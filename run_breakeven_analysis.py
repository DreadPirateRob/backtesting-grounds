#!/usr/bin/env python3
"""Breakeven Fee & Notional Analysis for 5 Validated Strategies x 3 Assets.

Computes:
  1. Breakeven fee: total round-trip cost (bps) where Sharpe crosses 0
  2. Fee buffer: margin between 10bps operating cost and breakeven
  3. Edge per trade: avg return per trade at 10bps cost

Output: ranked table with fragility warnings.
"""

import sys
import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics

from strategies.vol_spike_strategy import VolSpikeStrategy
from strategies.multi_bar_selloff_strategy import MultiBarSelloffStrategy
from strategies.extreme_range_strategy import ExtremeRangeStrategy
from strategies.donchian_vol_strategy import DonchianVolStrategy
from strategies.green_momentum_strategy import GreenMomentumStrategy


DATA_FILES = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

STRATEGIES = {
    "VolSpike": (VolSpikeStrategy, dict(
        vol_zscore_window=96, vol_zscore_threshold=2.5,
        delta_threshold_pct=10.0, hold_bars=8, buy_only=True,
    )),
    "Selloff": (MultiBarSelloffStrategy, dict(
        lookback=24, drop_pct=-8.0, hold_bars=16, cooldown_bars=3,
    )),
    "ExtrRange": (ExtremeRangeStrategy, dict(
        lookback=72, hold_bars=8, cooldown_bars=2,
    )),
    "Donchian": (DonchianVolStrategy, dict(
        entry_period=480, vol_mult=2.0, hold_bars=24, cooldown_bars=12,
    )),
    "GreenMom": (GreenMomentumStrategy, dict(
        min_green=4, min_gain_pct=4.0, hold_bars=16, cooldown_bars=3,
    )),
}

# Fee sweep: 0 to 50 bps in 2.5bps steps (total round-trip)
FEE_SWEEP_BPS = np.arange(0, 52.5, 2.5)  # [0, 2.5, 5, ..., 50]
OPERATING_COST_BPS = 10.0


def run_at_fee(df, signals, total_cost_bps):
    """Run backtest at a given total round-trip cost (bps).

    Splits total cost evenly between fee_rate and slippage_pct,
    so round-trip = fee_rate + slippage_pct = total_cost_bps.
    """
    half = total_cost_bps / 2.0 / 10000.0  # convert bps to fraction, split
    config = BacktestConfig(fee_rate=half, slippage_pct=half)
    result = run_backtest(df, signals, config)
    metrics = compute_metrics(result)
    return metrics


def find_breakeven(df, signals):
    """Binary search for exact breakeven fee (bps) where Sharpe crosses 0.

    First checks coarse sweep, then refines with bisection.
    Returns breakeven in bps, or >50 if still positive at 50bps.
    """
    # Check if already negative at 0 cost
    m0 = run_at_fee(df, signals, 0.0)
    if m0["sharpe_ratio"] <= 0:
        return 0.0

    # Check if still positive at 50bps
    m50 = run_at_fee(df, signals, 50.0)
    if m50["sharpe_ratio"] > 0:
        return 50.0  # cap at 50; edge survives even 50bps

    # Coarse sweep to find bracket
    lo, hi = 0.0, 50.0
    for fee_bps in FEE_SWEEP_BPS:
        m = run_at_fee(df, signals, fee_bps)
        if m["sharpe_ratio"] <= 0:
            hi = fee_bps
            lo = fee_bps - 2.5 if fee_bps > 0 else 0.0
            break

    # Bisection to 0.1 bps precision
    for _ in range(20):
        mid = (lo + hi) / 2.0
        m = run_at_fee(df, signals, mid)
        if m["sharpe_ratio"] > 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 0.1:
            break

    return (lo + hi) / 2.0


def main():
    t0 = time.time()
    p = lambda *a, **kw: print(*a, **kw, flush=True)

    p("=" * 105)
    p("  BREAKEVEN FEE & EDGE ANALYSIS: 5 Strategies x 3 Assets")
    p("=" * 105)

    # Load data
    p("\n  Loading data...")
    asset_dfs = {}
    for asset, path in DATA_FILES.items():
        df = load_candles(path, resample="1h",
                          extra_columns=["taker_buy_base_volume"])
        asset_dfs[asset] = df
        p(f"    {asset}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # Pre-generate signals (reused across fee sweeps)
    p("\n  Generating signals...")
    all_signals = {}  # (strat_name, asset) -> signals Series
    for asset, df in asset_dfs.items():
        for strat_name, (cls, params) in STRATEGIES.items():
            strat = cls(**params)
            signals = strat.generate_signals(df)
            all_signals[(strat_name, asset)] = signals
            # Quick check: count entries
            n_entries = (signals.diff().abs() > 0).sum()
            p(f"    {strat_name:>10} x {asset}: {n_entries} signal changes")

    # Run analysis
    p("\n  Running breakeven sweep + metrics...")
    results = []

    for strat_name in STRATEGIES:
        for asset in asset_dfs:
            key = (strat_name, asset)
            df = asset_dfs[asset]
            signals = all_signals[key]

            # 1. Metrics at operating cost (10bps)
            m10 = run_at_fee(df, signals, OPERATING_COST_BPS)
            sharpe_10 = m10["sharpe_ratio"]
            total_return_pct = m10["total_return_pct"]
            total_trades = m10["total_trades"]

            # 2. Edge per trade (total return / trades, in bps)
            if total_trades > 0:
                edge_per_trade_bps = (total_return_pct / total_trades) * 100  # pct -> bps
            else:
                edge_per_trade_bps = 0.0

            # 3. Breakeven fee
            breakeven_bps = find_breakeven(df, signals)

            # 4. Fee buffer
            buffer_bps = breakeven_bps - OPERATING_COST_BPS

            results.append({
                "strategy": strat_name,
                "asset": asset,
                "sharpe_10bps": sharpe_10,
                "breakeven_bps": breakeven_bps,
                "buffer_bps": buffer_bps,
                "edge_per_trade_bps": edge_per_trade_bps,
                "trades": total_trades,
                "total_return_pct": total_return_pct,
            })

            status = "*** FRAGILE ***" if buffer_bps < 5.0 else ""
            p(f"    {strat_name:>10} x {asset}: Sharpe={sharpe_10:+.2f}, "
              f"BE={breakeven_bps:.1f}bps, Buffer={buffer_bps:+.1f}bps "
              f"{status}")

    # Build results DataFrame
    df_results = pd.DataFrame(results)

    # ============================================================
    # MAIN TABLE
    # ============================================================
    p(f"\n{'=' * 105}")
    p(f"  RESULTS TABLE")
    p(f"{'=' * 105}")
    p(f"\n  {'Strategy':<12} {'Asset':<6} {'Sharpe@10bps':>13} {'Breakeven':>10} "
      f"{'Buffer':>8} {'Edge/Trade':>11} {'Trades':>8}  {'Note'}")
    p(f"  {'-'*12} {'-'*6} {'-'*13} {'-'*10} {'-'*8} {'-'*11} {'-'*8}  {'-'*15}")

    for _, row in df_results.iterrows():
        note = ""
        if row["buffer_bps"] < 5.0:
            note = "<<< FRAGILE"
        elif row["breakeven_bps"] >= 50.0:
            note = "(>50bps cap)"

        breakeven_str = f"{row['breakeven_bps']:.1f}bps"
        if row["breakeven_bps"] >= 50.0:
            breakeven_str = ">50bps"

        p(f"  {row['strategy']:<12} {row['asset']:<6} {row['sharpe_10bps']:>+13.2f} "
          f"{breakeven_str:>10} {row['buffer_bps']:>+7.1f} "
          f"{row['edge_per_trade_bps']:>+10.1f} {row['trades']:>8.0f}  {note}")

    # ============================================================
    # FRAGILE EDGES WARNING
    # ============================================================
    fragile = df_results[df_results["buffer_bps"] < 5.0]
    if len(fragile) > 0:
        p(f"\n  {'!' * 80}")
        p(f"  WARNING: {len(fragile)} strategy/asset combos have buffer < 5bps (fragile edge):")
        for _, row in fragile.iterrows():
            p(f"    - {row['strategy']} x {row['asset']}: "
              f"buffer={row['buffer_bps']:+.1f}bps, Sharpe@10bps={row['sharpe_10bps']:+.2f}")
        p(f"  {'!' * 80}")
    else:
        p(f"\n  All strategy/asset combos have buffer >= 5bps. No fragile edges detected.")

    # ============================================================
    # SUMMARY: RANK BY AVG BUFFER
    # ============================================================
    p(f"\n{'=' * 105}")
    p(f"  STRATEGY RANKING BY AVERAGE BUFFER ACROSS ASSETS")
    p(f"{'=' * 105}")

    summary = df_results.groupby("strategy").agg(
        avg_buffer=("buffer_bps", "mean"),
        min_buffer=("buffer_bps", "min"),
        avg_sharpe=("sharpe_10bps", "mean"),
        avg_edge=("edge_per_trade_bps", "mean"),
        total_trades=("trades", "sum"),
    ).sort_values("avg_buffer", ascending=False)

    p(f"\n  {'Rank':<6} {'Strategy':<12} {'Avg Buffer':>11} {'Min Buffer':>11} "
      f"{'Avg Sharpe':>11} {'Avg Edge/Tr':>12} {'Tot Trades':>11}")
    p(f"  {'-'*6} {'-'*12} {'-'*11} {'-'*11} {'-'*11} {'-'*12} {'-'*11}")

    for rank, (strat, row) in enumerate(summary.iterrows(), 1):
        p(f"  {rank:<6} {strat:<12} {row['avg_buffer']:>+10.1f} {row['min_buffer']:>+10.1f} "
          f"{row['avg_sharpe']:>+10.2f} {row['avg_edge']:>+11.1f} {row['total_trades']:>11.0f}")

    # ============================================================
    # PER-ASSET SUMMARY
    # ============================================================
    p(f"\n{'=' * 105}")
    p(f"  PER-ASSET SUMMARY")
    p(f"{'=' * 105}")

    asset_summary = df_results.groupby("asset").agg(
        avg_buffer=("buffer_bps", "mean"),
        min_buffer=("buffer_bps", "min"),
        avg_sharpe=("sharpe_10bps", "mean"),
        fragile_count=("buffer_bps", lambda x: (x < 5.0).sum()),
    )

    p(f"\n  {'Asset':<6} {'Avg Buffer':>11} {'Min Buffer':>11} {'Avg Sharpe':>11} {'Fragile':>8}")
    p(f"  {'-'*6} {'-'*11} {'-'*11} {'-'*11} {'-'*8}")
    for asset, row in asset_summary.iterrows():
        p(f"  {asset:<6} {row['avg_buffer']:>+10.1f} {row['min_buffer']:>+10.1f} "
          f"{row['avg_sharpe']:>+10.2f} {row['fragile_count']:>8.0f}")

    # ============================================================
    # FEE SENSITIVITY CURVES (condensed)
    # ============================================================
    p(f"\n{'=' * 105}")
    p(f"  FEE SENSITIVITY: Sharpe at key cost levels")
    p(f"{'=' * 105}")

    cost_levels = [0, 5, 10, 15, 20, 30, 40, 50]
    header = f"  {'Strategy':<12} {'Asset':<6}" + "".join(f" {c:>5}bps" for c in cost_levels)
    p(f"\n{header}")
    p(f"  {'-'*12} {'-'*6}" + " -------" * len(cost_levels))

    for strat_name in STRATEGIES:
        for asset in asset_dfs:
            df = asset_dfs[asset]
            signals = all_signals[(strat_name, asset)]
            sharpes = []
            for cost in cost_levels:
                m = run_at_fee(df, signals, float(cost))
                sharpes.append(m["sharpe_ratio"])
            vals = "".join(f" {s:>+7.2f}" for s in sharpes)
            p(f"  {strat_name:<12} {asset:<6}{vals}")

    elapsed = time.time() - t0
    p(f"\n  Total time: {elapsed:.0f}s")
    p("=" * 105)


if __name__ == "__main__":
    main()
