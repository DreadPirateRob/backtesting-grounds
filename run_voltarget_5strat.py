#!/usr/bin/env python3
"""Position-sizing overlays on 5-strategy x 3-asset mega portfolio.

Takes the 15 raw return streams (5 strategies x 3 assets), combines into
an equal-weight mega portfolio, then applies:
  1. Volatility Targeting (15%, 20%, 25% target at 30d/60d lookback)
  2. Drawdown-based Scaling (10%, 15%, 20% thresholds)
  3. Half-Kelly sizing (30d, 60d lookback)

Prints comparison table sorted by Calmar, per-year Sharpe for best vs raw,
and rolling 12-month stability for the best method.
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

CONFIG = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)

# Best validated params for each strategy (same as run_portfolio_5strat.py)
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


# ---------------------------------------------------------------------------
# Position sizing overlays (copied from run_wf_multiasset_voltarget.py)
# ---------------------------------------------------------------------------

def apply_vol_target(returns: pd.Series, target_vol: float,
                     lookback: int = 720, min_scale: float = 0.5,
                     max_scale: float = 2.0) -> pd.Series:
    """Scale returns to target a fixed annualized volatility."""
    rolling_std = returns.rolling(window=lookback, min_periods=max(lookback // 2, 60)).std()
    realized_vol = rolling_std * np.sqrt(8760)
    scale = target_vol / realized_vol
    scale = scale.clip(lower=min_scale, upper=max_scale)
    scale = scale.shift(1).fillna(1.0)
    return returns * scale


def apply_dd_scaling(returns: pd.Series, dd_threshold: float,
                     recovery_mult: float = 0.5,
                     reduced_size: float = 0.5) -> pd.Series:
    """Reduce position when portfolio drawdown exceeds threshold."""
    equity = (1 + returns).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax  # negative values

    scale = pd.Series(1.0, index=returns.index)
    in_reduction = False

    for i in range(len(returns)):
        dd = drawdown.iloc[i]
        if not in_reduction:
            if dd < -dd_threshold:
                in_reduction = True
                scale.iloc[i] = reduced_size
            else:
                scale.iloc[i] = 1.0
        else:
            if dd > -(dd_threshold * recovery_mult):
                in_reduction = False
                scale.iloc[i] = 1.0
            else:
                scale.iloc[i] = reduced_size

    scale = scale.shift(1).fillna(1.0)
    return returns * scale


def apply_half_kelly(returns: pd.Series, lookback: int = 720,
                     min_scale: float = 0.25,
                     max_scale: float = 1.5) -> pd.Series:
    """Apply half-Kelly sizing based on rolling Sharpe estimate."""
    rolling_mean = returns.rolling(window=lookback, min_periods=max(lookback // 2, 60)).mean()
    rolling_std = returns.rolling(window=lookback, min_periods=max(lookback // 2, 60)).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(8760)
    rolling_sharpe = rolling_sharpe.fillna(0)

    s2 = rolling_sharpe ** 2
    kelly = s2 / (s2 + 1)
    half_kelly = kelly / 2.0

    scale = half_kelly.clip(lower=min_scale, upper=max_scale)
    scale = scale.shift(1).fillna(min_scale)
    return returns * scale


def compute_equity_metrics(returns: pd.Series, initial_capital: float = 10_000):
    """Compute Sharpe, total return, max DD, Calmar from a return series (1h bars)."""
    equity = initial_capital * (1 + returns).cumprod()
    duration_days = (returns.index[-1] - returns.index[0]).total_seconds() / 86400
    years = duration_days / 365.25 if duration_days > 0 else 1.0

    mean_ret = returns.mean()
    std_ret = returns.std()
    bars_per_year = len(returns) / years if years > 0 else 8760
    sharpe = (mean_ret / std_ret * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cummax = equity.cummax()
    max_dd = float(((equity - cummax) / cummax).min()) * 100

    # Calmar = annualized return / abs(max DD)
    ann_ret = total_ret / years if years > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "total_return_pct": round(total_ret, 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 4),
        "ann_ret_pct": round(ann_ret, 2),
        "equity": equity,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    p = lambda *a, **kw: print(*a, **kw)

    p("=" * 120)
    p("  5-STRATEGY x 3-ASSET MEGA PORTFOLIO — POSITION-SIZING OVERLAYS")
    p("  Strategies: VolSpike, Selloff, ExtrRange, Donchian, GreenMom")
    p("  Assets: BTC, ETH, SOL | Resolution: 1h | Fee: 10bps")
    p("=" * 120)

    # ==================================================================
    # Step 1: Load data
    # ==================================================================
    asset_dfs = {}
    for asset, path in DATA_FILES.items():
        df = load_candles(path, resample="1h",
                          extra_columns=["taker_buy_base_volume"])
        asset_dfs[asset] = df
        p(f"  {asset}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # ==================================================================
    # Step 2: Run all 5 strategies on each asset -> 15 return streams
    # ==================================================================
    p(f"\n  Running 5 strategies x 3 assets = 15 streams...")

    all_returns = {}  # {key: pd.Series}
    individual_metrics = []

    for asset, df in asset_dfs.items():
        for strat_name, (cls, params) in STRATEGIES.items():
            key = f"{strat_name}_{asset}"
            strat = cls(**params)
            signals = strat.generate_signals(df)
            result = run_backtest(df, signals, CONFIG)
            all_returns[key] = result["strategy_returns"]

            metrics = compute_metrics(result)
            individual_metrics.append({
                "key": key,
                "sharpe": metrics["sharpe_ratio"],
                "return_pct": metrics["total_return_pct"],
                "max_dd_pct": metrics["max_drawdown_pct"],
            })

    p(f"  Done. {len(all_returns)} return streams generated.\n")

    # Individual performance summary
    p(f"  {'Stream':<20} {'Sharpe':>8} {'Return%':>10} {'MaxDD%':>8}")
    p(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8}")
    for m in sorted(individual_metrics, key=lambda x: x["sharpe"], reverse=True):
        p(f"  {m['key']:<20} {m['sharpe']:>+8.2f} {m['return_pct']:>+10.1f} {m['max_dd_pct']:>8.1f}")

    # ==================================================================
    # Step 3: Equal-weight mega portfolio (sum / 15)
    # ==================================================================
    returns_df = pd.DataFrame(all_returns).fillna(0.0).sort_index()
    n_streams = returns_df.shape[1]
    raw_portfolio = returns_df.mean(axis=1)  # equal-weight = mean of 15

    p(f"\n  Mega portfolio: {n_streams} streams, {len(raw_portfolio)} bars")
    p(f"  Date range: {raw_portfolio.index[0].date()} to {raw_portfolio.index[-1].date()}")

    raw_metrics = compute_equity_metrics(raw_portfolio)
    p(f"  Raw Sharpe: {raw_metrics['sharpe']:.4f}, "
      f"Return: {raw_metrics['total_return_pct']:.2f}%, "
      f"MaxDD: {raw_metrics['max_dd_pct']:.2f}%, "
      f"Calmar: {raw_metrics['calmar']:.4f}")

    # ==================================================================
    # Step 4: Apply overlays
    # ==================================================================
    p(f"\n{'=' * 120}")
    p(f"  APPLYING POSITION-SIZING OVERLAYS")
    p(f"{'=' * 120}")

    methods = {}
    methods["Raw EW Portfolio"] = raw_portfolio.copy()

    # Vol targeting: 15%, 20%, 25% at 30d and 60d lookback
    for target_pct in [15, 20, 25]:
        target = target_pct / 100.0
        for lb_days in [30, 60]:
            lb_bars = lb_days * 24  # 1h bars
            label = f"VolTarget {target_pct}% ({lb_days}d)"
            p(f"  Computing {label}...")
            methods[label] = apply_vol_target(
                raw_portfolio.copy(), target_vol=target, lookback=lb_bars
            )

    # Drawdown-based scaling: 10%, 15%, 20%
    for dd_pct in [10, 15, 20]:
        dd_thresh = dd_pct / 100.0
        label = f"DD-Scale {dd_pct}%"
        p(f"  Computing {label}...")
        methods[label] = apply_dd_scaling(
            raw_portfolio.copy(), dd_threshold=dd_thresh
        )

    # Half-Kelly: 30d, 60d
    for lb_days in [30, 60]:
        lb_bars = lb_days * 24
        label = f"Half-Kelly ({lb_days}d)"
        p(f"  Computing {label}...")
        methods[label] = apply_half_kelly(
            raw_portfolio.copy(), lookback=lb_bars
        )

    # ==================================================================
    # Step 5: Compute metrics for each method
    # ==================================================================
    p(f"\n{'=' * 120}")
    p(f"  COMPARISON TABLE (sorted by Calmar)")
    p(f"{'=' * 120}")

    results_table = []
    for method_name, rets in methods.items():
        m = compute_equity_metrics(rets)
        results_table.append({
            "method": method_name,
            "sharpe": m["sharpe"],
            "total_return_pct": m["total_return_pct"],
            "ann_ret_pct": m["ann_ret_pct"],
            "max_dd_pct": m["max_dd_pct"],
            "calmar": m["calmar"],
        })

    # Sort by Calmar descending
    results_table.sort(key=lambda x: x["calmar"], reverse=True)

    raw_result = next(r for r in results_table if r["method"] == "Raw EW Portfolio")

    header = (
        f"  {'Method':<28} {'Sharpe':>8} {'AnnRet%':>9} {'TotRet%':>9} "
        f"{'MaxDD%':>8} {'Calmar':>9}  Notes"
    )
    p(header)
    p(f"  {'-'*28} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*9}  {'-'*20}")

    for r in results_table:
        dd_improved = r["max_dd_pct"] > raw_result["max_dd_pct"]  # less negative = better
        sharpe_preserved = r["sharpe"] >= raw_result["sharpe"] * 0.85
        flag = ""
        if r["method"] == "Raw EW Portfolio":
            flag = "[BASELINE]"
        elif dd_improved and sharpe_preserved:
            flag = "DD improved + Sharpe kept"
        elif dd_improved:
            flag = "DD improved"
        elif r["calmar"] > raw_result["calmar"]:
            flag = "Calmar improved"

        p(f"  {r['method']:<28} {r['sharpe']:>+8.4f} {r['ann_ret_pct']:>+9.2f} "
          f"{r['total_return_pct']:>+9.2f} {r['max_dd_pct']:>8.2f} "
          f"{r['calmar']:>9.4f}  {flag}")

    p(f"  {'-'*100}")

    # ==================================================================
    # Step 6: Per-year Sharpe for best method vs raw
    # ==================================================================
    best = results_table[0]
    best_name = best["method"]
    best_returns = methods[best_name]

    p(f"\n{'=' * 120}")
    p(f"  PER-YEAR SHARPE: {best_name} vs Raw EW Portfolio")
    p(f"{'=' * 120}")

    years = sorted(raw_portfolio.index.year.unique())
    p(f"\n  {'Year':<8} {'Raw Sharpe':>12} {best_name + ' Sharpe':>28} {'Delta':>10}")
    p(f"  {'-'*8} {'-'*12} {'-'*28} {'-'*10}")

    for year in years:
        raw_year = raw_portfolio[raw_portfolio.index.year == year]
        best_year = best_returns[best_returns.index.year == year]

        if len(raw_year) < 200:
            continue

        raw_m = compute_equity_metrics(raw_year)
        best_m = compute_equity_metrics(best_year)

        delta = best_m["sharpe"] - raw_m["sharpe"]
        p(f"  {year:<8} {raw_m['sharpe']:>+12.4f} {best_m['sharpe']:>+28.4f} {delta:>+10.4f}")

    # ==================================================================
    # Step 7: Rolling 12-month window stability for best method
    # ==================================================================
    p(f"\n{'=' * 120}")
    p(f"  ROLLING 12-MONTH WINDOW STABILITY: {best_name} vs Raw")
    p(f"{'=' * 120}")

    def rolling_window_sharpes(rets, window_months=12, step_months=3):
        start = rets.index[0]
        end = rets.index[-1]
        current = start
        results = []
        while True:
            window_end = current + pd.DateOffset(months=window_months)
            if window_end > end:
                break
            sl = rets.loc[current:window_end]
            if len(sl) > 500:
                dur_days = (sl.index[-1] - sl.index[0]).total_seconds() / 86400
                yrs = dur_days / 365.25 if dur_days > 0 else 1
                bpy = len(sl) / yrs if yrs > 0 else 8760
                mu = sl.mean()
                sigma = sl.std()
                sh = mu / sigma * np.sqrt(bpy) if sigma > 0 else 0
                results.append((current.strftime("%Y-%m"), sh))
            current += pd.DateOffset(months=step_months)
        return results

    raw_windows = rolling_window_sharpes(raw_portfolio)
    best_windows = rolling_window_sharpes(best_returns)

    p(f"\n  {'Window Start':<14} {'Raw Sharpe':>12} {best_name + ' Sharpe':>28} {'Delta':>10}")
    p(f"  {'-'*14} {'-'*12} {'-'*28} {'-'*10}")

    for (ws_raw, sh_raw), (ws_best, sh_best) in zip(raw_windows, best_windows):
        delta = sh_best - sh_raw
        p(f"  {ws_raw:<14} {sh_raw:>+12.2f} {sh_best:>+28.2f} {delta:>+10.2f}")

    # Summary stats
    raw_sh_vals = [s for _, s in raw_windows]
    best_sh_vals = [s for _, s in best_windows]

    if raw_sh_vals and best_sh_vals:
        p(f"\n  {'Statistic':<20} {'Raw':>12} {best_name:>28}")
        p(f"  {'-'*20} {'-'*12} {'-'*28}")
        p(f"  {'Mean Sharpe':<20} {np.mean(raw_sh_vals):>+12.2f} {np.mean(best_sh_vals):>+28.2f}")
        p(f"  {'Median Sharpe':<20} {np.median(raw_sh_vals):>+12.2f} {np.median(best_sh_vals):>+28.2f}")
        p(f"  {'Min Sharpe':<20} {np.min(raw_sh_vals):>+12.2f} {np.min(best_sh_vals):>+28.2f}")
        p(f"  {'Max Sharpe':<20} {np.max(raw_sh_vals):>+12.2f} {np.max(best_sh_vals):>+28.2f}")

        n_raw = len(raw_sh_vals)
        n_best = len(best_sh_vals)
        frac_pos_raw = sum(1 for s in raw_sh_vals if s > 0) / n_raw if n_raw else 0
        frac_pos_best = sum(1 for s in best_sh_vals if s > 0) / n_best if n_best else 0
        frac_gt1_raw = sum(1 for s in raw_sh_vals if s > 1.0) / n_raw if n_raw else 0
        frac_gt1_best = sum(1 for s in best_sh_vals if s > 1.0) / n_best if n_best else 0

        p(f"  {'% Positive':<20} {frac_pos_raw*100:>11.0f}% {frac_pos_best*100:>27.0f}%")
        p(f"  {'% > 1.0':<20} {frac_gt1_raw*100:>11.0f}% {frac_gt1_best*100:>27.0f}%")
        p(f"  {'Windows':<20} {n_raw:>12} {n_best:>28}")

    # ==================================================================
    # Summary
    # ==================================================================
    p(f"\n{'=' * 120}")
    p(f"  SUMMARY")
    p(f"{'=' * 120}")
    p(f"  Best method by Calmar: {best_name}")
    p(f"    Sharpe:       {best['sharpe']:+.4f}  (raw: {raw_result['sharpe']:+.4f}, "
      f"delta: {best['sharpe'] - raw_result['sharpe']:+.4f})")
    p(f"    Ann. Return:  {best['ann_ret_pct']:+.2f}%  (raw: {raw_result['ann_ret_pct']:+.2f}%)")
    p(f"    Total Return: {best['total_return_pct']:+.2f}%  (raw: {raw_result['total_return_pct']:+.2f}%)")
    p(f"    Max Drawdown: {best['max_dd_pct']:.2f}%  (raw: {raw_result['max_dd_pct']:.2f}%)")
    p(f"    Calmar:       {best['calmar']:.4f}  (raw: {raw_result['calmar']:.4f})")

    dd_reduction = abs(raw_result["max_dd_pct"]) - abs(best["max_dd_pct"])
    if dd_reduction > 0:
        p(f"    DD Reduction:  {dd_reduction:.2f}pp")
    else:
        p(f"    DD Increase:   {abs(dd_reduction):.2f}pp")

    elapsed = time.time() - t0
    p(f"\n  Total time: {elapsed:.0f}s")
    p("=" * 120)


if __name__ == "__main__":
    main()
