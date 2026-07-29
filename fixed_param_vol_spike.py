"""
Fixed-parameter Vol Spike strategy stability test.

Tests VolSpikeStrategy with FIXED parameters (no optimization) across:
1. Rolling 12-month windows (3-month step) for temporal stability
2. Full-sample at 10bps and 5bps fee levels
3. Alternative param sets for sensitivity analysis

Eliminates overfitting concern entirely — if the strategy is robust,
fixed params should produce consistent positive Sharpe across windows.
"""

import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from strategies.vol_spike_strategy import VolSpikeStrategy

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
FIXED_PARAMS = {
    "vol_zscore_window": 96,
    "vol_zscore_threshold": 2.5,
    "delta_threshold_pct": 10.0,
    "hold_bars": 8,
    "buy_only": True,
}

ALT_PARAMS = [
    {"vol_zscore_window": 96, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 96, "vol_zscore_threshold": 2.0, "delta_threshold_pct": 10.0, "hold_bars": 8, "buy_only": True},
    {"vol_zscore_window": 48, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 8, "buy_only": True},
]

ALL_PARAM_SETS = [FIXED_PARAMS] + ALT_PARAMS
PARAM_LABELS = [
    "BASE(w96/z2.5/h8)",
    "ALT1(w96/z2.5/h4)",
    "ALT2(w96/z2.0/h8)",
    "ALT3(w48/z2.5/h8)",
]

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

WINDOW_MONTHS = 12
STEP_MONTHS = 3

CFG_10BPS = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
CFG_5BPS = BacktestConfig(fee_rate=0.00025, slippage_pct=0.00025)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_single(df: pd.DataFrame, params: dict, config: BacktestConfig) -> dict:
    """Run strategy with given params on df, return metrics dict."""
    strat = VolSpikeStrategy(**params)
    signals = strat.generate_signals(df)
    results = run_backtest(df, signals, config)
    metrics = compute_metrics(results)
    return metrics


def rolling_windows(df: pd.DataFrame, window_months: int, step_months: int):
    """Yield (start, end, df_slice) for each rolling window."""
    start_date = df.index[0]
    end_date = df.index[-1]

    current_start = start_date
    while True:
        current_end = current_start + pd.DateOffset(months=window_months)
        if current_end > end_date:
            break
        df_slice = df.loc[current_start:current_end]
        if len(df_slice) > 500:  # need at least ~500 bars (21 days at 1h)
            yield current_start, current_end, df_slice
        current_start += pd.DateOffset(months=step_months)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 100)
    print("FIXED-PARAM VOL SPIKE STRATEGY — ROLLING STABILITY TEST")
    print("=" * 100)
    print()

    # --- Load data ---
    data = {}
    for asset, path in ASSETS.items():
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {path} not found, skipping {asset}")
            continue
        print(f"Loading {asset} from {path}...", end=" ", flush=True)
        df = load_candles(str(p), resample="1h", extra_columns=["taker_buy_base_volume"])
        print(f"{len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        data[asset] = df
    print()

    # Collect results for summary table
    summary_rows = []

    for pi, (params, label) in enumerate(zip(ALL_PARAM_SETS, PARAM_LABELS)):
        print("-" * 100)
        print(f"PARAM SET {pi}: {label}")
        print(f"  Params: {params}")
        print("-" * 100)

        for asset, df in data.items():
            print(f"\n  === {asset} ({len(df)} bars) ===")

            # --- Full-sample backtests ---
            m10 = run_single(df, params, CFG_10BPS)
            m5 = run_single(df, params, CFG_5BPS)

            print(f"  Full-sample 10bps: Sharpe={m10['sharpe_ratio']:.2f}  "
                  f"Return={m10['total_return_pct']:.1f}%  "
                  f"Trades={m10['total_trades']}  "
                  f"WR={m10['win_rate_pct']:.1f}%  "
                  f"MaxDD={m10['max_drawdown_pct']:.1f}%  "
                  f"PF={m10['profit_factor']:.2f}")
            print(f"  Full-sample  5bps: Sharpe={m5['sharpe_ratio']:.2f}  "
                  f"Return={m5['total_return_pct']:.1f}%  "
                  f"Trades={m5['total_trades']}  "
                  f"WR={m5['win_rate_pct']:.1f}%  "
                  f"MaxDD={m5['max_drawdown_pct']:.1f}%  "
                  f"PF={m5['profit_factor']:.2f}")

            # --- Rolling windows ---
            window_results = []
            for ws, we, df_slice in rolling_windows(df, WINDOW_MONTHS, STEP_MONTHS):
                wm = run_single(df_slice, params, CFG_10BPS)
                window_results.append({
                    "start": ws.strftime("%Y-%m"),
                    "end": we.strftime("%Y-%m"),
                    "sharpe": wm["sharpe_ratio"],
                    "return_pct": wm["total_return_pct"],
                    "trades": wm["total_trades"],
                    "max_dd": wm["max_drawdown_pct"],
                    "win_rate": wm["win_rate_pct"],
                })

            n_windows = len(window_results)
            if n_windows > 0:
                sharpes = [w["sharpe"] for w in window_results]
                frac_pos = sum(1 for s in sharpes if s > 0) / n_windows
                frac_good = sum(1 for s in sharpes if s > 0.5) / n_windows
                avg_sharpe = np.mean(sharpes)
                med_sharpe = np.median(sharpes)
                min_sharpe = min(sharpes)
                max_sharpe = max(sharpes)
            else:
                frac_pos = frac_good = avg_sharpe = med_sharpe = min_sharpe = max_sharpe = 0.0

            print(f"\n  Rolling {WINDOW_MONTHS}mo windows ({STEP_MONTHS}mo step): {n_windows} windows")
            print(f"  Sharpe: avg={avg_sharpe:.2f}  med={med_sharpe:.2f}  min={min_sharpe:.2f}  max={max_sharpe:.2f}")
            print(f"  Frac(Sharpe>0) = {frac_pos:.0%}   Frac(Sharpe>0.5) = {frac_good:.0%}")

            # Print per-window detail
            print(f"\n  {'Window':<20} {'Sharpe':>7} {'Return%':>9} {'Trades':>7} {'WR%':>6} {'MaxDD%':>8}")
            for w in window_results:
                print(f"  {w['start']}–{w['end']}   {w['sharpe']:>7.2f} {w['return_pct']:>9.1f} "
                      f"{w['trades']:>7} {w['win_rate']:>6.1f} {w['max_dd']:>8.1f}")

            # Collect for summary
            summary_rows.append({
                "params": label,
                "asset": asset,
                "sharpe_10bps": m10["sharpe_ratio"],
                "sharpe_5bps": m5["sharpe_ratio"],
                "trades": m10["total_trades"],
                "win_rate": m10["win_rate_pct"],
                "return_pct": m10["total_return_pct"],
                "max_dd": m10["max_drawdown_pct"],
                "profit_factor": m10["profit_factor"],
                "n_windows": n_windows,
                "frac_pos": frac_pos,
                "frac_good": frac_good,
                "avg_window_sharpe": avg_sharpe,
                "med_window_sharpe": med_sharpe,
            })

    # --- Summary Table ---
    print("\n\n")
    print("=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    header = (f"{'Params':<22} {'Asset':<5} {'Sh10bp':>7} {'Sh5bp':>6} {'Trades':>7} "
              f"{'WR%':>6} {'Ret%':>8} {'MaxDD%':>7} {'PF':>6} "
              f"{'Win':>4} {'F>0':>5} {'F>0.5':>6} {'AvgSh':>6} {'MedSh':>6}")
    print(header)
    print("-" * 130)
    for r in summary_rows:
        print(f"{r['params']:<22} {r['asset']:<5} {r['sharpe_10bps']:>7.2f} {r['sharpe_5bps']:>6.2f} "
              f"{r['trades']:>7} {r['win_rate']:>6.1f} {r['return_pct']:>8.1f} {r['max_dd']:>7.1f} "
              f"{r['profit_factor']:>6.2f} {r['n_windows']:>4} {r['frac_pos']:>5.0%} {r['frac_good']:>6.0%} "
              f"{r['avg_window_sharpe']:>6.2f} {r['med_window_sharpe']:>6.2f}")

    # --- Cross-asset consistency ---
    print("\n\nCROSS-ASSET CONSISTENCY (Base params, 10bps)")
    print("-" * 60)
    base_rows = [r for r in summary_rows if r["params"] == PARAM_LABELS[0]]
    for r in base_rows:
        print(f"  {r['asset']}: Sharpe={r['sharpe_10bps']:.2f}, "
              f"Frac(>0)={r['frac_pos']:.0%}, "
              f"Frac(>0.5)={r['frac_good']:.0%}, "
              f"AvgWindowSharpe={r['avg_window_sharpe']:.2f}")

    # --- Sensitivity analysis ---
    print("\n\nSENSITIVITY ANALYSIS (10bps, full-sample)")
    print("-" * 80)
    for asset_name in data.keys():
        asset_rows = [r for r in summary_rows if r["asset"] == asset_name]
        print(f"\n  {asset_name}:")
        for r in asset_rows:
            delta = r["sharpe_10bps"] - [x for x in summary_rows if x["params"] == PARAM_LABELS[0] and x["asset"] == asset_name][0]["sharpe_10bps"]
            print(f"    {r['params']:<22}: Sharpe={r['sharpe_10bps']:.2f} (delta={delta:+.2f})")

    elapsed = time.time() - t0
    print(f"\n\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
