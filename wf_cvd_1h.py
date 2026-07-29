#!/usr/bin/env python3
"""Walk-forward validation: CVD Divergence strategy at 1h on BTC, ETH, SOL."""

import sys
import time

from backtester.walk_forward import run_walk_forward, WalkForwardConfig, print_wf_report
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from strategies.cvd_divergence_strategy import CvdDivergenceStrategy

ASSETS = [
    ("BTC", "data/binance_BTCUSDT_1m_klines.csv"),
    ("ETH", "data/binance_ETHUSDT_1m_klines.csv"),
    ("SOL", "data/binance_SOLUSDT_1m_klines.csv"),
]

wf_config = WalkForwardConfig(
    is_months=18,
    oos_months=3,
    mode="rolling",
    min_trades_oos=5,
)

bt_config = BacktestConfig(
    fee_rate=0.0005,
    slippage_pct=0.0005,
)

if __name__ == '__main__':
    strategy = CvdDivergenceStrategy()
    grid = strategy.param_grid()
    print(f"Param grid: {grid}", file=sys.stderr)
    combos = 1
    for v in grid.values():
        combos *= len(v)
    print(f"Total combinations: {combos}", file=sys.stderr)

    results = {}

    for asset_name, path in ASSETS:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"  {asset_name} — CVD Divergence @ 1h Walk-Forward", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)

        t0 = time.time()

        df = load_candles(path, resample="1h", extra_columns=["taker_buy_base_volume"])
        print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}", file=sys.stderr)

        result = run_walk_forward(
            strategy_class=CvdDivergenceStrategy,
            df=df,
            grid=grid,
            config=bt_config,
            wf_config=wf_config,
        )

        elapsed = time.time() - t0
        print(f"\n[{asset_name}] Completed in {elapsed:.1f}s", file=sys.stderr)

        print(f"\n--- {asset_name} REPORT ---", file=sys.stderr)
        print_wf_report(result)

        results[asset_name] = result

    # Cross-asset summary
    print("\n" + "=" * 80, file=sys.stderr)
    print("CROSS-ASSET SUMMARY: CVD Divergence @ 1h", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"{'Asset':<6}  {'Segments':>8}  {'OOS Sharpe':>11}  {'WFER':>6}  {'OOS Ret%':>9}  {'OOS MaxDD%':>11}  {'WFER Slope':>11}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    for asset_name, r in results.items():
        eq = r.combined_oos_equity
        total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        cummax = eq.cummax()
        max_dd = float(((eq - cummax) / cummax).min()) * 100
        print(
            f"{asset_name:<6}  {len(r.segments):>8}  {r.overall_oos_sharpe:>11.4f}  "
            f"{r.overall_wfer:>6.4f}  {total_ret:>9.2f}  {max_dd:>11.2f}  "
            f"{r.wfer_slope:>+11.6f}",
            file=sys.stderr,
        )

    print("=" * 80, file=sys.stderr)
