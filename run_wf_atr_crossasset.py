#!/usr/bin/env python3
"""Walk-forward analysis for ATR Breakout on ETH and SOL (cross-asset validation)."""

import sys
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.walk_forward import WalkForwardConfig, run_walk_forward, print_wf_report
from strategies.atr_breakout_strategy import AtrBreakoutStrategy


DATA_FILES = {
    "ETHUSDT": "data/binance_ETHUSDT_1m_klines.csv",
    "SOLUSDT": "data/binance_SOLUSDT_1m_klines.csv",
}


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    grid = {
        "sma_period": [10, 15, 20, 30, 40],
        "atr_period": [10, 14, 20],
        "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    }

    for symbol, path in DATA_FILES.items():
        for resample, tf_label in [("1h", "1h"), ("4h", "4h")]:
            print(f"\n{'=' * 80}", file=sys.stderr)
            print(f"ATR BREAKOUT — {symbol} — {tf_label} — Rolling 18mo IS / 3mo OOS", file=sys.stderr)
            print(f"{'=' * 80}", file=sys.stderr)

            try:
                df = load_candles(path, resample=resample)
            except Exception as e:
                print(f"Error loading {path}: {e}", file=sys.stderr)
                continue

            print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}", file=sys.stderr)

            # Need at least IS + OOS months of data
            min_bars = {"1h": 24 * 30 * 21, "4h": 6 * 30 * 21}  # ~21 months
            if len(df) < min_bars.get(tf_label, 10000):
                print(f"Insufficient data for walk-forward ({len(df)} bars). Skipping.", file=sys.stderr)
                continue

            try:
                wf = run_walk_forward(
                    strategy_class=AtrBreakoutStrategy,
                    df=df,
                    grid=grid,
                    config=config,
                    wf_config=WalkForwardConfig(is_months=18, oos_months=3, mode="rolling"),
                    rank_by="sharpe_ratio",
                )
                print_wf_report(wf)
            except ValueError as e:
                print(f"Walk-forward failed: {e}", file=sys.stderr)
                continue


if __name__ == "__main__":
    main()
