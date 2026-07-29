#!/usr/bin/env python3
"""Run walk-forward analysis on top strategies."""

import sys
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.walk_forward import WalkForwardConfig, run_walk_forward, print_wf_report
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy


DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # RSI+Bollinger — use optimized param grid (narrowed around known good region)
    grid = {
        "rsi_period": [21, 28, 30, 35],
        "overbought": [60, 65, 70],
        "oversold": [28, 30, 32, 38],
        "bb_period": [12, 15, 20],
        "bb_std": [1.8, 2.0, 2.2],
    }

    # --- 1h Walk-Forward ---
    print("=" * 80, file=sys.stderr)
    print("RSI+BOLLINGER — 1h — Rolling 18mo IS / 3mo OOS", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    df_1h = load_candles(DATA_PATH, resample="1h")
    print(f"Data: {len(df_1h)} bars, {df_1h.index[0].date()} to {df_1h.index[-1].date()}", file=sys.stderr)

    wf_1h = run_walk_forward(
        strategy_class=RsiBollingerStrategy,
        df=df_1h,
        grid=grid,
        config=config,
        wf_config=WalkForwardConfig(is_months=18, oos_months=3, mode="rolling"),
        rank_by="sharpe_ratio",
    )
    print_wf_report(wf_1h)

    # --- 4h Walk-Forward ---
    print("=" * 80, file=sys.stderr)
    print("RSI+BOLLINGER — 4h — Rolling 18mo IS / 3mo OOS", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    df_4h = load_candles(DATA_PATH, resample="4h")
    print(f"Data: {len(df_4h)} bars, {df_4h.index[0].date()} to {df_4h.index[-1].date()}", file=sys.stderr)

    wf_4h = run_walk_forward(
        strategy_class=RsiBollingerStrategy,
        df=df_4h,
        grid=grid,
        config=config,
        wf_config=WalkForwardConfig(is_months=18, oos_months=3, mode="rolling"),
        rank_by="sharpe_ratio",
    )
    print_wf_report(wf_4h)


if __name__ == "__main__":
    main()
