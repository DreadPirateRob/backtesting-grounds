#!/usr/bin/env python3
"""Walk-forward analysis for Bollinger Bounce strategy at 1h and 4h."""

import sys
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.walk_forward import WalkForwardConfig, run_walk_forward, print_wf_report
from strategies.bollinger_bounce_strategy import BollingerBounceStrategy


DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # Bollinger Bounce param grid (full grid from strategy)
    grid = {
        "period": [10, 15, 20, 25, 30, 40],
        "std": [1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
    }

    # --- 1h Walk-Forward ---
    print("=" * 80, file=sys.stderr)
    print("BOLLINGER BOUNCE — 1h — Rolling 18mo IS / 3mo OOS", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    df_1h = load_candles(DATA_PATH, resample="1h")
    print(f"Data: {len(df_1h)} bars, {df_1h.index[0].date()} to {df_1h.index[-1].date()}", file=sys.stderr)

    wf_1h = run_walk_forward(
        strategy_class=BollingerBounceStrategy,
        df=df_1h,
        grid=grid,
        config=config,
        wf_config=WalkForwardConfig(is_months=18, oos_months=3, mode="rolling"),
        rank_by="sharpe_ratio",
    )
    print_wf_report(wf_1h)

    # --- 4h Walk-Forward ---
    print("=" * 80, file=sys.stderr)
    print("BOLLINGER BOUNCE — 4h — Rolling 18mo IS / 3mo OOS", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    df_4h = load_candles(DATA_PATH, resample="4h")
    print(f"Data: {len(df_4h)} bars, {df_4h.index[0].date()} to {df_4h.index[-1].date()}", file=sys.stderr)

    wf_4h = run_walk_forward(
        strategy_class=BollingerBounceStrategy,
        df=df_4h,
        grid=grid,
        config=config,
        wf_config=WalkForwardConfig(is_months=18, oos_months=3, mode="rolling"),
        rank_by="sharpe_ratio",
    )
    print_wf_report(wf_4h)


if __name__ == "__main__":
    main()
