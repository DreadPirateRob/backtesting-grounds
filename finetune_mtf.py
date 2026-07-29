#!/usr/bin/env python3
"""Fine-tune the MTF RSI+Bollinger strategy on both timeframes."""

import time
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.sweep import run_sweep
from strategies.rsi_bb_mtf_strategy import RsiBbMtfStrategy


def to_float(v):
    try:
        return float(v) if v is not None else 0.0
    except:
        return 0.0


def print_results(title, result, top_n=10):
    print(f"\n{title}")
    print(f"{'#':<4} {'Sharpe':>7} {'Sortino':>8} {'Return%':>9} {'MaxDD%':>8} {'WinR%':>7} {'PF':>7} {'Calmar':>7} {'Trd':>5}  Params")
    print("-" * 130)
    for r in result["results"][:top_n]:
        m = r["metrics"]
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"#{r['rank']:<3} {to_float(m.get('sharpe_ratio')):>7.2f} {to_float(m.get('sortino_ratio')):>8.2f} "
              f"{to_float(m.get('total_return_pct')):>8.1f}% {to_float(m.get('max_drawdown_pct')):>7.1f}% "
              f"{to_float(m.get('win_rate_pct')):>6.1f}% {to_float(m.get('profit_factor')):>7.2f} "
              f"{to_float(m.get('calmar_ratio')):>7.2f} {int(to_float(m.get('total_trades'))):>5}  {params_str}")


if __name__ == "__main__":
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # ============================================================
    # SWING: MTF on 4h — fine-tune
    # Best was: rsi_period=14, overbought=65, oversold=30, bb_period=22, bb_std=2.25, daily_ema=30
    # ============================================================
    print("=" * 100)
    print("FINE-TUNING: MTF RSI+BB | 4h (SWING)")
    print("=" * 100)

    df_4h = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                         start="2025-01-01", end="2025-12-31", resample="4h")
    print(f"Data: {len(df_4h)} bars")

    grid_4h = {
        "rsi_period": [10, 12, 14, 16, 18],
        "overbought": [60, 62, 65, 67, 70],
        "oversold": [25, 28, 30, 32, 35],
        "bb_period": [15, 18, 20, 22, 25],
        "bb_std": [2.0, 2.25, 2.5],
        "daily_ema": [15, 21, 25, 30, 40],
    }

    combos = 1
    for v in grid_4h.values():
        combos *= len(v)
    print(f"Testing {combos} combinations...")
    t0 = time.time()

    r_4h = run_sweep(RsiBbMtfStrategy, df_4h, grid_4h, config=config,
                     rank_by="sharpe_ratio", top=10)
    print(f"Done in {time.time()-t0:.1f}s")
    print_results("TOP 10 MTF SWING (4h) — Sharpe:", r_4h)

    # Also rank by calmar
    r_4h_calmar = run_sweep(RsiBbMtfStrategy, df_4h, grid_4h, config=config,
                            rank_by="calmar_ratio", top=5)
    print_results("\nTOP 5 MTF SWING (4h) — Calmar:", r_4h_calmar)

    # Long-only
    config_long = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005, long_only=True)
    r_4h_long = run_sweep(RsiBbMtfStrategy, df_4h, grid_4h, config=config_long,
                          rank_by="sharpe_ratio", top=5)
    print_results("\nTOP 5 MTF SWING (4h) — Long-Only:", r_4h_long)

    # ============================================================
    # INTRADAY: MTF on 1h — fine-tune
    # Best was: rsi_period=30, overbought=65, oversold=32, bb_period=15, bb_std=2.0, daily_ema=21
    # ============================================================
    print("\n\n" + "=" * 100)
    print("FINE-TUNING: MTF RSI+BB | 1h (INTRADAY)")
    print("=" * 100)

    df_1h = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                         start="2025-01-01", end="2025-12-31", resample="1h")
    print(f"Data: {len(df_1h)} bars")

    grid_1h = {
        "rsi_period": [24, 26, 28, 30, 32, 35],
        "overbought": [60, 62, 65, 67, 70],
        "oversold": [25, 28, 30, 32, 35],
        "bb_period": [12, 14, 15, 17, 20],
        "bb_std": [1.75, 2.0, 2.25, 2.5],
        "daily_ema": [10, 15, 21, 25, 30],
    }

    combos = 1
    for v in grid_1h.values():
        combos *= len(v)
    print(f"Testing {combos} combinations...")
    t0 = time.time()

    r_1h = run_sweep(RsiBbMtfStrategy, df_1h, grid_1h, config=config,
                     rank_by="sharpe_ratio", top=10)
    print(f"Done in {time.time()-t0:.1f}s")
    print_results("TOP 10 MTF INTRADAY (1h) — Sharpe:", r_1h)

    r_1h_calmar = run_sweep(RsiBbMtfStrategy, df_1h, grid_1h, config=config,
                            rank_by="calmar_ratio", top=5)
    print_results("\nTOP 5 MTF INTRADAY (1h) — Calmar:", r_1h_calmar)

    r_1h_long = run_sweep(RsiBbMtfStrategy, df_1h, grid_1h, config=config_long,
                          rank_by="sharpe_ratio", top=5)
    print_results("\nTOP 5 MTF INTRADAY (1h) — Long-Only:", r_1h_long)

    print("\n\nDONE!")
