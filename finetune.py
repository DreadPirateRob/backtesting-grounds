#!/usr/bin/env python3
"""Fine-tune the winning strategies with expanded parameter grids."""

import sys
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.sweep import run_sweep
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy


def print_results(title, result):
    print(f"\n{title}")
    print(f"{'Rank':<5} {'Sharpe':>7} {'Sortino':>8} {'Return%':>9} {'MaxDD%':>8} {'WinR%':>7} {'PF':>7} {'Calmar':>7} {'Trades':>7}  Params")
    print("-" * 130)
    for r in result["results"]:
        m = r["metrics"]
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        sharpe = float(m.get("sharpe_ratio", 0) or 0)
        sortino = float(m.get("sortino_ratio", 0) or 0)
        ret = float(m.get("total_return_pct", 0) or 0)
        dd = float(m.get("max_drawdown_pct", 0) or 0)
        wr = float(m.get("win_rate_pct", 0) or 0)
        pf = float(m.get("profit_factor", 0) or 0)
        cal = float(m.get("calmar_ratio", 0) or 0)
        trades = int(float(m.get("total_trades", 0) or 0))
        print(f"#{r['rank']:<4} {sharpe:>7.2f} {sortino:>8.2f} {ret:>8.1f}% {dd:>7.1f}% {wr:>6.1f}% {pf:>7.2f} {cal:>7.2f} {trades:>7}  {params_str}")


if __name__ == "__main__":
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)
    config_long = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005, long_only=True)

    # ========================================
    # SWING: RSI Bollinger on 4h
    # ========================================
    print("=" * 80)
    print("FINE-TUNING: RSI Bollinger | 4h (SWING)")
    print("=" * 80)

    df_4h = load_candles("data/binance_BTCUSDT_1m_klines.csv", start="2025-01-01", end="2025-12-31", resample="4h")
    print(f"Data: {len(df_4h)} bars")

    fine_grid_swing = {
        "rsi_period": [10, 12, 14, 16, 18, 21],
        "overbought": [60, 62, 65, 67, 70],
        "oversold": [25, 28, 30, 32, 35],
        "bb_period": [15, 18, 20, 22, 25],
        "bb_std": [1.75, 2.0, 2.25, 2.5],
    }

    combos = 1
    for v in fine_grid_swing.values():
        combos *= len(v)
    print(f"Testing {combos} combinations...")

    result_swing = run_sweep(RsiBollingerStrategy, df_4h, fine_grid_swing, config=config,
                             rank_by="sharpe_ratio", top=10)
    print_results("TOP 10 SWING (4h) - Ranked by Sharpe:", result_swing)

    result_swing_calmar = run_sweep(RsiBollingerStrategy, df_4h, fine_grid_swing, config=config,
                                    rank_by="calmar_ratio", top=5)
    print_results("\nTOP 5 SWING (4h) - Ranked by Calmar:", result_swing_calmar)

    print("\n\n--- LONG-ONLY SWING TEST ---")
    result_swing_long = run_sweep(RsiBollingerStrategy, df_4h, fine_grid_swing, config=config_long,
                                  rank_by="sharpe_ratio", top=5)
    print_results("TOP 5 SWING (4h) LONG-ONLY:", result_swing_long)

    # ========================================
    # INTRADAY: RSI Bollinger on 1h
    # ========================================
    print("\n\n" + "=" * 80)
    print("FINE-TUNING: RSI Bollinger | 1h (INTRADAY)")
    print("=" * 80)

    df_1h = load_candles("data/binance_BTCUSDT_1m_klines.csv", start="2025-01-01", end="2025-12-31", resample="1h")
    print(f"Data: {len(df_1h)} bars")

    fine_grid_intraday = {
        "rsi_period": [24, 26, 28, 30, 32, 35],
        "overbought": [60, 62, 65, 67, 70],
        "oversold": [25, 28, 30, 32, 35],
        "bb_period": [12, 14, 15, 17, 20],
        "bb_std": [1.75, 2.0, 2.25, 2.5],
    }

    combos = 1
    for v in fine_grid_intraday.values():
        combos *= len(v)
    print(f"Testing {combos} combinations...")

    result_intraday = run_sweep(RsiBollingerStrategy, df_1h, fine_grid_intraday, config=config,
                                rank_by="sharpe_ratio", top=10)
    print_results("TOP 10 INTRADAY (1h) - Ranked by Sharpe:", result_intraday)

    result_intraday_calmar = run_sweep(RsiBollingerStrategy, df_1h, fine_grid_intraday, config=config,
                                       rank_by="calmar_ratio", top=5)
    print_results("\nTOP 5 INTRADAY (1h) - Ranked by Calmar:", result_intraday_calmar)

    print("\n\n--- LONG-ONLY INTRADAY TEST ---")
    result_intraday_long = run_sweep(RsiBollingerStrategy, df_1h, fine_grid_intraday, config=config_long,
                                     rank_by="sharpe_ratio", top=5)
    print_results("TOP 5 INTRADAY (1h) LONG-ONLY:", result_intraday_long)

    print("\n\nDONE!")
