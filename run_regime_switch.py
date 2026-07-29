#!/usr/bin/env python3
"""Regime-switching backtest across full 5yr BTC history.

Compares dynamic regime-based strategy allocation vs static best strategy.
Uses threshold-based regime detection + best params from regime sweep.
"""

import sys

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.regime_detector import (
    detect_regime_threshold, RegimeStrategyMap, generate_regime_signals,
    print_regime_summary,
)
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy
from strategies.dual_rsi_strategy import DualRsiStrategy
from strategies.atr_breakout_strategy import AtrBreakoutStrategy


DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"


def print_metrics(label: str, metrics: dict):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Sharpe:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino:      {metrics['sortino_ratio']:.2f}")
    print(f"  Total Return: {metrics['total_return_pct']:.1f}%")
    print(f"  Annual Return:{metrics['annualized_return_pct']:.1f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Max DD Days:  {metrics['max_drawdown_duration_days']:.0f}")
    print(f"  Calmar:       {metrics['calmar_ratio']:.2f}")
    print(f"  Win Rate:     {metrics['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:{metrics['profit_factor']:.2f}")
    print(f"  Trades:       {metrics['total_trades']}")
    print(f"  Time in Mkt:  {metrics['time_in_market_pct']:.1f}%")
    print(f"  Fees Paid:    ${metrics['total_fees_paid']:.2f}")


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    print("Loading 5yr 1h data...")
    df = load_candles(DATA_PATH, resample="1h")
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # --- Regime Detection ---
    print("\nDetecting regimes...")
    regimes = detect_regime_threshold(df)
    print_regime_summary(df, regimes)

    # --- Regime-Switching Strategy Map ---
    # Best params from regime sweep results:
    # sideways → rsi_bollinger (Sharpe 3.03 at 4h, 2.52 at 1h in 2025)
    # bull → dual_rsi (Sharpe 3.72 at 4h in 2021-H2) / atr_breakout (Sharpe 3.52 at 1h)
    # bear → dual_rsi (Sharpe 2.84 at 1h in 2022-H1)
    regime_map = RegimeStrategyMap(
        regime_strategy={
            "sideways": (
                RsiBollingerStrategy,
                {"rsi_period": 28, "overbought": 65, "oversold": 30, "bb_period": 15, "bb_std": 2.0},
            ),
            "bull": (
                AtrBreakoutStrategy,
                {"sma_period": 15, "atr_period": 10, "atr_mult": 2.5},
            ),
            "bear": (
                DualRsiStrategy,
                {"slow_period": 60, "fast_period": 20, "slow_overbought": 65,
                 "slow_oversold": 25, "fast_overbought": 80, "fast_oversold": 20},
            ),
        },
        default_strategy=(
            RsiBollingerStrategy,
            {"rsi_period": 28, "overbought": 65, "oversold": 30, "bb_period": 15, "bb_std": 2.0},
        ),
    )

    print("Running regime-switching backtest...")
    regime_signals = generate_regime_signals(df, regimes["regime"], regime_map)
    regime_results = run_backtest(df, regime_signals, config)
    regime_metrics = compute_metrics(regime_results)
    print_metrics("REGIME-SWITCHING (sideways=RSI+BB, bull=ATR Breakout, bear=Dual RSI)", regime_metrics)

    # --- Static Comparisons ---
    # Static RSI+BB (best overall range-bound params)
    print("\nRunning static RSI+Bollinger backtest...")
    static_rsi_bb = RsiBollingerStrategy(
        rsi_period=28, overbought=65, oversold=30, bb_period=15, bb_std=2.0,
    )
    static_signals = static_rsi_bb.generate_signals(df)
    static_results = run_backtest(df, static_signals, config)
    static_metrics = compute_metrics(static_results)
    print_metrics("STATIC RSI+BOLLINGER (same params, no regime filter)", static_metrics)

    # Static ATR Breakout
    print("\nRunning static ATR Breakout backtest...")
    static_atr = AtrBreakoutStrategy(sma_period=15, atr_period=10, atr_mult=2.5)
    atr_signals = static_atr.generate_signals(df)
    atr_results = run_backtest(df, atr_signals, config)
    atr_metrics = compute_metrics(atr_results)
    print_metrics("STATIC ATR BREAKOUT", atr_metrics)

    # Static Dual RSI
    print("\nRunning static Dual RSI backtest...")
    static_drsi = DualRsiStrategy(
        slow_period=60, fast_period=20, slow_overbought=65,
        slow_oversold=25, fast_overbought=80, fast_oversold=20,
    )
    drsi_signals = static_drsi.generate_signals(df)
    drsi_results = run_backtest(df, drsi_signals, config)
    drsi_metrics = compute_metrics(drsi_results)
    print_metrics("STATIC DUAL RSI", drsi_metrics)

    # --- Summary Comparison ---
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<45} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7}")
    print("-" * 70)
    for label, m in [
        ("Regime-Switching", regime_metrics),
        ("Static RSI+Bollinger", static_metrics),
        ("Static ATR Breakout", atr_metrics),
        ("Static Dual RSI", drsi_metrics),
    ]:
        print(f"{label:<45} {m['sharpe_ratio']:>8.2f} {m['total_return_pct']:>8.1f}% {m['max_drawdown_pct']:>7.1f}% {m['total_trades']:>7}")


if __name__ == "__main__":
    main()
