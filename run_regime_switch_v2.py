#!/usr/bin/env python3
"""Regime-switching backtest v2 — recalibrated regime detector.

Tests multiple detector calibrations (SMA 50/100 + momentum 1-3%)
to find settings that produce meaningful regime distribution,
then runs regime-switching and compares vs static ATR Breakout.
"""

import sys

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.regime_detector import (
    detect_regime_threshold, RegimeStrategyMap, generate_regime_signals,
)
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy
from strategies.dual_rsi_strategy import DualRsiStrategy
from strategies.atr_breakout_strategy import AtrBreakoutStrategy


DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"


def print_metrics_row(label, metrics):
    print(f"  {label:<50} {metrics['sharpe_ratio']:>7.2f} {metrics['total_return_pct']:>8.1f}% "
          f"{metrics['max_drawdown_pct']:>7.1f}% {metrics['total_trades']:>6} "
          f"{metrics['time_in_market_pct']:>6.1f}%")


def run_with_calibration(df, config, sma_period, momentum_thresh, momentum_bars=20):
    """Run regime detection + switching with given calibration."""
    regimes = detect_regime_threshold(
        df,
        sma_period=sma_period,
        momentum_bars=momentum_bars,
        momentum_thresh=momentum_thresh,
    )

    # Report regime distribution
    regime_counts = regimes["regime"].value_counts(normalize=True)
    bull_pct = regime_counts.get("bull", 0) * 100
    bear_pct = regime_counts.get("bear", 0) * 100
    sideways_pct = regime_counts.get("sideways", 0) * 100
    transitions = (regimes["regime"] != regimes["regime"].shift(1)).sum() - 1

    print(f"\n  SMA={sma_period}, mom_thresh={momentum_thresh:.0%}, mom_bars={momentum_bars}")
    print(f"  Distribution: bull={bull_pct:.1f}%, bear={bear_pct:.1f}%, sideways={sideways_pct:.1f}%")
    print(f"  Transitions: {transitions}")

    # Strategy map using best params from regime sweep
    regime_map = RegimeStrategyMap(
        regime_strategy={
            "sideways": (
                RsiBollingerStrategy,
                {"rsi_period": 28, "overbought": 65, "oversold": 30, "bb_period": 15, "bb_std": 2.0},
            ),
            "bull": (
                AtrBreakoutStrategy,
                {"sma_period": 10, "atr_period": 14, "atr_mult": 3.0},
            ),
            "bear": (
                DualRsiStrategy,
                {"slow_period": 60, "fast_period": 20, "slow_overbought": 65,
                 "slow_oversold": 25, "fast_overbought": 80, "fast_oversold": 20},
            ),
        },
        default_strategy=(
            AtrBreakoutStrategy,
            {"sma_period": 10, "atr_period": 14, "atr_mult": 3.0},
        ),
    )

    signals = generate_regime_signals(df, regimes["regime"], regime_map)
    results = run_backtest(df, signals, config)
    metrics = compute_metrics(results)

    return metrics, regimes


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    print("Loading 5yr 1h data...")
    df = load_candles(DATA_PATH, resample="1h")
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # --- Test multiple calibrations ---
    calibrations = [
        # (sma_period, momentum_thresh, momentum_bars)
        (200, 0.05, 20),   # Original (93.8% sideways)
        (100, 0.03, 20),   # Moderate recalibration
        (100, 0.02, 20),   # Aggressive recalibration
        (50,  0.02, 20),   # Very aggressive
        (50,  0.01, 10),   # Most sensitive
        (100, 0.03, 50),   # Longer momentum lookback
    ]

    print(f"\n{'=' * 90}")
    print("REGIME DETECTOR CALIBRATION SWEEP")
    print(f"{'=' * 90}")

    best_sharpe = -999
    best_label = ""
    all_results = []

    for sma_p, mom_t, mom_b in calibrations:
        metrics, regimes = run_with_calibration(df, config, sma_p, mom_t, mom_b)
        label = f"SMA{sma_p}_mom{mom_t:.0%}_bars{mom_b}"
        all_results.append((label, metrics, regimes))
        if metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_label = label

    # --- Static baselines ---
    print(f"\n{'=' * 90}")
    print("STATIC BASELINES")
    print(f"{'=' * 90}")

    # ATR Breakout (WF-validated params)
    static_atr = AtrBreakoutStrategy(sma_period=10, atr_period=14, atr_mult=3.0)
    atr_signals = static_atr.generate_signals(df)
    atr_results = run_backtest(df, atr_signals, config)
    atr_metrics = compute_metrics(atr_results)

    # RSI+BB
    static_rsi = RsiBollingerStrategy(rsi_period=28, overbought=65, oversold=30, bb_period=15, bb_std=2.0)
    rsi_signals = static_rsi.generate_signals(df)
    rsi_results = run_backtest(df, rsi_signals, config)
    rsi_metrics = compute_metrics(rsi_results)

    # --- Comparison table ---
    print(f"\n{'=' * 90}")
    print("FULL COMPARISON")
    print(f"{'=' * 90}")
    print(f"  {'Strategy':<50} {'Sharpe':>7} {'Return':>9} {'MaxDD':>8} {'Trades':>6} {'InMkt':>7}")
    print(f"  {'-' * 88}")

    for label, metrics, _ in all_results:
        tag = " <-- BEST" if label == best_label else ""
        print_metrics_row(f"Regime-Switch ({label}){tag}", metrics)

    print(f"  {'-' * 88}")
    print_metrics_row("Static ATR Breakout (sma10, atr14, mult3.0)", atr_metrics)
    print_metrics_row("Static RSI+BB (rsi28, ob65, os30, bb15, std2.0)", rsi_metrics)

    # --- Detailed results for best calibration ---
    best_entry = [e for e in all_results if e[0] == best_label][0]
    best_metrics = best_entry[1]
    best_regimes = best_entry[2]

    print(f"\n{'=' * 90}")
    print(f"BEST CALIBRATION: {best_label}")
    print(f"{'=' * 90}")
    regime_counts = best_regimes["regime"].value_counts()
    for r in ["bull", "bear", "sideways"]:
        count = regime_counts.get(r, 0)
        pct = count / len(df) * 100
        print(f"  {r:<12}: {count:>8,} bars ({pct:.1f}%)")
    print(f"\n  Sharpe:       {best_metrics['sharpe_ratio']:.2f}")
    print(f"  Return:       {best_metrics['total_return_pct']:.1f}%")
    print(f"  Max DD:       {best_metrics['max_drawdown_pct']:.1f}%")
    print(f"  Win Rate:     {best_metrics['win_rate_pct']:.1f}%")
    print(f"  Trades:       {best_metrics['total_trades']}")


if __name__ == "__main__":
    main()
