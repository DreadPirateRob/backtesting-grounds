#!/usr/bin/env python3
"""Walk-forward analysis for ATR Breakout + RSI+BB equal-weight portfolio.

For each WF segment:
1. Sweep ATR Breakout on IS → best params → OOS returns
2. Sweep RSI+BB on IS → best params → OOS returns
3. Combine OOS returns with equal weight
4. Compute portfolio OOS metrics
"""

import sys
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.sweep import run_sweep
from backtester.walk_forward import (
    WalkForwardConfig, _build_segment_boundaries, _ols_slope_tstat,
)
from strategies.atr_breakout_strategy import AtrBreakoutStrategy
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy


DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)
    wf_config = WalkForwardConfig(is_months=18, oos_months=3, mode="rolling")

    atr_grid = {
        "sma_period": [10, 15, 20, 30, 40],
        "atr_period": [10, 14, 20],
        "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    }
    rsi_grid = {
        "rsi_period": [21, 28, 30, 35],
        "overbought": [60, 65, 70],
        "oversold": [28, 30, 32, 38],
        "bb_period": [12, 15, 20],
        "bb_std": [1.8, 2.0, 2.2],
    }

    strategies = [
        ("atr_breakout", AtrBreakoutStrategy, atr_grid),
        ("rsi_bollinger", RsiBollingerStrategy, rsi_grid),
    ]

    for resample, tf_label in [("1h", "1h"), ("4h", "4h")]:
        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"PORTFOLIO WF (ATR Breakout + RSI+BB EW) — {tf_label} — Rolling 18mo IS / 3mo OOS", file=sys.stderr)
        print(f"{'=' * 80}", file=sys.stderr)

        df = load_candles(DATA_PATH, resample=resample)
        print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}", file=sys.stderr)

        boundaries = _build_segment_boundaries(df.index[0], df.index[-1], wf_config)
        print(f"[portfolio-wf] {len(boundaries)} segments", file=sys.stderr)

        segment_results = []

        for seg_id, is_start, is_end, oos_start, oos_end in boundaries:
            print(
                f"[portfolio-wf] Segment {seg_id}: "
                f"IS {is_start.strftime('%Y-%m-%d')} to {is_end.strftime('%Y-%m-%d')}, "
                f"OOS {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}",
                file=sys.stderr,
            )

            is_df = df.loc[is_start:is_end - pd.Timedelta(seconds=1)]
            oos_df = df.loc[oos_start:oos_end - pd.Timedelta(seconds=1)]
            if len(is_df) == 0 or len(oos_df) == 0:
                continue

            # Run each strategy's IS sweep and OOS evaluation
            oos_return_series = {}
            is_sharpes = {}

            for name, strategy_cls, grid in strategies:
                sweep_result = run_sweep(
                    strategy_class=strategy_cls,
                    df=is_df,
                    grid=grid,
                    config=config,
                    rank_by="sharpe_ratio",
                    top=1,
                )
                if not sweep_result["results"]:
                    continue

                best = sweep_result["results"][0]
                best_params = best["params"]
                is_sharpes[name] = best["metrics"]["sharpe_ratio"]

                strategy = strategy_cls(**best_params)
                oos_signals = strategy.generate_signals(oos_df)
                oos_results = run_backtest(oos_df, oos_signals, config)
                oos_return_series[name] = oos_results["strategy_returns"]

            if len(oos_return_series) < 2:
                print(f"[portfolio-wf] Segment {seg_id}: not enough strategies, skipping", file=sys.stderr)
                continue

            # Equal-weight portfolio returns
            returns_df = pd.DataFrame(oos_return_series)
            portfolio_returns = returns_df.mean(axis=1)

            # Portfolio equity curve
            portfolio_equity = config.initial_capital * (1 + portfolio_returns).cumprod()

            # Compute portfolio OOS metrics
            portfolio_results = {
                "equity_curve": portfolio_equity,
                "positions": pd.Series(1, index=oos_df.index),  # always invested
                "strategy_returns": portfolio_returns,
                "trades": [],  # no individual trades for portfolio
                "fees_paid": 0,  # fees already in individual returns
            }
            oos_metrics = compute_metrics(portfolio_results)

            # Individual strategy OOS metrics for comparison
            individual_oos = {}
            for name in oos_return_series:
                eq = config.initial_capital * (1 + oos_return_series[name]).cumprod()
                ind_results = {
                    "equity_curve": eq,
                    "positions": pd.Series(1, index=oos_df.index),
                    "strategy_returns": oos_return_series[name],
                    "trades": [],
                    "fees_paid": 0,
                }
                individual_oos[name] = compute_metrics(ind_results)

            # Average IS Sharpe for WFER calculation
            avg_is_sharpe = np.mean(list(is_sharpes.values()))
            wfer = oos_metrics["sharpe_ratio"] / avg_is_sharpe if avg_is_sharpe != 0 else 0.0

            # Correlation between strategies in OOS
            corr = returns_df.corr().iloc[0, 1] if returns_df.shape[1] >= 2 else 0.0

            segment_results.append({
                "seg_id": seg_id,
                "is_period": f"{is_start.strftime('%Y-%m-%d')} -> {is_end.strftime('%Y-%m-%d')}",
                "oos_period": f"{oos_start.strftime('%Y-%m-%d')} -> {oos_end.strftime('%Y-%m-%d')}",
                "portfolio_sharpe": oos_metrics["sharpe_ratio"],
                "portfolio_return": oos_metrics["total_return_pct"],
                "portfolio_dd": oos_metrics["max_drawdown_pct"],
                "wfer": wfer,
                "corr": corr,
                "atr_sharpe": individual_oos.get("atr_breakout", {}).get("sharpe_ratio", 0),
                "rsi_sharpe": individual_oos.get("rsi_bollinger", {}).get("sharpe_ratio", 0),
                "portfolio_equity": portfolio_equity,
            })

        if not segment_results:
            print(f"[portfolio-wf] No valid segments for {tf_label}!", file=sys.stderr)
            continue

        # --- Report ---
        print(f"\n{'=' * 120}", file=sys.stderr)
        print(f"PORTFOLIO WALK-FORWARD REPORT — {tf_label}", file=sys.stderr)
        print(f"{'=' * 120}", file=sys.stderr)

        print(f"{'Seg':>3}  {'OOS Period':>25}  {'Port Shrp':>9}  {'WFER':>6}  "
              f"{'Port Ret%':>9}  {'Port DD%':>8}  {'Corr':>6}  "
              f"{'ATR Shrp':>8}  {'RSI Shrp':>8}", file=sys.stderr)
        print("-" * 110, file=sys.stderr)

        for r in segment_results:
            flag = ""
            if r["wfer"] < 0.30 and r["wfer"] != 0:
                flag = " [WEAK]"
            print(
                f"{r['seg_id']:>3}  {r['oos_period']:>25}  {r['portfolio_sharpe']:>9.2f}  "
                f"{r['wfer']:>6.2f}  {r['portfolio_return']:>9.2f}  {r['portfolio_dd']:>8.2f}  "
                f"{r['corr']:>6.2f}  {r['atr_sharpe']:>8.2f}  {r['rsi_sharpe']:>8.2f}{flag}",
                file=sys.stderr,
            )

        # Combined OOS equity
        combined_parts = []
        running_capital = segment_results[0]["portfolio_equity"].iloc[0]
        for r in segment_results:
            eq = r["portfolio_equity"]
            scale = running_capital / eq.iloc[0] if eq.iloc[0] != 0 else 1.0
            scaled = eq * scale
            combined_parts.append(scaled)
            running_capital = scaled.iloc[-1]

        combined_equity = pd.concat(combined_parts)
        combined_equity = combined_equity[~combined_equity.index.duplicated(keep="last")]

        # Overall metrics
        sharpes = np.array([r["portfolio_sharpe"] for r in segment_results])
        wfers = np.array([r["wfer"] for r in segment_results])
        corrs = np.array([r["corr"] for r in segment_results])

        total_ret = (combined_equity.iloc[-1] / combined_equity.iloc[0] - 1) * 100
        cummax = combined_equity.cummax()
        max_dd = float(((combined_equity - cummax) / cummax).min()) * 100

        wfer_slope, wfer_tstat = _ols_slope_tstat(wfers)

        wfer_label = (
            "(EXCELLENT)" if wfers.mean() > 0.70 else
            "(GOOD)" if wfers.mean() > 0.50 else
            "(ACCEPTABLE)" if wfers.mean() > 0.30 else
            "(WEAK)"
        )

        print(f"\n{'SUMMARY':}", file=sys.stderr)
        print(f"  Segments:           {len(segment_results)}", file=sys.stderr)
        print(f"  Mean OOS Sharpe:    {sharpes.mean():.4f}", file=sys.stderr)
        print(f"  Mean WFER:          {wfers.mean():.4f}  {wfer_label}", file=sys.stderr)
        print(f"  WFER Slope:         {wfer_slope:+.6f} (t={wfer_tstat:+.2f})", file=sys.stderr)
        print(f"  Mean OOS Corr:      {corrs.mean():.4f}", file=sys.stderr)
        print(f"  Combined OOS Ret:   {total_ret:.2f}%", file=sys.stderr)
        print(f"  Combined OOS MaxDD: {max_dd:.2f}%", file=sys.stderr)
        print(f"  OOS Period:         {combined_equity.index[0].date()} to {combined_equity.index[-1].date()}", file=sys.stderr)

        # Compare vs individual strategy WF results
        atr_sharpes = np.array([r["atr_sharpe"] for r in segment_results])
        rsi_sharpes = np.array([r["rsi_sharpe"] for r in segment_results])
        print(f"\n  ATR Breakout alone: mean OOS Sharpe = {atr_sharpes.mean():.4f}", file=sys.stderr)
        print(f"  RSI+BB alone:       mean OOS Sharpe = {rsi_sharpes.mean():.4f}", file=sys.stderr)
        print(f"  Portfolio:          mean OOS Sharpe = {sharpes.mean():.4f}", file=sys.stderr)
        print(f"{'=' * 120}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
