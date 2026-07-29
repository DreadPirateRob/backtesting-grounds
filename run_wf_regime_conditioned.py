#!/usr/bin/env python3
"""Regime-conditioned walk-forward for RSI+Bollinger.

Runs standard walk-forward but post-filters: only evaluates OOS performance
during bars classified as 'sideways' regime. This tests whether RSI+BB
works when deployed with a regime filter (as intended).

Also runs with 6-month OOS windows to address the low-trade-count problem.
"""

import sys
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.walk_forward import (
    WalkForwardConfig, WFSegment, WalkForwardResult,
    label_regimes, compute_param_stability, print_wf_report,
    _build_segment_boundaries, _regime_pcts, _regime_mismatch, _ols_slope_tstat,
)
from backtester.sweep import run_sweep
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy


DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"


def run_regime_conditioned_wf(
    strategy_class,
    df: pd.DataFrame,
    grid: dict,
    config: BacktestConfig,
    wf_config: WalkForwardConfig,
    target_regime: str = "sideways",
    min_regime_pct: float = 0.30,
) -> WalkForwardResult:
    """Walk-forward with regime-conditioned OOS evaluation.

    For each segment:
    1. IS sweep finds best params (across all regimes in IS window)
    2. Apply best params to full OOS window to generate signals
    3. Zero out signals on bars where regime != target_regime
    4. Evaluate metrics only on filtered signals
    """
    regimes = label_regimes(df)
    data_start = df.index[0]
    data_end = df.index[-1]

    boundaries = _build_segment_boundaries(data_start, data_end, wf_config)
    if not boundaries:
        raise ValueError("No valid walk-forward segments.")

    print(
        f"[regime-wf] {len(boundaries)} segments, target_regime={target_regime}, "
        f"min_pct={min_regime_pct:.0%}",
        file=sys.stderr,
    )

    wf_segments = []

    for seg_id, is_start, is_end, oos_start, oos_end in boundaries:
        print(
            f"[regime-wf] Segment {seg_id}: "
            f"IS {is_start.strftime('%Y-%m-%d')} to {is_end.strftime('%Y-%m-%d')}, "
            f"OOS {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}",
            file=sys.stderr,
        )

        is_df = df.loc[is_start:is_end - pd.Timedelta(seconds=1)]
        oos_df = df.loc[oos_start:oos_end - pd.Timedelta(seconds=1)]
        if len(is_df) == 0 or len(oos_df) == 0:
            continue

        # Check OOS regime composition
        oos_regimes = regimes.loc[oos_df.index[0]:oos_df.index[-1]]
        oos_regime_pcts = _regime_pcts(oos_regimes)
        sideways_pct = oos_regime_pcts.get("sideways", 0)

        if sideways_pct < min_regime_pct:
            print(
                f"[regime-wf] Segment {seg_id}: only {sideways_pct:.0%} sideways "
                f"(min={min_regime_pct:.0%}), skipping",
                file=sys.stderr,
            )
            continue

        # IS sweep
        sweep_result = run_sweep(
            strategy_class=strategy_class,
            df=is_df,
            grid=grid,
            config=config,
            rank_by="sharpe_ratio",
            top=1,
            n_jobs=wf_config.n_jobs,
        )
        if not sweep_result["results"]:
            continue

        best = sweep_result["results"][0]
        best_params = best["params"]
        is_sharpe = best["metrics"]["sharpe_ratio"]

        # OOS: generate signals, then zero out non-sideways bars
        strategy = strategy_class(**best_params)
        oos_signals = strategy.generate_signals(oos_df)

        # Regime filter: only trade during target regime
        oos_bar_regimes = regimes.reindex(oos_df.index).fillna("sideways")
        regime_mask = oos_bar_regimes == target_regime
        filtered_signals = oos_signals.copy()
        filtered_signals[~regime_mask] = 0

        oos_results = run_backtest(oos_df, filtered_signals, config)
        oos_metrics = compute_metrics(oos_results)

        oos_sharpe = oos_metrics["sharpe_ratio"]
        oos_return_pct = oos_metrics["total_return_pct"]
        oos_max_dd_pct = oos_metrics["max_drawdown_pct"]
        oos_trades = oos_metrics["total_trades"]
        oos_equity = oos_results["equity_curve"]

        wfer = oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0

        is_regime_pcts = _regime_pcts(regimes.loc[is_df.index[0]:is_df.index[-1]])
        mismatch = _regime_mismatch(is_regime_pcts, oos_regime_pcts)

        seg = WFSegment(
            segment_id=seg_id,
            is_start=str(is_start.date()),
            is_end=str(is_end.date()),
            oos_start=str(oos_start.date()),
            oos_end=str(oos_end.date()),
            best_params=best_params,
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            oos_return_pct=oos_return_pct,
            oos_max_dd_pct=oos_max_dd_pct,
            oos_trades=oos_trades,
            wfer=round(wfer, 4),
            is_regime_pcts={k: round(v, 4) for k, v in is_regime_pcts.items()},
            oos_regime_pcts={k: round(v, 4) for k, v in oos_regime_pcts.items()},
            regime_mismatch=round(mismatch, 4),
            oos_equity=oos_equity,
        )
        wf_segments.append(seg)

        if oos_trades < wf_config.min_trades_oos:
            print(
                f"[regime-wf] Segment {seg_id}: {oos_trades} trades "
                f"(sideways={sideways_pct:.0%})",
                file=sys.stderr,
            )

    if not wf_segments:
        raise ValueError("No valid regime-conditioned segments.")

    # Combined OOS equity
    combined_parts = []
    running_capital = wf_segments[0].oos_equity.iloc[0]
    for seg in wf_segments:
        eq = seg.oos_equity
        scale = running_capital / eq.iloc[0] if eq.iloc[0] != 0 else 1.0
        scaled = eq * scale
        combined_parts.append(scaled)
        running_capital = scaled.iloc[-1]

    combined_oos_equity = pd.concat(combined_parts)
    combined_oos_equity = combined_oos_equity[~combined_oos_equity.index.duplicated(keep="last")]

    # Overall metrics
    is_sharpes = np.array([s.is_sharpe for s in wf_segments])
    oos_sharpes = np.array([s.oos_sharpe for s in wf_segments])
    mean_is = is_sharpes.mean()
    overall_wfer = oos_sharpes.mean() / mean_is if mean_is != 0 else 0.0

    combined_returns = combined_oos_equity.pct_change().dropna()
    if len(combined_returns) > 1:
        duration_days = (combined_returns.index[-1] - combined_returns.index[0]).total_seconds() / 86400
        years = duration_days / 365.25 if duration_days > 0 else 1.0
        bar_seconds = (combined_returns.index[1] - combined_returns.index[0]).total_seconds()
        if bar_seconds < 3600 and len(combined_returns) > 60:
            hourly = (1 + combined_returns).resample("1h").prod() - 1
            hourly = hourly.dropna()
            h_bpy = len(hourly) / years if years > 0 else 8760
            overall_oos_sharpe = float(hourly.mean() / hourly.std() * np.sqrt(h_bpy)) if hourly.std() > 0 else 0.0
        else:
            bpy = len(combined_returns) / years if years > 0 else 8760
            overall_oos_sharpe = float(combined_returns.mean() / combined_returns.std() * np.sqrt(bpy)) if combined_returns.std() > 0 else 0.0
    else:
        overall_oos_sharpe = 0.0

    wfers = np.array([s.wfer for s in wf_segments])
    wfer_slope, wfer_slope_tstat = _ols_slope_tstat(wfers)
    param_stability = compute_param_stability(wf_segments)

    return WalkForwardResult(
        segments=wf_segments,
        combined_oos_equity=combined_oos_equity,
        overall_wfer=round(overall_wfer, 4),
        overall_oos_sharpe=round(overall_oos_sharpe, 4),
        param_stability=param_stability,
        wfer_slope=round(wfer_slope, 6),
        wfer_slope_tstat=round(wfer_slope_tstat, 4),
    )


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    grid = {
        "rsi_period": [21, 28, 30, 35],
        "overbought": [60, 65, 70],
        "oversold": [28, 30, 32, 38],
        "bb_period": [12, 15, 20],
        "bb_std": [1.8, 2.0, 2.2],
    }

    # --- 1h, 3mo OOS, regime-conditioned ---
    print("=" * 80, file=sys.stderr)
    print("RSI+BB REGIME-CONDITIONED — 1h — Rolling 18mo IS / 3mo OOS — sideways only", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    df_1h = load_candles(DATA_PATH, resample="1h")
    print(f"Data: {len(df_1h)} bars, {df_1h.index[0].date()} to {df_1h.index[-1].date()}", file=sys.stderr)

    wf_1h_3mo = run_regime_conditioned_wf(
        strategy_class=RsiBollingerStrategy,
        df=df_1h,
        grid=grid,
        config=config,
        wf_config=WalkForwardConfig(is_months=18, oos_months=3, mode="rolling"),
        target_regime="sideways",
        min_regime_pct=0.30,
    )
    print_wf_report(wf_1h_3mo)

    # --- 1h, 6mo OOS, regime-conditioned ---
    print("=" * 80, file=sys.stderr)
    print("RSI+BB REGIME-CONDITIONED — 1h — Rolling 18mo IS / 6mo OOS — sideways only", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    wf_1h_6mo = run_regime_conditioned_wf(
        strategy_class=RsiBollingerStrategy,
        df=df_1h,
        grid=grid,
        config=config,
        wf_config=WalkForwardConfig(is_months=18, oos_months=6, mode="rolling"),
        target_regime="sideways",
        min_regime_pct=0.30,
    )
    print_wf_report(wf_1h_6mo)

    # --- 4h, 6mo OOS, regime-conditioned ---
    print("=" * 80, file=sys.stderr)
    print("RSI+BB REGIME-CONDITIONED — 4h — Rolling 18mo IS / 6mo OOS — sideways only", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    df_4h = load_candles(DATA_PATH, resample="4h")
    print(f"Data: {len(df_4h)} bars, {df_4h.index[0].date()} to {df_4h.index[-1].date()}", file=sys.stderr)

    wf_4h_6mo = run_regime_conditioned_wf(
        strategy_class=RsiBollingerStrategy,
        df=df_4h,
        grid=grid,
        config=config,
        wf_config=WalkForwardConfig(is_months=18, oos_months=6, mode="rolling"),
        target_regime="sideways",
        min_regime_pct=0.30,
    )
    print_wf_report(wf_4h_6mo)


if __name__ == "__main__":
    main()
