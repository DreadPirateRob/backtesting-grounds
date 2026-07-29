#!/usr/bin/env python3
"""Walk-forward multi-asset ATR Breakout portfolio with position-sizing overlays.

Extends run_wf_multiasset_portfolio.py by applying three sizing methods
on top of the raw equal-weight OOS portfolio returns:

  1. Volatility Targeting  — scale positions to hit a target annualized vol
  2. Drawdown-based Scaling — reduce exposure when in deep drawdown
  3. Half-Kelly            — size based on rolling Sharpe estimate

All sizing methods are post-processing transforms: the underlying WF sweeps
and OOS backtests are identical to the base multi-asset portfolio.
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


DATA_FILES = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

ASSET_NAMES = list(DATA_FILES.keys())

GRID = {
    "sma_period": [10, 15, 20, 30, 40],
    "atr_period": [10, 14, 20],
    "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
}


# ---------------------------------------------------------------------------
# Position sizing overlays — all operate on a pd.Series of per-bar returns
# ---------------------------------------------------------------------------

def apply_vol_target(returns: pd.Series, target_vol: float,
                     lookback: int = 720, min_scale: float = 0.5,
                     max_scale: float = 2.0) -> pd.Series:
    """Scale returns to target a fixed annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Raw portfolio returns (1h bars assumed).
    target_vol : float
        Target annualized volatility (e.g. 0.15 for 15%).
    lookback : int
        Number of trailing bars for realized vol estimation (720 = ~30d of 1h).
    min_scale, max_scale : float
        Leverage clamp bounds.
    """
    # Realized vol: annualize from hourly (8760 hours/year)
    rolling_std = returns.rolling(window=lookback, min_periods=max(lookback // 2, 60)).std()
    realized_vol = rolling_std * np.sqrt(8760)

    scale = target_vol / realized_vol
    scale = scale.clip(lower=min_scale, upper=max_scale)

    # Shift scale by 1 bar so we size based on past info only (no lookahead)
    scale = scale.shift(1).fillna(1.0)

    return returns * scale


def apply_dd_scaling(returns: pd.Series, dd_threshold: float,
                     recovery_mult: float = 0.5,
                     reduced_size: float = 0.5) -> pd.Series:
    """Reduce position when portfolio drawdown exceeds threshold.

    Parameters
    ----------
    returns : pd.Series
        Raw portfolio returns.
    dd_threshold : float
        Drawdown level to trigger reduction (e.g. 0.10 for 10%).
    recovery_mult : float
        Fraction of threshold at which to restore full size (e.g. 0.5 => 5% if threshold=10%).
    reduced_size : float
        Position size multiplier when in drawdown reduction mode.
    """
    equity = (1 + returns).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax  # negative values

    scale = pd.Series(1.0, index=returns.index)
    in_reduction = False

    for i in range(len(returns)):
        dd = drawdown.iloc[i]
        if not in_reduction:
            if dd < -dd_threshold:
                in_reduction = True
                scale.iloc[i] = reduced_size
            else:
                scale.iloc[i] = 1.0
        else:
            if dd > -(dd_threshold * recovery_mult):
                in_reduction = False
                scale.iloc[i] = 1.0
            else:
                scale.iloc[i] = reduced_size

    # Shift by 1 bar: decision at bar i applies at bar i+1
    scale = scale.shift(1).fillna(1.0)

    return returns * scale


def apply_half_kelly(returns: pd.Series, lookback: int = 720,
                     min_scale: float = 0.25,
                     max_scale: float = 1.5) -> pd.Series:
    """Apply half-Kelly sizing based on rolling Sharpe estimate.

    Kelly fraction for continuous returns ~ Sharpe^2 / (Sharpe^2 + 1).
    Half-Kelly = Kelly / 2.

    Parameters
    ----------
    returns : pd.Series
        Raw portfolio returns (1h bars).
    lookback : int
        Trailing window for Sharpe estimation.
    min_scale, max_scale : float
        Clamp bounds for the Kelly fraction.
    """
    rolling_mean = returns.rolling(window=lookback, min_periods=max(lookback // 2, 60)).mean()
    rolling_std = returns.rolling(window=lookback, min_periods=max(lookback // 2, 60)).std()

    # Annualized Sharpe
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(8760)
    rolling_sharpe = rolling_sharpe.fillna(0)

    # Kelly fraction: S^2 / (S^2 + 1)
    s2 = rolling_sharpe ** 2
    kelly = s2 / (s2 + 1)
    half_kelly = kelly / 2.0

    scale = half_kelly.clip(lower=min_scale, upper=max_scale)
    # Shift by 1 bar to avoid lookahead
    scale = scale.shift(1).fillna(min_scale)

    return returns * scale


# ---------------------------------------------------------------------------
# Metrics helpers for combined equity
# ---------------------------------------------------------------------------

def compute_equity_metrics(returns: pd.Series, initial_capital: float = 10_000):
    """Compute Sharpe, total return, max DD from a return series (1h bars)."""
    equity = initial_capital * (1 + returns).cumprod()
    duration_days = (returns.index[-1] - returns.index[0]).total_seconds() / 86400
    years = duration_days / 365.25 if duration_days > 0 else 1.0

    # Sharpe at 1h
    mean_ret = returns.mean()
    std_ret = returns.std()
    bars_per_year = len(returns) / years if years > 0 else 8760
    sharpe = (mean_ret / std_ret * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cummax = equity.cummax()
    max_dd = float(((equity - cummax) / cummax).min()) * 100

    return {
        "sharpe": round(sharpe, 4),
        "total_return_pct": round(total_ret, 2),
        "max_dd_pct": round(max_dd, 2),
        "equity": equity,
    }


def compute_per_segment_sharpe(returns: pd.Series):
    """Annualized Sharpe for a single OOS segment (1h bars)."""
    if len(returns) < 10:
        return 0.0
    duration_days = (returns.index[-1] - returns.index[0]).total_seconds() / 86400
    years = duration_days / 365.25 if duration_days > 0 else 1.0
    bars_per_year = len(returns) / years if years > 0 else 8760
    mean_ret = returns.mean()
    std_ret = returns.std()
    return float(mean_ret / std_ret * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)

    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)
    wf_config = WalkForwardConfig(is_months=18, oos_months=3, mode="rolling")
    resample = "1h"

    p(f"\n{'=' * 130}")
    p(f"MULTI-ASSET ATR BREAKOUT PORTFOLIO — VOLATILITY TARGETING / POSITION SIZING OVERLAY")
    p(f"Assets: {', '.join(ASSET_NAMES)} — Equal Weight (1/3 each) — {resample}")
    p(f"Rolling 18mo IS / 3mo OOS — fee=10bps + slip=5bps")
    p(f"{'=' * 130}")

    # ------------------------------------------------------------------
    # Step 1: Load all asset data
    # ------------------------------------------------------------------
    asset_dfs = {}
    for name, path in DATA_FILES.items():
        try:
            df = load_candles(path, resample=resample)
            asset_dfs[name] = df
            p(f"  {name}: {len(df):,} bars, {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            p(f"  {name}: FAILED to load — {e}")

    if len(asset_dfs) < 2:
        p("ERROR: Need at least 2 assets. Aborting.")
        return

    common_start = max(df.index[0] for df in asset_dfs.values())
    common_end = min(df.index[-1] for df in asset_dfs.values())
    p(f"\nCommon date range: {common_start.date()} to {common_end.date()}")

    boundaries = _build_segment_boundaries(common_start, common_end, wf_config)
    p(f"[wf] {len(boundaries)} segments\n")

    if not boundaries:
        p("ERROR: No valid segments. Aborting.")
        return

    # ------------------------------------------------------------------
    # Step 2: Run WF sweeps, collect raw OOS portfolio returns per segment
    # ------------------------------------------------------------------
    segment_oos_returns = []  # list of pd.Series (raw EW portfolio returns)
    segment_meta = []         # metadata per segment

    for seg_id, is_start, is_end, oos_start, oos_end in boundaries:
        p(
            f"[wf] Segment {seg_id}: "
            f"IS {is_start.strftime('%Y-%m-%d')} to {is_end.strftime('%Y-%m-%d')}, "
            f"OOS {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}"
        )

        oos_return_series = {}
        is_sharpes = {}

        for asset_name, df in asset_dfs.items():
            is_df = df.loc[is_start:is_end - pd.Timedelta(seconds=1)]
            oos_df = df.loc[oos_start:oos_end - pd.Timedelta(seconds=1)]

            if len(is_df) == 0 or len(oos_df) == 0:
                p(f"  {asset_name}: empty IS or OOS slice, skipping")
                continue

            sweep_result = run_sweep(
                strategy_class=AtrBreakoutStrategy,
                df=is_df,
                grid=GRID,
                config=config,
                rank_by="sharpe_ratio",
                top=1,
            )

            if not sweep_result["results"]:
                p(f"  {asset_name}: sweep returned no results, skipping")
                continue

            best = sweep_result["results"][0]
            best_params = best["params"]
            is_sharpes[asset_name] = best["metrics"]["sharpe_ratio"]

            strategy = AtrBreakoutStrategy(**best_params)
            oos_signals = strategy.generate_signals(oos_df)
            oos_results = run_backtest(oos_df, oos_signals, config)
            oos_return_series[asset_name] = oos_results["strategy_returns"]

            p(
                f"  {asset_name}: IS Sharpe={is_sharpes[asset_name]:.2f}, "
                f"params={best_params}"
            )

        if len(oos_return_series) < 2:
            p(f"  Segment {seg_id}: fewer than 2 assets succeeded, skipping\n")
            continue

        # Equal-weight portfolio returns
        returns_df = pd.DataFrame(oos_return_series)
        portfolio_returns = returns_df.mean(axis=1)

        segment_oos_returns.append(portfolio_returns)
        segment_meta.append({
            "seg_id": seg_id,
            "oos_period": f"{oos_start.strftime('%Y-%m-%d')} -> {oos_end.strftime('%Y-%m-%d')}",
            "avg_is_sharpe": np.mean(list(is_sharpes.values())),
        })
        p("")

    if not segment_oos_returns:
        p("ERROR: No valid segments produced results!")
        return

    p(f"[wf] {len(segment_oos_returns)} segments with valid OOS returns\n")

    # ------------------------------------------------------------------
    # Step 3: Concatenate all OOS returns into a single series
    # ------------------------------------------------------------------
    all_oos_returns = pd.concat(segment_oos_returns)
    all_oos_returns = all_oos_returns[~all_oos_returns.index.duplicated(keep="last")]
    all_oos_returns = all_oos_returns.sort_index()

    # ------------------------------------------------------------------
    # Step 4: Define all sizing methods
    # ------------------------------------------------------------------
    methods = {}

    # Raw (baseline)
    methods["Raw EW Portfolio"] = all_oos_returns.copy()

    # Vol targeting at different targets and lookbacks
    for target_pct in [15, 20, 25]:
        target = target_pct / 100.0
        for lb_days in [30, 60]:
            lb_bars = lb_days * 24  # 1h bars
            label = f"VolTarget {target_pct}% ({lb_days}d)"
            methods[label] = apply_vol_target(
                all_oos_returns.copy(), target_vol=target, lookback=lb_bars
            )

    # Drawdown-based scaling
    for dd_pct in [10, 15, 20]:
        dd_thresh = dd_pct / 100.0
        label = f"DD-Scale {dd_pct}%"
        methods[label] = apply_dd_scaling(
            all_oos_returns.copy(), dd_threshold=dd_thresh
        )

    # Half-Kelly with different lookbacks
    for lb_days in [30, 60]:
        lb_bars = lb_days * 24
        label = f"Half-Kelly ({lb_days}d)"
        methods[label] = apply_half_kelly(
            all_oos_returns.copy(), lookback=lb_bars
        )

    # ------------------------------------------------------------------
    # Step 5: Compute metrics for each method
    # ------------------------------------------------------------------
    results_table = []

    for method_name, rets in methods.items():
        # Combined metrics across all OOS
        combined = compute_equity_metrics(rets, initial_capital=config.initial_capital)

        # Per-segment Sharpe
        seg_sharpes = []
        for seg_ret in segment_oos_returns:
            # Extract the corresponding sized returns for this segment's time range
            mask = rets.index.isin(seg_ret.index)
            seg_sized = rets[mask]
            if len(seg_sized) > 10:
                seg_sharpes.append(compute_per_segment_sharpe(seg_sized))

        mean_seg_sharpe = np.mean(seg_sharpes) if seg_sharpes else 0.0
        median_seg_sharpe = np.median(seg_sharpes) if seg_sharpes else 0.0

        calmar = (combined["sharpe"] / abs(combined["max_dd_pct"] / 100.0)
                  if combined["max_dd_pct"] != 0 else 0.0)

        results_table.append({
            "method": method_name,
            "combined_sharpe": combined["sharpe"],
            "mean_seg_sharpe": round(mean_seg_sharpe, 4),
            "median_seg_sharpe": round(median_seg_sharpe, 4),
            "total_return_pct": combined["total_return_pct"],
            "max_dd_pct": combined["max_dd_pct"],
            "calmar": round(calmar, 4),
        })

    # ------------------------------------------------------------------
    # Step 6: Report
    # ------------------------------------------------------------------
    p(f"\n{'=' * 130}")
    p(f"POSITION SIZING OVERLAY — COMPARISON TABLE")
    p(f"{'=' * 130}")

    header = (
        f"{'Method':<30}  {'Comb Sharpe':>11}  {'Mean Seg Shrp':>13}  "
        f"{'Med Seg Shrp':>12}  {'Total Ret%':>10}  {'MaxDD%':>8}  {'Calmar':>8}"
    )
    p(header)
    p("-" * 130)

    # Sort by Calmar descending
    results_table.sort(key=lambda x: x["calmar"], reverse=True)

    raw_result = next(r for r in results_table if r["method"] == "Raw EW Portfolio")

    for r in results_table:
        # Flag improvements
        dd_improved = r["max_dd_pct"] > raw_result["max_dd_pct"]  # less negative = better
        sharpe_preserved = r["combined_sharpe"] >= raw_result["combined_sharpe"] * 0.85
        flag = ""
        if dd_improved and sharpe_preserved and r["method"] != "Raw EW Portfolio":
            flag = " ***"
        elif dd_improved and r["method"] != "Raw EW Portfolio":
            flag = " **"
        elif r["method"] == "Raw EW Portfolio":
            flag = " [BASELINE]"

        p(
            f"{r['method']:<30}  {r['combined_sharpe']:>11.4f}  "
            f"{r['mean_seg_sharpe']:>13.4f}  {r['median_seg_sharpe']:>12.4f}  "
            f"{r['total_return_pct']:>10.2f}  {r['max_dd_pct']:>8.2f}  "
            f"{r['calmar']:>8.4f}{flag}"
        )

    p("-" * 130)
    p("Legend: *** = DD improved + Sharpe preserved (>85% of baseline)")
    p("        **  = DD improved but Sharpe degraded")

    # ------------------------------------------------------------------
    # Step 7: Per-segment breakdown for top methods
    # ------------------------------------------------------------------
    p(f"\n{'=' * 130}")
    p(f"PER-SEGMENT SHARPE BREAKDOWN — TOP METHODS")
    p(f"{'=' * 130}")

    # Show raw + top 3 by Calmar
    show_methods = ["Raw EW Portfolio"] + [
        r["method"] for r in results_table
        if r["method"] != "Raw EW Portfolio"
    ][:3]

    header_parts = [f"{'Seg':>3}  {'OOS Period':>25}"]
    for m in show_methods:
        short_name = m[:18]
        header_parts.append(f"{short_name:>18}")
    p("  ".join(header_parts))
    p("-" * 130)

    for i, (seg_ret, meta) in enumerate(zip(segment_oos_returns, segment_meta)):
        line_parts = [f"{meta['seg_id']:>3}  {meta['oos_period']:>25}"]

        for method_name in show_methods:
            rets = methods[method_name]
            mask = rets.index.isin(seg_ret.index)
            seg_sized = rets[mask]
            if len(seg_sized) > 10:
                s = compute_per_segment_sharpe(seg_sized)
            else:
                s = float("nan")
            line_parts.append(f"{s:>18.2f}")

        p("  ".join(line_parts))

    p("-" * 130)

    # ------------------------------------------------------------------
    # Step 8: Detailed analysis of best method vs raw
    # ------------------------------------------------------------------
    best = results_table[0]  # already sorted by Calmar
    p(f"\nBEST METHOD BY CALMAR: {best['method']}")
    p(f"  Combined Sharpe: {best['combined_sharpe']:.4f}  (raw: {raw_result['combined_sharpe']:.4f}, "
      f"delta: {best['combined_sharpe'] - raw_result['combined_sharpe']:+.4f})")
    p(f"  Total Return:    {best['total_return_pct']:.2f}%  (raw: {raw_result['total_return_pct']:.2f}%)")
    p(f"  Max Drawdown:    {best['max_dd_pct']:.2f}%  (raw: {raw_result['max_dd_pct']:.2f}%)")
    p(f"  Calmar Ratio:    {best['calmar']:.4f}  (raw: {raw_result['calmar']:.4f})")

    dd_reduction = abs(best["max_dd_pct"]) - abs(raw_result["max_dd_pct"])
    if dd_reduction < 0:
        p(f"  DD Reduction:    {abs(dd_reduction):.2f}pp")
    else:
        p(f"  DD Increase:     {dd_reduction:.2f}pp")

    p(f"\n{'=' * 130}\n")


if __name__ == "__main__":
    main()
