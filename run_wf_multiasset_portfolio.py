#!/usr/bin/env python3
"""Walk-forward analysis for multi-asset ATR Breakout equal-weight portfolio.

For each WF segment:
1. Sweep ATR Breakout on each asset's IS data independently
2. Run best params on each asset's OOS data
3. Combine the 3 OOS return series with equal weight (1/3 each)
4. Compute portfolio OOS metrics (Sharpe, return, MaxDD)
5. Track pairwise correlations between assets
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

# Known individual WF results for comparison
INDIVIDUAL_WF = {
    "BTC": {"oos_sharpe": 0.45, "wfer": 0.38},
    "ETH": {"oos_sharpe": 0.59, "wfer": 0.55},
    "SOL": {"oos_sharpe": 1.40, "wfer": 0.72},
}


def main():
    p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)

    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)
    wf_config = WalkForwardConfig(is_months=18, oos_months=3, mode="rolling")
    resample = "1h"

    p(f"\n{'=' * 120}")
    p(f"MULTI-ASSET ATR BREAKOUT PORTFOLIO WF — {resample} — Rolling 18mo IS / 3mo OOS")
    p(f"Assets: {', '.join(ASSET_NAMES)} — Equal Weight (1/3 each)")
    p(f"{'=' * 120}")

    # Load all asset data
    asset_dfs = {}
    for name, path in DATA_FILES.items():
        try:
            df = load_candles(path, resample=resample)
            asset_dfs[name] = df
            p(f"  {name}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            p(f"  {name}: FAILED to load — {e}")

    if len(asset_dfs) < 2:
        p("ERROR: Need at least 2 assets. Aborting.")
        return

    # Use the common date range across all assets for segment boundaries
    common_start = max(df.index[0] for df in asset_dfs.values())
    common_end = min(df.index[-1] for df in asset_dfs.values())
    p(f"\nCommon date range: {common_start.date()} to {common_end.date()}")

    boundaries = _build_segment_boundaries(common_start, common_end, wf_config)
    p(f"[multi-asset-wf] {len(boundaries)} segments\n")

    if not boundaries:
        p("ERROR: No valid segments. Aborting.")
        return

    segment_results = []

    for seg_id, is_start, is_end, oos_start, oos_end in boundaries:
        p(
            f"[multi-asset-wf] Segment {seg_id}: "
            f"IS {is_start.strftime('%Y-%m-%d')} to {is_end.strftime('%Y-%m-%d')}, "
            f"OOS {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}"
        )

        oos_return_series = {}
        is_sharpes = {}
        oos_sharpes_ind = {}
        best_params_all = {}

        for asset_name, df in asset_dfs.items():
            is_df = df.loc[is_start:is_end - pd.Timedelta(seconds=1)]
            oos_df = df.loc[oos_start:oos_end - pd.Timedelta(seconds=1)]

            if len(is_df) == 0 or len(oos_df) == 0:
                p(f"  {asset_name}: empty IS or OOS slice, skipping")
                continue

            # IS sweep
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
            best_params_all[asset_name] = best_params

            # OOS run with best IS params
            strategy = AtrBreakoutStrategy(**best_params)
            oos_signals = strategy.generate_signals(oos_df)
            oos_results = run_backtest(oos_df, oos_signals, config)
            oos_return_series[asset_name] = oos_results["strategy_returns"]

            # Individual OOS metrics
            oos_metrics_ind = compute_metrics(oos_results)
            oos_sharpes_ind[asset_name] = oos_metrics_ind["sharpe_ratio"]

            p(
                f"  {asset_name}: IS Sharpe={is_sharpes[asset_name]:.2f}, "
                f"OOS Sharpe={oos_sharpes_ind[asset_name]:.2f}, "
                f"params={best_params}"
            )

        if len(oos_return_series) < 2:
            p(f"  Segment {seg_id}: fewer than 2 assets succeeded, skipping\n")
            continue

        # Equal-weight portfolio returns
        returns_df = pd.DataFrame(oos_return_series)
        portfolio_returns = returns_df.mean(axis=1)

        # Portfolio equity curve
        portfolio_equity = config.initial_capital * (1 + portfolio_returns).cumprod()

        # Compute portfolio OOS metrics
        portfolio_results = {
            "equity_curve": portfolio_equity,
            "positions": pd.Series(1, index=portfolio_equity.index),
            "strategy_returns": portfolio_returns,
            "trades": [],
            "fees_paid": 0,
        }
        oos_metrics = compute_metrics(portfolio_results)

        # Average IS Sharpe (across assets) for WFER
        avg_is_sharpe = np.mean(list(is_sharpes.values()))
        wfer = oos_metrics["sharpe_ratio"] / avg_is_sharpe if avg_is_sharpe != 0 else 0.0

        # Pairwise correlations
        corr_matrix = returns_df.corr()
        pairs = []
        pair_names = []
        assets_in_seg = list(oos_return_series.keys())
        for i in range(len(assets_in_seg)):
            for j in range(i + 1, len(assets_in_seg)):
                a, b = assets_in_seg[i], assets_in_seg[j]
                pair_names.append(f"{a}-{b}")
                pairs.append(corr_matrix.loc[a, b])

        segment_results.append({
            "seg_id": seg_id,
            "is_period": f"{is_start.strftime('%Y-%m-%d')} -> {is_end.strftime('%Y-%m-%d')}",
            "oos_period": f"{oos_start.strftime('%Y-%m-%d')} -> {oos_end.strftime('%Y-%m-%d')}",
            "portfolio_sharpe": oos_metrics["sharpe_ratio"],
            "portfolio_return": oos_metrics["total_return_pct"],
            "portfolio_dd": oos_metrics["max_drawdown_pct"],
            "wfer": wfer,
            "individual_sharpes": dict(oos_sharpes_ind),
            "pairwise_corrs": dict(zip(pair_names, pairs)),
            "avg_corr": np.mean(pairs) if pairs else 0.0,
            "best_params": best_params_all,
            "portfolio_equity": portfolio_equity,
        })
        p("")

    if not segment_results:
        p("ERROR: No valid segments produced results!")
        return

    # =========================================================================
    # REPORT
    # =========================================================================
    p(f"\n{'=' * 140}")
    p(f"MULTI-ASSET ATR BREAKOUT PORTFOLIO — WALK-FORWARD REPORT — {resample}")
    p(f"{'=' * 140}")

    # Per-segment table
    header = (
        f"{'Seg':>3}  {'OOS Period':>25}  {'Port Shrp':>9}  {'WFER':>6}  "
        f"{'Port Ret%':>9}  {'Port DD%':>8}  {'Avg Corr':>8}  "
    )
    for asset in ASSET_NAMES:
        header += f"  {asset + ' Shrp':>9}"
    for i in range(len(ASSET_NAMES)):
        for j in range(i + 1, len(ASSET_NAMES)):
            header += f"  {ASSET_NAMES[i][:1]+'-'+ASSET_NAMES[j][:1]+' Corr':>8}"
    p(header)
    p("-" * 140)

    for r in segment_results:
        flag = ""
        if r["wfer"] < 0.30 and r["wfer"] != 0:
            flag = " [WEAK]"

        line = (
            f"{r['seg_id']:>3}  {r['oos_period']:>25}  {r['portfolio_sharpe']:>9.2f}  "
            f"{r['wfer']:>6.2f}  {r['portfolio_return']:>9.2f}  {r['portfolio_dd']:>8.2f}  "
            f"{r['avg_corr']:>8.2f}  "
        )
        for asset in ASSET_NAMES:
            shrp = r["individual_sharpes"].get(asset, float("nan"))
            line += f"  {shrp:>9.2f}"

        for i in range(len(ASSET_NAMES)):
            for j in range(i + 1, len(ASSET_NAMES)):
                pair_key = f"{ASSET_NAMES[i]}-{ASSET_NAMES[j]}"
                corr_val = r["pairwise_corrs"].get(pair_key, float("nan"))
                line += f"  {corr_val:>8.2f}"

        line += flag
        p(line)

    p("-" * 140)

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

    # Overall summary
    sharpes = np.array([r["portfolio_sharpe"] for r in segment_results])
    wfers = np.array([r["wfer"] for r in segment_results])
    avg_corrs = np.array([r["avg_corr"] for r in segment_results])

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

    p(f"\nSUMMARY")
    p(f"  Segments:              {len(segment_results)}")
    p(f"  Mean OOS Sharpe:       {sharpes.mean():.4f}")
    p(f"  Median OOS Sharpe:     {np.median(sharpes):.4f}")
    p(f"  Mean WFER:             {wfers.mean():.4f}  {wfer_label}")
    p(f"  WFER Slope:            {wfer_slope:+.6f} per segment (t={wfer_tstat:+.2f})")
    p(f"  Mean Pairwise Corr:    {avg_corrs.mean():.4f}")
    p(f"  Combined OOS Return:   {total_ret:.2f}%")
    p(f"  Combined OOS Max DD:   {max_dd:.2f}%")
    p(f"  OOS Period:            {combined_equity.index[0].date()} to {combined_equity.index[-1].date()}")

    if wfer_slope < -0.03 and abs(wfer_tstat) > 1.5:
        p("  ** WARNING: Significantly declining WFER - edge is decaying **")
    elif wfer_slope < -0.02:
        p("  ** CAUTION: Declining WFER trend (not statistically significant) **")

    # Per-asset average OOS Sharpe across segments
    p(f"\nPER-ASSET AVERAGE OOS SHARPE (across segments)")
    for asset in ASSET_NAMES:
        asset_sharpes = [r["individual_sharpes"].get(asset, np.nan) for r in segment_results]
        asset_sharpes = [s for s in asset_sharpes if not np.isnan(s)]
        if asset_sharpes:
            p(f"  {asset:>5}: mean={np.mean(asset_sharpes):.4f}, median={np.median(asset_sharpes):.4f}")
        else:
            p(f"  {asset:>5}: no data")

    # Correlation breakdown
    p(f"\nPAIRWISE CORRELATION BREAKDOWN (across segments)")
    for i in range(len(ASSET_NAMES)):
        for j in range(i + 1, len(ASSET_NAMES)):
            pair_key = f"{ASSET_NAMES[i]}-{ASSET_NAMES[j]}"
            pair_corrs = [r["pairwise_corrs"].get(pair_key, np.nan) for r in segment_results]
            pair_corrs = [c for c in pair_corrs if not np.isnan(c)]
            if pair_corrs:
                p(f"  {pair_key:>8}: mean={np.mean(pair_corrs):.4f}, min={np.min(pair_corrs):.4f}, max={np.max(pair_corrs):.4f}")

    # Comparison vs individual asset WF results
    p(f"\nDIVERSIFICATION COMPARISON")
    p(f"  {'Asset':<12}  {'Individual WF Sharpe':>20}  {'Individual WF WFER':>18}  {'Portfolio Contrib Sharpe':>23}")
    p(f"  {'-'*12}  {'-'*20}  {'-'*18}  {'-'*23}")
    for asset in ASSET_NAMES:
        ind = INDIVIDUAL_WF.get(asset, {})
        ind_sharpe = ind.get("oos_sharpe", "N/A")
        ind_wfer = ind.get("wfer", "N/A")
        asset_sharpes_seg = [r["individual_sharpes"].get(asset, np.nan) for r in segment_results]
        asset_sharpes_seg = [s for s in asset_sharpes_seg if not np.isnan(s)]
        contrib = np.mean(asset_sharpes_seg) if asset_sharpes_seg else float("nan")
        p(
            f"  {asset:<12}  {str(ind_sharpe):>20}  {str(ind_wfer):>18}  "
            f"{contrib:>23.4f}"
        )

    p(f"\n  Portfolio mean OOS Sharpe:     {sharpes.mean():.4f}")
    weighted_ind_avg = np.mean([INDIVIDUAL_WF[a]["oos_sharpe"] for a in ASSET_NAMES if a in INDIVIDUAL_WF])
    p(f"  Weighted avg individual Sharpe: {weighted_ind_avg:.4f}")
    p(f"  Portfolio max DD:              {max_dd:.2f}%")

    diversification_benefit = sharpes.mean() - weighted_ind_avg
    p(f"  Diversification delta:         {diversification_benefit:+.4f}")
    if diversification_benefit > 0:
        p(f"  --> Portfolio IMPROVES on avg individual Sharpe")
    else:
        p(f"  --> Portfolio correlation too high for Sharpe improvement; check MaxDD for DD reduction")

    p(f"{'=' * 140}\n")


if __name__ == "__main__":
    main()
