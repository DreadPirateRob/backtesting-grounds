from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import compute_metrics
from .strategy import Strategy
from .sweep import run_sweep


@dataclass
class WalkForwardConfig:
    is_months: int = 18
    oos_months: int = 3
    mode: str = "rolling"       # "rolling" or "anchored"
    min_trades_oos: int = 5
    n_jobs: int | None = None


@dataclass
class WFSegment:
    segment_id: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    best_params: dict
    is_sharpe: float
    oos_sharpe: float
    oos_return_pct: float
    oos_max_dd_pct: float
    oos_trades: int
    wfer: float
    is_regime_pcts: dict
    oos_regime_pcts: dict
    regime_mismatch: float
    oos_equity: pd.Series


@dataclass
class WalkForwardResult:
    segments: list[WFSegment]
    combined_oos_equity: pd.Series
    overall_wfer: float
    overall_oos_sharpe: float
    param_stability: dict
    wfer_slope: float
    wfer_slope_tstat: float


def label_regimes(
    df: pd.DataFrame,
    sma_period: int = 200,
    momentum_bars: int = 20,
    momentum_thresh: float = 0.05,
) -> pd.Series:
    close = df["close"]
    sma_vals = close.rolling(sma_period).mean()
    momentum_ref = close.shift(momentum_bars)

    bull = (close > sma_vals) & (close > momentum_ref * (1 + momentum_thresh))
    bear = (close < sma_vals) & (close < momentum_ref * (1 - momentum_thresh))

    regime = pd.Series("sideways", index=df.index)
    regime[bull] = "bull"
    regime[bear] = "bear"

    warmup = max(sma_period, momentum_bars)
    regime.iloc[:warmup] = np.nan

    return regime


def _regime_pcts(regime_series: pd.Series) -> dict:
    valid = regime_series.dropna()
    if len(valid) == 0:
        return {"bull": 0.0, "bear": 0.0, "sideways": 0.0}
    counts = valid.value_counts(normalize=True)
    return {
        "bull": float(counts.get("bull", 0.0)),
        "bear": float(counts.get("bear", 0.0)),
        "sideways": float(counts.get("sideways", 0.0)),
    }


def _regime_mismatch(is_pcts: dict, oos_pcts: dict) -> float:
    return sum(abs(is_pcts[k] - oos_pcts[k]) for k in ("bull", "bear", "sideways"))


def _ols_slope_tstat(y: np.ndarray) -> tuple[float, float]:
    n = len(y)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    if ss_xx == 0:
        return 0.0, 0.0
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    mse = (residuals ** 2).sum() / (n - 2)
    se_slope = np.sqrt(mse / ss_xx) if mse > 0 else 0.0
    t_stat = slope / se_slope if se_slope > 0 else 0.0
    return float(slope), float(t_stat)


def compute_param_stability(segments: list[WFSegment]) -> dict:
    if not segments:
        return {}

    all_params = [seg.best_params for seg in segments]
    param_names = list(all_params[0].keys())
    stability = {}

    for name in param_names:
        values = [p[name] for p in all_params]
        counter = Counter(values)
        modal_value, modal_count = counter.most_common(1)[0]
        pcr = modal_count / len(values)
        stability[name] = {
            "pcr": round(pcr, 4),
            "modal_value": modal_value,
            "values_per_segment": values,
        }

    return stability


def _build_segment_boundaries(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    wf_config: WalkForwardConfig,
) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    boundaries = []
    seg_id = 0

    if wf_config.mode == "anchored":
        anchor_start = data_start
        oos_cursor = anchor_start + pd.DateOffset(months=wf_config.is_months)
        while True:
            is_end = oos_cursor
            oos_start = oos_cursor
            oos_end = oos_cursor + pd.DateOffset(months=wf_config.oos_months)
            if oos_end > data_end:
                break
            boundaries.append((seg_id, anchor_start, is_end, oos_start, oos_end))
            seg_id += 1
            oos_cursor = oos_end
    else:
        is_start = data_start
        while True:
            is_end = is_start + pd.DateOffset(months=wf_config.is_months)
            oos_start = is_end
            oos_end = oos_start + pd.DateOffset(months=wf_config.oos_months)
            if oos_end > data_end:
                break
            boundaries.append((seg_id, is_start, is_end, oos_start, oos_end))
            seg_id += 1
            is_start = is_start + pd.DateOffset(months=wf_config.oos_months)

    return boundaries


def run_walk_forward(
    strategy_class: Type[Strategy],
    df: pd.DataFrame,
    grid: dict[str, list],
    config: BacktestConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
    rank_by: str = "sharpe_ratio",
) -> WalkForwardResult:
    if config is None:
        config = BacktestConfig()
    if wf_config is None:
        wf_config = WalkForwardConfig()

    regimes = label_regimes(df)
    data_start = df.index[0]
    data_end = df.index[-1]

    boundaries = _build_segment_boundaries(data_start, data_end, wf_config)

    if not boundaries:
        raise ValueError(
            f"No valid walk-forward segments. Data range: {data_start} to {data_end}, "
            f"requires at least {wf_config.is_months + wf_config.oos_months} months."
        )

    print(
        f"[walk-forward] {len(boundaries)} segments, mode={wf_config.mode}, "
        f"IS={wf_config.is_months}mo, OOS={wf_config.oos_months}mo",
        file=sys.stderr,
    )

    wf_segments: list[WFSegment] = []

    for seg_id, is_start, is_end, oos_start, oos_end in boundaries:
        print(
            f"[walk-forward] Segment {seg_id}: "
            f"IS {is_start.strftime('%Y-%m-%d')} to {is_end.strftime('%Y-%m-%d')}, "
            f"OOS {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}",
            file=sys.stderr,
        )

        is_df = df.loc[is_start:is_end - pd.Timedelta(seconds=1)]
        oos_df = df.loc[oos_start:oos_end - pd.Timedelta(seconds=1)]

        if len(is_df) == 0 or len(oos_df) == 0:
            print(f"[walk-forward] Segment {seg_id}: empty IS or OOS, skipping", file=sys.stderr)
            continue

        sweep_result = run_sweep(
            strategy_class=strategy_class,
            df=is_df,
            grid=grid,
            config=config,
            rank_by=rank_by,
            top=1,
            n_jobs=wf_config.n_jobs,
        )

        if not sweep_result["results"]:
            print(f"[walk-forward] Segment {seg_id}: sweep returned no results, skipping", file=sys.stderr)
            continue

        best = sweep_result["results"][0]
        best_params = best["params"]
        is_sharpe = best["metrics"]["sharpe_ratio"]

        strategy = strategy_class(**best_params)
        oos_signals = strategy.generate_signals(oos_df)
        oos_results = run_backtest(oos_df, oos_signals, config)
        oos_metrics = compute_metrics(oos_results)

        oos_sharpe = oos_metrics["sharpe_ratio"]
        oos_return_pct = oos_metrics["total_return_pct"]
        oos_max_dd_pct = oos_metrics["max_drawdown_pct"]
        oos_trades = oos_metrics["total_trades"]
        oos_equity = oos_results["equity_curve"]

        wfer = oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0

        is_regime_pcts = _regime_pcts(regimes.loc[is_df.index[0]:is_df.index[-1]])
        oos_regime_pcts = _regime_pcts(regimes.loc[oos_df.index[0]:oos_df.index[-1]])
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
                f"[walk-forward] Segment {seg_id}: only {oos_trades} OOS trades "
                f"(min={wf_config.min_trades_oos})",
                file=sys.stderr,
            )

    if not wf_segments:
        raise ValueError("No valid walk-forward segments produced results.")

    # Combined OOS equity: rescale each segment to start where previous ended
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

    # Combined OOS Sharpe from the full concatenated equity curve
    combined_returns = combined_oos_equity.pct_change().dropna()
    if len(combined_returns) > 1:
        duration_days = (combined_returns.index[-1] - combined_returns.index[0]).total_seconds() / 86400
        years = duration_days / 365.25 if duration_days > 0 else 1.0
        bar_seconds = (combined_returns.index[1] - combined_returns.index[0]).total_seconds()
        if bar_seconds < 3600 and len(combined_returns) > 60:
            hourly = (1 + combined_returns).resample("1h").prod() - 1
            hourly = hourly.dropna()
            h_bpy = len(hourly) / years if years > 0 else 8760
            h_mean = hourly.mean()
            h_std = hourly.std()
            overall_oos_sharpe = float(h_mean / h_std * np.sqrt(h_bpy)) if h_std > 0 else 0.0
        else:
            bars_per_year = len(combined_returns) / years if years > 0 else 8760
            ret_mean = combined_returns.mean()
            ret_std = combined_returns.std()
            overall_oos_sharpe = float(ret_mean / ret_std * np.sqrt(bars_per_year)) if ret_std > 0 else 0.0
    else:
        overall_oos_sharpe = 0.0

    # WFER slope
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


def print_wf_report(result: WalkForwardResult):
    p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)

    p("\n" + "=" * 120)
    p("WALK-FORWARD ANALYSIS REPORT")
    p("=" * 120)

    header = (
        f"{'Seg':>3}  {'IS Period':>25}  {'OOS Period':>25}  "
        f"{'IS Shrp':>8}  {'OOS Shrp':>9}  {'WFER':>6}  "
        f"{'OOS Ret%':>9}  {'OOS DD%':>8}  {'Trades':>6}  "
        f"{'Mismatch':>8}  {'IS Bull%':>8}  {'IS Bear%':>8}  {'OOS Bull%':>9}  {'OOS Bear%':>9}"
    )
    p(header)
    p("-" * 120)

    for seg in result.segments:
        is_period = f"{seg.is_start} -> {seg.is_end}"
        oos_period = f"{seg.oos_start} -> {seg.oos_end}"
        flag = ""
        if seg.oos_trades < 5:
            flag += " [LOW TRADES]"
        if seg.regime_mismatch > 0.50:
            flag += " [HIGH MISMATCH]"
        if seg.wfer < 0.30 and seg.is_sharpe > 0:
            flag += " [WEAK WFER]"

        p(
            f"{seg.segment_id:>3}  {is_period:>25}  {oos_period:>25}  "
            f"{seg.is_sharpe:>8.2f}  {seg.oos_sharpe:>9.2f}  {seg.wfer:>6.2f}  "
            f"{seg.oos_return_pct:>9.2f}  {seg.oos_max_dd_pct:>8.2f}  {seg.oos_trades:>6}  "
            f"{seg.regime_mismatch:>8.2f}  "
            f"{seg.is_regime_pcts.get('bull', 0)*100:>7.1f}%  "
            f"{seg.is_regime_pcts.get('bear', 0)*100:>7.1f}%  "
            f"{seg.oos_regime_pcts.get('bull', 0)*100:>8.1f}%  "
            f"{seg.oos_regime_pcts.get('bear', 0)*100:>8.1f}%"
            f"{flag}"
        )

    p("-" * 120)

    p("\nSUMMARY")
    p(f"  Segments:              {len(result.segments)}")
    p(f"  Overall OOS Sharpe:    {result.overall_oos_sharpe:.4f}")

    wfer_label = (
        "(EXCELLENT)" if result.overall_wfer > 0.70 else
        "(GOOD)" if result.overall_wfer > 0.50 else
        "(ACCEPTABLE)" if result.overall_wfer > 0.30 else
        "(WEAK)"
    )
    p(f"  Overall WFER:          {result.overall_wfer:.4f}  {wfer_label}")

    p(f"  WFER Slope:            {result.wfer_slope:+.6f} per segment  (t={result.wfer_slope_tstat:+.2f})")

    if result.wfer_slope < -0.03 and abs(result.wfer_slope_tstat) > 1.5:
        p("  ** WARNING: Significantly declining WFER - edge is decaying **")
    elif result.wfer_slope < -0.02:
        p("  ** CAUTION: Declining WFER trend (not statistically significant) **")

    eq = result.combined_oos_equity
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    cummax = eq.cummax()
    max_dd = float(((eq - cummax) / cummax).min()) * 100
    p(f"  Combined OOS Return:   {total_ret:.2f}%")
    p(f"  Combined OOS Max DD:   {max_dd:.2f}%")
    p(f"  OOS Period:            {eq.index[0].date()} to {eq.index[-1].date()}")

    p("\nPARAMETER STABILITY")
    p(f"  {'Parameter':<20}  {'PCR':>6}  {'Modal Value':>14}  {'Values per Segment'}")
    p(f"  {'-'*20}  {'-'*6}  {'-'*14}  {'-'*40}")
    for name, info in result.param_stability.items():
        pcr = info["pcr"]
        modal = info["modal_value"]
        vals = info["values_per_segment"]
        stability_label = (
            "STABLE" if pcr > 0.70 else
            "MODERATE" if pcr > 0.50 else
            "BORDERLINE" if pcr > 0.30 else
            "UNSTABLE"
        )
        p(f"  {name:<20}  {pcr:>6.2f}  {str(modal):>14}  {vals}  [{stability_label}]")

    p("=" * 120 + "\n")
