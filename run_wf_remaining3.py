#!/usr/bin/env python3
"""Walk-forward analysis for the 3 remaining fixed-param strategies:
1. MultiBarSelloffStrategy (cumulative selloff mean-reversion)
2. ExtremeRangeStrategy (widest-range bullish bar breakout)
3. VolSpikeStrategy (buy-side volume spike momentum)

Runs across BTC, ETH, SOL at 1h with 10bps fees.
18mo IS / 3mo OOS rolling walk-forward.
"""

import sys
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.walk_forward import (
    WalkForwardConfig, run_walk_forward, print_wf_report,
)
from strategies.multi_bar_selloff_strategy import MultiBarSelloffStrategy
from strategies.extreme_range_strategy import ExtremeRangeStrategy
from strategies.vol_spike_strategy import VolSpikeStrategy


DATA_FILES = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CONFIG = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
WF_CONFIG = WalkForwardConfig(is_months=18, oos_months=3, mode="rolling")

p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)


def run_strategy_wf(name, strategy_class, grid, extra_columns=None):
    p(f"\n{'=' * 120}")
    p(f"  {name} — 1h — Rolling 18mo IS / 3mo OOS — 10bps fees")
    p(f"{'=' * 120}")

    results = {}

    for asset, path in DATA_FILES.items():
        p(f"\n--- {asset} ---")
        kwargs = {"resample": "1h"}
        if extra_columns:
            kwargs["extra_columns"] = extra_columns
        df = load_candles(path, **kwargs)
        p(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

        try:
            wf = run_walk_forward(
                strategy_class=strategy_class,
                df=df,
                grid=grid,
                config=CONFIG,
                wf_config=WF_CONFIG,
                rank_by="sharpe_ratio",
            )
            print_wf_report(wf)
            results[asset] = wf

            # Per-segment summary
            oos_sharpes = [s.oos_sharpe for s in wf.segments]
            frac_pos = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
            p(f"\n  {asset} SUMMARY:")
            p(f"    OOS Sharpe: {wf.overall_oos_sharpe:.4f}")
            p(f"    WFER: {wf.overall_wfer:.4f}")
            p(f"    Segments pos: {frac_pos*100:.0f}% ({sum(1 for s in oos_sharpes if s > 0)}/{len(oos_sharpes)})")
            p(f"    WFER slope: {wf.wfer_slope:+.6f} (t={wf.wfer_slope_tstat:+.2f})")
        except Exception as e:
            p(f"  ERROR: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            results[asset] = None

    return results


def main():
    # ═══════════════════════════════════════════════════════════════
    # 1. MULTI-BAR SELLOFF
    # ═══════════════════════════════════════════════════════════════
    selloff_grid = {
        "lookback": [12, 24, 36],
        "drop_pct": [-6.0, -8.0, -10.0],
        "hold_bars": [12, 16, 24],
        "cooldown_bars": [3],
    }

    selloff_results = run_strategy_wf(
        "MULTI-BAR SELLOFF",
        MultiBarSelloffStrategy,
        selloff_grid,
    )

    # ═══════════════════════════════════════════════════════════════
    # 2. EXTREME RANGE BAR
    # ═══════════════════════════════════════════════════════════════
    extreme_grid = {
        "lookback": [48, 72, 96],
        "hold_bars": [8, 10, 12],
        "cooldown_bars": [2],
    }

    extreme_results = run_strategy_wf(
        "EXTREME RANGE BAR",
        ExtremeRangeStrategy,
        extreme_grid,
    )

    # ═══════════════════════════════════════════════════════════════
    # 3. VOL SPIKE
    # ═══════════════════════════════════════════════════════════════
    volspike_grid = {
        "vol_zscore_window": [48, 96, 192],
        "vol_zscore_threshold": [2.0, 2.5, 3.0],
        "delta_threshold_pct": [5.0, 10.0, 15.0],
        "hold_bars": [4, 8, 16],
        "buy_only": [True],
    }

    volspike_results = run_strategy_wf(
        "VOL SPIKE",
        VolSpikeStrategy,
        volspike_grid,
        extra_columns=["taker_buy_base_volume"],
    )

    # ═══════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 120}")
    p(f"  COMBINED WALK-FORWARD SUMMARY")
    p(f"{'=' * 120}")

    p(f"\n  {'Strategy':<25} {'Asset':<6} {'OOS Sharpe':>10} {'WFER':>8} {'WFER Label':>12} "
      f"{'Segs Pos%':>10} {'Verdict':>8}")
    p(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*12} {'-'*10} {'-'*8}")

    all_strategies = [
        ("Multi-Bar Selloff", selloff_results),
        ("Extreme Range Bar", extreme_results),
        ("Vol Spike", volspike_results),
    ]

    for strat_name, results in all_strategies:
        for asset in DATA_FILES.keys():
            wf = results.get(asset)
            if wf is None:
                p(f"  {strat_name:<25} {asset:<6} {'FAILED':>10}")
                continue

            oos_sharpes = [s.oos_sharpe for s in wf.segments]
            frac_pos = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
            wfer_label = (
                "EXCELLENT" if wf.overall_wfer > 0.70 else
                "GOOD" if wf.overall_wfer > 0.50 else
                "ACCEPTABLE" if wf.overall_wfer > 0.30 else
                "WEAK"
            )
            passed = (
                wf.overall_oos_sharpe > 0
                and wf.overall_wfer > 0.20
                and frac_pos >= 0.40
            )
            verdict = "PASS" if passed else "FAIL"
            p(f"  {strat_name:<25} {asset:<6} {wf.overall_oos_sharpe:>10.4f} "
              f"{wf.overall_wfer:>8.4f} {wfer_label:>12} "
              f"{frac_pos*100:>9.0f}% {verdict:>8}")

    # Qualification verdicts with details
    p(f"\n  QUALIFICATION VERDICTS:")
    for strat_name, results in all_strategies:
        for asset in DATA_FILES.keys():
            wf = results.get(asset)
            if wf is None:
                continue

            oos_sharpes = [s.oos_sharpe for s in wf.segments]
            frac_pos = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)

            passed = (
                wf.overall_oos_sharpe > 0
                and wf.overall_wfer > 0.20
                and frac_pos >= 0.40
            )
            verdict = "PASS" if passed else "FAIL"
            p(f"    {strat_name} / {asset}: OOS Sharpe {wf.overall_oos_sharpe:.2f}, "
              f"WFER {wf.overall_wfer:.2f}, {frac_pos*100:.0f}% pos -> {verdict}")

    p(f"\n{'=' * 120}")


if __name__ == "__main__":
    main()
