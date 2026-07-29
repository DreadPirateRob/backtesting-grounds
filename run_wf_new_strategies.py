#!/usr/bin/env python3
"""Walk-forward analysis for the 2 new Round 7 strategies:
1. DonchianVolStrategy (breakout + volume filter)
2. GreenMomentumStrategy (consecutive bullish bars)

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
from strategies.donchian_vol_strategy import DonchianVolStrategy
from strategies.green_momentum_strategy import GreenMomentumStrategy


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
            results[asset] = None

    return results


def main():
    # ═══════════════════════════════════════════════════════════════
    # 1. DONCHIAN BREAKOUT + VOLUME
    # ═══════════════════════════════════════════════════════════════
    donchian_grid = {
        "entry_period": [240, 480, 720],
        "vol_mult": [1.5, 2.0, 2.5],
        "hold_bars": [24, 48],
        "cooldown_bars": [12],
    }

    donchian_results = run_strategy_wf(
        "DONCHIAN BREAKOUT + VOLUME",
        DonchianVolStrategy,
        donchian_grid,
    )

    # ═══════════════════════════════════════════════════════════════
    # 2. GREEN MOMENTUM
    # ═══════════════════════════════════════════════════════════════
    green_grid = {
        "min_green": [3, 4, 5],
        "min_gain_pct": [3.0, 4.0, 5.0],
        "hold_bars": [12, 16],
        "cooldown_bars": [3],
    }

    green_results = run_strategy_wf(
        "GREEN MOMENTUM",
        GreenMomentumStrategy,
        green_grid,
    )

    # ═══════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 120}")
    p(f"  COMBINED WALK-FORWARD SUMMARY")
    p(f"{'=' * 120}")

    p(f"\n  {'Strategy':<25} {'Asset':<6} {'OOS Sharpe':>10} {'WFER':>8} {'WFER Label':>12} "
      f"{'Segs Pos%':>10} {'WFER Slope':>12}")
    p(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*12} {'-'*10} {'-'*12}")

    for strat_name, results in [
        ("Donchian Vol", donchian_results),
        ("Green Momentum", green_results),
    ]:
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
            p(f"  {strat_name:<25} {asset:<6} {wf.overall_oos_sharpe:>10.4f} "
              f"{wf.overall_wfer:>8.4f} {wfer_label:>12} "
              f"{frac_pos*100:>9.0f}% {wf.wfer_slope:>+12.6f}")

    # Qualification verdict
    p(f"\n  QUALIFICATION VERDICTS:")
    for strat_name, results in [
        ("Donchian Vol", donchian_results),
        ("Green Momentum", green_results),
    ]:
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
              f"WFER {wf.overall_wfer:.2f}, {frac_pos*100:.0f}% pos → {verdict}")

    p(f"\n{'=' * 120}")


if __name__ == "__main__":
    main()
