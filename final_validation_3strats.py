"""Final comprehensive validation of 3 qualifying strategies at 1h.

Tests each strategy with:
- Multiple param sets to confirm robustness
- Full 5yr sample at 10bps and 5bps
- Rolling 12-month windows (16 windows, 3-month step)
- Per-year breakdown to check consistency across regimes
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from strategies.vol_spike_strategy import VolSpikeStrategy
from strategies.extreme_range_strategy import ExtremeRangeStrategy
from strategies.multi_bar_selloff_strategy import MultiBarSelloffStrategy

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CFG_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
CFG_5 = BacktestConfig(fee_rate=0.00025, slippage_pct=0.00025)


def rolling_windows(df, window_months=12, step_months=3):
    start = df.index[0]
    end = df.index[-1]
    current = start
    while True:
        window_end = current + pd.DateOffset(months=window_months)
        if window_end > end:
            break
        sl = df.loc[current:window_end]
        if len(sl) > 500:
            yield current, window_end, sl
        current += pd.DateOffset(months=step_months)


def per_year_analysis(df, signal_fn, cfg):
    """Break down performance by year."""
    years = sorted(df.index.year.unique())
    results = {}
    for year in years:
        yearly = df[df.index.year == year]
        if len(yearly) < 100:
            continue
        sig = signal_fn(yearly)
        res = run_backtest(yearly, sig, cfg)
        m = compute_metrics(res)
        results[year] = m
    return results


def full_evaluation(name, signal_fn, data, summary):
    """Complete evaluation of a strategy across all assets."""
    for asset, df in data.items():
        sig = signal_fn(df)
        m10 = compute_metrics(run_backtest(df, sig, CFG_10))
        m5 = compute_metrics(run_backtest(df, sig, CFG_5))

        # Rolling windows
        windows = []
        for ws, we, sl in rolling_windows(df):
            sw = signal_fn(sl)
            wm = compute_metrics(run_backtest(sl, sw, CFG_10))
            windows.append({"start": ws, "sharpe": wm['sharpe_ratio']})

        n = len(windows)
        sharpes = [w['sharpe'] for w in windows]
        frac_pos = sum(1 for s in sharpes if s > 0) / n if n else 0
        frac_good = sum(1 for s in sharpes if s > 0.5) / n if n else 0
        avg_sh = np.mean(sharpes) if sharpes else 0
        med_sh = np.median(sharpes) if sharpes else 0

        # Per-year
        yearly = per_year_analysis(df, signal_fn, CFG_10)

        print(f"\n  {asset}:")
        print(f"    Full sample (10bps): Sharpe={m10['sharpe_ratio']:.2f} Ret={m10['total_return_pct']:.1f}% "
              f"Trades={m10['total_trades']} WR={m10['win_rate_pct']:.1f}% DD={m10['max_drawdown_pct']:.1f}%")
        print(f"    Full sample ( 5bps): Sharpe={m5['sharpe_ratio']:.2f} Ret={m5['total_return_pct']:.1f}%")
        print(f"    Rolling ({n} windows): avg={avg_sh:.2f} med={med_sh:.2f} "
              f"frac>0={frac_pos:.0%} frac>0.5={frac_good:.0%}")

        # Per-year results
        yr_str = "    Per year: "
        for yr, ym in sorted(yearly.items()):
            yr_str += f"{yr}:{ym['sharpe_ratio']:+.2f}({ym['total_trades']}t) "
        print(yr_str)

        summary.append({
            "name": name, "asset": asset,
            "sh10": m10['sharpe_ratio'], "sh5": m5['sharpe_ratio'],
            "trades": m10['total_trades'], "wr": m10['win_rate_pct'],
            "ret": m10['total_return_pct'], "dd": m10['max_drawdown_pct'],
            "n_win": n, "frac_pos": frac_pos, "frac_good": frac_good,
            "avg_sh": avg_sh, "med_sh": med_sh,
        })


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 120)
    print("FINAL VALIDATION: 3 QUALIFYING STRATEGIES AT 1h")
    print("=" * 120)

    # Load data
    data = {}
    data_with_tbv = {}  # With taker_buy_base_volume for Vol Spike
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        df_tbv = load_candles(path, resample="1h", extra_columns=["taker_buy_base_volume"])
        print(f"{len(df)} bars")
        data[asset] = df
        data_with_tbv[asset] = df_tbv

    summary = []

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 1: VOL SPIKE (Volume anomaly + directional delta)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# STRATEGY 1: VOL SPIKE MOMENTUM")
    print(f"# Mechanism: Volume z-score spike + buy-side volume delta → long")
    print(f"# Params: vol_zscore_window=96, threshold=2.5, delta=10%, hold=8")
    print(f"{'#'*120}")

    strat_vs = VolSpikeStrategy(
        vol_zscore_window=96, vol_zscore_threshold=2.5,
        delta_threshold_pct=10.0, hold_bars=8, buy_only=True,
    )
    full_evaluation("VolSpike",
                    lambda df: strat_vs.generate_signals(df),
                    data_with_tbv, summary)

    # Vol Spike sensitivity: alt params
    print(f"\n  --- Vol Spike parameter sensitivity ---")
    vs_alt_params = [
        ("VS_z2.0_d10", VolSpikeStrategy(96, 2.0, 10.0, 8, True)),
        ("VS_z2.5_d5", VolSpikeStrategy(96, 2.5, 5.0, 8, True)),
        ("VS_z2.0_d5", VolSpikeStrategy(96, 2.0, 5.0, 8, True)),
        ("VS_z3.0_d10", VolSpikeStrategy(96, 3.0, 10.0, 8, True)),
        ("VS_w48_z2.5_d10", VolSpikeStrategy(48, 2.5, 10.0, 8, True)),
    ]
    for label, strat in vs_alt_params:
        row = f"  {label:<20}"
        for asset, df in data_with_tbv.items():
            sig = strat.generate_signals(df)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 2: MULTI-BAR SELLOFF (Crash dip buyer)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# STRATEGY 2: MULTI-BAR SELLOFF (Crash Dip Buyer)")
    print(f"# Mechanism: Cumulative return < -8% over 24 bars → buy, hold 16 bars")
    print(f"# Params: lookback=24, drop=-8%, hold=16, cooldown=3")
    print(f"{'#'*120}")

    strat_ms = MultiBarSelloffStrategy(lookback=24, drop_pct=-8.0, hold_bars=16, cooldown_bars=3)
    full_evaluation("Selloff8%",
                    lambda df: strat_ms.generate_signals(df),
                    data, summary)

    # Selloff sensitivity
    print(f"\n  --- Selloff parameter sensitivity ---")
    ms_alt_params = [
        ("MS_d6_h16", MultiBarSelloffStrategy(24, -6.0, 16, 3)),
        ("MS_d8_h24", MultiBarSelloffStrategy(24, -8.0, 24, 3)),
        ("MS_d10_h24", MultiBarSelloffStrategy(24, -10.0, 24, 3)),
        ("MS_lb18_d6_h24", MultiBarSelloffStrategy(18, -6.0, 24, 3)),
        ("MS_lb36_d10_h24", MultiBarSelloffStrategy(36, -10.0, 24, 3)),
    ]
    for label, strat in ms_alt_params:
        row = f"  {label:<20}"
        for asset, df in data.items():
            sig = strat.generate_signals(df)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 3: EXTREME RANGE BAR (Breakout momentum)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# STRATEGY 3: EXTREME RANGE BAR (Breakout Momentum)")
    print(f"# Mechanism: Widest range in 72 bars + bullish close → buy, hold 8")
    print(f"# Params: lookback=72, hold=8, cooldown=2")
    print(f"{'#'*120}")

    strat_er = ExtremeRangeStrategy(lookback=72, hold_bars=8, cooldown_bars=2)
    full_evaluation("ExtrRange",
                    lambda df: strat_er.generate_signals(df),
                    data, summary)

    # Extreme Range sensitivity
    print(f"\n  --- Extreme Range parameter sensitivity ---")
    er_alt_params = [
        ("ER_lb48_h8", ExtremeRangeStrategy(48, 8, 2)),
        ("ER_lb72_h12", ExtremeRangeStrategy(72, 12, 2)),
        ("ER_lb96_h8", ExtremeRangeStrategy(96, 8, 2)),
        ("ER_lb96_h12", ExtremeRangeStrategy(96, 12, 2)),
        ("ER_lb120_h12", ExtremeRangeStrategy(120, 12, 2)),
    ]
    for label, strat in er_alt_params:
        row = f"  {label:<20}"
        for asset, df in data.items():
            sig = strat.generate_signals(df)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════════════
    # CROSS-STRATEGY CORRELATION CHECK
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# SIGNAL OVERLAP ANALYSIS")
    print(f"{'#'*120}")

    for asset in ASSETS:
        df = data_with_tbv[asset]
        df_plain = data[asset]

        sig_vs = strat_vs.generate_signals(df)
        sig_ms = strat_ms.generate_signals(df_plain)
        sig_er = strat_er.generate_signals(df_plain)

        # Count bars in position
        vs_pos = (sig_vs != 0).sum()
        ms_pos = (sig_ms != 0).sum()
        er_pos = (sig_er != 0).sum()

        # Overlap: both in position at same time
        vs_ms_overlap = ((sig_vs != 0) & (sig_ms != 0)).sum()
        vs_er_overlap = ((sig_vs != 0) & (sig_er != 0)).sum()
        ms_er_overlap = ((sig_ms != 0) & (sig_er != 0)).sum()

        n = len(df)
        print(f"\n  {asset} ({n} bars):")
        print(f"    VolSpike in position: {vs_pos} bars ({vs_pos/n*100:.1f}%)")
        print(f"    Selloff  in position: {ms_pos} bars ({ms_pos/n*100:.1f}%)")
        print(f"    ExtrRange in position: {er_pos} bars ({er_pos/n*100:.1f}%)")
        print(f"    VS ∩ MS overlap: {vs_ms_overlap} bars ({vs_ms_overlap/max(min(vs_pos,ms_pos),1)*100:.1f}% of smaller)")
        print(f"    VS ∩ ER overlap: {vs_er_overlap} bars ({vs_er_overlap/max(min(vs_pos,er_pos),1)*100:.1f}% of smaller)")
        print(f"    MS ∩ ER overlap: {ms_er_overlap} bars ({ms_er_overlap/max(min(ms_pos,er_pos),1)*100:.1f}% of smaller)")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("FINAL SUMMARY")
    print(f"{'='*120}")
    print(f"{'Strategy':<12} {'Asset':<5} {'Sh10':>5} {'Sh5':>5} {'Tr':>5} {'WR%':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'Win':>4} {'F>0':>5} {'F>.5':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 120)
    for r in summary:
        print(f"{r['name']:<12} {r['asset']:<5} {r['sh10']:>5.2f} {r['sh5']:>5.2f} "
              f"{r['trades']:>5} {r['wr']:>5.1f} {r['ret']:>7.1f} {r['dd']:>6.1f} "
              f"{r['n_win']:>4} {r['frac_pos']:>5.0%} {r['frac_good']:>5.0%} "
              f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Qualification summary
    print(f"\n{'='*120}")
    print("QUALIFICATION ASSESSMENT")
    print(f"{'='*120}")
    print("Criteria: Sharpe > 0 at 10bps, frac>0 >= 60%, sufficient trades (>50)")
    print()

    for strat_name in ["VolSpike", "Selloff8%", "ExtrRange"]:
        strat_rows = [r for r in summary if r['name'] == strat_name]
        passing = [r for r in strat_rows if r['sh10'] > 0 and r['frac_pos'] >= 0.6 and r['trades'] > 50]
        borderline = [r for r in strat_rows if r['sh10'] > 0 and r['frac_pos'] >= 0.5 and r['trades'] > 20]
        print(f"  {strat_name}:")
        for r in strat_rows:
            status = "PASS" if r in passing else ("BORDERLINE" if r in borderline else "FAIL")
            print(f"    {r['asset']}: Sharpe={r['sh10']:.2f} Trades={r['trades']} "
                  f"frac>0={r['frac_pos']:.0%} avg_sh={r['avg_sh']:.2f} → {status}")
        print()

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
