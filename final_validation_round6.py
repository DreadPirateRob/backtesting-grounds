"""Final comprehensive validation of Round 6 qualifying strategies at 1h.

Tests each strategy with:
- Multiple param sets to confirm robustness (not lucky param pick)
- Full 5yr sample at 10bps and 5bps
- Rolling 12-month windows (16 windows, 3-month step)
- Per-year breakdown to check consistency across regimes
- Signal overlap with existing Round 5 strategies
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import rsi, bollinger_bands, atr

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


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 1: CASCADE BOUNCE
# ═══════════════════════════════════════════════════════════════════

def cascade_bounce(df, cum_move_pct=-8.0, lookback=10, vol_z_window=48,
                   vol_z_thresh=2.0, stabilize_bars=0, hold_bars=16,
                   cooldown=3):
    close = df['close'].values
    vol = df['volume'].values
    vol_mean = pd.Series(vol).rolling(vol_z_window).mean().values
    vol_std = pd.Series(vol).rolling(vol_z_window).std().values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    cascade_bar = -999
    warmup = max(lookback, vol_z_window) + max(stabilize_bars, 1)

    for i in range(warmup, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        cum_ret = (close[i] / close[i - lookback] - 1) * 100
        if vol_std[i] > 0 and not np.isnan(vol_std[i]):
            vol_z = (vol[i] - vol_mean[i]) / vol_std[i]
        else:
            vol_z = 0

        if cum_ret < cum_move_pct and vol_z > vol_z_thresh:
            cascade_bar = i

        if stabilize_bars == 0:
            # Immediate entry
            if cum_ret < cum_move_pct and vol_z > vol_z_thresh:
                sig[i] = 1
                hold_remaining = hold_bars - 1
                cooldown_remaining = cooldown
                cascade_bar = -999
        elif cascade_bar >= 0 and i - cascade_bar == stabilize_bars:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown
            cascade_bar = -999

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: ATR SPIKE REVERSAL
# ═══════════════════════════════════════════════════════════════════

def atr_spike_reversal(df, atr_period=14, atr_mult=3.5, hold_bars=8,
                       cooldown=2):
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    atr_vals = atr(df, atr_period).values

    bar_range = high - low
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(atr_period * 2, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        if np.isnan(atr_vals[i]) or atr_vals[i] == 0:
            continue

        if bar_range[i] > atr_mult * atr_vals[i] and close[i] > open_[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 3: SELLOFF + VOLUME CONFIRMATION
# ═══════════════════════════════════════════════════════════════════

def selloff_vol_confirm(df, lookback=24, drop_pct=-8.0, vol_mult=2.0,
                        hold_bars=16, cooldown=3):
    close = df['close'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(max(lookback, 20), n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        cum_ret = (close[i] / close[i - lookback] - 1) * 100
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue

        if cum_ret < drop_pct and vol[i] > vol_mult * vol_avg[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# EXISTING STRATEGIES (for overlap analysis)
# ═══════════════════════════════════════════════════════════════════

def multi_bar_selloff(df, lookback=24, drop_pct=-8.0, hold_bars=16,
                      cooldown_bars=3):
    close = df['close'].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(lookback, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        cum_ret = (close[i] / close[i - lookback] - 1) * 100
        if cum_ret < drop_pct:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype='int8')


def extreme_range_bar(df, lookback=72, hold_bars=8, cooldown_bars=2):
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    bar_range = high - low
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(lookback, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if bar_range[i] < np.max(bar_range[i - lookback:i]):
            continue
        if close[i] <= open_[i]:
            continue
        sig[i] = 1
        hold_remaining = hold_bars - 1
        cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# FULL EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════

def full_evaluation(name, signal_fn, data, summary):
    for asset, df in data.items():
        sig = signal_fn(df)
        m10 = compute_metrics(run_backtest(df, sig, CFG_10))
        m5 = compute_metrics(run_backtest(df, sig, CFG_5))

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

        yearly = per_year_analysis(df, signal_fn, CFG_10)

        print(f"\n  {asset}:")
        print(f"    Full sample (10bps): Sharpe={m10['sharpe_ratio']:.2f} "
              f"Ret={m10['total_return_pct']:.1f}% "
              f"Trades={m10['total_trades']} WR={m10['win_rate_pct']:.1f}% "
              f"DD={m10['max_drawdown_pct']:.1f}%")
        print(f"    Full sample ( 5bps): Sharpe={m5['sharpe_ratio']:.2f} "
              f"Ret={m5['total_return_pct']:.1f}%")
        print(f"    Rolling ({n} windows): avg={avg_sh:.2f} med={med_sh:.2f} "
              f"frac>0={frac_pos:.0%} frac>0.5={frac_good:.0%}")

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
    print("FINAL VALIDATION: ROUND 6 QUALIFYING STRATEGIES AT 1h")
    print("=" * 120)

    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        print(f"{len(df)} bars")
        data[asset] = df

    summary = []

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 1: CASCADE BOUNCE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# STRATEGY 1: CASCADE BOUNCE (Liquidation cascade → buy)")
    print("# Mechanism: Cumulative -8% over 10 bars + volume z>2 → buy immediately")
    print("# Best params: cum_move=-8%, lookback=10, vol_z=2.0, stabilize=0, hold=16")
    print(f"{'#'*120}")

    full_evaluation(
        "CascBounce",
        lambda df: cascade_bounce(df, cum_move_pct=-8, lookback=10,
                                  vol_z_thresh=2.0, stabilize_bars=0,
                                  hold_bars=16),
        data, summary)

    # Parameter sensitivity
    print(f"\n  --- Cascade Bounce parameter sensitivity ---")
    cb_params = [
        ("CB_d8_lb10_s0_h12",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=0, hold_bars=12)),
        ("CB_d8_lb10_s0_h20",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=0, hold_bars=20)),
        ("CB_d8_lb10_s0_h24",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=0, hold_bars=24)),
        ("CB_d8_lb10_s1_h16",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=1, hold_bars=16)),
        ("CB_d8_lb10_s2_h16",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=2, hold_bars=16)),
        ("CB_d8_lb10_s2_h24",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=2, hold_bars=24)),
        ("CB_d8_lb15_s0_h16",  dict(cum_move_pct=-8, lookback=15, stabilize_bars=0, hold_bars=16)),
        ("CB_d7_lb10_s0_h16",  dict(cum_move_pct=-7, lookback=10, stabilize_bars=0, hold_bars=16)),
        ("CB_d10_lb10_s0_h16", dict(cum_move_pct=-10, lookback=10, stabilize_bars=0, hold_bars=16)),
        ("CB_d10_lb10_s2_h24", dict(cum_move_pct=-10, lookback=10, stabilize_bars=2, hold_bars=24)),
        ("CB_d8_vz1.5_s0_h16", dict(cum_move_pct=-8, vol_z_thresh=1.5, stabilize_bars=0, hold_bars=16)),
        ("CB_d8_vz2.5_s0_h16", dict(cum_move_pct=-8, vol_z_thresh=2.5, stabilize_bars=0, hold_bars=16)),
    ]
    for label, params in cb_params:
        row = f"  {label:<24}"
        for asset, df in data.items():
            sig = cascade_bounce(df, **params)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 2: ATR SPIKE REVERSAL
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# STRATEGY 2: ATR SPIKE REVERSAL (Volatility anomaly → buy)")
    print("# Mechanism: Bar range > 3.5× ATR(14) + bullish close → buy, hold 8")
    print(f"{'#'*120}")

    full_evaluation(
        "ATRSpike",
        lambda df: atr_spike_reversal(df, atr_mult=3.5, hold_bars=8),
        data, summary)

    print(f"\n  --- ATR Spike parameter sensitivity ---")
    atr_params = [
        ("ATR_m2.5_h8",  dict(atr_mult=2.5, hold_bars=8)),
        ("ATR_m2.5_h12", dict(atr_mult=2.5, hold_bars=12)),
        ("ATR_m3.0_h8",  dict(atr_mult=3.0, hold_bars=8)),
        ("ATR_m3.0_h12", dict(atr_mult=3.0, hold_bars=12)),
        ("ATR_m3.5_h12", dict(atr_mult=3.5, hold_bars=12)),
        ("ATR_m4.0_h8",  dict(atr_mult=4.0, hold_bars=8)),
        ("ATR_m4.0_h12", dict(atr_mult=4.0, hold_bars=12)),
        ("ATR_m3.5_h6",  dict(atr_mult=3.5, hold_bars=6)),
        ("ATR_m3.5_h10", dict(atr_mult=3.5, hold_bars=10)),
        ("ATR_m3.0_h6",  dict(atr_mult=3.0, hold_bars=6)),
        ("ATR_m3.5_p20_h8", dict(atr_period=20, atr_mult=3.5, hold_bars=8)),
        ("ATR_m3.5_p10_h8", dict(atr_period=10, atr_mult=3.5, hold_bars=8)),
    ]
    for label, params in atr_params:
        row = f"  {label:<24}"
        for asset, df in data.items():
            sig = atr_spike_reversal(df, **params)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 3: SELLOFF + VOLUME CONFIRMATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# STRATEGY 3: SELLOFF + VOLUME CONFIRMATION")
    print("# Mechanism: Cumulative -8% over 24 bars + bar vol > 2× avg → buy, hold 16")
    print(f"{'#'*120}")

    full_evaluation(
        "SelloffVC",
        lambda df: selloff_vol_confirm(df, drop_pct=-8.0, vol_mult=2.0,
                                       hold_bars=16),
        data, summary)

    print(f"\n  --- Selloff+VC parameter sensitivity ---")
    svc_params = [
        ("SVC_d6_v1.5_h12",  dict(drop_pct=-6.0, vol_mult=1.5, hold_bars=12)),
        ("SVC_d6_v1.5_h16",  dict(drop_pct=-6.0, vol_mult=1.5, hold_bars=16)),
        ("SVC_d6_v2.0_h12",  dict(drop_pct=-6.0, vol_mult=2.0, hold_bars=12)),
        ("SVC_d6_v2.0_h16",  dict(drop_pct=-6.0, vol_mult=2.0, hold_bars=16)),
        ("SVC_d8_v1.5_h16",  dict(drop_pct=-8.0, vol_mult=1.5, hold_bars=16)),
        ("SVC_d8_v1.5_h24",  dict(drop_pct=-8.0, vol_mult=1.5, hold_bars=24)),
        ("SVC_d8_v2.0_h12",  dict(drop_pct=-8.0, vol_mult=2.0, hold_bars=12)),
        ("SVC_d8_v2.0_h24",  dict(drop_pct=-8.0, vol_mult=2.0, hold_bars=24)),
        ("SVC_d10_v1.5_h16", dict(drop_pct=-10.0, vol_mult=1.5, hold_bars=16)),
        ("SVC_d10_v2.0_h16", dict(drop_pct=-10.0, vol_mult=2.0, hold_bars=16)),
        ("SVC_d5_v2.0_h12",  dict(drop_pct=-5.0, vol_mult=2.0, lookback=18, hold_bars=12)),
        ("SVC_d8_v2.0_lb18_h16", dict(drop_pct=-8.0, vol_mult=2.0, lookback=18, hold_bars=16)),
    ]
    for label, params in svc_params:
        row = f"  {label:<24}"
        for asset, df in data.items():
            sig = selloff_vol_confirm(df, **params)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL OVERLAP ANALYSIS (with existing Round 5 strategies)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# SIGNAL OVERLAP: Round 6 vs Round 5 strategies")
    print(f"{'#'*120}")

    for asset, df in data.items():
        # Round 5 signals
        sig_ms = multi_bar_selloff(df)
        sig_er = extreme_range_bar(df)

        # Round 6 signals
        sig_cb = cascade_bounce(df, cum_move_pct=-8, lookback=10,
                                vol_z_thresh=2.0, stabilize_bars=0,
                                hold_bars=16)
        sig_atr = atr_spike_reversal(df, atr_mult=3.5, hold_bars=8)
        sig_svc = selloff_vol_confirm(df, drop_pct=-8.0, vol_mult=2.0,
                                      hold_bars=16)

        all_sigs = {
            "Selloff(R5)": sig_ms,
            "ExtrRange(R5)": sig_er,
            "CascBounce": sig_cb,
            "ATRSpike": sig_atr,
            "SelloffVC": sig_svc,
        }

        n = len(df)
        print(f"\n  {asset} ({n} bars):")

        # Time in position
        for name, sig in all_sigs.items():
            pos = (sig != 0).sum()
            print(f"    {name:<16} in position: {pos:>5} bars ({pos/n*100:.1f}%)")

        # Pairwise overlap
        print(f"    --- Pairwise overlap (% of smaller position set) ---")
        names = list(all_sigs.keys())
        for i_idx in range(len(names)):
            for j_idx in range(i_idx + 1, len(names)):
                n1 = names[i_idx]
                n2 = names[j_idx]
                s1 = all_sigs[n1]
                s2 = all_sigs[n2]
                overlap = ((s1 != 0) & (s2 != 0)).sum()
                smaller = max(min((s1 != 0).sum(), (s2 != 0).sum()), 1)
                pct = overlap / smaller * 100
                marker = " ***HIGH" if pct > 50 else ""
                print(f"    {n1} ∩ {n2}: {overlap} bars ({pct:.1f}%){marker}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("FINAL SUMMARY")
    print(f"{'='*120}")
    print(f"{'Strategy':<14} {'Asset':<5} {'Sh10':>5} {'Sh5':>5} {'Tr':>5} "
          f"{'WR%':>5} {'Ret%':>7} {'DD%':>6} {'Win':>4} {'F>0':>5} "
          f"{'F>.5':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 120)
    for r in summary:
        print(f"{r['name']:<14} {r['asset']:<5} {r['sh10']:>5.2f} "
              f"{r['sh5']:>5.2f} {r['trades']:>5} {r['wr']:>5.1f} "
              f"{r['ret']:>7.1f} {r['dd']:>6.1f} {r['n_win']:>4} "
              f"{r['frac_pos']:>5.0%} {r['frac_good']:>5.0%} "
              f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Qualification
    print(f"\n{'='*120}")
    print("QUALIFICATION ASSESSMENT")
    print(f"{'='*120}")
    print("Criteria: Sharpe > 0 at 10bps, frac>0 >= 50%, trades > 20")
    print()

    for strat_name in ["CascBounce", "ATRSpike", "SelloffVC"]:
        strat_rows = [r for r in summary if r['name'] == strat_name]
        passing = [r for r in strat_rows
                   if r['sh10'] > 0 and r['frac_pos'] >= 0.50 and r['trades'] > 20]
        strong = [r for r in strat_rows
                  if r['sh10'] > 0 and r['frac_pos'] >= 0.60 and r['trades'] > 20]
        print(f"  {strat_name}:")
        for r in strat_rows:
            if r['sh10'] > 0 and r['frac_pos'] >= 0.60:
                status = "STRONG PASS"
            elif r['sh10'] > 0 and r['frac_pos'] >= 0.50:
                status = "PASS"
            elif r['sh10'] > 0:
                status = "BORDERLINE"
            else:
                status = "FAIL"
            print(f"    {r['asset']}: Sharpe={r['sh10']:.2f} Trades={r['trades']} "
                  f"frac>0={r['frac_pos']:.0%} avg_sh={r['avg_sh']:.2f} → {status}")
        print()

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
