"""Round 7 FIX: Correct look-ahead bias in daily MACD strategies.

The previous round had a critical bug: daily MACD was shifted by only 1 hour
instead of a full day. This means bar i at 01:00 UTC would see a daily MACD
value computed from data through 23:00 UTC of the same day — look-ahead bias.

Fix: shift daily indicators by 24 bars (1 full day) to ensure no future info.

Also includes: comprehensive Donchian validation and signal overlap analysis.
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import rsi, bollinger_bands, atr, macd

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


def evaluate(signal_fn, df, cfg=CFG_10):
    sig = signal_fn(df)
    res = run_backtest(df, sig, cfg)
    m = compute_metrics(res)
    windows = []
    for ws, we, sl in rolling_windows(df):
        sw = signal_fn(sl)
        wm = compute_metrics(run_backtest(sl, sw, cfg))
        windows.append(wm['sharpe_ratio'])
    n = len(windows)
    return {
        "sharpe": m['sharpe_ratio'],
        "trades": m['total_trades'],
        "wr": m['win_rate_pct'],
        "ret": m['total_return_pct'],
        "dd": m['max_drawdown_pct'],
        "frac_pos": sum(1 for s in windows if s > 0) / n if n else 0,
        "avg_sh": np.mean(windows) if windows else 0,
        "med_sh": np.median(windows) if windows else 0,
        "n_win": n,
    }


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
# FIXED: DAILY MACD CROSS (shift by 24 bars, not 1)
# ═══════════════════════════════════════════════════════════════════

def daily_macd_cross_fixed(df, hold_bars=24, cooldown=12):
    """Buy when daily MACD histogram turns positive. FIXED: 24-bar shift."""
    close = df['close']

    daily = df.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])

    _, _, daily_hist = macd(daily['close'])

    # CRITICAL FIX: shift daily series by 1 DAY before reindexing to hourly
    # This ensures Tuesday's hourly bars see Monday's daily MACD, not Tuesday's
    daily_hist_shifted = daily_hist.shift(1)  # Now each day sees PRIOR day's value
    daily_hist_hourly = daily_hist_shifted.reindex(df.index, method='ffill')

    daily_hist_prev_shifted = daily_hist.shift(2)  # Two days prior
    daily_hist_prev_hourly = daily_hist_prev_shifted.reindex(df.index, method='ffill')

    vals = daily_hist_hourly.values
    prev_vals = daily_hist_prev_hourly.values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(700, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(vals[i]) or np.isnan(prev_vals[i]):
            continue

        if vals[i] > 0 and prev_vals[i] <= 0:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# FIXED: MTF TREND + PULLBACK
# ═══════════════════════════════════════════════════════════════════

def mtf_trend_pullback_fixed(df, rsi_period=14, rsi_thresh=35,
                             hold_bars=12, cooldown=3):
    """Daily MACD bullish + 1H RSI oversold pullback. FIXED: 24-bar shift."""
    close = df['close']

    daily = df.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])

    _, _, daily_hist = macd(daily['close'])
    daily_hist_shifted = daily_hist.shift(1)
    daily_hist_hourly = daily_hist_shifted.reindex(df.index, method='ffill')
    daily_hist_vals = daily_hist_hourly.values

    rsi_vals = rsi(close, rsi_period).values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    warmup = 700

    for i in range(warmup, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(daily_hist_vals[i]) or np.isnan(rsi_vals[i]):
            continue

        if daily_hist_vals[i] > 0 and rsi_vals[i] < rsi_thresh:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# DONCHIAN BREAKOUT (already correct, no bias)
# ═══════════════════════════════════════════════════════════════════

def donchian_breakout(df, entry_period=480, vol_mult=1.5, hold_bars=24,
                      cooldown=12):
    close = df['close'].values
    high = df['high'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(entry_period + 1, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        highest = np.max(high[i - entry_period:i])
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue

        if close[i] > highest and vol[i] > vol_mult * vol_avg[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# EXISTING STRATEGIES (for overlap analysis)
# ═══════════════════════════════════════════════════════════════════

def multi_bar_selloff(df):
    close = df['close'].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0
    for i in range(24, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        cum_ret = (close[i] / close[i - 24] - 1) * 100
        if cum_ret < -8.0:
            sig[i] = 1
            hold_remaining = 15
            cooldown = 3
    return pd.Series(sig, index=df.index, dtype='int8')


def extreme_range_bar(df):
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    bar_range = high - low
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0
    for i in range(72, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if bar_range[i] < np.max(bar_range[i - 72:i]):
            continue
        if close[i] <= open_[i]:
            continue
        sig[i] = 1
        hold_remaining = 7
        cooldown = 2
    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# GREEN MOMENTUM (from Round 7, legitimate)
# ═══════════════════════════════════════════════════════════════════

def consec_green_momentum(df, min_green=4, min_gain_pct=4.0, hold_bars=12,
                          cooldown=3):
    close = df['close'].values
    open_ = df['open'].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(min_green, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        green_count = 0
        for j in range(i, max(i - 10, -1), -1):
            if close[j] > open_[j]:
                green_count += 1
            else:
                break

        if green_count < min_green:
            continue

        cum_gain = (close[i] / close[i - green_count] - 1) * 100
        if cum_gain > min_gain_pct:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


def full_eval(name, signal_fn, data, summary):
    for asset, df in data.items():
        sig = signal_fn(df)
        m10 = compute_metrics(run_backtest(df, sig, CFG_10))
        m5 = compute_metrics(run_backtest(df, sig, CFG_5))

        windows = []
        for ws, we, sl in rolling_windows(df):
            sw = signal_fn(sl)
            wm = compute_metrics(run_backtest(sl, sw, CFG_10))
            windows.append(wm['sharpe_ratio'])

        n = len(windows)
        sharpes = [w for w in windows]
        frac_pos = sum(1 for s in sharpes if s > 0) / n if n else 0
        frac_good = sum(1 for s in sharpes if s > 0.5) / n if n else 0
        avg_sh = np.mean(sharpes) if sharpes else 0
        med_sh = np.median(sharpes) if sharpes else 0

        yearly = per_year_analysis(df, signal_fn, CFG_10)

        print(f"\n  {asset}:")
        print(f"    10bps: Sharpe={m10['sharpe_ratio']:.2f} Ret={m10['total_return_pct']:.1f}% "
              f"Trades={m10['total_trades']} WR={m10['win_rate_pct']:.1f}% DD={m10['max_drawdown_pct']:.1f}%")
        print(f"     5bps: Sharpe={m5['sharpe_ratio']:.2f} Ret={m5['total_return_pct']:.1f}%")
        print(f"    Rolling ({n} win): avg={avg_sh:.2f} med={med_sh:.2f} "
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
    print("ROUND 7 FIX: BIAS-CORRECTED DAILY MACD + DONCHIAN VALIDATION")
    print("=" * 120)

    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        print(f"{len(df)} bars")
        data[asset] = df

    summary = []

    def test_quick(name, signal_fn):
        row = f"  {name:<42}"
        for asset, df in data.items():
            try:
                r = evaluate(signal_fn, df)
                row += f"  {asset}:{r['sharpe']:+.2f}({r['trades']}t,{r['frac_pos']:.0%}w)"
            except Exception as e:
                row += f"  {asset}:ERR({str(e)[:30]})"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # 1. DAILY MACD CROSS — BIAS FIXED
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("1. DAILY MACD CROSS — BIAS FIXED (shifted by full day)")
    print(f"{'='*120}")

    macd_params = [
        ("h24_c12", dict(hold_bars=24, cooldown=12)),
        ("h48_c24", dict(hold_bars=48, cooldown=24)),
        ("h72_c24", dict(hold_bars=72, cooldown=24)),
        ("h96_c48", dict(hold_bars=96, cooldown=48)),
        ("h12_c6",  dict(hold_bars=12, cooldown=6)),
        ("h120_c48", dict(hold_bars=120, cooldown=48)),
    ]
    for label, params in macd_params:
        test_quick(f"MACDFix_{label}",
                   lambda df, p=params: daily_macd_cross_fixed(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 2. MTF TREND + PULLBACK — BIAS FIXED
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("2. MTF TREND + PULLBACK — BIAS FIXED")
    print(f"{'='*120}")

    mtf_params = [
        ("RSI35_h8",  dict(rsi_thresh=35, hold_bars=8)),
        ("RSI35_h12", dict(rsi_thresh=35, hold_bars=12)),
        ("RSI30_h8",  dict(rsi_thresh=30, hold_bars=8)),
        ("RSI40_h12", dict(rsi_thresh=40, hold_bars=12)),
    ]
    for label, params in mtf_params:
        test_quick(f"MTFFix_{label}",
                   lambda df, p=params: mtf_trend_pullback_fixed(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 3. DONCHIAN BREAKOUT — COMPREHENSIVE VALIDATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# DONCHIAN BREAKOUT — COMPREHENSIVE VALIDATION")
    print("# Best params: entry_period=480, vol_mult=2.0, hold=24, cooldown=12")
    print(f"{'#'*120}")

    full_eval("Donchian",
              lambda df: donchian_breakout(df, entry_period=480, vol_mult=2.0,
                                           hold_bars=24, cooldown=12),
              data, summary)

    print(f"\n  --- Donchian parameter sensitivity ---")
    don_params = [
        ("p240_v1.5_h24",  dict(entry_period=240, vol_mult=1.5, hold_bars=24)),
        ("p240_v2.0_h24",  dict(entry_period=240, vol_mult=2.0, hold_bars=24)),
        ("p360_v1.5_h24",  dict(entry_period=360, vol_mult=1.5, hold_bars=24)),
        ("p360_v2.0_h24",  dict(entry_period=360, vol_mult=2.0, hold_bars=24)),
        ("p480_v1.5_h24",  dict(entry_period=480, vol_mult=1.5, hold_bars=24)),
        ("p480_v2.0_h48",  dict(entry_period=480, vol_mult=2.0, hold_bars=48)),
        ("p480_v1.5_h48",  dict(entry_period=480, vol_mult=1.5, hold_bars=48)),
        ("p480_v2.5_h24",  dict(entry_period=480, vol_mult=2.5, hold_bars=24)),
        ("p720_v1.5_h24",  dict(entry_period=720, vol_mult=1.5, hold_bars=24)),
        ("p720_v1.5_h48",  dict(entry_period=720, vol_mult=1.5, hold_bars=48)),
        ("p720_v2.0_h24",  dict(entry_period=720, vol_mult=2.0, hold_bars=24)),
        ("p720_v2.0_h48",  dict(entry_period=720, vol_mult=2.0, hold_bars=48)),
    ]
    for label, params in don_params:
        row = f"  {label:<24}"
        for asset, df in data.items():
            sig = donchian_breakout(df, **params)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # 4. GREEN MOMENTUM — COMPREHENSIVE VALIDATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# GREEN MOMENTUM — COMPREHENSIVE VALIDATION")
    print("# Best params: min_green=4, min_gain=4%, hold=12")
    print(f"{'#'*120}")

    full_eval("GreenMom",
              lambda df: consec_green_momentum(df, min_green=4,
                                               min_gain_pct=4.0,
                                               hold_bars=12),
              data, summary)

    print(f"\n  --- Green Momentum parameter sensitivity ---")
    green_params = [
        ("g3_3p_h8",   dict(min_green=3, min_gain_pct=3.0, hold_bars=8)),
        ("g3_3p_h12",  dict(min_green=3, min_gain_pct=3.0, hold_bars=12)),
        ("g4_3p_h8",   dict(min_green=4, min_gain_pct=3.0, hold_bars=8)),
        ("g4_3p_h12",  dict(min_green=4, min_gain_pct=3.0, hold_bars=12)),
        ("g4_4p_h8",   dict(min_green=4, min_gain_pct=4.0, hold_bars=8)),
        ("g4_5p_h12",  dict(min_green=4, min_gain_pct=5.0, hold_bars=12)),
        ("g4_5p_h16",  dict(min_green=4, min_gain_pct=5.0, hold_bars=16)),
        ("g5_4p_h12",  dict(min_green=5, min_gain_pct=4.0, hold_bars=12)),
        ("g5_5p_h12",  dict(min_green=5, min_gain_pct=5.0, hold_bars=12)),
        ("g5_5p_h16",  dict(min_green=5, min_gain_pct=5.0, hold_bars=16)),
        ("g5_5p_h24",  dict(min_green=5, min_gain_pct=5.0, hold_bars=24)),
        ("g4_4p_h16",  dict(min_green=4, min_gain_pct=4.0, hold_bars=16)),
    ]
    for label, params in green_params:
        row = f"  {label:<24}"
        for asset, df in data.items():
            sig = consec_green_momentum(df, **params)
            m = compute_metrics(run_backtest(df, sig, CFG_10))
            row += f"  {asset}:{m['sharpe_ratio']:+.2f}({m['total_trades']}t)"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL OVERLAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'#'*120}")
    print("# SIGNAL OVERLAP: New strategies vs existing Round 5 strategies")
    print(f"{'#'*120}")

    for asset, df in data.items():
        sig_ms = multi_bar_selloff(df)
        sig_er = extreme_range_bar(df)
        sig_don = donchian_breakout(df, entry_period=480, vol_mult=2.0,
                                    hold_bars=24, cooldown=12)
        sig_gm = consec_green_momentum(df, min_green=4, min_gain_pct=4.0,
                                        hold_bars=12)

        all_sigs = {
            "Selloff(R5)": sig_ms,
            "ExtrRange(R5)": sig_er,
            "Donchian": sig_don,
            "GreenMom": sig_gm,
        }

        n = len(df)
        print(f"\n  {asset} ({n} bars):")
        for name, sig in all_sigs.items():
            pos = (sig != 0).sum()
            print(f"    {name:<16} in position: {pos:>5} bars ({pos/n*100:.1f}%)")

        print(f"    --- Pairwise overlap ---")
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
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"{'Strategy':<14} {'Asset':<5} {'Sh10':>5} {'Sh5':>5} {'Tr':>5} "
          f"{'WR%':>5} {'Ret%':>7} {'DD%':>6} {'Win':>4} {'F>0':>5} "
          f"{'F>.5':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 110)
    for r in summary:
        print(f"{r['name']:<14} {r['asset']:<5} {r['sh10']:>5.2f} "
              f"{r['sh5']:>5.2f} {r['trades']:>5} {r['wr']:>5.1f} "
              f"{r['ret']:>7.1f} {r['dd']:>6.1f} {r['n_win']:>4} "
              f"{r['frac_pos']:>5.0%} {r['frac_good']:>5.0%} "
              f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Qualification
    print(f"\n{'='*120}")
    print("QUALIFICATION")
    print(f"{'='*120}")
    for name in ["Donchian", "GreenMom"]:
        rows = [r for r in summary if r['name'] == name]
        print(f"\n  {name}:")
        for r in rows:
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

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
