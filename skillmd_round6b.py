"""Round 6b: Refinement of promising leads + new extreme-event detectors.

Key insight from Round 6: only RARE, EXTREME events produce reliable bounces
that survive 10bps fees at 1h. Moderate indicator thresholds (RSI<25) fail on BTC/ETH.

This round focuses on:
1. Cascade Bounce extended param sweep (best cross-asset lead)
2. Consecutive Red Bars crash detector (new mechanism)
3. ATR Spike Reversal (volatility anomaly)
4. Volume-Weighted Selloff (combines price + volume decline)
5. Session-restricted Triple MR (MR best 20:00-08:00 UTC per SKILL.md)
6. VWAP Deviation extreme (price far below daily VWAP)
7. Triple MR with even wider BTC thresholds
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import rsi, bollinger_bands, adx, atr, vwap_daily

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


# ═══════════════════════════════════════════════════════════════════
# 1. CASCADE BOUNCE — EXTENDED PARAM SWEEP
# ═══════════════════════════════════════════════════════════════════

def cascade_bounce(df, cum_move_pct=-8.0, lookback=10, vol_z_window=48,
                   vol_z_thresh=2.0, stabilize_bars=2, hold_bars=16,
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
    warmup = max(lookback, vol_z_window) + stabilize_bars

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

        if cascade_bar >= 0 and i - cascade_bar == stabilize_bars:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown
            cascade_bar = -999

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 2. CONSECUTIVE RED BARS + CRASH DETECTOR
# N consecutive down bars + cumulative drop > threshold → buy
# ═══════════════════════════════════════════════════════════════════

def consec_red_crash(df, min_red_bars=4, min_drop_pct=-4.0, hold_bars=8,
                     cooldown=3):
    close = df['close'].values
    open_ = df['open'].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(min_red_bars, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # Count consecutive red (bearish) bars ending at i
        red_count = 0
        for j in range(i, max(i - 10, -1), -1):
            if close[j] < open_[j]:
                red_count += 1
            else:
                break

        if red_count < min_red_bars:
            continue

        # Cumulative drop over those bars
        cum_ret = (close[i] / close[i - red_count] - 1) * 100
        if cum_ret < min_drop_pct:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 3. ATR SPIKE REVERSAL
# Current bar range > mult × ATR(period) + bullish close → buy
# ═══════════════════════════════════════════════════════════════════

def atr_spike_reversal(df, atr_period=14, atr_mult=3.0, hold_bars=8,
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
# 4. VOLUME-WEIGHTED SELLOFF
# Cumulative (return × volume) falls below threshold → buy
# Combines crash magnitude with volume intensity
# ═══════════════════════════════════════════════════════════════════

def vol_weighted_selloff(df, lookback=24, threshold=-1e8, hold_bars=12,
                         cooldown=3):
    """threshold is in absolute terms (price_return × volume).
    We normalize by computing z-score of the vol-weighted return."""
    close = df['close'].values
    vol = df['volume'].values
    n = len(df)

    # Compute per-bar return × volume
    ret_vol = np.zeros(n)
    for i in range(1, n):
        ret_vol[i] = (close[i] / close[i - 1] - 1) * vol[i]

    # Rolling sum and z-score
    rv_series = pd.Series(ret_vol)
    rv_sum = rv_series.rolling(lookback).sum().values
    rv_mean = rv_series.rolling(lookback * 10).apply(
        lambda x: pd.Series(x).rolling(lookback).sum().mean(), raw=False
    ).values if False else None  # Too slow, use simpler approach

    # Simpler: z-score of rolling sum
    rv_sum_series = pd.Series(rv_sum)
    rv_rolling_mean = rv_sum_series.rolling(lookback * 4).mean().values
    rv_rolling_std = rv_sum_series.rolling(lookback * 4).std().values

    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    warmup = lookback * 5

    for i in range(warmup, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        if (np.isnan(rv_rolling_std[i]) or rv_rolling_std[i] == 0 or
                np.isnan(rv_sum[i]) or np.isnan(rv_rolling_mean[i])):
            continue

        z = (rv_sum[i] - rv_rolling_mean[i]) / rv_rolling_std[i]
        if z < threshold:  # threshold is negative z-score
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 5. SESSION-RESTRICTED TRIPLE MR
# Triple MR but only during 20:00-08:00 UTC (prime MR window)
# ═══════════════════════════════════════════════════════════════════

def session_triple_mr(df, rsi_period=14, rsi_thresh=20, bb_period=20,
                      bb_std=2.5, vol_mult=2.0, hold_bars=5, cooldown=2,
                      session_start=20, session_end=8):
    rsi_vals = rsi(df['close'], rsi_period).values
    _, _, lower = bollinger_bands(df['close'], bb_period, bb_std)
    lower_vals = lower.values
    close = df['close'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values
    hours = df.index.hour

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    warmup = max(rsi_period, bb_period, 20) + 1

    for i in range(warmup, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # Session filter (wraps around midnight)
        h = hours[i]
        if session_start > session_end:
            in_session = h >= session_start or h < session_end
        else:
            in_session = session_start <= h < session_end

        if not in_session:
            continue

        if np.isnan(rsi_vals[i]) or np.isnan(lower_vals[i]) or np.isnan(vol_avg[i]):
            continue
        if (rsi_vals[i] < rsi_thresh and
                close[i] < lower_vals[i] and
                vol[i] > vol_mult * vol_avg[i]):
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 6. VWAP DEVIATION EXTREME
# Price at -2σ or more below daily VWAP → buy
# ═══════════════════════════════════════════════════════════════════

def vwap_extreme(df, sigma_mult=2.0, hold_bars=6, cooldown=2):
    vwap_df = vwap_daily(df, bands=True)
    vwap_lower = vwap_df['vwap_lower'].values if 'vwap_lower' in vwap_df.columns else None
    vwap_vals = vwap_df['vwap'].values
    vwap_std_vals = (vwap_vals - vwap_lower) if vwap_lower is not None else None
    close = df['close'].values

    # Compute deviation from VWAP in terms of std
    # vwap_lower = vwap - 1*std, so std = vwap - vwap_lower
    vwap_std = vwap_df['vwap'].values - vwap_df['vwap_lower'].values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(1, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        if np.isnan(vwap_vals[i]) or np.isnan(vwap_std[i]) or vwap_std[i] == 0:
            continue

        z = (close[i] - vwap_vals[i]) / vwap_std[i]
        if z < -sigma_mult:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 7. TRIPLE MR WITH WIDER BTC THRESHOLDS
# RSI < 30 AND below BB(20,2.0) AND vol > 1.5x → buy
# ═══════════════════════════════════════════════════════════════════

def triple_mr_wide(df, rsi_period=14, rsi_thresh=30, bb_period=20,
                   bb_std=2.0, vol_mult=1.5, hold_bars=8, cooldown=2,
                   adx_period=14, adx_thresh=0):
    """Wide-threshold Triple MR with optional ADX filter."""
    rsi_vals = rsi(df['close'], rsi_period).values
    _, _, lower = bollinger_bands(df['close'], bb_period, bb_std)
    lower_vals = lower.values
    close = df['close'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    use_adx = adx_thresh > 0
    if use_adx:
        adx_vals = adx(df, adx_period)['adx'].values
    else:
        adx_vals = None

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    warmup = max(rsi_period, bb_period, 20, adx_period * 3 if use_adx else 0) + 1

    for i in range(warmup, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(rsi_vals[i]) or np.isnan(lower_vals[i]) or np.isnan(vol_avg[i]):
            continue
        if use_adx and (np.isnan(adx_vals[i]) or adx_vals[i] > adx_thresh):
            continue
        if (rsi_vals[i] < rsi_thresh and
                close[i] < lower_vals[i] and
                vol[i] > vol_mult * vol_avg[i]):
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 8. MULTI-BAR SELLOFF + VOLUME CONFIRMATION
# Cumulative drop + volume spike on the trigger bar
# ═══════════════════════════════════════════════════════════════════

def selloff_vol_confirm(df, lookback=24, drop_pct=-6.0, vol_mult=1.5,
                        hold_bars=12, cooldown=3):
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
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        print(f"{len(df)} bars")
        data[asset] = df

    all_results = []

    def test_strategy(name, signal_fn):
        row = f"  {name:<42}"
        for asset, df in data.items():
            try:
                r = evaluate(signal_fn, df)
                all_results.append({"name": name, "asset": asset, **r})
                row += f"  {asset}:{r['sharpe']:+.2f}({r['trades']}t,{r['frac_pos']:.0%}w)"
            except Exception as e:
                row += f"  {asset}:ERR({str(e)[:30]})"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # 1. CASCADE BOUNCE — EXTENDED SWEEP
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("1. CASCADE BOUNCE — EXTENDED SWEEP (around d8/vz2 sweet spot)")
    print(f"{'='*120}")

    cascade_params = [
        # Core: vary hold around d8
        ("d8_lb10_s2_h12",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=2, hold_bars=12)),
        ("d8_lb10_s2_h20",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=2, hold_bars=20)),
        ("d8_lb10_s2_h24",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=2, hold_bars=24)),
        ("d8_lb10_s2_h32",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=2, hold_bars=32)),
        # Vary lookback
        ("d8_lb15_s2_h16",  dict(cum_move_pct=-8, lookback=15, stabilize_bars=2, hold_bars=16)),
        ("d8_lb20_s2_h16",  dict(cum_move_pct=-8, lookback=20, stabilize_bars=2, hold_bars=16)),
        ("d8_lb24_s2_h16",  dict(cum_move_pct=-8, lookback=24, stabilize_bars=2, hold_bars=16)),
        # Vary threshold
        ("d6_lb10_s2_h16",  dict(cum_move_pct=-6, lookback=10, stabilize_bars=2, hold_bars=16)),
        ("d6_lb15_s2_h16",  dict(cum_move_pct=-6, lookback=15, stabilize_bars=2, hold_bars=16)),
        ("d7_lb10_s2_h16",  dict(cum_move_pct=-7, lookback=10, stabilize_bars=2, hold_bars=16)),
        ("d10_lb10_s2_h16", dict(cum_move_pct=-10, lookback=10, stabilize_bars=2, hold_bars=16)),
        ("d10_lb10_s2_h24", dict(cum_move_pct=-10, lookback=10, stabilize_bars=2, hold_bars=24)),
        # Lower vol threshold
        ("d8_lb10_vz1.5_s2_h16", dict(cum_move_pct=-8, lookback=10, vol_z_thresh=1.5, stabilize_bars=2, hold_bars=16)),
        # No stabilization
        ("d8_lb10_s0_h16",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=0, hold_bars=16)),
        ("d8_lb10_s1_h16",  dict(cum_move_pct=-8, lookback=10, stabilize_bars=1, hold_bars=16)),
    ]

    for label, params in cascade_params:
        test_strategy(f"Cascade_{label}",
                      lambda df, p=params: cascade_bounce(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 2. CONSECUTIVE RED BARS CRASH
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("2. CONSECUTIVE RED BARS + CRASH DETECTOR")
    print(f"{'='*120}")

    consec_params = [
        ("r4_d3_h8",   dict(min_red_bars=4, min_drop_pct=-3.0, hold_bars=8)),
        ("r4_d3_h12",  dict(min_red_bars=4, min_drop_pct=-3.0, hold_bars=12)),
        ("r4_d4_h8",   dict(min_red_bars=4, min_drop_pct=-4.0, hold_bars=8)),
        ("r4_d4_h12",  dict(min_red_bars=4, min_drop_pct=-4.0, hold_bars=12)),
        ("r5_d3_h12",  dict(min_red_bars=5, min_drop_pct=-3.0, hold_bars=12)),
        ("r5_d5_h12",  dict(min_red_bars=5, min_drop_pct=-5.0, hold_bars=12)),
        ("r5_d5_h16",  dict(min_red_bars=5, min_drop_pct=-5.0, hold_bars=16)),
        ("r3_d3_h8",   dict(min_red_bars=3, min_drop_pct=-3.0, hold_bars=8)),
        ("r6_d5_h16",  dict(min_red_bars=6, min_drop_pct=-5.0, hold_bars=16)),
    ]

    for label, params in consec_params:
        test_strategy(f"ConsecRed_{label}",
                      lambda df, p=params: consec_red_crash(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 3. ATR SPIKE REVERSAL
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("3. ATR SPIKE REVERSAL (bar range > N × ATR + bullish)")
    print(f"{'='*120}")

    atr_params = [
        ("m2.5_h8",   dict(atr_mult=2.5, hold_bars=8)),
        ("m2.5_h12",  dict(atr_mult=2.5, hold_bars=12)),
        ("m3.0_h8",   dict(atr_mult=3.0, hold_bars=8)),
        ("m3.0_h12",  dict(atr_mult=3.0, hold_bars=12)),
        ("m3.5_h8",   dict(atr_mult=3.5, hold_bars=8)),
        ("m3.5_h12",  dict(atr_mult=3.5, hold_bars=12)),
        ("m4.0_h12",  dict(atr_mult=4.0, hold_bars=12)),
        ("m2.0_h8",   dict(atr_mult=2.0, hold_bars=8)),
    ]

    for label, params in atr_params:
        test_strategy(f"ATRSpike_{label}",
                      lambda df, p=params: atr_spike_reversal(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 4. VOLUME-WEIGHTED SELLOFF (z-score)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("4. VOLUME-WEIGHTED SELLOFF (z-score of rolling vol×return)")
    print(f"{'='*120}")

    vws_params = [
        ("z-2_lb24_h12",   dict(threshold=-2.0, lookback=24, hold_bars=12)),
        ("z-2_lb24_h16",   dict(threshold=-2.0, lookback=24, hold_bars=16)),
        ("z-2.5_lb24_h12", dict(threshold=-2.5, lookback=24, hold_bars=12)),
        ("z-2.5_lb24_h16", dict(threshold=-2.5, lookback=24, hold_bars=16)),
        ("z-3_lb24_h16",   dict(threshold=-3.0, lookback=24, hold_bars=16)),
        ("z-2_lb12_h12",   dict(threshold=-2.0, lookback=12, hold_bars=12)),
        ("z-2_lb36_h16",   dict(threshold=-2.0, lookback=36, hold_bars=16)),
    ]

    for label, params in vws_params:
        test_strategy(f"VolWtSelloff_{label}",
                      lambda df, p=params: vol_weighted_selloff(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 5. SESSION-RESTRICTED TRIPLE MR
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("5. SESSION-RESTRICTED TRIPLE MR (20:00-08:00 UTC = prime MR)")
    print(f"{'='*120}")

    sess_tmr_params = [
        ("R20_BB2.5_V2_h5_night",  dict(rsi_thresh=20, bb_std=2.5, vol_mult=2.0, hold_bars=5, session_start=20, session_end=8)),
        ("R20_BB2.5_V2_h8_night",  dict(rsi_thresh=20, bb_std=2.5, vol_mult=2.0, hold_bars=8, session_start=20, session_end=8)),
        ("R25_BB2.0_V1.5_h8_night", dict(rsi_thresh=25, bb_std=2.0, vol_mult=1.5, hold_bars=8, session_start=20, session_end=8)),
        ("R20_BB2.5_V2_h5_allday", dict(rsi_thresh=20, bb_std=2.5, vol_mult=2.0, hold_bars=5, session_start=0, session_end=24)),
        ("R25_BB2.0_V2_h8_peak",   dict(rsi_thresh=25, bb_std=2.0, vol_mult=2.0, hold_bars=8, session_start=8, session_end=20)),
    ]

    for label, params in sess_tmr_params:
        test_strategy(f"SessTMR_{label}",
                      lambda df, p=params: session_triple_mr(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 6. VWAP DEVIATION EXTREME
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("6. VWAP DEVIATION EXTREME (price far below daily VWAP)")
    print(f"{'='*120}")

    vwap_params = [
        ("s2.0_h6",  dict(sigma_mult=2.0, hold_bars=6)),
        ("s2.0_h8",  dict(sigma_mult=2.0, hold_bars=8)),
        ("s2.5_h6",  dict(sigma_mult=2.5, hold_bars=6)),
        ("s2.5_h8",  dict(sigma_mult=2.5, hold_bars=8)),
        ("s3.0_h8",  dict(sigma_mult=3.0, hold_bars=8)),
        ("s1.5_h6",  dict(sigma_mult=1.5, hold_bars=6)),
    ]

    for label, params in vwap_params:
        test_strategy(f"VWAPExtr_{label}",
                      lambda df, p=params: vwap_extreme(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 7. TRIPLE MR WIDE (for BTC/ETH)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("7. TRIPLE MR WIDE + ADX FILTER (wider thresholds for BTC)")
    print(f"{'='*120}")

    tmr_wide_params = [
        ("R30_BB2.0_V1.5_h8",       dict(rsi_thresh=30, bb_std=2.0, vol_mult=1.5, hold_bars=8)),
        ("R30_BB2.0_V1.5_h12",      dict(rsi_thresh=30, bb_std=2.0, vol_mult=1.5, hold_bars=12)),
        ("R30_BB1.5_V1.5_h8",       dict(rsi_thresh=30, bb_std=1.5, vol_mult=1.5, hold_bars=8)),
        ("R35_BB2.0_V1.5_h8",       dict(rsi_thresh=35, bb_std=2.0, vol_mult=1.5, hold_bars=8)),
        ("R30_BB2.0_V1.5_h8_ADX25", dict(rsi_thresh=30, bb_std=2.0, vol_mult=1.5, hold_bars=8, adx_thresh=25)),
        ("R30_BB2.0_V1.5_h12_ADX25", dict(rsi_thresh=30, bb_std=2.0, vol_mult=1.5, hold_bars=12, adx_thresh=25)),
        ("R30_BB2.0_V1.5_h8_ADX30", dict(rsi_thresh=30, bb_std=2.0, vol_mult=1.5, hold_bars=8, adx_thresh=30)),
        ("R25_BB2.0_V1.5_h8_ADX25", dict(rsi_thresh=25, bb_std=2.0, vol_mult=1.5, hold_bars=8, adx_thresh=25)),
    ]

    for label, params in tmr_wide_params:
        test_strategy(f"TMRWide_{label}",
                      lambda df, p=params: triple_mr_wide(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 8. SELLOFF + VOLUME CONFIRMATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("8. MULTI-BAR SELLOFF + VOLUME CONFIRMATION")
    print(f"{'='*120}")

    svc_params = [
        ("d6_v1.5_lb24_h12",   dict(drop_pct=-6.0, vol_mult=1.5, lookback=24, hold_bars=12)),
        ("d6_v1.5_lb24_h16",   dict(drop_pct=-6.0, vol_mult=1.5, lookback=24, hold_bars=16)),
        ("d6_v2.0_lb24_h12",   dict(drop_pct=-6.0, vol_mult=2.0, lookback=24, hold_bars=12)),
        ("d8_v1.5_lb24_h16",   dict(drop_pct=-8.0, vol_mult=1.5, lookback=24, hold_bars=16)),
        ("d8_v2.0_lb24_h16",   dict(drop_pct=-8.0, vol_mult=2.0, lookback=24, hold_bars=16)),
        ("d5_v1.5_lb18_h12",   dict(drop_pct=-5.0, vol_mult=1.5, lookback=18, hold_bars=12)),
        ("d5_v2.0_lb18_h12",   dict(drop_pct=-5.0, vol_mult=2.0, lookback=18, hold_bars=12)),
        ("d8_v1.5_lb24_h24",   dict(drop_pct=-8.0, vol_mult=1.5, lookback=24, hold_bars=24)),
    ]

    for label, params in svc_params:
        test_strategy(f"SelloffVC_{label}",
                      lambda df, p=params: selloff_vol_confirm(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("ROUND 6b SUMMARY — TOP RESULTS (Sharpe > 0, frac>0 >= 50%, trades > 20)")
    print(f"{'='*120}")

    qualifying = [r for r in all_results
                  if r['sharpe'] > 0 and r['frac_pos'] >= 0.50 and r['trades'] > 20]
    qualifying.sort(key=lambda x: x['avg_sh'], reverse=True)

    print(f"\n{'Name':<44} {'Asset':<5} {'Sh':>5} {'Tr':>5} {'WR':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'F>0':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 110)

    for r in qualifying[:40]:
        print(f"{r['name']:<44} {r['asset']:<5} {r['sharpe']:>5.2f} "
              f"{r['trades']:>5} {r['wr']:>5.1f} {r['ret']:>7.1f} "
              f"{r['dd']:>6.1f} {r['frac_pos']:>5.0%} "
              f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Cross-asset analysis for top strategies
    print(f"\n{'='*120}")
    print("CROSS-ASSET QUALIFICATION (strategies positive on 2+ assets)")
    print(f"{'='*120}")

    strat_names = set(r['name'] for r in all_results)
    for name in sorted(strat_names):
        rows = [r for r in all_results if r['name'] == name]
        positive = [r for r in rows if r['sharpe'] > 0]
        stable = [r for r in rows if r['sharpe'] > 0 and r['frac_pos'] >= 0.50]
        if len(positive) >= 2:
            assets_str = " | ".join(
                f"{r['asset']}:{r['sharpe']:+.2f}(F{r['frac_pos']:.0%})"
                for r in rows
            )
            mark = "***" if len(stable) >= 2 else ""
            print(f"  {mark}{name:<42} {assets_str}")

    # Strategy family summary
    print(f"\n{'='*120}")
    print("FAMILY SCORECARD")
    print(f"{'='*120}")

    families = {}
    for r in all_results:
        family = r['name'].split('_')[0]
        if family not in families:
            families[family] = []
        families[family].append(r)

    for family, results in sorted(families.items()):
        n_total = len(results)
        n_qual = len([r for r in results if r['sharpe'] > 0
                      and r['frac_pos'] >= 0.50 and r['trades'] > 20])
        n_pos = len([r for r in results if r['sharpe'] > 0])
        avg_sh = np.mean([r['sharpe'] for r in results])
        best = max(results, key=lambda x: x['avg_sh'])
        print(f"  {family:<14}: {n_qual}/{n_total} qual, {n_pos}/{n_total} pos | "
              f"avg={avg_sh:+.2f} | best: {best['name']} {best['asset']} "
              f"Sh={best['sharpe']:.2f} F>0={best['frac_pos']:.0%}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
