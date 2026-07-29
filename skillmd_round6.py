"""Round 6: Strategy catalog test from SKILL.md suggestions.

Tests 8 strategy families from the Research-Driven Strategy Catalog,
all adapted to buy-only + fixed-hold architecture at 1h.

Strategies tested:
1. Regime-Filtered RSI MR (C8/A6) - RSI oversold + ADX ranging filter
2. Triple MR Confirmation (C13) - RSI + BB + Volume confluence
3. Volume Exhaustion Fade (C9) - Counter-trend after vol exhaustion
4. Volatility Squeeze Breakout (A2) - BB Width percentile squeeze → breakout
5. Liquidation Cascade Bounce (B4) - Big drop + vol spike → buy bounce
6. Opening Range Volume Momentum (C11) - US open vol anomaly
7. RSI + BB Confluence (simplified) - RSI oversold + below BB
8. Multi-Lookback Momentum (A3) - Weighted composite returns
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import rsi, bollinger_bands, adx, atr

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CFG_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)


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
    """Full-sample + rolling windows evaluation."""
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
# STRATEGY 1: REGIME-FILTERED RSI MEAN REVERSION (C8/A6)
# RSI < threshold + ADX < threshold → buy, fixed hold
# ═══════════════════════════════════════════════════════════════════

def regime_rsi_mr(df, rsi_period=14, rsi_thresh=25, adx_period=14,
                  adx_thresh=25, hold_bars=6, cooldown=2):
    rsi_vals = rsi(df['close'], rsi_period).values
    adx_df = adx(df, adx_period)
    adx_vals = adx_df['adx'].values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    warmup = max(rsi_period, adx_period) * 3

    for i in range(warmup, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(rsi_vals[i]) or np.isnan(adx_vals[i]):
            continue
        if rsi_vals[i] < rsi_thresh and adx_vals[i] < adx_thresh:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: TRIPLE MR CONFIRMATION (C13)
# RSI < 20 AND below BB(20,2.5) AND vol > 2x avg → buy
# ═══════════════════════════════════════════════════════════════════

def triple_mr(df, rsi_period=14, rsi_thresh=20, bb_period=20, bb_std=2.5,
              vol_mult=2.0, hold_bars=5, cooldown=2):
    rsi_vals = rsi(df['close'], rsi_period).values
    _, _, lower = bollinger_bands(df['close'], bb_period, bb_std)
    lower_vals = lower.values
    close = df['close'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

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
# STRATEGY 3: VOLUME EXHAUSTION FADE (C9)
# 3+ bars with vol > 2x avg going DOWN → buy counter-trend
# ═══════════════════════════════════════════════════════════════════

def vol_exhaustion_fade(df, consec_bars=3, vol_mult=2.0, hold_bars=5,
                        cooldown=2):
    close = df['close'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(20 + consec_bars, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # Check for consecutive high-volume bars
        high_vol_count = 0
        for j in range(consec_bars):
            idx = i - j
            if np.isnan(vol_avg[idx]) or vol_avg[idx] == 0:
                break
            if vol[idx] > vol_mult * vol_avg[idx]:
                high_vol_count += 1
            else:
                break

        if high_vol_count >= consec_bars:
            # Was it a selloff? (price declined over the consecutive bars)
            if close[i] < close[i - consec_bars]:
                sig[i] = 1
                hold_remaining = hold_bars - 1
                cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 4: VOLATILITY SQUEEZE BREAKOUT (A2)
# BB Width at low percentile → squeeze. Enter bullish on expansion.
# ═══════════════════════════════════════════════════════════════════

def vol_squeeze_breakout(df, bb_period=20, bb_std=2.0, width_lookback=100,
                         squeeze_pctl=20, expansion_pctl=50, hold_bars=8,
                         cooldown=2):
    upper, middle, lower = bollinger_bands(df['close'], bb_period, bb_std)
    bb_width = ((upper - lower) / middle).values
    close = df['close'].values
    open_ = df['open'].values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    in_squeeze = False

    for i in range(width_lookback + bb_period, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        window = bb_width[i - width_lookback:i]
        if np.any(np.isnan(window)):
            continue
        pctl = np.sum(window < bb_width[i]) / width_lookback * 100

        if pctl <= squeeze_pctl:
            in_squeeze = True

        if in_squeeze and pctl >= expansion_pctl:
            if close[i] > open_[i]:  # Bullish bar → buy
                sig[i] = 1
                hold_remaining = hold_bars - 1
                cooldown_remaining = cooldown
                in_squeeze = False

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 5: LIQUIDATION CASCADE BOUNCE (B4)
# Big cumulative drop + volume spike → buy bounce after stabilization
# ═══════════════════════════════════════════════════════════════════

def cascade_bounce(df, cum_move_pct=-5.0, lookback=10, vol_z_window=48,
                   vol_z_thresh=2.0, stabilize_bars=2, hold_bars=12,
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
# STRATEGY 6: OPENING RANGE VOLUME MOMENTUM (C11)
# US open vol > threshold × avg → buy if bullish
# ═══════════════════════════════════════════════════════════════════

def opening_range_momentum(df, vol_pct_threshold=12.0, hold_bars=6,
                           cooldown=2):
    close = df['close'].values
    open_ = df['open'].values
    vol = df['volume'].values
    hours = df.index.hour

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(24, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # Only trigger at US open (14:00 UTC)
        if hours[i] != 14:
            continue

        vol_24h = np.sum(vol[max(0, i - 24):i])
        if vol_24h == 0:
            continue
        vol_pct = vol[i] / vol_24h * 100

        if vol_pct > vol_pct_threshold and close[i] > open_[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 7: RSI + BB CONFLUENCE (simplified mean reversion)
# RSI < threshold AND below lower BB → buy, hold N bars
# ═══════════════════════════════════════════════════════════════════

def rsi_bb_confluence(df, rsi_period=14, rsi_thresh=25, bb_period=20,
                      bb_std=2.0, hold_bars=6, cooldown=2):
    rsi_vals = rsi(df['close'], rsi_period).values
    _, _, lower = bollinger_bands(df['close'], bb_period, bb_std)
    lower_vals = lower.values
    close = df['close'].values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    warmup = max(rsi_period, bb_period) + 1

    for i in range(warmup, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(rsi_vals[i]) or np.isnan(lower_vals[i]):
            continue
        if rsi_vals[i] < rsi_thresh and close[i] < lower_vals[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 8: MULTI-LOOKBACK MOMENTUM (A3)
# Weighted composite returns > threshold → buy
# ═══════════════════════════════════════════════════════════════════

def multi_momentum(df, short_bars=168, med_bars=720, long_bars=2160,
                   w_short=0.2, w_med=0.5, w_long=0.3, threshold=5.0,
                   hold_bars=12, cooldown=3):
    close = df['close'].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(long_bars, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        ret_short = (close[i] / close[i - short_bars] - 1) * 100
        ret_med = (close[i] / close[i - med_bars] - 1) * 100
        ret_long = (close[i] / close[i - long_bars] - 1) * 100

        composite = w_short * ret_short + w_med * ret_med + w_long * ret_long

        if composite > threshold:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    # Load data
    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        print(f"{len(df)} bars")
        data[asset] = df

    all_results = []

    def test_strategy(name, signal_fn):
        """Test a strategy across all assets and print results."""
        row = f"  {name:<38}"
        for asset, df in data.items():
            try:
                r = evaluate(signal_fn, df)
                all_results.append({"name": name, "asset": asset, **r})
                row += f"  {asset}:{r['sharpe']:+.2f}({r['trades']}t,{r['frac_pos']:.0%}w)"
            except Exception as e:
                row += f"  {asset}:ERR({str(e)[:20]})"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # 1. REGIME-FILTERED RSI MEAN REVERSION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("1. REGIME-FILTERED RSI MEAN REVERSION (C8/A6)")
    print(f"   RSI < threshold + ADX < threshold → buy, fixed hold")
    print(f"{'='*120}")

    regime_rsi_params = [
        ("RSI25_ADX25_h6",  dict(rsi_thresh=25, adx_thresh=25, hold_bars=6, cooldown=2)),
        ("RSI25_ADX25_h8",  dict(rsi_thresh=25, adx_thresh=25, hold_bars=8, cooldown=2)),
        ("RSI25_ADX25_h12", dict(rsi_thresh=25, adx_thresh=25, hold_bars=12, cooldown=2)),
        ("RSI20_ADX25_h6",  dict(rsi_thresh=20, adx_thresh=25, hold_bars=6, cooldown=2)),
        ("RSI20_ADX25_h8",  dict(rsi_thresh=20, adx_thresh=25, hold_bars=8, cooldown=2)),
        ("RSI25_ADX20_h8",  dict(rsi_thresh=25, adx_thresh=20, hold_bars=8, cooldown=2)),
        ("RSI30_ADX25_h6",  dict(rsi_thresh=30, adx_thresh=25, hold_bars=6, cooldown=2)),
        ("RSI30_ADX30_h8",  dict(rsi_thresh=30, adx_thresh=30, hold_bars=8, cooldown=2)),
    ]

    for label, params in regime_rsi_params:
        test_strategy(f"RegRSI_{label}",
                      lambda df, p=params: regime_rsi_mr(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 2. TRIPLE MR CONFIRMATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("2. TRIPLE MR CONFIRMATION (C13)")
    print(f"   RSI < thresh AND below BB AND vol > mult × avg → buy")
    print(f"{'='*120}")

    triple_mr_params = [
        ("R20_BB2.5_V2_h5",  dict(rsi_thresh=20, bb_std=2.5, vol_mult=2.0, hold_bars=5)),
        ("R20_BB2.5_V2_h8",  dict(rsi_thresh=20, bb_std=2.5, vol_mult=2.0, hold_bars=8)),
        ("R20_BB2.0_V2_h5",  dict(rsi_thresh=20, bb_std=2.0, vol_mult=2.0, hold_bars=5)),
        ("R20_BB2.0_V2_h8",  dict(rsi_thresh=20, bb_std=2.0, vol_mult=2.0, hold_bars=8)),
        ("R25_BB2.5_V2_h5",  dict(rsi_thresh=25, bb_std=2.5, vol_mult=2.0, hold_bars=5)),
        ("R25_BB2.5_V1.5_h5", dict(rsi_thresh=25, bb_std=2.5, vol_mult=1.5, hold_bars=5)),
        ("R25_BB2.0_V1.5_h8", dict(rsi_thresh=25, bb_std=2.0, vol_mult=1.5, hold_bars=8)),
        ("R20_BB2.5_V1.5_h8", dict(rsi_thresh=20, bb_std=2.5, vol_mult=1.5, hold_bars=8)),
    ]

    for label, params in triple_mr_params:
        test_strategy(f"TripleMR_{label}",
                      lambda df, p=params: triple_mr(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 3. VOLUME EXHAUSTION FADE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("3. VOLUME EXHAUSTION FADE (C9)")
    print(f"   3+ bars vol > 2x avg + selloff → buy counter-trend")
    print(f"{'='*120}")

    vol_exh_params = [
        ("c3_v2_h5",   dict(consec_bars=3, vol_mult=2.0, hold_bars=5, cooldown=2)),
        ("c3_v2_h8",   dict(consec_bars=3, vol_mult=2.0, hold_bars=8, cooldown=2)),
        ("c3_v2_h12",  dict(consec_bars=3, vol_mult=2.0, hold_bars=12, cooldown=2)),
        ("c3_v1.5_h5", dict(consec_bars=3, vol_mult=1.5, hold_bars=5, cooldown=2)),
        ("c3_v1.5_h8", dict(consec_bars=3, vol_mult=1.5, hold_bars=8, cooldown=2)),
        ("c2_v2_h5",   dict(consec_bars=2, vol_mult=2.0, hold_bars=5, cooldown=2)),
        ("c2_v2_h8",   dict(consec_bars=2, vol_mult=2.0, hold_bars=8, cooldown=2)),
        ("c4_v2_h8",   dict(consec_bars=4, vol_mult=2.0, hold_bars=8, cooldown=2)),
    ]

    for label, params in vol_exh_params:
        test_strategy(f"VolExh_{label}",
                      lambda df, p=params: vol_exhaustion_fade(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 4. VOLATILITY SQUEEZE BREAKOUT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("4. VOLATILITY SQUEEZE BREAKOUT (A2)")
    print(f"   BB Width at low percentile → squeeze → enter bullish expansion")
    print(f"{'='*120}")

    squeeze_params = [
        ("sq20_ex50_h8",   dict(squeeze_pctl=20, expansion_pctl=50, hold_bars=8)),
        ("sq20_ex50_h12",  dict(squeeze_pctl=20, expansion_pctl=50, hold_bars=12)),
        ("sq15_ex50_h8",   dict(squeeze_pctl=15, expansion_pctl=50, hold_bars=8)),
        ("sq15_ex40_h8",   dict(squeeze_pctl=15, expansion_pctl=40, hold_bars=8)),
        ("sq20_ex60_h8",   dict(squeeze_pctl=20, expansion_pctl=60, hold_bars=8)),
        ("sq25_ex50_h12",  dict(squeeze_pctl=25, expansion_pctl=50, hold_bars=12)),
        ("sq10_ex50_h12",  dict(squeeze_pctl=10, expansion_pctl=50, hold_bars=12)),
    ]

    for label, params in squeeze_params:
        test_strategy(f"Squeeze_{label}",
                      lambda df, p=params: vol_squeeze_breakout(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 5. LIQUIDATION CASCADE BOUNCE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("5. LIQUIDATION CASCADE BOUNCE (B4)")
    print(f"   Big drop + vol z-score spike → buy after stabilization")
    print(f"{'='*120}")

    cascade_params = [
        ("d5_vz2_s2_h12",  dict(cum_move_pct=-5.0, vol_z_thresh=2.0, stabilize_bars=2, hold_bars=12)),
        ("d5_vz2_s2_h16",  dict(cum_move_pct=-5.0, vol_z_thresh=2.0, stabilize_bars=2, hold_bars=16)),
        ("d5_vz2_s3_h12",  dict(cum_move_pct=-5.0, vol_z_thresh=2.0, stabilize_bars=3, hold_bars=12)),
        ("d3_vz2_s2_h8",   dict(cum_move_pct=-3.0, vol_z_thresh=2.0, stabilize_bars=2, hold_bars=8)),
        ("d3_vz2_s2_h12",  dict(cum_move_pct=-3.0, vol_z_thresh=2.0, stabilize_bars=2, hold_bars=12)),
        ("d5_vz1.5_s2_h12", dict(cum_move_pct=-5.0, vol_z_thresh=1.5, stabilize_bars=2, hold_bars=12)),
        ("d8_vz2_s2_h16",  dict(cum_move_pct=-8.0, vol_z_thresh=2.0, stabilize_bars=2, hold_bars=16)),
        ("d8_vz2_s3_h24",  dict(cum_move_pct=-8.0, vol_z_thresh=2.0, stabilize_bars=3, hold_bars=24)),
    ]

    for label, params in cascade_params:
        test_strategy(f"Cascade_{label}",
                      lambda df, p=params: cascade_bounce(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 6. OPENING RANGE VOLUME MOMENTUM
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("6. OPENING RANGE VOLUME MOMENTUM (C11)")
    print(f"   US open (14:00 UTC) vol > threshold% of 24h → buy if bullish")
    print(f"{'='*120}")

    or_params = [
        ("v12_h6",  dict(vol_pct_threshold=12.0, hold_bars=6)),
        ("v12_h8",  dict(vol_pct_threshold=12.0, hold_bars=8)),
        ("v10_h6",  dict(vol_pct_threshold=10.0, hold_bars=6)),
        ("v10_h8",  dict(vol_pct_threshold=10.0, hold_bars=8)),
        ("v8_h6",   dict(vol_pct_threshold=8.0, hold_bars=6)),
        ("v15_h8",  dict(vol_pct_threshold=15.0, hold_bars=8)),
    ]

    for label, params in or_params:
        test_strategy(f"OpenRange_{label}",
                      lambda df, p=params: opening_range_momentum(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 7. RSI + BB CONFLUENCE (simplified)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("7. RSI + BB CONFLUENCE (simplified mean reversion)")
    print(f"   RSI < threshold AND below lower BB → buy, fixed hold")
    print(f"{'='*120}")

    rsi_bb_params = [
        ("R25_BB2.0_h6",  dict(rsi_thresh=25, bb_std=2.0, hold_bars=6)),
        ("R25_BB2.0_h8",  dict(rsi_thresh=25, bb_std=2.0, hold_bars=8)),
        ("R25_BB2.0_h12", dict(rsi_thresh=25, bb_std=2.0, hold_bars=12)),
        ("R25_BB2.5_h8",  dict(rsi_thresh=25, bb_std=2.5, hold_bars=8)),
        ("R20_BB2.0_h6",  dict(rsi_thresh=20, bb_std=2.0, hold_bars=6)),
        ("R20_BB2.0_h8",  dict(rsi_thresh=20, bb_std=2.0, hold_bars=8)),
        ("R30_BB2.0_h8",  dict(rsi_thresh=30, bb_std=2.0, hold_bars=8)),
        ("R30_BB2.5_h8",  dict(rsi_thresh=30, bb_std=2.5, hold_bars=8)),
    ]

    for label, params in rsi_bb_params:
        test_strategy(f"RSIBB_{label}",
                      lambda df, p=params: rsi_bb_confluence(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 8. MULTI-LOOKBACK MOMENTUM
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("8. MULTI-LOOKBACK MOMENTUM (A3)")
    print(f"   Weighted 7d/30d/90d returns > threshold → buy")
    print(f"{'='*120}")

    mom_params = [
        ("t5_h12",   dict(threshold=5.0, hold_bars=12)),
        ("t5_h24",   dict(threshold=5.0, hold_bars=24)),
        ("t10_h12",  dict(threshold=10.0, hold_bars=12)),
        ("t10_h24",  dict(threshold=10.0, hold_bars=24)),
        ("t3_h8",    dict(threshold=3.0, hold_bars=8)),
        ("t15_h24",  dict(threshold=15.0, hold_bars=24)),
    ]

    for label, params in mom_params:
        test_strategy(f"Momentum_{label}",
                      lambda df, p=params: multi_momentum(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("ROUND 6 SUMMARY — TOP RESULTS")
    print(f"{'='*120}")

    # Filter for qualifying results
    qualifying = [r for r in all_results
                  if r['sharpe'] > 0 and r['frac_pos'] >= 0.50 and r['trades'] > 20]

    if qualifying:
        # Sort by avg_sh (rolling window average Sharpe)
        qualifying.sort(key=lambda x: x['avg_sh'], reverse=True)

        print(f"\n{'Name':<40} {'Asset':<5} {'Sh':>5} {'Tr':>5} {'WR':>5} "
              f"{'Ret%':>7} {'DD%':>6} {'F>0':>5} {'AvgSh':>6} {'MedSh':>6}")
        print("-" * 100)

        for r in qualifying[:30]:  # top 30
            print(f"{r['name']:<40} {r['asset']:<5} {r['sharpe']:>5.2f} "
                  f"{r['trades']:>5} {r['wr']:>5.1f} {r['ret']:>7.1f} "
                  f"{r['dd']:>6.1f} {r['frac_pos']:>5.0%} "
                  f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Strategy family summary
    print(f"\n{'='*120}")
    print("STRATEGY FAMILY SCORECARD")
    print(f"{'='*120}")

    families = {}
    for r in all_results:
        family = r['name'].split('_')[0]
        if family not in families:
            families[family] = []
        families[family].append(r)

    for family, results in families.items():
        n_total = len(results)
        n_qual = len([r for r in results if r['sharpe'] > 0
                      and r['frac_pos'] >= 0.50 and r['trades'] > 20])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        best = max(results, key=lambda x: x['avg_sh'])
        print(f"  {family:<12}: {n_qual}/{n_total} qualifying | "
              f"avg Sharpe={avg_sharpe:+.2f} | "
              f"best: {best['name']} {best['asset']} "
              f"Sh={best['sharpe']:.2f} F>0={best['frac_pos']:.0%}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
