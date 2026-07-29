"""Round 7: Genuinely different strategy mechanisms.

Previous rounds found that ALL qualifying strategies detect the same phenomenon:
rare market overreactions → buy the aftermath. This round tests structurally
different mechanisms to find truly independent signals:

1. Donchian Breakout + Vol Filter (A5) — buy above N-bar high
2. MTF Trend Filter (A1) — daily trend + hourly pullback entry
3. Volume Drought → Breakout (from SKILL.md §Volume Anomaly) — quiet → expansion
4. Consecutive Green Bars Momentum — sustained buying pressure
5. Session Open Breakout — Asian range breakout during EU session
6. Daily MACD Trend Buy — daily MACD histogram positive + 1h bar bullish
7. ADX Trend Confirmation — ADX > threshold + DI+ > DI- → buy
8. Bollinger Band Upper Breakout — price breaks above upper BB → momentum
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import (
    rsi, bollinger_bands, adx, atr, ema, sma, macd
)

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
# 1. DONCHIAN BREAKOUT + VOLUME FILTER (A5)
# Buy above N-bar high if volume confirms
# ═══════════════════════════════════════════════════════════════════

def donchian_breakout(df, entry_period=480, vol_mult=1.5, hold_bars=24,
                      cooldown=12):
    """Buy when close > highest high of entry_period bars AND vol > mult×avg."""
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
# 2. MTF TREND FILTER (A1)
# Daily MACD bullish + RSI oversold on 1H → buy pullback
# ═══════════════════════════════════════════════════════════════════

def mtf_trend_pullback(df, daily_fast=12, daily_slow=26, daily_signal=9,
                       rsi_period=14, rsi_thresh=35, hold_bars=8,
                       cooldown=3):
    """Buy when daily MACD histogram > 0 AND 1H RSI < threshold (pullback)."""
    close = df['close']

    # Resample to daily for MACD
    daily = df.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])

    _, _, daily_hist = macd(daily['close'], daily_fast, daily_slow, daily_signal)
    # Forward-fill daily MACD to hourly, shifted by 1 to prevent look-ahead
    daily_hist_hourly = daily_hist.reindex(df.index, method='ffill').shift(1)
    daily_hist_vals = daily_hist_hourly.values

    rsi_vals = rsi(close, rsi_period).values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0
    warmup = max(daily_slow * 24, rsi_period) + 48  # Enough for daily MACD warmup

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

        # Daily trend bullish AND 1H RSI showing pullback
        if daily_hist_vals[i] > 0 and rsi_vals[i] < rsi_thresh:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 3. VOLUME DROUGHT → BREAKOUT
# Low volume for N bars, then volume spike + bullish → buy
# ═══════════════════════════════════════════════════════════════════

def vol_drought_breakout(df, drought_bars=5, drought_mult=0.5,
                         spike_mult=1.5, hold_bars=8, cooldown=2):
    """Buy after sustained low volume followed by a bullish volume spike."""
    close = df['close'].values
    open_ = df['open'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(20 + drought_bars + 1, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue

        # Check for drought: N consecutive bars with vol < drought_mult × avg
        drought = True
        for j in range(1, drought_bars + 1):
            if vol[i - j] >= drought_mult * vol_avg[i - j]:
                drought = False
                break

        if not drought:
            continue

        # Current bar: volume spike + bullish
        if vol[i] > spike_mult * vol_avg[i] and close[i] > open_[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 4. CONSECUTIVE GREEN BARS MOMENTUM
# N consecutive bullish bars with cumulative gain > threshold → buy
# ═══════════════════════════════════════════════════════════════════

def consec_green_momentum(df, min_green=4, min_gain_pct=3.0, hold_bars=8,
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


# ═══════════════════════════════════════════════════════════════════
# 5. ASIAN RANGE BREAKOUT (session-based)
# Asian session (00:00-08:00 UTC) high → buy breakout during EU session
# ═══════════════════════════════════════════════════════════════════

def asian_range_breakout(df, hold_bars=8, cooldown=2):
    """Track Asian session high/low, buy if EU session breaks above."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    hours = df.index.hour

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    asian_high = 0
    asian_low = 1e18
    tracking_asian = False
    asian_set = False

    for i in range(1, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        h = hours[i]

        # During Asian session (00:00-08:00), track high/low
        if h >= 0 and h < 8:
            if h == 0 and (i == 0 or hours[i - 1] != 0):
                # New Asian session
                asian_high = high[i]
                asian_low = low[i]
                asian_set = False
            else:
                asian_high = max(asian_high, high[i])
                asian_low = min(asian_low, low[i])

            if h == 7:
                asian_set = True

        # During EU session (08:00-16:00), trade breakouts
        elif h >= 8 and h < 16 and asian_set:
            if close[i] > asian_high:
                sig[i] = 1
                hold_remaining = hold_bars - 1
                cooldown_remaining = cooldown
                asian_set = False  # Only one trade per session

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 6. DAILY MACD TREND BUY
# Daily MACD histogram positive + 1H bar bullish → buy
# ═══════════════════════════════════════════════════════════════════

def daily_macd_trend(df, hold_bars=24, cooldown=12):
    """Buy when daily MACD histogram turns positive (cross above 0)."""
    close = df['close']

    daily = df.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])

    _, _, daily_hist = macd(daily['close'])
    # Forward-fill to hourly, shift by 1
    daily_hist_hourly = daily_hist.reindex(df.index, method='ffill').shift(1)
    daily_hist_prev = daily_hist.shift(1).reindex(df.index, method='ffill').shift(1)
    daily_hist_vals = daily_hist_hourly.values
    daily_hist_prev_vals = daily_hist_prev.values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(700, n):  # Warmup for daily MACD
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(daily_hist_vals[i]) or np.isnan(daily_hist_prev_vals[i]):
            continue

        # MACD histogram crosses from negative to positive
        if daily_hist_vals[i] > 0 and daily_hist_prev_vals[i] <= 0:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 7. ADX TREND CONFIRMATION
# ADX > threshold + DI+ > DI- + 1H bullish bar → buy
# ═══════════════════════════════════════════════════════════════════

def adx_trend_buy(df, adx_period=14, adx_thresh=25, hold_bars=12,
                  cooldown=6):
    """Buy when ADX shows strong uptrend (ADX > thresh AND DI+ > DI-)."""
    adx_df = adx(df, adx_period)
    adx_vals = adx_df['adx'].values
    di_plus = adx_df['di_plus'].values
    di_minus = adx_df['di_minus'].values
    close = df['close'].values
    open_ = df['open'].values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(adx_period * 3, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(adx_vals[i]) or np.isnan(di_plus[i]) or np.isnan(di_minus[i]):
            continue

        if (adx_vals[i] > adx_thresh and
                di_plus[i] > di_minus[i] and
                close[i] > open_[i]):
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# 8. BOLLINGER BAND UPPER BREAKOUT
# Price closes above upper BB → momentum continuation buy
# ═══════════════════════════════════════════════════════════════════

def bb_upper_breakout(df, bb_period=20, bb_std=2.0, hold_bars=8,
                      cooldown=2, vol_mult=1.5):
    """Buy when price closes above upper BB with volume confirmation."""
    upper, _, _ = bollinger_bands(df['close'], bb_period, bb_std)
    upper_vals = upper.values
    close = df['close'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown_remaining = 0

    for i in range(bb_period + 1, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if np.isnan(upper_vals[i]) or np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue

        if close[i] > upper_vals[i] and vol[i] > vol_mult * vol_avg[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cooldown_remaining = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# MAIN
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
    # 1. DONCHIAN BREAKOUT + VOLUME
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("1. DONCHIAN BREAKOUT + VOLUME (buy above N-bar high)")
    print(f"{'='*120}")

    don_params = [
        ("p480_v1.5_h24_c12", dict(entry_period=480, vol_mult=1.5, hold_bars=24, cooldown=12)),
        ("p480_v1.5_h48_c24", dict(entry_period=480, vol_mult=1.5, hold_bars=48, cooldown=24)),
        ("p720_v1.5_h24_c12", dict(entry_period=720, vol_mult=1.5, hold_bars=24, cooldown=12)),
        ("p720_v1.5_h48_c24", dict(entry_period=720, vol_mult=1.5, hold_bars=48, cooldown=24)),
        ("p240_v1.5_h24_c12", dict(entry_period=240, vol_mult=1.5, hold_bars=24, cooldown=12)),
        ("p480_v2.0_h24_c12", dict(entry_period=480, vol_mult=2.0, hold_bars=24, cooldown=12)),
        ("p480_v1.0_h24_c12", dict(entry_period=480, vol_mult=1.0, hold_bars=24, cooldown=12)),
    ]
    for label, params in don_params:
        test_strategy(f"Donchian_{label}", lambda df, p=params: donchian_breakout(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 2. MTF TREND + PULLBACK
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("2. MTF TREND FILTER (daily MACD bull + 1H RSI pullback)")
    print(f"{'='*120}")

    mtf_params = [
        ("RSI35_h8",   dict(rsi_thresh=35, hold_bars=8)),
        ("RSI35_h12",  dict(rsi_thresh=35, hold_bars=12)),
        ("RSI30_h8",   dict(rsi_thresh=30, hold_bars=8)),
        ("RSI30_h12",  dict(rsi_thresh=30, hold_bars=12)),
        ("RSI40_h8",   dict(rsi_thresh=40, hold_bars=8)),
        ("RSI40_h12",  dict(rsi_thresh=40, hold_bars=12)),
        ("RSI25_h8",   dict(rsi_thresh=25, hold_bars=8)),
    ]
    for label, params in mtf_params:
        test_strategy(f"MTFPull_{label}", lambda df, p=params: mtf_trend_pullback(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 3. VOLUME DROUGHT → BREAKOUT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("3. VOLUME DROUGHT → BREAKOUT (low vol streak → spike)")
    print(f"{'='*120}")

    drought_params = [
        ("d5_s1.5_h8",   dict(drought_bars=5, spike_mult=1.5, hold_bars=8)),
        ("d5_s1.5_h12",  dict(drought_bars=5, spike_mult=1.5, hold_bars=12)),
        ("d5_s2.0_h8",   dict(drought_bars=5, spike_mult=2.0, hold_bars=8)),
        ("d5_s2.0_h12",  dict(drought_bars=5, spike_mult=2.0, hold_bars=12)),
        ("d3_s1.5_h8",   dict(drought_bars=3, spike_mult=1.5, hold_bars=8)),
        ("d3_s2.0_h8",   dict(drought_bars=3, spike_mult=2.0, hold_bars=8)),
        ("d8_s1.5_h12",  dict(drought_bars=8, spike_mult=1.5, hold_bars=12)),
        ("d5_s2.5_h12",  dict(drought_bars=5, spike_mult=2.5, hold_bars=12)),
    ]
    for label, params in drought_params:
        test_strategy(f"Drought_{label}", lambda df, p=params: vol_drought_breakout(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 4. CONSECUTIVE GREEN BARS MOMENTUM
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("4. CONSECUTIVE GREEN BARS MOMENTUM (sustained buying)")
    print(f"{'='*120}")

    green_params = [
        ("g4_3p_h8",   dict(min_green=4, min_gain_pct=3.0, hold_bars=8)),
        ("g4_3p_h12",  dict(min_green=4, min_gain_pct=3.0, hold_bars=12)),
        ("g4_5p_h12",  dict(min_green=4, min_gain_pct=5.0, hold_bars=12)),
        ("g5_5p_h12",  dict(min_green=5, min_gain_pct=5.0, hold_bars=12)),
        ("g5_5p_h16",  dict(min_green=5, min_gain_pct=5.0, hold_bars=16)),
        ("g3_3p_h8",   dict(min_green=3, min_gain_pct=3.0, hold_bars=8)),
        ("g4_4p_h12",  dict(min_green=4, min_gain_pct=4.0, hold_bars=12)),
    ]
    for label, params in green_params:
        test_strategy(f"GreenMom_{label}", lambda df, p=params: consec_green_momentum(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 5. ASIAN RANGE BREAKOUT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("5. ASIAN RANGE BREAKOUT (overnight range → EU breakout)")
    print(f"{'='*120}")

    arb_params = [
        ("h6",  dict(hold_bars=6)),
        ("h8",  dict(hold_bars=8)),
        ("h12", dict(hold_bars=12)),
        ("h4",  dict(hold_bars=4)),
    ]
    for label, params in arb_params:
        test_strategy(f"AsianBrk_{label}", lambda df, p=params: asian_range_breakout(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 6. DAILY MACD CROSS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("6. DAILY MACD CROSS (histogram turns positive)")
    print(f"{'='*120}")

    macd_params = [
        ("h24_c12", dict(hold_bars=24, cooldown=12)),
        ("h48_c24", dict(hold_bars=48, cooldown=24)),
        ("h72_c24", dict(hold_bars=72, cooldown=24)),
        ("h96_c48", dict(hold_bars=96, cooldown=48)),
    ]
    for label, params in macd_params:
        test_strategy(f"MACDCross_{label}", lambda df, p=params: daily_macd_trend(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 7. ADX TREND BUY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("7. ADX TREND BUY (ADX > thresh + DI+ dominant)")
    print(f"{'='*120}")

    adx_params = [
        ("a25_h12_c6",  dict(adx_thresh=25, hold_bars=12, cooldown=6)),
        ("a25_h24_c12", dict(adx_thresh=25, hold_bars=24, cooldown=12)),
        ("a30_h12_c6",  dict(adx_thresh=30, hold_bars=12, cooldown=6)),
        ("a30_h24_c12", dict(adx_thresh=30, hold_bars=24, cooldown=12)),
        ("a35_h12_c6",  dict(adx_thresh=35, hold_bars=12, cooldown=6)),
        ("a20_h12_c6",  dict(adx_thresh=20, hold_bars=12, cooldown=6)),
    ]
    for label, params in adx_params:
        test_strategy(f"ADXTrend_{label}", lambda df, p=params: adx_trend_buy(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # 8. BB UPPER BREAKOUT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("8. BB UPPER BREAKOUT (price above upper BB + volume)")
    print(f"{'='*120}")

    bb_params = [
        ("BB2.0_v1.5_h8",  dict(bb_std=2.0, vol_mult=1.5, hold_bars=8)),
        ("BB2.0_v1.5_h12", dict(bb_std=2.0, vol_mult=1.5, hold_bars=12)),
        ("BB2.5_v1.5_h8",  dict(bb_std=2.5, vol_mult=1.5, hold_bars=8)),
        ("BB2.5_v1.5_h12", dict(bb_std=2.5, vol_mult=1.5, hold_bars=12)),
        ("BB2.0_v2.0_h8",  dict(bb_std=2.0, vol_mult=2.0, hold_bars=8)),
        ("BB2.0_v2.0_h12", dict(bb_std=2.0, vol_mult=2.0, hold_bars=12)),
        ("BB2.5_v2.0_h12", dict(bb_std=2.5, vol_mult=2.0, hold_bars=12)),
    ]
    for label, params in bb_params:
        test_strategy(f"BBBreak_{label}", lambda df, p=params: bb_upper_breakout(df, **p))

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("ROUND 7 SUMMARY — TOP RESULTS (Sharpe > 0, frac>0 >= 50%, trades > 20)")
    print(f"{'='*120}")

    qualifying = [r for r in all_results
                  if r['sharpe'] > 0 and r['frac_pos'] >= 0.50 and r['trades'] > 20]
    qualifying.sort(key=lambda x: x['avg_sh'], reverse=True)

    print(f"\n{'Name':<44} {'Asset':<5} {'Sh':>5} {'Tr':>5} {'WR':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'F>0':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 110)

    for r in qualifying[:30]:
        print(f"{r['name']:<44} {r['asset']:<5} {r['sharpe']:>5.2f} "
              f"{r['trades']:>5} {r['wr']:>5.1f} {r['ret']:>7.1f} "
              f"{r['dd']:>6.1f} {r['frac_pos']:>5.0%} "
              f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Cross-asset
    print(f"\n{'='*120}")
    print("CROSS-ASSET (positive on 2+ assets)")
    print(f"{'='*120}")

    strat_names = set(r['name'] for r in all_results)
    for name in sorted(strat_names):
        rows = [r for r in all_results if r['name'] == name]
        positive = [r for r in rows if r['sharpe'] > 0]
        if len(positive) >= 2:
            assets_str = " | ".join(
                f"{r['asset']}:{r['sharpe']:+.2f}(F{r['frac_pos']:.0%})"
                for r in rows)
            stable = [r for r in rows if r['sharpe'] > 0 and r['frac_pos'] >= 0.50]
            mark = "***" if len(stable) >= 2 else ""
            print(f"  {mark}{name:<42} {assets_str}")

    # Family scorecard
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
