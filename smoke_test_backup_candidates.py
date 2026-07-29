"""Smoke test additional event-driven strategy ideas as backups."""

import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, ".")

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import rsi, bollinger_bands, atr, sma

print("Loading BTC data...")
df_30 = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample="30min")
df_1h = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample="1h")

cfg_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
cfg_0 = BacktestConfig(fee_rate=0.0, slippage_pct=0.0)


def test_signals(name, signals, df, params=""):
    for label, cfg in [("10bps", cfg_10), ("0bps", cfg_0)]:
        res = run_backtest(df, signals, cfg)
        m = compute_metrics(res)
        if label == "10bps" or m["sharpe_ratio"] > 0:
            print(f"  {name} {params} | {label} | Sharpe={m['sharpe_ratio']:.2f} Trades={m['total_trades']} WR={m['win_rate_pct']:.0f}% DD={m['max_drawdown_pct']:.1f}% Ret={m['total_return_pct']:.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. EXTREME RSI REVERSAL - only trade very extreme RSI values
#    This is very selective → few trades → low fee drag
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("1. EXTREME RSI REVERSAL")
print("="*60)

for tf_label, df in [("30min", df_30), ("1h", df_1h)]:
    print(f"\n--- {tf_label} ---")
    close = df["close"]
    for rsi_p in [14, 21]:
        r = rsi(close, rsi_p)
        for extreme in [10, 15, 20]:
            for hold in [4, 8, 16, 24]:
                # Buy when RSI < extreme, sell when RSI > (100-extreme)
                # Hold for N bars then go flat
                n = len(df)
                sig = np.zeros(n, dtype=np.int8)
                hold_remaining = 0
                for i in range(n):
                    if hold_remaining > 0:
                        sig[i] = sig[i-1]
                        hold_remaining -= 1
                        continue
                    rv = r.iloc[i]
                    if np.isnan(rv):
                        continue
                    if rv < extreme:
                        sig[i] = 1
                        hold_remaining = hold - 1
                    elif rv > (100 - extreme):
                        sig[i] = -1
                        hold_remaining = hold - 1
                signals = pd.Series(sig, index=df.index, dtype="int8")
                test_signals("ExtremeRSI", signals, df, f"p={rsi_p} ext={extreme} hold={hold}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CONSECUTIVE BAR STREAK REVERSAL
#    After N consecutive same-direction bars, fade
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("2. CONSECUTIVE BAR STREAK REVERSAL")
print("="*60)

for tf_label, df in [("30min", df_30), ("1h", df_1h)]:
    print(f"\n--- {tf_label} ---")
    close = df["close"].values
    bar_dir = np.sign(np.diff(close, prepend=close[0]))
    n = len(df)

    for streak_min in [3, 4, 5, 6]:
        for hold in [2, 4, 8]:
            sig = np.zeros(n, dtype=np.int8)
            streak = 0
            streak_dir = 0
            hold_remaining = 0

            for i in range(1, n):
                if hold_remaining > 0:
                    sig[i] = sig[i-1]
                    hold_remaining -= 1
                    continue

                d = bar_dir[i]
                if d == streak_dir:
                    streak += 1
                else:
                    streak = 1
                    streak_dir = d

                if streak >= streak_min and d != 0:
                    sig[i] = -int(d)  # Fade the streak
                    hold_remaining = hold - 1
                    streak = 0

            signals = pd.Series(sig, index=df.index, dtype="int8")
            test_signals("Streak", signals, df, f"min={streak_min} hold={hold}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ATR EXPANSION MOMENTUM
#    When current bar range > N * ATR, trade in bar direction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("3. ATR EXPANSION MOMENTUM")
print("="*60)

for tf_label, df in [("30min", df_30), ("1h", df_1h)]:
    print(f"\n--- {tf_label} ---")
    for atr_p in [14, 24]:
        atr_val = atr(df, atr_p)
        bar_range = df["high"] - df["low"]
        bar_dir_v = np.sign(df["close"].values - df["open"].values)
        for mult in [2.0, 2.5, 3.0]:
            for hold in [2, 4, 8]:
                expansion = bar_range > mult * atr_val
                n = len(df)
                sig = np.zeros(n, dtype=np.int8)
                hold_remaining = 0

                for i in range(n):
                    if hold_remaining > 0:
                        sig[i] = sig[i-1]
                        hold_remaining -= 1
                        continue
                    if expansion.iloc[i] and bar_dir_v[i] != 0:
                        sig[i] = int(bar_dir_v[i])
                        hold_remaining = hold - 1

                signals = pd.Series(sig, index=df.index, dtype="int8")
                test_signals("ATRExpand", signals, df, f"atr={atr_p} mult={mult} hold={hold}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. INSIDE BAR BREAKOUT
#    When bar is fully inside previous bar (lower vol), enter on breakout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("4. INSIDE BAR BREAKOUT")
print("="*60)

for tf_label, df in [("30min", df_30), ("1h", df_1h)]:
    print(f"\n--- {tf_label} ---")
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    for min_inside in [1, 2, 3]:  # Require N consecutive inside bars
        for hold in [4, 8, 12]:
            sig = np.zeros(n, dtype=np.int8)
            inside_count = 0
            ref_high = 0.0
            ref_low = 0.0
            hold_remaining = 0

            for i in range(1, n):
                if hold_remaining > 0:
                    sig[i] = sig[i-1]
                    hold_remaining -= 1
                    continue

                if inside_count == 0:
                    ref_high = high[i-1]
                    ref_low = low[i-1]

                if high[i] <= ref_high and low[i] >= ref_low:
                    inside_count += 1
                else:
                    if inside_count >= min_inside:
                        # Breakout detected
                        if close[i] > ref_high:
                            sig[i] = 1
                            hold_remaining = hold - 1
                        elif close[i] < ref_low:
                            sig[i] = -1
                            hold_remaining = hold - 1
                    inside_count = 0

            signals = pd.Series(sig, index=df.index, dtype="int8")
            test_signals("InsideBar", signals, df, f"min={min_inside} hold={hold}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. BB WIDTH PERCENTILE + TREND (refined Vol Compression variant)
#    Use BB width percentile instead of RV percentile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("5. BB WIDTH PERCENTILE + TREND")
print("="*60)

for tf_label, df in [("30min", df_30), ("1h", df_1h)]:
    print(f"\n--- {tf_label} ---")
    close = df["close"]
    for bb_p in [20, 30]:
        for bb_s in [2.0, 2.5]:
            upper, mid, lower = bollinger_bands(close, bb_p, bb_s)
            bb_width = (upper - lower) / mid
            for lk in [480, 720, 1440]:
                width_pctl = bb_width.rolling(lk).rank(pct=True) * 100
                trend = sma(close, 200)
                for comp_thr in [10.0, 15.0]:
                    for exp_thr in [25.0, 35.0]:
                        n = len(df)
                        sig = np.zeros(n, dtype=np.int8)
                        in_compression = False
                        comp_high = 0.0
                        comp_low = 0.0
                        current_signal = 0

                        for i in range(n):
                            pv = width_pctl.iloc[i]
                            if np.isnan(pv):
                                continue

                            if not in_compression:
                                if pv < comp_thr:
                                    in_compression = True
                                    start = max(0, i-3)
                                    comp_high = df["high"].values[start:i+1].max()
                                    comp_low = df["low"].values[start:i+1].min()
                                sig[i] = current_signal
                            else:
                                h = df["high"].values[i]
                                l = df["low"].values[i]
                                if h > comp_high:
                                    comp_high = h
                                if l < comp_low:
                                    comp_low = l

                                if pv > exp_thr:
                                    c = close.values[i]
                                    if c > comp_high:
                                        current_signal = 1
                                    elif c < comp_low:
                                        current_signal = -1
                                    else:
                                        tv = trend.iloc[i]
                                        if not np.isnan(tv):
                                            current_signal = 1 if c > tv else -1
                                    in_compression = False
                                    sig[i] = current_signal
                                else:
                                    sig[i] = current_signal

                        signals = pd.Series(sig, index=df.index, dtype="int8")
                        test_signals("BBWidth", signals, df, f"bb={bb_p}/{bb_s} lk={lk} comp={comp_thr} exp={exp_thr}")

print("\n" + "="*60)
print("BACKUP CANDIDATE SMOKE TEST COMPLETE")
print("="*60)
