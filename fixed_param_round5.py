"""Round 5: Multi-bar pattern and price-action anomaly strategies at 1h.

These strategies detect STRUCTURAL market events, not indicator thresholds:
1. Consecutive Down Bars (dip buyer): N bars in a row closing lower
2. Multi-bar Selloff: cumulative return < -X% over N bars
3. Hammer Reversal: large range bar with close in upper portion after selloff
4. Extreme Range Bar: widest range in N bars + bullish close
5. Volume Climax Reversal: highest volume in N bars + bullish reversal
6. Vol Spike variants: tweaked params for ETH/SOL robustness
"""

import sys
import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import atr, sma

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CFG_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
CFG_5 = BacktestConfig(fee_rate=0.00025, slippage_pct=0.00025)


def run_and_report(name, signals, df, cfg):
    res = run_backtest(df, signals, cfg)
    return compute_metrics(res)


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


def test_strategy(name, signal_fn, data, summary):
    """Test a strategy across all assets with rolling windows."""
    for asset, df in data.items():
        signals = signal_fn(df)
        m10 = run_and_report(name, signals, df, CFG_10)
        m5 = run_and_report(name, signals, df, CFG_5)

        print(f"  {asset}: 10bps Sharpe={m10['sharpe_ratio']:.2f} Ret={m10['total_return_pct']:.1f}% "
              f"Trades={m10['total_trades']} WR={m10['win_rate_pct']:.1f}% DD={m10['max_drawdown_pct']:.1f}%")
        print(f"  {asset}:  5bps Sharpe={m5['sharpe_ratio']:.2f} Ret={m5['total_return_pct']:.1f}%")

        # Rolling windows
        windows = []
        for ws, we, sl in rolling_windows(df):
            sig_w = signal_fn(sl)
            wm = run_and_report(name, sig_w, sl, CFG_10)
            windows.append(wm['sharpe_ratio'])

        n = len(windows)
        frac_pos = sum(1 for s in windows if s > 0) / n if n else 0
        frac_good = sum(1 for s in windows if s > 0.5) / n if n else 0
        avg_sh = np.mean(windows) if windows else 0
        med_sh = np.median(windows) if windows else 0

        print(f"  {asset}: {n} windows: avg={avg_sh:.2f} med={med_sh:.2f} "
              f"frac>0={frac_pos:.0%} frac>0.5={frac_good:.0%}")

        summary.append({
            "name": name, "asset": asset,
            "sh10": m10['sharpe_ratio'], "sh5": m5['sharpe_ratio'],
            "trades": m10['total_trades'], "wr": m10['win_rate_pct'],
            "ret": m10['total_return_pct'], "dd": m10['max_drawdown_pct'],
            "n_win": n, "frac_pos": frac_pos, "frac_good": frac_good,
            "avg_sh": avg_sh, "med_sh": med_sh,
        })


# ── Strategy signal generators ──────────────────────────────────────────

def consecutive_down_bars(df, threshold=5, hold_bars=12, cooldown_bars=2):
    """Buy after N consecutive down bars (close < open). Classic dip buyer."""
    close = df["close"].values
    open_ = df["open"].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)

    consec_down = 0
    hold_remaining = 0
    cooldown = 0

    for i in range(n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            # Still count consecutive bars during cooldown
            if close[i] < open_[i]:
                consec_down += 1
            else:
                consec_down = 0
            continue

        if close[i] < open_[i]:
            consec_down += 1
        else:
            consec_down = 0

        if consec_down >= threshold:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            consec_down = 0
            cooldown = cooldown_bars  # Applied after hold ends... actually after position exits

    return pd.Series(sig, index=df.index, dtype="int8")


def multi_bar_selloff(df, lookback=6, drop_pct=-3.0, hold_bars=12, cooldown_bars=3):
    """Buy after cumulative return over N bars drops below threshold."""
    close = df["close"].values
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

    return pd.Series(sig, index=df.index, dtype="int8")


def hammer_reversal(df, atr_mult=2.0, body_ratio=0.3, prior_down=3,
                    hold_bars=8, cooldown_bars=2):
    """Buy on hammer candle: large range, close in upper portion, after selloff.

    Hammer = range > atr_mult * ATR, body in upper 'body_ratio' of range,
    preceded by N down bars.
    """
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr_val = atr(df, 14).values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(prior_down, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        bar_range = high[i] - low[i]
        if np.isnan(atr_val[i]) or atr_val[i] == 0 or bar_range == 0:
            continue

        # Large range bar
        if bar_range < atr_mult * atr_val[i]:
            continue

        # Close in upper portion of range (hammer-like)
        close_pos = (close[i] - low[i]) / bar_range
        if close_pos < (1.0 - body_ratio):  # close must be in upper body_ratio of range
            continue

        # Preceded by selloff
        down_count = sum(1 for j in range(i - prior_down, i) if close[j] < open_[j])
        if down_count < prior_down:
            continue

        sig[i] = 1
        hold_remaining = hold_bars - 1
        cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype="int8")


def extreme_range_bar(df, lookback=48, hold_bars=10, cooldown_bars=2):
    """Buy when current bar has the widest range in N bars AND closes bullish."""
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values

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

        # Is this the widest range bar in lookback period?
        if bar_range[i] < np.max(bar_range[i - lookback:i]):
            continue

        # Must close bullish (close > open)
        if close[i] <= open_[i]:
            continue

        sig[i] = 1
        hold_remaining = hold_bars - 1
        cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype="int8")


def volume_climax_reversal(df, vol_lookback=48, hold_bars=10, cooldown_bars=2):
    """Buy on highest volume bar in N bars IF bar is bullish reversal.

    Detects volume climax: highest volume in lookback period +
    price closes in upper half of range (absorbing selling).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(vol_lookback, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        # Highest volume in lookback period
        if volume[i] < np.max(volume[i - vol_lookback:i]):
            continue

        bar_range = high[i] - low[i]
        if bar_range == 0:
            continue

        # Close in upper half of range (bullish absorption)
        close_pos = (close[i] - low[i]) / bar_range
        if close_pos < 0.5:
            continue

        sig[i] = 1
        hold_remaining = hold_bars - 1
        cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype="int8")


def selloff_volume_spike(df, vol_z_window=48, vol_z_thresh=2.0,
                          ret_lookback=6, ret_thresh=-2.0,
                          hold_bars=10, cooldown_bars=2):
    """Buy after selloff + volume spike combination.

    Requires BOTH:
    1. Cumulative return over ret_lookback bars < ret_thresh (selloff)
    2. Current volume z-score > vol_z_thresh (volume spike)

    This is a "capitulation" detector.
    """
    close = df["close"].values
    volume = df["volume"]

    vol_mean = volume.rolling(vol_z_window).mean().values
    vol_std = volume.rolling(vol_z_window).std().values
    vol_vals = volume.values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(max(vol_z_window, ret_lookback), n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        # Check selloff
        cum_ret = (close[i] / close[i - ret_lookback] - 1) * 100
        if cum_ret >= ret_thresh:
            continue

        # Check volume spike
        if vol_std[i] == 0 or np.isnan(vol_std[i]):
            continue
        vol_z = (vol_vals[i] - vol_mean[i]) / vol_std[i]
        if vol_z < vol_z_thresh:
            continue

        sig[i] = 1
        hold_remaining = hold_bars - 1
        cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype="int8")


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 110)
    print("ROUND 5: MULTI-BAR PATTERN & PRICE-ACTION ANOMALY STRATEGIES at 1h")
    print("=" * 110)

    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        print(f"{len(df)} bars")
        data[asset] = df

    summary = []

    # 1. Consecutive down bars
    print(f"\n{'─'*110}")
    print("1. CONSECUTIVE DOWN BARS (threshold=5, hold=12)")
    print(f"{'─'*110}")
    test_strategy("ConsecDown5", lambda df: consecutive_down_bars(df, threshold=5, hold_bars=12), data, summary)

    print(f"\n{'─'*110}")
    print("2. CONSECUTIVE DOWN BARS (threshold=6, hold=12)")
    print(f"{'─'*110}")
    test_strategy("ConsecDown6", lambda df: consecutive_down_bars(df, threshold=6, hold_bars=12), data, summary)

    print(f"\n{'─'*110}")
    print("3. CONSECUTIVE DOWN BARS (threshold=7, hold=16)")
    print(f"{'─'*110}")
    test_strategy("ConsecDown7", lambda df: consecutive_down_bars(df, threshold=7, hold_bars=16), data, summary)

    # 2. Multi-bar selloff
    print(f"\n{'─'*110}")
    print("4. MULTI-BAR SELLOFF (lookback=6, drop=-3%, hold=12)")
    print(f"{'─'*110}")
    test_strategy("Selloff3pct", lambda df: multi_bar_selloff(df, lookback=6, drop_pct=-3.0, hold_bars=12), data, summary)

    print(f"\n{'─'*110}")
    print("5. MULTI-BAR SELLOFF (lookback=12, drop=-5%, hold=16)")
    print(f"{'─'*110}")
    test_strategy("Selloff5pct", lambda df: multi_bar_selloff(df, lookback=12, drop_pct=-5.0, hold_bars=16), data, summary)

    print(f"\n{'─'*110}")
    print("6. MULTI-BAR SELLOFF (lookback=24, drop=-8%, hold=24)")
    print(f"{'─'*110}")
    test_strategy("Selloff8pct", lambda df: multi_bar_selloff(df, lookback=24, drop_pct=-8.0, hold_bars=24), data, summary)

    # 3. Hammer reversal
    print(f"\n{'─'*110}")
    print("7. HAMMER REVERSAL (atr_mult=2.0, body_ratio=0.3, prior_down=3, hold=8)")
    print(f"{'─'*110}")
    test_strategy("Hammer_2x", lambda df: hammer_reversal(df, atr_mult=2.0, body_ratio=0.3, prior_down=3, hold_bars=8), data, summary)

    print(f"\n{'─'*110}")
    print("8. HAMMER REVERSAL (atr_mult=2.5, body_ratio=0.25, prior_down=3, hold=12)")
    print(f"{'─'*110}")
    test_strategy("Hammer_2.5x", lambda df: hammer_reversal(df, atr_mult=2.5, body_ratio=0.25, prior_down=3, hold_bars=12), data, summary)

    # 4. Extreme range bar
    print(f"\n{'─'*110}")
    print("9. EXTREME RANGE BAR (lookback=48, hold=10)")
    print(f"{'─'*110}")
    test_strategy("ExtrRange48", lambda df: extreme_range_bar(df, lookback=48, hold_bars=10), data, summary)

    print(f"\n{'─'*110}")
    print("10. EXTREME RANGE BAR (lookback=96, hold=12)")
    print(f"{'─'*110}")
    test_strategy("ExtrRange96", lambda df: extreme_range_bar(df, lookback=96, hold_bars=12), data, summary)

    # 5. Volume climax reversal
    print(f"\n{'─'*110}")
    print("11. VOLUME CLIMAX REVERSAL (vol_lookback=48, hold=10)")
    print(f"{'─'*110}")
    test_strategy("VolClimax48", lambda df: volume_climax_reversal(df, vol_lookback=48, hold_bars=10), data, summary)

    print(f"\n{'─'*110}")
    print("12. VOLUME CLIMAX REVERSAL (vol_lookback=96, hold=12)")
    print(f"{'─'*110}")
    test_strategy("VolClimax96", lambda df: volume_climax_reversal(df, vol_lookback=96, hold_bars=12), data, summary)

    # 6. Selloff + volume spike (capitulation detector)
    print(f"\n{'─'*110}")
    print("13. SELLOFF + VOL SPIKE (vol_z=2.0, ret_lookback=6, ret=-2%, hold=10)")
    print(f"{'─'*110}")
    test_strategy("CapitA", lambda df: selloff_volume_spike(df, vol_z_thresh=2.0, ret_lookback=6, ret_thresh=-2.0, hold_bars=10), data, summary)

    print(f"\n{'─'*110}")
    print("14. SELLOFF + VOL SPIKE (vol_z=2.5, ret_lookback=12, ret=-3%, hold=12)")
    print(f"{'─'*110}")
    test_strategy("CapitB", lambda df: selloff_volume_spike(df, vol_z_thresh=2.5, ret_lookback=12, ret_thresh=-3.0, hold_bars=12), data, summary)

    print(f"\n{'─'*110}")
    print("15. SELLOFF + VOL SPIKE (vol_z=1.5, ret_lookback=6, ret=-2%, hold=12)")
    print(f"{'─'*110}")
    test_strategy("CapitC", lambda df: selloff_volume_spike(df, vol_z_thresh=1.5, ret_lookback=6, ret_thresh=-2.0, hold_bars=12), data, summary)

    # Summary table
    print(f"\n\n{'='*130}")
    print("SUMMARY TABLE")
    print(f"{'='*130}")
    print(f"{'Name':<15} {'Asset':<5} {'Sh10':>5} {'Sh5':>5} {'Tr':>5} {'WR%':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'Win':>4} {'F>0':>5} {'F>.5':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 130)
    for r in summary:
        print(f"{r['name']:<15} {r['asset']:<5} {r['sh10']:>5.2f} {r['sh5']:>5.2f} "
              f"{r['trades']:>5} {r['wr']:>5.1f} {r['ret']:>7.1f} {r['dd']:>6.1f} "
              f"{r['n_win']:>4} {r['frac_pos']:>5.0%} {r['frac_good']:>5.0%} "
              f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Highlight candidates
    print(f"\n\nPOTENTIAL QUALIFYING STRATEGIES (Sharpe > 0.3 at 10bps, frac>0 >= 60%):")
    print("-" * 100)
    qualifying = [r for r in summary if r['sh10'] > 0.3 and r['frac_pos'] >= 0.6]
    if qualifying:
        for r in qualifying:
            print(f"  {r['name']} on {r['asset']}: Sharpe={r['sh10']:.2f} Trades={r['trades']} "
                  f"Frac>0={r['frac_pos']:.0%} AvgWindowSharpe={r['avg_sh']:.2f}")
    else:
        print("  NONE found at strict thresholds.")

    # Relaxed threshold
    print(f"\n  Near-qualifying (Sharpe > 0.2 OR frac>0 >= 50%):")
    near = [r for r in summary if r['sh10'] > 0.2 or (r['frac_pos'] >= 0.5 and r['trades'] > 10)]
    for r in near:
        print(f"  {r['name']} on {r['asset']}: Sharpe={r['sh10']:.2f} Trades={r['trades']} "
              f"WR={r['wr']:.1f}% Frac>0={r['frac_pos']:.0%} AvgSh={r['avg_sh']:.2f}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
