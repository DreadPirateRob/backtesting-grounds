"""Fixed-parameter CVD Divergence + additional event-driven strategies at 1h.

Tests:
1. CVD Divergence with known-good 4h params
2. Stochastic RSI extreme reversal (new)
3. Multi-indicator confluence extreme (new)
"""

import sys
import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from backtester.strategy import rsi, bollinger_bands, stoch_rsi, atr, sma
from strategies.cvd_divergence_strategy import CvdDivergenceStrategy

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CFG_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
CFG_5 = BacktestConfig(fee_rate=0.00025, slippage_pct=0.00025)


def run_and_report(name, signals, df, cfg):
    res = run_backtest(df, signals, cfg)
    m = compute_metrics(res)
    return m


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


# ── Strategy signal generators ──

def cvd_div_signals(df):
    """CVD Divergence with known-good params."""
    strat = CvdDivergenceStrategy(
        pivot_lookback=10, cvd_smoothing=5,
        confirmation_bars=2, atr_stop_mult=2.0,
    )
    return strat.generate_signals(df)


def cvd_div_tight(df):
    """CVD Divergence with tighter params (more selective)."""
    strat = CvdDivergenceStrategy(
        pivot_lookback=15, cvd_smoothing=5,
        confirmation_bars=2, atr_stop_mult=1.5,
    )
    return strat.generate_signals(df)


def stoch_rsi_extreme(df, k_period=14, d_period=3, rsi_period=14,
                       extreme=10, hold_bars=8, buy_only=True):
    """StochRSI extreme reversal with fixed hold."""
    close = df["close"]
    r = rsi(close, rsi_period)
    # StochRSI = (RSI - min(RSI, k)) / (max(RSI, k) - min(RSI, k))
    r_min = r.rolling(k_period).min()
    r_max = r.rolling(k_period).max()
    sr = (r - r_min) / (r_max - r_min).replace(0, np.nan) * 100
    sr_k = sr.rolling(d_period).mean()  # %K line

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(n):
        if hold_remaining > 0:
            sig[i] = sig[i - 1]
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        v = sr_k.iloc[i]
        if np.isnan(v):
            continue

        if v < extreme:
            sig[i] = 1  # Buy on extreme oversold
            hold_remaining = hold_bars - 1
        elif not buy_only and v > (100 - extreme):
            sig[i] = -1  # Short on extreme overbought
            hold_remaining = hold_bars - 1
        elif i > 0 and sig[i - 1] != 0:
            cooldown = 1

    return pd.Series(sig, index=df.index, dtype="int8")


def rsi_extreme_reversal(df, rsi_period=14, extreme=15, hold_bars=8, buy_only=True):
    """Plain RSI extreme with fixed hold. Very selective."""
    close = df["close"]
    r = rsi(close, rsi_period)

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(n):
        if hold_remaining > 0:
            sig[i] = sig[i - 1]
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        rv = r.iloc[i]
        if np.isnan(rv):
            continue

        if rv < extreme:
            sig[i] = 1
            hold_remaining = hold_bars - 1
        elif not buy_only and rv > (100 - extreme):
            sig[i] = -1
            hold_remaining = hold_bars - 1
        elif i > 0 and sig[i - 1] != 0:
            cooldown = 1

    return pd.Series(sig, index=df.index, dtype="int8")


def bb_touch_reversal(df, bb_period=20, bb_std=2.5, hold_bars=8, buy_only=True):
    """Bollinger Band touch with fixed hold. Entry only on touch."""
    close = df["close"]
    upper, mid, lower = bollinger_bands(close, bb_period, bb_std)

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(n):
        if hold_remaining > 0:
            sig[i] = sig[i - 1]
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        c = close.iloc[i]
        lo = lower.iloc[i]
        up = upper.iloc[i]
        if np.isnan(lo) or np.isnan(up):
            continue

        if c <= lo:
            sig[i] = 1
            hold_remaining = hold_bars - 1
        elif not buy_only and c >= up:
            sig[i] = -1
            hold_remaining = hold_bars - 1
        elif i > 0 and sig[i - 1] != 0:
            cooldown = 1

    return pd.Series(sig, index=df.index, dtype="int8")


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 110)
    print("ROUND 4: FIXED-PARAM EVENT-DRIVEN STRATEGIES at 1h")
    print("=" * 110)

    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h", extra_columns=["taker_buy_base_volume"])
        print(f"{len(df)} bars")
        data[asset] = df

    summary = []

    # 1. CVD Divergence
    print(f"\n{'─'*110}")
    print("1. CVD DIVERGENCE (pivot=10, smooth=5, atr_stop=2.0)")
    print(f"{'─'*110}")
    test_strategy("CVD_base", cvd_div_signals, data, summary)

    print(f"\n{'─'*110}")
    print("2. CVD DIVERGENCE TIGHT (pivot=15, smooth=5, atr_stop=1.5)")
    print(f"{'─'*110}")
    test_strategy("CVD_tight", cvd_div_tight, data, summary)

    # 3. StochRSI extreme
    print(f"\n{'─'*110}")
    print("3. STOCH RSI EXTREME (extreme=10, hold=8, buy_only)")
    print(f"{'─'*110}")
    test_strategy("StochRSI_10", lambda df: stoch_rsi_extreme(df, extreme=10, hold_bars=8, buy_only=True), data, summary)

    print(f"\n{'─'*110}")
    print("4. STOCH RSI EXTREME (extreme=5, hold=12, buy_only)")
    print(f"{'─'*110}")
    test_strategy("StochRSI_5", lambda df: stoch_rsi_extreme(df, extreme=5, hold_bars=12, buy_only=True), data, summary)

    # 5. Plain RSI extreme
    print(f"\n{'─'*110}")
    print("5. RSI EXTREME (extreme=15, hold=8, buy_only)")
    print(f"{'─'*110}")
    test_strategy("RSI_ext15", lambda df: rsi_extreme_reversal(df, extreme=15, hold_bars=8, buy_only=True), data, summary)

    print(f"\n{'─'*110}")
    print("6. RSI EXTREME (extreme=10, hold=12, buy_only)")
    print(f"{'─'*110}")
    test_strategy("RSI_ext10", lambda df: rsi_extreme_reversal(df, extreme=10, hold_bars=12, buy_only=True), data, summary)

    # 7. BB touch reversal
    print(f"\n{'─'*110}")
    print("7. BB TOUCH REVERSAL (bb=20, std=2.5, hold=8, buy_only)")
    print(f"{'─'*110}")
    test_strategy("BB_touch", lambda df: bb_touch_reversal(df, bb_std=2.5, hold_bars=8, buy_only=True), data, summary)

    print(f"\n{'─'*110}")
    print("8. BB TOUCH REVERSAL (bb=20, std=3.0, hold=12, buy_only)")
    print(f"{'─'*110}")
    test_strategy("BB_3sig", lambda df: bb_touch_reversal(df, bb_std=3.0, hold_bars=12, buy_only=True), data, summary)

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
        print("\n  Near-qualifying (Sharpe > 0.1 at 10bps):")
        near = [r for r in summary if r['sh10'] > 0.1]
        for r in near:
            print(f"  {r['name']} on {r['asset']}: Sharpe={r['sh10']:.2f} Trades={r['trades']} "
                  f"Frac>0={r['frac_pos']:.0%}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
