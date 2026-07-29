"""Parameter sensitivity test for top 3 candidate strategies.

Tests multiple parameter variations to confirm the signal is robust
(not a lucky param pick). A strategy passes if MULTIPLE nearby param
sets produce positive results.
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CFG_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)


def extreme_range_bar(df, lookback=96, hold_bars=12, cooldown_bars=2):
    """Widest range in lookback bars + bullish close → buy."""
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
        if bar_range[i] < np.max(bar_range[i - lookback:i]):
            continue
        if close[i] <= open_[i]:
            continue
        sig[i] = 1
        hold_remaining = hold_bars - 1
        cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype="int8")


def multi_bar_selloff(df, lookback=24, drop_pct=-8.0, hold_bars=24, cooldown_bars=3):
    """Cumulative return < threshold over lookback bars → buy."""
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


def vol_spike(df, vol_z_window=96, vol_z_thresh=2.5, delta_thresh=10.0,
              hold_bars=8, cooldown_bars=1):
    """Volume z-score spike + price delta → buy only."""
    close = df["close"].values
    open_ = df["open"].values
    volume = df["volume"]

    vol_mean = volume.rolling(vol_z_window).mean().values
    vol_std = volume.rolling(vol_z_window).std().values
    vol_vals = volume.values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cooldown = 0

    for i in range(vol_z_window, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        if vol_std[i] == 0 or np.isnan(vol_std[i]):
            continue
        vol_z = (vol_vals[i] - vol_mean[i]) / vol_std[i]
        if vol_z < vol_z_thresh:
            continue

        # Price delta as %
        if close[i] == 0:
            continue
        delta_pct = abs(close[i] - open_[i]) / open_[i] * 100
        if delta_pct < delta_thresh:
            continue

        # Buy only: must be bullish bar
        if close[i] <= open_[i]:
            continue

        sig[i] = 1
        hold_remaining = hold_bars - 1
        cooldown = cooldown_bars

    return pd.Series(sig, index=df.index, dtype="int8")


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


if __name__ == '__main__':
    t0 = time.time()

    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        print(f"{len(df)} bars")
        data[asset] = df

    # ═══════════════════════════════════════════════════════════════
    # 1. EXTREME RANGE BAR — Parameter sensitivity
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("EXTREME RANGE BAR — PARAMETER SENSITIVITY")
    print(f"{'='*120}")

    er_params = [
        {"lookback": 48, "hold_bars": 8, "cooldown_bars": 2},
        {"lookback": 48, "hold_bars": 10, "cooldown_bars": 2},
        {"lookback": 48, "hold_bars": 12, "cooldown_bars": 2},
        {"lookback": 72, "hold_bars": 8, "cooldown_bars": 2},
        {"lookback": 72, "hold_bars": 10, "cooldown_bars": 2},
        {"lookback": 72, "hold_bars": 12, "cooldown_bars": 2},
        {"lookback": 96, "hold_bars": 8, "cooldown_bars": 2},
        {"lookback": 96, "hold_bars": 10, "cooldown_bars": 2},
        {"lookback": 96, "hold_bars": 12, "cooldown_bars": 2},
        {"lookback": 96, "hold_bars": 16, "cooldown_bars": 2},
        {"lookback": 120, "hold_bars": 10, "cooldown_bars": 2},
        {"lookback": 120, "hold_bars": 12, "cooldown_bars": 2},
    ]

    print(f"\n{'Params':<30} {'Asset':<5} {'Sh':>5} {'Tr':>5} {'WR':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'F>0':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 100)

    er_results = []
    for p in er_params:
        label = f"lb={p['lookback']}/h={p['hold_bars']}"
        for asset, df in data.items():
            r = evaluate(lambda df, _p=p: extreme_range_bar(df, **_p), df)
            er_results.append({"label": label, "asset": asset, **p, **r})
            print(f"{label:<30} {asset:<5} {r['sharpe']:>5.2f} {r['trades']:>5} {r['wr']:>5.1f} "
                  f"{r['ret']:>7.1f} {r['dd']:>6.1f} {r['frac_pos']:>5.0%} "
                  f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    # Count qualifying param sets per asset
    print(f"\nQualifying param sets (Sharpe > 0 AND frac>0 >= 50%):")
    for asset in ASSETS:
        q = [r for r in er_results if r['asset'] == asset
             and r['sharpe'] > 0 and r['frac_pos'] >= 0.5]
        total = [r for r in er_results if r['asset'] == asset]
        print(f"  {asset}: {len(q)}/{len(total)} param sets qualify")
        if q:
            best = max(q, key=lambda x: x['avg_sh'])
            print(f"    Best: {best['label']} Sharpe={best['sharpe']:.2f} "
                  f"frac>0={best['frac_pos']:.0%} avg_sh={best['avg_sh']:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 2. MULTI-BAR SELLOFF — Parameter sensitivity
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("MULTI-BAR SELLOFF — PARAMETER SENSITIVITY")
    print(f"{'='*120}")

    ms_params = [
        {"lookback": 12, "drop_pct": -5.0, "hold_bars": 12, "cooldown_bars": 3},
        {"lookback": 12, "drop_pct": -5.0, "hold_bars": 16, "cooldown_bars": 3},
        {"lookback": 12, "drop_pct": -5.0, "hold_bars": 24, "cooldown_bars": 3},
        {"lookback": 18, "drop_pct": -6.0, "hold_bars": 16, "cooldown_bars": 3},
        {"lookback": 18, "drop_pct": -6.0, "hold_bars": 24, "cooldown_bars": 3},
        {"lookback": 24, "drop_pct": -6.0, "hold_bars": 16, "cooldown_bars": 3},
        {"lookback": 24, "drop_pct": -6.0, "hold_bars": 24, "cooldown_bars": 3},
        {"lookback": 24, "drop_pct": -8.0, "hold_bars": 16, "cooldown_bars": 3},
        {"lookback": 24, "drop_pct": -8.0, "hold_bars": 24, "cooldown_bars": 3},
        {"lookback": 24, "drop_pct": -8.0, "hold_bars": 32, "cooldown_bars": 3},
        {"lookback": 24, "drop_pct": -10.0, "hold_bars": 24, "cooldown_bars": 3},
        {"lookback": 36, "drop_pct": -10.0, "hold_bars": 24, "cooldown_bars": 3},
    ]

    print(f"\n{'Params':<30} {'Asset':<5} {'Sh':>5} {'Tr':>5} {'WR':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'F>0':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 100)

    ms_results = []
    for p in ms_params:
        label = f"lb={p['lookback']}/d={p['drop_pct']}/h={p['hold_bars']}"
        for asset, df in data.items():
            r = evaluate(lambda df, _p=p: multi_bar_selloff(df, **_p), df)
            ms_results.append({"label": label, "asset": asset, **p, **r})
            print(f"{label:<30} {asset:<5} {r['sharpe']:>5.2f} {r['trades']:>5} {r['wr']:>5.1f} "
                  f"{r['ret']:>7.1f} {r['dd']:>6.1f} {r['frac_pos']:>5.0%} "
                  f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    print(f"\nQualifying param sets (Sharpe > 0 AND frac>0 >= 50%):")
    for asset in ASSETS:
        q = [r for r in ms_results if r['asset'] == asset
             and r['sharpe'] > 0 and r['frac_pos'] >= 0.5]
        total = [r for r in ms_results if r['asset'] == asset]
        print(f"  {asset}: {len(q)}/{len(total)} param sets qualify")
        if q:
            best = max(q, key=lambda x: x['avg_sh'])
            print(f"    Best: {best['label']} Sharpe={best['sharpe']:.2f} "
                  f"frac>0={best['frac_pos']:.0%} avg_sh={best['avg_sh']:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 3. VOL SPIKE — Parameter sensitivity (especially for ETH)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("VOL SPIKE — PARAMETER SENSITIVITY")
    print(f"{'='*120}")

    vs_params = [
        {"vol_z_window": 48, "vol_z_thresh": 2.0, "delta_thresh": 5.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 48, "vol_z_thresh": 2.5, "delta_thresh": 5.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 48, "vol_z_thresh": 2.0, "delta_thresh": 10.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 96, "vol_z_thresh": 2.0, "delta_thresh": 5.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 96, "vol_z_thresh": 2.0, "delta_thresh": 10.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 96, "vol_z_thresh": 2.5, "delta_thresh": 5.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 96, "vol_z_thresh": 2.5, "delta_thresh": 10.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 96, "vol_z_thresh": 2.5, "delta_thresh": 10.0, "hold_bars": 12, "cooldown_bars": 1},
        {"vol_z_window": 96, "vol_z_thresh": 3.0, "delta_thresh": 10.0, "hold_bars": 8, "cooldown_bars": 1},
        {"vol_z_window": 144, "vol_z_thresh": 2.5, "delta_thresh": 10.0, "hold_bars": 8, "cooldown_bars": 1},
    ]

    print(f"\n{'Params':<42} {'Asset':<5} {'Sh':>5} {'Tr':>5} {'WR':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'F>0':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 110)

    vs_results = []
    for p in vs_params:
        label = f"zw={p['vol_z_window']}/zt={p['vol_z_thresh']}/dt={p['delta_thresh']}/h={p['hold_bars']}"
        for asset, df in data.items():
            r = evaluate(lambda df, _p=p: vol_spike(df, **_p), df)
            vs_results.append({"label": label, "asset": asset, **p, **r})
            print(f"{label:<42} {asset:<5} {r['sharpe']:>5.2f} {r['trades']:>5} {r['wr']:>5.1f} "
                  f"{r['ret']:>7.1f} {r['dd']:>6.1f} {r['frac_pos']:>5.0%} "
                  f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    print(f"\nQualifying param sets (Sharpe > 0 AND frac>0 >= 50%):")
    for asset in ASSETS:
        q = [r for r in vs_results if r['asset'] == asset
             and r['sharpe'] > 0 and r['frac_pos'] >= 0.5]
        total = [r for r in vs_results if r['asset'] == asset]
        print(f"  {asset}: {len(q)}/{len(total)} param sets qualify")
        if q:
            best = max(q, key=lambda x: x['avg_sh'])
            print(f"    Best: {best['label']} Sharpe={best['sharpe']:.2f} "
                  f"frac>0={best['frac_pos']:.0%} avg_sh={best['avg_sh']:.2f}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
