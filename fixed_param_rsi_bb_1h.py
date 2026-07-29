"""Fixed-parameter RSI+BB stability test at 1h.

Uses known-good parameters from 4h regime sweep:
  rsi_period=14, oversold=30, overbought=70, bb_period=20, bb_std=2.0

Tests rolling window stability across BTC, ETH, SOL at 1h.
"""

import sys
import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

PARAM_SETS = [
    ("BASE(r14/ob70/os30/bb20/s2.0)", {"rsi_period": 14, "overbought": 70, "oversold": 30, "bb_period": 20, "bb_std": 2.0}),
    ("ALT1(r21/ob70/os30/bb20/s2.0)", {"rsi_period": 21, "overbought": 70, "oversold": 30, "bb_period": 20, "bb_std": 2.0}),
    ("ALT2(r14/ob75/os25/bb20/s2.5)", {"rsi_period": 14, "overbought": 75, "oversold": 25, "bb_period": 20, "bb_std": 2.5}),
    ("ALT3(r14/ob70/os30/bb20/s2.5)", {"rsi_period": 14, "overbought": 70, "oversold": 30, "bb_period": 20, "bb_std": 2.5}),
]

CFG_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
CFG_5 = BacktestConfig(fee_rate=0.00025, slippage_pct=0.00025)


def run_single(df, params, cfg):
    strat = RsiBollingerStrategy(**params)
    sig = strat.generate_signals(df)
    res = run_backtest(df, sig, cfg)
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


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 110)
    print("FIXED-PARAM RSI+BB at 1h — ROLLING STABILITY TEST")
    print("=" * 110)

    data = {}
    for asset, path in ASSETS.items():
        print(f"Loading {asset}...", end=" ", flush=True)
        df = load_candles(path, resample="1h")
        print(f"{len(df)} bars")
        data[asset] = df

    summary = []
    for label, params in PARAM_SETS:
        print(f"\n{'─'*110}")
        print(f"PARAMS: {label}")
        print(f"{'─'*110}")

        for asset, df in data.items():
            m10 = run_single(df, params, CFG_10)
            m5 = run_single(df, params, CFG_5)

            print(f"\n  {asset}: 10bps Sharpe={m10['sharpe_ratio']:.2f} Ret={m10['total_return_pct']:.1f}% "
                  f"Trades={m10['total_trades']} WR={m10['win_rate_pct']:.1f}% DD={m10['max_drawdown_pct']:.1f}%")
            print(f"  {asset}:  5bps Sharpe={m5['sharpe_ratio']:.2f} Ret={m5['total_return_pct']:.1f}% "
                  f"Trades={m5['total_trades']} WR={m5['win_rate_pct']:.1f}%")

            windows = []
            for ws, we, sl in rolling_windows(df):
                wm = run_single(sl, params, CFG_10)
                windows.append(wm['sharpe_ratio'])

            n = len(windows)
            frac_pos = sum(1 for s in windows if s > 0) / n if n else 0
            frac_good = sum(1 for s in windows if s > 0.5) / n if n else 0
            avg_sh = np.mean(windows) if windows else 0
            med_sh = np.median(windows) if windows else 0

            print(f"  {asset}: {n} rolling windows: avg={avg_sh:.2f} med={med_sh:.2f} "
                  f"frac>0={frac_pos:.0%} frac>0.5={frac_good:.0%}")

            summary.append({
                "label": label, "asset": asset,
                "sh10": m10['sharpe_ratio'], "sh5": m5['sharpe_ratio'],
                "trades": m10['total_trades'], "wr": m10['win_rate_pct'],
                "ret": m10['total_return_pct'], "dd": m10['max_drawdown_pct'],
                "n_win": n, "frac_pos": frac_pos, "frac_good": frac_good,
                "avg_sh": avg_sh, "med_sh": med_sh,
            })

    # Summary table
    print(f"\n\n{'='*130}")
    print("SUMMARY TABLE")
    print(f"{'='*130}")
    print(f"{'Params':<32} {'Asset':<5} {'Sh10':>5} {'Sh5':>5} {'Tr':>5} {'WR%':>5} "
          f"{'Ret%':>7} {'DD%':>6} {'Win':>4} {'F>0':>5} {'F>.5':>5} {'AvgSh':>6} {'MedSh':>6}")
    print("-" * 130)
    for r in summary:
        print(f"{r['label']:<32} {r['asset']:<5} {r['sh10']:>5.2f} {r['sh5']:>5.2f} "
              f"{r['trades']:>5} {r['wr']:>5.1f} {r['ret']:>7.1f} {r['dd']:>6.1f} "
              f"{r['n_win']:>4} {r['frac_pos']:>5.0%} {r['frac_good']:>5.0%} "
              f"{r['avg_sh']:>6.2f} {r['med_sh']:>6.2f}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
