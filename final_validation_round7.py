"""Final Validation Round 7: Donchian Breakout + Green Momentum

Comprehensive validation of the two genuinely new strategies found in Round 7:
1. Donchian Breakout + Volume (breakout/momentum — ZERO overlap with dip-buying)
2. Green Momentum (consecutive bullish bars — trend continuation)

Tests:
- Full eval at 10bps and 5bps fees
- Rolling 12-month window stability (16 windows, 3-month steps)
- Per-year Sharpe breakdown (2021-2025)
- Parameter sensitivity (12+ param sets per strategy)
- Signal overlap vs ALL 3 existing Round 5 strategies
- Combined portfolio diversification analysis
"""

import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

ASSETS = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CFG_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
CFG_5 = BacktestConfig(fee_rate=0.00025, slippage_pct=0.00025)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

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
        "frac_gt05": sum(1 for s in windows if s > 0.5) / n if n else 0,
        "avg_sh": np.mean(windows) if windows else 0,
        "med_sh": np.median(windows) if windows else 0,
        "n_win": n,
        "windows": windows,
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
# Strategy signal functions
# ═══════════════════════════════════════════════════════════════════

# --- New strategies (Round 7) ---

def donchian_vol(df, entry_period=480, vol_mult=2.0, hold_bars=24, cooldown=12):
    close = df['close'].values
    high = df['high'].values
    vol = df['volume'].values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cool = 0

    for i in range(entry_period + 1, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cool > 0:
            cool -= 1
            continue

        highest = np.max(high[i - entry_period:i])
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue

        if close[i] > highest and vol[i] > vol_mult * vol_avg[i]:
            sig[i] = 1
            hold_remaining = hold_bars - 1
            cool = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


def green_momentum(df, min_green=4, min_gain_pct=4.0, hold_bars=16, cooldown=3):
    close = df['close'].values
    open_ = df['open'].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cool = 0

    for i in range(min_green, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cool > 0:
            cool -= 1
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
            cool = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# --- Existing strategies (Round 5) for overlap analysis ---

def vol_spike(df, window=96, threshold=2.5, delta_thresh=10.0, hold=8):
    volume = df['volume']
    close = df['close']
    open_ = df['open']
    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std()
    vol_zscore = (volume - vol_mean) / vol_std.replace(0, np.nan)

    if 'taker_buy_base_volume' in df.columns:
        buy_vol = df['taker_buy_base_volume']
        delta_pct = (2 * buy_vol / volume.replace(0, np.nan) - 1) * 100
    else:
        delta_pct = pd.Series(np.sign(close.values - open_.values) * 100,
                              index=df.index, dtype='float64')

    buy_spike = (vol_zscore > threshold) & (delta_pct > delta_thresh)

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    buy_vals = buy_spike.values
    hold_remaining = 0
    cooldown = 0

    for i in range(n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if buy_vals[i]:
            sig[i] = 1
            hold_remaining = hold - 1
            cooldown = 1

    return pd.Series(sig, index=df.index, dtype='int8')


def multi_bar_selloff(df, lookback=24, drop_pct=-8.0, hold=16, cooldown=3):
    close = df['close'].values
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cool = 0

    for i in range(lookback, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cool > 0:
            cool -= 1
            continue
        cum_ret = (close[i] / close[i - lookback] - 1) * 100
        if cum_ret < drop_pct:
            sig[i] = 1
            hold_remaining = hold - 1
            cool = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


def extreme_range(df, lookback=72, hold=8, cooldown=2):
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    bar_range = high - low
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    hold_remaining = 0
    cool = 0

    for i in range(lookback, n):
        if hold_remaining > 0:
            sig[i] = 1
            hold_remaining -= 1
            continue
        if cool > 0:
            cool -= 1
            continue
        if bar_range[i] < np.max(bar_range[i - lookback:i]):
            continue
        if close[i] <= open_[i]:
            continue
        sig[i] = 1
        hold_remaining = hold - 1
        cool = cooldown

    return pd.Series(sig, index=df.index, dtype='int8')


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print("  FINAL VALIDATION ROUND 7: Donchian Breakout + Green Momentum")
    print("=" * 80)

    # Load data
    data = {}
    for asset, path in ASSETS.items():
        df = load_candles(path, resample="1h",
                          extra_columns=["taker_buy_base_volume"])
        data[asset] = df
        print(f"  {asset}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # ═══════════════════════════════════════════════════════════════
    # 1. DONCHIAN BREAKOUT — Full Evaluation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  1. DONCHIAN BREAKOUT + VOLUME FILTER")
    print("=" * 80)

    print("\n  --- Best params: entry_period=480, vol_mult=2.0, hold=24 ---")
    for asset, df in data.items():
        r10 = evaluate(lambda d: donchian_vol(d, 480, 2.0, 24), df, CFG_10)
        r5 = evaluate(lambda d: donchian_vol(d, 480, 2.0, 24), df, CFG_5)
        print(f"\n  {asset} @10bps: Sharpe {r10['sharpe']:+.2f}, "
              f"Ret {r10['ret']:+.1f}%, DD {r10['dd']:.1f}%, "
              f"Trades {r10['trades']}, WR {r10['wr']:.0f}%")
        print(f"    Windows: {r10['frac_pos']*100:.0f}% pos, "
              f"{r10['frac_gt05']*100:.0f}% >0.5, "
              f"avg {r10['avg_sh']:+.2f}, med {r10['med_sh']:+.2f}")
        print(f"  {asset} @5bps:  Sharpe {r5['sharpe']:+.2f}, "
              f"Ret {r5['ret']:+.1f}%, Trades {r5['trades']}")

    # Per-year breakdown
    print("\n  --- Per-year Sharpe (10bps) ---")
    for asset, df in data.items():
        py = per_year_analysis(df,
                               lambda d: donchian_vol(d, 480, 2.0, 24),
                               CFG_10)
        yrs = " | ".join(f"{y}: {m['sharpe_ratio']:+.2f}" for y, m in py.items())
        print(f"  {asset}: {yrs}")

    # Parameter sensitivity
    print("\n  --- Parameter sensitivity (12 combos) ---")
    donchian_params = [
        ("p240_v1.5_h24", dict(entry_period=240, vol_mult=1.5, hold_bars=24)),
        ("p240_v2.0_h24", dict(entry_period=240, vol_mult=2.0, hold_bars=24)),
        ("p240_v2.5_h24", dict(entry_period=240, vol_mult=2.5, hold_bars=24)),
        ("p480_v1.5_h24", dict(entry_period=480, vol_mult=1.5, hold_bars=24)),
        ("p480_v2.0_h24", dict(entry_period=480, vol_mult=2.0, hold_bars=24)),
        ("p480_v2.0_h48", dict(entry_period=480, vol_mult=2.0, hold_bars=48)),
        ("p480_v2.5_h24", dict(entry_period=480, vol_mult=2.5, hold_bars=24)),
        ("p480_v2.5_h48", dict(entry_period=480, vol_mult=2.5, hold_bars=48)),
        ("p720_v1.5_h24", dict(entry_period=720, vol_mult=1.5, hold_bars=24)),
        ("p720_v2.0_h24", dict(entry_period=720, vol_mult=2.0, hold_bars=24)),
        ("p720_v2.0_h48", dict(entry_period=720, vol_mult=2.0, hold_bars=48)),
        ("p720_v2.5_h48", dict(entry_period=720, vol_mult=2.5, hold_bars=48)),
    ]
    print(f"  {'Params':<18} {'BTC':>8} {'ETH':>8} {'SOL':>8}")
    n_all_pos = 0
    for label, params in donchian_params:
        sharpes = {}
        for asset, df in data.items():
            r = evaluate(lambda d, p=params: donchian_vol(d, **p), df, CFG_10)
            sharpes[asset] = r['sharpe']
        all_pos = all(s > 0 for s in sharpes.values())
        if all_pos:
            n_all_pos += 1
        marker = " ***" if all_pos else ""
        print(f"  {label:<18} {sharpes['BTC']:+8.2f} {sharpes['ETH']:+8.2f} "
              f"{sharpes['SOL']:+8.2f}{marker}")
    print(f"\n  Cross-asset positive: {n_all_pos}/{len(donchian_params)}")

    # ═══════════════════════════════════════════════════════════════
    # 2. GREEN MOMENTUM — Full Evaluation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  2. GREEN MOMENTUM (Consecutive Bullish Bars)")
    print("=" * 80)

    print("\n  --- Best cross-asset params: min_green=4, gain=4%, hold=16 ---")
    for asset, df in data.items():
        r10 = evaluate(lambda d: green_momentum(d, 4, 4.0, 16), df, CFG_10)
        r5 = evaluate(lambda d: green_momentum(d, 4, 4.0, 16), df, CFG_5)
        print(f"\n  {asset} @10bps: Sharpe {r10['sharpe']:+.2f}, "
              f"Ret {r10['ret']:+.1f}%, DD {r10['dd']:.1f}%, "
              f"Trades {r10['trades']}, WR {r10['wr']:.0f}%")
        print(f"    Windows: {r10['frac_pos']*100:.0f}% pos, "
              f"{r10['frac_gt05']*100:.0f}% >0.5, "
              f"avg {r10['avg_sh']:+.2f}, med {r10['med_sh']:+.2f}")
        print(f"  {asset} @5bps:  Sharpe {r5['sharpe']:+.2f}, "
              f"Ret {r5['ret']:+.1f}%, Trades {r5['trades']}")

    # Also test BTC-dominant params: h12
    print("\n  --- BTC-dominant params: min_green=4, gain=4%, hold=12 ---")
    for asset, df in data.items():
        r10 = evaluate(lambda d: green_momentum(d, 4, 4.0, 12), df, CFG_10)
        print(f"  {asset} @10bps: Sharpe {r10['sharpe']:+.2f}, "
              f"Ret {r10['ret']:+.1f}%, WR {r10['wr']:.0f}%, "
              f"Trades {r10['trades']}, "
              f"{r10['frac_pos']*100:.0f}%w pos")

    # Per-year breakdown
    print("\n  --- Per-year Sharpe (10bps, h16) ---")
    for asset, df in data.items():
        py = per_year_analysis(df,
                               lambda d: green_momentum(d, 4, 4.0, 16),
                               CFG_10)
        yrs = " | ".join(f"{y}: {m['sharpe_ratio']:+.2f}" for y, m in py.items())
        print(f"  {asset}: {yrs}")

    # Parameter sensitivity
    print("\n  --- Parameter sensitivity (12 combos) ---")
    green_params = [
        ("g3_3p_h12", dict(min_green=3, min_gain_pct=3.0, hold_bars=12)),
        ("g3_3p_h16", dict(min_green=3, min_gain_pct=3.0, hold_bars=16)),
        ("g3_4p_h12", dict(min_green=3, min_gain_pct=4.0, hold_bars=12)),
        ("g3_4p_h16", dict(min_green=3, min_gain_pct=4.0, hold_bars=16)),
        ("g4_3p_h12", dict(min_green=4, min_gain_pct=3.0, hold_bars=12)),
        ("g4_3p_h16", dict(min_green=4, min_gain_pct=3.0, hold_bars=16)),
        ("g4_4p_h12", dict(min_green=4, min_gain_pct=4.0, hold_bars=12)),
        ("g4_4p_h16", dict(min_green=4, min_gain_pct=4.0, hold_bars=16)),
        ("g4_5p_h16", dict(min_green=4, min_gain_pct=5.0, hold_bars=16)),
        ("g5_4p_h12", dict(min_green=5, min_gain_pct=4.0, hold_bars=12)),
        ("g5_5p_h12", dict(min_green=5, min_gain_pct=5.0, hold_bars=12)),
        ("g5_5p_h16", dict(min_green=5, min_gain_pct=5.0, hold_bars=16)),
    ]
    print(f"  {'Params':<18} {'BTC':>8} {'ETH':>8} {'SOL':>8}")
    n_all_pos = 0
    for label, params in green_params:
        sharpes = {}
        for asset, df in data.items():
            r = evaluate(lambda d, p=params: green_momentum(d, **p), df, CFG_10)
            sharpes[asset] = r['sharpe']
        all_pos = all(s > 0 for s in sharpes.values())
        if all_pos:
            n_all_pos += 1
        marker = " ***" if all_pos else ""
        print(f"  {label:<18} {sharpes['BTC']:+8.2f} {sharpes['ETH']:+8.2f} "
              f"{sharpes['SOL']:+8.2f}{marker}")
    print(f"\n  Cross-asset positive: {n_all_pos}/{len(green_params)}")

    # ═══════════════════════════════════════════════════════════════
    # 3. SIGNAL OVERLAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  3. SIGNAL OVERLAP ANALYSIS")
    print("=" * 80)

    # Generate all signals for each asset
    for asset, df in data.items():
        sigs = {
            "Donchian": donchian_vol(df, 480, 2.0, 24),
            "GreenMom": green_momentum(df, 4, 4.0, 16),
            "VolSpike": vol_spike(df),
            "Selloff": multi_bar_selloff(df),
            "ExtrRange": extreme_range(df),
        }

        print(f"\n  {asset}:")
        # Entry-bar overlap (signal transitions from 0→1)
        entries = {}
        for name, sig in sigs.items():
            entry = (sig == 1) & (sig.shift(1).fillna(0) == 0)
            entries[name] = entry
            print(f"    {name}: {entry.sum()} entry signals")

        # Pairwise overlap: new strategies vs existing
        new_strats = ["Donchian", "GreenMom"]
        existing = ["VolSpike", "Selloff", "ExtrRange"]

        print(f"\n    Pairwise overlap (entry bars ±2h):")
        for ns in new_strats:
            for es in existing:
                # Count entries in ns that have an entry in es within ±2 bars
                ns_entries = entries[ns]
                es_entries = entries[es]
                ns_idx = ns_entries[ns_entries].index
                es_idx = es_entries[es_entries].index

                overlap = 0
                for idx in ns_idx:
                    window_start = idx - pd.Timedelta(hours=2)
                    window_end = idx + pd.Timedelta(hours=2)
                    nearby = es_idx[(es_idx >= window_start) & (es_idx <= window_end)]
                    if len(nearby) > 0:
                        overlap += 1

                pct = overlap / len(ns_idx) * 100 if len(ns_idx) > 0 else 0
                print(f"      {ns} ∩ {es}: {overlap}/{len(ns_idx)} = {pct:.1f}%")

        # Also check new vs new overlap
        ns1, ns2 = "Donchian", "GreenMom"
        ns1_idx = entries[ns1][entries[ns1]].index
        ns2_idx = entries[ns2][entries[ns2]].index
        overlap = 0
        for idx in ns1_idx:
            window_start = idx - pd.Timedelta(hours=2)
            window_end = idx + pd.Timedelta(hours=2)
            nearby = ns2_idx[(ns2_idx >= window_start) & (ns2_idx <= window_end)]
            if len(nearby) > 0:
                overlap += 1
        pct = overlap / len(ns1_idx) * 100 if len(ns1_idx) > 0 else 0
        print(f"      {ns1} ∩ {ns2}: {overlap}/{len(ns1_idx)} = {pct:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 4. PORTFOLIO DIVERSIFICATION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  4. PORTFOLIO DIVERSIFICATION — All 5 strategies combined")
    print("=" * 80)

    for asset, df in data.items():
        # Get equity curves from each strategy
        strategies = {
            "VolSpike": lambda d: vol_spike(d),
            "Selloff": lambda d: multi_bar_selloff(d),
            "ExtrRange": lambda d: extreme_range(d),
            "Donchian": lambda d: donchian_vol(d, 480, 2.0, 24),
            "GreenMom": lambda d: green_momentum(d, 4, 4.0, 16),
        }

        returns_list = []
        for name, fn in strategies.items():
            sig = fn(df)
            res = run_backtest(df, sig, CFG_10)
            # Compute hourly returns from equity curve
            eq = res['equity_curve']
            ret = eq.pct_change().fillna(0)
            returns_list.append(ret)

        # Equal-weight portfolio
        combined = sum(returns_list) / len(returns_list)
        combined_eq = (1 + combined).cumprod()

        # Metrics from combined
        ann_ret = combined.mean() * 8760
        ann_vol = combined.std() * np.sqrt(8760)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cum_max = combined_eq.cummax()
        dd = ((combined_eq - cum_max) / cum_max).min() * 100

        # Correlation matrix
        ret_df = pd.DataFrame({name: fn(df) for name, fn in strategies.items()})
        # Actually compute returns for correlation
        ret_mat = pd.DataFrame()
        for name, fn in strategies.items():
            sig = fn(df)
            res = run_backtest(df, sig, CFG_10)
            ret_mat[name] = res['equity_curve'].pct_change().fillna(0)

        corr = ret_mat.corr()

        print(f"\n  {asset} — Equal-weight 5-strategy portfolio:")
        print(f"    Sharpe: {sharpe:.2f}, Max DD: {dd:.1f}%")
        print(f"    Ann Return: {ann_ret*100:.1f}%, Ann Vol: {ann_vol*100:.1f}%")

        print(f"\n    Return correlations:")
        for i, n1 in enumerate(strategies.keys()):
            row = "    "
            for j, n2 in enumerate(strategies.keys()):
                if j <= i:
                    row += f"  {corr.loc[n1, n2]:+.2f}"
                else:
                    row += "      "
            print(f"    {n1:<12} {row}")

    # ═══════════════════════════════════════════════════════════════
    # 5. QUALIFICATION SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  5. QUALIFICATION SUMMARY")
    print("=" * 80)

    for strat_name, sig_fn, params_label in [
        ("Donchian Breakout",
         lambda d: donchian_vol(d, 480, 2.0, 24),
         "entry_period=480, vol_mult=2.0, hold=24"),
        ("Green Momentum",
         lambda d: green_momentum(d, 4, 4.0, 16),
         "min_green=4, gain=4%, hold=16"),
    ]:
        print(f"\n  {strat_name} ({params_label}):")
        for asset, df in data.items():
            r = evaluate(sig_fn, df, CFG_10)
            qual = "PASS" if (
                r['sharpe'] > 0
                and r['trades'] >= 20
                and r['frac_pos'] >= 0.50
            ) else "FAIL"
            print(f"    {asset}: Sharpe {r['sharpe']:+.2f}, "
                  f"Trades {r['trades']}, "
                  f"WR {r['wr']:.0f}%, "
                  f"Windows {r['frac_pos']*100:.0f}% pos → {qual}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 80)
