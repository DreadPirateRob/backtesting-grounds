#!/usr/bin/env python3
"""Fine-tune the top full-range MTF params with tighter grids."""

import time
import numpy as np
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.sweep import run_sweep
from strategies.rsi_bb_mtf_strategy import RsiBbMtfStrategy


def to_float(v):
    try:
        return float(v) if v is not None else 0.0
    except:
        return 0.0


REGIMES = [
    ("2021-H1 Bull",       "2021-02-20", "2021-06-30", "BULL"),
    ("2021-H2 Recovery",   "2021-07-01", "2021-12-31", "BULL"),
    ("2022-H1 Bear",       "2022-01-01", "2022-06-30", "BEAR"),
    ("2022-H2 Bottom",     "2022-07-01", "2022-12-31", "SIDEWAYS"),
    ("2023 Recovery",      "2023-01-01", "2023-12-31", "BULL"),
    ("2024-H1 ETF Bull",   "2024-01-01", "2024-06-30", "BULL"),
    ("2024-H2 Consolidate","2024-07-01", "2024-12-31", "BULL"),
    ("2025 Sideways",      "2025-01-01", "2025-12-31", "SIDEWAYS"),
]


def print_results(title, result, top_n=10):
    print(f"\n{title}")
    print(f"{'#':<4} {'Sharpe':>7} {'Sortino':>8} {'Return%':>9} {'MaxDD%':>8} {'WinR%':>7} {'PF':>7} {'Calmar':>7} {'Trd':>5}  Params")
    print("-" * 130)
    for r in result["results"][:top_n]:
        m = r["metrics"]
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"#{r['rank']:<3} {to_float(m.get('sharpe_ratio')):>7.2f} {to_float(m.get('sortino_ratio')):>8.2f} "
              f"{to_float(m.get('total_return_pct')):>8.1f}% {to_float(m.get('max_drawdown_pct')):>7.1f}% "
              f"{to_float(m.get('win_rate_pct')):>6.1f}% {to_float(m.get('profit_factor')):>7.2f} "
              f"{to_float(m.get('calmar_ratio')):>7.2f} {int(to_float(m.get('total_trades'))):>5}  {params_str}")


def validate_params(params, timeframe, config):
    results = []
    for name, start, end, regime_type in REGIMES:
        df = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                          start=start, end=end, resample=timeframe)
        if len(df) < 50:
            continue
        strategy = RsiBbMtfStrategy(**params)
        signals = strategy.generate_signals(df)
        bt = run_backtest(df, signals, config)
        m = compute_metrics(bt)
        results.append({
            "period": name,
            "regime": regime_type,
            "sharpe": to_float(m.get("sharpe_ratio")),
            "ret": to_float(m.get("total_return_pct")),
            "dd": to_float(m.get("max_drawdown_pct")),
            "wr": to_float(m.get("win_rate_pct")),
            "pf": to_float(m.get("profit_factor")),
            "trades": int(to_float(m.get("total_trades"))),
        })
    return results


def robustness_score(regime_results):
    if not regime_results:
        return 0.0
    sharpes = [r["sharpe"] for r in regime_results]
    rets = [r["ret"] for r in regime_results]
    dds = [r["dd"] for r in regime_results]

    pct_positive_sharpe = sum(1 for s in sharpes if s > 0) / len(sharpes)
    avg_sharpe = np.mean(sharpes)
    worst_dd = min(dds)
    pct_positive_ret = sum(1 for r in rets if r > 0) / len(rets)

    score = 0
    score += pct_positive_sharpe * 40
    score += min(max(avg_sharpe / 3.0, 0), 1.0) * 30
    score += min(max((worst_dd + 70) / 40, 0), 1.0) * 15
    score += pct_positive_ret * 15
    return round(score, 1)


if __name__ == "__main__":
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # ============================================================
    # SWING 4h — fine-tune around winning area
    # Winner: rsi=14, ob=65, os=30, bb=30, bb_std=2.5, daily_ema=30
    # ============================================================
    print("=" * 100)
    print("FINE-TUNING MTF RSI+BB: SWING (4h) — Full Range")
    print("=" * 100)

    df_4h = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample="4h")
    print(f"Data: {len(df_4h)} bars")

    grid_4h = {
        "rsi_period": [10, 12, 14, 16, 18],
        "overbought": [60, 63, 65, 67, 70],
        "oversold": [25, 28, 30, 32, 35],
        "bb_period": [22, 25, 28, 30, 33],
        "bb_std": [2.0, 2.25, 2.5, 2.75, 3.0],
        "daily_ema": [21, 25, 30, 35, 40, 50],
    }
    combos = 1
    for v in grid_4h.values():
        combos *= len(v)
    print(f"Testing {combos} combinations...")
    t0 = time.time()

    r_4h = run_sweep(RsiBbMtfStrategy, df_4h, grid_4h, config=config,
                     rank_by="sharpe_ratio", top=15)
    print(f"Done in {time.time()-t0:.1f}s")
    print_results("MTF SWING (4h) — Full Range — by Sharpe:", r_4h, top_n=15)

    r_4h_cal = run_sweep(RsiBbMtfStrategy, df_4h, grid_4h, config=config,
                         rank_by="calmar_ratio", top=10)
    print_results("\nMTF SWING (4h) — Full Range — by Calmar:", r_4h_cal, top_n=10)

    # Validate top 5 Sharpe + top 3 Calmar
    print(f"\n--- ROBUSTNESS VALIDATION: SWING (4h) ---")
    candidates = []
    seen = set()
    for r in r_4h["results"][:7]:
        key = str(r["params"])
        if key not in seen:
            seen.add(key)
            candidates.append((r["params"], r["metrics"]))
    for r in r_4h_cal["results"][:5]:
        key = str(r["params"])
        if key not in seen:
            seen.add(key)
            candidates.append((r["params"], r["metrics"]))

    print(f"Validating {len(candidates)} candidates across {len(REGIMES)} regimes...\n")
    scored = []
    for params, full_metrics in candidates:
        regime_results = validate_params(params, "4h", config)
        rob_score = robustness_score(regime_results)
        scored.append({
            "params": params,
            "full_sharpe": to_float(full_metrics.get("sharpe_ratio")),
            "full_ret": to_float(full_metrics.get("total_return_pct")),
            "full_dd": to_float(full_metrics.get("max_drawdown_pct")),
            "full_trades": int(to_float(full_metrics.get("total_trades"))),
            "robustness": rob_score,
            "regimes": regime_results,
        })

    scored.sort(key=lambda x: x["robustness"], reverse=True)

    print(f"{'#':<3} {'Rob':>5} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8} {'Trd':>5}  Params")
    print("-" * 110)
    for i, s in enumerate(scored[:10]):
        params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
        print(f"#{i+1:<2} {s['robustness']:>5.1f} {s['full_sharpe']:>7.2f} "
              f"{s['full_ret']:>7.1f}% {s['full_dd']:>7.1f}% {s['full_trades']:>5}  {params_str}")

    # Top 3 breakdown
    for i, s in enumerate(scored[:3]):
        params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
        print(f"\n#{i+1} (Rob={s['robustness']:.1f}): {params_str}")
        print(f"  {'Period':<22} {'Regime':<10} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8} {'WinR%':>7} {'PF':>6} {'Trd':>5}")
        print(f"  {'-' * 80}")
        for r in s["regimes"]:
            print(f"  {r['period']:<22} {r['regime']:<10} {r['sharpe']:>7.2f} {r['ret']:>7.1f}% "
                  f"{r['dd']:>7.1f}% {r['wr']:>6.1f}% {r['pf']:>6.2f} {r['trades']:>5}")
        sharpes = [r["sharpe"] for r in s["regimes"]]
        rets = [r["ret"] for r in s["regimes"]]
        dds = [r["dd"] for r in s["regimes"]]
        pos = sum(1 for r in rets if r > 0)
        print(f"  {'SUMMARY':<22} {'':10} {np.mean(sharpes):>7.2f} {sum(rets):>7.1f}% "
              f"{min(dds):>7.1f}% {pos}/{len(s['regimes'])} profitable")

    # ============================================================
    # INTRADAY 1h — fine-tune around winning area
    # Winner: rsi=24, ob=75, os=30, bb=15, bb_std=2.5, daily_ema=30
    # ============================================================
    print("\n\n" + "=" * 100)
    print("FINE-TUNING MTF RSI+BB: INTRADAY (1h) — Full Range")
    print("=" * 100)

    df_1h = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample="1h")
    print(f"Data: {len(df_1h)} bars")

    grid_1h = {
        "rsi_period": [20, 22, 24, 26, 28, 30],
        "overbought": [68, 70, 73, 75, 78],
        "oversold": [25, 28, 30, 32, 35],
        "bb_period": [12, 15, 18, 20, 25],
        "bb_std": [2.0, 2.25, 2.5, 2.75, 3.0],
        "daily_ema": [21, 25, 30, 35, 40, 50],
    }
    combos = 1
    for v in grid_1h.values():
        combos *= len(v)
    print(f"Testing {combos} combinations...")
    t0 = time.time()

    r_1h = run_sweep(RsiBbMtfStrategy, df_1h, grid_1h, config=config,
                     rank_by="sharpe_ratio", top=15)
    print(f"Done in {time.time()-t0:.1f}s")
    print_results("MTF INTRADAY (1h) — Full Range — by Sharpe:", r_1h, top_n=15)

    r_1h_cal = run_sweep(RsiBbMtfStrategy, df_1h, grid_1h, config=config,
                         rank_by="calmar_ratio", top=10)
    print_results("\nMTF INTRADAY (1h) — Full Range — by Calmar:", r_1h_cal, top_n=10)

    # Validate
    print(f"\n--- ROBUSTNESS VALIDATION: INTRADAY (1h) ---")
    candidates = []
    seen = set()
    for r in r_1h["results"][:7]:
        key = str(r["params"])
        if key not in seen:
            seen.add(key)
            candidates.append((r["params"], r["metrics"]))
    for r in r_1h_cal["results"][:5]:
        key = str(r["params"])
        if key not in seen:
            seen.add(key)
            candidates.append((r["params"], r["metrics"]))

    print(f"Validating {len(candidates)} candidates across {len(REGIMES)} regimes...\n")
    scored = []
    for params, full_metrics in candidates:
        regime_results = validate_params(params, "1h", config)
        rob_score = robustness_score(regime_results)
        scored.append({
            "params": params,
            "full_sharpe": to_float(full_metrics.get("sharpe_ratio")),
            "full_ret": to_float(full_metrics.get("total_return_pct")),
            "full_dd": to_float(full_metrics.get("max_drawdown_pct")),
            "full_trades": int(to_float(full_metrics.get("total_trades"))),
            "robustness": rob_score,
            "regimes": regime_results,
        })

    scored.sort(key=lambda x: x["robustness"], reverse=True)

    print(f"{'#':<3} {'Rob':>5} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8} {'Trd':>5}  Params")
    print("-" * 110)
    for i, s in enumerate(scored[:10]):
        params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
        print(f"#{i+1:<2} {s['robustness']:>5.1f} {s['full_sharpe']:>7.2f} "
              f"{s['full_ret']:>7.1f}% {s['full_dd']:>7.1f}% {s['full_trades']:>5}  {params_str}")

    for i, s in enumerate(scored[:3]):
        params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
        print(f"\n#{i+1} (Rob={s['robustness']:.1f}): {params_str}")
        print(f"  {'Period':<22} {'Regime':<10} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8} {'WinR%':>7} {'PF':>6} {'Trd':>5}")
        print(f"  {'-' * 80}")
        for r in s["regimes"]:
            print(f"  {r['period']:<22} {r['regime']:<10} {r['sharpe']:>7.2f} {r['ret']:>7.1f}% "
                  f"{r['dd']:>7.1f}% {r['wr']:>6.1f}% {r['pf']:>6.2f} {r['trades']:>5}")
        sharpes = [r["sharpe"] for r in s["regimes"]]
        rets = [r["ret"] for r in s["regimes"]]
        dds = [r["dd"] for r in s["regimes"]]
        pos = sum(1 for r in rets if r > 0)
        print(f"  {'SUMMARY':<22} {'':10} {np.mean(sharpes):>7.2f} {sum(rets):>7.1f}% "
              f"{min(dds):>7.1f}% {pos}/{len(s['regimes'])} profitable")

    print("\n\nDONE!")
