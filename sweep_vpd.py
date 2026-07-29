#!/usr/bin/env python3
"""Sweep Value-Price Divergence strategy across timeframes, regimes, and assets.

Phases:
1. Full-range sweep on BTC at 4h and 1h (top by Sharpe + Calmar)
2. Regime validation of top candidates across 8 sub-periods
3. Cross-asset validation on ETH and SOL
"""

import time
import numpy as np
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.sweep import run_sweep
from strategies.value_price_divergence_strategy import ValuePriceDivergenceStrategy


def to_float(v):
    try:
        return float(v) if v is not None else 0.0
    except:
        return 0.0


REGIMES = [
    ("2021-H1 Bull",        "2021-02-20", "2021-06-30", "BULL"),
    ("2021-H2 Recovery",    "2021-07-01", "2021-12-31", "BULL"),
    ("2022-H1 Bear",        "2022-01-01", "2022-06-30", "BEAR"),
    ("2022-H2 Bottom",      "2022-07-01", "2022-12-31", "SIDEWAYS"),
    ("2023 Recovery",       "2023-01-01", "2023-12-31", "BULL"),
    ("2024-H1 ETF Bull",    "2024-01-01", "2024-06-30", "BULL"),
    ("2024-H2 Consolidate", "2024-07-01", "2024-12-31", "BULL"),
    ("2025 Sideways",       "2025-01-01", "2025-12-31", "SIDEWAYS"),
]

ASSETS = [
    ("BTCUSDT", "data/binance_BTCUSDT_1m_klines.csv"),
    ("ETHUSDT", "data/binance_ETHUSDT_1m_klines.csv"),
    ("SOLUSDT", "data/binance_SOLUSDT_1m_klines.csv"),
]


def print_results(title, result, top_n=10):
    print(f"\n{title}")
    print(f"{'#':<4} {'Sharpe':>7} {'Sortino':>8} {'Return%':>9} {'MaxDD%':>8} {'WinR%':>7} {'PF':>7} {'Calmar':>7} {'Trd':>5}  Params")
    print("-" * 140)
    for r in result["results"][:top_n]:
        m = r["metrics"]
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"#{r['rank']:<3} {to_float(m.get('sharpe_ratio')):>7.2f} {to_float(m.get('sortino_ratio')):>8.2f} "
              f"{to_float(m.get('total_return_pct')):>8.1f}% {to_float(m.get('max_drawdown_pct')):>7.1f}% "
              f"{to_float(m.get('win_rate_pct')):>6.1f}% {to_float(m.get('profit_factor')):>7.2f} "
              f"{to_float(m.get('calmar_ratio')):>7.2f} {int(to_float(m.get('total_trades'))):>5}  {params_str}")


def needs_cvd(params):
    """Check if param set uses CVD (needs taker_buy_base_volume)."""
    return params.get("value_source", "obv") == "cvd"


def validate_regimes(params, timeframe, config, data_path="data/binance_BTCUSDT_1m_klines.csv"):
    """Run strategy across sub-periods and return per-regime metrics."""
    results = []
    use_cvd = needs_cvd(params)
    extra = ["taker_buy_base_volume"] if use_cvd else None
    for name, start, end, regime_type in REGIMES:
        try:
            df = load_candles(data_path, start=start, end=end, resample=timeframe,
                              extra_columns=extra)
            if len(df) < 50:
                continue
            strategy = ValuePriceDivergenceStrategy(**params)
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
        except Exception as e:
            results.append({
                "period": name, "regime": regime_type,
                "sharpe": 0, "ret": 0, "dd": -100, "wr": 0, "pf": 0, "trades": 0,
            })
    return results


def robustness_score(regime_results):
    """Score 0-100 based on regime consistency."""
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


def run_single(params, data_path, timeframe, config):
    """Run single backtest and return metrics dict."""
    use_cvd = needs_cvd(params)
    extra = ["taker_buy_base_volume"] if use_cvd else None
    df = load_candles(data_path, resample=timeframe, extra_columns=extra)
    strategy = ValuePriceDivergenceStrategy(**params)
    signals = strategy.generate_signals(df)
    bt = run_backtest(df, signals, config)
    m = compute_metrics(bt)
    return {
        "sharpe": to_float(m.get("sharpe_ratio")),
        "ret": to_float(m.get("total_return_pct")),
        "dd": to_float(m.get("max_drawdown_pct")),
        "wr": to_float(m.get("win_rate_pct")),
        "pf": to_float(m.get("profit_factor")),
        "calmar": to_float(m.get("calmar_ratio")),
        "trades": int(to_float(m.get("total_trades"))),
    }


if __name__ == "__main__":
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    grid = {
        "lookback": [10, 15, 20, 30, 50],
        "smooth": [5, 10, 15, 20],
        "z_window": [50, 100, 150],
        "entry_z": [1.5, 2.0, 2.5, 3.0],
        "exit_z": [0.0, 0.25, 0.5],
        "atr_stop_mult": [0.0, 2.0, 3.0],
        "value_source": ["obv", "cvd"],
    }
    combos = 1
    for v in grid.values():
        combos *= len(v)
    print(f"Total grid: {combos} combinations per timeframe")

    # ============================================================
    # PHASE 1: FULL-RANGE SWEEP (BTC, 4h + 1h)
    # ============================================================
    for tf, tf_label in [("4h", "SWING"), ("1h", "INTRADAY")]:
        print("\n" + "=" * 110)
        print(f"PHASE 1: {tf_label} ({tf}) — FULL-RANGE SWEEP")
        print("=" * 110)

        # Load both OBV and CVD variants (CVD needs taker_buy_base_volume)
        df = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample=tf,
                          extra_columns=["taker_buy_base_volume"])
        print(f"Data: {len(df)} bars ({str(df.index[0])[:10]} to {str(df.index[-1])[:10]})")

        t0 = time.time()
        r_sharpe = run_sweep(ValuePriceDivergenceStrategy, df, grid, config=config,
                             rank_by="sharpe_ratio", top=15)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s ({r_sharpe['total_combos']} combos)")
        print_results(f"VPD {tf_label} ({tf}) — Top 15 by Sharpe:", r_sharpe, top_n=15)

        r_calmar = run_sweep(ValuePriceDivergenceStrategy, df, grid, config=config,
                             rank_by="calmar_ratio", top=10)
        print_results(f"\nVPD {tf_label} ({tf}) — Top 10 by Calmar:", r_calmar, top_n=10)

        # ============================================================
        # PHASE 2: REGIME VALIDATION
        # ============================================================
        print(f"\n{'=' * 110}")
        print(f"PHASE 2: REGIME VALIDATION — {tf_label} ({tf})")
        print(f"{'=' * 110}")

        candidates = []
        seen = set()
        for r in r_sharpe["results"][:10]:
            key = str(sorted(r["params"].items()))
            if key not in seen:
                seen.add(key)
                candidates.append((r["params"], r["metrics"]))
        for r in r_calmar["results"][:7]:
            key = str(sorted(r["params"].items()))
            if key not in seen:
                seen.add(key)
                candidates.append((r["params"], r["metrics"]))

        print(f"Validating {len(candidates)} candidates across {len(REGIMES)} regimes...\n")
        scored = []
        for params, full_metrics in candidates:
            regime_results = validate_regimes(params, tf, config)
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
        print("-" * 120)
        for i, s in enumerate(scored[:10]):
            params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            print(f"#{i+1:<2} {s['robustness']:>5.1f} {s['full_sharpe']:>7.2f} "
                  f"{s['full_ret']:>7.1f}% {s['full_dd']:>7.1f}% {s['full_trades']:>5}  {params_str}")

        # Top 3 regime breakdown
        for i, s in enumerate(scored[:3]):
            params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            print(f"\n#{i+1} (Rob={s['robustness']:.1f}): {params_str}")
            print(f"  {'Period':<22} {'Regime':<10} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8} {'WinR%':>7} {'PF':>6} {'Trd':>5}")
            print(f"  {'-' * 85}")
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
        # PHASE 3: CROSS-ASSET VALIDATION (top 5 by robustness)
        # ============================================================
        print(f"\n{'=' * 110}")
        print(f"PHASE 3: CROSS-ASSET VALIDATION — {tf_label} ({tf})")
        print(f"{'=' * 110}")

        top_for_xasset = scored[:5]
        print(f"Testing top {len(top_for_xasset)} robust params on BTC, ETH, SOL...\n")

        print(f"{'#':<3} {'Rob':>5}  {'BTC':>7} {'ETH':>7} {'SOL':>7}  {'AvgSh':>6} {'MinSh':>6}  Params")
        print("-" * 120)
        for i, s in enumerate(top_for_xasset):
            params = s["params"]
            asset_sharpes = {}
            for asset_name, data_path in ASSETS:
                try:
                    m = run_single(params, data_path, tf, config)
                    asset_sharpes[asset_name] = m["sharpe"]
                except Exception:
                    asset_sharpes[asset_name] = 0.0

            sharpe_vals = list(asset_sharpes.values())
            avg_sh = np.mean(sharpe_vals)
            min_sh = min(sharpe_vals)
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"#{i+1:<2} {s['robustness']:>5.1f}  "
                  f"{asset_sharpes.get('BTCUSDT', 0):>7.2f} "
                  f"{asset_sharpes.get('ETHUSDT', 0):>7.2f} "
                  f"{asset_sharpes.get('SOLUSDT', 0):>7.2f}  "
                  f"{avg_sh:>6.2f} {min_sh:>6.2f}  {params_str}")

        # Detailed breakdown for top 1
        if top_for_xasset:
            best = top_for_xasset[0]
            params = best["params"]
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"\n--- BEST CANDIDATE: {params_str} ---")
            for asset_name, data_path in ASSETS:
                try:
                    m = run_single(params, data_path, tf, config)
                    print(f"\n  {asset_name} (full range):")
                    print(f"    Sharpe={m['sharpe']:.2f}  Ret={m['ret']:.1f}%  "
                          f"MaxDD={m['dd']:.1f}%  WinR={m['wr']:.1f}%  "
                          f"PF={m['pf']:.2f}  Trades={m['trades']}")
                except Exception as e:
                    print(f"\n  {asset_name}: ERROR — {e}")

    print("\n\n" + "=" * 110)
    print("DONE!")
    print("=" * 110)
