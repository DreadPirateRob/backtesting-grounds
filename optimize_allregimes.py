#!/usr/bin/env python3
"""Full-range optimization across all market regimes (2021-2026).

Phase 1: Sweep on full range for both strategies x both timeframes
Phase 2: Validate top params on each sub-period (bull, bear, sideways)
Phase 3: Score robustness (consistency across regimes)
"""

import time
import numpy as np
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.sweep import run_sweep
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy
from strategies.rsi_bb_mtf_strategy import RsiBbMtfStrategy


def to_float(v):
    try:
        return float(v) if v is not None else 0.0
    except:
        return 0.0


# Sub-periods representing different regimes
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


def validate_params(strategy_class, params, timeframe, config):
    """Run a single param set across all regimes and return per-regime metrics."""
    results = []
    for name, start, end, regime_type in REGIMES:
        df = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                          start=start, end=end, resample=timeframe)
        if len(df) < 50:
            continue
        strategy = strategy_class(**params)
        signals = strategy.generate_signals(df)
        bt = run_backtest(df, signals, config)
        m = compute_metrics(bt)
        results.append({
            "period": name,
            "regime": regime_type,
            "bars": len(df),
            "sharpe": to_float(m.get("sharpe_ratio")),
            "ret": to_float(m.get("total_return_pct")),
            "dd": to_float(m.get("max_drawdown_pct")),
            "wr": to_float(m.get("win_rate_pct")),
            "pf": to_float(m.get("profit_factor")),
            "trades": int(to_float(m.get("total_trades"))),
        })
    return results


def robustness_score(regime_results):
    """Score how consistent a strategy is across regimes.

    Factors:
    - % of periods with positive Sharpe (max 40 pts)
    - Average Sharpe across all periods (max 30 pts)
    - Worst single-period drawdown (max 15 pts)
    - % of periods with positive return (max 15 pts)
    """
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
    score += min(max(avg_sharpe / 3.0, 0), 1.0) * 30  # cap at sharpe=3
    score += min(max((worst_dd + 70) / 40, 0), 1.0) * 15  # -70%=0, -30%=max
    score += pct_positive_ret * 15

    return round(score, 1)


if __name__ == "__main__":
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # Wide grids for full-range optimization
    baseline_grid = {
        "rsi_period": [10, 14, 18, 24, 30],
        "overbought": [60, 65, 70, 75],
        "oversold": [25, 30, 35, 40],
        "bb_period": [15, 20, 25, 30],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
    }
    mtf_grid = {
        "rsi_period": [10, 14, 18, 24, 30],
        "overbought": [60, 65, 70, 75],
        "oversold": [25, 30, 35, 40],
        "bb_period": [15, 20, 25, 30],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
        "daily_ema": [15, 21, 30, 50],
    }

    for tf, tf_label in [("4h", "SWING"), ("1h", "INTRADAY")]:
        print("=" * 100)
        print(f"FULL-RANGE OPTIMIZATION: {tf_label} ({tf}) — 2021-2026")
        print("=" * 100)

        df = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample=tf)
        print(f"Data: {len(df)} bars ({tf}, {str(df.index[0])[:10]} to {str(df.index[-1])[:10]})")

        # --- Baseline RSI+BB ---
        print(f"\n[1/2] Baseline RSI+BB ({tf})...")
        t0 = time.time()
        r_base = run_sweep(RsiBollingerStrategy, df, baseline_grid, config=config,
                           rank_by="sharpe_ratio", top=15)
        print(f"  Done in {time.time()-t0:.1f}s ({r_base['total_combos']} combos)")
        print_results(f"BASELINE RSI+BB ({tf}) — Full Range — by Sharpe:", r_base, top_n=10)

        # Also rank by calmar
        r_base_cal = run_sweep(RsiBollingerStrategy, df, baseline_grid, config=config,
                               rank_by="calmar_ratio", top=10)
        print_results(f"\nBASELINE RSI+BB ({tf}) — Full Range — by Calmar:", r_base_cal, top_n=5)

        # --- MTF RSI+BB ---
        print(f"\n[2/2] MTF RSI+BB ({tf})...")
        t0 = time.time()
        r_mtf = run_sweep(RsiBbMtfStrategy, df, mtf_grid, config=config,
                          rank_by="sharpe_ratio", top=15)
        print(f"  Done in {time.time()-t0:.1f}s ({r_mtf['total_combos']} combos)")
        print_results(f"MTF RSI+BB ({tf}) — Full Range — by Sharpe:", r_mtf, top_n=10)

        r_mtf_cal = run_sweep(RsiBbMtfStrategy, df, mtf_grid, config=config,
                              rank_by="calmar_ratio", top=10)
        print_results(f"\nMTF RSI+BB ({tf}) — Full Range — by Calmar:", r_mtf_cal, top_n=5)

        # ============================================================
        # ROBUSTNESS VALIDATION — top 5 from each strategy
        # ============================================================
        print(f"\n{'=' * 100}")
        print(f"ROBUSTNESS VALIDATION: {tf_label} ({tf})")
        print(f"{'=' * 100}")

        candidates = []

        # Top 5 baseline by Sharpe
        for r in r_base["results"][:5]:
            candidates.append(("Baseline", RsiBollingerStrategy, r["params"], r["metrics"]))

        # Top 5 MTF by Sharpe
        for r in r_mtf["results"][:5]:
            candidates.append(("MTF", RsiBbMtfStrategy, r["params"], r["metrics"]))

        # Top 3 baseline by Calmar
        for r in r_base_cal["results"][:3]:
            p = r["params"]
            if not any(c[3] == r["metrics"] for c in candidates):
                candidates.append(("Baseline", RsiBollingerStrategy, p, r["metrics"]))

        # Top 3 MTF by Calmar
        for r in r_mtf_cal["results"][:3]:
            p = r["params"]
            if not any(c[3] == r["metrics"] for c in candidates):
                candidates.append(("MTF", RsiBbMtfStrategy, p, r["metrics"]))

        print(f"\nValidating {len(candidates)} candidate param sets across {len(REGIMES)} market regimes...\n")

        scored = []
        for strat_name, strat_class, params, full_metrics in candidates:
            regime_results = validate_params(strat_class, params, tf, config)
            rob_score = robustness_score(regime_results)
            scored.append({
                "strategy": strat_name,
                "params": params,
                "full_sharpe": to_float(full_metrics.get("sharpe_ratio")),
                "full_ret": to_float(full_metrics.get("total_return_pct")),
                "full_dd": to_float(full_metrics.get("max_drawdown_pct")),
                "full_trades": int(to_float(full_metrics.get("total_trades"))),
                "robustness": rob_score,
                "regimes": regime_results,
            })

        # Sort by robustness score
        scored.sort(key=lambda x: x["robustness"], reverse=True)

        print(f"{'#':<3} {'Type':<9} {'Rob':>5} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8} {'Trd':>5}  Params")
        print("-" * 120)
        for i, s in enumerate(scored[:15]):
            params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            print(f"#{i+1:<2} {s['strategy']:<9} {s['robustness']:>5.1f} {s['full_sharpe']:>7.2f} "
                  f"{s['full_ret']:>7.1f}% {s['full_dd']:>7.1f}% {s['full_trades']:>5}  {params_str}")

        # Detailed regime breakdown for top 3
        print(f"\n--- REGIME BREAKDOWN: Top 3 most robust ({tf}) ---")
        for i, s in enumerate(scored[:3]):
            params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            print(f"\n#{i+1} {s['strategy']} (Rob={s['robustness']:.1f}): {params_str}")
            print(f"  {'Period':<22} {'Regime':<10} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8} {'WinR%':>7} {'PF':>6} {'Trd':>5}")
            print(f"  {'-' * 80}")
            for r in s["regimes"]:
                print(f"  {r['period']:<22} {r['regime']:<10} {r['sharpe']:>7.2f} {r['ret']:>7.1f}% "
                      f"{r['dd']:>7.1f}% {r['wr']:>6.1f}% {r['pf']:>6.2f} {r['trades']:>5}")
            # Summary stats
            sharpes = [r["sharpe"] for r in s["regimes"]]
            rets = [r["ret"] for r in s["regimes"]]
            dds = [r["dd"] for r in s["regimes"]]
            pos_periods = sum(1 for r in rets if r > 0)
            print(f"  {'SUMMARY':<22} {'':10} {np.mean(sharpes):>7.2f} {sum(rets):>7.1f}% "
                  f"{min(dds):>7.1f}% {pos_periods}/{len(s['regimes'])} profitable periods")

        print()

    # ============================================================
    # LONG-ONLY VARIANTS
    # ============================================================
    print("\n" + "=" * 100)
    print("LONG-ONLY OPTIMIZATION (for one-directional accounts)")
    print("=" * 100)

    config_long = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005, long_only=True)

    for tf, tf_label in [("4h", "SWING"), ("1h", "INTRADAY")]:
        df = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample=tf)
        print(f"\n{tf_label} ({tf}) — Long-Only:")

        r_mtf_long = run_sweep(RsiBbMtfStrategy, df, mtf_grid, config=config_long,
                               rank_by="sharpe_ratio", top=5)
        print_results(f"MTF RSI+BB Long-Only ({tf}):", r_mtf_long, top_n=5)

        r_base_long = run_sweep(RsiBollingerStrategy, df, baseline_grid, config=config_long,
                                rank_by="sharpe_ratio", top=5)
        print_results(f"Baseline RSI+BB Long-Only ({tf}):", r_base_long, top_n=5)

    print("\n\nDONE!")
