#!/usr/bin/env python3
"""Cross-asset sweep: find MTF RSI+BB params that work across BTC, ETH, SOL.

Goal: find "fluent" params that are profitable across multiple assets and timeframes,
not just overfit to a single pair.

Approach:
1. Sweep each asset independently on full range
2. For the top N params per asset, cross-validate on other assets
3. Score by average robustness across all assets
4. Report the most "universal" param sets
"""

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


ASSETS = [
    ("BTCUSDT", "data/binance_BTCUSDT_1m_klines.csv"),
    ("ETHUSDT", "data/binance_ETHUSDT_1m_klines.csv"),
    ("SOLUSDT", "data/binance_SOLUSDT_1m_klines.csv"),
]

# Sub-periods for robustness validation
REGIMES = [
    ("2022-H1 Bear",       "2022-01-01", "2022-06-30"),
    ("2022-H2 Bottom",     "2022-07-01", "2022-12-31"),
    ("2023 Recovery",      "2023-01-01", "2023-12-31"),
    ("2024-H1 Bull",       "2024-01-01", "2024-06-30"),
    ("2024-H2 Consolidate","2024-07-01", "2024-12-31"),
    ("2025 Sideways",      "2025-01-01", "2025-12-31"),
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


def run_single(params, data_path, timeframe, config):
    """Run a single backtest and return metrics dict."""
    df = load_candles(data_path, resample=timeframe)
    strategy = RsiBbMtfStrategy(**params)
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


def run_regimes(params, data_path, timeframe, config):
    """Run across sub-periods and return list of per-regime metrics."""
    results = []
    for name, start, end in REGIMES:
        try:
            df = load_candles(data_path, start=start, end=end, resample=timeframe)
            if len(df) < 50:
                continue
            strategy = RsiBbMtfStrategy(**params)
            signals = strategy.generate_signals(df)
            bt = run_backtest(df, signals, config)
            m = compute_metrics(bt)
            results.append({
                "period": name,
                "sharpe": to_float(m.get("sharpe_ratio")),
                "ret": to_float(m.get("total_return_pct")),
                "dd": to_float(m.get("max_drawdown_pct")),
            })
        except Exception as e:
            results.append({"period": name, "sharpe": 0, "ret": 0, "dd": -100})
    return results


def cross_asset_score(params, timeframe, config):
    """Score a param set across all assets.

    Returns a dict with per-asset metrics and an aggregate fluency score.
    Fluency = how well it works everywhere, not just one asset.
    """
    asset_results = {}
    all_sharpes = []
    all_rets = []
    all_dds = []
    all_regime_sharpes = []

    for name, path in ASSETS:
        try:
            full = run_single(params, path, timeframe, config)
            regimes = run_regimes(params, path, timeframe, config)
            asset_results[name] = {"full": full, "regimes": regimes}
            all_sharpes.append(full["sharpe"])
            all_rets.append(full["ret"])
            all_dds.append(full["dd"])
            for r in regimes:
                all_regime_sharpes.append(r["sharpe"])
        except Exception as e:
            asset_results[name] = {"full": {"sharpe": 0, "ret": 0, "dd": -100}, "regimes": []}
            all_sharpes.append(0)
            all_rets.append(0)
            all_dds.append(-100)

    # Fluency score components:
    # 1. Min Sharpe across assets (30 pts) — worst-case performance
    # 2. Avg Sharpe across assets (20 pts) — overall quality
    # 3. % of regime periods with positive Sharpe across ALL assets (25 pts)
    # 4. Worst drawdown across all assets (15 pts)
    # 5. Consistency: 1 - std(sharpes)/mean(sharpes) if mean > 0 (10 pts)

    min_sharpe = min(all_sharpes) if all_sharpes else 0
    avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0
    worst_dd = min(all_dds) if all_dds else -100
    pct_pos_regime = (sum(1 for s in all_regime_sharpes if s > 0) / len(all_regime_sharpes)
                      if all_regime_sharpes else 0)

    score = 0
    score += min(max(min_sharpe / 1.5, 0), 1.0) * 30  # min sharpe across assets
    score += min(max(avg_sharpe / 2.0, 0), 1.0) * 20   # avg sharpe
    score += pct_pos_regime * 25                          # regime consistency
    score += min(max((worst_dd + 60) / 30, 0), 1.0) * 15  # drawdown
    # Consistency bonus
    if avg_sharpe > 0 and len(all_sharpes) > 1:
        cv = np.std(all_sharpes) / avg_sharpe  # coefficient of variation
        consistency = max(0, 1 - cv)
        score += consistency * 10

    return {
        "assets": asset_results,
        "min_sharpe": min_sharpe,
        "avg_sharpe": avg_sharpe,
        "worst_dd": worst_dd,
        "fluency": round(score, 1),
    }


if __name__ == "__main__":
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # Grid covering the space informed by per-asset winners
    grid = {
        "rsi_period": [10, 14, 16, 20, 24, 26, 30],
        "overbought": [63, 65, 70, 75],
        "oversold": [25, 28, 30, 35],
        "bb_period": [12, 15, 20, 25, 30],
        "bb_std": [2.0, 2.5, 2.75, 3.0],
        "daily_ema": [25, 30, 35, 40, 50],
    }

    for tf, tf_label in [("4h", "SWING"), ("1h", "INTRADAY")]:
        print("=" * 110)
        print(f"CROSS-ASSET SWEEP: {tf_label} ({tf})")
        print("=" * 110)

        # Phase 1: Sweep each asset independently to find top params
        all_top_params = {}
        for asset_name, data_path in ASSETS:
            print(f"\n--- {asset_name} ({tf}) ---")
            t0 = time.time()
            df = load_candles(data_path, resample=tf)
            print(f"  Data: {len(df)} bars ({str(df.index[0])[:10]} to {str(df.index[-1])[:10]})")

            r = run_sweep(RsiBbMtfStrategy, df, grid, config=config,
                          rank_by="sharpe_ratio", top=10)
            print(f"  Done in {time.time()-t0:.1f}s ({r['total_combos']} combos)")
            print_results(f"  {asset_name} ({tf}) — Top 10 by Sharpe:", r, top_n=10)
            all_top_params[asset_name] = r["results"][:10]

        # Phase 2: Collect unique param sets from all assets' top 10
        unique_params = {}
        for asset_name, results in all_top_params.items():
            for r in results:
                key = str(sorted(r["params"].items()))
                if key not in unique_params:
                    unique_params[key] = r["params"]

        print(f"\n{'=' * 110}")
        print(f"CROSS-ASSET VALIDATION: {tf_label} ({tf}) — {len(unique_params)} unique param sets")
        print(f"{'=' * 110}")

        # Phase 3: Score each param set across all assets
        scored = []
        for i, (key, params) in enumerate(unique_params.items()):
            result = cross_asset_score(params, tf, config)
            result["params"] = params
            scored.append(result)
            if (i + 1) % 5 == 0:
                print(f"  Validated {i+1}/{len(unique_params)} param sets...")

        scored.sort(key=lambda x: x["fluency"], reverse=True)

        # Print ranked results
        print(f"\n{'#':<3} {'Fluency':>7}  {'BTC':>7} {'ETH':>7} {'SOL':>7}  {'AvgSh':>6} {'MinSh':>6} {'WrstDD':>7}  Params")
        print("-" * 120)
        for i, s in enumerate(scored[:20]):
            params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            btc_sh = s["assets"].get("BTCUSDT", {}).get("full", {}).get("sharpe", 0)
            eth_sh = s["assets"].get("ETHUSDT", {}).get("full", {}).get("sharpe", 0)
            sol_sh = s["assets"].get("SOLUSDT", {}).get("full", {}).get("sharpe", 0)
            print(f"#{i+1:<2} {s['fluency']:>7.1f}  {btc_sh:>7.2f} {eth_sh:>7.2f} {sol_sh:>7.2f}  "
                  f"{s['avg_sharpe']:>6.2f} {s['min_sharpe']:>6.2f} {s['worst_dd']:>6.1f}%  {params_str}")

        # Detailed breakdown for top 3
        print(f"\n{'=' * 110}")
        print(f"DETAILED BREAKDOWN: Top 3 most fluent ({tf})")
        print(f"{'=' * 110}")

        for rank, s in enumerate(scored[:3], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            print(f"\n--- #{rank} Fluency={s['fluency']:.1f}: {params_str} ---")

            for asset_name in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                a = s["assets"].get(asset_name, {})
                f_data = a.get("full", {})
                print(f"\n  {asset_name} (full range):")
                print(f"    Sharpe={f_data.get('sharpe',0):.2f}  Ret={f_data.get('ret',0):.1f}%  "
                      f"MaxDD={f_data.get('dd',0):.1f}%  WinR={f_data.get('wr',0):.1f}%  "
                      f"PF={f_data.get('pf',0):.2f}  Trades={f_data.get('trades',0)}")

                regimes = a.get("regimes", [])
                if regimes:
                    print(f"    {'Period':<22} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8}")
                    for r in regimes:
                        print(f"    {r['period']:<22} {r['sharpe']:>7.2f} {r['ret']:>7.1f}% {r['dd']:>7.1f}%")
                    pos = sum(1 for r in regimes if r["ret"] > 0)
                    print(f"    -> {pos}/{len(regimes)} profitable periods")

        print()

    print("\n" + "=" * 110)
    print("DONE!")
    print("=" * 110)
