#!/usr/bin/env python3
"""Run parameter sweeps across all historical BTC regimes for top strategies.

Produces:
- Per-regime x strategy x timeframe top-3 results
- Summary table printed to stdout
- Full results JSON at data/runs/regime_sweep_full.json
- Strategy-Regime matrix matching KNOWLEDGE.md format
"""

import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.strategy import Strategy
from backtester.sweep import run_sweep

DATA_PATH = Path("data/binance_BTCUSDT_1m_klines.csv")
OUTPUT_PATH = Path("data/runs/regime_sweep_full.json")

REGIMES: list[tuple[str, str, str, str]] = [
    ("2021-H1 Bull",        "2021-02-20", "2021-06-30", "bull"),
    ("2021-H2 Recovery",    "2021-07-01", "2021-12-31", "bull"),
    ("2022-H1 Bear",        "2022-01-01", "2022-06-30", "bear"),
    ("2022-H2 Bottom",      "2022-07-01", "2022-12-31", "range_bound"),
    ("2023 Recovery",       "2023-01-01", "2023-12-31", "bull"),
    ("2024-H1 ETF Bull",    "2024-01-01", "2024-06-30", "bull"),
    ("2024-H2 Consolidate", "2024-07-01", "2024-12-31", "range_bound"),
    ("2025 Sideways",       "2025-01-01", "2025-12-31", "range_bound"),
]

STRATEGIES: dict[str, str] = {
    "rsi_bollinger":   "mean_reversion",
    "dual_rsi":        "mean_reversion",
    "cvd_divergence":  "volume",
    "ema_crossover":   "trend_following",
    "supertrend":      "trend_following",
    "macd":            "trend_following",
    "atr_breakout":    "breakout",
}

TIMEFRAMES = ["1h", "4h"]


def discover_strategy(name: str) -> type[Strategy]:
    module = importlib.import_module(f"strategies.{name}_strategy")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Strategy) and attr is not Strategy:
            return attr
    raise ValueError(f"No Strategy subclass in strategies/{name}_strategy.py")


def format_params(params: dict) -> str:
    parts = [f"{k}={v}" for k, v in params.items()]
    return "{" + ", ".join(parts) + "}"


def print_summary_table(all_results: list[dict]) -> None:
    header = (
        f"{'Regime':<25} {'Strategy':<18} {'TF':<4} "
        f"{'Sharpe':>7} {'Return%':>9} {'Trades':>6} {'Best Params'}"
    )
    print()
    print("=" * 120)
    print("REGIME SWEEP RESULTS")
    print("=" * 120)
    print(header)
    print("-" * 120)

    current_regime = ""
    for r in all_results:
        regime_label = r["regime_name"] if r["regime_name"] != current_regime else ""
        current_regime = r["regime_name"]
        print(
            f"{regime_label:<25} {r['strategy']:<18} {r['timeframe']:<4} "
            f"{r['best_sharpe']:>7.2f} {r['best_return']:>8.2f}% "
            f"{r['best_trades']:>6} {r['best_params']}"
        )


def print_family_winners(all_results: list[dict]) -> None:
    by_regime_type: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_regime_type[r["regime_type"]].append(r)

    print()
    print("=" * 80)
    print("BEST STRATEGY PER REGIME TYPE")
    print("=" * 80)

    for regime_type in ["bull", "bear", "range_bound"]:
        entries = by_regime_type.get(regime_type, [])
        if not entries:
            continue
        best = max(entries, key=lambda x: x["best_sharpe"])
        family = STRATEGIES.get(best["strategy"], "unknown")
        print(
            f"  {regime_type:<15} -> {best['strategy']} ({family}) "
            f"| Sharpe {best['best_sharpe']:.2f} | {best['timeframe']} "
            f"| {best['regime_name']}"
        )


def print_matrix(all_results: list[dict]) -> None:
    regime_types = ["bull", "bear", "range_bound"]
    families = sorted(set(STRATEGIES.values()))

    matrix: dict[str, dict[str, tuple[float, str]]] = {
        f: {rt: (float("-inf"), "") for rt in regime_types} for f in families
    }

    for r in all_results:
        family = STRATEGIES.get(r["strategy"], "unknown")
        rt = r["regime_type"]
        if rt in regime_types and family in matrix:
            if r["best_sharpe"] > matrix[family][rt][0]:
                matrix[family][rt] = (r["best_sharpe"], r["strategy"])

    print()
    print("=" * 80)
    print("STRATEGY-REGIME MATRIX (best Sharpe | strategy | confidence)")
    print("=" * 80)
    print(f"{'Family':<18} {'bull':>25} {'bear':>25} {'range_bound':>25}")
    print("-" * 95)

    for family in families:
        cells = []
        for rt in regime_types:
            sharpe, strat = matrix[family][rt]
            if sharpe == float("-inf"):
                cells.append("")
            else:
                n_regimes = sum(1 for _, _, _, rtype in REGIMES if rtype == rt)
                conf = "L" if n_regimes <= 3 else ("M" if n_regimes <= 7 else "H")
                cells.append(f"{sharpe:.2f} | {strat} | {conf}")
        print(f"{family:<18} {cells[0]:>25} {cells[1]:>25} {cells[2]:>25}")


def main() -> None:
    config = BacktestConfig(
        initial_capital=10_000,
        fee_rate=0.001,
        slippage_pct=0.0005,
    )

    print("Loading base 1-min data...", file=sys.stderr)
    t_load = time.time()
    df_base = load_candles(str(DATA_PATH))
    print(
        f"Loaded {len(df_base)} bars in {time.time() - t_load:.1f}s",
        file=sys.stderr,
    )

    strat_classes: dict[str, type[Strategy]] = {}
    for name in STRATEGIES:
        strat_classes[name] = discover_strategy(name)

    all_results: list[dict] = []
    full_results: dict[str, dict] = {}

    total_tasks = len(REGIMES) * len(STRATEGIES) * len(TIMEFRAMES)
    completed = 0

    for regime_name, start, end, regime_type in REGIMES:
        print(
            f"\n{'='*60}\n"
            f"REGIME: {regime_name} ({regime_type}) [{start} to {end}]\n"
            f"{'='*60}",
            file=sys.stderr,
        )

        for tf in TIMEFRAMES:
            print(f"\n  Timeframe: {tf}", file=sys.stderr)
            df = load_candles(str(DATA_PATH), start=start, end=end, resample=tf)
            print(f"  {len(df)} bars", file=sys.stderr)

            if len(df) < 50:
                print(f"  SKIP: too few bars ({len(df)})", file=sys.stderr)
                completed += len(STRATEGIES)
                continue

            for strat_name in STRATEGIES:
                completed += 1
                strat_class = strat_classes[strat_name]
                instance = strat_class()
                grid = instance.param_grid()

                combo_count = 1
                for v in grid.values():
                    combo_count *= len(v)

                print(
                    f"    [{completed}/{total_tasks}] {strat_name} "
                    f"({combo_count} combos)...",
                    file=sys.stderr,
                    end=" ",
                    flush=True,
                )

                t0 = time.time()
                try:
                    sweep_result = run_sweep(
                        strat_class, df, grid,
                        config=config,
                        rank_by="sharpe_ratio",
                        top=3,
                    )
                    elapsed = time.time() - t0
                    print(f"done in {elapsed:.1f}s", file=sys.stderr)

                    top_results = sweep_result["results"]
                    best = top_results[0] if top_results else None

                    key = f"{regime_name}__{strat_name}__{tf}"
                    full_results[key] = {
                        "regime_name": regime_name,
                        "regime_type": regime_type,
                        "start": start,
                        "end": end,
                        "strategy": strat_name,
                        "family": STRATEGIES[strat_name],
                        "timeframe": tf,
                        "total_combos": sweep_result["total_combos"],
                        "completed": sweep_result["completed"],
                        "top_results": top_results,
                    }

                    best_sharpe = best["metrics"]["sharpe_ratio"] if best else 0.0
                    best_return = best["metrics"]["total_return_pct"] if best else 0.0
                    best_trades = best["metrics"]["total_trades"] if best else 0
                    best_params = format_params(best["params"]) if best else "{}"

                    all_results.append({
                        "regime_name": regime_name,
                        "regime_type": regime_type,
                        "strategy": strat_name,
                        "family": STRATEGIES[strat_name],
                        "timeframe": tf,
                        "best_sharpe": best_sharpe,
                        "best_return": best_return,
                        "best_trades": best_trades,
                        "best_params": best_params,
                    })

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"FAILED in {elapsed:.1f}s: {e}", file=sys.stderr)
                    key = f"{regime_name}__{strat_name}__{tf}"
                    full_results[key] = {
                        "regime_name": regime_name,
                        "regime_type": regime_type,
                        "strategy": strat_name,
                        "family": STRATEGIES[strat_name],
                        "timeframe": tf,
                        "error": str(e),
                    }
                    all_results.append({
                        "regime_name": regime_name,
                        "regime_type": regime_type,
                        "strategy": strat_name,
                        "family": STRATEGIES[strat_name],
                        "timeframe": tf,
                        "best_sharpe": 0.0,
                        "best_return": 0.0,
                        "best_trades": 0,
                        "best_params": "ERROR",
                    })

    print_summary_table(all_results)
    print_family_winners(all_results)
    print_matrix(all_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nFull results saved to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
