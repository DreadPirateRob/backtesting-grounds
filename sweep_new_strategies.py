#!/usr/bin/env python3
"""Sweep new Tier 1 strategies across multiple timeframes on 2025 data."""

import json
import sys
import time
from pathlib import Path

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.sweep import run_sweep

DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"
TIMEFRAMES = ["5min", "15min", "1h", "4h"]
START = "2025-01-01"
END = "2025-12-31"

STRATEGIES = {
    "session_vwap": {
        "module": "strategies.session_vwap_strategy",
        "class": "SessionVwapStrategy",
        "extra_columns": None,
    },
    "cvd_divergence": {
        "module": "strategies.cvd_divergence_strategy",
        "class": "CvdDivergenceStrategy",
        "extra_columns": ["taker_buy_base_volume"],
    },
    "session_momentum": {
        "module": "strategies.session_momentum_strategy",
        "class": "SessionMomentumStrategy",
        "extra_columns": None,
    },
}


def main():
    config = BacktestConfig()
    all_results = {}
    total_start = time.time()

    for tf in TIMEFRAMES:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"TIMEFRAME: {tf}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Load data for this timeframe (with and without extra columns)
        t0 = time.time()
        df = load_candles(DATA_PATH, start=START, end=END, resample=tf)
        df_cvd = load_candles(
            DATA_PATH, start=START, end=END, resample=tf,
            extra_columns=["taker_buy_base_volume"],
        )
        load_time = time.time() - t0
        print(f"  Loaded {len(df)} bars in {load_time:.1f}s", file=sys.stderr)

        for strat_name, strat_info in STRATEGIES.items():
            import importlib
            mod = importlib.import_module(strat_info["module"])
            strat_class = getattr(mod, strat_info["class"])

            data = df_cvd if strat_info["extra_columns"] else df
            grid = strat_class().param_grid()

            combos = 1
            for v in grid.values():
                combos *= len(v)

            print(f"\n  --- {strat_name} @ {tf} ({combos} combos) ---", file=sys.stderr)
            t0 = time.time()

            try:
                sweep_result = run_sweep(
                    strat_class, data, grid, config,
                    rank_by="sharpe_ratio", top=5,
                )
                elapsed = time.time() - t0
                print(f"  Completed in {elapsed:.1f}s ({sweep_result['completed']}/{sweep_result['total_combos']})", file=sys.stderr)

                # Print top result
                if sweep_result["results"]:
                    best = sweep_result["results"][0]
                    m = best["metrics"]
                    print(f"  Best: Sharpe={m['sharpe_ratio']}, Return={m['total_return_pct']}%, "
                          f"Trades={m['total_trades']}, WR={m['win_rate_pct']}%", file=sys.stderr)
                    print(f"  Params: {best['params']}", file=sys.stderr)

                key = f"{strat_name}_{tf}"
                all_results[key] = {
                    "strategy": strat_name,
                    "timeframe": tf,
                    "bars": len(data),
                    "total_combos": sweep_result["total_combos"],
                    "completed": sweep_result["completed"],
                    "elapsed_seconds": round(elapsed, 1),
                    "top_5": sweep_result["results"],
                }
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                all_results[f"{strat_name}_{tf}"] = {"error": str(e)}

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total sweep time: {total_elapsed:.0f}s", file=sys.stderr)

    # Save results
    output_path = Path("data/runs/sweep_new_strategies_2025.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}", file=sys.stderr)

    # Also print summary table
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"{'Strategy':<22} {'TF':>5} {'Sharpe':>8} {'Return%':>9} {'Trades':>7} {'WR%':>6}", file=sys.stderr)
    print(f"{'-'*22} {'-'*5} {'-'*8} {'-'*9} {'-'*7} {'-'*6}", file=sys.stderr)

    for key, data in sorted(all_results.items()):
        if "error" in data:
            print(f"{data.get('strategy','?'):<22} {data.get('timeframe','?'):>5} ERROR: {data['error']}", file=sys.stderr)
            continue
        if data["top_5"]:
            best = data["top_5"][0]["metrics"]
            print(f"{data['strategy']:<22} {data['timeframe']:>5} "
                  f"{best['sharpe_ratio']:>8.2f} {best['total_return_pct']:>9.2f} "
                  f"{best['total_trades']:>7} {best['win_rate_pct']:>6.1f}", file=sys.stderr)

    # Print JSON to stdout
    json.dump(all_results, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
