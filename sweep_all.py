#!/usr/bin/env python3
"""Run parameter sweeps for ALL strategies across ALL timeframes.

Outputs a comprehensive JSON results file for analysis.
"""

import importlib
import json
import sys
import time
from pathlib import Path

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.strategy import Strategy
from backtester.sweep import run_sweep

DATA_PATH = Path("data/binance_BTCUSDT_1m_klines.csv")

# All strategies to test
STRATEGIES = [
    # Existing
    "rsi", "ema_crossover", "macd", "bollinger_bounce", "bb_squeeze",
    "atr_breakout", "rsi_ema", "rsi_bollinger", "dual_rsi",
    # New
    "supertrend", "donchian", "keltner", "zscore_mr", "momentum_roc",
    "vwap", "ichimoku", "triple_ema", "macd_histogram", "volume_breakout",
]

# Timeframes: intraday + swing
TIMEFRAMES = {
    "intraday": ["5min", "15min", "1h"],
    "swing": ["4h", "1D"],
}

def discover_strategy(name: str):
    module = importlib.import_module(f"strategies.{name}_strategy")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Strategy) and attr is not Strategy:
            return attr
    raise ValueError(f"No Strategy subclass found in strategies/{name}_strategy.py")


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # Pre-load data at 1min, then resample per timeframe
    print("Loading base 1-min data...", file=sys.stderr)
    all_results = {}

    for category, timeframes in TIMEFRAMES.items():
        for tf in timeframes:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"TIMEFRAME: {tf} ({category})", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            df = load_candles(str(DATA_PATH), start="2025-01-01", end="2025-12-31", resample=tf)
            print(f"  Loaded {len(df)} bars", file=sys.stderr)

            for strat_name in STRATEGIES:
                key = f"{strat_name}__{tf}"
                try:
                    strat_class = discover_strategy(strat_name)
                    instance = strat_class()
                    grid = instance.param_grid()

                    # Count combos
                    combos = 1
                    for v in grid.values():
                        combos *= len(v)

                    print(f"  {strat_name} ({combos} combos)...", file=sys.stderr, end=" ", flush=True)
                    t0 = time.time()

                    result = run_sweep(strat_class, df, grid, config=config,
                                       rank_by="sharpe_ratio", top=3)

                    elapsed = time.time() - t0
                    print(f"done in {elapsed:.1f}s", file=sys.stderr)

                    # Store top 3 results
                    all_results[key] = {
                        "strategy": strat_name,
                        "timeframe": tf,
                        "category": category,
                        "total_combos": result["total_combos"],
                        "top_results": result["results"],
                    }
                except Exception as e:
                    print(f"FAILED: {e}", file=sys.stderr)
                    all_results[key] = {
                        "strategy": strat_name,
                        "timeframe": tf,
                        "category": category,
                        "error": str(e),
                    }

    # Output JSON
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
