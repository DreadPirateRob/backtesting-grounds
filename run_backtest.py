#!/usr/bin/env python3
"""CLI entry point for the backtesting framework.

Usage:
    # Single run with default params
    python3 run_backtest.py --strategy rsi --resample 1h

    # With specific params
    python3 run_backtest.py --strategy rsi --resample 1h --params '{"period":14,"overbought":75,"oversold":25}'

    # Parameter sweep
    python3 run_backtest.py --strategy rsi --resample 1h --sweep --rank-by sharpe_ratio --top 20

    # Date range + config
    python3 run_backtest.py --strategy rsi --resample 1h --start 2023-01-01 --end 2024-01-01 --fee 0.001

All results are output as JSON to stdout. Progress/errors go to stderr.
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.sweep import run_sweep


DATA_PATH = Path(__file__).parent / "data" / "binance_BTCUSDT_1m_klines.csv"


def discover_strategy(name: str):
    """Dynamically import a strategy class from strategies/{name}_strategy.py."""
    module_name = f"strategies.{name}_strategy"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Error: strategy module '{module_name}' not found.", file=sys.stderr)
        sys.exit(1)

    # Find the Strategy subclass in the module
    from backtester.strategy import Strategy
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, Strategy)
            and attr is not Strategy
        ):
            return attr

    print(f"Error: no Strategy subclass found in '{module_name}'.", file=sys.stderr)
    sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


W = 48  # column width for display


def _color(val: float, fmt: str = "+.2f", inverse: bool = False) -> str:
    """Color a number green/red based on sign. inverse flips (for drawdown)."""
    positive = val >= 0
    if inverse:
        positive = not positive
    color = "\033[32m" if positive else "\033[31m"
    reset = "\033[0m"
    return f"{color}{val:{fmt}}{reset}"


def _row(label: str, value: str) -> str:
    dots = "." * (W - len(label) - len(_strip_ansi(value)))
    return f"  {label} {dots} {value}"


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _header(title: str) -> str:
    pad = W - len(title) - 2
    return f"\033[1m{'─' * 2} {title} {'─' * max(0, pad)}\033[0m"


def _print_single_summary(strategy: str, params: dict, data_info: dict, metrics: dict, config: dict):
    """Print a formatted summary for a single backtest run."""
    err = sys.stderr
    m = metrics
    print("", file=err)
    print(_header(f"{strategy.upper()} Backtest"), file=err)

    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  {param_str}", file=err)
    start = data_info["start"][:10]
    end = data_info["end"][:10]
    tf = data_info.get("resample") or "1m"
    print(f"  {start} to {end}  |  {data_info['rows']:,} bars @ {tf}", file=err)

    # Returns
    print("", file=err)
    print(_header("Returns"), file=err)
    ret = m["total_return_pct"]
    ann = m["annualized_return_pct"]
    capital = config["initial_capital"]
    final = capital * (1 + ret / 100)
    print(_row("Total Return", f"{_color(ret)}%"), file=err)
    print(_row("Annualized", f"{_color(ann)}%"), file=err)
    print(_row("Capital", f"${capital:,.0f} -> ${final:,.0f}"), file=err)

    # Risk
    print("", file=err)
    print(_header("Risk"), file=err)
    print(_row("Sharpe Ratio", _color(m["sharpe_ratio"], "+.3f")), file=err)
    print(_row("Sortino Ratio", _color(m["sortino_ratio"], "+.3f")), file=err)
    dd = m["max_drawdown_pct"]
    print(_row("Max Drawdown", f"{_color(dd, '+.2f')}%"), file=err)
    dd_days = m["max_drawdown_duration_days"]
    print(_row("Max DD Duration", f"{dd_days:.0f} days"), file=err)
    print(_row("Calmar Ratio", _color(m["calmar_ratio"], "+.3f")), file=err)

    # Trades
    print("", file=err)
    print(_header("Trades"), file=err)
    print(_row("Total Trades", str(m["total_trades"])), file=err)
    wr = m["win_rate_pct"]
    wr_color = "\033[32m" if wr >= 50 else "\033[31m"
    print(_row("Win Rate", f"{wr_color}{wr:.1f}%\033[0m"), file=err)
    pf = m["profit_factor"]
    pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
    pf_val = float(pf) if isinstance(pf, (int, float)) else 0
    pf_color = "\033[32m" if pf_val >= 1 else "\033[31m"
    print(_row("Profit Factor", f"{pf_color}{pf_str}\033[0m"), file=err)
    print(_row("Avg Win", f"{_color(m['avg_win_pct'])}%"), file=err)
    print(_row("Avg Loss", f"{_color(m['avg_loss_pct'])}%"), file=err)
    print(_row("Best Trade", f"{_color(m['best_trade_pct'])}%"), file=err)
    print(_row("Worst Trade", f"{_color(m['worst_trade_pct'])}%"), file=err)

    # Exposure
    print("", file=err)
    print(_header("Exposure"), file=err)
    print(_row("Time in Market", f"{m['time_in_market_pct']:.1f}%"), file=err)
    print(_row("Fees Paid", f"${m['total_fees_paid']:,.2f}"), file=err)
    print("", file=err)


def _print_sweep_summary(strategy: str, data_info: dict, sweep_meta: dict, results: list, config: dict):
    """Print a formatted table for sweep results."""
    err = sys.stderr
    print("", file=err)
    print(_header(f"{strategy.upper()} Sweep Results"), file=err)

    start = data_info["start"][:10]
    end = data_info["end"][:10]
    tf = data_info.get("resample") or "1m"
    print(f"  {start} to {end}  |  {data_info['rows']:,} bars @ {tf}", file=err)
    print(f"  {sweep_meta['completed']}/{sweep_meta['total_combos']} combos in {sweep_meta['elapsed_seconds']}s  |  ranked by {sweep_meta['rank_by']}", file=err)
    print("", file=err)

    if not results:
        print("  No results.", file=err)
        return

    # Build table
    # Collect param keys from first result
    param_keys = list(results[0]["params"].keys())

    # Determine column widths for params (min 6, max = max of header/values)
    param_widths = {}
    for pk in param_keys:
        val_width = max(len(str(r["params"][pk])) for r in results)
        param_widths[pk] = max(len(pk), val_width, 6)

    # Header
    hdr = f"  {'#':>3}  "
    for pk in param_keys:
        hdr += f"  {pk:>{param_widths[pk]}}"
    hdr += f"  {'Return':>9}  {'Sharpe':>7}  {'MaxDD':>8}  {'WinR':>6}  {'PF':>6}  {'Trades':>6}"
    print(f"\033[1m{hdr}\033[0m", file=err)
    print(f"  {'─' * (len(hdr) - 2)}", file=err)

    for r in results:
        m = r["metrics"]
        row = f"  {r['rank']:>3}  "
        for pk in param_keys:
            row += f"  {r['params'][pk]:>{param_widths[pk]}}"

        ret = m["total_return_pct"]
        ret_c = "\033[32m" if ret >= 0 else "\033[31m"
        sharpe = m["sharpe_ratio"]
        sh_c = "\033[32m" if sharpe >= 0 else "\033[31m"
        dd = m["max_drawdown_pct"]
        dd_c = "\033[32m" if dd > -20 else "\033[33m" if dd > -50 else "\033[31m"
        wr = m["win_rate_pct"]
        wr_c = "\033[32m" if wr >= 50 else "\033[31m"
        pf = m["profit_factor"]
        pf_val = float(pf) if isinstance(pf, (int, float)) else 0
        pf_c = "\033[32m" if pf_val >= 1 else "\033[31m"
        rst = "\033[0m"

        row += f"  {ret_c}{ret:>+8.1f}%{rst}"
        row += f"  {sh_c}{sharpe:>+7.3f}{rst}"
        row += f"  {dd_c}{dd:>+7.1f}%{rst}"
        row += f"  {wr_c}{wr:>5.1f}%{rst}"
        row += f"  {pf_c}{pf_val:>6.2f}{rst}"
        row += f"  {m['total_trades']:>6}"
        print(row, file=err)

    print("", file=err)


def main():
    parser = argparse.ArgumentParser(description="Run backtests on BTCUSDT data")
    parser.add_argument("--strategy", "-s", default=None, help="Strategy name (e.g. 'rsi')")
    parser.add_argument("--resample", "-r", default=None, help="Resample frequency (e.g. '1h', '4h', '1D')")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--params", default=None, help="JSON string of strategy parameters")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--rank-by", default="sharpe_ratio", help="Metric to rank sweep results")
    parser.add_argument("--top", type=int, default=20, help="Number of top results to show")
    parser.add_argument("--fee", type=float, default=0.0005, help="Fee rate per side (default 5 bps)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage per side (default 5 bps)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--long-only", action="store_true", help="Long-only mode")
    parser.add_argument("--impact-coeff", type=float, default=0.0, help="Tier 2: market impact coefficient (0.20 recommended for BTC)")
    parser.add_argument("--adv", type=float, default=20e9, help="Tier 2: average daily volume (default $20B)")
    parser.add_argument("--funding-rate", type=float, default=0.0, help="Perpetual funding rate per 8h (e.g. 0.0001)")
    parser.add_argument("--data", default=None, help="Path to CSV data file")
    parser.add_argument("--jobs", type=int, default=None, help="Number of parallel workers for sweep")
    parser.add_argument("--list-strategies", action="store_true", help="List all available strategies and exit")

    args = parser.parse_args()

    if args.list_strategies:
        from strategies import list_strategies
        strategies = list_strategies()
        if not strategies:
            print("No strategies found.", file=sys.stderr)
            sys.exit(0)
        # Print table
        name_w = max(len(s["name"]) for s in strategies)
        class_w = max(len(s["class"].__name__) for s in strategies)
        print(f"  {'Name':<{name_w}}  {'Class':<{class_w}}  File", file=sys.stderr)
        print(f"  {'-' * name_w}  {'-' * class_w}  {'-' * 40}", file=sys.stderr)
        for s in strategies:
            print(f"  {s['name']:<{name_w}}  {s['class'].__name__:<{class_w}}  {s['file']}", file=sys.stderr)
        sys.exit(0)

    if not args.strategy:
        parser.error("--strategy is required (or use --list-strategies)")

    data_path = args.data or str(DATA_PATH)

    # Load data
    print(f"[info] Loading data from {data_path}...", file=sys.stderr)
    t0 = time.time()
    df = load_candles(data_path, start=args.start, end=args.end, resample=args.resample)
    load_time = time.time() - t0
    print(f"[info] Loaded {len(df)} bars in {load_time:.1f}s", file=sys.stderr)

    # Discover strategy
    strategy_class = discover_strategy(args.strategy)

    # Build config
    config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_pct=args.slippage,
        long_only=args.long_only,
        impact_coefficient=args.impact_coeff,
        adv_daily=args.adv,
        funding_rate_8h=args.funding_rate,
    )

    data_info = {
        "rows": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "resample": args.resample,
    }

    config_info = {
        "initial_capital": config.initial_capital,
        "fee_rate": config.fee_rate,
        "slippage_pct": config.slippage_pct,
        "long_only": config.long_only,
    }

    if args.sweep:
        # Parameter sweep mode
        print("[info] Running parameter sweep...", file=sys.stderr)
        t0 = time.time()
        grid = strategy_class().param_grid()
        sweep_results = run_sweep(
            strategy_class, df, grid, config,
            rank_by=args.rank_by, top=args.top, n_jobs=args.jobs,
        )
        sweep_time = time.time() - t0
        print(f"[info] Sweep completed in {sweep_time:.1f}s", file=sys.stderr)

        output = {
            "mode": "sweep",
            "strategy": args.strategy,
            "data_info": data_info,
            "config": config_info,
            "sweep_meta": {
                "total_combos": sweep_results["total_combos"],
                "completed": sweep_results["completed"],
                "rank_by": sweep_results["rank_by"],
                "elapsed_seconds": round(sweep_time, 1),
            },
            "results": [
                {
                    "rank": r["rank"],
                    "params": r["params"],
                    "metrics": {
                        "total_return_pct": r["metrics"]["total_return_pct"],
                        "sharpe_ratio": r["metrics"]["sharpe_ratio"],
                        "max_drawdown_pct": r["metrics"]["max_drawdown_pct"],
                        "win_rate_pct": r["metrics"]["win_rate_pct"],
                        "profit_factor": r["metrics"]["profit_factor"],
                        "total_trades": r["metrics"]["total_trades"],
                    },
                }
                for r in sweep_results["results"]
            ],
        }
    else:
        # Single run mode
        params = {}
        if args.params:
            params = json.loads(args.params)

        strategy = strategy_class(**params)
        print(f"[info] Running {args.strategy} strategy...", file=sys.stderr)

        t0 = time.time()
        signals = strategy.generate_signals(df)
        results = run_backtest(df, signals, config)
        metrics = compute_metrics(results)
        run_time = time.time() - t0
        print(f"[info] Backtest completed in {run_time:.1f}s", file=sys.stderr)

        # Sample equity curve (max 500 points)
        equity = results["equity_curve"]
        step = max(1, len(equity) // 500)
        equity_sample = [
            {"time": str(t), "equity": round(v, 2)}
            for t, v in zip(equity.index[::step], equity.values[::step])
        ]

        output = {
            "mode": "single",
            "strategy": args.strategy,
            "params": params or {k: getattr(strategy, k) for k in strategy.param_grid().keys()},
            "data_info": data_info,
            "config": config_info,
            "metrics": metrics,
            "trades": results["trades"],
            "equity_curve_sample": equity_sample,
        }

    # Print human-readable summary to stderr
    if args.sweep:
        _print_sweep_summary(args.strategy, data_info, output["sweep_meta"], output["results"], config_info)
    else:
        _print_single_summary(args.strategy, output["params"], data_info, metrics, config_info)

    json.dump(output, sys.stdout, indent=2, cls=NumpyEncoder)
    print(file=sys.stdout)  # trailing newline


if __name__ == "__main__":
    main()
