import io
import itertools
import multiprocessing
import sys
from typing import Type

import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import compute_metrics
from .strategy import Strategy


def _init_worker(parquet_bytes: bytes):
    """Initialize worker with deserialized DataFrame."""
    global _worker_df
    _worker_df = pd.read_parquet(io.BytesIO(parquet_bytes))


def _run_single(args: tuple) -> dict | None:
    """Run a single backtest with given params."""
    strategy_class, params, config_dict = args
    try:
        config = BacktestConfig(**config_dict)
        strategy = strategy_class(**params)
        signals = strategy.generate_signals(_worker_df)
        results = run_backtest(_worker_df, signals, config)
        metrics = compute_metrics(results)
        return {"params": params, "metrics": metrics}
    except Exception as e:
        print(f"[sweep] Error with params {params}: {e}", file=sys.stderr)
        return None


def run_sweep(
    strategy_class: Type[Strategy],
    df: pd.DataFrame,
    grid: dict[str, list],
    config: BacktestConfig | None = None,
    rank_by: str = "sharpe_ratio",
    top: int = 20,
    n_jobs: int | None = None,
) -> dict:
    """Run parallel parameter sweep.

    Parameters
    ----------
    strategy_class : Type[Strategy]
        Strategy class to instantiate with each param combo.
    df : pd.DataFrame
        OHLCV data.
    grid : dict
        Parameter grid, e.g. {"period": [7,14], "threshold": [20,30]}.
    config : BacktestConfig, optional
    rank_by : str
        Metric to rank results by (descending). Use "-metric" for ascending.
    top : int
        Number of top results to return.
    n_jobs : int, optional
        Number of parallel workers. Defaults to CPU count.

    Returns
    -------
    dict with sweep metadata and ranked results.
    """
    if config is None:
        config = BacktestConfig()

    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    # Safety net: auto-fallback to n_jobs=1 when no __main__ guard detected.
    # On macOS, multiprocessing.Pool spawns new processes that re-import __main__,
    # which causes infinite recursion if there's no `if __name__ == "__main__":` guard.
    if n_jobs > 1:
        import __main__
        if not hasattr(__main__, "__file__"):
            print(
                "[sweep] Warning: no __main__.__file__ detected (interactive session). "
                "Falling back to n_jobs=1 to avoid multiprocessing crash.",
                file=sys.stderr,
            )
            n_jobs = 1

    # Generate all parameter combinations
    keys = list(grid.keys())
    values = list(grid.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    total = len(combos)
    print(f"[sweep] {total} parameter combinations, {n_jobs} workers", file=sys.stderr)

    # Serialize DataFrame to parquet bytes for efficient transfer
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    parquet_bytes = buf.getvalue()

    config_dict = {
        "initial_capital": config.initial_capital,
        "fee_rate": config.fee_rate,
        "slippage_pct": config.slippage_pct,
        "long_only": config.long_only,
        "impact_coefficient": config.impact_coefficient,
        "adv_daily": config.adv_daily,
        "funding_rate_8h": config.funding_rate_8h,
    }

    args_list = [(strategy_class, params, config_dict) for params in combos]

    with multiprocessing.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(parquet_bytes,),
    ) as pool:
        raw_results = pool.map(_run_single, args_list)

    results = [r for r in raw_results if r is not None]

    # Rank results
    ascending = False
    sort_key = rank_by
    if rank_by.startswith("-"):
        sort_key = rank_by[1:]
        ascending = True
    # For drawdown, more negative is worse, so ascending sorts best first
    if sort_key == "max_drawdown_pct":
        ascending = not ascending

    results.sort(
        key=lambda r: r["metrics"].get(sort_key, 0) or 0,
        reverse=not ascending,
    )

    top_results = results[:top]
    for i, r in enumerate(top_results):
        r["rank"] = i + 1

    return {
        "total_combos": total,
        "completed": len(results),
        "rank_by": rank_by,
        "results": top_results,
    }
