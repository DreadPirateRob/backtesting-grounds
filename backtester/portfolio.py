from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import compute_metrics


@dataclass
class PortfolioConfig:
    method: str = "equal_weight"
    rebalance_freq: str = "1M"
    lookback_bars: int = 252
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    strategy_returns: dict[str, pd.Series]
    combined_returns: pd.Series
    metrics: dict
    correlation_matrix: pd.DataFrame


def _align_returns(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Align all strategy return series into a single DataFrame, fill missing with 0."""
    df = pd.DataFrame(strategy_returns)
    df = df.sort_index().fillna(0.0)
    return df


def _compute_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Compute rebalance dates at the given frequency, starting from the first date."""
    if len(index) == 0:
        return []
    start = index[0]
    end = index[-1]
    dates = pd.date_range(start=start, end=end, freq=freq)
    if len(dates) == 0 or dates[0] != start:
        dates = dates.insert(0, start)
    return list(dates)


def _equal_weight(n_strategies: int, config: PortfolioConfig) -> np.ndarray:
    w = np.full(n_strategies, 1.0 / n_strategies)
    w = np.clip(w, config.min_weight, config.max_weight)
    total = w.sum()
    if total > 0:
        w /= total
    return w


def _risk_parity_weights(
    returns_df: pd.DataFrame,
    lookback: int,
    end_loc: int,
    config: PortfolioConfig,
) -> np.ndarray:
    start_loc = max(0, end_loc - lookback)
    window = returns_df.iloc[start_loc:end_loc]

    if len(window) < 2:
        return _equal_weight(returns_df.shape[1], config)

    vols = window.std().values
    vols = np.where(vols > 0, vols, 1e-10)
    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum()
    w = np.clip(w, config.min_weight, config.max_weight)
    total = w.sum()
    if total > 0:
        w /= total
    return w


def _max_sharpe_weights(
    returns_df: pd.DataFrame,
    lookback: int,
    end_loc: int,
    config: PortfolioConfig,
) -> np.ndarray:
    start_loc = max(0, end_loc - lookback)
    window = returns_df.iloc[start_loc:end_loc]

    if len(window) < 2:
        return _equal_weight(returns_df.shape[1], config)

    means = window.mean().values
    stds = window.std().values
    stds = np.where(stds > 0, stds, 1e-10)
    sharpes = means / stds
    sharpes = np.maximum(sharpes, 0.0)
    total = sharpes.sum()

    if total <= 0:
        return _equal_weight(returns_df.shape[1], config)

    w = sharpes / total
    w = np.clip(w, config.min_weight, config.max_weight)
    total = w.sum()
    if total > 0:
        w /= total
    return w


def build_portfolio(
    strategy_returns: dict[str, pd.Series],
    config: PortfolioConfig | None = None,
    initial_capital: float = 10_000,
) -> PortfolioResult:
    if config is None:
        config = PortfolioConfig()

    returns_df = _align_returns(strategy_returns)
    names = list(returns_df.columns)
    n = len(names)
    index = returns_df.index

    rebalance_dates = _compute_rebalance_dates(index, config.rebalance_freq)

    weights = np.zeros((len(index), n))
    weight_records: list[tuple[pd.Timestamp, np.ndarray]] = []

    rebal_positions: list[int] = []
    for rd in rebalance_dates:
        loc = index.searchsorted(rd)
        if loc < len(index):
            rebal_positions.append(int(loc))
    rebal_positions = sorted(set(rebal_positions))

    if len(rebal_positions) == 0:
        rebal_positions = [0]

    weight_func = {
        "equal_weight": lambda end_loc: _equal_weight(n, config),
        "risk_parity": lambda end_loc: _risk_parity_weights(
            returns_df, config.lookback_bars, end_loc, config,
        ),
        "inverse_vol": lambda end_loc: _risk_parity_weights(
            returns_df, config.lookback_bars, end_loc, config,
        ),
        "max_sharpe": lambda end_loc: _max_sharpe_weights(
            returns_df, config.lookback_bars, end_loc, config,
        ),
    }

    compute_w = weight_func.get(config.method)
    if compute_w is None:
        raise ValueError(f"Unknown portfolio method: {config.method}")

    for i, pos in enumerate(rebal_positions):
        w = compute_w(pos)
        next_pos = rebal_positions[i + 1] if i + 1 < len(rebal_positions) else len(index)
        weights[pos:next_pos, :] = w
        weight_records.append((index[pos], w.copy()))

    weights_history = pd.DataFrame(
        [r[1] for r in weight_records],
        index=pd.DatetimeIndex([r[0] for r in weight_records]),
        columns=names,
    )

    combined_returns = (returns_df.values * weights).sum(axis=1)
    combined_returns = pd.Series(combined_returns, index=index)

    equity_curve = initial_capital * (1 + combined_returns).cumprod()

    correlation_matrix = returns_df.corr()

    positions_synthetic = pd.Series(1, index=index, dtype="int8")
    synthetic_results = {
        "equity_curve": equity_curve,
        "positions": positions_synthetic,
        "strategy_returns": combined_returns,
        "trades": [],
        "fees_paid": 0.0,
    }
    metrics = compute_metrics(synthetic_results)

    return PortfolioResult(
        equity_curve=equity_curve,
        weights_history=weights_history,
        strategy_returns=dict(strategy_returns),
        combined_returns=combined_returns,
        metrics=metrics,
        correlation_matrix=correlation_matrix,
    )


def run_portfolio_backtest(
    strategies: dict[str, tuple[type, dict]],
    df: pd.DataFrame,
    backtest_config: BacktestConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> PortfolioResult:
    if backtest_config is None:
        backtest_config = BacktestConfig()

    strategy_returns: dict[str, pd.Series] = {}

    for name, (strategy_class, params) in strategies.items():
        strat = strategy_class(**params)
        signals = strat.generate_signals(df)
        result = run_backtest(df, signals, backtest_config)
        strategy_returns[name] = result["strategy_returns"]

    return build_portfolio(
        strategy_returns,
        config=portfolio_config,
        initial_capital=backtest_config.initial_capital,
    )


def build_regime_portfolio(
    strategy_returns: dict[str, pd.Series],
    regimes: pd.Series,
    regime_weights: dict[str, dict[str, float]],
    initial_capital: float = 10_000,
) -> PortfolioResult:
    returns_df = _align_returns(strategy_returns)
    names = list(returns_df.columns)
    n = len(names)
    index = returns_df.index

    regimes_aligned = regimes.reindex(index, method="ffill")

    weights = np.zeros((len(index), n))

    for regime_name, strat_weights in regime_weights.items():
        mask = (regimes_aligned == regime_name).values
        w = np.array([strat_weights.get(name, 0.0) for name in names])
        total = w.sum()
        if total > 0:
            w /= total
        weights[mask] = w

    weight_records: list[tuple[pd.Timestamp, np.ndarray]] = []
    prev_regime = None
    for i, ts in enumerate(index):
        regime = regimes_aligned.iloc[i]
        if regime != prev_regime:
            weight_records.append((ts, weights[i].copy()))
            prev_regime = regime

    if weight_records:
        weights_history = pd.DataFrame(
            [r[1] for r in weight_records],
            index=pd.DatetimeIndex([r[0] for r in weight_records]),
            columns=names,
        )
    else:
        weights_history = pd.DataFrame(columns=names)

    combined_returns = (returns_df.values * weights).sum(axis=1)
    combined_returns = pd.Series(combined_returns, index=index)

    equity_curve = initial_capital * (1 + combined_returns).cumprod()

    correlation_matrix = returns_df.corr()

    positions_synthetic = pd.Series(1, index=index, dtype="int8")
    synthetic_results = {
        "equity_curve": equity_curve,
        "positions": positions_synthetic,
        "strategy_returns": combined_returns,
        "trades": [],
        "fees_paid": 0.0,
    }
    metrics = compute_metrics(synthetic_results)

    return PortfolioResult(
        equity_curve=equity_curve,
        weights_history=weights_history,
        strategy_returns=dict(strategy_returns),
        combined_returns=combined_returns,
        metrics=metrics,
        correlation_matrix=correlation_matrix,
    )


def print_portfolio_report(
    result: PortfolioResult,
    strategy_names: list[str] | None = None,
) -> None:
    if strategy_names is None:
        strategy_names = list(result.strategy_returns.keys())

    print("=" * 70)
    print("PORTFOLIO REPORT")
    print("=" * 70)

    m = result.metrics
    print(f"\n{'Portfolio Metrics':^70}")
    print("-" * 70)
    print(f"  Total Return:       {m['total_return_pct']:>10.2f}%")
    print(f"  Annualized Return:  {m['annualized_return_pct']:>10.2f}%")
    print(f"  Sharpe Ratio:       {m['sharpe_ratio']:>10.4f}")
    print(f"  Sortino Ratio:      {m['sortino_ratio']:>10.4f}")
    print(f"  Max Drawdown:       {m['max_drawdown_pct']:>10.2f}%")
    print(f"  Max DD Duration:    {m['max_drawdown_duration_days']:>10.1f} days")
    print(f"  Calmar Ratio:       {m['calmar_ratio']:>10.4f}")

    print(f"\n{'Per-Strategy Metrics':^70}")
    print("-" * 70)
    header = (
        f"  {'Strategy':<25} {'Return%':>10} {'Sharpe':>10} "
        f"{'MaxDD%':>10} {'Sortino':>10}"
    )
    print(header)
    print("  " + "-" * 65)

    for name in strategy_names:
        if name not in result.strategy_returns:
            continue
        sr = result.strategy_returns[name]
        eq = 10_000 * (1 + sr).cumprod()
        positions_s = pd.Series(1, index=sr.index, dtype="int8")
        strat_result = {
            "equity_curve": eq,
            "positions": positions_s,
            "strategy_returns": sr,
            "trades": [],
            "fees_paid": 0.0,
        }
        sm = compute_metrics(strat_result)
        print(
            f"  {name:<25} {sm['total_return_pct']:>10.2f} "
            f"{sm['sharpe_ratio']:>10.4f} {sm['max_drawdown_pct']:>10.2f} "
            f"{sm['sortino_ratio']:>10.4f}"
        )

    print(f"\n{'Correlation Matrix':^70}")
    print("-" * 70)
    corr = result.correlation_matrix
    col_w = 12
    print(
        "  " + " " * 20
        + "".join(f"{c[:col_w]:>{col_w}}" for c in corr.columns)
    )
    for row_name in corr.index:
        vals = "".join(
            f"{corr.loc[row_name, c]:>{col_w}.4f}" for c in corr.columns
        )
        print(f"  {row_name:<20}{vals}")

    print(f"\n{'Weight History Summary':^70}")
    print("-" * 70)
    wh = result.weights_history
    if len(wh) > 0:
        header = f"  {'Strategy':<25} {'Min':>10} {'Max':>10} {'Avg':>10}"
        print(header)
        print("  " + "-" * 55)
        for col in wh.columns:
            print(
                f"  {col:<25} {wh[col].min():>10.4f} "
                f"{wh[col].max():>10.4f} {wh[col].mean():>10.4f}"
            )
        print(f"\n  Rebalance count: {len(wh)}")
    else:
        print("  No weight history recorded.")

    print(f"\n{'Diversification':^70}")
    print("-" * 70)
    returns_df = _align_returns(result.strategy_returns)
    individual_vols = returns_df.std().values
    if len(wh) > 0:
        avg_weights = wh.mean().values
    else:
        avg_weights = np.ones(len(individual_vols)) / len(individual_vols)
    weighted_avg_vol = (avg_weights * individual_vols).sum()
    portfolio_vol = result.combined_returns.std()
    div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0.0
    print(f"  Portfolio Vol (per bar):    {portfolio_vol:.6f}")
    print(f"  Wtd Avg Individual Vol:     {weighted_avg_vol:.6f}")
    print(f"  Diversification Ratio:      {div_ratio:.4f}")
    print("=" * 70)
