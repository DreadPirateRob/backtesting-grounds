import numpy as np
import pandas as pd


def compute_metrics(results: dict) -> dict:
    """Compute performance metrics from backtest results.

    Parameters
    ----------
    results : dict
        Output from run_backtest().

    Returns
    -------
    dict of metric name -> value.
    """
    equity: pd.Series = results["equity_curve"]
    returns: pd.Series = results["strategy_returns"]
    positions: pd.Series = results["positions"]
    trades: list[dict] = results["trades"]

    total_bars = len(returns)
    if total_bars == 0:
        return _empty_metrics()

    # Total return
    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # Annualized return (assume continuous trading ~365.25 days)
    # Infer bars per year from the index
    duration_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    if duration_days > 0:
        years = duration_days / 365.25
        annualized_return_pct = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
    else:
        annualized_return_pct = 0.0

    # Sharpe ratio (annualized)
    # For sub-hourly data, resample returns to 1h to avoid autocorrelation inflation
    bars_per_year = total_bars / years if duration_days > 0 else 252
    if len(returns) > 1:
        bar_seconds = (returns.index[1] - returns.index[0]).total_seconds()
    else:
        bar_seconds = 3600  # default 1h
    if bar_seconds < 3600 and len(returns) > 60:
        # Resample to hourly: compound sub-hourly returns
        hourly_returns = (1 + returns).resample("1h").prod() - 1
        hourly_returns = hourly_returns.dropna()
        h_bars = len(hourly_returns)
        h_bars_per_year = h_bars / years if duration_days > 0 else 8760
        h_mean = hourly_returns.mean()
        h_std = hourly_returns.std()
        sharpe_ratio = (h_mean / h_std * np.sqrt(h_bars_per_year)) if h_std > 0 else 0.0
        h_downside = hourly_returns[hourly_returns < 0]
        h_downside_std = h_downside.std() if len(h_downside) > 0 else 0.0
        sortino_ratio = (h_mean / h_downside_std * np.sqrt(h_bars_per_year)) if h_downside_std > 0 else 0.0
    else:
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe_ratio = (mean_ret / std_ret * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0
        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0.0
        sortino_ratio = (mean_ret / downside_std * np.sqrt(bars_per_year)) if downside_std > 0 else 0.0

    # Drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown_pct = float(drawdown.min()) * 100

    # Max drawdown duration
    underwater = drawdown < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_lengths = underwater.groupby(groups).sum()
        max_dd_bars = int(dd_lengths.max())
        # Convert bars to approximate days
        bar_duration_days = duration_days / total_bars if total_bars > 0 else 0
        max_drawdown_duration_days = round(max_dd_bars * bar_duration_days, 1)
    else:
        max_drawdown_duration_days = 0.0

    # Calmar ratio
    calmar_ratio = (annualized_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0.0

    # Trade statistics
    total_trades = len(trades)
    if total_trades > 0:
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate_pct = len(wins) / total_trades * 100
        avg_win_pct = np.mean(wins) if wins else 0.0
        avg_loss_pct = np.mean(losses) if losses else 0.0
        best_trade_pct = max(pnls)
        worst_trade_pct = min(pnls)

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.99
    else:
        win_rate_pct = 0.0
        avg_win_pct = 0.0
        avg_loss_pct = 0.0
        best_trade_pct = 0.0
        worst_trade_pct = 0.0
        profit_factor = 0.0

    # Direction-split trade statistics
    long_trades_list = [t for t in trades if t["direction"] == "long"]
    short_trades_list = [t for t in trades if t["direction"] == "short"]

    long_trades = len(long_trades_list)
    short_trades = len(short_trades_list)

    long_wins = [t for t in long_trades_list if t["pnl_pct"] > 0]
    short_wins = [t for t in short_trades_list if t["pnl_pct"] > 0]

    long_win_rate_pct = (len(long_wins) / long_trades * 100) if long_trades > 0 else 0.0
    short_win_rate_pct = (len(short_wins) / short_trades * 100) if short_trades > 0 else 0.0

    long_avg_bars_held = (
        np.mean([t["bars_held"] for t in long_trades_list]) if long_trades > 0 else 0.0
    )
    short_avg_bars_held = (
        np.mean([t["bars_held"] for t in short_trades_list]) if short_trades > 0 else 0.0
    )

    # Time in market
    time_in_market_pct = float((positions != 0).mean()) * 100

    return {
        "total_return_pct": round(total_return_pct, 4),
        "annualized_return_pct": round(annualized_return_pct, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "max_drawdown_duration_days": max_drawdown_duration_days,
        "calmar_ratio": round(calmar_ratio, 4),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate_pct, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_win_pct": round(avg_win_pct, 4),
        "avg_loss_pct": round(avg_loss_pct, 4),
        "best_trade_pct": round(best_trade_pct, 4),
        "worst_trade_pct": round(worst_trade_pct, 4),
        "time_in_market_pct": round(time_in_market_pct, 4),
        "total_fees_paid": round(results["fees_paid"], 2),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_win_rate_pct": round(long_win_rate_pct, 4),
        "short_win_rate_pct": round(short_win_rate_pct, 4),
        "long_avg_bars_held": round(float(long_avg_bars_held), 2),
        "short_avg_bars_held": round(float(short_avg_bars_held), 2),
    }


def _empty_metrics() -> dict:
    return {
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_duration_days": 0.0,
        "calmar_ratio": 0.0,
        "total_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "best_trade_pct": 0.0,
        "worst_trade_pct": 0.0,
        "time_in_market_pct": 0.0,
        "total_fees_paid": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "long_win_rate_pct": 0.0,
        "short_win_rate_pct": 0.0,
        "long_avg_bars_held": 0.0,
        "short_avg_bars_held": 0.0,
    }
