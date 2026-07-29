from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0005       # 0.05% (5 bps) taker fee per side
    slippage_pct: float = 0.0005   # 0.05% (5 bps) slippage per side
    long_only: bool = False
    # Tier 2: impact-adjusted model (set impact_coefficient > 0 to enable)
    impact_coefficient: float = 0.0    # 0.20 recommended for BTC
    adv_daily: float = 20_000_000_000  # Average daily volume ($20B BTC default)
    # Funding rate for perpetuals (set > 0 to enable)
    funding_rate_8h: float = 0.0       # 0.0001 = 1 bp per 8h settlement


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
) -> dict:
    """Run a vectorized backtest.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    signals : pd.Series
        Signal series aligned to df index: 1=long, -1=short, 0=flat.
    config : BacktestConfig, optional
        Backtest configuration.

    Returns
    -------
    dict with keys: equity_curve, positions, strategy_returns, trades, fees_paid.
    """
    if config is None:
        config = BacktestConfig()

    signals = signals.reindex(df.index, fill_value=0)

    if config.long_only:
        signals = signals.clip(lower=0)

    # Next-bar execution: signal at bar[i] executes at bar[i+1]
    positions = signals.shift(1, fill_value=0)

    # Close-to-close returns
    close_returns = df["close"].pct_change().fillna(0)
    strategy_returns = positions * close_returns

    # Fee/slippage on position changes
    position_changes = positions.diff().abs().fillna(0)
    flat_cost = config.fee_rate + config.slippage_pct

    if config.impact_coefficient > 0:
        # Tier 2: impact-adjusted model
        # impact = coefficient * sqrt(trade_notional / ADV)
        # trade_notional approximated from equity curve built so far
        running_equity = config.initial_capital * (1 + strategy_returns.cumsum())
        trade_notional = running_equity * position_changes
        impact = config.impact_coefficient * np.sqrt(
            trade_notional.clip(lower=0) / config.adv_daily
        )
        costs = position_changes * config.fee_rate + impact
    else:
        # Tier 1: flat model
        costs = position_changes * flat_cost

    # Funding rate drag on held positions (perpetuals)
    if config.funding_rate_8h > 0:
        # Infer bar duration in hours from index
        if len(df) > 1:
            bar_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
        else:
            bar_hours = 1.0
        funding_per_bar = config.funding_rate_8h * (bar_hours / 8.0)
        funding_costs = positions.abs() * funding_per_bar
        costs = costs + funding_costs

    strategy_returns = strategy_returns - costs

    total_fees = float((costs * config.initial_capital).sum())

    # Equity curve
    equity_curve = config.initial_capital * (1 + strategy_returns).cumprod()

    # Extract trades
    trades = _extract_trades(positions, df["close"], equity_curve)

    return {
        "equity_curve": equity_curve,
        "positions": positions,
        "strategy_returns": strategy_returns,
        "trades": trades,
        "fees_paid": total_fees,
    }


def _extract_trades(
    positions: pd.Series,
    close: pd.Series,
    equity: pd.Series,
) -> list[dict]:
    """Extract individual trades from position changes."""
    pos_values = positions.values
    changes = np.where(pos_values[1:] != pos_values[:-1])[0] + 1

    # Add index 0 if first position is non-zero
    if pos_values[0] != 0:
        changes = np.concatenate([[0], changes])

    trades = []
    idx = positions.index
    i = 0

    while i < len(changes):
        enter_idx = changes[i]
        direction = pos_values[enter_idx]

        if direction == 0:
            i += 1
            continue

        # Find exit
        if i + 1 < len(changes):
            exit_idx = changes[i + 1]
        else:
            exit_idx = len(pos_values) - 1

        entry_time = str(idx[enter_idx])
        exit_time = str(idx[exit_idx])
        entry_price = float(close.iloc[enter_idx])
        exit_price = float(close.iloc[exit_idx])

        if direction == 1:
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (1 - exit_price / entry_price) * 100

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": "long" if direction == 1 else "short",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 4),
            "bars_held": int(exit_idx - enter_idx),
        })

        i += 1
        # Skip the exit-change index if it introduced a new position
        if i < len(changes) and changes[i] == exit_idx and pos_values[exit_idx] == 0:
            i += 1

    return trades
