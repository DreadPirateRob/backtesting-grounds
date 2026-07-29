"""Pairs trading utilities: cointegration, spread computation, z-score."""

import numpy as np
import pandas as pd


def align_pair(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames to their common index."""
    common = df_a.index.intersection(df_b.index)
    return df_a.loc[common].copy(), df_b.loc[common].copy()


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS hedge ratio: y = beta * x + alpha. Returns beta."""
    x_arr = x.values.astype(float)
    y_arr = y.values.astype(float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    if len(x_arr) < 2:
        return 1.0
    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    ss_xx = ((x_arr - x_mean) ** 2).sum()
    if ss_xx == 0:
        return 1.0
    return float(((x_arr - x_mean) * (y_arr - y_mean)).sum() / ss_xx)


def rolling_hedge_ratio(
    y: pd.Series, x: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling OLS hedge ratio."""
    betas = pd.Series(np.nan, index=y.index)
    y_arr = y.values.astype(float)
    x_arr = x.values.astype(float)
    for i in range(window, len(y_arr)):
        y_win = y_arr[i - window : i]
        x_win = x_arr[i - window : i]
        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if mask.sum() < 10:
            continue
        x_m = x_win[mask]
        y_m = y_win[mask]
        x_mean = x_m.mean()
        ss_xx = ((x_m - x_mean) ** 2).sum()
        if ss_xx == 0:
            continue
        beta = ((x_m - x_mean) * (y_m - y_m.mean())).sum() / ss_xx
        betas.iloc[i] = beta
    return betas.ffill()


def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float | pd.Series,
    use_log: bool = True,
) -> pd.Series:
    """Compute spread = log(A) - beta * log(B) or A - beta * B."""
    if use_log:
        a = np.log(price_a)
        b = np.log(price_b)
    else:
        a = price_a
        b = price_b
    return a - hedge_ratio * b


def zscore(
    spread: pd.Series, lookback: int = 60
) -> pd.Series:
    """Rolling z-score of the spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std.replace(0, np.nan)


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Hurst exponent via rescaled range (R/S) analysis.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending

    Returns H value.
    """
    arr = series.dropna().values.astype(float)
    n = len(arr)
    if n < max_lag * 2:
        return 0.5

    lags = range(2, min(max_lag + 1, n // 2))
    rs_values = []

    for lag in lags:
        # Split into subseries of length lag
        n_sub = n // lag
        if n_sub < 1:
            continue
        rs_list = []
        for i in range(n_sub):
            sub = arr[i * lag : (i + 1) * lag]
            mean_sub = sub.mean()
            deviations = sub - mean_sub
            cumdev = np.cumsum(deviations)
            r = cumdev.max() - cumdev.min()
            s = sub.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

    if len(rs_values) < 3:
        return 0.5

    log_lags = np.array([v[0] for v in rs_values])
    log_rs = np.array([v[1] for v in rs_values])

    # OLS fit: log(R/S) = H * log(lag) + c
    slope = np.polyfit(log_lags, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def engle_granger_coint(
    y: pd.Series, x: pd.Series, max_lag: int | None = None
) -> tuple[float, float, float]:
    """Engle-Granger cointegration test without statsmodels.

    1. Run OLS: y = alpha + beta * x + epsilon
    2. Test residuals for unit root using ADF-like test

    Returns (adf_stat, critical_value_5pct, hedge_ratio).
    Critical value at 5%: approximately -3.37 for 2-variable case.
    If adf_stat < critical_value, the pair is cointegrated.
    """
    x_arr = x.values.astype(float)
    y_arr = y.values.astype(float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    n = len(x_arr)
    if n < 30:
        return 0.0, -3.37, 1.0

    # OLS: y = alpha + beta * x
    X = np.column_stack([np.ones(n), x_arr])
    coef = np.linalg.lstsq(X, y_arr, rcond=None)[0]
    residuals = y_arr - X @ coef
    beta = float(coef[1])

    # ADF test on residuals (no constant — residuals should be mean-zero)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** 0.25))
    max_lag = max(1, min(max_lag, n // 4))

    # Simple ADF: delta_resid = gamma * resid_lag + ... + error
    dr = np.diff(residuals)
    resid_lag = residuals[:-1]

    # Include lagged differences
    T = len(dr)
    lag_terms = max_lag
    if T <= lag_terms + 2:
        return 0.0, -3.37, beta

    y_adf = dr[lag_terms:]
    X_adf = resid_lag[lag_terms:].reshape(-1, 1)

    for lag in range(1, lag_terms + 1):
        X_adf = np.column_stack([X_adf, dr[lag_terms - lag : T - lag]])

    coef_adf = np.linalg.lstsq(X_adf, y_adf, rcond=None)[0]
    gamma = coef_adf[0]

    resid_adf = y_adf - X_adf @ coef_adf
    se = np.sqrt((resid_adf ** 2).sum() / (len(y_adf) - len(coef_adf)))
    se_gamma = se / np.sqrt((X_adf[:, 0] ** 2).sum()) if (X_adf[:, 0] ** 2).sum() > 0 else 1.0
    adf_stat = gamma / se_gamma if se_gamma > 0 else 0.0

    # MacKinnon critical value for 2-variable Engle-Granger at 5%
    critical_5pct = -3.37

    return float(adf_stat), critical_5pct, beta


def pairs_backtest(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    signals: pd.Series,
    hedge_ratio: float | pd.Series,
    fee_rate: float = 0.001,
    initial_capital: float = 10_000.0,
) -> dict:
    """Backtest a pairs trading strategy.

    Signal: +1 = long A / short B, -1 = short A / long B, 0 = flat.
    Each leg gets half the capital. Fees are charged on both legs.

    Returns dict compatible with compute_metrics().
    """
    signals = signals.reindex(df_a.index, fill_value=0)

    # Next-bar execution
    positions = signals.shift(1, fill_value=0)

    # Returns for each leg
    ret_a = df_a["close"].pct_change().fillna(0)
    ret_b = df_b["close"].pct_change().fillna(0)

    # Handle hedge ratio (constant or rolling)
    if isinstance(hedge_ratio, (int, float)):
        hr = hedge_ratio
    else:
        hr = hedge_ratio.reindex(df_a.index).ffill().fillna(1.0)

    # Spread return: position * (ret_a - beta * ret_b)
    # Normalized by (1 + |beta|) so each leg gets proportional capital
    weight = 1.0 / (1.0 + np.abs(hr))
    spread_return = positions * (ret_a * weight - hr * ret_b * (1 - weight))

    # Alternatively, simpler: half capital each leg
    spread_return = positions * (ret_a * 0.5 - hr * ret_b * 0.5)

    # Fees: charged on BOTH legs for each position change
    position_changes = positions.diff().abs().fillna(0)
    # Double fees for two-legged execution
    costs = position_changes * fee_rate * 2

    strategy_returns = spread_return - costs

    # Equity curve
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    total_fees = float((costs * initial_capital).sum())

    # Extract trades
    trades = _extract_pair_trades(positions, df_a["close"], df_b["close"], equity_curve)

    return {
        "equity_curve": equity_curve,
        "positions": positions,
        "strategy_returns": strategy_returns,
        "trades": trades,
        "fees_paid": total_fees,
    }


def _extract_pair_trades(
    positions: pd.Series,
    close_a: pd.Series,
    close_b: pd.Series,
    equity: pd.Series,
) -> list[dict]:
    """Extract trades from position changes in pairs strategy."""
    pos_values = positions.values
    changes = np.where(pos_values[1:] != pos_values[:-1])[0] + 1

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

        if i + 1 < len(changes):
            exit_idx = changes[i + 1]
        else:
            exit_idx = len(pos_values) - 1

        entry_equity = float(equity.iloc[enter_idx])
        exit_equity = float(equity.iloc[exit_idx])
        pnl_pct = (exit_equity / entry_equity - 1) * 100 if entry_equity > 0 else 0.0

        trades.append({
            "entry_time": str(idx[enter_idx]),
            "exit_time": str(idx[exit_idx]),
            "direction": "long_spread" if direction == 1 else "short_spread",
            "entry_price": round(float(close_a.iloc[enter_idx]), 2),
            "exit_price": round(float(close_a.iloc[exit_idx]), 2),
            "pnl_pct": round(pnl_pct, 4),
            "bars_held": int(exit_idx - enter_idx),
        })

        i += 1
        if i < len(changes) and changes[i] == exit_idx and pos_values[exit_idx] == 0:
            i += 1

    return trades
