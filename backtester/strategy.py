from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Base class for trading strategies."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of signals: 1 = long, -1 = short, 0 = flat."""
        ...

    @abstractmethod
    def param_grid(self) -> dict[str, list]:
        """Return parameter grid for sweep, e.g. {"period": [7,14,21]}."""
        ...


# ---------------------------------------------------------------------------
# Indicator functions
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using exponential moving average of gains/losses."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator.

    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series, period: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns (upper, middle, lower).
    """
    middle = sma(series, period)
    rolling_std = series.rolling(period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def keltner_channels(
    df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 1.5
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels: EMA ± ATR multiplier.

    Returns (upper, middle, lower).
    """
    middle = ema(df["close"], ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    return upper, middle, lower


def vwap_daily(df: pd.DataFrame, bands: bool = True) -> pd.DataFrame:
    """Daily-resetting VWAP with optional sigma bands.

    Resets at 00:00 UTC each day. Requires 'volume' column.

    Returns DataFrame with columns: vwap, vwap_upper, vwap_lower (if bands=True).
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    # Group by calendar date for daily reset
    date_groups = df.index.date

    cum_tp_vol = tp_vol.groupby(date_groups).cumsum()
    cum_vol = df["volume"].groupby(date_groups).cumsum()

    vwap_vals = cum_tp_vol / cum_vol.replace(0, np.nan)

    result = pd.DataFrame({"vwap": vwap_vals}, index=df.index)

    if bands:
        # Volume-weighted variance for sigma bands
        sq_diff = (typical_price - vwap_vals) ** 2 * df["volume"]
        cum_sq_diff = sq_diff.groupby(date_groups).cumsum()
        vwap_std = np.sqrt(cum_sq_diff / cum_vol.replace(0, np.nan))
        result["vwap_upper"] = vwap_vals + vwap_std
        result["vwap_lower"] = vwap_vals - vwap_std

    return result


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with DI+ and DI-.

    Returns DataFrame with columns: adx, di_plus, di_minus.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # True range
    atr_vals = atr(df, period)

    # Smoothed DI
    alpha = 1 / period
    smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    di_plus = 100 * smooth_plus / atr_vals.replace(0, np.nan)
    di_minus = 100 * smooth_minus / atr_vals.replace(0, np.nan)

    # DX and ADX
    di_sum = di_plus + di_minus
    dx = 100 * (di_plus - di_minus).abs() / di_sum.replace(0, np.nan)
    adx_vals = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx_vals, "di_plus": di_plus, "di_minus": di_minus},
        index=df.index,
    )


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator.

    Returns (%K, %D) where %D is SMA of %K.
    """
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    denom = highest - lowest
    k = 100 * (df["close"] - lowest) / denom.replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def stoch_rsi(
    series: pd.Series, rsi_period: int = 14, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI.

    Applies Stochastic formula to RSI values instead of price.
    Returns (stoch_rsi_k, stoch_rsi_d).
    """
    rsi_vals = rsi(series, rsi_period)
    lowest = rsi_vals.rolling(k_period).min()
    highest = rsi_vals.rolling(k_period).max()
    denom = highest - lowest
    k = 100 * (rsi_vals - lowest) / denom.replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def cvd(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume Delta.

    Requires 'taker_buy_base_volume' column. Falls back to close-based
    approximation (buy if close > open, sell otherwise) if not available.
    """
    if "taker_buy_base_volume" in df.columns:
        buy_vol = df["taker_buy_base_volume"]
        sell_vol = df["volume"] - buy_vol
        delta = buy_vol - sell_vol
    else:
        # Approximation: assign volume direction based on bar direction
        direction = np.sign(df["close"] - df["open"])
        delta = df["volume"] * direction

    return delta.cumsum()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (df["volume"] * direction).cumsum()


def funding_rate_zscore(funding: pd.Series, period: int = 30) -> pd.Series:
    """Z-score of funding rate over rolling window.
    High positive = crowded longs (contrarian short signal).
    High negative = crowded shorts (contrarian long signal)."""
    mean = funding.rolling(period).mean()
    std = funding.rolling(period).std()
    return (funding - mean) / std.replace(0, np.nan)


def oi_momentum(oi: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """OI momentum: fast EMA of OI change vs slow EMA.
    Rising OI + rising price = trend confirmation.
    Rising OI + falling price = bearish pressure."""
    oi_pct = oi.pct_change()
    fast_ema = ema(oi_pct, fast)
    slow_ema = ema(oi_pct, slow)
    return fast_ema - slow_ema


def oi_price_divergence(close: pd.Series, oi: pd.Series, period: int = 20) -> pd.Series:
    """Divergence between price direction and OI direction.
    Returns: +1 (price up, OI down), -1 (price down, OI up), 0 (aligned)."""
    price_direction = np.sign(close - sma(close, period))
    oi_direction = np.sign(oi - sma(oi, period))
    divergence = price_direction * oi_direction
    return pd.Series(np.where(divergence < 0, -price_direction, 0), index=close.index)
