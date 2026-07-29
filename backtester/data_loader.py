import numpy as np
import pandas as pd


def enrich_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add session and time-of-day columns to a DataFrame with UTC DatetimeIndex.

    Adds columns:
    - hour: hour of day (0-23)
    - session: 'asian' | 'european' | 'us' | 'eu_us_overlap'
    - is_weekend: True for Saturday/Sunday
    - is_peak_liquidity: True during 08:00-20:00 UTC weekdays
    """
    hour = df.index.hour
    day_of_week = df.index.dayofweek  # 0=Mon, 6=Sun

    # Session classification (UTC)
    conditions = [
        (hour >= 0) & (hour < 8),
        (hour >= 8) & (hour < 13),
        (hour >= 13) & (hour < 17),
        (hour >= 17) & (hour < 22),
        hour >= 22,
    ]
    choices = ["asian", "european", "eu_us_overlap", "us", "asian"]
    session = pd.Series(np.select(conditions, choices, default="asian"), index=df.index)

    df = df.copy()
    df["hour"] = hour
    df["session"] = session
    df["is_weekend"] = day_of_week >= 5
    df["is_peak_liquidity"] = (hour >= 8) & (hour < 20) & (day_of_week < 5)

    return df


def load_candles(
    path: str,
    start: str | None = None,
    end: str | None = None,
    resample: str | None = None,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load OHLCV candles from a Binance klines CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    start, end : str, optional
        Date strings (YYYY-MM-DD) to filter the data range.
    resample : str, optional
        Pandas frequency string to resample to (e.g. "5min", "1h", "1D").
    extra_columns : list[str], optional
        Additional columns to load (e.g. ["taker_buy_base_volume", "trades"]).

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
        (plus any extra_columns requested).
    """
    base_cols = ["open_time", "open", "high", "low", "close", "volume"]
    base_dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    }

    if extra_columns:
        base_cols = base_cols + [c for c in extra_columns if c not in base_cols]
        for c in extra_columns:
            if c not in base_dtypes:
                base_dtypes[c] = "float64"

    df = pd.read_csv(path, usecols=base_cols, dtype=base_dtypes)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df.index.name = "time"

    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]

    if resample is not None:
        agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        if extra_columns:
            for c in extra_columns:
                if c not in agg_dict:
                    agg_dict[c] = "sum"
        df = df.resample(resample).agg(agg_dict).dropna(subset=["close"])

    return df


def load_funding_rates(
    path: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load funding rate data. Returns Series with DatetimeIndex, values = funding rate."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index.name = "time"
    df["funding_rate"] = df["funding_rate"].astype(float)

    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]

    return df["funding_rate"]


def load_open_interest(
    path: str,
    start: str | None = None,
    end: str | None = None,
    resample: str | None = None,
) -> pd.DataFrame:
    """Load open interest data. Returns DataFrame with open_interest and open_interest_value."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index.name = "time"
    df["open_interest"] = df["open_interest"].astype(float)
    df["open_interest_value"] = df["open_interest_value"].astype(float)

    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]

    if resample is not None:
        df = df.resample(resample).last().dropna()

    return df


def merge_funding_oi(
    df: pd.DataFrame,
    funding_path: str | None = None,
    oi_path: str | None = None,
) -> pd.DataFrame:
    """Merge funding rate and OI data into an existing OHLCV DataFrame."""
    df = df.copy()

    if funding_path is not None:
        funding = load_funding_rates(funding_path)
        df["funding_rate"] = funding.reindex(df.index, method="ffill")

    if oi_path is not None:
        oi = load_open_interest(oi_path)
        df["open_interest"] = oi["open_interest"].reindex(df.index, method="ffill")
        df["oi_change_pct"] = df["open_interest"].pct_change() * 100

    return df
