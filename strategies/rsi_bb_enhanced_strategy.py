import numpy as np
import pandas as pd

from backtester.strategy import Strategy, rsi, bollinger_bands, ema


class RsiBbEnhancedStrategy(Strategy):
    """RSI + Bollinger with volume confirmation + session + weekend filters.

    Enhancements over base RSI+Bollinger:
    1. Volume confirmation: only enter when bar volume > N * rolling average
    2. Session filter: only trade during active hours (08:00-20:00 UTC)
    3. Weekend filter: go flat during weekends (Sat-Sun)
    4. Daily trend filter (MTF): only trade in direction of daily EMA

    Combines the highest-priority improvements from SKILL.md research.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: int = 65,
        oversold: int = 30,
        bb_period: int = 20,
        bb_std: float = 2.0,
        vol_mult: float = 1.5,
        vol_lookback: int = 20,
        daily_ema: int = 21,
        use_session_filter: int = 1,
        use_weekend_filter: int = 1,
    ):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.vol_mult = vol_mult
        self.vol_lookback = vol_lookback
        self.daily_ema = daily_ema
        self.use_session_filter = use_session_filter
        self.use_weekend_filter = use_weekend_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        # --- Base signals (RSI + Bollinger confluence) ---
        rsi_values = rsi(close, self.rsi_period)
        upper, middle, lower = bollinger_bands(close, self.bb_period, self.bb_std)

        raw_triggers = pd.Series(0, index=df.index, dtype="int8")
        raw_triggers[(rsi_values < self.oversold) & (close < lower)] = 1
        raw_triggers[(rsi_values > self.overbought) & (close > upper)] = -1

        # --- Volume confirmation filter ---
        avg_vol = df["volume"].rolling(self.vol_lookback).mean()
        vol_ok = df["volume"] > (self.vol_mult * avg_vol)
        # Only allow NEW entries when volume is high enough
        # (existing positions can persist)
        vol_triggers = raw_triggers.copy()
        vol_triggers[(raw_triggers != 0) & ~vol_ok] = 0

        # --- Daily trend filter (MTF) ---
        df_daily = df.resample("1D", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])

        daily_ema_val = ema(df_daily["close"], self.daily_ema)
        daily_trend = pd.Series(0, index=df_daily.index, dtype="int8")
        daily_trend[df_daily["close"] > daily_ema_val] = 1
        daily_trend[df_daily["close"] < daily_ema_val] = -1

        htf_trend = daily_trend.reindex(df.index, method="ffill").shift(1).fillna(0).astype("int8")

        # Filter: zero out signals against daily trend
        mtf_triggers = vol_triggers.copy()
        mtf_triggers[(vol_triggers == 1) & (htf_trend == -1)] = 0
        mtf_triggers[(vol_triggers == -1) & (htf_trend == 1)] = 0

        # --- Forward-fill to hold positions ---
        signals = mtf_triggers.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        # --- Session filter (08:00-20:00 UTC) ---
        if self.use_session_filter and hasattr(df.index, 'hour'):
            hour = df.index.hour
            outside_session = (hour < 8) | (hour >= 20)
            # Go flat outside active session
            signals[outside_session] = 0

        # --- Weekend filter ---
        if self.use_weekend_filter and hasattr(df.index, 'dayofweek'):
            is_weekend = df.index.dayofweek >= 5  # Saturday=5, Sunday=6
            signals[is_weekend] = 0

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [14, 18, 28, 30],
            "overbought": [62, 65, 67],
            "oversold": [28, 30, 32],
            "bb_period": [15, 17, 20, 22],
            "bb_std": [2.0, 2.25, 2.5],
            "vol_mult": [1.0, 1.5, 2.0],
            "daily_ema": [15, 21, 30],
        }
