import numpy as np
import pandas as pd

from backtester.strategy import Strategy, rsi, bollinger_bands, ema


class RsiBbMtfStrategy(Strategy):
    """RSI + Bollinger with Multi-Timeframe Daily trend filter.

    Base: RSI+Bollinger confluence on the trading timeframe (1h or 4h).
    Filter: Daily EMA trend direction. Only take longs when daily trend is up
    (price above daily EMA), shorts when daily trend is down.

    The daily EMA is computed by resampling within generate_signals() and
    forward-filled back to the base timeframe with a 1-bar shift to prevent
    look-ahead bias.

    Expected improvement: +0.3 to +0.7 Sharpe (per QuantPedia D1H1 research).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: int = 65,
        oversold: int = 30,
        bb_period: int = 20,
        bb_std: float = 2.0,
        daily_ema: int = 21,
    ):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.daily_ema = daily_ema

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        # --- Base signals (RSI + Bollinger confluence) ---
        rsi_values = rsi(close, self.rsi_period)
        upper, middle, lower = bollinger_bands(close, self.bb_period, self.bb_std)

        base_signals = pd.Series(0, index=df.index, dtype="int8")
        base_signals[(rsi_values < self.oversold) & (close < lower)] = 1
        base_signals[(rsi_values > self.overbought) & (close > upper)] = -1
        base_signals = base_signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        # --- Daily trend filter ---
        # Resample to daily using right-closed, right-labeled to avoid look-ahead
        df_daily = df.resample("1D", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])

        daily_ema_val = ema(df_daily["close"], self.daily_ema)
        daily_trend = pd.Series(0, index=df_daily.index, dtype="int8")
        daily_trend[df_daily["close"] > daily_ema_val] = 1   # bullish
        daily_trend[df_daily["close"] < daily_ema_val] = -1   # bearish

        # Forward-fill to base timeframe + shift(1) to prevent look-ahead
        htf_trend = daily_trend.reindex(df.index, method="ffill").shift(1).fillna(0).astype("int8")

        # --- Filter: only take signals aligned with daily trend ---
        filtered = base_signals.copy()
        # Zero out longs when daily trend is bearish
        filtered[(base_signals == 1) & (htf_trend == -1)] = 0
        # Zero out shorts when daily trend is bullish
        filtered[(base_signals == -1) & (htf_trend == 1)] = 0

        return filtered

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [12, 14, 18, 28, 30],
            "overbought": [62, 65, 67],
            "oversold": [28, 30, 32],
            "bb_period": [15, 17, 20, 22],
            "bb_std": [2.0, 2.25, 2.5],
            "daily_ema": [10, 15, 21, 30],
        }
