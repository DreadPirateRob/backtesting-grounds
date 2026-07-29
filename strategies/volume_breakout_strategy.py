import pandas as pd

from backtester.strategy import Strategy, ema, atr


class VolumeBreakoutStrategy(Strategy):
    """Volume-confirmed breakout strategy.

    Combines price breakout with volume surge confirmation.
    Enters only when price breaks a recent range AND volume is significantly
    above average. This filters out false breakouts that lack conviction.
    """

    def __init__(self, price_lookback: int = 20, vol_lookback: int = 20,
                 vol_mult: float = 1.5, ema_period: int = 50):
        self.price_lookback = price_lookback
        self.vol_lookback = vol_lookback
        self.vol_mult = vol_mult
        self.ema_period = ema_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]

        # Price range
        high_channel = df["high"].rolling(self.price_lookback).max()
        low_channel = df["low"].rolling(self.price_lookback).min()

        # Volume confirmation
        avg_vol = volume.rolling(self.vol_lookback).mean()
        high_volume = volume > (avg_vol * self.vol_mult)

        # Trend filter
        trend = ema(close, self.ema_period)

        signals = pd.Series(0, index=df.index, dtype="int8")

        # Long: price breaks above range + high volume + above EMA
        long_break = (close > high_channel.shift(1)) & high_volume & (close > trend)
        # Short: price breaks below range + high volume + below EMA
        short_break = (close < low_channel.shift(1)) & high_volume & (close < trend)

        signals[long_break] = 1
        signals[short_break] = -1

        # Forward-fill to hold
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")
        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "price_lookback": [10, 15, 20, 30, 40],
            "vol_lookback": [10, 20, 30],
            "vol_mult": [1.2, 1.5, 2.0, 2.5],
            "ema_period": [20, 50, 100],
        }
