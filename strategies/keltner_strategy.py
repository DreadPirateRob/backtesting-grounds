import pandas as pd

from backtester.strategy import Strategy, ema, atr


class KeltnerStrategy(Strategy):
    """Keltner Channel Breakout strategy.

    Uses EMA-based channels with ATR width. Goes long on upper break,
    short on lower break, exits on middle (EMA) crossback.
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mid = ema(df["close"], self.ema_period)
        atr_val = atr(df, self.atr_period)
        upper = mid + self.atr_mult * atr_val
        lower = mid - self.atr_mult * atr_val

        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[df["close"] > upper] = 1
        signals[df["close"] < lower] = -1

        # Forward-fill to hold
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")
        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "ema_period": [10, 15, 20, 30, 40],
            "atr_period": [10, 14, 20],
            "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
        }
