import pandas as pd

from backtester.strategy import Strategy, atr, sma


class AtrBreakoutStrategy(Strategy):
    """ATR breakout strategy.

    Goes long when price moves above SMA + N*ATR (upside breakout).
    Goes short when price moves below SMA - N*ATR (downside breakout).
    Mean-reverts back to flat when price returns within the bands.
    """

    def __init__(self, sma_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0):
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mid = sma(df["close"], self.sma_period)
        atr_val = atr(df, self.atr_period)
        upper = mid + self.atr_mult * atr_val
        lower = mid - self.atr_mult * atr_val

        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[df["close"] > upper] = 1
        signals[df["close"] < lower] = -1

        # Forward-fill to hold position until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "sma_period": [10, 15, 20, 30, 40],
            "atr_period": [10, 14, 20],
            "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
        }
