import pandas as pd

from backtester.strategy import Strategy, rsi


class RsiStrategy(Strategy):
    """RSI mean-reversion strategy.

    Goes long when RSI drops below oversold, short when RSI rises above overbought.
    Holds position until the opposite signal fires.
    """

    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi_values = rsi(df["close"], self.period)

        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[rsi_values < self.oversold] = 1
        signals[rsi_values > self.overbought] = -1

        # Forward-fill to hold position until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "period": [7, 10, 14, 21, 28],
            "overbought": [65, 70, 75, 80],
            "oversold": [20, 25, 30, 35],
        }
