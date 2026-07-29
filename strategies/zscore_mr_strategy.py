import pandas as pd

from backtester.strategy import Strategy, sma


class ZscoreMrStrategy(Strategy):
    """Z-Score Mean Reversion strategy.

    Computes z-score of price relative to a rolling window.
    Goes long when price is N standard deviations below mean,
    short when N standard deviations above. Exits at mean reversion.
    """

    def __init__(self, lookback: int = 50, entry_z: float = 2.0, exit_z: float = 0.0):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()
        zscore = (close - rolling_mean) / rolling_std

        signals = pd.Series(0, index=df.index, dtype="int8")

        position = 0
        for i in range(self.lookback, len(df)):
            z = zscore.iloc[i]
            if position == 0:
                if z < -self.entry_z:
                    position = 1
                elif z > self.entry_z:
                    position = -1
            elif position == 1:
                if z > self.exit_z:
                    position = 0
                elif z > self.entry_z:
                    position = -1
            elif position == -1:
                if z < -self.exit_z:
                    position = 0
                elif z < -self.entry_z:
                    position = 1
            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "lookback": [20, 30, 50, 75, 100, 150],
            "entry_z": [1.5, 2.0, 2.5, 3.0],
            "exit_z": [0.0, 0.25, 0.5],
        }
