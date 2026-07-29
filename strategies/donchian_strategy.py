import pandas as pd

from backtester.strategy import Strategy


class DonchianStrategy(Strategy):
    """Donchian Channel Breakout (Turtle Trading inspired).

    Goes long when price breaks above the highest high of N periods.
    Goes short when price breaks below the lowest low of N periods.
    Uses a shorter exit channel to lock profits.
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        self.entry_period = entry_period
        self.exit_period = exit_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper = df["high"].rolling(self.entry_period).max()
        lower = df["low"].rolling(self.entry_period).min()
        exit_upper = df["high"].rolling(self.exit_period).max()
        exit_lower = df["low"].rolling(self.exit_period).min()

        signals = pd.Series(0, index=df.index, dtype="int8")

        position = 0
        for i in range(self.entry_period, len(df)):
            if position == 0:
                if df["close"].iloc[i] > upper.iloc[i - 1]:
                    position = 1
                elif df["close"].iloc[i] < lower.iloc[i - 1]:
                    position = -1
            elif position == 1:
                if df["close"].iloc[i] < exit_lower.iloc[i - 1]:
                    position = 0
                elif df["close"].iloc[i] < lower.iloc[i - 1]:
                    position = -1
            elif position == -1:
                if df["close"].iloc[i] > exit_upper.iloc[i - 1]:
                    position = 0
                elif df["close"].iloc[i] > upper.iloc[i - 1]:
                    position = 1
            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "entry_period": [10, 15, 20, 30, 40, 55],
            "exit_period": [5, 7, 10, 15, 20],
        }
