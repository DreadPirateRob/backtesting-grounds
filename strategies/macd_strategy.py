import pandas as pd

from backtester.strategy import Strategy, macd


class MacdStrategy(Strategy):
    """MACD crossover momentum strategy.

    Goes long when MACD line crosses above signal line.
    Goes short (or flat in long-only mode) when MACD crosses below signal.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        macd_line, signal_line, _ = macd(df["close"], self.fast, self.slow, self.signal)

        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[macd_line > signal_line] = 1
        signals[macd_line < signal_line] = -1

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "fast": [8, 10, 12, 15],
            "slow": [20, 26, 30, 35],
            "signal": [5, 7, 9, 12],
        }
