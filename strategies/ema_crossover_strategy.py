import pandas as pd

from backtester.strategy import Strategy, ema


class EmaCrossoverStrategy(Strategy):
    """EMA crossover trend-following strategy.

    Goes long when fast EMA crosses above slow EMA.
    Goes short (or flat in long-only mode) when fast crosses below slow.
    """

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_ema = ema(df["close"], self.fast)
        slow_ema = ema(df["close"], self.slow)

        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[fast_ema > slow_ema] = 1
        signals[fast_ema < slow_ema] = -1

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "fast": [5, 10, 15, 20, 30],
            "slow": [30, 50, 75, 100, 150, 200],
        }
