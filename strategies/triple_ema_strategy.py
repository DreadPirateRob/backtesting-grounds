import pandas as pd

from backtester.strategy import Strategy, ema


class TripleEmaStrategy(Strategy):
    """Triple EMA trend-following strategy.

    Uses three EMAs (fast, medium, slow) for trend confirmation.
    Long when fast > medium > slow (strong uptrend).
    Short when fast < medium < slow (strong downtrend).
    Flat when EMAs are mixed (choppy/transitional market).
    Reduces whipsaws compared to dual EMA crossover.
    """

    def __init__(self, fast: int = 8, medium: int = 21, slow: int = 55):
        self.fast = fast
        self.medium = medium
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ema_fast = ema(close, self.fast)
        ema_med = ema(close, self.medium)
        ema_slow = ema(close, self.slow)

        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[(ema_fast > ema_med) & (ema_med > ema_slow)] = 1
        signals[(ema_fast < ema_med) & (ema_med < ema_slow)] = -1

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "fast": [5, 8, 10, 13],
            "medium": [15, 21, 30, 40],
            "slow": [45, 55, 75, 100, 150],
        }
