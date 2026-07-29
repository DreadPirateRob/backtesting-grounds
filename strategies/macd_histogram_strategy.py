import pandas as pd

from backtester.strategy import Strategy, macd, ema


class MacdHistogramStrategy(Strategy):
    """MACD Histogram Divergence strategy.

    Instead of simple MACD line/signal crossover, this uses histogram momentum.
    Enters when histogram reverses direction (turns from negative to less negative
    or positive to less positive), confirming with EMA trend filter.
    Catches early momentum shifts before the actual MACD cross.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 ema_filter: int = 100):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.ema_filter = ema_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        _, _, hist = macd(close, self.fast, self.slow, self.signal)
        trend = ema(close, self.ema_filter)

        # Histogram direction change
        hist_rising = hist > hist.shift(1)
        hist_falling = hist < hist.shift(1)

        signals = pd.Series(0, index=df.index, dtype="int8")

        # Long: histogram turning up + price above trend
        signals[(hist_rising) & (hist.shift(1) < 0) & (close > trend)] = 1
        # Short: histogram turning down + price below trend
        signals[(hist_falling) & (hist.shift(1) > 0) & (close < trend)] = -1

        # Forward-fill to hold
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")
        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "fast": [8, 10, 12, 15],
            "slow": [20, 26, 30],
            "signal": [5, 7, 9, 12],
            "ema_filter": [50, 100, 150, 200],
        }
