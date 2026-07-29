import pandas as pd

from backtester.strategy import Strategy, rsi


class DualRsiStrategy(Strategy):
    """Dual-timeframe RSI for minute-level data.

    Slow RSI (long period) determines directional bias.
    Fast RSI (short period) provides entry timing within that bias.

    - Long: slow RSI < slow_oversold AND fast RSI < fast_oversold
    - Short: slow RSI > slow_overbought AND fast RSI > fast_overbought
    """

    def __init__(
        self,
        slow_period: int = 60,
        fast_period: int = 10,
        slow_overbought: int = 70,
        slow_oversold: int = 30,
        fast_overbought: int = 75,
        fast_oversold: int = 25,
    ):
        self.slow_period = slow_period
        self.fast_period = fast_period
        self.slow_overbought = slow_overbought
        self.slow_oversold = slow_oversold
        self.fast_overbought = fast_overbought
        self.fast_oversold = fast_oversold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        slow_rsi = rsi(df["close"], self.slow_period)
        fast_rsi = rsi(df["close"], self.fast_period)

        signals = pd.Series(0, index=df.index, dtype="int8")
        # Confluence: slow RSI sets direction, fast RSI times entry
        signals[(slow_rsi < self.slow_oversold) & (fast_rsi < self.fast_oversold)] = 1
        signals[
            (slow_rsi > self.slow_overbought) & (fast_rsi > self.fast_overbought)
        ] = -1

        # Forward-fill to hold position until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "slow_period": [30, 60, 120, 240],
            "fast_period": [5, 10, 15, 20],
            "slow_overbought": [65, 70, 75],
            "slow_oversold": [25, 30, 35],
            "fast_overbought": [70, 75, 80],
            "fast_oversold": [20, 25, 30],
        }
