import pandas as pd

from backtester.strategy import Strategy, bollinger_bands


class BollingerBounceStrategy(Strategy):
    """Bollinger Band mean-reversion strategy.

    Goes long when price touches the lower band (oversold bounce).
    Goes short when price touches the upper band (overbought rejection).
    Exits at the middle band (SMA).
    """

    def __init__(self, period: int = 20, std: float = 2.0):
        self.period = period
        self.std = std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper, middle, lower = bollinger_bands(df["close"], self.period, self.std)

        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[df["close"] < lower] = 1   # Bounce off lower band -> long
        signals[df["close"] > upper] = -1  # Reject at upper band -> short

        # Forward-fill to hold position until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "period": [10, 15, 20, 25, 30, 40],
            "std": [1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
        }