import pandas as pd

from backtester.strategy import Strategy, ema


class MomentumRocStrategy(Strategy):
    """Rate of Change (ROC) Momentum strategy.

    Uses price ROC to identify strong momentum. Goes long when ROC exceeds
    a positive threshold and a trend filter (EMA) confirms. Short on negative ROC.
    """

    def __init__(self, roc_period: int = 14, threshold: float = 2.0, ema_filter: int = 50):
        self.roc_period = roc_period
        self.threshold = threshold
        self.ema_filter = ema_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        roc = ((close - close.shift(self.roc_period)) / close.shift(self.roc_period)) * 100
        ema_val = ema(close, self.ema_filter)

        signals = pd.Series(0, index=df.index, dtype="int8")
        # Long: strong positive momentum + price above EMA
        signals[(roc > self.threshold) & (close > ema_val)] = 1
        # Short: strong negative momentum + price below EMA
        signals[(roc < -self.threshold) & (close < ema_val)] = -1

        # Forward-fill to hold
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")
        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "roc_period": [5, 10, 14, 20, 30],
            "threshold": [1.0, 2.0, 3.0, 5.0],
            "ema_filter": [20, 50, 100, 200],
        }
