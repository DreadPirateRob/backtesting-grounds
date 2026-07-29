import pandas as pd

from backtester.strategy import Strategy, rsi, bollinger_bands


class RsiBollingerStrategy(Strategy):
    """RSI + Bollinger Band confluence strategy.

    Requires both indicators to agree:
    - Long: RSI oversold AND price below lower Bollinger Band
    - Short: RSI overbought AND price above upper Bollinger Band
    """

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: int = 70,
        oversold: int = 30,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi_values = rsi(df["close"], self.rsi_period)
        upper, middle, lower = bollinger_bands(df["close"], self.bb_period, self.bb_std)

        signals = pd.Series(0, index=df.index, dtype="int8")
        # Confluence: both must agree
        signals[(rsi_values < self.oversold) & (df["close"] < lower)] = 1
        signals[(rsi_values > self.overbought) & (df["close"] > upper)] = -1

        # Forward-fill to hold position until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [10, 14, 21, 28],
            "overbought": [65, 70, 75],
            "oversold": [25, 30, 35],
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
        }