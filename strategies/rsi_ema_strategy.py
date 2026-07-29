import pandas as pd

from backtester.strategy import Strategy, rsi, ema


class RsiEmaStrategy(Strategy):
    """RSI mean-reversion with EMA trend context.

    Uses the slope of a slow EMA as a trend filter:
    - Only takes RSI long signals when the slow EMA is rising (uptrend context)
    - This avoids buying dips in a downtrend while still catching oversold bounces

    The key insight: we can't require price > EMA at the moment RSI is oversold
    (they're contradictory), but we CAN require the broader trend to be up.
    """

    def __init__(
        self,
        rsi_period: int = 30,
        overbought: int = 65,
        oversold: int = 27,
        ema_period: int = 200,
        slope_lookback: int = 48,
    ):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.ema_period = ema_period
        self.slope_lookback = slope_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi_values = rsi(df["close"], self.rsi_period)
        ema_values = ema(df["close"], self.ema_period)
        ema_rising = ema_values.diff(self.slope_lookback) > 0

        signals = pd.Series(0, index=df.index, dtype="int8")
        # Long when RSI oversold AND EMA slope is positive (uptrend)
        signals[(rsi_values < self.oversold) & ema_rising] = 1
        # Exit when RSI overbought
        signals[rsi_values > self.overbought] = -1

        # Forward-fill to hold position until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [22, 26, 30, 34],
            "overbought": [60, 65, 70],
            "oversold": [22, 25, 27, 30],
            "ema_period": [100, 150, 200],
            "slope_lookback": [24, 48, 72],
        }
