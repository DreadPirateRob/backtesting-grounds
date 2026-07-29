import numpy as np
import pandas as pd

from backtester.strategy import Strategy, atr, ema


class SupertrendStrategy(Strategy):
    """Supertrend trend-following strategy.

    Uses ATR-based trailing stop bands. When price closes above the upper band,
    go long; when below the lower band, go short. The bands flip on trend reversal.
    """

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0):
        self.atr_period = atr_period
        self.multiplier = multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        atr_val = atr(df, self.atr_period)
        hl2 = (df["high"] + df["low"]) / 2

        upper_band = hl2 + self.multiplier * atr_val
        lower_band = hl2 - self.multiplier * atr_val

        supertrend = pd.Series(np.nan, index=df.index)
        direction = pd.Series(1, index=df.index, dtype="int8")

        for i in range(1, len(df)):
            # Adjust bands based on previous values
            if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["close"].iloc[i - 1] < lower_band.iloc[i - 1]:
                pass  # keep lower_band as is
            else:
                lower_band.iloc[i] = lower_band.iloc[i - 1]

            if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["close"].iloc[i - 1] > upper_band.iloc[i - 1]:
                pass  # keep upper_band as is
            else:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

            # Determine direction
            if direction.iloc[i - 1] == 1:
                if df["close"].iloc[i] < lower_band.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = 1
            else:
                if df["close"].iloc[i] > upper_band.iloc[i]:
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = -1

        signals = direction.copy()
        return signals.astype("int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "atr_period": [7, 10, 14, 20, 30],
            "multiplier": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        }
