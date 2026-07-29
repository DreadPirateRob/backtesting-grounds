import numpy as np
import pandas as pd

from backtester.strategy import Strategy


class VwapStrategy(Strategy):
    """VWAP (Volume Weighted Average Price) strategy.

    Computes a rolling VWAP and standard deviation bands.
    Goes long when price dips below lower VWAP band (value area),
    short when price exceeds upper band. Exits at VWAP mean.
    Great for intraday mean-reversion.
    """

    def __init__(self, lookback: int = 50, band_mult: float = 2.0):
        self.lookback = lookback
        self.band_mult = band_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"]

        # Rolling VWAP
        tp_vol = typical_price * vol
        rolling_tp_vol = tp_vol.rolling(self.lookback).sum()
        rolling_vol = vol.rolling(self.lookback).sum()
        vwap = rolling_tp_vol / rolling_vol

        # Rolling VWAP std deviation
        vwap_sq = (typical_price ** 2 * vol).rolling(self.lookback).sum() / rolling_vol
        vwap_std = np.sqrt(np.maximum(vwap_sq - vwap ** 2, 0))

        upper = vwap + self.band_mult * vwap_std
        lower = vwap - self.band_mult * vwap_std

        signals = pd.Series(0, index=df.index, dtype="int8")

        position = 0
        for i in range(self.lookback, len(df)):
            price = df["close"].iloc[i]
            if position == 0:
                if price < lower.iloc[i]:
                    position = 1
                elif price > upper.iloc[i]:
                    position = -1
            elif position == 1:
                if price > vwap.iloc[i]:
                    position = 0
                elif price > upper.iloc[i]:
                    position = -1
            elif position == -1:
                if price < vwap.iloc[i]:
                    position = 0
                elif price < lower.iloc[i]:
                    position = 1
            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "lookback": [20, 30, 50, 75, 100, 150],
            "band_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
        }
