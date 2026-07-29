import pandas as pd
import numpy as np

from backtester.strategy import Strategy, bollinger_bands, ema


class BbSqueezeStrategy(Strategy):
    """Bollinger Band squeeze breakout strategy.

    Detects volatility compression (BB width falls below threshold percentile)
    then enters in the direction of momentum when BB width expands.

    - Squeeze: BB width < squeeze_pctl percentile of rolling BB width
    - Expansion: BB width rises above expansion_pctl percentile
    - Direction: determined by EMA slope (price vs EMA)
    """

    def __init__(
        self,
        bb_period: int = 120,
        bb_std: float = 2.0,
        squeeze_lookback: int = 480,
        squeeze_pctl: float = 20.0,
        expansion_pctl: float = 50.0,
        ema_period: int = 60,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pctl = squeeze_pctl
        self.expansion_pctl = expansion_pctl
        self.ema_period = ema_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper, middle, lower = bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )
        bb_width = (upper - lower) / middle

        # Rolling percentile of BB width
        width_pctl = bb_width.rolling(self.squeeze_lookback).rank(pct=True) * 100

        # Detect squeeze (low volatility) and expansion
        in_squeeze = width_pctl < self.squeeze_pctl
        expanding = width_pctl > self.expansion_pctl

        # Was recently in squeeze (within last squeeze_lookback/4 bars)
        lookback = max(1, self.squeeze_lookback // 4)
        was_squeezed = in_squeeze.rolling(lookback).max().fillna(0).astype(bool)

        # Direction: EMA trend
        ema_val = ema(df["close"], self.ema_period)
        bullish = df["close"] > ema_val
        bearish = df["close"] < ema_val

        signals = pd.Series(0, index=df.index, dtype="int8")
        # Enter when expanding out of squeeze, in direction of trend
        signals[was_squeezed & expanding & bullish] = 1
        signals[was_squeezed & expanding & bearish] = -1

        # Forward-fill to hold position until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "bb_period": [60, 120, 240],
            "bb_std": [1.5, 2.0, 2.5],
            "squeeze_lookback": [240, 480, 720],
            "squeeze_pctl": [15.0, 20.0, 30.0],
            "expansion_pctl": [40.0, 50.0, 60.0],
            "ema_period": [30, 60, 120],
        }
