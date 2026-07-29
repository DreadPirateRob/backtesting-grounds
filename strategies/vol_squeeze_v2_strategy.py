import pandas as pd
import numpy as np

from backtester.strategy import Strategy, bollinger_bands, keltner_channels, ema


class VolSqueezeV2Strategy(Strategy):
    """Volatility Squeeze V2: Bollinger Bands inside Keltner Channels.

    Detects when BB contracts inside KC (squeeze state), indicating compressed
    volatility. Enters in the direction of momentum when the squeeze fires
    (BB expands back outside KC).

    Squeeze condition: BB upper < KC upper AND BB lower > KC lower
    Fire condition: BB expands outside KC after being in squeeze for min_squeeze_bars
    Direction: EMA momentum (close vs EMA, plus EMA slope confirmation)

    Designed for 15min-30min timeframes where squeezes correspond to real
    session-based consolidation (Asia -> London/NY breakout).
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_ema_period: int = 20,
        kc_atr_period: int = 14,
        kc_atr_mult: float = 1.5,
        min_squeeze_bars: int = 6,
        momentum_period: int = 12,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_ema_period = kc_ema_period
        self.kc_atr_period = kc_atr_period
        self.kc_atr_mult = kc_atr_mult
        self.min_squeeze_bars = min_squeeze_bars
        self.momentum_period = momentum_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )

        # Keltner Channels
        kc_upper, kc_mid, kc_lower = keltner_channels(
            df, self.kc_ema_period, self.kc_atr_period, self.kc_atr_mult
        )

        # Squeeze detection: BB is inside KC
        in_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # Count consecutive squeeze bars
        squeeze_groups = (~in_squeeze).cumsum()
        consecutive_squeeze = in_squeeze.groupby(squeeze_groups).cumsum()

        # Squeeze was active for minimum duration
        qualified_squeeze = consecutive_squeeze >= self.min_squeeze_bars

        # Squeeze fire: was in qualified squeeze, now BB expands outside KC
        squeeze_fired = qualified_squeeze.shift(1, fill_value=False) & ~in_squeeze

        # Momentum direction: EMA slope + price relative to EMA
        ema_val = ema(df["close"], self.momentum_period)
        ema_slope = ema_val - ema_val.shift(1)
        bullish = (df["close"] > ema_val) & (ema_slope > 0)
        bearish = (df["close"] < ema_val) & (ema_slope < 0)

        # Generate signals on squeeze fire in direction of momentum
        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[squeeze_fired & bullish] = 1
        signals[squeeze_fired & bearish] = -1

        # Forward-fill to hold position until opposite signal fires
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "bb_period": [15, 20, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "kc_ema_period": [15, 20, 30],
            "kc_atr_period": [10, 14, 20],
            "kc_atr_mult": [1.0, 1.5, 2.0],
            "min_squeeze_bars": [4, 6, 8],
            "momentum_period": [8, 12, 20],
        }
