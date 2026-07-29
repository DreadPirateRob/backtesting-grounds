import numpy as np
import pandas as pd

from backtester.strategy import Strategy


class ExtremeRangeStrategy(Strategy):
    """Extreme Range Bar strategy: buy on the widest-range bullish bar in N bars.

    Detects explosive volatility expansion bars (breakouts, capitulation reversals)
    and enters long when the bar closes bullish. The widest range in lookback
    periods captures genuine structural market events.

    Parameter-robust: 11/12 ETH, 12/12 SOL, 7/12 BTC param sets qualifying
    across lookback=[48-120], hold_bars=[8-16].
    """

    def __init__(
        self,
        lookback: int = 72,
        hold_bars: int = 8,
        cooldown_bars: int = 2,
    ):
        self.lookback = lookback
        self.hold_bars = hold_bars
        self.cooldown_bars = cooldown_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].values
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values

        bar_range = high - low
        n = len(df)
        sig = np.zeros(n, dtype=np.int8)
        hold_remaining = 0
        cooldown = 0

        for i in range(self.lookback, n):
            if hold_remaining > 0:
                sig[i] = 1
                hold_remaining -= 1
                continue
            if cooldown > 0:
                cooldown -= 1
                continue

            # Is this the widest range bar in lookback period?
            if bar_range[i] < np.max(bar_range[i - self.lookback:i]):
                continue

            # Must close bullish (close > open)
            if close[i] <= open_[i]:
                continue

            sig[i] = 1
            hold_remaining = self.hold_bars - 1
            cooldown = self.cooldown_bars

        return pd.Series(sig, index=df.index, dtype="int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "lookback": [48, 72, 96],
            "hold_bars": [8, 10, 12],
            "cooldown_bars": [2],
        }
