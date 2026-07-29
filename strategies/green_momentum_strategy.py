import numpy as np
import pandas as pd

from backtester.strategy import Strategy


class GreenMomentumStrategy(Strategy):
    """Consecutive Green Bars Momentum: buy after sustained bullish runs.

    Buys when at least min_green consecutive bars close > open AND the
    cumulative gain over those bars exceeds min_gain_pct. This captures
    strong momentum moves with genuine follow-through.

    Structurally different from dip-buying/mean-reversion strategies —
    this is a momentum/trend-continuation signal with ZERO overlap with
    selloff-based strategies.

    Best cross-asset params: min_green=4, min_gain_pct=4.0, hold=16
      BTC: Sharpe 0.46 (88% windows positive)
      ETH: Sharpe 0.39
      SOL: Sharpe 0.29
    """

    def __init__(
        self,
        min_green: int = 4,
        min_gain_pct: float = 4.0,
        hold_bars: int = 16,
        cooldown_bars: int = 3,
    ):
        self.min_green = min_green
        self.min_gain_pct = min_gain_pct
        self.hold_bars = hold_bars
        self.cooldown_bars = cooldown_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].values
        open_ = df["open"].values
        n = len(df)
        sig = np.zeros(n, dtype=np.int8)
        hold_remaining = 0
        cooldown = 0

        for i in range(self.min_green, n):
            if hold_remaining > 0:
                sig[i] = 1
                hold_remaining -= 1
                continue
            if cooldown > 0:
                cooldown -= 1
                continue

            green_count = 0
            for j in range(i, max(i - 10, -1), -1):
                if close[j] > open_[j]:
                    green_count += 1
                else:
                    break

            if green_count < self.min_green:
                continue

            cum_gain = (close[i] / close[i - green_count] - 1) * 100
            if cum_gain > self.min_gain_pct:
                sig[i] = 1
                hold_remaining = self.hold_bars - 1
                cooldown = self.cooldown_bars

        return pd.Series(sig, index=df.index, dtype="int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "min_green": [3, 4, 5],
            "min_gain_pct": [3.0, 4.0, 5.0],
            "hold_bars": [12, 16],
            "cooldown_bars": [3],
        }
