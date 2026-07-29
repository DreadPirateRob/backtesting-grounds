import numpy as np
import pandas as pd

from backtester.strategy import Strategy, atr


class AtrExpansionStrategy(Strategy):
    """ATR Expansion Momentum strategy.

    Detects single-bar range expansions (bar range > mult * ATR) and trades
    in the direction of the bar's close-vs-open. Holds for a fixed number
    of bars then goes flat. Cooldown period prevents immediate re-entry.

    Rationale: large-range bars at 1h indicate institutional order flow or
    breakout momentum. The bar direction predicts continuation over the
    next few bars. High ATR multiplier filters noise, keeping only
    genuinely unusual expansions (~88 trades/yr at 1h with mult=3.0).
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 3.0,
        hold_bars: int = 8,
        cooldown_bars: int = 1,
        use_close_direction: bool = True,
    ):
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.hold_bars = hold_bars
        self.cooldown_bars = cooldown_bars
        self.use_close_direction = use_close_direction

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        atr_val = atr(df, self.atr_period)
        bar_range = df["high"] - df["low"]
        expansion = bar_range > self.atr_mult * atr_val

        if self.use_close_direction:
            bar_dir = np.sign(df["close"].values - df["open"].values)
        else:
            # Use close vs previous close
            bar_dir = np.sign(df["close"].values - np.roll(df["close"].values, 1))
            bar_dir[0] = 0

        n = len(df)
        sig = np.zeros(n, dtype=np.int8)
        expansion_vals = expansion.values
        hold_remaining = 0
        cooldown = 0

        for i in range(n):
            if hold_remaining > 0:
                sig[i] = sig[i - 1]
                hold_remaining -= 1
                continue

            if cooldown > 0:
                cooldown -= 1
                continue

            if expansion_vals[i] and bar_dir[i] != 0:
                sig[i] = int(bar_dir[i])
                hold_remaining = self.hold_bars - 1
            elif i > 0 and sig[i - 1] != 0:
                # Position just expired, enter cooldown
                cooldown = self.cooldown_bars

        return pd.Series(sig, index=df.index, dtype="int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "atr_period": [14, 24],
            "atr_mult": [2.5, 3.0, 3.5, 4.0],
            "hold_bars": [4, 6, 8, 12],
            "cooldown_bars": [0, 1, 2],
            "use_close_direction": [True],
        }
