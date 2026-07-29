import numpy as np
import pandas as pd

from backtester.strategy import Strategy, atr, sma


class CompositeMomentumStrategy(Strategy):
    """Composite momentum strategy requiring multiple confirming signals.

    Combines ATR expansion + volume spike + optional trend filter for
    ultra-selective entries. Each trigger requires:
    1. Bar range > atr_mult * ATR (unusual price expansion)
    2. Volume z-score > vol_z_threshold (unusual volume)
    3. Both signals agree on direction
    4. Optional: price above/below SMA for trend confirmation

    Holds for fixed bars then goes flat. Very selective (~20-60 trades/yr)
    but each trade has multiple confirming signals.
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 2.5,
        vol_zscore_window: int = 48,
        vol_z_threshold: float = 2.0,
        hold_bars: int = 8,
        cooldown_bars: int = 1,
        use_trend_filter: bool = False,
        trend_sma_period: int = 200,
    ):
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.vol_zscore_window = vol_zscore_window
        self.vol_z_threshold = vol_z_threshold
        self.hold_bars = hold_bars
        self.cooldown_bars = cooldown_bars
        self.use_trend_filter = use_trend_filter
        self.trend_sma_period = trend_sma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]

        # --- ATR expansion detection ---
        atr_val = atr(df, self.atr_period)
        bar_range = df["high"] - df["low"]
        range_expansion = bar_range > self.atr_mult * atr_val

        # --- Volume spike detection ---
        vol_mean = volume.rolling(self.vol_zscore_window).mean()
        vol_std = volume.rolling(self.vol_zscore_window).std()
        vol_z = (volume - vol_mean) / vol_std.replace(0, np.nan)
        vol_spike = vol_z > self.vol_z_threshold

        # --- Direction from bar close vs open ---
        bar_dir = np.sign(close.values - df["open"].values)

        # --- Composite trigger: range expansion AND volume spike ---
        trigger = range_expansion.values & vol_spike.values

        # --- Optional trend filter ---
        if self.use_trend_filter:
            trend = sma(close, self.trend_sma_period)
            trend_vals = trend.values
        else:
            trend_vals = None

        n = len(df)
        sig = np.zeros(n, dtype=np.int8)
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

            if trigger[i] and bar_dir[i] != 0:
                direction = int(bar_dir[i])

                # Trend filter: only trade in trend direction
                if trend_vals is not None and not np.isnan(trend_vals[i]):
                    if direction == 1 and close.values[i] < trend_vals[i]:
                        continue  # Skip: long against downtrend
                    if direction == -1 and close.values[i] > trend_vals[i]:
                        continue  # Skip: short against uptrend

                sig[i] = direction
                hold_remaining = self.hold_bars - 1
            elif i > 0 and sig[i - 1] != 0:
                cooldown = self.cooldown_bars

        return pd.Series(sig, index=df.index, dtype="int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "atr_period": [14, 24],
            "atr_mult": [2.0, 2.5, 3.0],
            "vol_zscore_window": [24, 48, 96],
            "vol_z_threshold": [1.5, 2.0, 2.5],
            "hold_bars": [4, 8, 12],
            "cooldown_bars": [0, 1],
            "use_trend_filter": [False, True],
            "trend_sma_period": [200],
        }
