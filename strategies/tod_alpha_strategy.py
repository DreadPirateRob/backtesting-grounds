import numpy as np
import pandas as pd

from backtester.strategy import Strategy


class TodAlphaStrategy(Strategy):
    """Time-of-Day Alpha strategy.

    Based on academic evidence (Vojtko & Javorska SSRN 2024, QuantPedia)
    that BTC exhibits significant returns during specific UTC hours,
    particularly the 21:00-23:00 window.

    Only trades during a configurable hour window each day.
    Flat outside the window — many flat periods by design.
    """

    def __init__(
        self,
        entry_hour: int = 21,
        exit_hour: int = 23,
        direction: str = "long",
        day_filter: str = "all",
        vol_filter_period: int = 0,
    ):
        self.entry_hour = entry_hour
        self.exit_hour = exit_hour
        self.direction = direction
        self.day_filter = day_filter
        self.vol_filter_period = vol_filter_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype="int8")

        # df.index.hour returns a numpy array, NOT pandas — don't call .values
        hours = df.index.hour
        day_of_week = df.index.dayofweek  # Monday=0, Sunday=6

        # --- Build the active-hour mask ---
        # Handle wrap-around case (e.g., entry_hour=22, exit_hour=1)
        if self.entry_hour < self.exit_hour:
            # Simple case: e.g., 21 to 23
            active = (hours >= self.entry_hour) & (hours < self.exit_hour)
        else:
            # Wrap-around case: e.g., 22 to 1 means 22,23,0
            active = (hours >= self.entry_hour) | (hours < self.exit_hour)

        # --- Day filter ---
        if self.day_filter == "weekday":
            # Monday(0) through Friday(4)
            day_mask = day_of_week <= 4
        elif self.day_filter == "weekend":
            # Saturday(5) and Sunday(6)
            day_mask = day_of_week >= 5
        elif self.day_filter == "thu_sun":
            # Thursday(3), Friday(4), Saturday(5), Sunday(6)
            day_mask = day_of_week >= 3
        else:
            # "all" — no filter
            day_mask = np.ones(len(df), dtype=bool)

        active = active & day_mask

        # --- Volatility filter ---
        if self.vol_filter_period > 0:
            log_ret = np.log(df["close"] / df["close"].shift(1))
            # Rolling realized vol over vol_filter_period days (in bars)
            vol_window = self.vol_filter_period * 24
            median_window = 60 * 24
            realized_vol = log_ret.rolling(vol_window, min_periods=vol_window).std()
            vol_median = realized_vol.rolling(median_window, min_periods=vol_window).median()
            vol_mask = realized_vol > vol_median
            active = active & vol_mask.values

        # --- Assign direction ---
        if self.direction == "long":
            signals[active] = 1
        elif self.direction == "short":
            signals[active] = -1
        elif self.direction == "both":
            # "both" means long during the window
            signals[active] = 1

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "entry_hour": [20, 21, 22],
            "exit_hour": [23, 0, 1],  # 0 and 1 mean next day
            "direction": ["long"],
            "day_filter": ["all", "thu_sun"],
            "vol_filter_period": [0, 7],
        }
