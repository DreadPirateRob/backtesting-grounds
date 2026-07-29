import numpy as np
import pandas as pd

from backtester.strategy import Strategy


class VolSpikeStrategy(Strategy):
    """Volume spike momentum strategy.

    Based on order flow research: buy-side volume spikes at 1h resolution
    show +15bp average next-bar return, well above 10bps fee threshold.
    Buy spikes strongly predict continuation; sell spikes do NOT (asymmetric).

    Uses taker_buy_base_volume when available; falls back to bar direction
    (close > open = buy volume) as a crude approximation.

    Logic:
    1. Compute volume z-score over rolling window.
    2. Compute directional delta (buy vs sell volume proportion).
    3. Buy spike: z-score > threshold AND delta_pct > delta_threshold → long.
    4. Sell spike: z-score > threshold AND delta_pct < -delta_threshold → short
       (only if buy_only=False).
    5. Hold for hold_bars bars then go flat.
    6. After going flat, skip at least 1 bar before re-entering.
    """

    def __init__(
        self,
        vol_zscore_window: int = 96,
        vol_zscore_threshold: float = 2.5,
        delta_threshold_pct: float = 10.0,
        hold_bars: int = 8,
        buy_only: bool = True,
    ):
        self.vol_zscore_window = vol_zscore_window
        self.vol_zscore_threshold = vol_zscore_threshold
        self.delta_threshold_pct = delta_threshold_pct
        self.hold_bars = hold_bars
        self.buy_only = buy_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        volume = df["volume"]
        close = df["close"]
        open_ = df["open"]

        # --- Volume z-score ---
        vol_mean = volume.rolling(self.vol_zscore_window).mean()
        vol_std = volume.rolling(self.vol_zscore_window).std()
        vol_zscore = (volume - vol_mean) / vol_std.replace(0, np.nan)

        # --- Directional delta ---
        if "taker_buy_base_volume" in df.columns:
            buy_vol = df["taker_buy_base_volume"]
            # delta_pct: (2 * buy_vol / total_vol - 1) * 100
            # Ranges from -100 (all sell) to +100 (all buy)
            delta_pct = (2 * buy_vol / volume.replace(0, np.nan) - 1) * 100
        else:
            # Crude approximation: bar direction
            delta_pct = pd.Series(
                np.sign(close.values - open_.values) * 100,
                index=df.index,
                dtype="float64",
            )

        # --- Spike detection (vectorized) ---
        vol_spike = vol_zscore > self.vol_zscore_threshold
        buy_spike = vol_spike & (delta_pct > self.delta_threshold_pct)
        sell_spike = vol_spike & (delta_pct < -self.delta_threshold_pct)

        # --- Stateful hold logic via loop ---
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        buy_spike_vals = buy_spike.values
        sell_spike_vals = sell_spike.values

        hold_remaining = 0  # bars left to hold current position
        cooldown = 0  # bars of cooldown after position exit

        for i in range(n):
            if hold_remaining > 0:
                # Continue holding current position
                signals[i] = signals[i - 1]
                hold_remaining -= 1
                continue

            if cooldown > 0:
                # In cooldown period: stay flat
                signals[i] = 0
                cooldown -= 1
                continue

            # Check for new spike triggers
            if buy_spike_vals[i]:
                signals[i] = 1
                hold_remaining = self.hold_bars - 1  # current bar counts as 1
                continue

            if not self.buy_only and sell_spike_vals[i]:
                signals[i] = -1
                hold_remaining = self.hold_bars - 1
                continue

            # No trigger, check if previous bar was a hold that just ended
            if i > 0 and signals[i - 1] != 0:
                # Position just expired; enter cooldown
                cooldown = 1  # skip at least 1 bar
                signals[i] = 0
            else:
                signals[i] = 0

        return pd.Series(signals, index=df.index, dtype="int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "vol_zscore_window": [48, 96, 192],
            "vol_zscore_threshold": [2.0, 2.5, 3.0],
            "delta_threshold_pct": [5.0, 10.0, 15.0],
            "hold_bars": [4, 8, 16],
            "buy_only": [True, False],
        }
