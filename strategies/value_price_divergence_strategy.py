import numpy as np
import pandas as pd

from backtester.strategy import Strategy, ema, atr, obv, cvd


class ValuePriceDivergenceStrategy(Strategy):
    """Value-Price Divergence (VPD) strategy.

    Measures divergence between volume flow (OBV or CVD) momentum and price
    momentum using Z-scores. When volume-weighted accumulation diverges
    significantly from price action, the volume side tends to be correct.

    Positive divergence (value_z >> price_z) → bullish accumulation → LONG
    Negative divergence (value_z << price_z) → bearish distribution → SHORT
    Exit when divergence normalizes back toward zero.

    Requires taker_buy_base_volume column for CVD variant.
    """

    def __init__(
        self,
        lookback: int = 20,
        smooth: int = 10,
        z_window: int = 100,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        atr_stop_mult: float = 0.0,
        value_source: str = "obv",
    ):
        self.lookback = lookback
        self.smooth = smooth
        self.z_window = z_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.atr_stop_mult = atr_stop_mult
        self.value_source = value_source

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        # Compute value metric
        if self.value_source == "cvd":
            raw_value = cvd(df)
        else:
            raw_value = obv(df)

        # Smooth and compute rate-of-change
        smoothed_value = ema(raw_value, self.smooth)
        value_roc = smoothed_value.diff(self.lookback)

        # Price rate-of-change (pct)
        price_roc = close.pct_change(self.lookback)

        # Rolling Z-scores
        value_mean = value_roc.rolling(self.z_window).mean()
        value_std = value_roc.rolling(self.z_window).std().replace(0, np.nan)
        value_z = (value_roc - value_mean) / value_std

        price_mean = price_roc.rolling(self.z_window).mean()
        price_std = price_roc.rolling(self.z_window).std().replace(0, np.nan)
        price_z = (price_roc - price_mean) / price_std

        # Divergence signal
        divergence = (value_z - price_z).fillna(0)

        # ATR for stops
        atr_vals = atr(df, 14) if self.atr_stop_mult > 0 else None

        # Warmup
        warmup = self.z_window + self.lookback

        # Convert to numpy for loop performance
        div_arr = divergence.values
        close_arr = close.values
        atr_arr = atr_vals.values if atr_vals is not None else None

        signals = pd.Series(0, index=df.index, dtype="int8")
        position = 0
        entry_price = 0.0
        stop_price = 0.0

        for i in range(warmup, len(df)):
            d = div_arr[i]
            price = close_arr[i]

            # Stop-loss check first
            if self.atr_stop_mult > 0 and atr_arr is not None:
                if position == 1 and stop_price > 0 and price < stop_price:
                    position = 0
                elif position == -1 and stop_price > 0 and price > stop_price:
                    position = 0

            if position == 0:
                if d > self.entry_z:
                    position = 1
                    entry_price = price
                    if atr_arr is not None:
                        a = atr_arr[i]
                        stop_price = price - self.atr_stop_mult * a if not np.isnan(a) else 0.0
                elif d < -self.entry_z:
                    position = -1
                    entry_price = price
                    if atr_arr is not None:
                        a = atr_arr[i]
                        stop_price = price + self.atr_stop_mult * a if not np.isnan(a) else 0.0
            elif position == 1:
                if d < self.exit_z:
                    position = 0
                elif d < -self.entry_z:
                    # Direct flip to short
                    position = -1
                    entry_price = price
                    if atr_arr is not None:
                        a = atr_arr[i]
                        stop_price = price + self.atr_stop_mult * a if not np.isnan(a) else 0.0
            elif position == -1:
                if d > -self.exit_z:
                    position = 0
                elif d > self.entry_z:
                    # Direct flip to long
                    position = 1
                    entry_price = price
                    if atr_arr is not None:
                        a = atr_arr[i]
                        stop_price = price - self.atr_stop_mult * a if not np.isnan(a) else 0.0

            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "lookback": [10, 15, 20, 30, 50],
            "smooth": [5, 10, 15, 20],
            "z_window": [50, 100, 150],
            "entry_z": [1.5, 2.0, 2.5, 3.0],
            "exit_z": [0.0, 0.25, 0.5],
            "atr_stop_mult": [0.0, 2.0, 3.0],
            "value_source": ["obv", "cvd"],
        }
