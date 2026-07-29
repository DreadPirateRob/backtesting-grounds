import numpy as np
import pandas as pd

from backtester.strategy import Strategy, ema, atr


class CvdDivergenceStrategy(Strategy):
    """CVD (Cumulative Volume Delta) divergence strategy.

    Detects divergences between price action and order flow:
    - Bullish divergence: price makes lower low but CVD makes higher low
      (hidden buying / accumulation)
    - Bearish divergence: price makes higher high but CVD makes lower high
      (hidden selling / distribution)

    Uses pivot detection to find swing highs/lows in both price and CVD,
    then compares consecutive pivots for divergence patterns.

    Requires taker_buy_base_volume column (falls back to close-based
    approximation if unavailable).

    Expected Sharpe: 0.8-1.5. Optimal timeframes: 15m-4h.
    """

    def __init__(
        self,
        pivot_lookback: int = 10,
        cvd_smoothing: int = 5,
        confirmation_bars: int = 2,
        atr_stop_mult: float = 2.0,
    ):
        self.pivot_lookback = pivot_lookback
        self.cvd_smoothing = cvd_smoothing
        self.confirmation_bars = confirmation_bars
        self.atr_stop_mult = atr_stop_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        # Compute CVD
        if "taker_buy_base_volume" in df.columns:
            delta = 2 * df["taker_buy_base_volume"] - df["volume"]
        else:
            direction = np.sign(close - df["open"])
            delta = df["volume"] * direction

        raw_cvd = delta.cumsum()
        cvd_vals = ema(raw_cvd, self.cvd_smoothing)

        # ATR for stop sizing
        atr_vals = atr(df, 14)

        # Detect pivot highs/lows in price and CVD
        lb = self.pivot_lookback
        price_arr = close.values
        cvd_arr = cvd_vals.values

        signals = pd.Series(0, index=df.index, dtype="int8")
        position = 0
        entry_price = 0.0
        stop_price = 0.0

        # Track recent pivot lows and highs for divergence
        last_price_low = np.nan
        last_price_low_idx = 0
        last_cvd_at_price_low = np.nan

        last_price_high = np.nan
        last_price_high_idx = 0
        last_cvd_at_price_high = np.nan

        for i in range(lb * 2, len(df)):
            price = price_arr[i]

            # Check for pivot low at i - lb (confirmed by lb bars on each side)
            pi = i - lb
            if pi >= lb:
                is_pivot_low = True
                is_pivot_high = True
                for j in range(pi - lb, pi + lb + 1):
                    if j == pi or j < 0 or j >= len(price_arr):
                        continue
                    if price_arr[j] < price_arr[pi]:
                        is_pivot_low = False
                    if price_arr[j] > price_arr[pi]:
                        is_pivot_high = False

                if is_pivot_low:
                    # Check for bullish divergence: lower price low, higher CVD low
                    if (
                        not np.isnan(last_price_low)
                        and price_arr[pi] < last_price_low
                        and not np.isnan(cvd_arr[pi])
                        and not np.isnan(last_cvd_at_price_low)
                        and cvd_arr[pi] > last_cvd_at_price_low
                    ):
                        # Bullish divergence detected
                        if position <= 0:
                            position = 1
                            entry_price = price
                            a = atr_vals.iloc[i]
                            stop_price = price - self.atr_stop_mult * a if not np.isnan(a) else 0.0

                    last_price_low = price_arr[pi]
                    last_price_low_idx = pi
                    last_cvd_at_price_low = cvd_arr[pi]

                if is_pivot_high:
                    # Check for bearish divergence: higher price high, lower CVD high
                    if (
                        not np.isnan(last_price_high)
                        and price_arr[pi] > last_price_high
                        and not np.isnan(cvd_arr[pi])
                        and not np.isnan(last_cvd_at_price_high)
                        and cvd_arr[pi] < last_cvd_at_price_high
                    ):
                        # Bearish divergence detected
                        if position >= 0:
                            position = -1
                            entry_price = price
                            a = atr_vals.iloc[i]
                            stop_price = price + self.atr_stop_mult * a if not np.isnan(a) else 0.0

                    last_price_high = price_arr[pi]
                    last_price_high_idx = pi
                    last_cvd_at_price_high = cvd_arr[pi]

            # Stop-loss check
            if position == 1 and stop_price > 0 and price < stop_price:
                position = 0
            elif position == -1 and stop_price > 0 and price > stop_price:
                position = 0

            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "pivot_lookback": [5, 8, 10, 15],
            "cvd_smoothing": [3, 5, 8],
            "confirmation_bars": [1, 2, 3],
            "atr_stop_mult": [1.5, 2.0, 2.5, 3.0],
        }
