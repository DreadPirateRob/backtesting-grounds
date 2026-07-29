import numpy as np
import pandas as pd

from backtester.strategy import Strategy, sma


class VolCompressionStrategy(Strategy):
    """Volatility compression breakout strategy.

    Based on Fidelity Digital Assets research: when realized volatility drops
    below the 10th percentile of its trailing distribution, the subsequent
    vol expansion is directional ~72% of the time (resolving upward).

    Unlike BB Squeeze, this strategy uses raw realized volatility percentile
    rather than indicator band width.

    Logic:
    1. Compute annualized realized vol (rolling std of log returns).
    2. Rank current RV within trailing rv_lookback bars (percentile).
    3. Enter compression state when RV percentile < compression_pctl.
    4. On expansion (RV percentile > expansion_pctl after compression):
       - Breakout direction: close vs highest high / lowest low of compression.
       - Ambiguous case: fall back to trend SMA bias.
    5. Forward-fill signal until opposite signal fires.
    """

    def __init__(
        self,
        rv_window: int = 96,
        rv_lookback: int = 2880,
        compression_pctl: float = 10.0,
        expansion_pctl: float = 25.0,
        breakout_bars: int = 4,
        trend_sma_period: int = 200,
    ):
        self.rv_window = rv_window
        self.rv_lookback = rv_lookback
        self.compression_pctl = compression_pctl
        self.expansion_pctl = expansion_pctl
        self.breakout_bars = breakout_bars
        self.trend_sma_period = trend_sma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # --- Realized volatility ---
        log_returns = np.log(close / close.shift(1))
        # Annualize: assume 365.25 days, infer bars_per_year from data length
        # Use a generic multiplier; sqrt(bars_per_year) cancels in percentile rank
        rv = log_returns.rolling(self.rv_window).std()

        # --- RV percentile within trailing window ---
        rv_pctl = rv.rolling(self.rv_lookback).rank(pct=True) * 100

        # --- Trend filter ---
        trend_sma = sma(close, self.trend_sma_period)

        # --- Stateful compression tracking via loop ---
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        rv_pctl_vals = rv_pctl.values
        close_vals = close.values
        high_vals = high.values
        low_vals = low.values
        trend_sma_vals = trend_sma.values

        in_compression = False
        compression_high = np.nan
        compression_low = np.nan
        current_signal = 0

        for i in range(n):
            pctl_val = rv_pctl_vals[i]

            if np.isnan(pctl_val):
                signals[i] = 0
                continue

            if not in_compression:
                # Check if entering compression
                if pctl_val < self.compression_pctl:
                    in_compression = True
                    # Initialize compression high/low from recent breakout_bars
                    start = max(0, i - self.breakout_bars + 1)
                    compression_high = np.nanmax(high_vals[start : i + 1])
                    compression_low = np.nanmin(low_vals[start : i + 1])
                # Hold current signal while not in compression
                signals[i] = current_signal
            else:
                # In compression: track high/low of compression period
                if not np.isnan(high_vals[i]):
                    compression_high = max(compression_high, high_vals[i])
                if not np.isnan(low_vals[i]):
                    compression_low = min(compression_low, low_vals[i])

                # Check for expansion
                if pctl_val > self.expansion_pctl:
                    # Expansion detected — determine breakout direction
                    c = close_vals[i]
                    if c > compression_high:
                        current_signal = 1
                    elif c < compression_low:
                        current_signal = -1
                    else:
                        # Ambiguous: use trend SMA
                        sma_val = trend_sma_vals[i]
                        if not np.isnan(sma_val):
                            current_signal = 1 if c > sma_val else -1
                        else:
                            current_signal = 0
                    in_compression = False
                    signals[i] = current_signal
                else:
                    # Still compressed: hold previous signal
                    signals[i] = current_signal

        return pd.Series(signals, index=df.index, dtype="int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "rv_window": [48, 96, 192],
            "rv_lookback": [1440, 2880, 5760],
            "compression_pctl": [5.0, 10.0, 15.0],
            "expansion_pctl": [20.0, 25.0, 35.0],
            "breakout_bars": [3, 4, 6],
            "trend_sma_period": [100, 200],
        }
