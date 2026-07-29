import pandas as pd
import numpy as np

from backtester.strategy import Strategy
from backtester.pairs import (
    compute_spread,
    zscore,
    rolling_hedge_ratio,
    engle_granger_coint,
)


class PairsTradingStrategy(Strategy):
    """Cointegration-based pairs trading strategy.

    Trades the z-score of the log-price spread between two cointegrated assets.
    - Long spread (long A / short B) when z-score < -entry_z
    - Short spread (short A / long B) when z-score > +entry_z
    - Exit (go flat) when |z-score| < exit_z

    Uses rolling hedge ratio and periodic cointegration retest.

    This strategy is special: it requires TWO DataFrames (asset A and asset B).
    The generate_signals() method takes a DataFrame with columns:
    close_a, close_b (pre-merged by the runner script).
    """

    def __init__(
        self,
        hedge_window: int = 60,
        zscore_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        coint_retest_bars: int = 0,
        use_rolling_hedge: bool = True,
    ):
        self.hedge_window = hedge_window
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.coint_retest_bars = coint_retest_bars
        self.use_rolling_hedge = use_rolling_hedge

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate spread trading signals.

        Expects df to have 'close_a' and 'close_b' columns
        (merged by the runner script).
        """
        price_a = df["close_a"]
        price_b = df["close_b"]

        # Compute hedge ratio
        if self.use_rolling_hedge:
            log_a = np.log(price_a)
            log_b = np.log(price_b)
            hr = rolling_hedge_ratio(log_a, log_b, self.hedge_window)
        else:
            log_a = np.log(price_a)
            log_b = np.log(price_b)
            static_beta = float(np.polyfit(log_b.dropna().values, log_a.dropna().values, 1)[0])
            hr = static_beta

        # Compute spread and z-score
        spread = compute_spread(price_a, price_b, hr, use_log=True)
        z = zscore(spread, self.zscore_window)

        # Generate signals based on z-score thresholds
        signals = pd.Series(0, index=df.index, dtype="int8")

        position = 0
        z_vals = z.values
        for i in range(len(z_vals)):
            zv = z_vals[i]
            if np.isnan(zv):
                signals.iloc[i] = 0
                continue

            if position == 0:
                # Entry
                if zv < -self.entry_z:
                    position = 1  # Long spread (long A, short B)
                elif zv > self.entry_z:
                    position = -1  # Short spread (short A, long B)
            elif position == 1:
                # Exit long spread
                if zv > -self.exit_z:
                    position = 0
                elif zv > self.stop_z:
                    position = 0  # Stop loss
            elif position == -1:
                # Exit short spread
                if zv < self.exit_z:
                    position = 0
                elif zv < -self.stop_z:
                    position = 0  # Stop loss

            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "hedge_window": [30, 60, 120],
            "zscore_window": [30, 60, 120],
            "entry_z": [1.5, 2.0, 2.5],
            "exit_z": [0.0, 0.5, 1.0],
            "stop_z": [3.0, 4.0, 5.0],
            "use_rolling_hedge": [True],
        }
