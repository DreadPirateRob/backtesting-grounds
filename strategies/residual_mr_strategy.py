import pandas as pd
import numpy as np

from backtester.strategy import Strategy
from backtester.pairs import rolling_hedge_ratio, compute_spread, zscore, hurst_exponent


class ResidualMrStrategy(Strategy):
    """Beta-adjusted residual mean reversion strategy.

    Unlike cointegration-based pairs trading, this strategy:
    1. Does NOT require a stable long-run equilibrium (cointegration)
    2. Uses a SHORT rolling window for beta (adapts to structural changes)
    3. Trades SHORT-TERM mean reversion of the OLS residual

    The residual = log(asset_A) - beta * log(asset_B) where beta is
    re-estimated every bar via rolling OLS. The z-score of this residual
    is the trading signal.

    When z < -entry_z: residual is abnormally low → long A / short B
    When z > +entry_z: residual is abnormally high → short A / long B
    Exit when |z| < exit_z (residual has reverted)

    Expects DataFrame with columns: close_a, close_b
    """

    def __init__(
        self,
        beta_window: int = 60,
        zscore_window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        min_hurst_check: bool = False,
        hurst_window: int = 200,
        hurst_threshold: float = 0.5,
    ):
        self.beta_window = beta_window
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.min_hurst_check = min_hurst_check
        self.hurst_window = hurst_window
        self.hurst_threshold = hurst_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        price_a = df["close_a"]
        price_b = df["close_b"]
        log_a = np.log(price_a)
        log_b = np.log(price_b)

        # Rolling beta
        hr = rolling_hedge_ratio(log_a, log_b, self.beta_window)

        # Residual spread and z-score
        spread = compute_spread(price_a, price_b, hr, use_log=True)
        z = zscore(spread, self.zscore_window)

        # Optional: Hurst exponent filter (only trade if residual is mean-reverting)
        if self.min_hurst_check:
            # Compute rolling Hurst on the spread
            spread_vals = spread.dropna()
            if len(spread_vals) > self.hurst_window:
                h = hurst_exponent(spread_vals.iloc[-self.hurst_window:])
                if h >= self.hurst_threshold:
                    # Spread is trending, not mean-reverting — don't trade
                    return pd.Series(0, index=df.index, dtype="int8")

        # Generate signals
        signals = pd.Series(0, index=df.index, dtype="int8")
        z_vals = z.values
        position = 0

        for i in range(len(z_vals)):
            zv = z_vals[i]
            if np.isnan(zv):
                signals.iloc[i] = 0
                continue

            if position == 0:
                if zv < -self.entry_z:
                    position = 1   # Long spread (long A, short B)
                elif zv > self.entry_z:
                    position = -1  # Short spread (short A, long B)
            elif position == 1:
                if zv > -self.exit_z:
                    position = 0   # Reversion complete
                elif zv < -self.stop_z:
                    position = 0   # Stop loss — spread diverging further
            elif position == -1:
                if zv < self.exit_z:
                    position = 0
                elif zv > self.stop_z:
                    position = 0

            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "beta_window": [30, 60, 120],
            "zscore_window": [20, 30, 60],
            "entry_z": [1.5, 2.0, 2.5],
            "exit_z": [0.0, 0.5],
            "stop_z": [3.0, 4.0],
        }
