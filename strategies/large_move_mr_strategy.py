import numpy as np
import pandas as pd

from backtester.strategy import Strategy


class LargeMoveMrStrategy(Strategy):
    """Post-Large-Move Mean Reversion strategy.

    Based on academic evidence (Cheah, Fry & Masih 2019) showing
    significant negative first-order autocorrelation at 1-4h in BTC,
    especially after large moves. Downside overreactions revert faster.

    Triggers a fade trade when a large move occurs, then holds until
    price retraces a fraction of the move or a max holding period elapses.
    Each trigger is a discrete event — not forward-filled.
    """

    def __init__(
        self,
        lookback_bars: int = 4,
        move_threshold_pct: float = 2.0,
        hold_bars: int = 16,
        target_retracement: float = 0.5,
        use_asymmetric_sizing: bool = False,
    ):
        self.lookback_bars = lookback_bars
        self.move_threshold_pct = move_threshold_pct
        self.hold_bars = hold_bars
        self.target_retracement = target_retracement
        self.use_asymmetric_sizing = use_asymmetric_sizing

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype="int8")
        close = df["close"].values

        # Precompute rolling return over lookback_bars
        threshold = self.move_threshold_pct / 100.0
        n = len(close)

        # Vectorized rolling return (close / close[lookback_bars ago] - 1)
        ret = np.full(n, np.nan)
        if self.lookback_bars < n:
            ret[self.lookback_bars:] = (
                close[self.lookback_bars:] / close[:n - self.lookback_bars] - 1
            )

        # Stateful loop: track position, entry price, triggering move, bars held
        position = 0        # 0=flat, 1=long, -1=short
        entry_price = 0.0
        trigger_move = 0.0  # the return that triggered the trade
        bars_held = 0

        sig_arr = np.zeros(n, dtype=np.int8)

        for i in range(self.lookback_bars, n):
            r = ret[i]
            price = close[i]

            if position == 0:
                # Check for trigger
                if np.isnan(r):
                    continue
                if r > threshold:
                    # Large up move -> fade with short
                    position = -1
                    entry_price = price
                    trigger_move = r
                    bars_held = 0
                elif r < -threshold:
                    # Large down move -> fade with long
                    position = 1
                    entry_price = price
                    trigger_move = r
                    bars_held = 0
            else:
                bars_held += 1

                # Check exit conditions
                exit_trade = False

                # (a) Retracement target hit
                # For a long (fading down move): entry was at bottom, we want
                # price to rise by target_retracement * abs(trigger_move) * entry_price
                move_in_price = abs(trigger_move) * entry_price
                if position == 1:
                    # Long: exit if price rose enough from entry
                    if price >= entry_price + self.target_retracement * move_in_price:
                        exit_trade = True
                elif position == -1:
                    # Short: exit if price fell enough from entry
                    if price <= entry_price - self.target_retracement * move_in_price:
                        exit_trade = True

                # (b) Max hold period elapsed
                if bars_held >= self.hold_bars:
                    exit_trade = True

                if exit_trade:
                    position = 0
                    entry_price = 0.0
                    trigger_move = 0.0
                    bars_held = 0

            sig_arr[i] = position

        signals[:] = sig_arr
        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "lookback_bars": [4, 8, 12],  # 1h, 2h, 3h at 15min
            "move_threshold_pct": [1.5, 2.0, 2.5, 3.0],
            "hold_bars": [8, 16, 24],  # 2h, 4h, 6h at 15min
            "target_retracement": [0.3, 0.5, 0.7],
        }
