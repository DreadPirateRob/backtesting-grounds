import numpy as np
import pandas as pd

from backtester.strategy import Strategy, ema, funding_rate_zscore, oi_momentum


class FundingOiStrategy(Strategy):
    """Funding Rate Mean-Reversion with OI Confirmation.

    - When funding z-score is extremely positive (crowded longs), go short
    - When funding z-score is extremely negative (crowded shorts), go long
    - OI momentum confirms: only take signal if OI momentum supports the direction

    Requires 'funding_rate' column in the DataFrame (added by merge_funding_oi).
    Optionally uses 'open_interest' column if use_oi_confirm=True.
    """

    def __init__(
        self,
        funding_zscore_period: int = 30,
        funding_entry_z: float = 1.5,
        funding_exit_z: float = 0.5,
        use_oi_confirm: bool = True,
        oi_fast: int = 10,
        oi_slow: int = 30,
    ):
        self.funding_zscore_period = funding_zscore_period
        self.funding_entry_z = funding_entry_z
        self.funding_exit_z = funding_exit_z
        self.use_oi_confirm = use_oi_confirm
        self.oi_fast = oi_fast
        self.oi_slow = oi_slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype="int8")

        if "funding_rate" not in df.columns:
            return signals

        funding = df["funding_rate"].fillna(method="ffill").fillna(0)
        zscore = funding_rate_zscore(funding, self.funding_zscore_period)

        # OI confirmation
        oi_confirm_long = pd.Series(True, index=df.index)
        oi_confirm_short = pd.Series(True, index=df.index)

        if self.use_oi_confirm and "open_interest" in df.columns:
            oi = df["open_interest"].fillna(method="ffill")
            oi_mom = oi_momentum(oi, self.oi_fast, self.oi_slow)
            # For long: OI momentum should be negative or neutral (shorts unwinding)
            oi_confirm_long = oi_mom <= 0
            # For short: OI momentum should be positive or neutral (longs building)
            oi_confirm_short = oi_mom >= 0

        position = 0
        for i in range(len(df)):
            z = zscore.iloc[i]
            if np.isnan(z):
                signals.iloc[i] = position
                continue

            if position == 0:
                # Entry
                if z < -self.funding_entry_z and oi_confirm_long.iloc[i]:
                    position = 1
                elif z > self.funding_entry_z and oi_confirm_short.iloc[i]:
                    position = -1
            elif position == 1:
                # Exit long when z-score reverts above -exit_z
                if z > -self.funding_exit_z:
                    position = 0
                # Flip to short if z-score goes extreme positive
                if z > self.funding_entry_z and oi_confirm_short.iloc[i]:
                    position = -1
            elif position == -1:
                # Exit short when z-score reverts below +exit_z
                if z < self.funding_exit_z:
                    position = 0
                # Flip to long if z-score goes extreme negative
                if z < -self.funding_entry_z and oi_confirm_long.iloc[i]:
                    position = 1

            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "funding_zscore_period": [20, 30, 50],
            "funding_entry_z": [1.0, 1.5, 2.0, 2.5],
            "funding_exit_z": [0.3, 0.5, 0.8],
            "use_oi_confirm": [True, False],
            "oi_fast": [5, 10],
            "oi_slow": [20, 30],
        }
