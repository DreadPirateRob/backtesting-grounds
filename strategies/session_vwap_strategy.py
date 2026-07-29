import numpy as np
import pandas as pd

from backtester.strategy import Strategy, vwap_daily, atr, adx


class SessionVwapStrategy(Strategy):
    """Session-anchored VWAP mean-reversion strategy.

    Uses a proper daily-resetting VWAP (anchored at 00:00 UTC) instead of
    a rolling window. Price tends to revert to the volume-weighted fair value
    within a session. Deviations beyond band_mult standard deviations from
    session VWAP represent statistically stretched conditions.

    v2 improvements:
    - ATR-based stop loss to cut trend breakouts fast
    - ADX trend filter: only enter when ADX < threshold (ranging market)
    - Session-boundary position reset: go flat at 00:00 UTC each day

    Backed by: Zarattini & Aziz (SSRN, 2023) — Sharpe 2.1 on equities.
    Optimal timeframes: 5m-15m (intraday), 1h (swing entries).
    """

    def __init__(
        self,
        band_mult: float = 2.0,
        atr_stop_mult: float = 1.5,
        adx_max: float = 25.0,
        session_flat: bool = True,
    ):
        self.band_mult = band_mult
        self.atr_stop_mult = atr_stop_mult
        self.adx_max = adx_max
        self.session_flat = session_flat

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vwap_df = vwap_daily(df, bands=True)
        vwap_vals = vwap_df["vwap"]
        vwap_upper = vwap_df["vwap_upper"]
        vwap_lower = vwap_df["vwap_lower"]

        # Scale bands by band_mult (default bands are 1 sigma)
        upper = vwap_vals + self.band_mult * (vwap_upper - vwap_vals)
        lower = vwap_vals - self.band_mult * (vwap_vals - vwap_lower)

        # ATR for stop-loss
        atr_vals = atr(df, 14)

        # ADX trend filter
        adx_df = adx(df, 14)
        adx_vals = adx_df["adx"]

        # Detect session boundaries (00:00 UTC)
        is_session_start = np.array(df.index.hour == 0)
        if len(df) > 1:
            bar_minutes = int((df.index[1] - df.index[0]).total_seconds() / 60)
        else:
            bar_minutes = 5
        # For bars >= 4h, session_flat doesn't apply (each bar spans multiple hours)
        apply_session_flat = self.session_flat and bar_minutes < 240

        signals = pd.Series(0, index=df.index, dtype="int8")
        close_arr = df["close"].values
        vwap_arr = vwap_vals.values
        upper_arr = upper.values
        lower_arr = lower.values
        atr_arr = atr_vals.values
        adx_arr = adx_vals.values
        session_arr = is_session_start

        position = 0
        stop_price = 0.0

        for i in range(1, len(df)):
            price = close_arr[i]
            v = vwap_arr[i]
            a = atr_arr[i]

            if np.isnan(v) or np.isnan(upper_arr[i]):
                signals.iloc[i] = position
                continue

            # Session boundary: go flat
            if apply_session_flat and session_arr[i] and position != 0:
                position = 0
                stop_price = 0.0

            # ATR stop-loss check
            if position == 1 and stop_price > 0 and price < stop_price:
                position = 0
                stop_price = 0.0
            elif position == -1 and stop_price > 0 and price > stop_price:
                position = 0
                stop_price = 0.0

            # ADX filter: only enter in ranging markets
            trending = (not np.isnan(adx_arr[i])) and adx_arr[i] > self.adx_max

            if position == 0 and not trending:
                if price < lower_arr[i]:
                    position = 1
                    stop_price = price - self.atr_stop_mult * a if not np.isnan(a) else 0.0
                elif price > upper_arr[i]:
                    position = -1
                    stop_price = price + self.atr_stop_mult * a if not np.isnan(a) else 0.0
            elif position == 1:
                if price >= v:
                    position = 0
                    stop_price = 0.0
            elif position == -1:
                if price <= v:
                    position = 0
                    stop_price = 0.0

            signals.iloc[i] = position

        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "band_mult": [1.0, 1.5, 2.0, 2.5],
            "atr_stop_mult": [1.0, 1.5, 2.0, 3.0],
            "adx_max": [20, 25, 30, 40],
            "session_flat": [True, False],
        }
