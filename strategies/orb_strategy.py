import pandas as pd
import numpy as np

from backtester.strategy import Strategy, ema


class OrbStrategy(Strategy):
    """Opening Range Breakout (ORB) strategy — vectorized.

    Computes the opening range (high/low of first N bars after session start)
    and enters on breakout above/below the range. Two modes:
    - hold_until_session_end=True: go flat at session boundary (pure intraday)
    - hold_until_session_end=False: hold until opposite signal (swing ORB)

    Session boundaries (UTC):
    - daily: 00:00-24:00 (full day — breakout from midnight range)
    - asian: 00:00-08:00
    - london: 08:00-13:00
    - us: 13:00-21:00

    At 15min resolution, or_bars=2 means a 30-minute opening range.
    """

    SESSIONS = {
        "daily": (0, 24),
        "asian": (0, 8),
        "london": (8, 13),
        "us": (13, 21),
    }

    def __init__(
        self,
        session: str = "us",
        or_bars: int = 2,
        trend_ema_period: int = 50,
        use_trend_filter: bool = True,
        hold_until_session_end: bool = False,
    ):
        self.session = session
        self.or_bars = or_bars
        self.trend_ema_period = trend_ema_period
        self.use_trend_filter = use_trend_filter
        self.hold_until_session_end = hold_until_session_end

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        session_start, session_end = self.SESSIONS[self.session]
        hour = df.index.hour
        dates = df.index.date

        # Trend filter
        if self.use_trend_filter:
            trend_ema = ema(df["close"], self.trend_ema_period)
            trend_up = (df["close"] > trend_ema).values
            trend_dn = (df["close"] < trend_ema).values
        else:
            trend_up = np.ones(len(df), dtype=bool)
            trend_dn = np.ones(len(df), dtype=bool)

        # Session mask
        if session_end == 24:
            in_session = hour >= session_start
        else:
            in_session = (hour >= session_start) & (hour < session_end)

        # Detect session starts: first bar where in_session is True after being False
        in_session_arr = np.asarray(in_session)
        session_start_mask = np.zeros(len(df), dtype=bool)
        session_start_mask[0] = in_session_arr[0]
        session_start_mask[1:] = in_session_arr[1:] & ~in_session_arr[:-1]

        # Also detect new date transitions within session (for daily session)
        date_arr = np.array(dates)
        date_changed = np.zeros(len(df), dtype=bool)
        date_changed[1:] = date_arr[1:] != date_arr[:-1]
        session_start_mask |= (date_changed & in_session_arr)

        # Assign each bar to a session group
        session_group = np.cumsum(session_start_mask)

        # Compute signals via groupby
        close_arr = df["close"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        signals = np.zeros(len(df), dtype=np.int8)

        unique_groups = np.unique(session_group)
        for grp in unique_groups:
            grp_mask = session_group == grp
            grp_indices = np.where(grp_mask)[0]

            # Only process bars in session
            in_sess_grp = in_session_arr[grp_indices]
            sess_indices = grp_indices[in_sess_grp]

            if len(sess_indices) < self.or_bars + 1:
                continue

            # Opening range
            or_indices = sess_indices[:self.or_bars]
            or_high = high_arr[or_indices].max()
            or_low = low_arr[or_indices].min()

            if or_high == or_low:
                continue

            # Trading bars after opening range
            trade_indices = sess_indices[self.or_bars:]

            # Find first breakout
            for idx in trade_indices:
                c = close_arr[idx]
                if c > or_high and trend_up[idx]:
                    signals[idx] = 1
                    break
                elif c < or_low and trend_dn[idx]:
                    signals[idx] = -1
                    break

        signals_series = pd.Series(signals, index=df.index, dtype="int8")

        # Forward-fill to hold position
        signals_series = signals_series.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        if self.hold_until_session_end:
            # Go flat outside session
            if session_end == 24:
                outside = hour < session_start
            else:
                outside = np.asarray((hour < session_start) | (hour >= session_end))
            signals_series[outside] = 0

        return signals_series

    def param_grid(self) -> dict[str, list]:
        return {
            "session": ["asian", "london", "us"],
            "or_bars": [2, 3, 4],
            "trend_ema_period": [30, 50, 100],
            "use_trend_filter": [True, False],
            "hold_until_session_end": [True, False],
        }
