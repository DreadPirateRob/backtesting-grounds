import numpy as np
import pandas as pd

from backtester.strategy import Strategy, rsi, ema, sma, atr, bollinger_bands


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample to higher timeframe. Right-closed/labeled prevents look-ahead."""
    return (
        df.resample(timeframe, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )


class HtfTrendLtfMrStrategy(Strategy):
    """HTF Trend + LTF Mean Reversion strategy.

    Uses higher-timeframe (1h/4h) trend signals to establish directional bias,
    then enters at lower-timeframe (15min/30min) mean-reversion levels ONLY
    in the direction of the HTF trend.

    When HTF says "long": buy LTF RSI oversold + BB lower band touches
    When HTF says "short": sell LTF RSI overbought + BB upper band touches
    All counter-trend signals are filtered out.

    Based on QuantPedia D1H1 research: daily trend filter on hourly signals
    improved Sharpe from 0.33 to 0.80 (+142%). Combined with trailing stop
    exit, Sharpe reached 1.07.

    HTF trend methods:
    - "atr_breakout": Use ATR Breakout logic (our proven 1h strategy)
    - "ema": Price above/below EMA
    - "sma_slope": SMA direction
    """

    def __init__(
        self,
        # LTF (base timeframe) parameters
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        bb_period: int = 20,
        bb_std: float = 2.0,
        # HTF parameters
        htf_timeframe: str = "1h",
        htf_method: str = "atr_breakout",
        htf_sma_period: int = 10,
        htf_atr_period: int = 14,
        htf_atr_mult: float = 3.0,
        htf_ema_period: int = 21,
        # Exit
        use_trailing_stop: bool = False,
        trailing_atr_mult: float = 2.0,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.htf_timeframe = htf_timeframe
        self.htf_method = htf_method
        self.htf_sma_period = htf_sma_period
        self.htf_atr_period = htf_atr_period
        self.htf_atr_mult = htf_atr_mult
        self.htf_ema_period = htf_ema_period
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_mult = trailing_atr_mult

    def _compute_htf_trend(self, df_htf: pd.DataFrame) -> pd.Series:
        """Compute trend direction on the higher timeframe."""
        if self.htf_method == "atr_breakout":
            mid = sma(df_htf["close"], self.htf_sma_period)
            atr_val = atr(df_htf, self.htf_atr_period)
            upper = mid + self.htf_atr_mult * atr_val
            lower = mid - self.htf_atr_mult * atr_val
            trend = pd.Series(0, index=df_htf.index, dtype="int8")
            trend[df_htf["close"] > upper] = 1
            trend[df_htf["close"] < lower] = -1
            trend = trend.replace(0, pd.NA).ffill().fillna(0).astype("int8")
            return trend

        if self.htf_method == "ema":
            ema_val = ema(df_htf["close"], self.htf_ema_period)
            trend = pd.Series(0, index=df_htf.index, dtype="int8")
            trend[df_htf["close"] > ema_val] = 1
            trend[df_htf["close"] < ema_val] = -1
            return trend

        if self.htf_method == "sma_slope":
            sma_val = sma(df_htf["close"], self.htf_ema_period)
            slope = sma_val.diff()
            trend = pd.Series(0, index=df_htf.index, dtype="int8")
            trend[slope > 0] = 1
            trend[slope < 0] = -1
            return trend

        raise ValueError(f"Unknown htf_method: {self.htf_method}")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        # --- LTF base signals: RSI + Bollinger confluence ---
        rsi_values = rsi(close, self.rsi_period)
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, self.bb_period, self.bb_std)

        base_long = (rsi_values < self.rsi_oversold) & (close < bb_lower)
        base_short = (rsi_values > self.rsi_overbought) & (close > bb_upper)

        base_signals = pd.Series(0, index=df.index, dtype="int8")
        base_signals[base_long] = 1
        base_signals[base_short] = -1
        # Forward-fill to hold position until opposite signal
        base_signals = base_signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        # --- HTF trend direction ---
        df_htf = _resample_ohlcv(df, self.htf_timeframe)
        htf_trend_raw = self._compute_htf_trend(df_htf)
        # Map back to LTF with shift(1) to prevent look-ahead
        htf_trend = (
            htf_trend_raw
            .reindex(df.index, method="ffill")
            .shift(1)
            .fillna(0)
            .astype("int8")
        )

        # --- Apply hard filter: only allow signals aligned with HTF trend ---
        filtered = base_signals.copy()
        filtered[(base_signals == 1) & (htf_trend != 1)] = 0
        filtered[(base_signals == -1) & (htf_trend != -1)] = 0

        # --- Optional: trailing stop exit on LTF ---
        if self.use_trailing_stop:
            atr_val = atr(df, self.rsi_period)  # reuse rsi_period for ATR
            trail_dist = self.trailing_atr_mult * atr_val

            position = 0
            trailing_stop = 0.0
            result = np.zeros(len(df), dtype=np.int8)

            for i in range(len(df)):
                sig = filtered.iloc[i]
                c = close.iloc[i]
                h = df["high"].iloc[i]
                l = df["low"].iloc[i]

                if position == 0:
                    if sig != 0:
                        position = int(sig)
                        if position == 1:
                            trailing_stop = c - trail_dist.iloc[i]
                        else:
                            trailing_stop = c + trail_dist.iloc[i]
                elif position == 1:
                    # Update trailing stop
                    new_stop = h - trail_dist.iloc[i]
                    if new_stop > trailing_stop:
                        trailing_stop = new_stop
                    # Check stop hit
                    if l < trailing_stop:
                        position = 0
                    # Check for signal flip
                    elif sig == -1:
                        position = -1
                        trailing_stop = c + trail_dist.iloc[i]
                elif position == -1:
                    new_stop = l + trail_dist.iloc[i]
                    if new_stop < trailing_stop:
                        trailing_stop = new_stop
                    if h > trailing_stop:
                        position = 0
                    elif sig == 1:
                        position = 1
                        trailing_stop = c - trail_dist.iloc[i]

                result[i] = position

            return pd.Series(result, index=df.index, dtype="int8")

        return filtered

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [14, 21],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "htf_timeframe": ["1h", "4h"],
            "htf_method": ["atr_breakout", "ema"],
            "htf_ema_period": [21, 50],
        }
