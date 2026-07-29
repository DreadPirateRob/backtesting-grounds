import numpy as np
import pandas as pd

from backtester.strategy import Strategy, rsi, sma, ema, bollinger_bands, adx


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample base-timeframe OHLCV to a higher timeframe.

    Uses right-closed, right-labeled bars to prevent look-ahead bias.
    """
    return (
        df.resample(timeframe, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )


class RsiBbMtfV2Strategy(Strategy):
    """RSI + Bollinger with configurable multi-timeframe trend filter.

    Improvements over v1:
    - Configurable HTF timeframe and trend detection method (EMA, SMA slope, ADX+DI).
    - Soft/neutral_only filter modes to avoid over-filtering in range-bound regimes.
    - Optional HTF RSI momentum guard against entering at HTF extremes.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: int = 65,
        oversold: int = 30,
        bb_period: int = 20,
        bb_std: float = 2.0,
        htf_timeframe: str = "1D",
        htf_method: str = "ema",
        htf_ema_period: int = 21,
        filter_mode: str = "hard",
        use_htf_momentum: bool = False,
        htf_rsi_period: int = 14,
    ):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.htf_timeframe = htf_timeframe
        self.htf_method = htf_method
        self.htf_ema_period = htf_ema_period
        self.filter_mode = filter_mode
        self.use_htf_momentum = use_htf_momentum
        self.htf_rsi_period = htf_rsi_period

    def _compute_htf_trend(self, df_htf: pd.DataFrame) -> pd.Series:
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

        if self.htf_method == "adx_di":
            adx_df = adx(df_htf, self.htf_ema_period)
            trend = pd.Series(0, index=df_htf.index, dtype="int8")
            directional = adx_df["adx"] >= 20
            trend[directional & (adx_df["di_plus"] > adx_df["di_minus"])] = 1
            trend[directional & (adx_df["di_minus"] > adx_df["di_plus"])] = -1
            return trend

        raise ValueError(f"Unknown htf_method: {self.htf_method}")

    def _compute_htf_adx(self, df_htf: pd.DataFrame) -> pd.Series:
        adx_df = adx(df_htf, self.htf_ema_period)
        return adx_df["adx"]

    def _apply_filter(
        self,
        base_signals: pd.Series,
        htf_trend: pd.Series,
        htf_adx_vals: pd.Series | None,
    ) -> pd.Series:
        filtered = base_signals.copy()

        if self.filter_mode == "hard":
            filtered[(base_signals == 1) & (htf_trend == -1)] = 0
            filtered[(base_signals == -1) & (htf_trend == 1)] = 0

        elif self.filter_mode == "soft":
            filtered[(base_signals == 1) & (htf_trend == -1)] = 0
            filtered[(base_signals == -1) & (htf_trend == 1)] = 0

        elif self.filter_mode == "neutral_only":
            if htf_adx_vals is not None:
                strong_trend = htf_adx_vals > 25
                filtered[(base_signals == 1) & (htf_trend == -1) & strong_trend] = 0
                filtered[(base_signals == -1) & (htf_trend == 1) & strong_trend] = 0
            else:
                filtered[(base_signals == 1) & (htf_trend == -1)] = 0
                filtered[(base_signals == -1) & (htf_trend == 1)] = 0

        return filtered

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        # Base signals (RSI + Bollinger confluence)
        rsi_values = rsi(close, self.rsi_period)
        upper, _middle, lower = bollinger_bands(close, self.bb_period, self.bb_std)

        base_signals = pd.Series(0, index=df.index, dtype="int8")
        base_signals[(rsi_values < self.oversold) & (close < lower)] = 1
        base_signals[(rsi_values > self.overbought) & (close > upper)] = -1
        base_signals = base_signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")

        # Resample to HTF
        df_htf = _resample_ohlcv(df, self.htf_timeframe)

        # HTF trend direction (shift(1) to prevent look-ahead)
        htf_trend_htf = self._compute_htf_trend(df_htf)
        htf_trend = (
            htf_trend_htf
            .reindex(df.index, method="ffill")
            .shift(1)
            .fillna(0)
            .astype("int8")
        )

        # HTF ADX for neutral_only mode
        htf_adx_vals: pd.Series | None = None
        if self.filter_mode == "neutral_only":
            htf_adx_raw = self._compute_htf_adx(df_htf)
            htf_adx_vals = (
                htf_adx_raw
                .reindex(df.index, method="ffill")
                .shift(1)
                .fillna(0)
            )

        # Apply trend filter
        filtered = self._apply_filter(base_signals, htf_trend, htf_adx_vals)

        # Optional HTF momentum guard
        if self.use_htf_momentum:
            htf_rsi_raw = rsi(df_htf["close"], self.htf_rsi_period)
            htf_rsi_vals = (
                htf_rsi_raw
                .reindex(df.index, method="ffill")
                .shift(1)
                .fillna(50)
            )
            filtered[(filtered == 1) & (htf_rsi_vals > 70)] = 0
            filtered[(filtered == -1) & (htf_rsi_vals < 30)] = 0

        return filtered.astype("int8")

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [14, 21, 28],
            "overbought": [62, 65, 70],
            "oversold": [28, 30, 35],
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "htf_timeframe": ["4h", "1D"],
            "htf_method": ["ema", "sma_slope"],
            "filter_mode": ["hard", "soft", "neutral_only"],
            "htf_ema_period": [15, 21, 50],
        }
