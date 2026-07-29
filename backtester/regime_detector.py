from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import numpy as np
import pandas as pd

from .strategy import Strategy, sma, ema, atr

try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Threshold-based regime detector
# ---------------------------------------------------------------------------

def detect_regime_threshold(
    df: pd.DataFrame,
    sma_period: int = 200,
    momentum_bars: int = 20,
    momentum_thresh: float = 0.05,
    vol_window: int = 20,
    high_vol_annual: float = 0.80,
    low_vol_annual: float = 0.40,
) -> pd.DataFrame:
    """Classify each bar into a regime using simple threshold rules.

    Look-ahead free: only uses data up to the current bar.

    Returns DataFrame with columns:
        regime: bull / bear / sideways
        volatility: high / mid / low
        regime_combined: e.g. "bull_high_vol"
    """
    close = df["close"]

    # Trend: SMA filter + momentum
    sma_vals = sma(close, sma_period)
    prev_close = close.shift(momentum_bars)
    bull_mask = (close > sma_vals) & (close > prev_close * (1 + momentum_thresh))
    bear_mask = (close < sma_vals) & (close < prev_close * (1 - momentum_thresh))

    regime = pd.Series("sideways", index=df.index)
    regime[bull_mask] = "bull"
    regime[bear_mask] = "bear"

    # Volatility: annualized realized vol of log returns
    log_ret = np.log(close / close.shift(1))
    if len(df) > 1:
        bar_seconds = (df.index[1] - df.index[0]).total_seconds()
    else:
        bar_seconds = 3600.0
    bars_per_year = 365.25 * 24 * 3600 / bar_seconds
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(bars_per_year)

    vol_label = pd.Series("mid", index=df.index)
    vol_label[realized_vol > high_vol_annual] = "high"
    vol_label[realized_vol < low_vol_annual] = "low"

    return pd.DataFrame(
        {
            "regime": regime,
            "volatility": vol_label,
            "regime_combined": regime + "_" + vol_label + "_vol",
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# HMM-based regime detector
# ---------------------------------------------------------------------------

def detect_regime_hmm(
    df: pd.DataFrame,
    n_regimes: int = 3,
    features: list[str] | None = None,
    lookback_bars: int | None = None,
) -> pd.DataFrame:
    """Classify each bar into a regime using a Gaussian HMM.

    Features extracted from price data (all look-ahead free):
        returns: 1-bar log returns
        volatility: 20-bar rolling realized vol
        trend: normalized distance from 50-bar EMA  (close - ema50) / atr14

    If lookback_bars is None, fits on the full dataset (look-ahead bias -- suitable
    for historical analysis only). If provided, uses an expanding window: fits on
    data[0:i] and predicts state at bar i, which is truly look-ahead free but slow.

    Falls back to detect_regime_threshold if hmmlearn is not installed.

    Returns DataFrame with columns:
        regime: bull / bear / sideways
        regime_prob: probability of assigned state
        state_id: raw HMM state number
    """
    if not _HMM_AVAILABLE:
        print(
            "WARNING: hmmlearn not installed. "
            "Falling back to threshold-based regime detector."
        )
        result = detect_regime_threshold(df)
        result["regime_prob"] = np.nan
        result["state_id"] = -1
        return result[["regime", "regime_prob", "state_id"]]

    # Build feature matrix
    feat_df = _build_hmm_features(df, features)
    valid_mask = feat_df.notna().all(axis=1)
    feat_clean = feat_df.loc[valid_mask]

    if len(feat_clean) < 50:
        raise ValueError(
            f"Only {len(feat_clean)} valid bars after feature computation; need >= 50."
        )

    X = feat_clean.values

    if lookback_bars is None:
        # Full-sample fit (has look-ahead bias)
        states, probs = _fit_predict_hmm(X, n_regimes)
    else:
        # Expanding-window fit (look-ahead free)
        min_fit = max(lookback_bars, 100)
        states = np.full(len(X), -1, dtype=int)
        probs = np.full(len(X), np.nan)
        for i in range(min_fit, len(X)):
            window_start = max(0, i - lookback_bars)
            X_train = X[window_start:i]
            try:
                model = GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42,
                )
                model.fit(X_train)
                logprob, state_seq = model.decode(X[i : i + 1])
                posteriors = model.predict_proba(X[i : i + 1])
                states[i] = state_seq[0]
                probs[i] = posteriors[0, state_seq[0]]
            except Exception:
                states[i] = -1
                probs[i] = np.nan

    # Map states to regime labels by mean return
    regime_labels = _map_states_to_labels(X, states, n_regimes, ret_col=0)

    # Build output aligned to original index
    regime_series = pd.Series(np.nan, index=df.index, dtype=object)
    prob_series = pd.Series(np.nan, index=df.index)
    state_series = pd.Series(-1, index=df.index, dtype=int)

    valid_idx = feat_clean.index
    regime_series.loc[valid_idx] = [regime_labels.get(s, "sideways") for s in states]
    prob_series.loc[valid_idx] = probs
    state_series.loc[valid_idx] = states

    # Forward-fill NaN regime labels at the start (warm-up period)
    regime_series = regime_series.ffill().fillna("sideways")

    return pd.DataFrame(
        {
            "regime": regime_series,
            "regime_prob": prob_series,
            "state_id": state_series,
        },
        index=df.index,
    )


def _build_hmm_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    close = df["close"]
    default_features = ["returns", "volatility", "trend"]
    use_features = features if features is not None else default_features

    feat = pd.DataFrame(index=df.index)

    if "returns" in use_features:
        feat["returns"] = np.log(close / close.shift(1))

    if "volatility" in use_features:
        log_ret = np.log(close / close.shift(1))
        feat["volatility"] = log_ret.rolling(20).std()

    if "trend" in use_features:
        ema50 = ema(close, 50)
        atr14 = atr(df, 14)
        feat["trend"] = (close - ema50) / atr14.replace(0, np.nan)

    return feat


def _fit_predict_hmm(
    X: np.ndarray,
    n_regimes: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    model.fit(X)
    states = model.predict(X)
    posteriors = model.predict_proba(X)
    probs = posteriors[np.arange(len(states)), states]
    return states, probs


def _map_states_to_labels(
    X: np.ndarray,
    states: np.ndarray,
    n_regimes: int,
    ret_col: int = 0,
) -> dict[int, str]:
    labels = ["bear", "sideways", "bull"]
    if n_regimes != 3:
        labels = ["bear"] + ["sideways"] * (n_regimes - 2) + ["bull"]

    valid = states >= 0
    mean_ret = {}
    for s in range(n_regimes):
        mask = (states == s) & valid
        if mask.any():
            mean_ret[s] = X[mask, ret_col].mean()
        else:
            mean_ret[s] = 0.0

    sorted_states = sorted(mean_ret, key=lambda s: mean_ret[s])
    return {s: labels[i] for i, s in enumerate(sorted_states)}


# ---------------------------------------------------------------------------
# Regime switching helper
# ---------------------------------------------------------------------------

@dataclass
class RegimeStrategyMap:
    """Maps regime labels to strategy configurations."""

    regime_strategy: dict[str, tuple[Type[Strategy], dict]] = field(default_factory=dict)
    default_strategy: tuple[Type[Strategy], dict] | None = None


def generate_regime_signals(
    df: pd.DataFrame,
    regimes: pd.Series,
    regime_map: RegimeStrategyMap,
) -> pd.Series:
    """Generate signals by switching strategies based on regime.

    For each contiguous block of the same regime, runs the mapped strategy
    and combines the signals into a single Series.
    """
    regimes = regimes.reindex(df.index).ffill().fillna("sideways")
    signals = pd.Series(0, index=df.index, dtype=int)

    # Identify contiguous regime blocks
    regime_shifted = regimes.shift(1, fill_value=regimes.iloc[0])
    block_starts = regimes.index[regimes != regime_shifted].tolist()
    if len(block_starts) == 0 or block_starts[0] != df.index[0]:
        block_starts = [df.index[0]] + block_starts

    for i, start in enumerate(block_starts):
        end = block_starts[i + 1] if i + 1 < len(block_starts) else df.index[-1]
        regime_label = regimes.loc[start]

        if regime_label in regime_map.regime_strategy:
            strategy_cls, params = regime_map.regime_strategy[regime_label]
        elif regime_map.default_strategy is not None:
            strategy_cls, params = regime_map.default_strategy
        else:
            continue  # flat

        # Slice data with enough lookback for indicator warm-up
        lookback_start = max(0, df.index.get_loc(start) - 300)
        df_slice = df.iloc[lookback_start : df.index.get_loc(end) + 1]

        strategy = strategy_cls(**params)
        block_signals = strategy.generate_signals(df_slice)
        block_signals = block_signals.reindex(df.index, fill_value=0)

        # Only use signals within this regime block
        if i + 1 < len(block_starts):
            mask = (df.index >= start) & (df.index < block_starts[i + 1])
        else:
            mask = df.index >= start
        signals.loc[mask] = block_signals.loc[mask]

    return signals


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_regime_summary(df: pd.DataFrame, regimes: pd.DataFrame) -> None:
    total_bars = len(df)

    print(f"\n{'=' * 60}")
    print(f"REGIME SUMMARY  ({total_bars:,} bars)")
    print(f"{'=' * 60}")

    if "regime" in regimes.columns:
        regime_col = regimes["regime"]
        print(f"\n{'Regime':<12} {'Bars':>10} {'Pct':>8}  {'First':>20}  {'Last':>20}")
        print("-" * 74)
        for label in sorted(regime_col.unique()):
            mask = regime_col == label
            count = mask.sum()
            pct = count / total_bars * 100
            first = df.index[mask].min()
            last = df.index[mask].max()
            print(f"{label:<12} {count:>10,} {pct:>7.1f}%  {str(first)[:19]:>20}  {str(last)[:19]:>20}")

    if "regime_combined" in regimes.columns:
        combined = regimes["regime_combined"]
        print(f"\n{'Combined':<24} {'Bars':>10} {'Pct':>8}")
        print("-" * 44)
        for label in sorted(combined.unique()):
            mask = combined == label
            count = mask.sum()
            pct = count / total_bars * 100
            print(f"{label:<24} {count:>10,} {pct:>7.1f}%")

    if "regime" in regimes.columns:
        regime_col = regimes["regime"]
        transitions = (regime_col != regime_col.shift(1)).sum() - 1
        print(f"\nRegime transitions: {transitions:,}")

    print(f"{'=' * 60}\n")
