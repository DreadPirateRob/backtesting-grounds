#!/usr/bin/env python3
"""5-Strategy Multi-Asset Portfolio Construction.

Combines all 5 validated strategies across BTC/ETH/SOL:
  1. Vol Spike (dip-buy / volume anomaly)
  2. Multi-bar Selloff (dip-buy / price decline)
  3. Extreme Range Bar (dip-buy / volatility expansion)
  4. Donchian Breakout (breakout / new high + volume)
  5. Green Momentum (momentum / bullish continuation)

Tests portfolio methods: equal_weight, risk_parity, max_sharpe.
Per-year breakdown, rolling window stability, correlation analysis.
"""

import sys
import time
import numpy as np
import pandas as pd

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.portfolio import (
    PortfolioConfig, build_portfolio, print_portfolio_report,
)

from strategies.vol_spike_strategy import VolSpikeStrategy
from strategies.multi_bar_selloff_strategy import MultiBarSelloffStrategy
from strategies.extreme_range_strategy import ExtremeRangeStrategy
from strategies.donchian_vol_strategy import DonchianVolStrategy
from strategies.green_momentum_strategy import GreenMomentumStrategy


DATA_FILES = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CONFIG = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)

# Best validated params for each strategy
STRATEGIES = {
    "VolSpike": (VolSpikeStrategy, dict(
        vol_zscore_window=96, vol_zscore_threshold=2.5,
        delta_threshold_pct=10.0, hold_bars=8, buy_only=True,
    )),
    "Selloff": (MultiBarSelloffStrategy, dict(
        lookback=24, drop_pct=-8.0, hold_bars=16, cooldown_bars=3,
    )),
    "ExtrRange": (ExtremeRangeStrategy, dict(
        lookback=72, hold_bars=8, cooldown_bars=2,
    )),
    "Donchian": (DonchianVolStrategy, dict(
        entry_period=480, vol_mult=2.0, hold_bars=24, cooldown_bars=12,
    )),
    "GreenMom": (GreenMomentumStrategy, dict(
        min_green=4, min_gain_pct=4.0, hold_bars=16, cooldown_bars=3,
    )),
}


def rolling_windows(df, window_months=12, step_months=3):
    start = df.index[0]
    end = df.index[-1]
    current = start
    while True:
        window_end = current + pd.DateOffset(months=window_months)
        if window_end > end:
            break
        sl = df.loc[current:window_end]
        if len(sl) > 500:
            yield current, window_end, sl
        current += pd.DateOffset(months=step_months)


def compute_portfolio_in_window(df_window, config):
    """Run all 5 strategies on a window and return portfolio returns."""
    strategy_returns = {}
    for name, (cls, params) in STRATEGIES.items():
        strat = cls(**params)
        signals = strat.generate_signals(df_window)
        result = run_backtest(df_window, signals, config)
        strategy_returns[name] = result["strategy_returns"]
    return strategy_returns


def main():
    t0 = time.time()
    p = lambda *a, **kw: print(*a, **kw)

    p("=" * 100)
    p("  5-STRATEGY MULTI-ASSET PORTFOLIO CONSTRUCTION")
    p("=" * 100)

    # ═══════════════════════════════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════════════════════════════
    asset_dfs = {}
    for asset, path in DATA_FILES.items():
        df = load_candles(path, resample="1h",
                          extra_columns=["taker_buy_base_volume"])
        asset_dfs[asset] = df
        p(f"  {asset}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # ═══════════════════════════════════════════════════════════════
    # PER-ASSET INDIVIDUAL STRATEGY PERFORMANCE
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 100}")
    p(f"  1. INDIVIDUAL STRATEGY PERFORMANCE (10bps)")
    p(f"{'=' * 100}")

    all_strategy_returns = {}  # {asset: {strat_name: returns_series}}

    p(f"\n  {'Strategy':<15} {'Asset':<6} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
      f"{'Trades':>7} {'WR%':>6}")
    p(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*7} {'-'*6}")

    for asset, df in asset_dfs.items():
        all_strategy_returns[asset] = {}
        for strat_name, (cls, params) in STRATEGIES.items():
            strat = cls(**params)
            signals = strat.generate_signals(df)
            result = run_backtest(df, signals, CONFIG)
            metrics = compute_metrics(result)
            all_strategy_returns[asset][strat_name] = result["strategy_returns"]

            p(f"  {strat_name:<15} {asset:<6} {metrics['sharpe_ratio']:>+8.2f} "
              f"{metrics['total_return_pct']:>+9.1f} {metrics['max_drawdown_pct']:>8.1f} "
              f"{metrics['total_trades']:>7} {metrics['win_rate_pct']:>5.0f}%")

    # ═══════════════════════════════════════════════════════════════
    # PER-ASSET PORTFOLIO CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 100}")
    p(f"  2. PER-ASSET PORTFOLIOS (5 strategies combined)")
    p(f"{'=' * 100}")

    methods = ["equal_weight", "risk_parity", "max_sharpe"]

    p(f"\n  {'Method':<15} {'Asset':<6} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
      f"{'Sortino':>8} {'Calmar':>8} {'DivRatio':>9}")
    p(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")

    best_portfolios = {}  # {asset: best PortfolioResult}

    for asset in asset_dfs:
        strat_returns = all_strategy_returns[asset]
        best_sharpe = -999
        best_result = None

        for method in methods:
            pc = PortfolioConfig(method=method, rebalance_freq="1M",
                                 lookback_bars=720)
            portfolio = build_portfolio(strat_returns, config=pc)
            m = portfolio.metrics

            # Diversification ratio
            returns_df = pd.DataFrame(strat_returns).fillna(0)
            ind_vols = returns_df.std().values
            avg_w = 1.0 / len(strat_returns)
            wtd_vol = (avg_w * ind_vols).sum()
            port_vol = portfolio.combined_returns.std()
            div_ratio = wtd_vol / port_vol if port_vol > 0 else 0

            p(f"  {method:<15} {asset:<6} {m['sharpe_ratio']:>+8.2f} "
              f"{m['total_return_pct']:>+9.1f} {m['max_drawdown_pct']:>8.1f} "
              f"{m['sortino_ratio']:>8.2f} {m['calmar_ratio']:>8.2f} "
              f"{div_ratio:>9.2f}")

            if m['sharpe_ratio'] > best_sharpe:
                best_sharpe = m['sharpe_ratio']
                best_result = portfolio

        best_portfolios[asset] = best_result

    # ═══════════════════════════════════════════════════════════════
    # CORRELATION MATRICES
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 100}")
    p(f"  3. CORRELATION MATRICES (strategy returns)")
    p(f"{'=' * 100}")

    for asset in asset_dfs:
        strat_returns = all_strategy_returns[asset]
        returns_df = pd.DataFrame(strat_returns).fillna(0)
        corr = returns_df.corr()

        p(f"\n  {asset}:")
        names = list(corr.columns)
        header = "  " + " " * 12 + "".join(f"{n:>12}" for n in names)
        p(header)
        for row in names:
            vals = "".join(f"{corr.loc[row, c]:>12.3f}" for c in names)
            p(f"  {row:<12}{vals}")

        # Average pairwise correlation (off-diagonal)
        n_strats = len(names)
        off_diag = []
        for i in range(n_strats):
            for j in range(i + 1, n_strats):
                off_diag.append(corr.iloc[i, j])
        p(f"  Avg pairwise corr: {np.mean(off_diag):.3f}")

    # ═══════════════════════════════════════════════════════════════
    # ROLLING WINDOW STABILITY
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 100}")
    p(f"  4. ROLLING WINDOW STABILITY (12mo windows, equal-weight portfolio)")
    p(f"{'=' * 100}")

    for asset, df in asset_dfs.items():
        window_sharpes = []
        for ws, we, sl in rolling_windows(df):
            strat_returns_w = {}
            for strat_name, (cls, params) in STRATEGIES.items():
                strat = cls(**params)
                signals = strat.generate_signals(sl)
                result = run_backtest(sl, signals, CONFIG)
                strat_returns_w[strat_name] = result["strategy_returns"]

            combined = sum(strat_returns_w.values()) / len(strat_returns_w)
            # Compute Sharpe from combined returns
            if len(combined) > 100:
                dur_days = (combined.index[-1] - combined.index[0]).total_seconds() / 86400
                years = dur_days / 365.25 if dur_days > 0 else 1
                bpy = len(combined) / years if years > 0 else 8760
                mu = combined.mean()
                sigma = combined.std()
                sh = mu / sigma * np.sqrt(bpy) if sigma > 0 else 0
            else:
                sh = 0
            window_sharpes.append(sh)

        n = len(window_sharpes)
        frac_pos = sum(1 for s in window_sharpes if s > 0) / n if n else 0
        frac_gt05 = sum(1 for s in window_sharpes if s > 0.5) / n if n else 0

        p(f"\n  {asset}: {n} windows")
        p(f"    Mean Sharpe: {np.mean(window_sharpes):+.2f}, "
          f"Median: {np.median(window_sharpes):+.2f}")
        p(f"    Pos: {frac_pos*100:.0f}%, >0.5: {frac_gt05*100:.0f}%")
        p(f"    Min: {np.min(window_sharpes):+.2f}, Max: {np.max(window_sharpes):+.2f}")

    # ═══════════════════════════════════════════════════════════════
    # PER-YEAR BREAKDOWN
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 100}")
    p(f"  5. PER-YEAR PORTFOLIO SHARPE (equal-weight)")
    p(f"{'=' * 100}")

    for asset, df in asset_dfs.items():
        years = sorted(df.index.year.unique())
        year_sharpes = {}
        for year in years:
            yearly = df[df.index.year == year]
            if len(yearly) < 200:
                continue
            strat_returns_y = {}
            for strat_name, (cls, params) in STRATEGIES.items():
                strat = cls(**params)
                signals = strat.generate_signals(yearly)
                result = run_backtest(yearly, signals, CONFIG)
                strat_returns_y[strat_name] = result["strategy_returns"]

            combined = sum(strat_returns_y.values()) / len(strat_returns_y)
            dur_days = (combined.index[-1] - combined.index[0]).total_seconds() / 86400
            yrs = dur_days / 365.25 if dur_days > 0 else 1
            bpy = len(combined) / yrs if yrs > 0 else 8760
            mu = combined.mean()
            sigma = combined.std()
            sh = mu / sigma * np.sqrt(bpy) if sigma > 0 else 0
            year_sharpes[year] = sh

        year_str = " | ".join(f"{y}: {s:+.2f}" for y, s in year_sharpes.items())
        p(f"  {asset}: {year_str}")

    # ═══════════════════════════════════════════════════════════════
    # CROSS-ASSET COMBINED (15 strategy-asset combinations)
    # ═══════════════════════════════════════════════════════════════
    p(f"\n{'=' * 100}")
    p(f"  6. MEGA PORTFOLIO (5 strats × 3 assets = 15 streams, equal-weight)")
    p(f"{'=' * 100}")

    mega_returns = {}
    for asset, df in asset_dfs.items():
        for strat_name in STRATEGIES:
            key = f"{strat_name}_{asset}"
            mega_returns[key] = all_strategy_returns[asset][strat_name]

    for method in methods:
        pc = PortfolioConfig(method=method, rebalance_freq="1M", lookback_bars=720)
        mega = build_portfolio(mega_returns, config=pc)
        m = mega.metrics

        p(f"\n  {method}:")
        p(f"    Sharpe: {m['sharpe_ratio']:+.2f}, Return: {m['total_return_pct']:+.1f}%, "
          f"MaxDD: {m['max_drawdown_pct']:.1f}%")
        p(f"    Sortino: {m['sortino_ratio']:.2f}, Calmar: {m['calmar_ratio']:.2f}")

    # Full report for equal-weight mega portfolio
    p(f"\n  --- Equal-Weight Mega Portfolio Full Report ---")
    pc_eq = PortfolioConfig(method="equal_weight")
    mega_eq = build_portfolio(mega_returns, config=pc_eq)
    print_portfolio_report(mega_eq)

    # Rolling stability of mega portfolio
    p(f"\n  --- Mega Portfolio Rolling Window Stability ---")
    # Use BTC's timeframe as reference
    ref_df = asset_dfs["BTC"]
    mega_window_sharpes = []
    for ws, we, sl in rolling_windows(ref_df):
        window_returns = {}
        for asset, df in asset_dfs.items():
            asset_window = df.loc[ws:we]
            if len(asset_window) < 200:
                continue
            for strat_name, (cls, params) in STRATEGIES.items():
                strat = cls(**params)
                signals = strat.generate_signals(asset_window)
                result = run_backtest(asset_window, signals, CONFIG)
                key = f"{strat_name}_{asset}"
                window_returns[key] = result["strategy_returns"]

        if len(window_returns) < 5:
            continue

        combined = sum(window_returns.values()) / len(window_returns)
        dur_days = (combined.index[-1] - combined.index[0]).total_seconds() / 86400
        yrs = dur_days / 365.25 if dur_days > 0 else 1
        bpy = len(combined) / yrs if yrs > 0 else 8760
        mu = combined.mean()
        sigma = combined.std()
        sh = mu / sigma * np.sqrt(bpy) if sigma > 0 else 0
        mega_window_sharpes.append(sh)

    if mega_window_sharpes:
        n = len(mega_window_sharpes)
        frac_pos = sum(1 for s in mega_window_sharpes if s > 0) / n
        frac_gt1 = sum(1 for s in mega_window_sharpes if s > 1.0) / n
        p(f"    {n} windows, Mean: {np.mean(mega_window_sharpes):+.2f}, "
          f"Median: {np.median(mega_window_sharpes):+.2f}")
        p(f"    Pos: {frac_pos*100:.0f}%, >1.0: {frac_gt1*100:.0f}%")
        p(f"    Min: {np.min(mega_window_sharpes):+.2f}, "
          f"Max: {np.max(mega_window_sharpes):+.2f}")

    elapsed = time.time() - t0
    p(f"\n  Total time: {elapsed:.0f}s")
    p("=" * 100)


if __name__ == "__main__":
    main()
