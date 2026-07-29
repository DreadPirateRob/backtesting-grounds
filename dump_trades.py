#!/usr/bin/env python3
"""Dump trade lists for all validated strategies × 3 assets."""

import pandas as pd
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig, run_backtest

from strategies.donchian_vol_strategy import DonchianVolStrategy
from strategies.multi_bar_selloff_strategy import MultiBarSelloffStrategy
from strategies.extreme_range_strategy import ExtremeRangeStrategy
from strategies.vol_spike_strategy import VolSpikeStrategy

DATA_FILES = {
    "BTC": "data/binance_BTCUSDT_1m_klines.csv",
    "ETH": "data/binance_ETHUSDT_1m_klines.csv",
    "SOL": "data/binance_SOLUSDT_1m_klines.csv",
}

CONFIG = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)

STRATEGIES = {
    "Donchian Vol": (DonchianVolStrategy, dict(
        entry_period=240, vol_mult=2.0, hold_bars=24, cooldown_bars=12,
    )),
    "Multi-bar Selloff": (MultiBarSelloffStrategy, dict(
        lookback=24, drop_pct=-8.0, hold_bars=16, cooldown_bars=3,
    )),
    "Extreme Range": (ExtremeRangeStrategy, dict(
        lookback=72, hold_bars=8, cooldown_bars=2,
    )),
    "Vol Spike": (VolSpikeStrategy, dict(
        vol_zscore_window=96, vol_zscore_threshold=2.5,
        delta_threshold_pct=10.0, hold_bars=8, buy_only=True,
    )),
}


def main():
    # Load data
    asset_dfs = {}
    for asset, path in DATA_FILES.items():
        asset_dfs[asset] = load_candles(path, resample="1h",
                                         extra_columns=["taker_buy_base_volume"])

    for strat_name, (cls, params) in STRATEGIES.items():
        print(f"\n{'='*120}")
        print(f"  {strat_name}  |  Params: {params}")
        print(f"{'='*120}")

        for asset, df in asset_dfs.items():
            strat = cls(**params)
            signals = strat.generate_signals(df)
            result = run_backtest(df, signals, CONFIG)
            trades = result["trades"]

            print(f"\n  {asset} — {len(trades)} trades")
            if not trades:
                print("  (no trades)")
                continue

            # Print table header
            print(f"  {'#':>4}  {'Entry Time':<20} {'Exit Time':<20} {'Dir':<6} "
                  f"{'Entry Price':>12} {'Exit Price':>12} {'PnL%':>8} {'Bars':>5}")
            print(f"  {'-'*4}  {'-'*20} {'-'*20} {'-'*6} "
                  f"{'-'*12} {'-'*12} {'-'*8} {'-'*5}")

            total_pnl = 0
            wins = 0
            for i, t in enumerate(trades, 1):
                entry_t = t["entry_time"][:19]
                exit_t = t["exit_time"][:19]
                print(f"  {i:>4}  {entry_t:<20} {exit_t:<20} {t['direction']:<6} "
                      f"{t['entry_price']:>12.2f} {t['exit_price']:>12.2f} "
                      f"{t['pnl_pct']:>+8.2f} {t['bars_held']:>5}")
                total_pnl += t["pnl_pct"]
                if t["pnl_pct"] > 0:
                    wins += 1

            wr = wins / len(trades) * 100 if trades else 0
            avg_pnl = total_pnl / len(trades) if trades else 0
            print(f"\n  Summary: {len(trades)} trades, WR {wr:.0f}%, "
                  f"Avg PnL {avg_pnl:+.2f}%, Total PnL {total_pnl:+.2f}%")


if __name__ == "__main__":
    main()
