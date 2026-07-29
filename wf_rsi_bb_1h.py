#!/usr/bin/env python3
"""Walk-forward validation: RSI+Bollinger (tight grid) at 1h on BTC, ETH, SOL."""

import sys
import time

from backtester.walk_forward import run_walk_forward, WalkForwardConfig, print_wf_report
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy


class RsiBollingerTight(RsiBollingerStrategy):
    """Tight param grid (16 combos) to reduce overfitting in walk-forward."""

    def param_grid(self):
        return {
            "rsi_period": [14, 21],
            "overbought": [70, 75],
            "oversold": [25, 30],
            "bb_period": [20],
            "bb_std": [2.0, 2.5],
        }


ASSETS = [
    ("BTC", "data/binance_BTCUSDT_1m_klines.csv"),
    ("ETH", "data/binance_ETHUSDT_1m_klines.csv"),
    ("SOL", "data/binance_SOLUSDT_1m_klines.csv"),
]

wf_config = WalkForwardConfig(
    is_months=18,
    oos_months=3,
    mode="rolling",
    min_trades_oos=5,
)

bt_config = BacktestConfig(
    fee_rate=0.0005,      # 5 bps taker fee per side
    slippage_pct=0.0005,  # 5 bps slippage per side  => 10 bps total round-trip
)

if __name__ == '__main__':
    grid = RsiBollingerTight().param_grid()

    print(f"Grid: {grid}", file=sys.stderr)
    print(f"Total combinations: {len(grid['rsi_period']) * len(grid['overbought']) * len(grid['oversold']) * len(grid['bb_period']) * len(grid['bb_std'])}", file=sys.stderr)
    print(f"WF Config: IS={wf_config.is_months}mo, OOS={wf_config.oos_months}mo, mode={wf_config.mode}", file=sys.stderr)
    print(f"BT Config: fee={bt_config.fee_rate}, slippage={bt_config.slippage_pct}", file=sys.stderr)
    print("", file=sys.stderr)

    results = {}

    for asset_name, path in ASSETS:
        print(f"\n{'#' * 120}", file=sys.stderr)
        print(f"# {asset_name} — Loading 1h candles from {path}", file=sys.stderr)
        print(f"{'#' * 120}", file=sys.stderr)

        t0 = time.time()
        df = load_candles(path, resample="1h")
        print(f"  Loaded {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}", file=sys.stderr)

        result = run_walk_forward(
            strategy_class=RsiBollingerTight,
            df=df,
            grid=grid,
            config=bt_config,
            wf_config=wf_config,
            rank_by="sharpe_ratio",
        )

        elapsed = time.time() - t0
        print(f"\n  {asset_name} walk-forward completed in {elapsed:.1f}s", file=sys.stderr)

        print(f"\n{'#' * 120}", file=sys.stderr)
        print(f"# {asset_name} WALK-FORWARD REPORT", file=sys.stderr)
        print(f"{'#' * 120}", file=sys.stderr)
        print_wf_report(result)

        results[asset_name] = result

    # ── Cross-asset summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 120}", file=sys.stderr)
    print("CROSS-ASSET SUMMARY — RSI+Bollinger Tight Grid @ 1h", file=sys.stderr)
    print(f"{'=' * 120}", file=sys.stderr)
    print(f"{'Asset':<6}  {'Segments':>8}  {'OOS Sharpe':>10}  {'WFER':>6}  {'WFER Slope':>11}  {'OOS Ret%':>9}  {'OOS MaxDD%':>10}", file=sys.stderr)
    print(f"{'-'*6}  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*11}  {'-'*9}  {'-'*10}", file=sys.stderr)

    for asset_name, res in results.items():
        eq = res.combined_oos_equity
        total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        cummax = eq.cummax()
        max_dd = float(((eq - cummax) / cummax).min()) * 100
        print(
            f"{asset_name:<6}  {len(res.segments):>8}  {res.overall_oos_sharpe:>10.4f}  "
            f"{res.overall_wfer:>6.4f}  {res.wfer_slope:>+11.6f}  "
            f"{total_ret:>9.2f}  {max_dd:>10.2f}",
            file=sys.stderr,
        )

    print(f"{'=' * 120}\n", file=sys.stderr)
    print("Done.", file=sys.stderr)
