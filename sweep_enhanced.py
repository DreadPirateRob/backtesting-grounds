#!/usr/bin/env python3
"""Sweep enhanced strategies and compare against baseline RSI+Bollinger."""

import json
import sys
import time
from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.sweep import run_sweep

from strategies.rsi_bollinger_strategy import RsiBollingerStrategy
from strategies.rsi_bb_mtf_strategy import RsiBbMtfStrategy
from strategies.rsi_bb_cvd_strategy import RsiBbCvdStrategy
from strategies.rsi_bb_enhanced_strategy import RsiBbEnhancedStrategy


def to_float(v):
    try:
        return float(v) if v is not None else 0.0
    except:
        return 0.0


def print_results(title, result, top_n=5):
    print(f"\n{title}")
    print(f"{'#':<4} {'Sharpe':>7} {'Sortino':>8} {'Return%':>9} {'MaxDD%':>8} {'WinR%':>7} {'PF':>7} {'Calmar':>7} {'Trd':>5}  Params")
    print("-" * 130)
    for r in result["results"][:top_n]:
        m = r["metrics"]
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"#{r['rank']:<3} {to_float(m.get('sharpe_ratio')):>7.2f} {to_float(m.get('sortino_ratio')):>8.2f} "
              f"{to_float(m.get('total_return_pct')):>8.1f}% {to_float(m.get('max_drawdown_pct')):>7.1f}% "
              f"{to_float(m.get('win_rate_pct')):>6.1f}% {to_float(m.get('profit_factor')):>7.2f} "
              f"{to_float(m.get('calmar_ratio')):>7.2f} {int(to_float(m.get('total_trades'))):>5}  {params_str}")


if __name__ == "__main__":
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    # ============================================================
    # INTRADAY (1h)
    # ============================================================
    print("=" * 100)
    print("INTRADAY (1h) — ENHANCED STRATEGIES vs BASELINE")
    print("=" * 100)

    df_1h = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                         start="2025-01-01", end="2025-12-31", resample="1h")
    df_1h_cvd = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                             start="2025-01-01", end="2025-12-31", resample="1h",
                             extra_columns=["taker_buy_base_volume"])
    print(f"Data: {len(df_1h)} bars (1h, 2025)")

    # --- Baseline ---
    print("\n[1/4] Baseline RSI+Bollinger...")
    t0 = time.time()
    baseline_grid = {
        "rsi_period": [14, 18, 28, 30],
        "overbought": [62, 65, 67],
        "oversold": [28, 30, 32],
        "bb_period": [15, 17, 20, 22],
        "bb_std": [2.0, 2.25, 2.5],
    }
    r_base = run_sweep(RsiBollingerStrategy, df_1h, baseline_grid, config=config,
                       rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_base['total_combos']} combos)")
    print_results("BASELINE RSI+Bollinger (1h):", r_base)

    # --- MTF ---
    print("\n[2/4] MTF (Daily trend filter)...")
    t0 = time.time()
    mtf_grid = {
        "rsi_period": [14, 18, 28, 30],
        "overbought": [62, 65, 67],
        "oversold": [28, 30, 32],
        "bb_period": [15, 17, 20, 22],
        "bb_std": [2.0, 2.25, 2.5],
        "daily_ema": [10, 15, 21, 30],
    }
    r_mtf = run_sweep(RsiBbMtfStrategy, df_1h, mtf_grid, config=config,
                      rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_mtf['total_combos']} combos)")
    print_results("MTF RSI+BB (1h + Daily filter):", r_mtf)

    # --- CVD ---
    print("\n[3/4] CVD (Order flow filter)...")
    t0 = time.time()
    cvd_grid = {
        "rsi_period": [14, 18, 28, 30],
        "overbought": [62, 65, 67],
        "oversold": [28, 30, 32],
        "bb_period": [15, 17, 20, 22],
        "bb_std": [2.0, 2.25, 2.5],
        "cvd_lookback": [10, 20, 30, 50],
    }
    r_cvd = run_sweep(RsiBbCvdStrategy, df_1h_cvd, cvd_grid, config=config,
                      rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_cvd['total_combos']} combos)")
    print_results("CVD RSI+BB (1h + Order flow):", r_cvd)

    # --- Enhanced (all filters) ---
    print("\n[4/4] Enhanced (Volume + Session + Weekend + MTF)...")
    t0 = time.time()
    enhanced_grid = {
        "rsi_period": [14, 18, 28, 30],
        "overbought": [62, 65, 67],
        "oversold": [28, 30, 32],
        "bb_period": [15, 17, 20, 22],
        "bb_std": [2.0, 2.25, 2.5],
        "vol_mult": [1.0, 1.5, 2.0],
        "daily_ema": [15, 21, 30],
    }
    r_enh = run_sweep(RsiBbEnhancedStrategy, df_1h, enhanced_grid, config=config,
                      rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_enh['total_combos']} combos)")
    print_results("ENHANCED RSI+BB (1h + all filters):", r_enh)

    # ============================================================
    # SWING (4h)
    # ============================================================
    print("\n\n" + "=" * 100)
    print("SWING (4h) — ENHANCED STRATEGIES vs BASELINE")
    print("=" * 100)

    df_4h = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                         start="2025-01-01", end="2025-12-31", resample="4h")
    df_4h_cvd = load_candles("data/binance_BTCUSDT_1m_klines.csv",
                             start="2025-01-01", end="2025-12-31", resample="4h",
                             extra_columns=["taker_buy_base_volume"])
    print(f"Data: {len(df_4h)} bars (4h, 2025)")

    # --- Baseline ---
    print("\n[1/4] Baseline RSI+Bollinger...")
    t0 = time.time()
    r_base_4h = run_sweep(RsiBollingerStrategy, df_4h, baseline_grid, config=config,
                          rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_base_4h['total_combos']} combos)")
    print_results("BASELINE RSI+Bollinger (4h):", r_base_4h)

    # --- MTF ---
    print("\n[2/4] MTF (Daily trend filter)...")
    t0 = time.time()
    r_mtf_4h = run_sweep(RsiBbMtfStrategy, df_4h, mtf_grid, config=config,
                         rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_mtf_4h['total_combos']} combos)")
    print_results("MTF RSI+BB (4h + Daily filter):", r_mtf_4h)

    # --- CVD ---
    print("\n[3/4] CVD (Order flow filter)...")
    t0 = time.time()
    r_cvd_4h = run_sweep(RsiBbCvdStrategy, df_4h_cvd, cvd_grid, config=config,
                         rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_cvd_4h['total_combos']} combos)")
    print_results("CVD RSI+BB (4h + Order flow):", r_cvd_4h)

    # --- Enhanced ---
    print("\n[4/4] Enhanced (Volume + Session + Weekend + MTF)...")
    t0 = time.time()
    # For 4h, session filter doesn't apply well (bars span 4 hours), disable it
    enhanced_grid_4h = {
        "rsi_period": [14, 18, 28, 30],
        "overbought": [62, 65, 67],
        "oversold": [28, 30, 32],
        "bb_period": [15, 17, 20, 22],
        "bb_std": [2.0, 2.25, 2.5],
        "vol_mult": [1.0, 1.5, 2.0],
        "daily_ema": [15, 21, 30],
    }
    r_enh_4h = run_sweep(RsiBbEnhancedStrategy, df_4h, enhanced_grid_4h, config=config,
                         rank_by="sharpe_ratio", top=5)
    print(f"  Done in {time.time()-t0:.1f}s ({r_enh_4h['total_combos']} combos)")
    print_results("ENHANCED RSI+BB (4h + all filters):", r_enh_4h)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    def best(r):
        if not r.get("results"):
            return {"sharpe": 0, "ret": 0, "dd": 0, "wr": 0}
        m = r["results"][0]["metrics"]
        return {
            "sharpe": to_float(m.get("sharpe_ratio")),
            "ret": to_float(m.get("total_return_pct")),
            "dd": to_float(m.get("max_drawdown_pct")),
            "wr": to_float(m.get("win_rate_pct")),
            "trades": int(to_float(m.get("total_trades"))),
        }

    print(f"\n{'Strategy':<35} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'WinR%':>7} {'Trades':>7}")
    print("-" * 80)
    for name, r in [
        ("Baseline RSI+BB (1h)", r_base),
        ("+ MTF Daily Filter (1h)", r_mtf),
        ("+ CVD Order Flow (1h)", r_cvd),
        ("+ All Filters (1h)", r_enh),
        ("Baseline RSI+BB (4h)", r_base_4h),
        ("+ MTF Daily Filter (4h)", r_mtf_4h),
        ("+ CVD Order Flow (4h)", r_cvd_4h),
        ("+ All Filters (4h)", r_enh_4h),
    ]:
        b = best(r)
        print(f"{name:<35} {b['sharpe']:>8.2f} {b['ret']:>8.1f}% {b['dd']:>7.1f}% {b['wr']:>6.1f}% {b['trades']:>7}")

    print("\nDONE!")
