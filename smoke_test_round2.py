"""Smoke test all 5 Round 2 strategies on BTC at 15min and 30min."""

import sys
import time
sys.path.insert(0, ".")

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics

# Load BTC data with taker_buy_base_volume for VolSpike
print("Loading BTC data...")
t0 = time.time()
df_15 = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample="15min",
                     extra_columns=["taker_buy_base_volume"])
df_30 = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample="30min",
                     extra_columns=["taker_buy_base_volume"])
print(f"  Loaded: 15min={len(df_15)} bars, 30min={len(df_30)} bars ({time.time()-t0:.1f}s)")

configs_10bps = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)  # 10bps total
configs_0bps = BacktestConfig(fee_rate=0.0, slippage_pct=0.0)

# ────────────────────────────────────────────
# 1. TodAlphaStrategy
# ────────────────────────────────────────────
from strategies.tod_alpha_strategy import TodAlphaStrategy

print("\n" + "="*60)
print("1. TOD ALPHA STRATEGY")
print("="*60)

tod_configs = [
    {"entry_hour": 21, "exit_hour": 23, "direction": "long", "day_filter": "all", "vol_filter_period": 0},
    {"entry_hour": 20, "exit_hour": 23, "direction": "long", "day_filter": "all", "vol_filter_period": 0},
    {"entry_hour": 21, "exit_hour": 0, "direction": "long", "day_filter": "all", "vol_filter_period": 0},
    {"entry_hour": 21, "exit_hour": 23, "direction": "long", "day_filter": "thu_sun", "vol_filter_period": 0},
    {"entry_hour": 21, "exit_hour": 23, "direction": "long", "day_filter": "all", "vol_filter_period": 7},
]

for tf_label, df in [("15min", df_15), ("30min", df_30)]:
    print(f"\n--- {tf_label} ---")
    for params in tod_configs:
        strat = TodAlphaStrategy(**params)
        sig = strat.generate_signals(df)
        for fee_label, cfg in [("10bps", configs_10bps), ("0bps", configs_0bps)]:
            res = run_backtest(df, sig, cfg)
            m = compute_metrics(res)
            if fee_label == "10bps" or m["sharpe_ratio"] > 0:
                print(f"  {params} | {fee_label} | Sharpe={m['sharpe_ratio']:.2f} Trades={m['total_trades']} WR={m['win_rate_pct']:.0f}% MaxDD={m['max_drawdown_pct']:.1f}% Ret={m['total_return_pct']:.1f}%")

# ────────────────────────────────────────────
# 2. LargeMoveMrStrategy
# ────────────────────────────────────────────
from strategies.large_move_mr_strategy import LargeMoveMrStrategy

print("\n" + "="*60)
print("2. LARGE MOVE MR STRATEGY")
print("="*60)

lm_configs = [
    {"lookback_bars": 4, "move_threshold_pct": 2.0, "hold_bars": 16, "target_retracement": 0.5},
    {"lookback_bars": 8, "move_threshold_pct": 1.5, "hold_bars": 16, "target_retracement": 0.5},
    {"lookback_bars": 4, "move_threshold_pct": 2.5, "hold_bars": 24, "target_retracement": 0.3},
    {"lookback_bars": 4, "move_threshold_pct": 3.0, "hold_bars": 8, "target_retracement": 0.5},
    {"lookback_bars": 12, "move_threshold_pct": 2.0, "hold_bars": 16, "target_retracement": 0.7},
]

for tf_label, df in [("15min", df_15), ("30min", df_30)]:
    print(f"\n--- {tf_label} ---")
    for params in lm_configs:
        strat = LargeMoveMrStrategy(**params)
        sig = strat.generate_signals(df)
        for fee_label, cfg in [("10bps", configs_10bps), ("0bps", configs_0bps)]:
            res = run_backtest(df, sig, cfg)
            m = compute_metrics(res)
            if fee_label == "10bps" or m["sharpe_ratio"] > 0:
                print(f"  {params} | {fee_label} | Sharpe={m['sharpe_ratio']:.2f} Trades={m['total_trades']} WR={m['win_rate_pct']:.0f}% MaxDD={m['max_drawdown_pct']:.1f}% Ret={m['total_return_pct']:.1f}%")

# ────────────────────────────────────────────
# 3. HtfTrendLtfMrStrategy
# ────────────────────────────────────────────
from strategies.htf_trend_ltf_mr_strategy import HtfTrendLtfMrStrategy

print("\n" + "="*60)
print("3. HTF TREND + LTF MR STRATEGY")
print("="*60)

htf_configs = [
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "1h", "htf_method": "atr_breakout"},
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "4h", "htf_method": "atr_breakout"},
    {"rsi_period": 21, "rsi_oversold": 25, "rsi_overbought": 75, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "1h", "htf_method": "ema", "htf_ema_period": 21},
    {"rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65, "bb_period": 15, "bb_std": 2.5, "htf_timeframe": "1h", "htf_method": "atr_breakout"},
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "1h", "htf_method": "ema", "htf_ema_period": 50},
]

for tf_label, df in [("15min", df_15), ("30min", df_30)]:
    print(f"\n--- {tf_label} ---")
    for params in htf_configs:
        strat = HtfTrendLtfMrStrategy(**params)
        sig = strat.generate_signals(df)
        for fee_label, cfg in [("10bps", configs_10bps), ("0bps", configs_0bps)]:
            res = run_backtest(df, sig, cfg)
            m = compute_metrics(res)
            if fee_label == "10bps" or m["sharpe_ratio"] > 0:
                print(f"  htf={params['htf_timeframe']} method={params['htf_method']} rsi_os={params['rsi_oversold']} rsi_ob={params['rsi_overbought']} bb={params['bb_period']}/{params['bb_std']} | {fee_label} | Sharpe={m['sharpe_ratio']:.2f} Trades={m['total_trades']} WR={m['win_rate_pct']:.0f}% MaxDD={m['max_drawdown_pct']:.1f}% Ret={m['total_return_pct']:.1f}%")

# ────────────────────────────────────────────
# 4. VolCompressionStrategy
# ────────────────────────────────────────────
from strategies.vol_compression_strategy import VolCompressionStrategy

print("\n" + "="*60)
print("4. VOL COMPRESSION BREAKOUT STRATEGY")
print("="*60)

vc_configs = [
    {"rv_window": 96, "rv_lookback": 2880, "compression_pctl": 10.0, "expansion_pctl": 25.0, "breakout_bars": 4, "trend_sma_period": 200},
    {"rv_window": 48, "rv_lookback": 1440, "compression_pctl": 10.0, "expansion_pctl": 25.0, "breakout_bars": 4, "trend_sma_period": 200},
    {"rv_window": 192, "rv_lookback": 5760, "compression_pctl": 15.0, "expansion_pctl": 35.0, "breakout_bars": 6, "trend_sma_period": 100},
    {"rv_window": 96, "rv_lookback": 2880, "compression_pctl": 5.0, "expansion_pctl": 20.0, "breakout_bars": 3, "trend_sma_period": 200},
]

for tf_label, df in [("15min", df_15), ("30min", df_30)]:
    print(f"\n--- {tf_label} ---")
    for params in vc_configs:
        strat = VolCompressionStrategy(**params)
        sig = strat.generate_signals(df)
        for fee_label, cfg in [("10bps", configs_10bps), ("0bps", configs_0bps)]:
            res = run_backtest(df, sig, cfg)
            m = compute_metrics(res)
            if fee_label == "10bps" or m["sharpe_ratio"] > 0:
                print(f"  rv={params['rv_window']} lkbk={params['rv_lookback']} comp={params['compression_pctl']} exp={params['expansion_pctl']} | {fee_label} | Sharpe={m['sharpe_ratio']:.2f} Trades={m['total_trades']} WR={m['win_rate_pct']:.0f}% MaxDD={m['max_drawdown_pct']:.1f}% Ret={m['total_return_pct']:.1f}%")

# ────────────────────────────────────────────
# 5. VolSpikeStrategy
# ────────────────────────────────────────────
from strategies.vol_spike_strategy import VolSpikeStrategy

print("\n" + "="*60)
print("5. VOL SPIKE MOMENTUM STRATEGY")
print("="*60)

vs_configs = [
    {"vol_zscore_window": 96, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 8, "buy_only": True},
    {"vol_zscore_window": 48, "vol_zscore_threshold": 2.0, "delta_threshold_pct": 5.0, "hold_bars": 8, "buy_only": True},
    {"vol_zscore_window": 96, "vol_zscore_threshold": 3.0, "delta_threshold_pct": 15.0, "hold_bars": 16, "buy_only": True},
    {"vol_zscore_window": 192, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 96, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 8, "buy_only": False},
]

for tf_label, df in [("15min", df_15), ("30min", df_30)]:
    print(f"\n--- {tf_label} ---")
    for params in vs_configs:
        strat = VolSpikeStrategy(**params)
        sig = strat.generate_signals(df)
        for fee_label, cfg in [("10bps", configs_10bps), ("0bps", configs_0bps)]:
            res = run_backtest(df, sig, cfg)
            m = compute_metrics(res)
            if fee_label == "10bps" or m["sharpe_ratio"] > 0:
                print(f"  zwin={params['vol_zscore_window']} zthr={params['vol_zscore_threshold']} dthr={params['delta_threshold_pct']} hold={params['hold_bars']} buy_only={params['buy_only']} | {fee_label} | Sharpe={m['sharpe_ratio']:.2f} Trades={m['total_trades']} WR={m['win_rate_pct']:.0f}% MaxDD={m['max_drawdown_pct']:.1f}% Ret={m['total_return_pct']:.1f}%")

print("\n" + "="*60)
print("SMOKE TEST COMPLETE")
print("="*60)
