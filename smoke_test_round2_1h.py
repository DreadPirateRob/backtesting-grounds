"""Test Round 2 strategies at 1h — conditional strategies may work better here."""

import sys
import time
sys.path.insert(0, ".")

from backtester.data_loader import load_candles
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import compute_metrics

print("Loading BTC 1h data...")
t0 = time.time()
df_1h = load_candles("data/binance_BTCUSDT_1m_klines.csv", resample="1h",
                     extra_columns=["taker_buy_base_volume"])
print(f"  Loaded: 1h={len(df_1h)} bars ({time.time()-t0:.1f}s)")

cfg_10 = BacktestConfig(fee_rate=0.0005, slippage_pct=0.0005)
cfg_0 = BacktestConfig(fee_rate=0.0, slippage_pct=0.0)

def test(name, strat, df, params_label=""):
    sig = strat.generate_signals(df)
    for label, cfg in [("10bps", cfg_10), ("0bps", cfg_0)]:
        res = run_backtest(df, sig, cfg)
        m = compute_metrics(res)
        if label == "10bps" or m["sharpe_ratio"] > 0:
            print(f"  {name} {params_label} | {label} | Sharpe={m['sharpe_ratio']:.2f} Trades={m['total_trades']} WR={m['win_rate_pct']:.0f}% MaxDD={m['max_drawdown_pct']:.1f}% Ret={m['total_return_pct']:.1f}%")

# ── 1. TOD Alpha at 1h ──
from strategies.tod_alpha_strategy import TodAlphaStrategy
print("\n" + "="*60)
print("1. TOD ALPHA @ 1h")
print("="*60)
for params in [
    {"entry_hour": 21, "exit_hour": 23, "direction": "long"},
    {"entry_hour": 20, "exit_hour": 23, "direction": "long"},
    {"entry_hour": 21, "exit_hour": 0, "direction": "long"},
    {"entry_hour": 21, "exit_hour": 23, "direction": "long", "vol_filter_period": 7},
    {"entry_hour": 20, "exit_hour": 0, "direction": "long"},
    {"entry_hour": 19, "exit_hour": 23, "direction": "long"},
    {"entry_hour": 21, "exit_hour": 1, "direction": "long"},
]:
    test("TOD", TodAlphaStrategy(**params), df_1h, str(params))

# ── 2. Large Move MR at 1h ──
from strategies.large_move_mr_strategy import LargeMoveMrStrategy
print("\n" + "="*60)
print("2. LARGE MOVE MR @ 1h")
print("="*60)
for params in [
    {"lookback_bars": 1, "move_threshold_pct": 2.0, "hold_bars": 4, "target_retracement": 0.5},
    {"lookback_bars": 1, "move_threshold_pct": 2.5, "hold_bars": 4, "target_retracement": 0.5},
    {"lookback_bars": 1, "move_threshold_pct": 3.0, "hold_bars": 4, "target_retracement": 0.5},
    {"lookback_bars": 2, "move_threshold_pct": 2.0, "hold_bars": 4, "target_retracement": 0.5},
    {"lookback_bars": 2, "move_threshold_pct": 2.0, "hold_bars": 8, "target_retracement": 0.3},
    {"lookback_bars": 1, "move_threshold_pct": 2.0, "hold_bars": 8, "target_retracement": 0.3},
    {"lookback_bars": 1, "move_threshold_pct": 1.5, "hold_bars": 4, "target_retracement": 0.5},
    {"lookback_bars": 4, "move_threshold_pct": 3.0, "hold_bars": 8, "target_retracement": 0.5},
    {"lookback_bars": 1, "move_threshold_pct": 2.0, "hold_bars": 12, "target_retracement": 0.7},
    {"lookback_bars": 2, "move_threshold_pct": 3.0, "hold_bars": 6, "target_retracement": 0.5},
]:
    test("LargeMR", LargeMoveMrStrategy(**params), df_1h, str(params))

# ── 3. HTF Trend + LTF MR at 1h (HTF=4h) ──
from strategies.htf_trend_ltf_mr_strategy import HtfTrendLtfMrStrategy
print("\n" + "="*60)
print("3. HTF TREND + LTF MR @ 1h")
print("="*60)
for params in [
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "4h", "htf_method": "atr_breakout"},
    {"rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "4h", "htf_method": "atr_breakout"},
    {"rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "4h", "htf_method": "atr_breakout"},
    {"rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65, "bb_period": 15, "bb_std": 2.5, "htf_timeframe": "4h", "htf_method": "atr_breakout"},
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "4h", "htf_method": "ema", "htf_ema_period": 21},
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "4h", "htf_method": "ema", "htf_ema_period": 50},
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.5, "htf_timeframe": "4h", "htf_method": "atr_breakout"},
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "1D", "htf_method": "atr_breakout"},
    {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20, "bb_std": 2.0, "htf_timeframe": "1D", "htf_method": "ema", "htf_ema_period": 21},
]:
    test("HTF", HtfTrendLtfMrStrategy(**params), df_1h, f"htf={params['htf_timeframe']} m={params['htf_method']}")

# ── 4. Vol Compression at 1h ──
from strategies.vol_compression_strategy import VolCompressionStrategy
print("\n" + "="*60)
print("4. VOL COMPRESSION @ 1h")
print("="*60)
for params in [
    {"rv_window": 24, "rv_lookback": 720, "compression_pctl": 10.0, "expansion_pctl": 25.0, "breakout_bars": 4, "trend_sma_period": 200},
    {"rv_window": 48, "rv_lookback": 1440, "compression_pctl": 10.0, "expansion_pctl": 25.0, "breakout_bars": 4, "trend_sma_period": 200},
    {"rv_window": 96, "rv_lookback": 2880, "compression_pctl": 10.0, "expansion_pctl": 25.0, "breakout_bars": 4, "trend_sma_period": 200},
    {"rv_window": 48, "rv_lookback": 1440, "compression_pctl": 5.0, "expansion_pctl": 20.0, "breakout_bars": 3, "trend_sma_period": 200},
    {"rv_window": 48, "rv_lookback": 720, "compression_pctl": 15.0, "expansion_pctl": 35.0, "breakout_bars": 6, "trend_sma_period": 100},
    {"rv_window": 24, "rv_lookback": 720, "compression_pctl": 5.0, "expansion_pctl": 20.0, "breakout_bars": 3, "trend_sma_period": 200},
    {"rv_window": 24, "rv_lookback": 1440, "compression_pctl": 10.0, "expansion_pctl": 25.0, "breakout_bars": 4, "trend_sma_period": 100},
    {"rv_window": 96, "rv_lookback": 2880, "compression_pctl": 15.0, "expansion_pctl": 35.0, "breakout_bars": 6, "trend_sma_period": 100},
]:
    test("VC", VolCompressionStrategy(**params), df_1h, f"rv={params['rv_window']} lk={params['rv_lookback']}")

# ── 5. Vol Spike at 1h ──
from strategies.vol_spike_strategy import VolSpikeStrategy
print("\n" + "="*60)
print("5. VOL SPIKE @ 1h")
print("="*60)
for params in [
    {"vol_zscore_window": 24, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 48, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 24, "vol_zscore_threshold": 2.0, "delta_threshold_pct": 5.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 24, "vol_zscore_threshold": 3.0, "delta_threshold_pct": 15.0, "hold_bars": 8, "buy_only": True},
    {"vol_zscore_window": 24, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 8, "buy_only": True},
    {"vol_zscore_window": 48, "vol_zscore_threshold": 2.0, "delta_threshold_pct": 10.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 24, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 4, "buy_only": False},
    {"vol_zscore_window": 96, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 10.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 24, "vol_zscore_threshold": 2.5, "delta_threshold_pct": 5.0, "hold_bars": 4, "buy_only": True},
    {"vol_zscore_window": 24, "vol_zscore_threshold": 2.0, "delta_threshold_pct": 10.0, "hold_bars": 8, "buy_only": True},
]:
    test("VS", VolSpikeStrategy(**params), df_1h, f"zw={params['vol_zscore_window']} zt={params['vol_zscore_threshold']} dt={params['delta_threshold_pct']} h={params['hold_bars']} bo={params['buy_only']}")

print("\nDone!")
