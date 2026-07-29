# Backtesting Analysis Protocol — Autonomous Optimization Loop

## When to Use This Protocol

Invoke this protocol when the user asks to **analyze**, **optimize**, **backtest**, **find best params**, or **evaluate** a trading strategy. Follow the 8-phase workflow sequentially. Do not skip the robustness phases.

**Loop mode:** When the user asks to **find the best strategy**, **optimize everything**, **run the loop**, or says **keep going until you find something good** — enter the Optimization Loop (Section 2). The loop wraps the 8-phase protocol, derives new strategies automatically, and converges when criteria are met.

**What it produces:**
- **Single strategy mode:** A ranked recommendation with robustness evidence, comparison to buy-and-hold, and a single copy-paste command to reproduce the result.
- **Loop mode:** A Leaderboard of all tested strategies with a Final Report ranking the best robustness-validated candidates.

---

## Quick Reference — CLI & Python API

### CLI: `run_backtest.py`

```
python3 run_backtest.py \
  --strategy NAME          # Required. Maps to strategies/{name}_strategy.py
  --resample FREQ          # Pandas freq: 5min, 15min, 1h, 4h, 1D
  --sweep                  # Enable parameter sweep mode
  --rank-by METRIC         # Default: sharpe_ratio. Prefix - for ascending
  --top N                  # Number of top results (default: 20)
  --params '{"key": val}'  # JSON string of strategy params
  --start YYYY-MM-DD       # Start date filter (inclusive)
  --end YYYY-MM-DD         # End date filter (inclusive)
  --fee 0.001              # Fee rate per trade (default: 0.1%)
  --slippage 0.0005        # Slippage per trade (default: 0.05%)
  --long-only              # Restrict to long positions only
  --capital 10000          # Initial capital in USD
  --jobs N                 # Parallel workers (default: CPU count)
  --data PATH              # CSV path (default: data/binance_BTCUSDT_1m_klines.csv)
  --list-strategies        # List all discovered strategies and exit
```

**Output:** JSON to stdout, human-readable summary to stderr.

### CLI: `fetch_binance_klines.py`

```
python3 fetch_binance_klines.py \
  --interval INTERVAL      # 1s, 5s, 10s, 30s, 1m, 5m, 15m, 1h, etc.
  --symbol SYMBOL          # Default: BTCUSDT
  --years-back N           # Years of history (default: 5 for bulk)
  --days-back N            # Days of history (overrides --years-back)
  --output-dir PATH        # Output directory (default: data/)
```

Sub-minute intervals (1s, 5s, 10s, 30s) automatically use the REST API.

### Python API — Signatures

```python
from backtester import (
    load_candles, enrich_sessions, BacktestConfig, run_backtest, compute_metrics, run_sweep,
    Strategy, rsi, sma, ema, macd, bollinger_bands, atr,
    vwap_daily, adx, stochastic, stoch_rsi, cvd, obv,
)
from strategies import list_strategies

# Load data
df = load_candles(
    path: str,                    # Path to Binance klines CSV
    start: str | None = None,     # "YYYY-MM-DD" inclusive
    end: str | None = None,       # "YYYY-MM-DD" inclusive
    resample: str | None = None,  # "1h", "4h", "1D", etc.
    extra_columns: list[str] | None = None,  # e.g. ["taker_buy_base_volume"]
) -> pd.DataFrame                 # DatetimeIndex, columns: open high low close volume

# Enrich with session data (optional)
df = enrich_sessions(df)  # Adds: hour, session, is_weekend, is_peak_liquidity

# Configure backtest
config = BacktestConfig(
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.0005,         # 5 bps taker fee per side (was 10 bps)
    slippage_pct: float = 0.0005,     # 5 bps slippage per side
    long_only: bool = False,
    impact_coefficient: float = 0.0,  # Tier 2: 0.20 recommended for BTC
    adv_daily: float = 20e9,          # Tier 2: average daily volume
    funding_rate_8h: float = 0.0,     # Perps: 0.0001 = 1 bp per 8h
)

# Run single backtest
results = run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,                   # 1=long, -1=short, 0=flat
    config: BacktestConfig | None = None,
) -> dict  # keys: equity_curve, positions, strategy_returns, trades, fees_paid

# Compute metrics
metrics = compute_metrics(results: dict) -> dict
# Keys: total_return_pct, annualized_return_pct, sharpe_ratio, sortino_ratio,
#        max_drawdown_pct, max_drawdown_duration_days, calmar_ratio,
#        total_trades, win_rate_pct, profit_factor, avg_win_pct, avg_loss_pct,
#        best_trade_pct, worst_trade_pct, time_in_market_pct, total_fees_paid,
#        long_trades, short_trades, long_win_rate_pct, short_win_rate_pct,
#        long_avg_bars_held, short_avg_bars_held

# Run parameter sweep
sweep = run_sweep(
    strategy_class: Type[Strategy],
    df: pd.DataFrame,
    grid: dict[str, list],
    config: BacktestConfig | None = None,
    rank_by: str = "sharpe_ratio",
    top: int = 20,
    n_jobs: int | None = None,
) -> dict  # keys: total_combos, completed, rank_by, results (list of {rank, params, metrics})

# Discover strategies
strategies = list_strategies()
# Returns: [{"name": "rsi", "class": RsiStrategy, "file": "strategies/rsi_strategy.py"}, ...]
```

### Indicators

| Function | Signature | Returns |
|----------|-----------|---------|
| `sma` | `sma(series, period)` | `pd.Series` |
| `ema` | `ema(series, period)` | `pd.Series` |
| `rsi` | `rsi(series, period=14)` | `pd.Series` (0-100) |
| `macd` | `macd(series, fast=12, slow=26, signal=9)` | `(macd_line, signal_line, histogram)` |
| `bollinger_bands` | `bollinger_bands(series, period=20, std=2.0)` | `(upper, middle, lower)` |
| `atr` | `atr(df, period=14)` | `pd.Series` — note: takes full DataFrame |
| `vwap_daily` | `vwap_daily(df, bands=True)` | `pd.DataFrame` with vwap, vwap_upper, vwap_lower — daily-resetting at 00:00 UTC |
| `adx` | `adx(df, period=14)` | `pd.DataFrame` with adx, di_plus, di_minus |
| `stochastic` | `stochastic(df, k_period=14, d_period=3)` | `(%K, %D)` — takes full DataFrame |
| `stoch_rsi` | `stoch_rsi(series, rsi_period=14, k_period=14, d_period=3)` | `(stoch_rsi_k, stoch_rsi_d)` |
| `cvd` | `cvd(df)` | `pd.Series` — uses taker_buy_base_volume if available, else close-based approx |
| `obv` | `obv(df)` | `pd.Series` — On-Balance Volume |

### Indicator Warm-Up Requirements

Indicators require a warm-up period before producing valid values. The first N bars are NaN or distorted. **Discard signals generated during warm-up** — the backtest period begins after `max(all indicator warm-up bars)`.

| Indicator | Minimum Bars | Recommended Warm-Up | Notes |
|-----------|-------------|---------------------|-------|
| SMA(N) | N | N | Exact after N bars |
| EMA(N) | N+1 | 3*N (good), 5*N (precise) | Asymptotic convergence; weight of oldest bar at 3*N is < 0.25% |
| RSI(14) | 15 | 100+ | Uses Wilder smoothing (EMA internally) |
| MACD(12,26,9) | 35 | 200+ | 26-period EMA + 9-period signal, both need convergence |
| Bollinger(20,2) | 20 | 20 | Std dev calculation is exact after N bars |
| ATR(14) | 14 | 100+ | Wilder smoothing like RSI |
| ADX(14) | 28+ | 150+ | Double-smoothed DI calculations |
| Stochastic(14,3) | 14 | 17+ | K needs 14 bars, D adds 3 |
| StochRSI(14,14,3) | 28+ | 200+ | RSI warm-up + Stochastic on RSI |
| VWAP_daily | 1 | 1+ | Resets each day; valid from first bar of session |
| CVD | 1 | 50+ | Cumulative; early values lack context |
| OBV | 1 | 50+ | Cumulative; same as CVD |

**Multi-timeframe trap:** An EMA(200) on 4h within a 1h backtest needs 200 * 4 = 800 hourly bars of warm-up.

**Data sufficiency rule:** Reject any strategy/timeframe combination where `available_bars - warm_up_bars < 10 * max_indicator_period`. An EMA(200) strategy on 1h data needs at least 2,000 valid post-warmup bars.

### Indicator Redundancy Matrix

Before combining indicators, check for redundancy. HIGH redundancy pairs measure the same underlying dimension and should not be combined. Prefer LOW redundancy pairs for complementary signal construction.

| | MA/EMA | MACD | RSI | Stochastic | BB | ATR | ADX | Volume |
|---|---|---|---|---|---|---|---|---|
| **MA/EMA** | - | HIGH | LOW | LOW | MED | LOW | MED | LOW |
| **MACD** | HIGH | - | MED | MED | LOW | LOW | MED | LOW |
| **RSI** | LOW | MED | - | HIGH | LOW | LOW | LOW | LOW |
| **Stochastic** | LOW | MED | HIGH | - | LOW | LOW | LOW | LOW |
| **BB** | MED | LOW | LOW | LOW | - | HIGH | LOW | LOW |
| **ATR** | LOW | LOW | LOW | LOW | HIGH | - | MED | LOW |
| **ADX** | MED | MED | LOW | LOW | LOW | MED | - | LOW |
| **Volume** | LOW | LOW | LOW | LOW | LOW | LOW | LOW | - |

**Best complementary pairings:** EMA (trend) + RSI (momentum) + ATR (volatility) + Volume. Each measures a different market dimension.

### Additional Data Columns (Binance Klines)

The Binance CSV includes columns beyond basic OHLCV that are currently unused by `data_loader.py` but valuable for advanced strategies:

| Column | Description | Use Case |
|--------|-------------|----------|
| `taker_buy_base_volume` | Volume where buyer was the aggressor (market buys) | Order flow analysis, CVD, taker buy ratio |
| `taker_buy_quote_volume` | Same in quote currency (USD) | Dollar-weighted flow analysis |
| `quote_volume` | Total volume in quote currency | Amihud illiquidity, dollar-normalized metrics |
| `trades` | Number of individual trades in the bar | Trade intensity, average trade size, whale detection |

**To use these:** Pass `extra_columns=["taker_buy_base_volume"]` to `load_candles()`. The `cvd()` indicator and `CvdDivergenceStrategy` use this automatically. Strategies can access `df["taker_buy_base_volume"]` etc. in `generate_signals()`.

---

## Order Flow & Microstructure Analysis

Order flow analysis exploits the information in *how* trades happen, not just the resulting OHLCV bars. Since Binance data includes `taker_buy_base_volume`, you have direct aggressor-classified order flow — no proxies needed.

### Core Metrics

**Taker Buy Ratio (TBR):**
```
TBR = taker_buy_base_volume / volume
// TBR = 0.50: balanced  |  > 0.60: strong buyer aggression  |  < 0.40: strong seller aggression
```

**Taker Buy/Sell Imbalance (normalized -1 to +1):**
```
imbalance = 2 * TBR - 1
// +1 = all buyers  |  0 = balanced  |  -1 = all sellers
```

**Cumulative Volume Delta (CVD):**
```
delta[i] = 2 * taker_buy_base_volume[i] - volume[i]
CVD[i] = CVD[i-1] + delta[i]
// CVD rising while price flat/falling = hidden accumulation
// CVD falling while price flat/rising = hidden distribution
```

### VWAP (Volume-Weighted Average Price)

**Intraday VWAP (resets daily at 00:00 UTC):**
```
typical_price = (high + low + close) / 3
VWAP = cumulative(typical_price * volume) / cumulative(volume)

Standard deviation bands:
  variance = cumulative((typical_price - VWAP)^2 * volume) / cumulative_vol
  Upper_N = VWAP + N * sqrt(variance)     // N = 1, 2, 3
  Lower_N = VWAP - N * sqrt(variance)
```

**Anchored VWAP (multi-day):** Start from a significant anchor point (swing high/low, weekly open, monthly open) and cumulate forward without resetting. Weekly AVWAP (Monday 00:00 UTC) and monthly AVWAP (1st of month) are standard institutional references.

**VWAP strategy concepts:**
- Mean reversion at +/- 2 sigma bands (win rate ~55-60%, best during 08:00-20:00 UTC)
- VWAP reclaim/loss as trend signal (price crossing VWAP after being on other side for 60+ bars)
- Multi-day AVWAP confluence as support/resistance filter

**BTC-specific:** Skip first 5 bars of each session (VWAP unstable). Use 2-sigma for mean reversion (1-sigma generates too many signals in BTC's volatility). In strongly trending markets, VWAP mean reversion generates repeated losers.

### Volume Profile (VPVR)

Aggregate volume at each price level to identify structural support/resistance:

```
1. Divide the price range into N bins (e.g., 50-100 bins)
2. For each bar, assign its volume to the bin containing its close (or distribute across high-low range)
3. Point of Control (POC) = price bin with highest volume
4. Value Area = price range containing 70% of total volume (centered on POC)
5. Value Area High (VAH) / Value Area Low (VAL) = boundaries
```

**Use as filter:** Strategies entering near POC face mean-reversion pressure. Strategies entering outside the Value Area face breakout/continuation dynamics. Prefer entries where signal direction aligns with the Value Area structure.

### Liquidation Cascade Detection

Detect cascading liquidations from OHLCV + volume without L2 data:

```
cascade_flag = TRUE if ALL of:
  1. |bar_return| > 0.5% in a single 1m bar
  2. volume z-score > 3.0 (vs 120-bar rolling mean)
  3. Same-direction returns for 3+ consecutive bars
  4. Cumulative move over last 10 bars > 3%
```

**Post-cascade mean reversion:** After a >5% drop in <30 minutes, wait for 3 bars of stabilization (|return| < 0.1%), then enter long targeting 40% retracement. Historical win rate ~60-70% for drops of 5-15%.

| Cascade Size | Typical 4h Bounce | Typical 24h Bounce | Win Rate |
|---|---|---|---|
| -5% to -8% | +3% to +5% | +5% to +8% | ~60% |
| -8% to -15% | +5% to +8% | +8% to +12% | ~65% |
| -15% to -25% | +8% to +15% | +10% to +17% | ~70% |

**Regime filter required:** During bear markets (price below 200-day MA), cascade bounces have substantially lower win rates. Use only when longer-term trend is neutral or bullish.

### Microstructure Signals Surviving at 1m Resolution

| Signal | Formula | BTC-Specific Notes |
|---|---|---|
| **Roll Spread** | `2 * sqrt(-cov(delta_p[i], delta_p[i-1]))` over 20 bars | Spike > 2σ above mean = liquidity deterioration. ~30-40% of estimates are NaN (positive cov); set to 0. |
| **Corwin-Schultz Spread** | Uses consecutive high/low pairs | More stable than Roll. Produces negative values ~25% of time; clip to 0. |
| **Amihud Illiquidity** | `|return| / quote_volume` | Compute on 5m bars (resample from 1m) to reduce microstructure noise bias. |
| **Trade Intensity** | `avg_trade_size = volume / trades` | When > 2x its 60-bar EMA → large player activity; next 30-60 bars show continuation ~58% of time. |
| **Volatility Signature** | `RV_1m / RV_5m` over 60 bars | Ratio > 1.3 = high microstructure noise regime. Ratio ~1.0 = efficient. |

**Best use:** These are **regime/filter signals** (Sharpe improvement +0.2-0.4 when overlaid), not standalone strategies.

### Implementation Priority for Order Flow

| Priority | Signal | Type | Expected Sharpe Impact | Difficulty |
|---|---|---|---|---|
| 1 | VWAP mean reversion (2σ) | Standalone | 0.5-0.9 | Low |
| 2 | CVD divergence | Filter/Standalone | 0.4-0.7 | Low (load extra column) |
| 3 | Liquidation cascade bounce | Opportunistic overlay | 1.0-1.5 per-trade | Medium |
| 4 | Taker buy ratio imbalance | Filter | +0.1-0.3 improvement | Low |
| 5 | Volume profile POC/VA | Filter | +0.1-0.2 improvement | Medium |
| 6 | VPIN toxicity filter | Regime filter | +0.2-0.4 improvement | High (volume-time bucketing) |
| 7 | Microstructure regime (Roll/Amihud) | Regime filter | +0.2-0.4 improvement | Medium |

---

### Strategy Interface

```python
class MyStrategy(Strategy):
    def __init__(self, param1=default, ...):
        self.param1 = param1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Return: 1=long, -1=short, 0=flat
        ...

    def param_grid(self) -> dict[str, list]:
        return {"param1": [...], "param2": [...]}
```

**File location:** `strategies/{name}_strategy.py` — the CLI finds the first `Strategy` subclass automatically.

### Available Strategies

List dynamically: `python3 run_backtest.py --list-strategies`

Or in Python: `from strategies import list_strategies; list_strategies()`

### New Strategies (2026-02-19) — Tier 1 Intraday/Swing

Three new strategies were implemented and validated across 4 timeframes (5min, 15min, 1h, 4h) and 8 market regimes (2021-2025):

| Strategy | File | Type | Best TF | 2025 Sharpe | Regime Robustness | Verdict |
|----------|------|------|---------|-------------|-------------------|---------|
| `session_vwap` | `session_vwap_strategy.py` | VWAP mean-reversion with ADX filter + ATR stop | N/A | Negative everywhere | 0/8 positive | **Not viable** — wins too small vs costs |
| `cvd_divergence` | `cvd_divergence_strategy.py` | Order flow divergence (price vs CVD pivots) | **4h** | **+0.58** | **5/8 positive** (avg 0.48) | **Keep** — diversifier, excels in bear markets |
| `session_momentum` | `session_momentum_strategy.py` | European session open (08:00 UTC) predicts daily direction | 5m/15m | +0.45-0.53 | 3/8 positive | Conditional — sideways only |

**CVD Divergence at 4h** is the key finding. Best params: `pivot_lookback=8, cvd_smoothing=3, atr_stop_mult=2.0`. Regime performance:
- 2022-H1 Bear: **Sharpe 2.23** (+85%) — excels when trend-following fails
- 2022-H2 Sideways: Sharpe 1.79 (+41%)
- 2024-H1 ETF Bull: Sharpe 1.07 (+21%)
- Weak in: 2023 Recovery (-1.37), 2024-H2 Consolidation (-0.97)

**Session Momentum** works at 5min/15min with European session open. Best params: `session_start_utc=8, opening_range_minutes=30, min_opening_return_pct=0.25, hold_hours=18, volume_filter=True`. Academically backed by Shen et al. (2022) but regime-dependent.

### Session Enrichment

```python
from backtester import enrich_sessions
df = enrich_sessions(df)
# Adds: hour (0-23), session ('asian'|'european'|'eu_us_overlap'|'us'),
#        is_weekend (bool), is_peak_liquidity (bool, 08:00-20:00 UTC weekdays)
```

---

## Known Framework Limitations

Be aware of these when interpreting results. These do not invalidate the protocol but affect how aggressively you should trust numbers.

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| ~~**Flat fee model**~~ **FIXED** — Tier 2 impact-adjusted model now implemented | `BacktestConfig(impact_coefficient=0.20, funding_rate_8h=0.0001)` enables square-root market impact + funding drag. Default Tier 1 corrected to 10 bps (was 15 bps). | Use `--impact-coeff 0.20 --funding-rate 0.0001` via CLI for realistic costs. |
| **No position sizing** — all-in or all-out (1 or 0 or -1) | Cannot model fractional Kelly, volatility-scaled sizing, or pyramiding | Treat backtest as signal quality assessment, not portfolio simulation. |
| ~~**Sharpe inflation on minute data**~~ **FIXED** — `compute_metrics()` now auto-resamples sub-hourly returns to 1h before Sharpe/Sortino calculation | Sharpe values are now comparable across timeframes. A 5min Sharpe of 1.0 now correctly reflects the hourly-equivalent risk-adjusted return. | No action needed — fix is automatic. |
| **RSI division by zero** — when all price moves are gains, `avg_loss=0` produces NaN | NaN propagates through signals silently | Inspect for NaN in signals before trusting results. Flag if detected. |
| ~~**Profit factor "inf"**~~ **FIXED** — now returns `999.99` instead of string `"inf"` | Numeric comparisons work correctly. | No action needed. |
| **Forward-fill signal persistence** — `replace(0, pd.NA).ffill()` holds positions indefinitely until opposite signal | RSI/Bollinger strategies may hold through entire trends without explicit exit | Understand this is a design choice, not a bug. Consider exit conditions when designing strategies. |
| **No explicit exit mechanisms** — no stop-loss, take-profit, trailing stop, or time-based exits | Drawdowns can grow unchecked; mean-reversion trades held through regime changes | Design exit logic within `generate_signals()`. See Exit Strategy Guidance below. New strategies (CVD Divergence, Session VWAP) implement ATR-based stops as examples. |
| ~~**No funding rate modeling**~~ **FIXED** — `funding_rate_8h` in BacktestConfig applies per-bar funding drag | Set `funding_rate_8h=0.0001` (~10.9% annualized) for perpetual futures backtests. | Use `--funding-rate 0.0001` via CLI. |

### Realistic Execution Cost Model

The flat fee model underestimates real costs. Use these conservative all-in estimates (entry + exit) as baselines:

| Venue & Method | Round-Trip Cost (bps) | Notes |
|----------------|----------------------|-------|
| Perps, taker, base tier | 15-20 | Binance/Bybit/OKX standard taker fee + spread |
| Perps, maker, base tier | 6-10 | Assumes 60-80% fill rate on limit orders |
| Spot, taker, base tier | 22-30 | Higher fees than perps on most exchanges |
| Perps, taker, VIP tier | 6-12 | High-volume traders with fee discounts |

**Slippage by order size (BTC, normal conditions):**

| Order Size | Expected Slippage | Total Cost (+ taker fee) |
|------------|-------------------|--------------------------|
| < $100K | 1-3 bps | ~13-23 bps round trip |
| $100K-$500K | 3-8 bps | ~16-28 bps round trip |
| $500K-$2M | 8-20 bps | ~28-40 bps round trip |
| > $2M | 20-50+ bps | Use TWAP/VWAP execution |

**Time-of-day adjustment:** Slippage varies ~1.5-2x between peak liquidity (12:00-16:00 UTC) and low liquidity (20:00-04:00 UTC). Apply a 1.5x multiplier when backtesting strategies that trade outside peak hours.

### Advanced Transaction Cost Modeling

The flat fee model (`fee_rate + slippage_pct`) is a first-pass filter only. For strategies approaching viability thresholds, use the tiered cost models below. The current engine's 15 bps total (10 fee + 5 slippage) overstates costs at small sizes (<$100K) and understates them at institutional scale.

**Square Root Market Impact Model:**

Empirical evidence shows market impact scales sublinearly with order size:

```
slippage_fraction = base_slippage + impact_coefficient × sqrt(trade_notional / ADV_daily)
```

Where `impact_coefficient` captures permanent + temporary price impact beyond the spread, and ADV_daily is derivable from the existing `quote_volume` column as a 20-day rolling sum.

**Impact Coefficient by Regime (BTC on Binance):**

| Regime | ADV Context | `impact_coefficient` | Impact at 0.1% Participation |
|---|---|---|---|
| High liquidity, low vol (<40%) | ADV $25B+ | 0.10 – 0.15 | 1.0 – 1.5 bps |
| Normal liquidity, moderate vol (40-70%) | ADV $15-25B | 0.15 – 0.25 | 1.5 – 2.5 bps |
| Stress / low liquidity (vol >70%) | ADV $8-15B | 0.30 – 0.50 | 3.0 – 5.0 bps |
| Weekend illiquid session | ADV <$8B | 0.50 – 0.80 | 5.0 – 8.0 bps |

**Default:** `impact_coefficient = 0.20` with `ADV = $20B`.

Example at $10K notional: `slippage = 0.0002 + 0.20 × sqrt(10000 / 20B) = 0.000341` (~3.4 bps). At $100K: ~6.5 bps.

**Intraday Impact Multipliers (applied to base coefficient):**

| Time Window (UTC) | Session | Multiplier |
|---|---|---|
| 00:00 – 06:00 | Asia | 1.3x – 2.5x |
| 06:00 – 08:00 | Pre-EU | 1.1x – 1.3x |
| 08:00 – 13:00 | EU active | 0.8x – 1.0x |
| 13:00 – 17:00 | EU/US overlap | 0.6x – 0.8x |
| 17:00 – 20:00 | US afternoon | 0.9x – 1.1x |
| 20:00 – 00:00 | Late US / early Asia | 1.2x – 2.0x |

**Volatility Regime Impact Multipliers:**

| Realized 30-day Vol (annualized) | Multiplier |
|---|---|
| < 30% (very low) | 0.7x |
| 30-50% (normal BTC) | 1.0x |
| 50-80% (elevated) | 1.5x |
| 80-120% (high, trending) | 2.0x – 2.5x |
| > 120% (extreme crash/squeeze) | 3.0x – 5.0x |

Combined example: 01:00 UTC during 80% vol regime → `effective_coefficient = 0.20 × 2.2 × 2.0 = 0.88`. Slippage on $100K rises from 6.5 bps to ~28 bps.

### Funding Rate Drag (Perpetual Futures)

Binance perpetuals settle funding every 8 hours (00:00, 08:00, 16:00 UTC). Longs pay shorts when funding is positive (most of the time in bull markets).

```
annualized_funding_drag = avg_funding_rate × (24/8) × 365 = avg_funding_rate × 1095
```

**Historical BTC Funding Rate Distribution (Binance, 2020-2025):**

| Metric | Value |
|---|---|
| Mean 8-hour rate | +0.012% to +0.018% |
| Median 8-hour rate | +0.008% to +0.012% |
| 90th percentile (bull markets) | +0.030% to +0.075% |
| Annualized drag at mean rate | 13% – 20% per year |
| Annualized drag at 90th pct | 33% – 82% per year |

**Impact by Strategy Type:**
- **Long trend-following:** Most affected. During 2020-2021 bull, avg funding >0.05%/8h = 54% annualized drag. Strategies "profitable" before funding become losers.
- **Short trend-following:** Receives funding when positive (most of the time). Short strategies in bear markets (negative funding) must pay.
- **Intraday mean-reversion (<8h hold):** Largely immune — only pays if a settlement point (00:00, 08:00, 16:00 UTC) is crossed. Probability per trade ~25-50% for 2-4h average holds.
- **Carry strategies:** Explicitly harvest funding premium. Not currently supported in backtester but documented for awareness.

**Default for backtesting:** Use `funding_rate = 0.0001` per 8h period (~10.9% annualized). For each position held through a settlement, apply `funding_cost = position_sign × funding_rate × position_notional`.

### Spread-Variable Cost Model

The actual BTC half-spread on Binance is 0.1-4.0 bps depending on session — dramatically below the current 5 bps flat slippage assumption.

**Effective Half-Spread by Session (Binance BTCUSDT Perp, 2024-2025):**

| Session | Time (UTC) | Half-Spread (bps) | + Taker Fee (5 bps) | Round-Trip (bps) |
|---|---|---|---|---|
| EU/US Overlap | 13:00-17:00 weekday | 0.15 – 0.30 | 5.15 – 5.30 | 10.3 – 10.6 |
| EU Active | 08:00-13:00 weekday | 0.25 – 0.50 | 5.25 – 5.50 | 10.5 – 11.0 |
| Asia Active | 02:00-06:00 weekday | 0.50 – 1.00 | 5.50 – 6.00 | 11.0 – 12.0 |
| Off-Peak | 20:00-02:00 weekday | 1.00 – 2.00 | 6.00 – 7.00 | 12.0 – 14.0 |
| Weekend Day | Sat-Sun 09:00-20:00 | 1.50 – 3.00 | 6.50 – 8.00 | 13.0 – 16.0 |
| Weekend Night | Sat-Sun 20:00-09:00 | 2.00 – 4.00 | 7.00 – 9.00 | 14.0 – 18.0 |

**Simplified 3-regime model:** half-spread = 0.30 bps (EU/US overlap weekday), 0.90 bps (Asia/off-peak weekday), 2.00 bps (weekend any time).

### Three-Tier Cost Model

**Tier 1 — Flat Model (default engine, quick filter): ✅ IMPLEMENTED**
```
cost_per_trade = (fee_rate + slippage_pct) × |position_change|
```
Default: fee_rate=0.0005 (5 bps) + slippage_pct=0.0005 (5 bps) = 10 bps/side, 20 bps round-trip.
CLI: `python3 run_backtest.py --fee 0.0005 --slippage 0.0005`

**Tier 2 — Impact-Adjusted Model (recommended for strategy evaluation): ✅ IMPLEMENTED**
```
fee_cost = fee_rate × |Δposition|
impact_cost = impact_coefficient × sqrt(trade_notional / ADV) × |Δposition|
funding_cost = funding_rate_8h × (bar_hours / 8) × |position|
total = fee_cost + impact_cost + funding_cost
```
CLI: `python3 run_backtest.py --impact-coeff 0.20 --funding-rate 0.0001`

**Cost at Key Trade Sizes (ADV=$20B, coefficient=0.20):**

| Trade Notional | Fee + Spread (bps) | Impact (bps) | Funding (8h hold) | Total Round-Trip |
|---|---|---|---|---|
| $1,000 | 10.0 | 0.45 | negligible | ~10.5 bps |
| $10,000 | 10.0 | 1.41 | 0.08 | ~11.5 bps |
| $100,000 | 10.0 | 4.47 | 0.8 | ~15.3 bps |
| $1,000,000 | 10.0 | 14.14 | 8.0 | ~32.1 bps |

**Tier 3 — Full Model (session-aware + impact + funding + adverse selection):**
```
total_one_way = session_spread(timestamp) + adverse_selection + impact + taker_fee + funding
```
Where `adverse_selection = half_spread × max(0, proxy_VPIN - 0.25) × 4.0` adds cost during high-toxicity flow periods (proxy_VPIN from taker_buy_volume imbalance).

### Breakeven Analysis as Function of Position Size

**Minimum viable gross edge for the backtester's $10K capital:**
```
E_min = 2 × (fee + spread + impact_at_10K) = 2 × (5 + 0.30 + 1.41) = ~13.4 bps
```

Any strategy with gross edge below 13-14 bps per trade is not viable even at $10K. At $100K: minimum viable edge rises to ~19 bps.

**Breakeven notional formula (where edge is entirely consumed by costs):**
```
N_breakeven = ADV × ([E/2 - fee - spread] / (impact_coeff × 10000))²
```

| Gross Edge E (bps) | Breakeven Notional N_max |
|---|---|
| 12 | $2,450 |
| 15 | $24,200 |
| 20 | $110,450 |
| 30 | $471,100 |
| 50 | $1,943,000 |
| 100 | $10,010,000 |

**Key insight:** For every 10 bps of additional gross edge, breakeven notional increases ~4-5x (due to sqrt in impact model). Off-peak execution adds 3-5 bps to the cost floor, shrinking breakeven by 20-40%.

### Recommended Default Parameters by Use Case

| Use Case | Fee (bps) | Slippage (bps) | Impact Coeff | Funding Rate | Notes |
|---|---|---|---|---|---|
| Quick filter (Tier 1 corrected) | 5 | 5 | none | none | Fix current 15→10 bps |
| Retail intraday (<$50K) | 5 | 3 | 0.20 | 0.01%/8h | Primary recommendation |
| Retail swing (overnight) | 5 | 4 | 0.20 | 0.01%/8h | Add funding drag per hold |
| Institutional ($1M+) | 3 | — | 0.25 | 0.008%/8h | Maker rebates offset spread |
| Conservative stress test | 5 | — | 0.40 | 0.02%/8h | Test robustness |

---

## Pre-Backtest Validation

Run these checks before every backtest to catch data quality issues and signal bugs early. Skipping these leads to silent failures.

### Data Quality Checks

1. **OHLC consistency:** Verify `low <= open <= high` and `low <= close <= high` for every bar. Violations = corrupted data.
2. **Gap detection:** Compute timestamp diffs. Any gap > 1.5x expected interval is a missing bar. Gap fraction > 1% -> reject or explicitly fill.
3. **Price spike detection:** Flag bars where `|close_return| > 20%` for review. Use rolling IQR (window=100) rather than z-score for robustness.
4. **Zero-volume bars:** If > 5% of bars have zero volume, the pair may be too illiquid for reliable backtesting.
5. **Timestamp monotonicity:** Assert all timestamps strictly increasing. Duplicates indicate assembly bug.

### Signal Quality Pre-Checks

After generating signals but before running a full backtest:

1. **NaN check:** `signals.isna().sum()` must be 0 after warm-up period. Any NaN = indicator bug.
2. **Signal distribution:** If buy signals < 0.1% of bars or > 50% of bars, thresholds are miscalibrated.
3. **Value check:** `signals.isin([0, 1, -1]).all()` must be True. Other values break the engine.
4. **Indicator range sanity:** RSI must be in [0, 100]. ATR and BB width must be positive.

### Look-Ahead Bias Checklist

Look-ahead bias extends beyond peeking at future prices. These subtle forms are more dangerous because they are harder to detect. **Check every item before trusting any backtest result.**

**Resampling look-ahead (critical for this framework):**
- When resampling 1m to 1h: the hourly bar timestamped 10:00 contains data from 10:00-10:59 (with default `closed='left', label='left'`). Signals computed on this bar should only execute at 11:00 or later. The engine handles this via next-bar execution — **do not add manual shifts on top of the engine's shift.**
- Volume-weighted indicators (VWAP, OBV) use the full bar's volume, which is only known at bar close. If a strategy trades intrabar, indicators must use only completed bars.

**Pandas-specific pitfalls:**
- `.bfill()` / `.fillna(method='bfill')` — fills with the **next** valid value. Pure look-ahead. **Always use `.ffill()` for time series.**
- `.interpolate()` — default linear uses both past and future. Use `.interpolate(limit_direction='forward')` only.
- `.rolling(center=True)` — centers the window, using future data. **Always use `center=False`** (the default).
- `.shift(-1)` — shifts data backward in time (look-ahead). For lagged features, use `.shift(1)` (positive).
- `groupby().transform('mean')` on intraday data — computes the daily mean using all intraday values. Look-ahead if applied to intraday signals.
- Z-score normalization using `(x - x.mean()) / x.std()` on the full series — uses future values. Use `.expanding().mean()` and `.expanding().std()` instead.
- Index alignment in `.join()` / `.merge()` — can silently align future data to past timestamps when DataFrames have different frequencies. Verify alignment after every join.

### Error Recovery: Zero-Trade Diagnostic

When a sweep produces zero or near-zero trades:

```
Step 1: Check NaN ratio in signals -> if high, warm-up period too short
Step 2: Check signal distribution -> are thresholds ever crossed?
Step 3: Check indicator value ranges -> is RSI ever reaching 30/70?
Step 4: If trades exist but all losing -> test without fees first (cost-eating edge?)
Step 5: Try inverting signal -> if inverted works, logic is reversed
```

---

## Enhanced Data Preprocessing

Beyond the basic pre-backtest checks above, apply these preprocessing steps for production-quality results.

### Outlier Detection Methods

Simple z-scores fail for BTC because legitimate 10-20% moves happen regularly. Use robust methods:

**IQR Method (preferred for price spikes):**
```
Q1, Q3 = rolling_quantile(returns, 100, [0.25, 0.75])
IQR = Q3 - Q1
outlier = (returns < Q1 - 3.0 * IQR) | (returns > Q3 + 3.0 * IQR)
// Use 3.0x (not 1.5x) for BTC — tighter thresholds flag legitimate moves
```

**Hampel Filter (preferred for detecting isolated spikes in indicator values):**
```
rolling_median = series.rolling(window).median()
MAD = 1.4826 * (series - rolling_median).abs().rolling(window).median()
outlier = (series - rolling_median).abs() > 3.0 * MAD
// MAD is robust to the outliers it's trying to detect, unlike std
```

**Decision: Remove vs Winsorize vs Keep:**

| Situation | Action |
|-----------|--------|
| Single-bar spike with volume < 1% of rolling median | Winsorize high/low to 5x ATR from close (likely flash wick) |
| Multi-bar move with high volume | Keep (legitimate market move) |
| Post-maintenance gap first bar | Exclude from signal generation for 5 bars |
| Bar with zero volume | Flag but keep; treat as unreliable for volume-based indicators |

### Gap Filling Strategies

When minute bars are missing from the dataset:

| Gap Size | Action |
|----------|--------|
| 1-5 bars | Forward-fill close price. Set volume to 0 for filled bars. Mark as synthetic. |
| 6-60 bars (maintenance) | Do not fill. Log gap timestamps. Exclude 5 bars after gap end from signal generation. |
| > 60 bars | Reject the segment. Split into before/after and test separately. |

**Never interpolate prices** — any interpolation between known prices is a form of look-ahead bias. Forward-fill is the only safe method.

### Binance-Specific Data Issues

| Issue | Detection | Handling |
|-------|-----------|---------|
| Maintenance windows | Gaps > 30 min in timestamps | Mark, do not fill. Log exact timestamps. |
| Post-maintenance spikes | First 5 bars after any gap > 10 min | Exclude from signal generation |
| Flash wicks (thin liquidity) | `(high - low) > 5x ATR(14)` | Winsorize high/low to 5x ATR from close |
| Thin volume bars | `volume < 1% of rolling_median(1440)` | Flag; treat volume-based indicators as unreliable |

### Resampling Correctness

The current `data_loader.py` uses `closed='left', label='left'` — this is **correct** for Binance crypto data. These settings must not be changed.

**Why:** A Binance 1m bar timestamped 10:00 contains price action from 10:00:00.000 to 10:00:59.999. Resampling to hourly with `closed='left'` correctly groups 10:00-10:59 minute bars into the 10:00 hourly bar. Using `closed='right'` shifts every bar by one source bar, creating systematic look-ahead bias (+0.5-2% Sharpe inflation in trend-following).

**Volume aggregation must be `sum`**, not mean/first/last. The current code is correct.

**After resampling:** Use `dropna(subset=["close"])` instead of plain `dropna()` to keep bars with price data but zero volume.

### Data Vintage Tracking

Record the SHA-256 hash and download date of every data file in the run report JSON. If re-running a backtest and the hash differs, the data has been retroactively modified — investigate before trusting results.

**BTC fork adjustments:** Backtests spanning 2017+ should account for the BCH fork (Aug 2017, ~5% of BTC price received as BCH). Ignoring forks understates cumulative return for long strategies.

### Survivorship Bias Disclosure

The current framework backtests on BTCUSDT — the ultimate survivor asset. Always include this disclosure in cross-symbol results:

> "Backtest performed on BTCUSDT. BTC is the highest-survivorship crypto asset. Results should not be generalized to altcoins without survivorship-free multi-asset testing."

### Wash Trading Awareness

For BTC on Binance (Tier 1 exchange), wash trading is a minor concern. For altcoins or Tier 2-3 exchanges, check:
- High volume with near-zero price change (volume z > 2, |return| < 0.05%)
- Taker buy ratio near exactly 50% for extended periods
- Volume clustering at round numbers

If wash trading signals are present, do not trust volume-based strategy signals.

---

## Backtest Engine Validation

The engine must be validated with synthetic data before trusting any strategy result. A single-engine test suite catches internal bugs; cross-engine validation catches systematic errors. Run these checks whenever `engine.py`, `metrics.py`, `data_loader.py`, or `strategy.py` are modified.

### Fill Model Specification

Document every behavioral choice so each decision is unambiguous:

**Signal Bar vs. Execution Bar:** Signal at bar `i` close → position at bar `i+1`. Implemented via `signals.shift(1, fill_value=0)`. The return earned is `close[i+2] / close[i+1] - 1`. This prevents the most dangerous look-ahead: using the same bar's close to both generate and profit from a signal.

**Fee Application:** `|position[i] - position[i-1]| × (fee_rate + slippage_pct)` subtracted from gross return of bar `i`. Long-to-short reversal: `|-1 - 1| = 2` → fee = 2× combined rate (correct — pays for both legs). No flat intermediate bar is inserted during reversals.

**Equity Curve:** Geometric compounding via `initial_capital × cumprod(1 + strategy_return)`. A -50% followed by +50% = 75% of starting (0.5 × 1.5 = 0.75). Flat-period bars contribute `(1+0) = 1.0` — correctly leaving equity unchanged.

**Short P&L:** `position × close_return` where position = -1. Bar-by-bar accumulation (geometrically correct for multi-bar holds).

**Last Bar:** Open positions are NOT explicitly closed. No exit fee charged at data end. Trade list may show fewer trades than expected. Account for this in any trade-count assertions.

### Synthetic Test Cases

Run these 5 tests to validate engine correctness:

**Case 1 — Perfect Uptrend:** 200 bars of monotonic +0.1%/bar. EMA crossover (fast=10, slow=30). Expected: positive return, first position at bar 12 (not 11). Verify no look-ahead by checking first non-zero position = signal bar + 1.

**Case 2 — Sine Wave:** `price = 100 + 10×sin(2π×i/50)` for 200 bars. Tests entry/exit timing, long-to-short transitions, and fee charging at reversals. Verify fee at reversal bar = `2 × combined_rate`.

**Case 3 — Random Walk:** Gaussian returns, 200 bars. Expected: strategy return near zero (no systematic edge). Sharpe should be within ±1.0 of zero. Confirms the engine does not create spurious alpha.

**Case 4 — Volatility Spike:** 190 flat bars + 10 extreme bars (-10%, +12%, -8%, +15%, -6%, +4%, -3%, +2%, -1%, +7%). Verifies geometric compounding handles extreme returns correctly. After -50% then +100%, equity ≈ pre-spike level (not higher, which would indicate arithmetic compounding).

**Case 5 — Flat Market (Fee Isolation):** Constant price, strategy generates known position changes. Expected: `total_return = -total_fees_paid`. Verify `expected_fees = Σ|Δposition| × combined_rate × initial_capital` matches within ±$0.01.

### Audit Trail Requirements

For every bar, store or reconstruct: `bar_index`, `timestamp`, `close_price`, `raw_signal`, `executed_position`, `close_return`, `gross_strategy_return`, `position_change`, `fee_charged`, `net_strategy_return`, `equity`.

**Reconstruction verification:** `reconstructed_terminal = initial_capital × Π(1 + net_strategy_return[i])`. Must match `equity_curve.iloc[-1]` within 1e-6. Run this check after every backtest, not just during testing.

**Per-trade audit fields (currently missing — priority enhancement):** `entry_signal_bar`, `fees_entry`/`fees_exit` split, `equity_at_entry`, `equity_at_exit`. Without equity-at-entry/exit, trade-level P&L cannot be confirmed against the equity curve.

### Common Engine Bugs

| Bug | Manifestation | Detection |
|---|---|---|
| Off-by-one signal shift | Strategy appears far too profitable | Case 1: first non-zero position must be signal_bar + 1 |
| Resampling look-ahead | `closed='right'` leaks future data | Verify resampled daily close = last bar's close (not next day's) |
| Fee double/under-counting | Fee totals don't match analytically | Case 5: assert `fees_paid == Σ|Δpos| × rate × capital` within $0.01 |
| Flat-period survivorship | Equity jumps during position=0 bars | Case 5: assert equity has N bars, flat periods unchanged |
| Signal dtype overflow | int8 wraps -1 to 127 on intermediate values | Assert `signals.isin([-1, 0, 1]).all()` before engine shift |
| Floating-point accumulation | Equity drifts on 50K+ bars | Run 50K bar flat test; assert terminal = initial within 1e-6 |

**Resampling safety:** Always pass `label='left', closed='left'` explicitly to `df.resample()` — pandas defaults change for weekly/monthly frequencies.

### Cross-Engine Validation

Validate against **vectorbt** `Portfolio.from_signals()` as the reference engine:

**Configuration alignment:** Match `freq`, `fees` (combined rate), `init_cash = initial_capital`, `size=1.0`, `size_type='percent'`, `direction='both'`.

**Watch for mismatches:** vectorbt defaults to fill at next bar's *open* (not close). Set `upon_long_conflict='ignore'` and verify fill behavior. Annualization: this engine uses 365.25 days; vectorbt defaults to 252.

**Metric match tolerances:**

| Metric | Tolerance | Notes |
|---|---|---|
| total_return_pct | ±0.1 pp | Use total, not annualized (different year lengths) |
| sharpe_ratio | ±0.05 | Annualization convention may differ |
| max_drawdown_pct | ±0.1 pp | Percentage-based, not duration |
| total_trades | exact ±1 | Boundary handling may differ by 1 |
| total_fees_paid | ±0.1% relative | Slippage compounding can produce small differences |

**Procedure:** Use 90-day 1-hour BTC bars (~2,160 bars), EMA crossover (fast=10, slow=30). Export equity curves from both, compute `max(|equity_A[i] - equity_B[i]| / equity_A[i])` — must be <0.5%.

### Validation Checklist (Gate Before Any Strategy Evaluation)

- [ ] All 5 synthetic test cases pass
- [ ] Equity reconstruction within 1e-6 tolerance
- [ ] Cross-validation equity divergence <0.5% at all points
- [ ] Signal values ∈ {-1, 0, 1}, no NaN
- [ ] Fee isolation (flat market) exact within $0.01
- [ ] Determinism: identical results on two independent runs
- [ ] Resampling uses explicit `closed='left'` convention

Re-run whenever `engine.py`, `metrics.py`, `data_loader.py`, or `strategy.py` are modified. Strategy-only changes require only the signal-value assertion test.

---

## Run History & Cross-Run Learning

Each Optimization Loop run is ephemeral — the agent starts fresh every time with no memory of what worked, what failed, what market regimes were observed, or what parameter regions are promising. This section defines a structured capture/retrieval system so that future runs benefit from historical knowledge.

### Storage Layout

```
data/runs/
  run-YYYY-MM-DD-HHmmss.json   # Per-run structured report
  KNOWLEDGE.md                   # Cumulative cross-run knowledge base (template at data/runs/KNOWLEDGE.md)
```

### Run Report JSON Schema

Each completed loop produces a structured JSON file:

```json
{
  "schema_version": "1.0",
  "run_id": "run-2026-02-19-143052",
  "timestamp": "2026-02-19T14:30:52Z",

  "data_source": {
    "file": "data/binance_BTCUSDT_1m_klines.csv",
    "symbol": "BTCUSDT",
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "total_bars_1m": 525600,
    "resample_used": "1h"
  },

  "market_regime": {
    "buy_and_hold_return_pct": -7.2,
    "start_price": 94401.14,
    "end_price": 87648.22,
    "regime_label": "range_bound",
    "sub_period_character": [
      {"period": "2025-01 to 2025-04", "bh_return_pct": 5.2, "label": "choppy"},
      {"period": "2025-05 to 2025-08", "bh_return_pct": -8.1, "label": "mild_bearish"},
      {"period": "2025-09 to 2025-12", "bh_return_pct": -4.3, "label": "recovery"}
    ]
  },

  "convergence": {
    "reason": "stagnation",
    "criterion_number": 2,
    "total_strategies_tested": 9,
    "total_iterations": 9,
    "families_explored": ["mean_reversion", "trend_following", "momentum", "confluence", "breakout"]
  },

  "leaderboard": [
    {
      "rank": 1,
      "strategy_name": "rsi_bollinger",
      "strategy_file": "strategies/rsi_bollinger_strategy.py",
      "mode": "long+short",
      "timeframe": "1h",
      "best_params": {"rsi_period": 30, "overbought": 65, "oversold": 32, "bb_period": 15, "bb_std": 2.0},
      "metrics": {
        "sharpe_ratio": 2.97,
        "sortino_ratio": 4.12,
        "total_return_pct": 237.2,
        "max_drawdown_pct": -23.3,
        "win_rate_pct": 80.6,
        "profit_factor": 5.69,
        "total_trades": 31,
        "time_in_market_pct": 99.0,
        "long_trades": 16,
        "short_trades": 15,
        "long_win_rate_pct": 80.0,
        "short_win_rate_pct": 81.2
      },
      "composite_score": 1.80,
      "robustness": "PASS",
      "robustness_details": {
        "sub_period_results": [
          {"period": "2025-01 to 2025-04", "sharpe": 2.84, "return_pct": 58.8, "profitable": true},
          {"period": "2025-05 to 2025-08", "sharpe": 1.22, "return_pct": 12.5, "profitable": true},
          {"period": "2025-09 to 2025-12", "sharpe": 4.63, "return_pct": 83.7, "profitable": true}
        ],
        "walk_forward": {"is_sharpe": 2.735, "oos_sharpe": 2.394, "oos_is_ratio_pct": 87.5},
        "neighbors_profitable": "10/10",
        "multi_timeframe_4h": "FAIL (only 4 trades)"
      },
      "derivation": {"method": "Method 3: Combine", "parent": "rsi + bollinger_bounce", "parent_sharpe": 2.45}
    }
  ],

  "derivation_genealogy": [
    {"step": 1, "strategy": "rsi", "sharpe": 2.45, "method": "Seed", "parent": null, "family": "mean_reversion"},
    {"step": 2, "strategy": "ema_crossover", "sharpe": -0.83, "method": "Seed", "parent": null, "family": "trend_following"}
  ],

  "key_insights": {
    "what_worked": ["RSI + Bollinger confluence outperformed pure RSI", "Short side profitable in range-bound market"],
    "what_failed": ["All trend-following strategies (EMA, MACD) destroyed by chop", "EMA trend filter too restrictive"],
    "market_observations": ["2025 range-bound (-7.2% B&H)", "Mean-reversion dominates"],
    "derivation_method_effectiveness": {
      "Method 1: Add filter": {"attempts": 1, "improvements": 0},
      "Method 3: Combine": {"attempts": 1, "improvements": 1},
      "Method 4: Change TF": {"attempts": 2, "improvements": 0},
      "Method 5: New concept": {"attempts": 4, "improvements": 1}
    }
  },

  "winner_report": "Full Phase 8 markdown text...",
  "runner_up_report": "Full Phase 8 markdown text..."
}
```

### Regime Classification Table

| B&H Return | Label |
|------------|-------|
| > +100% | `strong_uptrend` |
| +20% to +100% | `mild_uptrend` |
| -20% to +20% | `range_bound` |
| < -20% | `bear_market` |

### Capture Protocol

Execute after convergence, before presenting the Final Report to the user.

1. `mkdir -p data/runs`
2. Assemble JSON from loop state (leaderboard, genealogy, convergence, insights, winner/runner-up reports)
3. Write `data/runs/run-YYYY-MM-DD-HHmmss.json`
4. Read existing `data/runs/KNOWLEDGE.md` (or create from template if first run)
5. Update each section of KNOWLEDGE.md:
   - **Strategy-Regime Performance Matrix** — add/update cells for each tested strategy under the current regime column
   - **Robust Parameter Regions** — confirm or discover new regions from this run's best parameters
   - **Derivation Method Effectiveness** — aggregate attempt/improvement counts
   - **Run Summaries** — prepend new run summary (cap at 10 most recent)
   - **Anti-Patterns** — add if a failure pattern is confirmed across 2+ runs (cap at 15)
   - **Open Questions** — regenerate based on untested families and unexplored regimes
6. Write updated KNOWLEDGE.md
7. Announce file paths to user: "Run report saved to data/runs/run-{timestamp}.json. Knowledge base updated at data/runs/KNOWLEDGE.md."

### Retrieval Protocol

Execute before Phase 1 of each new Optimization Loop.

1. Check for prior runs: `ls data/runs/*.json 2>/dev/null | wc -l`
   - If 0: skip retrieval, note "No prior run history. Starting fresh."
   - If > 0: continue
2. Read `data/runs/KNOWLEDGE.md` in full
3. After Phase 1 regime classification, look up the matching column in the Strategy-Regime Performance Matrix
4. Adjust loop behavior:
   - **Seed selection**: Pick the best historical family for this regime (not always RSI)
   - **Parameter centering**: Use Robust Parameter Regions for initial grids
   - **Derivation priority**: Reorder derivation methods by historical improvement rate
   - **Anti-pattern avoidance**: Skip configurations known to fail in this regime
   - **Family exploration**: Prioritize untested families from Open Questions
5. Announce context to user: "Historical context: [N] prior runs. Regime: [label]. Best historical family: [X] (Sharpe [Y]). Anti-patterns: [list]."

### Similarity Matching

When the exact regime has fewer than 2 data points in the knowledge base, use similarity matching to find the closest historical run:

```
similarity(current, historical) = 1 / (1 + |bh_diff| / 100)
```

Where `bh_diff` is the difference in buy-and-hold return percentage between the current period and the historical run. Load the 1-2 most similar runs' full JSON files for detailed guidance on seed selection, parameter regions, and anti-patterns.

---

## The Optimization Loop

The Optimization Loop wraps the 8-Phase Protocol and adds strategy derivation, convergence checking, and leaderboard tracking. It runs autonomously until convergence criteria are met.

### Loop Architecture

```
OUTER LOOP (strategy-level):
  0. Load run history (see Retrieval Protocol)
  1. Pick/create strategy (use historical regime data to inform seed selection)
  2. INNER LOOP: Run 8-Phase Protocol
     -> If fine-grid Sharpe improvement < 0.02 over broad -> stop refining
     -> Always test BOTH long+short and long-only in Phase 2
     -> Carry better mode (or both if both Sharpe > 0.3) through Phases 3-8
  3. Record best robustness-validated result on Leaderboard
  4. Derive next strategy (use historical derivation effectiveness to prioritize methods)
  5. Convergence check -> if triggered, produce Final Report, then execute Capture Protocol
  6. Else -> go to step 1
```

### Loop State

Maintain this state throughout the loop (track mentally or in notes):

```
Loop iteration:       N
Strategies tested:    [list]
Leaderboard:          [see Leaderboard Format]
Convergence log:      [improvement deltas for last 3 iterations]
Derivation path:      seed -> derived1 -> derived2 -> ...
Total variants tried: N_total  (for multiple testing correction)
```

### Entry Points

- **Fresh start:** Load run history first. If history exists and regime match found, seed with historically best family. Otherwise default to RSI.
- **Resume:** Continue from existing leaderboard.
- **User-directed:** Use named strategy as seed. Still load history for derivation guidance and anti-pattern avoidance.
- **History-informed:** When history exists, announce context to user before Phase 1.

---

## Long+Short Evaluation Protocol

Both long and short modes are first-class. Never assume long-only by default — always test both.

### Phase 2 Dual-Mode Sweep

In Phase 2 of the 8-Phase Protocol, **always** run two sweeps:

```bash
# Long + Short (bidirectional)
python3 run_backtest.py --strategy {name} --resample {tf} --sweep --rank-by sharpe_ratio --top 20

# Long-only
python3 run_backtest.py --strategy {name} --resample {tf} --sweep --rank-by sharpe_ratio --top 20 --long-only
```

### Mode Selection Rules

| Condition | Decision |
|-----------|----------|
| Bidirectional Sharpe > Long-only Sharpe + 0.1 | Use bidirectional |
| Long-only Sharpe > Bidirectional Sharpe + 0.1 | Use long-only |
| Gap <= 0.1 between modes | Prefer long-only (simpler) |
| Both modes have Sharpe > 0.3 | Carry BOTH through Phases 3-8, report both |

### Short-Side Diagnostics

After Phase 2, if bidirectional mode is selected, check the short-side health:

```python
# From metrics:
# long_trades, short_trades, long_win_rate_pct, short_win_rate_pct,
# long_avg_bars_held, short_avg_bars_held

# Short-side drag detection:
if short_win_rate_pct < 40 and avg_short_pnl < 0:
    # Short side is a drag — switch to long-only
```

Compute average short PnL from the trades list:
```python
short_pnls = [t["pnl_pct"] for t in results["trades"] if t["direction"] == "short"]
avg_short_pnl = sum(short_pnls) / len(short_pnls) if short_pnls else 0
```

**Decision:** If avg short PnL < 0 AND short win rate < 40% -> short side is a drag, switch to long-only and note the reason on the Leaderboard.

---

## Multi-Resolution Protocol

Different intervals have different performance characteristics and data requirements. Use the appropriate tier.

### Resolution Tiers

| Tier | Intervals | Data Source | Max Grid for Sweeps | Default Range |
|------|-----------|-------------|---------------------|---------------|
| Ultra-high | 1s, 5s | REST API (`fetch_binance_klines.py --interval 1s`) | 50 combos | 1 day |
| High | 10s, 30s | REST API (`fetch_binance_klines.py --interval 10s`) | 100 combos | 7 days |
| Standard | 1m, 5m, 15m | Existing CSV resample | 200 combos | Full dataset |
| Medium | 1h, 4h | Existing CSV resample | 500+ combos | Full dataset |
| Low | 1D | Existing CSV resample | Full grid | Full dataset |

### Data Acquisition for Sub-Minute

Before running sub-minute analysis, fetch the data:

```bash
# 1-second, last 24 hours
python3 fetch_binance_klines.py --interval 1s --days-back 1

# 10-second, last 7 days
python3 fetch_binance_klines.py --interval 10s --days-back 7

# Custom symbol + output
python3 fetch_binance_klines.py --interval 5s --symbol ETHUSDT --days-back 3 --output-dir data/
```

Then point the backtest at the generated file:
```bash
python3 run_backtest.py --strategy {name} --data data/binance_BTCUSDT_1s_klines.csv --sweep --top 10
```

### Sub-Minute Caveats

- **No resampling needed** — data is already at target resolution.
- **Smaller grid** — sub-minute data is large; keep sweeps lean.
- **Higher fee sensitivity** — more trades = more fees. Test fee sensitivity aggressively.
- **Shorter validation periods** — sub-period splits may use hours instead of years.
- **Sharpe inflation** — minute-bar Sharpe is inflated ~38x vs daily due to autocorrelation. Always note the timeframe when reporting Sharpe.

---

## Crypto Session Framework

Crypto trades 24/7, but volume and volatility are far from uniform. Session awareness improves both execution quality and signal reliability.

### Session Definitions

| Session | UTC Time | Volume (% of avg) | Character |
|---------|----------|-------------------|-----------|
| Late Asia / Pacific | 00:00-04:00 | 60-70% | Thinnest liquidity, flash-crash risk |
| Asia Core | 04:00-08:00 | 70-80% | Moderate, building |
| European Morning | 08:00-12:00 | 100-110% | First vol expansion, institutional flow |
| EU/US Overlap | 12:00-16:00 | 130-150% | **Peak liquidity, tightest spreads** |
| US Core | 16:00-20:00 | 110-130% | Strong volume, second-highest |
| US Evening / Pre-Asia | 20:00-00:00 | 60-80% | Declining, second flash-crash window |

### Intraday Patterns

- **Volatility peaks** at European open (~08:00 UTC) and US equity open (~14:30 UTC). The EU/US overlap (13:00-16:00 UTC) has the highest and most reliable volume.
- **Weekend volume** is 30-50% lower than weekdays. Spread per unit of volume is higher — flash crashes are disproportionately likely on weekends (especially Sunday 00:00-08:00 UTC).
- **Day-of-week:** Monday often flat/negative (CME gap dynamics), Wed-Thu slightly positive in some studies. Effects are small (2-8 bps) and regime-dependent — do not build strategies solely on calendar effects.
- **Funding rate timestamps:** Binance settles at 00:00, 08:00, 16:00 UTC. Position changes between timestamps avoid paying/receiving that period's funding.

### Session-Aware Strategy Guidelines

- **Breakout/momentum strategies:** Prefer EU/US overlap (12:00-16:00 UTC) for execution — highest liquidity, most reliable directional moves.
- **Mean-reversion strategies:** The 20:00-04:00 UTC window produces more false breakouts due to thin liquidity — prime territory for mean-reversion.
- **Adapted Opening Range Breakout:** Define the "overnight range" as the Asian session high/low (00:00-08:00 UTC). Trade breakouts during the European session. Filter: only trade if overnight range < 1.5x the 20-day average.
- **Execution timing:** When possible, execute signals during peak liquidity (12:00-16:00 UTC) even if the signal fires off-hours. Saves 1-3 bps in slippage.

---

## Intraday Volume Patterns & Session Microstructure

BTC exhibits a persistent W-shaped (double-humped) intraday volume profile that has evolved significantly since the 2024 spot ETF approval. Understanding these patterns enables session-aware filtering, volume-clock sampling, and execution optimization.

### Intraday Volume Profile

BTC 24h volume follows a W-shape with two peaks and two troughs:

| UTC Window | Session | % of Daily Volume | Relative Volume | Vol-per-Unit-Volume |
|---|---|---|---|---|
| 05:00-08:00 | Late Asia | 8-10% | 0.65-0.80x | 1.1-1.3x |
| 08:00-13:00 | European Morning | 22-26% | 1.10-1.30x | 0.8-1.0x |
| 13:00-16:00 | EU/US Overlap | 18-22% | 1.50-1.80x | 0.7-0.9x (lowest) |
| 16:00-20:00 | US Core | 16-20% | 1.00-1.25x | 0.9-1.1x |
| 20:00-00:00 | US Evening | 10-14% | 0.50-0.70x | 1.3-1.8x (highest) |
| 00:00-05:00 | Asia Core | 12-15% | 0.50-0.70x | 1.2-1.5x |

**Post-ETF shift (2024+):** US session peak is now dominant (ETF-related volume concentrated 14:30-21:00 UTC). Weekend volume dropped to 50-65% of weekday levels (was 70-80% pre-2021). The pattern is now closer to a skewed W with a dominant US peak.

**Day-of-week variations:** Tuesday and Wednesday are highest volume (108-115% of Monday baseline), driven by macro data releases (FOMC, CPI). Saturday/Sunday volume drops to 50-70% of weekday levels.

### Volume Clock (Dollar/Volume Bars)

Standard time bars oversample quiet periods and undersample active periods. Volume/dollar bars create new bars when a fixed volume/dollar amount transacts, producing returns closer to IID (Lopez de Prado, 2018).

**Statistical quality hierarchy (best to worst):** Dollar bars > Volume bars > Tick bars > Time bars.

**Advantages for BTC:**
- Eliminates noise-laden low-volume bars during 20:00-04:00 UTC trough
- Increases resolution during active EU/US overlap
- Reduces excess kurtosis by ~30-40% (Aste & Di Matteo, 2019)
- ML classifiers trained on dollar bars outperform time bars by 5-15% accuracy

**Recommended dollar bar sizes (Binance BTCUSDT perpetual, 2024-2025):**
- Scalping/HF: $1-5M per bar (~500-2000 bars/day)
- Intraday: $10-25M per bar (~100-300 bars/day)
- Swing: $50-100M per bar (~20-60 bars/day)

**Implementation caveat:** Indicators using time-based concepts (session VWAP, funding rate) must remain on a time axis. Use dual-axis approach: volume bars for signals, time bars for time-dependent calculations. Look-ahead bias: bar size must be calibrated from prior data, not full dataset.

### Volume Anomaly Detection

**Volume spike classification (vs 20-bar session-normalized average):**

| Multiplier | Classification | Expected Significance |
|---|---|---|
| 1.5-2.0x | Elevated | Position building; mild interest |
| 2.0-3.0x | Spike | Notable event; ~62-68% same-direction continuation within 4h |
| 3.0-5.0x | Major spike | High probability sustained move |
| >5.0x | Extreme | Often capitulation/stop cascade; ~55-60% reversal probability |

**Volume drought detection:** Volume below 50% of hourly average for 5+ consecutive hours precedes breakouts. First bar exceeding 1.5x average after drought leads to >1.5% move within 6h ~58-64% of the time. Direction requires a separate signal.

**Session-normalized Z-scores:** Compute 30-day rolling average and std for each hour-of-day separately. Z = (current - hourly_mean) / hourly_std. Z > 2.0 = significant spike; Z < -1.5 = significant drought. This eliminates systematic W-shape bias.

### Volume-Price Relationships

**Volume confirmation (1h BTC bars, 2022-2025):**
- Up bar + volume > 1.5x avg → ~56-60% continuation probability
- Up bar + volume < 0.7x avg → ~47-50% continuation (random)
- Down bar + volume > 2x avg → ~60-65% continuation (liquidation cascading)

**Volume-price divergence:** 3-swing bearish divergence (three higher highs with declining volume) → >2% correction within 24h ~60-65% of time. Bullish divergence less reliable (~55%).

**Volume exhaustion signal:** After 3+ consecutive bars with volume > 2x average, expect mean reversion. Probability of >0.5% retracement within next 3 bars: ~65-72% (uptrend), ~60-68% (downtrend). Combined with RSI extremes: ~70-78%.

**VPIN (Volume-Synchronized Probability of Informed Trading):** BTC background VPIN ~0.20-0.25. VPIN > 0.35 = heightened alert (~65-70% chance of significant move within 1-4h). Spiked to >0.50 before FTX collapse and >0.45 before ETF approval.

### Optimal Execution Windows

**Minimum slippage windows (Binance BTCUSDT perpetual, 2024-2025):**

| UTC Window | Avg Spread (bps) | Relative Slippage | Market Depth |
|---|---|---|---|
| 13:00-16:00 (EU/US Overlap) | 0.3-0.6 | 0.5-0.6x baseline (best) | Highest |
| 10:00-13:00 (EU Core) | 0.5-0.7 | 0.7x baseline | Very High |
| 16:00-20:00 (US Core) | 0.5-0.8 | 0.7x baseline | High |
| 20:00-23:00 (Post-US) | 1.0-2.0 | 1.5-2.5x baseline (worst) | Low |
| Weekend trough (Sat 20:00-Sun 04:00) | 2.0-3.0 | 3-5x baseline | Very Low |

Market impact scales as ~sqrt(order_size / ADV). Doubling order size increases impact by ~40%.

### Volume-Based Strategy Enhancements

**Volume confirmation filter:** Only take signals when bar volume > 1.5x session-normalized 20-bar average. Expected impact: -30-45% trade count, +3-7pp win rate, +0.1-0.3 Sharpe.

**Opening range volume:** If US open (14:00-15:00 UTC) volume > 12% of prior 24h total → trend day signal. Trade first-hour direction: ~58-63% win rate, 1.5-2x avg winner vs avg loser.

**Session restriction (highest impact, lowest complexity):** Restricting trend-following to 08:00-20:00 UTC weekdays can improve Sharpe by +0.2-0.5 by eliminating low-quality thin-market signals.

### Volume Pattern Implementation Priority

| Priority | Enhancement | Complexity | Expected Impact |
|---|---|---|---|
| 1 | Session-time restriction for trend strategies | Low | +0.2-0.5 Sharpe |
| 2 | Volume confirmation filter (1.5x threshold) | Low | +0.1-0.3 Sharpe |
| 3 | Weekend sizing reduction (50-60%) | Low | Drawdown reduction |
| 4 | Session-normalized volume Z-scores | Medium | Better anomaly detection |
| 5 | Volume exhaustion mean-reversion signal | Medium | New signal source |
| 6 | Dollar/volume bar resampling | High | Better statistical properties |
| 7 | VPIN computation | High | Informed trading detection |

---

## BTC Structural Patterns

BTC has unique structural features that create exploitable patterns. These are distinct from indicator-based signals — they arise from market mechanics, not price/volume calculations.

### Halving Cycle Effects

Bitcoin's supply issuance halves every ~210,000 blocks (~4 years). Historical price behavior by cycle phase:

| Phase | Timing (relative to halving) | Historical Return | Character |
|-------|-----|------|------|
| Pre-halving accumulation | -12 to -6 months | +50-100% | Anticipatory buying, supply squeeze narrative |
| Pre-halving rally | -6 to 0 months | +30-80% | Accelerating momentum, media coverage |
| Post-halving consolidation | 0 to +6 months | -10% to +30% | Choppy, miner capitulation, supply adjustment |
| Parabolic phase | +6 to +18 months | +200-500% | Strongest returns, narrative-driven mania |
| Distribution/bear | +18 to +36 months | -50% to -85% | Extended decline, capitulation |

**2024 halving (April 19, 2024):** BTC was already at ~$64K (pre-halving ATH). Post-halving consolidation was prolonged (~3 months, $49K-$73K range). Unique in having spot ETFs as structural demand.

**Integration as regime filter:** Compute `months_since_halving` from the known halving dates. During parabolic phase (+6 to +18 months), bias strategies toward trend-following and increase long sizing. During distribution/bear phase (+18 to +36 months), bias toward mean-reversion and reduce long exposure.

**Halving dates:** 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-19, ~2028-04 (estimated).

**Caveat:** Each cycle has shown diminishing returns. The +500% post-halving moves of 2012-2016 may not repeat as BTC matures. Sample size is 4 halvings — too small for statistical confidence.

### CME Gap Dynamics

BTC futures on CME close Friday 5:00 PM CT and reopen Sunday 5:00 PM CT. Price moves during the weekend create "gaps" at Monday's open.

**Historical statistics:**
- CME gaps > $100 are filled (price returns to pre-gap level) approximately 77% of the time
- Median fill time: 1-7 days for small gaps (<3%), weeks for large gaps (>5%)
- Large gaps (>5%) have a lower fill rate (~60%) and can take months

**Strategy concept:**
```
gap_size = monday_open - friday_close
IF gap_size > 0.5% of price:
    signal = SHORT (expect gap fill downward)
IF gap_size < -0.5% of price:
    signal = LONG (expect gap fill upward)
target = friday_close (gap fill level)
stop = monday_open + gap_direction * 1.5 * gap_size
max_hold = 168 bars (1 week on 1h data)
```

**Caveat:** This strategy has low trade frequency (~40-50 trades per year). The edge exists but may be arbitraged away as more participants trade it.

### Options Expiry Effects (Deribit)

**Schedule:** Last Friday of each month, 08:00 UTC. Quarterly expiries (Mar, Jun, Sep, Dec) are 3-5x larger.

**Pre-expiry (48h before):** Max pain gravitation — price gravitates toward the strike with highest open interest. Gamma pinning suppresses volatility. Consider reducing position size or going flat.

**Post-expiry (48h after):** Volatility expansion as gamma pinning releases. Good window for breakout strategies. Quarterly expiries produce 3-7% moves within 48 hours post-expiry.

**Integration:** Add expiry dates as a calendar column. Strategies reduce position sizing 48h pre-expiry and increase 48h post-expiry. Quarterly expiries deserve stronger adjustment than monthly.

### Weekend Effects

**Volume:** Weekend volume is 30-50% lower than weekdays and declining (27% drop 2022-2024 as institutions trade only weekdays).

**Flash crash risk:** Major weekend flash crashes occur 3-5 times per year. Examples: Aug 2024 ($49K, yen carry unwind), Feb 2026 ($60K, tariff panic), Dec 2025 ($24K flash wick on thin book).

**Actionable edge:** The edge is in **risk management**, not directional alpha. Reduce position size by 50% on weekends (Saturday 00:00 UTC to Monday 00:00 UTC) to avoid tail events with minimal opportunity cost.

### BTC-SPX Correlation Regimes

BTC-SPX correlation shifts between regimes:

| Regime | 60-day Correlation | When | Strategy Implication |
|---|---|---|---|
| Macro-driven | > 0.5 | Rate hikes, risk-off, post-ETF | Use SPX/VIX as leading indicators for BTC |
| Crypto-native | < 0.2 | Crypto-specific catalysts, early cycle | Focus on on-chain and crypto-specific signals |
| Transition | 0.2 - 0.5 | Regime shifts | Reduce position sizing, wait for clarity |

**Post-ETF (2024+):** Baseline correlation is structurally elevated (~0.5+) due to institutional crossover flows. The decorrelated "digital gold" narrative applies less frequently.

**Integration:** Fetch SPX daily close (Yahoo Finance, free). Compute 60-day rolling correlation. In macro-driven regime, overlay SPX momentum as additional filter. In crypto-native regime, rely solely on price/on-chain signals.

### DXY (Dollar Index) Impact

BTC shows inverse correlation with DXY ranging from -0.4 to -0.8 over 2022-2025:
- DXY breakdowns (below 20-day and 50-day SMA) have preceded BTC rallies by 1-4 weeks
- DXY at 20-year highs (2022, DXY=114) coincided with BTC's cycle low ($15,500)

**Integration:** DXY downtrend → allow full long exposure. DXY uptrend → reduce long exposure or allow shorts. Expected improvement: +0.1-0.2 Sharpe as regime filter. Data: Yahoo Finance ticker `DX-Y.NYB`.

**Caveat:** The relationship breaks during crypto-specific events (exchange failures, protocol upgrades, ETF launches).

### Futures Basis as Sentiment Indicator

```
basis_annualized = ((futures_price / spot_price) - 1) * (365 / days_to_expiry)
```

| Annualized Basis | Interpretation | Strategy Bias |
|---|---|---|
| > 20% | Euphoria / extreme contango | Reduce long exposure, tighten stops |
| 5-15% | Normal contango / healthy market | Standard operation |
| 0-5% | Low contango / cautious | Market weakening |
| < 0% | Backwardation / capitulation | Strong accumulation signal (5/5 at cycle bottoms) |

**Data:** CoinGlass (free), CME DataMine. Backwardation sustained for days/weeks is the signal — hourly backwardation is noise.

### Hash Ribbons (Miner Capitulation)

**Signal:** 30-day SMA of hash rate crosses below 60-day SMA = miner capitulation. **Buy signal:** when 30-day crosses back above 60-day (recovery).

**Track record:** 6/6 profitable on a 12-month horizon since 2018 (Dec 2018, Mar 2020, Jun 2021, Nov 2022, May 2024, Nov 2025). Average 12-month return after recovery signal: +200%+.

**Caveat:** Very infrequent (1-2 times per cycle). Drawdown from signal to actual bottom can be 10-30%. This is a macro positioning signal, not a timing tool. Data: blockchain.com (free).

### Structural Pattern Implementation Priority

| Priority | Pattern | External Data Needed | Expected Impact |
|---|---|---|---|
| 1 | Halving cycle position | None (hardcoded dates) | Regime filter, +0.1-0.3 Sharpe |
| 2 | Weekend regime filter | None (day-of-week) | Risk reduction, fewer tail events |
| 3 | Options expiry calendar | None (deterministic dates) | Reduced whipsaw pre-expiry |
| 4 | CME gap strategy | None (hour-of-week) | Standalone ~0.5-0.8 Sharpe, low frequency |
| 5 | DXY regime filter | Yahoo Finance (free) | +0.1-0.2 Sharpe as filter |
| 6 | BTC-SPX correlation | Yahoo Finance (free) | Regime-dependent strategy selection |
| 7 | Hash ribbons | blockchain.com (free) | Macro accumulation signal |
| 8 | Futures basis | CoinGlass (free) | Sentiment/cycle positioning |

---

## Multi-Timeframe Analysis (MTF) Framework

Higher timeframes set directional bias; lower timeframes provide entry timing. MTF filtering is one of the most documented ways to improve strategy Sharpe on BTC — the QuantPedia D1H1 study showed Sharpe improvement from 0.33 to 0.80 on BTC hourly MACD by adding a daily trend filter.

### Timeframe Ratios

Use a **4:1 to 6:1 ratio** between adjacent analysis timeframes. Ratios below 3:1 cause excessive signal overlap; above 8:1 miss intermediate structure.

| Ratio | Example Chain | Notes |
|-------|---------------|-------|
| 4:1 | 1H → 4H → Daily | Most popular; maps to standard candle intervals |
| 6:1 | 4H → Daily → Weekly | Wider separation; fewer but higher-conviction signals |

**The "1:4:16" principle:** If your trading TF is 1H, context TF is 4H (4x), trend TF is Daily/16H (16x).

### Practical BTC Timeframe Combinations

| Use Case | Trend TF | Setup TF | Entry TF | BTC Evidence |
|----------|----------|----------|----------|-------------|
| Swing trading (best documented) | Daily | 4H | 1H | Sharpe 0.33 → 0.80 with D1H1 filter (QuantPedia) |
| Position trading | Weekly | Daily | 4H | Captures BTC halving macro cycles |
| Active swing | 4H | 1H | 15m | More trades; needs tight spreads |
| Day trading | 1H | 15m | 5m | High frequency; less BTC-specific evidence |

### Hierarchical Signal Construction

```
Layer 1 (Trend TF — highest weight):
  Determine directional bias. Only trade in this direction.
  Example: Daily MACD histogram rising → bullish bias only

Layer 2 (Setup TF — moderate weight):
  Identify pullback/setup within the trend
  Example: 4H Stochastic dropping to oversold during daily uptrend

Layer 3 (Entry TF — execution):
  Time the entry precisely within the setup
  Example: 1H candle breakout above prior bar high
```

### Timeframe Alignment Scoring (Harmony Index)

Quantify alignment across 3-4 timeframes to set position sizing:

```
Per-timeframe bias: +1 (bullish), -1 (bearish), 0 (neutral)
  Based on: price vs key MA AND confirming indicator (RSI > 50, MACD histogram > 0)

Harmony Index (HI) = Sum(Bias_i * Weight_i) / Sum(Weight_i)
  Normalized to [-1.0, +1.0]
```

**Recommended weights:**

| Timeframe Role | Weight | Rationale |
|----------------|--------|-----------|
| Execution (1H) | 0.10 | Noisy, fast-changing |
| Tactical (4H) | 0.20 | Moderate conviction |
| Setup (Daily) | 0.30 | Strong structural signal |
| Trend (Weekly) | 0.40 | Highest conviction, slowest to change |

**Action thresholds:**

| HI Value | Action |
|----------|--------|
| +0.80 to +1.00 | Full position long; highest conviction |
| +0.40 to +0.79 | Reduced position long |
| -0.39 to +0.39 | No trade; stay flat (conflicting signals) |
| -0.79 to -0.40 | Reduced position short |
| -1.00 to -0.80 | Full position short; highest conviction |

**Evidence:** Trades aligned across 2+ timeframes show ~58% win rate vs 39% for non-aligned trades. The QuantPedia D1H1 filter cut max drawdown from -23.9% to -12.4% (48% reduction).

### Triple Screen Trading System (Elder) — BTC Adaptation

| Screen | Swing Trading (BTC) | Intraday (BTC) |
|--------|---------------------|----------------|
| First (Trend) | Weekly MACD histogram slope | Daily MACD histogram slope |
| Second (Oscillator) | Daily Stochastic %K/%D in oversold zone | 4H RSI(14) or Stochastic RSI |
| Third (Entry) | 4H candle breakout / trailing buy-stop | 1H candle breakout / limit at support |

**Rules:** First screen determines direction — only trade with it. Second screen identifies counter-trend pullbacks within the trend. Third screen provides entry timing. Ignore oscillator signals that oppose the first screen.

### BTC 60-Day Intermediate Cycle

BTC exhibits a roughly 60-day intermediate cycle (~24 cycles per 4-year macro cycle):
- Identify cycle lows: daily RSI(14) below 30 while weekly trend is still up
- Enter at cycle low; target the cycle high (~30 trading days later)
- Exit when daily RSI exceeds 70

### Implementing MTF in a Single-Timeframe Backtester

Since `generate_signals(df)` receives a single-resolution DataFrame, compute higher-TF indicators by resampling within the strategy:

1. Receive `df` at base resolution (e.g., 1H)
2. Resample: `df_daily = df.resample('1D', label='right', closed='right').agg({...}).dropna()`
3. Compute higher-TF indicator on `df_daily`
4. Forward-fill back: `htf_signal = htf_indicator.reindex(df.index, method='ffill').shift(1)`
5. Combine: `final_signal = ltf_signal.where(htf_signal == expected_direction, 0)`

**Critical — the `.shift(1)` after forward-fill prevents look-ahead bias.** Without it, the daily indicator value appears at the start of the day before the day's price action has occurred. Estimated inflation from this bug: +0.3 to +1.0 Sharpe.

### MTF Resampling Rules

| Base TF | Higher TF | Resample Rule | Shift |
|---------|-----------|---------------|-------|
| 1H | 4H | `'4h', label='right', closed='right'` | `.shift(1)` on 1H index |
| 1H | 1D | `'1D', label='right', closed='right'` | `.shift(1)` on 1H index |
| 4H | 1D | `'1D', label='right', closed='right'` | `.shift(1)` on 4H index |
| 4H | 1W | `'W', label='right', closed='right'` | `.shift(1)` on 4H index |

**Always use `label='right', closed='right'` for MTF resampling.** This assigns the aggregated bar to the *end* of the period (when the data is actually available). The default `label='left'` creates systematic look-ahead bias.

### MTF Pitfalls

1. **Over-filtering:** Each TF filter reduces trades by 40-60%. With 3+ filters, you may drop below 100 trades. Monitor trade reduction ratio — if a filter eliminates >70% of trades, it's too restrictive.
2. **Conflict paralysis:** When TFs disagree, strategy goes flat for months. Define explicit conflict rules: go flat, or follow higher TF at reduced size.
3. **Indicator mismatch:** RSI-14 on weekly (14 weeks) ≠ RSI-14 on hourly (14 hours). Scale parameters by calendar time, or use different indicators at different TFs (Elder approach).
4. **Diminishing returns:** Adding a 3rd or 4th TF filter rarely improves Sharpe by more than +0.1. Two timeframes (one trend + one execution) is the sweet spot.

### MTF Implementation Priority

| Priority | Combination | Expected Sharpe Impact | Difficulty |
|----------|-------------|----------------------|------------|
| 1 | Daily trend filter + 1H signals | +0.3 to +0.7 | Low (resample + filter) |
| 2 | Weekly trend + Daily entry | +0.2 to +0.5 | Low |
| 3 | Harmony Index (3-4 TFs) | +0.3 to +0.8 | Medium (scoring system) |
| 4 | Triple Screen full implementation | +0.4 to +0.8 | Medium (3 indicators) |

---

## Regime Detection Protocol

Market regime awareness prevents deploying trend-following strategies in ranging markets and vice versa. Use regime labels to evaluate and filter strategies.

### Regime Detection Methods

Use at least one method. Method 1 is mandatory; Methods 2-3 are recommended for deeper analysis.

**Method 1: Rolling Volatility Percentile (Mandatory)**

```python
# Simple, deterministic, no model risk
rolling_vol = returns.rolling(20).std() * np.sqrt(365)
vol_percentile = rolling_vol.rolling(252).rank(pct=True)

# Regime labels:
# Low vol:  percentile < 0.33
# Med vol:  0.33 <= percentile <= 0.67
# High vol: percentile > 0.67
```

**Method 2: Trend Strength (ADX / Efficiency Ratio)**

```python
# ADX-based:
# ADX > 25 -> trending regime (favor trend-following)
# ADX < 20 -> ranging regime (favor mean-reversion)
# 20-25 -> transition zone (reduce position sizes)

# Efficiency Ratio (Kaufman):
# ER = |price_t - price_{t-n}| / sum(|price_i - price_{i-1}|)
# ER close to 1.0 -> strong trend
# ER close to 0.0 -> choppy, mean-reverting
```

**Method 3: Hurst Exponent (Advanced)**

```python
# H > 0.5 -> persistent/trending -> trend-following
# H < 0.5 -> anti-persistent/mean-reverting -> mean-reversion
# H ≈ 0.5 -> random walk -> no directional edge
# Compute over rolling 60-100 day windows
```

**BTC Hurst exponent by timeframe (empirical averages):**

| Timeframe | Typical H | Implication |
|-----------|----------|-------------|
| 1m-5m | ~0.50 | Near random walk — minimal directional edge at micro scale |
| 1h | 0.52-0.55 | Slight persistence, near-efficient — mean-reversion edges are small |
| 4h | 0.55-0.60 | Mild persistence — both trend and reversion strategies may work |
| 1D | ~0.64 | Moderate persistence — trend-following has documented edge |
| 1W | 0.60-0.70 | Persistent — strongest trend-following regime |

**Implication for strategy selection:** Persistence increases with timeframe. Sub-hourly strategies should not expect large directional edges; they rely on microstructure or execution advantages. Daily+ strategies benefit most from trend-following. The hourly timeframe (H ≈ 0.52-0.55) is near the mean-reversion/trend boundary — both strategy types can work, making it the most versatile analysis frequency. Note: H is time-varying and moves toward 0.5 as markets mature (increased efficiency).

### BTC Drawdown Statistics (Historical Context)

Use these empirical baselines when assessing whether a strategy's drawdown is reasonable or anomalous:

| Event Type | Avg Drawdown | Avg Recovery Time | Frequency |
|-----------|-------------|-------------------|-----------|
| Bull market pullback | 27% (median 27%) | Weeks to months | Every 2-3 months |
| Moderate correction | 30-49% | 2-6 months | 1-2 per year |
| Bear market | 70-85% | 2-3 years | Every 3-4 years (cycle-aligned) |
| Flash crash (intraday) | 10-30% | Hours to days | 2-4 per year |

**Practical implications:**
- A strategy max drawdown of 20% is within normal BTC bull-market behavior. Anything under 30% during a ranging/bear year is acceptable.
- Monte Carlo stress-test drawdowns > 50% suggest the strategy may not survive a full bear cycle.
- Average bull-market rally following a 20%+ pullback is ~91% (median 75%). Strategies that exit during pullbacks and re-enter on recovery confirmation can exploit this asymmetry.
- 20%+ BTC drops occur roughly every 2 months even in bull markets. Circuit breaker thresholds must account for this — a -20% trigger would fire constantly.

### Regime-Conditional Evaluation

After generating regime labels, segment strategy performance by regime:

```python
# For each regime in [low_vol, med_vol, high_vol] (or [trending, ranging]):
#   Compute: Sharpe, win rate, avg P&L, max drawdown, trade count
#   A robust strategy should be profitable in at least 2 of 3 regimes
#   Flag if strategy has negative Sharpe in any regime occurring > 20% of the time
```

**Integration with Phase 5:** Report per-regime metrics alongside the existing robustness tests. A strategy that only works in one regime is fragile — note this on the Leaderboard.

---

## On-Chain Metrics as Supplementary Signals

On-chain data provides information about BTC holder behavior that is invisible in price/volume data. These metrics operate at longer timescales (daily to monthly) and are best used as regime filters and position sizing overlays, not intraday signals.

### Signal Hierarchy (Fastest to Slowest)

```
Layer 1 - CYCLE POSITION (monthly):
  MVRV Z-Score + NVT Signal → overall exposure budget (0% to 150%)

Layer 2 - REGIME (weekly):
  SOPR 30d + Exchange Netflow 14d → bull/bear classification, directional filter

Layer 3 - LEVERAGE CONTEXT (daily):
  OI/MCap ratio + Funding Rate Z-score → risk adjustment, fragile market detection

Layer 4 - ENTRY TIMING (intraday to daily):
  Existing RSI, EMA, MACD, Bollinger strategies → filtered and sized by Layers 1-3

Layer 5 - CONFIRMATION (daily):
  Active Addresses trend + Hash Rate momentum → signal quality adjustment
```

### Key Metrics

**Exchange Netflow (accumulation/distribution):**
```
net_flow = exchange_inflow - exchange_outflow
signal: 30-day SMA of net_flow
  Positive (coins entering exchanges) → distribution / selling pressure
  Negative (coins leaving exchanges) → accumulation / bullish
Threshold: flow > 10,000 BTC in a day = "large" — historically precedes 3-7% moves
```

**MVRV Z-Score (cycle positioning):**
```
MVRV = market_cap / realized_cap
MVRV_Z = (market_cap - realized_cap) / std(market_cap)

Historical cycle levels:
  Z < 0:    Deep undervaluation (cycle bottom zone) — 2015, 2019, 2022
  Z 0-2:    Accumulation / early cycle
  Z 2-5:    Mid-cycle
  Z 5-7:    Late cycle / elevated — consider reducing exposure
  Z > 7:    Extreme overvaluation — historically within weeks of cycle tops
```
- Each successive cycle has shown lower Z peaks (10 → 8 → 7). The Z > 7 threshold may never be reached again.
- Daily resolution, available free from CoinMetrics community API.

**SOPR (Spent Output Profit Ratio):**
```
SOPR = USD value at spend / USD value at creation (aggregate over all UTXOs)

Key behavior:
  Bull market: SOPR = 1.0 acts as SUPPORT (holders refuse to sell at loss)
  Bear market: SOPR = 1.0 acts as RESISTANCE (holders sell at breakeven)
Smoothing: 7-day SMA for swing trading, 30-day SMA for regime identification
```
- The 1.0 support/resistance flip has held across every major BTC cycle.
- UTXO-only metric — works for BTC/LTC, not ETH.

**Open Interest Dynamics:**
```
OI/MCap ratio:
  3-4%:  Healthy / normalized
  5-7%:  Elevated leverage, increasing fragility
  8-10%: Dangerous, high cascade probability
  > 10%: Extreme, deleveraging event highly likely

OI-Price divergence:
  Price up + OI down → longs exiting, rally weakening (bearish)
  Price down + OI up → shorts loading, squeeze potential (bullish)
```

### Data Availability

| Metric | Free Source | Frequency | Lag |
|---|---|---|---|
| Exchange Netflow | CryptoQuant | Daily | ~6h (entity attribution) |
| MVRV Z-Score | CoinMetrics community API | Daily | ~24h |
| SOPR | CryptoQuant, Glassnode | Daily | ~24h |
| NVT Signal | CoinMetrics, Woobull | Daily | ~24h (90d MA makes lag moot) |
| Funding Rate | CoinGlass (free) | 8-hourly | Near real-time |
| Open Interest | CoinGlass, Coinalyze | Near real-time | Near real-time |
| Active Addresses | CoinMetrics, blockchain.com | Daily | ~24h |
| Hash Rate | blockchain.com | Daily | ~24h |

### Look-Ahead Bias with On-Chain Data

**Critical:** In backtesting, always shift on-chain signals forward by at least 1 day:
```python
signal = on_chain_metric.shift(1)  # use yesterday's value for today's decision
```
On-chain metrics are end-of-day aggregates. Using same-day values is look-ahead bias.

### Integration with Existing Framework

On-chain data should be:
1. Fetched separately and saved to CSV (e.g., `data/onchain_daily.csv`)
2. Merged into the OHLCV DataFrame by date before signal generation
3. Strategies check for column presence and degrade gracefully if missing

**Implementation priority:** Funding Rate + OI (highest — real-time, free, directly actionable) → MVRV Z-Score (high — excellent cycle positioning) → SOPR (medium) → Exchange Netflow (medium) → NVT (lower) → Active Addresses/Hash Rate (lowest as standalone).

---

## Sentiment & Alternative Data Integration

Sentiment data provides information about market psychology invisible in price/volume data. The primary value is **risk management and position sizing**, not standalone alpha generation. Realistic out-of-sample Sharpe for sentiment-enhanced BTC strategies: 0.9-1.3 (vs 0.8-1.0 for raw buy-and-hold).

### Fear & Greed Index (Primary Sentiment Signal)

**Source:** Alternative.me API (completely free, no API key, daily since Feb 2018)
**API:** `GET https://api.alternative.me/fng/?limit=0` — returns all historical data as JSON

**Construction:** Composite of volatility (25%), market momentum/volume (25%), social media (15%), BTC dominance (10%), Google Trends (10%). Surveys component (15%) has been paused.

**Signal interpretation (contrarian):**

| Reading | Classification | Action |
|---------|---------------|--------|
| 0-10 | Extreme Fear | Strong contrarian buy zone. Best historical forward returns. |
| 11-25 | Fear | Moderate buy zone. Accumulation territory. |
| 26-45 | Moderate Fear | Neutral-to-mild buy bias |
| 46-55 | Neutral | No signal |
| 56-75 | Greed | Caution. Begin reducing position size or tightening stops. |
| 76-90 | Extreme Greed | Contrarian sell/reduce. Historically precedes corrections. |
| 91-100 | Peak Greed | Strongest sell signal |

**Historical evidence:**
- Contrarian scaling strategy (buy 1% at FGI ≤ 20, sell 1% at FGI ≥ 80): 1,145% ROI vs 1,046% B&H over Feb 2018-2024
- FGI filter improved algorithmic strategy Sharpe from 1.33 to 1.52 (SAGE Journals, 2025)
- March 2020 (FGI ~5): subsequent 12-month return >1,500%
- Nov 2021 (FGI ~95): preceded -77% drawdown

**Critical:** FGI works via **gradual scaling over cycles**, not binary signals. Acting on a single extreme day is unreliable. Conditions can remain fearful/greedy for weeks.

### Google Trends ("Bitcoin" Search Interest)

**Source:** `pytrends` Python library (free, unofficial scraper). Weekly resolution for periods >5 months.

**Correlation with price:** ~82% globally, but the relationship is primarily **contemporaneous or lagging** — retail searches follow price, not vice versa.

**Use as:** Rate-of-change contrarian/confirmation signal. Search spikes >3x 90-day average often coincide with market exhaustion.

**CRITICAL LOOK-AHEAD BIAS WARNING:** Google Trends values are **retroactively renormalized**. The entire historical series changes every time you query. A value of 50 from 2017 may become 10 when re-queried in 2025 because the 2021 peak was higher.

**Safe backtesting approach:**
- Use **rate-of-change within rolling windows** (z-scores), never absolute levels
- Apply minimum +7 day lag (weekly data) plus +1 day publication delay
- Never download the full historical series at a single point in time

### Social Media Sentiment

**Key finding:** Tweet **volume** (mention counts) is more predictive than tweet **polarity** (positive/negative) for BTC. Engagement-weighted signals outperform raw polarity.

**Platforms:** LunarCrush ($24-240/mo), Santiment (free tier + paid), CryptoPanic (free API)

**Practical thresholds:**
- Social volume spike >2-3σ above 30-day average: potential contrarian signal
- Cross-platform alignment of extreme sentiment: strongest but rarest signal
- Sentiment polarity ratio >0.7 bullish or bearish sustained for 3+ days: moderate signal

**Signal-to-noise challenges:** High bot contamination on crypto Twitter. Social sentiment is more predictive for altcoins than for BTC. Most retail activity is reactive (follows price).

### News Sentiment

**NLP tools:** CryptoBERT (free, HuggingFace, F1 ~0.86), VADER (free, rule-based), FinBERT (free, finance-specific)
**Data sources:** CryptoPanic (free tier, community-voted sentiment), NewsAPI (100 req/day free, 1-month history)

**Assessment:** News sentiment has **weak standalone alpha** for BTC. Negative news impact is much stronger than positive. Predictive power decays within hours to 1-2 days. Primary value: suppress trades during extreme negative news regimes.

### Composite Sentiment Architecture

**Tier 1 (Primary — free, longest history):**
- Fear & Greed Index: daily, contrarian scaling
- On-chain metrics (MVRV, SOPR): from the On-Chain Metrics section

**Tier 2 (Confirmation — free with careful implementation):**
- Google Trends rate-of-change (z-scored, rolling window, NOT absolute levels)
- Social media volume spikes (>2σ, cross-platform corroborated)

**Tier 3 (Filter only — lower standalone reliability):**
- News sentiment (NLP-scored), Twitter polarity, individual influencer sentiment

**Signal hierarchy:** Tier 1 sets direction; Tier 2 confirms; Tier 3 filters. Only trade when Tier 1 + at least one Tier 2 align. Tier 3 disagreement reduces position size but does not veto.

### Integration with Backtesting

**Data lag requirements:**

| Source | Minimum Shift | Recommended Shift | Rationale |
|--------|--------------|-------------------|-----------|
| Fear & Greed Index | +1 day | +1 day | Published daily; reflects prior day's data |
| Google Trends | +7 days | +8-14 days | Weekly resolution + retroactive renormalization |
| Social media volume | +1 day | +1-2 days | Aggregation + publication delay |
| News sentiment | +4-24 hours | +1 day | Real-time news but NLP processing delay |

**Implementation priority:**
1. Fear & Greed Index — free, daily, longest history, simplest
2. Google Trends rate-of-change — free but requires careful rolling-window implementation
3. CryptoPanic news sentiment — free tier, community-voted, no NLP pipeline needed
4. Composite score — combine after validating each individually
5. Social media (LunarCrush/Santiment) — paid; add only after free sources validated

**Expected improvement:** +0.1-0.3 Sharpe as overlay on technical strategies. 10-30% max drawdown reduction via risk regime filtering. Primary value is risk management, not alpha generation.

---

## Funding Rate Carry Strategies

Perpetual futures funding rates represent one of the most documented edges in crypto — historical Sharpe ratios of 3.0-6.0+ for delta-neutral carry. The BIS estimates average crypto carry at ~7-8% per year, frequently exceeding 20% during bull phases.

### Funding Rate Mechanics

Exchanges impose funding payments every 8 hours (00:00, 08:00, 16:00 UTC) to anchor perp price to spot:
- **Positive funding** (perp > spot): Longs pay shorts
- **Negative funding** (perp < spot): Shorts pay longs

```
Funding Rate = Premium Index + clamp(Interest Rate - Premium Index, -0.05%, +0.05%)
Default rate: 0.01% per 8h on Binance (~10.95% annualized)
```

### Historical BTC Funding Rate Statistics

| Metric | Value |
|--------|-------|
| Mean (per 8h) | ~+0.010% to +0.015% |
| % Time Positive | ~70-75% |
| % Time Negative | ~25-30% |
| Annualized Mean Carry (short bias) | ~7-8% (BIS estimate) |
| Distribution | Positively skewed, leptokurtic |

**Year-by-year:**

| Year | Regime | Avg Rate (per 8h) | Annualized Carry |
|------|--------|-------------------|-----------------|
| 2020 | Recovery | +0.01% to +0.02% | ~11-22% |
| 2021 | Parabolic Bull | +0.03% to +0.06% | ~33-66% |
| 2022 | Bear Market | -0.005% to +0.005% | ~-5% to +5% |
| 2023 | Recovery | +0.01% to +0.02% | ~11-22% |
| 2024 | Bull (ETF) | +0.02% to +0.04% | ~22-44% |
| 2025 | Mixed | Variable | Declining post-ETF efficiency |

**Post-ETF impact (Jan 2024+):** Carry decreased by ~3 percentage points across all exchanges, reflecting improved arbitrage efficiency from institutional participants.

### Strategy Variants

**1. Pure Carry (Delta-Neutral Cash-and-Carry):**
Buy BTC spot + short BTC perpetual in equal notional. Collect positive funding.
- Expected return: 7-16% annualized
- Sharpe: 3.0-6.0+ (academic studies)
- Max drawdown: <2% (SSRN)
- Capital requirement: 2x notional (or ~1.33x with 3x leverage on perp)
- **Risk:** At 10x leverage, liquidation in >50% of months (BIS). Use 2-3x max.

**2. Contrarian Z-Score (Directional):**
Fade extreme funding rates without hedge. Fully directional.

**3. Funding Rate as Filter (Overlay on existing strategies):**
Use funding regime to suppress/allow signals from other strategies. Easiest to implement.

### Z-Score Approach

```
Z = (Current_Rate - Rolling_Mean) / Rolling_StdDev
Lookback: 60 days (180 observations at 3x/day) is the recommended default
```

| Z Threshold | Frequency | Use |
|-------------|-----------|-----|
| |Z| > 1.0 | ~30% of time | Filter only |
| |Z| > 1.5 | ~13% of time | Moderate contrarian signal |
| |Z| > 2.0 | ~5% of time | High confidence contrarian (primary threshold) |
| |Z| > 2.5 | ~1-2% of time | Very high confidence but rare |

**Signal logic:**
- Z > +2.0 → short / fade longs (longs overcrowded, funding will mean-revert)
- Z < -2.0 → long / fade shorts (shorts overcrowded, capitulation likely near)
- |Z| < 1.0 → no action

**Refinements:** Add 24-48h cooldown after signal fires. Require Z-score to cross back through ±1.0 before re-triggering. Use asymmetric thresholds (Z > +2.0 for shorts, Z < -1.5 for longs — negative funding periods are shorter-lived).

**Historical win rates:** Contrarian trades at Z > 2.0 on 60-day lookback produce favorable outcomes in next 24-72 hours approximately 55-65% of the time. The Z-score works better as a **regime filter** than a standalone signal (funding alone has ~0% R-squared for next-period prediction — Presto Labs).

### Risk Factors

| Risk | Severity | Mitigation |
|------|----------|------------|
| Funding regime reversal | High | Monitor 30-day rolling; unwind if negative for >7 days |
| Liquidation during divergence | Critical | Keep leverage ≤ 3x; maintain 20%+ excess margin |
| Exchange counterparty failure | Catastrophic | Split across 2-3 exchanges; max 40% per venue |
| Extended negative funding | High | Jun-Dec 2022: ~6 months of negative; spot leg loses simultaneously |

### Data Sources

| Source | Access | Coverage | Notes |
|--------|--------|----------|-------|
| Binance API | Free (no key needed) | 2019-present, per 8h | `GET /fapi/v1/fundingRate`, limit 1000/request |
| CoinGlass | Free charts, paid API | Multi-exchange | Best for cross-exchange comparison |
| CoinAPI | Paid | Multi-exchange | Good for normalization |

### Backtesting Integration

1. Fetch historical funding rates to CSV (e.g., `data/funding_rate_btc.csv`)
2. Align to 8h bars centered on 00:00, 08:00, 16:00 UTC
3. Apply funding as **discrete cost/credit** at each settlement — do NOT interpolate between settlements
4. For Z-score signals: shift by 1 period (signal at 08:00 → execute at 16:00)
5. For basis trade: track two separate equity legs (spot + perp + funding credits)

**Look-ahead note:** Exchanges publish the *next* funding rate ~8h in advance. For backtesting, use only **realized/settled** rates unless specifically modeling the prediction window.

### Implementation Priority

| Priority | Strategy | Effort | Expected Sharpe |
|----------|----------|--------|----------------|
| 1 | Z-score filter on existing strategies | Low | +0.1-0.3 improvement |
| 2 | Contrarian Z-score (standalone) | Low-Medium | 0.5-1.5 (regime-dependent) |
| 3 | Cash-and-carry backtest | Medium | 3.0-6.0 |
| 4 | Cross-exchange arb | High | 1.5-3.5 |

---

## Cross-Asset Correlation Signals

BTC's correlation with traditional assets has increased structurally since the spot ETF launch (Jan 2024). Cross-asset signals are most useful as **regime filters and position sizing overlays** — they reduce drawdowns more than they boost returns. Expected Sharpe improvement: +0.15-0.30 out-of-sample.

### BTC-DXY (Dollar Index) — Primary Cross-Asset Signal

The strongest and most documented cross-asset relationship for BTC:

| Period | 90-Day Rolling Correlation | Notes |
|--------|---------------------------|-------|
| Q2-Q4 2022 (DXY 100→114) | -0.50 to -0.70 | BTC fell $47K→$16K as dollar surged |
| Q4 2022-Q1 2023 (DXY 114→101) | -0.40 to -0.60 | BTC rallied $16K→$28K |
| H2 2023 (range-bound DXY) | -0.10 to -0.30 | Crypto-specific catalysts dominated |
| Post-ETF (2024) | -0.30 to -0.50 | Institutional flows reconnected macro link |
| **Full sample average** | **-0.30 to -0.40** | Consistently inverse |

**DXY as regime filter (recommended):**
- DXY below 50-day SMA and declining → bullish for BTC, allow full long exposure
- DXY above 50-day SMA and rising → bearish for BTC, reduce long exposure or allow shorts
- DXY within 1% of 50-day SMA → neutral, rely on crypto-specific signals

**Lead time:** DXY trend reversals have led BTC rallies by approximately 2-6 weeks (March 2020, September 2022, October 2023).

**Data:** Yahoo Finance ticker `DX-Y.NYB`, daily. Free via `yfinance`. Apply T+1 lag in backtests (DXY closes 5 PM ET, before BTC midnight UTC close, but T+1 is the safe conservative approach).

### BTC-SPX/Nasdaq Correlation

Post-ETF, BTC-Nasdaq correlation is structurally elevated:

| Period | BTC-Nasdaq 90d Corr | BTC-SPX 90d Corr |
|--------|---------------------|-------------------|
| Pre-COVID (2019) | 0.00 to 0.15 | 0.00 to 0.10 |
| 2020-2021 | 0.30 to 0.50 | 0.25 to 0.45 |
| 2022 bear market | 0.50 to 0.70 | 0.45 to 0.65 |
| Post-ETF (2024-2025) | 0.40 to 0.60 | 0.35 to 0.55 |

**Practical use:** Nasdaq in downtrend (below 20-day SMA) is a warning for BTC longs. Nasdaq drop >2% in a single day is followed by BTC weakness within 24-48h in ~65-70% of cases.

**Data:** Yahoo Finance `QQQ` (Nasdaq 100 ETF) or `^NDX`. Apply T+1 lag.

### VIX as Risk-Off Indicator

| VIX Level | BTC Regime | Position Sizing |
|-----------|-----------|-----------------|
| < 15 | Risk-on, favorable | 100% (full size) |
| 15-25 | Normal | 100% |
| 25-35 | Elevated fear | 50-70% sizing |
| > 35 | Panic | 20-30% sizing, OR wait for VIX mean-reversion buy signal |

**VIX mean-reversion buy signal:** When VIX spikes >35 then begins declining (VIX 5-day SMA crosses below 20-day SMA), this is historically a strong BTC recovery signal. Example: Aug 2024, VIX >60 → BTC dropped to $49K → recovered to $60K+ within weeks.

**Implementation:** Use VIX as position-sizing multiplier: `size_mult = max(0.2, 1.0 - max(0, VIX - 25) / 25)`.

**Data:** Yahoo Finance `^VIX`. T+1 lag.

### BTC-ETH Ratio as Crypto Risk Indicator

| ETH/BTC Signal | Threshold | Interpretation |
|----------------|-----------|----------------|
| ETH/BTC 14d ROC > +15% | Extreme | Late-cycle euphoria; caution for BTC longs |
| ETH/BTC 14d ROC +0% to +10% | Healthy | Risk-on, favorable for crypto |
| ETH/BTC 14d ROC < -5% | Stress | Risk-off; capital consolidating into BTC |
| ETH/BTC 14d ROC < -10% | Extreme stress | Preceded BTC drawdown >10% in ~60% of cases |
| 30d BTC-ETH correlation < 0.70 | Decorrelation | ETH-specific event; crypto regime shift |

**Historical range:** ETH/BTC 0.016 (early 2020) to 0.088 (late 2021). Common range since 2021: 0.03-0.06.

**Data:** Binance `ETHBTC` or compute from `ETHUSDT / BTCUSDT`. Same-day signals are valid (both crypto, same close time).

### BTC-Gold Correlation

**Assessment:** Weakest cross-asset signal. Full-sample correlation ~0.05-0.15. Occasionally rises to 0.30-0.35 during "store of value" narratives (Q1 2024, March 2023). Gold is better as a portfolio diversifier than as a BTC predictor.

**Use only as confirmation:** When gold and BTC are both rising with 30-day correlation >0.30, it confirms a "hard asset" regime that supports BTC. Otherwise, ignore gold for BTC signal generation.

**Data:** Yahoo Finance `GLD` (SPDR Gold ETF). T+1 lag.

### Composite Cross-Asset Regime Score (CARS)

Combine all cross-asset signals into a single regime score:

```
Component scores: each -1 (bearish) to +1 (bullish)

CARS = 0.30 * DXY_score +      # DXY trend (primary)
       0.25 * Nasdaq_score +    # Equity trend
       0.20 * VIX_score +       # Volatility regime
       0.15 * ETH_BTC_score +   # Crypto risk appetite
       0.10 * Gold_score        # Hard asset confirmation

CARS > +0.3:   Bullish regime → full position sizing
CARS -0.3 to +0.3: Neutral → reduced sizing (50-70%)
CARS < -0.3:   Bearish regime → minimal/no long exposure
```

**Expected improvement:** +0.3-0.6 Sharpe over buy-and-hold in backtests. After realistic OOS degradation (30-50%), expect +0.15-0.30 live. Primary value is drawdown reduction (30-50% less max DD).

### Look-Ahead Bias with Cross-Asset Data

**Critical:** TradFi markets close before BTC's daily close. The safe approach:

| Data Source | Close Time | BTC Backtest Lag |
|-------------|-----------|-----------------|
| DXY | 5:00 PM ET | T+1 (use yesterday's value) |
| SPX/Nasdaq | 4:00 PM ET | T+1 |
| VIX | 4:15 PM ET | T+1 |
| Gold (GLD) | 4:00 PM ET | T+1 |
| ETH | Same as BTC | T+0 (same close time) |
| FRED macro data | Variable publication lag | T+2 |

**Weekend handling:** Forward-fill Friday's TradFi close for Saturday and Sunday. Never interpolate.

### Cross-Asset Implementation Priority

| Priority | Signal | Expected Sharpe Impact | Difficulty |
|----------|--------|----------------------|------------|
| 1 | DXY trend filter | +0.2-0.4 | Low (single data source) |
| 2 | VIX position sizing | +0.1-0.2 (drawdown reduction) | Low |
| 3 | Nasdaq trend filter | +0.1-0.2 | Low |
| 4 | ETH/BTC ratio momentum | +0.05-0.15 | Low (already have Binance data) |
| 5 | Composite CARS | +0.15-0.30 (after all components) | Medium |
| 6 | Gold confirmation | +0.02-0.05 | Low but marginal value |

---

## Strategy Derivation Engine

When a strategy is exhausted (optimization complete, robustness tested), the loop derives the next strategy to test. This section defines the 7 derivation methods in priority order.

### Strategy Family Taxonomy

The derivation engine should explore across these families systematically, not just variations within one family:

| Family | Core Signal | BTC Edge | Examples |
|--------|-------------|----------|----------|
| **Trend-following** | Price direction persistence | Strong in halving cycles, narrative-driven trends | EMA crossover, MACD, Donchian breakout |
| **Mean-reversion** | Price deviation from mean | Strong in ranging/consolidation phases | RSI, Bollinger bounce, z-score reversion |
| **Breakout** | Volatility expansion from compression | Very strong — BTC consolidation-to-explosion pattern | Bollinger squeeze, ATR expansion, Donchian |
| **Momentum factor** | Multi-lookback return strength | Documented BTC momentum premium at 30-day horizon | Time-series momentum, cross-sectional momentum |
| **Volatility** | Vol compression/expansion cycles | BTC vol clustering is among the strongest of any asset | VIX-like strategies, vol selling, squeeze detection |
| **Carry** | Yield from holding/funding | Perpetual futures funding rate has shown Sharpe 1.0-1.5 | Funding rate, basis trade |
| **Order flow** | Aggressor imbalance / volume microstructure | Taker buy data available from Binance; CVD divergence is actionable | CVD divergence, VWAP reversion, taker imbalance |
| **Event-driven** | Calendar/structural patterns | BTC has unique structural events with documented effects | CME gap fill, options expiry, liquidation cascade bounce |
| **Multi-timeframe** | Higher-TF trend filters lower-TF entries | D1H1 filter doubled BTC Sharpe from 0.33 to 0.80 (QuantPedia) | Triple Screen, Harmony Index, MTF trend filter |
| **Sentiment-driven** | Fear/greed extremes, social volume | FGI contrarian has documented edge; best as overlay | FGI contrarian scaling, social volume spike fade |
| **Funding rate** | Perp funding rate extremes/carry | 7-8% annualized carry (BIS); Z-score contrarian Sharpe 0.5-1.5 | Z-score contrarian, cash-and-carry, funding filter |
| **Cross-asset** | DXY/equity/VIX regime filtering | DXY inverse correlation -0.3 to -0.7; post-ETF equity correlation 0.4-0.6 | DXY trend filter, VIX sizing, CARS composite |
| **Volume-microstructure** | Intraday volume anomalies, session patterns | W-shaped volume profile; session restriction alone +0.2-0.5 Sharpe | Volume confirmation filter, exhaustion fade, opening range momentum, dollar bars |
| **Drawdown-managed** | Meta-family: any strategy + dynamic risk management | 25-40% max DD reduction; layered architecture (vol target + anti-martingale + DD reduction) | Drawdown-managed trend, equity curve trading, risk-budgeted sizing |
| **Portfolio-construction** | Multi-strategy allocation, risk parity, exposure management | +1.0-2.5% net Sharpe improvement from regime-conditional allocation; correlation breakdown protection | Risk parity ensemble, regime-parity portfolio, WFER-decay adaptive |
| **Validation-driven** | CPCV, walk-forward, cost-adjusted strategies selected by robustness metrics | Higher OOS retention; WFER > 0.50 filter eliminates ~60% of false positives | CPCV-validated momentum, walk-forward ensemble, cost-aware trend |

**Rule:** Before deriving a new strategy via Methods 1-4, check if other families in the taxonomy have been explored. If 3+ consecutive strategies are from the same family, force Method 5 (New Concept) from an unexplored family.

### Method 1: Add Filter

**When:** Current strategy has Sharpe > 0.3 but MaxDD > 30% or win rate < 45%.

**What:** Add a filter to reduce bad trades. Examples:
- RSI strategy -> add EMA slope filter (only trade when trend agrees)
- Any strategy -> add volatility filter (skip low-ATR periods)
- Any strategy -> add volume filter (require above-average volume)
- Any strategy -> add regime filter (only trade in the regime that suits the strategy type)

**Template:**
```python
# Add to generate_signals():
ema_val = ema(df["close"], 200)
trend_up = ema_val.diff(24) > 0
signals[~trend_up & (signals == 1)] = 0  # cancel longs in downtrend
signals[trend_up & (signals == -1)] = 0  # cancel shorts in uptrend
```

**Indicator complementarity check:** Before adding a filter, consult the Indicator Redundancy Matrix. Do not add a filter based on an indicator that is highly redundant with the strategy's core indicator (e.g., don't add Stochastic filter to an RSI strategy).

### Method 2: Swap Indicator

**When:** Current strategy has plateaued (fine-grid improvement < 0.02 Sharpe over broad).

**What:** Keep the strategy structure (entry/exit logic shape), swap the core indicator. Examples:
- RSI -> Stochastic RSI (same overbought/oversold logic)
- EMA crossover -> SMA crossover, or DEMA crossover
- Bollinger bounce -> Keltner channel bounce
- Fixed MA period -> Adaptive MA (KAMA, Hull MA)

### Method 3: Combine Strategies

**When:** 2+ leaderboard entries have Sharpe > 0.2.

**What:** Create a confluence strategy that requires signals from both. Examples:
- RSI oversold + MACD bullish crossover -> long
- Bollinger squeeze + RSI divergence -> breakout entry

**Template:**
```python
def generate_signals(self, df):
    sig_a = strategy_a_logic(df)  # 1, -1, 0
    sig_b = strategy_b_logic(df)  # 1, -1, 0
    # Confluence: both must agree
    signals = pd.Series(0, index=df.index, dtype="int8")
    signals[(sig_a == 1) & (sig_b == 1)] = 1
    signals[(sig_a == -1) & (sig_b == -1)] = -1
    signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")
    return signals
```

**Combination rule:** Only combine strategies whose core indicators have LOW redundancy in the matrix (e.g., EMA trend + RSI momentum is good; RSI + Stochastic is bad).

### Method 4: Change Timeframe

**When:** Strategy works at one resolution but hasn't been tested at others.

**What:** Run the same strategy at a different resolution tier. Common useful transitions:
- 1h -> 4h (smoother, fewer trades, often better Sharpe)
- 1h -> 15m (more trades, may improve or degrade)
- Standard -> Sub-minute (completely different regime)

### Method 5: New Concept

**When:** Methods 1-4 exhausted, or 3 consecutive derivations show no improvement, or an entire strategy family is unexplored.

**What:** Introduce an untested strategy concept. Prioritize by documented BTC edge strength:

**Tier 1 — Strongest documented BTC edge:**
- **Volatility squeeze/expansion:** BB Width falls to 20th percentile -> wait for expansion above 50th percentile -> enter in direction of MACD. One of the most reliable BTC patterns.
- **Multi-lookback momentum factor:** Compute trailing returns over 7d, 30d, 90d. Weighted composite signal. 30-day horizon has strongest BTC signal (Sharpe 0.8-1.2 in studies).
- **Donchian breakout with filters:** Buy above 20-day high, sell below 10-day low. Add volume > 1.5x average AND ATR expansion > 1.2x to filter false breakouts.

**Tier 2 — Good documented edge:**
- **ATR breakout:** Price moves > N * ATR from close -> trend entry
- **VWAP reversion:** Price deviation from VWAP -> mean-reversion (requires VWAP indicator)
- **Dual-timeframe:** Signal on fast TF, confirmed by slow TF trend
- **MACD histogram divergence:** Price makes new high/low but histogram doesn't

**Tier 3 — Requires additional data / framework support:**
- **Funding rate carry:** Z-score of funding rate -> counter-trade extremes (requires funding rate data). Historical Sharpe ~0.8-1.5 through a full cycle. Positive funding (longs pay shorts) averages ~0.01-0.02% per 8h (10-18% annualized carry). Use 30-day rolling Z-score: Z > 2.0 = contrarian short bias, Z < -2.0 = contrarian long bias. Combine with other signals rather than standalone.
- **Liquidation cascade mean-reversion:** After aggregate liquidations > $200M in 1 hour + price move > 3%, fade the move with 30-60% retrace target over 4-24 hours (requires open interest/liquidation data)
- **Cross-asset risk filter:** SPX momentum + VIX + DXY as regime filter for BTC strategies (requires external data)
- **Options max pain:** BTC price gravitates toward Deribit max pain in 24-48h before major monthly expiries (last Friday). Post-expiry vol expansion as gamma pinning releases. Low-frequency overlay (12 signals/year)

### Method 6: Add Regime Awareness

**When:** Strategy shows Sharpe > 0.3 overall but with high variance across sub-periods, suggesting regime dependency.

**What:** Add a regime detection filter so the strategy only trades in favorable regimes:

```python
# Example: Only trade mean-reversion when ADX < 20
adx_values = compute_adx(df, 14)
ranging = adx_values < 20
signals[~ranging] = 0  # Cancel all signals outside ranging regime
```

**Regime-strategy alignment:**
- Trend-following strategies -> only trade when ADX > 25 or Hurst > 0.55
- Mean-reversion strategies -> only trade when ADX < 20 or Hurst < 0.45
- Breakout strategies -> only trade when volatility is compressing (ATR ratio < 0.8, anticipating expansion)

### Method 7: Adaptive Parameters

**When:** Strategy shows parameter sensitivity (narrow optimum, cliff effects in Phase 4).

**What:** Replace static parameters with adaptive ones that adjust to market conditions:

- **Dynamic lookback:** `period = base_period * (long_term_vol / current_vol)`, capped between reasonable bounds
- **Adaptive thresholds:** RSI overbought adjusts based on trend strength: `overbought = 70 + clip((price - SMA50) / ATR50 * 5, -10, 10)`
- **Volatility-scaled entry:** Only enter when signal strength exceeds a volatility-adjusted threshold

### Derivation Priority

```
1. Add filter          (when Sharpe > 0.3 but risk metrics poor)
2. Swap indicator      (when plateaued)
3. Combine             (when 2+ entries have Sharpe > 0.2)
4. Change TF           (when untested at other resolutions)
5. New concept         (when all else exhausted, or unexplored family exists)
6. Add regime filter   (when high sub-period variance suggests regime dependency)
7. Adaptive params     (when narrow optimum / cliff effects detected)
```

### Genealogy Tracking

Track the derivation chain for the Final Report:

```
Seed: RSI (Sharpe 0.45)
  -> RSI + EMA filter (Sharpe 0.62)  [Method 1: Add filter]
  -> RSI + EMA + Vol filter (Sharpe 0.58)  [Method 1: Add filter — no improvement]
  -> MACD crossover (Sharpe 0.38)  [Method 5: New concept — trend-following family]
  -> MACD + RSI confluence (Sharpe 0.71)  [Method 3: Combine]
  -> BB Squeeze (Sharpe 0.68)  [Method 5: New concept — breakout family]
```

### Research-Driven Strategy Suggestions

When deriving new strategies (Method 5 or when the loop needs fresh ideas), use this catalog of concrete, buildable strategies ranked by documented evidence and implementation feasibility. Each suggestion includes the strategy family, core logic, expected parameters, and data requirements — providing enough detail for the executor to build a working `Strategy` subclass.

**Tier A — Strongest Evidence, Buildable with Current Framework:**

| # | Strategy Name | Family | Core Logic | Key Parameters | Expected Sharpe | Data Needed |
|---|---|---|---|---|---|---|
| A1 | **MTF Trend Filter** | Multi-timeframe | Daily MACD histogram direction filters 1H MACD crossover signals. Only take 1H signals aligned with daily trend. | `daily_fast`, `daily_slow`, `daily_signal`, `hourly_fast`, `hourly_slow` | 0.6-1.0 | Standard OHLCV (resample internally) |
| A2 | **Volatility Squeeze Breakout** | Breakout | BB Width falls to 20th percentile of 100-bar rolling rank. Enter in MACD direction when BB Width crosses above 50th percentile. | `bb_period`, `bb_std`, `width_lookback`, `squeeze_pctl`, `expansion_pctl` | 0.8-1.2 | Standard OHLCV |
| A3 | **Multi-Lookback Momentum** | Momentum factor | Weighted composite of returns over 7d, 30d, 90d horizons. 30d horizon dominates. Long when composite > +threshold, short when < -threshold. | `w_7d`, `w_30d`, `w_90d`, `entry_threshold`, `exit_threshold` | 0.8-1.2 | Standard OHLCV |
| A4 | **CVD Divergence** | Order flow | Price makes new high but CVD does not (bearish div), or price makes new low but CVD does not (bullish div). Compute CVD from `taker_buy_base_volume`. | `divergence_lookback`, `price_threshold`, `cvd_smooth` | 0.4-0.7 | OHLCV + `taker_buy_base_volume` column |
| A5 | **Donchian Breakout + Volume/ATR Filter** | Breakout | Buy above 20-day high, sell below 10-day low. Filter: volume > 1.5x 20-day avg AND ATR expansion > 1.2x 20-day avg ATR. | `entry_period`, `exit_period`, `vol_mult`, `atr_mult` | 0.6-1.0 | Standard OHLCV |
| A6 | **Adaptive RSI (Regime-Filtered)** | Mean-reversion | RSI mean-reversion but only active when ADX < 20 (ranging regime). Adaptive overbought/oversold thresholds based on trend strength. | `rsi_period`, `adx_period`, `adx_threshold`, `ob_base`, `os_base` | 0.5-0.9 | Standard OHLCV |

**Tier B — Good Evidence, Requires Additional Data Columns or External Data:**

| # | Strategy Name | Family | Core Logic | Key Parameters | Expected Sharpe | Data Needed |
|---|---|---|---|---|---|---|
| B1 | **Funding Rate Contrarian** | Funding rate | Compute rolling Z-score of funding rate. Short when Z > +2.0, long when Z < -2.0. Optional: scale position by Z magnitude. | `z_lookback`, `z_long_threshold`, `z_short_threshold`, `cooldown_bars` | 0.5-1.5 | Funding rate CSV (Binance API, free) |
| B2 | **VWAP Mean Reversion (2σ)** | Order flow | Compute intraday VWAP with σ bands. Enter long at -2σ, short at +2σ. Exit at VWAP. Skip first 5 bars of session. | `sigma_entry`, `sigma_exit`, `session_skip_bars` | 0.5-0.9 | Standard OHLCV (compute VWAP from typical price * volume) |
| B3 | **Fear & Greed Overlay** | Sentiment-driven | Use FGI as position sizing multiplier on any base strategy. FGI < 20 → 1.5x long sizing. FGI > 80 → 0.5x long sizing, 1.5x short sizing. | `fear_threshold`, `greed_threshold`, `fear_mult`, `greed_mult` | +0.1-0.3 improvement | FGI CSV (alternative.me API, free) |
| B4 | **Liquidation Cascade Bounce** | Event-driven | Detect cascade (>0.5% 1m bar + volume z > 3 + 3 consecutive same-direction bars + 10-bar move > 3%). Wait 3 bars stabilization, enter targeting 40% retracement. | `cascade_pct`, `vol_z_thresh`, `consec_bars`, `cum_move_pct`, `retrace_target` | 1.0-1.5 per-trade | Standard 1m OHLCV |
| B5 | **CME Gap Fill** | Event-driven | Detect weekend gaps (Monday open vs Friday close). Short if gap > +0.5%, long if gap < -0.5%. Target: gap fill level. Stop: 1.5x gap size. Max hold: 1 week. | `gap_threshold_pct`, `stop_mult`, `max_hold_bars` | 0.5-0.8 | Standard OHLCV (hour-of-week logic) |
| B6 | **Triple Screen (Elder BTC)** | Multi-timeframe | Weekly MACD histogram direction (trend). Daily Stochastic oversold/overbought (setup). 4H breakout (entry). | `weekly_macd_fast/slow/signal`, `daily_stoch_period`, `daily_stoch_ob/os` | 0.6-1.0 | Standard OHLCV (resample internally) |
| B7 | **OI Leverage Filter** | Carry | Reduce all position sizes by 50% when OI/MCap > 7%. Go flat when OI/MCap > 10%. Overlay on any base strategy. | `oi_caution_threshold`, `oi_halt_threshold`, `size_reduction` | +0.1-0.2 improvement | OI CSV (CoinGlass, free) |

**Tier C — Exploratory, Requires More Infrastructure or Has Higher Uncertainty:**

| # | Strategy Name | Family | Core Logic | Key Parameters | Expected Sharpe | Data Needed |
|---|---|---|---|---|---|---|
| C1 | **Funding Rate Cash-and-Carry** | Funding rate | Spot long + perp short. Collect funding. Requires dual-leg equity tracking. | `entry_z`, `exit_z`, `leverage`, `rebalance_freq` | 3.0-6.0 | Funding rate + dual-leg engine modification |
| C2 | **MVRV Cycle Overlay** | Sentiment-driven | MVRV Z < 0 → full long bias. Z 3-5 → reduce longs 50%. Z > 7 → exit all longs. Monthly rebalance. | `underval_z`, `overval_z`, `extreme_z` | Macro filter only | MVRV CSV (CoinMetrics, free) |
| C3 | **Hash Ribbon Buy** | Event-driven | 30-day hash rate SMA crosses above 60-day SMA after capitulation. Buy signal. Hold for 12 months. 6/6 profitable since 2018. | `short_ma`, `long_ma`, `hold_days` | Very high per-trade but 1-2 signals/cycle | Hash rate CSV (blockchain.com, free) |
| C4 | **Harmony Index Composite** | Multi-timeframe | Compute Harmony Index across 4 TFs (Weekly/Daily/4H/1H). Only trade when HI > 0.40 (long) or HI < -0.40 (short). | `weights`, `hi_threshold`, `indicator_per_tf` | 0.6-1.0 | Standard OHLCV (resample internally) |
| C5 | **DXY Trend Filter** | Cross-asset | Long BTC only when DXY < 50d SMA. Short bias when DXY > 50d SMA and rising. DXY trend reversals lead BTC by 2-6 weeks. | `dxy_sma_period`, `trend_lookback` | +0.2-0.4 improvement | DXY CSV (Yahoo Finance, free) |
| C6 | **VIX-Scaled Position Sizing** | Cross-asset | Scale all positions by `max(0.2, 1 - max(0, VIX-25)/25)`. Reduces exposure in fear, full exposure in calm. | `vix_floor`, `vix_start`, `vix_scale` | +0.1-0.2 improvement | VIX CSV (Yahoo Finance, free) |
| C7 | **CARS Composite Regime** | Cross-asset | Weighted composite of DXY, Nasdaq, VIX, ETH/BTC, Gold signals. CARS > +0.3 → full long; CARS < -0.3 → reduce/short. | `weights`, `bull_threshold`, `bear_threshold` | +0.15-0.30 improvement | Multiple CSVs (all free) |
| C8 | **Regime-Filtered RSI Mean Reversion** | Mean-reversion | RSI(14) < 25 for longs, > 80 for shorts. ADX < 25 regime filter. Partial reversion target (z=-1.0). Time exit at 7 bars. Stop 2.5x ATR. | `rsi_period`, `rsi_long/short_thresh`, `adx_filter`, `max_hold`, `atr_stop_mult` | 0.6-1.1 | Standard OHLCV |
| C9 | **Volume Exhaustion Fade** | Mean-reversion | After 3+ consecutive bars with volume > 2x 20-bar avg, enter counter-trend. RSI confirmation (>80 or <20). Exit at partial reversion or 5-bar time stop. | `consec_bars`, `vol_mult`, `rsi_confirm`, `exit_bars` | 0.4-0.8 | Standard OHLCV |
| C10 | **Drawdown-Managed Trend** | Trend-following | Any trend strategy + layered risk management: vol targeting (20% target), anti-martingale (threshold 20, cap 2x), drawdown reduction (go-flat at -35%). | Base strategy params + `vol_target`, `am_threshold`, `am_cap`, `dd_flat_level` | Base Sharpe + 0.1-0.3 improvement | Standard OHLCV |
| C11 | **Opening Range Volume Momentum** | Momentum factor | If US open (14:00-15:00 UTC) volume > 12% of prior 24h total, trade first-hour direction. Hold until 21:00 UTC or 1.5x ATR stop. | `volume_pct_threshold`, `hold_end_utc`, `atr_stop_mult` | 0.5-0.8 | Standard 1h OHLCV with UTC timestamps |
| C12 | **Dollar Bar Momentum** | Momentum factor | Resample to $15M dollar bars, apply multi-lookback momentum composite. Eliminates low-vol noise, increases resolution during active periods. | `dollar_bar_size`, `w_short`, `w_med`, `w_long`, `threshold` | 0.7-1.1 | Tick/minute data for dollar bar construction |
| C13 | **Triple MR Confirmation** | Mean-reversion | RSI(14) < 20 AND price below BB(20, 2.5) AND volume > 2x 20-bar avg. All three must align. Partial reversion target. Time stop 5 bars. | `rsi_thresh`, `bb_std`, `vol_mult`, `reversion_target`, `max_hold` | 0.8-1.5+ | Standard OHLCV |
| C14 | **Walk-Forward Validated Ensemble** | Multi-strategy | Run 3+ strategies through walk-forward validation. Allocate via inverse-vol risk parity. Apply regime-conditional weights (ADX-based tanh smoothing). Portfolio-level circuit breakers (Yellow/Orange/Red alerts). | `wf_is_months`, `wf_oos_months`, `risk_parity_lookback`, `adx_period`, `regime_blend_alpha` | 0.8-1.3 (portfolio Sharpe) | Standard OHLCV + ADX |
| C15 | **Cost-Aware Trend with Funding** | Trend-following | EMA crossover with Tier 2 transaction cost model (sqrt impact + funding drag). Restrict to EU/US overlap for lower spread. Funding drag reduces hold time vs standard trend. | Base EMA params + `impact_coeff`, `adv_estimate`, `funding_rate`, `session_filter` | 0.5-0.9 (realistic after costs) | Standard OHLCV + session timestamps |
| C16 | **WFER-Decay Adaptive Sizing** | Meta-strategy | Any validated strategy + decay-adjusted sizing. Monitor WFER quarterly, estimate λ_decay, auto-reduce size per `exp(-λt)`. CUSUM + IC monitoring for structural break detection. | Base params + `lambda_decay`, `size_floor`, `cusum_h`, `ic_window` | Base Sharpe × WFER | Standard OHLCV + monitoring data |
| C17 | **Regime-Parity Portfolio** | Multi-strategy | 3-strategy (trend + MR + carry) with dual-layer allocation: risk parity base + regime-conditional overlay. Smooth rebalancing at 20% drift trigger. Heat limit at 1.5x target vol. | `vol_lookback`, `drift_trigger`, `heat_limit`, `regime_blend_alpha`, `alert_thresholds` | 0.8-1.3 (portfolio) | Standard OHLCV + ADX |
| C18 | **CPCV-Validated Momentum** | Momentum factor | Multi-lookback momentum scored via CPCV (N=6, k=2). Only deploy if all 5 robustness criteria pass (μ_OOS > 0.50, positive_rate > 75%). Embargo = 2.5x max lookback. | `lookback_short/med/long`, `n_groups`, `k_oos`, `embargo_mult` | 0.6-1.0 | Standard OHLCV |

**How to use this catalog:**
1. During Method 5 (New Concept) derivation, select from Tier A first, then Tier B
2. Check which families remain unexplored in the current loop and pick from that family
3. The executor builds the strategy class using the Strategy Creation Template
4. Parameter ranges are provided in the Key Parameters column — create a `param_grid()` from them
5. Data requirements indicate whether additional columns or external CSVs are needed
6. Expected Sharpe is the documented range from research — actual results will vary by market regime

---

## Mean Reversion Deep Dive — BTC-Specific

Mean reversion is the second most important strategy family for BTC after trend-following, but BTC's trending character (Hurst exponent ~0.55-0.65 on daily data) makes standard equity mean-reversion parameters fail. This section provides BTC-calibrated parameters and regime-conditional guidance.

### Best Metrics for BTC Mean Reversion

**Metric effectiveness ranking (BTC daily data, 2018-2025):**

| Metric | BTC Optimal Thresholds | Win Rate Range | Notes |
|---|---|---|---|
| RSI(14) | Long < 20/25, Short > 75/80 | 55-68% | Wider than equity 30/70 |
| Bollinger Band (20, 2.0-2.5) | Z-score < -2.0 / > +2.0 | 52-62% | Use 2.5σ for more selective |
| RSI + BB confluence | RSI < 25 AND below lower BB | 60-68% | +5-10pp over single indicator |
| RSI + BB + Volume spike | Triple confirmation | 68-78% | Best quality, fewest signals |
| Rolling Hurst exponent | H < 0.50 as master switch | — | Regime gate, not direct signal |

**Key finding:** Standard equity RSI 30/70 thresholds are too tight for BTC. The 20/80 threshold on RSI-14 daily provides the best risk-adjusted return per trade. The 25/75 level offers a better balance between frequency and quality.

**Asymmetric thresholds:** BTC's structural bullish bias means longs from oversold work better than shorts from overbought. Use RSI < 25 for longs, RSI > 80-85 for shorts. During confirmed bear markets (price below 200-day SMA), the asymmetry reverses.

### Regime-Conditional Mean Reversion

Mean reversion works in ranging markets and fails in trending markets. Regime filtering is the single most impactful improvement:

**ADX-based filtering:** Restrict mean reversion to ADX < 20-25. Typical impact: +0.3-0.7 Sharpe improvement (e.g., 0.4 unfiltered → 0.8-1.1 filtered). Win rate improvement: +8-15pp. Tradeoff: filters out ~50-65% of signals.

**Volatility regime filtering:** Mean revert in low-vol, trend-follow in high-vol. Compute 20-day realized vol percentile over trailing 252 days. Below 50th percentile → mean reversion. Above → trend-following or flat. Sharpe improvement: +0.4-0.8.

**Recommended dual filter:** ADX < 25 AND realized volatility percentile < 50. This captures the "ranging and calm" regime where mean reversion has the highest expected payoff.

**Rolling Hurst exponent as master switch:**
- H < 0.45: Full mean reversion sizing
- H 0.45-0.55: 50% sizing
- H > 0.55: No mean reversion trades (trending regime)
- Compute over 200-day rolling window. Lagging indicator — combine with ADX for faster detection.

### Mean Reversion Exit Strategies

**Critical finding:** Mean reversion payoff is front-loaded. ~60-70% of reversion occurs within the first 5-10 bars (1h TF) or 3-7 bars (daily TF). Beyond that, edge dissipates.

**Optimal holding periods:**
- 1h timeframe: 3-12 bars (3-12 hours), median winner closes in 4-6 bars
- 4h timeframe: 3-8 bars (12-32 hours), median winner closes in 3-5 bars
- Daily timeframe: 3-7 bars (3-7 days), beyond 10 days → position becomes random directional bet

**Winners vs losers profile:** Winners resolve quickly (modal: 2-5 bars). Losers linger (modal: 8-15 bars, ~2-3x winner duration). Time-based stop: if no profit within 5-7 bars, it is more likely a loser.

**Target-based exits:**
- Full reversion to mean (z-score = 0): reached ~55-65% of time from z < -2.0 entries
- Partial reversion (z-score = -1.0): reached ~70-80% of time (recommended default)
- RSI-based exit: enter RSI < 20, exit when RSI crosses 50

**Stop-loss calibration:**
- **Tight stops destroy mean reversion.** Entries are inherently catching a falling knife.
- Optimal distance: 1.5-3.0x ATR at entry. Tighter (0.5x ATR) reduces win rate by 15-25pp.
- Catastrophic stop: 5-8% below entry for black swan protection
- Use MAE analysis: median MAE for BTC daily winning trades is 2-5%, 90th percentile is 6-10%. Stops tighter than 5-8% convert eventual winners into realized losers.

### Multi-Indicator Mean Reversion Performance

| Approach | Win Rate | Sharpe Ratio | Trades/Year (Daily) |
|---|---|---|---|
| RSI only (25/75) | 55-62% | 0.4-0.8 | 15-25 |
| BB z-score only (2.0) | 52-58% | 0.3-0.6 | 20-30 |
| RSI + BB confluence | 60-68% | 0.6-1.1 | 8-15 |
| RSI + BB + Volume | 68-78% | 0.8-1.5+ | 5-12 |

Multi-indicator confirmation trades signal count for quality. The consistent pattern: each additional orthogonal confirmation improves win rate by ~5-10pp at the cost of ~30-50% fewer signals.

### Mean Reversion by Market Phase

| Phase | Sharpe Range | Notes |
|---|---|---|
| Post-halving accumulation (0-6 mo) | 0.5-1.0 | Moderate vol, mild upward bias |
| Bull run acceleration (6-18 mo) | 0.0-0.3 | Long dip-buys work; shorts destroyed |
| Blow-off top (18-24 mo) | -0.5-0.2 | Most dangerous phase; drawdowns 30-50% |
| Bear market decline (24-36 mo) | 0.5-1.0 | Buying oversold capitulation works (WR 62-72%) |
| Pre-halving accumulation (36-48 mo) | 1.0-1.5+ | Best phase for MR; low vol, range-bound |

**Range-bound markets:** Sharpe 1.0-2.0, win rate 65-80%. The challenge is identifying ranges in real-time (use ADX + Hurst).

### Common Mean Reversion Failures on BTC

**Why equity parameters fail:**
1. BTC vol ~60-100% vs equity ~15-25% → standard thresholds too tight
2. Excess kurtosis 5-15 vs ~3 → 2σ events far more frequent
3. Hurst ~0.55-0.65 → persistent trending character (H > 0.50)
4. 24/7 trading → no opening gap reversion effect
5. Asymmetric liquidity → thin books cause overshoots, creating false signals

**The "mean reversion trap":** Entry during regime change. Warning signs:
- Volume expanding on break below range (breakout volume, not capitulation)
- ADX rising above 25 and increasing
- Z-score persistently negative for unusually long period (mean itself is shifting)
- Moving averages converging toward death/golden cross

### Mean Reversion Parameter Summary

| Parameter | Recommended Range | Notes |
|---|---|---|
| RSI period | 7-14 (intraday), 14-21 (daily) | Shorter = more sensitive |
| RSI thresholds | 20/80 or 25/75 | Wider than equity 30/70 |
| BB period | 20 (standard) | 50-100 for intraday |
| BB sigma | 2.0-2.5 | 2.5 for selective signals |
| ADX regime filter | ADX < 20-25 | Critical for performance |
| Volatility filter | Below 50th percentile | Realized vol ranking |
| Optimal TF for MR | 15m-1h | Strongest MR evidence |
| Holding period (daily) | 3-7 bars | Front-loaded payoff |
| Stop-loss (daily) | 2-3x ATR (5-10%) | Tight stops destructive |
| Multi-indicator | RSI + BB minimum | Volume for third confirmation |
| Hurst threshold | < 0.50 for MR | Pause when H > 0.55-0.60 |

---

## Convergence & Scoring System

### Composite Score

For leaderboard ranking, use a composite score that balances return and risk:

```
Score = (Sharpe * 0.35) + (Sortino * 0.15) + (Calmar * 0.10) + (WinRate/100 * 0.10) + (ProfitFactor_capped/3 * 0.10) + (RegimeConsistency * 0.10) + (WalkForwardEfficiency * 0.10)

Where:
  ProfitFactor_capped = min(profit_factor, 3.0)  # cap to avoid outlier dominance; treat "inf" as 3.0
  RegimeConsistency = (regimes_profitable / regimes_tested)  # 0 to 1; default 0.5 if not yet tested
  WalkForwardEfficiency = min(OOS_Sharpe / IS_Sharpe, 1.0)  # 0 to 1; default 0.5 if not yet tested
```

This weights risk-adjusted return (Sharpe/Sortino) most heavily, with bonus for consistency (win rate, profit factor), drawdown management (Calmar), regime robustness, and walk-forward validation quality.

**Supplementary scoring (for tiebreaking and risk assessment):** When two strategies have similar composite scores, prefer the one with: higher Tail Ratio (> 0.8), lower CVaR magnitude, higher GPR, and higher haircut-adjusted Sharpe. These supplementary metrics do not enter the composite formula to keep it stable, but they inform the qualitative recommendation.

### Multiple Testing Correction

**Critical rule:** When selecting the best result from a parameter sweep, the observed Sharpe is inflated by luck. The more combinations tested, the higher the bar for significance.

After every parameter sweep, compute the **Deflated Sharpe Ratio (DSR)**:

```
DSR = CDF[ (SR_observed - SR_0) / SE(SR) ]

Where:
  SR_observed = best Sharpe from the sweep
  SR_0 = expected max Sharpe under pure noise, given N trials:
    SR_0 ≈ sqrt(Var(all_SR_values)) * ((1 - 0.5772) * Z_inv(1 - 1/N) + 0.5772 * Z_inv(1 - 1/(N*e)))
  SE(SR) = sqrt((1 - skew*SR + ((kurtosis-1)/4)*SR^2) / (T-1))
    where T = number of return observations, skew/kurtosis from the best strategy's returns
  N = total parameter combinations tested (across all sweeps for this strategy)
```

**Thresholds:**
- DSR > 0.95 -> PASS: result is significant even after correcting for multiple testing
- DSR 0.90-0.95 -> MARGINAL: proceed with heavy caveats
- DSR < 0.90 -> FAIL: likely a spurious result from testing too many combinations

**Note on correlated parameters:** When adjacent parameter values produce nearly identical return streams (e.g., SMA(49) vs SMA(50)), the effective N is smaller than the raw N. If most top results cluster in a tight parameter region, the effective N is lower and DSR is more favorable. When results are scattered, effective N is closer to raw N.

**Practical shortcut:** If DSR computation is complex, use the **Bonferroni haircut** as a simpler alternative: multiply the p-value of the best Sharpe by N. If the adjusted p-value > 0.05, the result is not significant. This is conservative but easy to compute.

### Convergence Criteria

Any ONE of these triggers STOP:

| # | Criterion | Description |
|---|-----------|-------------|
| 1 | **Hard stop** | 10 strategies tested total |
| 2 | **Stagnation** | 3 consecutive strategies with no leaderboard improvement (new Sharpe doesn't exceed #1 by > 0.05) |
| 3 | **Diminishing returns** | Last 2 improvements each < 0.02 Sharpe gain |
| 4 | **All-fail** | 3 consecutive FAIL strategies (fail robustness Phase 5) |
| 5 | **User interrupt** | Always honored immediately |

### Self-Check Protocol

After each strategy completes the inner loop, perform this self-check:

```
1. Record result on Leaderboard
2. Compare to current #1:
   - If new Sharpe > #1 Sharpe + 0.05 -> "IMPROVEMENT" (reset stagnation counter)
   - Else -> "NO IMPROVEMENT" (increment stagnation counter)
3. Check convergence criteria 1-4
4. If not converged:
   - Log improvement delta
   - Check strategy family taxonomy coverage
   - Select derivation method based on current state
   - Announce: "Strategy N complete. Sharpe=X. Leaderboard position: #Y.
     Convergence: [not triggered]. Families explored: [list].
     Next: [derivation method + rationale]."
5. If converged:
   - Announce convergence reason
   - Produce Final Report
```

---

## Leaderboard Format

Track all tested strategies in a table. Update after each strategy completes.

```
| # | Strategy          | Mode      | TF  | Sharpe | Sortino | MaxDD   | WinR  | Trades | Score | DSR  | Robust | Derivation        |
|---|-------------------|-----------|-----|--------|---------|---------|-------|--------|-------|------|--------|-------------------|
| 1 | RSI+EMA filter    | long+short| 1h  | 0.62   | 0.85    | -18.3%  | 52.1% | 84     | 0.71  | 0.97 | PASS   | Method 1 from #2  |
| 2 | RSI               | long-only | 1h  | 0.45   | 0.61    | -22.7%  | 48.3% | 120    | 0.52  | 0.94 | PASS   | Seed              |
| 3 | MACD crossover    | long-only | 1h  | 0.38   | 0.44    | -31.2%  | 44.1% | 95     | 0.41  | 0.91 | MARG   | Method 5          |
| 4 | EMA crossover     | long+short| 4h  | 0.22   | 0.28    | -35.8%  | 41.0% | 45     | 0.28  | 0.88 | FAIL   | Method 2 from #2  |
```

**Columns:**
- **#**: Rank by Score (descending)
- **Strategy**: Name + key modifier
- **Mode**: `long-only` or `long+short`
- **TF**: Timeframe tested
- **Sharpe/Sortino/MaxDD/WinR/Trades**: Key metrics
- **Score**: Composite score (see formula above)
- **DSR**: Deflated Sharpe Ratio (multiple testing correction)
- **Robust**: `PASS`, `MARG` (marginal), or `FAIL` from Phase 5
- **Derivation**: How this strategy was derived

**Extended columns** (track for Phase 8 report but omit from main table for readability):
- **CVaR95**: Conditional Value at Risk (95% daily)
- **TailR**: Tail Ratio (95th/5th percentile)
- **GPR**: Gain-to-Pain Ratio
- **Haircut Sharpe**: Sharpe * (1 - haircut_factor) based on optimization level

---

## The 8-Phase Analysis Protocol

### Phase 1 — Structural Assessment

**Goal:** Understand the data and establish baselines before any optimization.

**Steps:**

1. Load data and inspect the date range and bar count:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from backtester import load_candles
df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='1h')
print(f'Range: {df.index[0]} to {df.index[-1]}')
print(f'Bars: {len(df):,}')
print(f'Start price: {df.close.iloc[0]:.2f}')
print(f'End price: {df.close.iloc[-1]:.2f}')
bh_return = (df.close.iloc[-1] / df.close.iloc[0] - 1) * 100
print(f'Buy-and-hold return: {bh_return:.1f}%')
"
```

2. Run a single backtest with default params to establish a baseline:
```bash
python3 run_backtest.py --strategy {name} --resample {tf}
```

3. Assess market structure — is BTC trending or mean-reverting over this period?
   - Compare start and end prices
   - If end price > 2x start price -> strong uptrend -> favor trend-following
   - If end price roughly equals start price -> range-bound -> mean-reversion may work
   - Compute rolling volatility to identify regime changes within the period

4. **Load and apply run history** (if available):
   - Classify the regime using the Regime Classification table
   - Read `data/runs/KNOWLEDGE.md`
   - Look up current regime in the Strategy-Regime Performance Matrix
   - Report: "Historical context: [N] prior runs. Regime: [label]. Best historical family: [X] (Sharpe [Y]). Anti-patterns: [list]."
   - Adjust seed strategy and param grid centers per Robust Parameter Regions
   - If no history: "No prior run history. Starting fresh."

**Note on resolution:** Use the resolution appropriate for the strategy. Default to 1h for standard analysis. For sub-minute strategies, ensure data has been fetched first (see Multi-Resolution Protocol).

**Decision gate:**
- If strategy type conflicts with market structure (e.g., mean-reversion on strongly trending data), flag the mismatch. Consider both modes and alternative strategies.

---

### Phase 2 — Broad Landscape Sweep

**Goal:** Map the full parameter space and evaluate both trading modes.

**Steps:**

1. Run the strategy's full `param_grid()` sweep in **both modes**:
```bash
# Bidirectional (long + short)
python3 run_backtest.py --strategy {name} --resample {tf} --sweep --rank-by sharpe_ratio --top 20

# Long-only
python3 run_backtest.py --strategy {name} --resample {tf} --sweep --rank-by sharpe_ratio --top 20 --long-only
```

2. **Record total combinations tested (N)** for later DSR calculation. This includes both mode sweeps.

3. Inspect results. Key questions:
   - Are ANY combos profitable?
   - What's the range of Sharpe ratios?
   - Is long-only consistently better than long+short?
   - What do the direction-split metrics show? (`long_trades`, `short_trades`, `long_win_rate_pct`, `short_win_rate_pct`)

4. Apply mode selection rules (see Long+Short Evaluation Protocol).

**Fast-fail heuristics** — check these before investing more compute:

| Signal | Threshold | Action |
|--------|-----------|--------|
| Zero trades across entire grid | 0 trades | Diagnostic, not strategy failure. Check NaN/warm-up (see Pre-Backtest Validation). |
| All-negative Sharpe across grid | Max Sharpe < 0 | Signal hypothesis has no merit. Abandon immediately. |
| Profit factor < 1.0 everywhere | PF < 1.0 | Strategy is a net loser. No amount of optimization fixes a wrong directional bet. |
| Too few trades for statistical inference | < 30 trades | Cannot draw conclusions. Check timeframe/param ranges. |
| Trades exist but Sharpe marginal | Best Sharpe < 0.5 | Weak signal. Proceed only if trade count > 100 and profit factor > 1.1. |

**Minimum viable signal (MVS) by stage:**

| Stage | Min Sharpe | Min Trades | Purpose |
|-------|-----------|-----------|---------|
| Broad sweep -> Phase 3 | > 0.3 | > 30 | Worth narrowing |
| Fine grid -> Phase 5 | > 0.5 | > 50 | Worth robustness testing |
| Walk-forward -> recommend | > 1.0 OOS | > 100 | Worth presenting to user |

**Statistical significance by Sharpe level** — the number of trades needed for 95% confidence that the strategy's true mean return is positive:

| Annualized Sharpe | Approx Trades Needed (95% conf) | Practical Implication |
|-------------------|------|------|
| 0.5 | ~4,000 | Requires multi-year daily data; near-impossible for weekly strategies |
| 1.0 | ~1,000 | Achievable with 3-4 years of daily trading |
| 1.5 | ~425 | ~2 years of daily trading |
| 2.0 | ~240 | ~1 year of daily trading |
| 3.0 | ~107 | ~6 months of daily trading; sub-hour strategies reach this fast |

**Minimum Track Record Length (MinTRL)** — Lopez de Prado's formula, accounting for non-normal returns:

```
MinTRL (years) = 1 + (1 - skew*SR + ((kurtosis-1)/4)*SR^2) * (1.96/SR)^2
```

BTC daily returns have excess kurtosis ~6-10 and skewness -1 to +1. This substantially increases MinTRL vs the Gaussian case. For SR=1.0 with kurtosis=8: MinTRL ≈ 1 + (1 + 0 + 7/4) * 3.84 ≈ **11.6 years**. For SR=2.0 with kurtosis=8: MinTRL ≈ **3.6 years**. Crypto's fat tails demand longer track records than traditional assets for the same Sharpe level. When data is insufficient, supplement with bootstrap confidence intervals (see Phase 5).

**Decision gate:**

| Best Sharpe (either mode) | Action |
|---------------------------|--------|
| > 0.3 | Strong signal. Proceed to Phase 3 with selected mode(s). |
| 0 to 0.3 | Marginal signal. Proceed with caution. |
| -0.5 to 0 | Weak. Try both modes if not done. If both tried, PIVOT. |
| < -0.5 | Strategy is broken for this data. Abandon and PIVOT immediately. |

---

### Phase 3 — Statistical Filtering & Trade Quality Analysis

**Goal:** Remove noise, identify robust parameter regions, and assess trade quality.

**Steps:**

1. Capture sweep results and filter for statistical significance:
```bash
result=$(python3 run_backtest.py --strategy {name} --resample {tf} --sweep --rank-by sharpe_ratio --top 80 {mode_flag} 2>/dev/null)
echo "$result" | python3 -c "
import json, sys
data = json.load(sys.stdin)
# Filter: at least 20 trades
filtered = [r for r in data['results'] if r['metrics']['total_trades'] >= 20]
if not filtered:
    # Relax to 10 trades but flag low confidence
    filtered = [r for r in data['results'] if r['metrics']['total_trades'] >= 10]
    print('WARNING: Relaxed to 10-trade minimum. Low confidence.')
filtered.sort(key=lambda r: r['metrics']['sharpe_ratio'], reverse=True)
for r in filtered[:10]:
    m = r['metrics']
    print(f\"Sharpe={m['sharpe_ratio']:.2f}  Trades={m['total_trades']}  Return={m['total_return_pct']:.1f}%  Params={r['params']}\")
"
```

2. Check for parameter clusters:
   - Are the top 10 combos in a tight parameter region? -> Robust (broad hill)
   - Are they scattered across the grid? -> Likely overfit (narrow spike)
   - **Broad hill test:** If the top 10 results span less than 30% of the parameter range, this is a positive sign.

3. Compute parameter sensitivity for the best combo:
   - For each parameter, check how Sharpe changes when moving +/-1 grid step
   - If Sharpe drops by > 50% from a single step change -> fragile, flag concern

4. **Complexity penalization:** Check if the strategy has enough trades per free parameter:

| Parameters | Minimum Trades Required | Reliability |
|-----------|------------------------|-------------|
| 1-2 | 40-60 | Marginal |
| 3 | 60-90 | Marginal |
| 4-5 | 100-150 | Acceptable |
| 6+ | 150+ | Required for any confidence |

   If `total_trades < 30 * num_parameters`, flag the result as **statistically unreliable**. The more parameters optimized, the higher the multiple testing penalty. Prefer simpler strategies (fewer params) when Sharpe is similar.

5. **Trade quality analysis:** Check if returns are concentrated or distributed:
```python
# From the best parameter set results:
pnls = [t["pnl_pct"] for t in results["trades"]]
# Concentration check: does the top 10% of trades account for >50% of total return?
sorted_pnls = sorted(pnls, reverse=True)
top_10pct = sorted_pnls[:max(1, len(sorted_pnls)//10)]
total_pnl = sum(pnls)
top_10pct_pnl = sum(top_10pct)
concentration = top_10pct_pnl / total_pnl if total_pnl > 0 else 0
# If concentration > 0.5: returns depend on a few lucky trades — flag concern
# Max consecutive losses:
max_consec_loss = max(len(list(g)) for k, g in itertools.groupby(pnls, key=lambda x: x <= 0) if k)
```

---

### Phase 4 — Fine-Grained Refinement

**Goal:** Zoom into the sweet spot with a custom fine grid.

**Steps:**

1. Define a fine grid around the best region from Phase 3:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from backtester import load_candles, BacktestConfig
from backtester.sweep import run_sweep
from strategies.{name}_strategy import {ClassName}

df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='{tf}')
config = BacktestConfig(long_only={long_only})

# Example fine grid (adjust ranges based on Phase 3 results)
grid = {
    'param1': [v-2, v-1, v, v+1, v+2],
    'param2': [w-2, w-1, w, w+1, w+2],
}

sweep = run_sweep({ClassName}, df, grid, config, rank_by='sharpe_ratio', top=20)
for r in sweep['results']:
    if r['metrics']['total_trades'] >= 10:
        m = r['metrics']
        print(f\"Sharpe={m['sharpe_ratio']:.2f}  Sortino={m['sortino_ratio']:.2f}  \"
              f\"Return={m['total_return_pct']:.1f}%  Trades={m['total_trades']}  \"
              f\"MaxDD={m['max_drawdown_pct']:.1f}%  Params={r['params']}\")
"
```

2. **Update total N** with the fine grid combinations for DSR calculation.

3. **Loop mode refinement check:** If the fine-grid best Sharpe improves by < 0.02 over the broad-sweep best -> stop refining, move to Phase 5 with the broad-sweep winner. Do not over-optimize.

4. Identify top 3-5 candidates for robustness testing.

5. **Prefer plateau centers over peaks:** Select parameter values at the center of a broad, well-performing region rather than the single best point. The plateau center is more likely to generalize.

6. **Parameter search method by dimensionality:**

| Free Parameters | Recommended Method | Budget |
|----------------|-------------------|--------|
| 1-2 | Exhaustive grid search | Full grid |
| 3 | Grid if < 1000 combos; else random search (200 samples) | 200-1000 |
| 4-5 | Bayesian optimization (50-100 iterations after 30 random init) | 80-130 |
| 6+ | Random search (200 samples) then neighborhood refinement around top 3 | 250-350 |

**Warm-starting from KNOWLEDGE.md:** When prior run data exists for this strategy/regime pair, initialize the search with known good parameter regions. For grid search, center the fine grid on the historical best. For Bayesian optimization, seed with 5-10 known good points from Robust Parameter Regions. This typically saves 30-50% of the search budget.

**Expanding neighborhood search (refinement):** After identifying the best region from the broad search, check all +/-1 step neighbors. If any neighbor is better, move to it and expand to +/-2 steps. Continue until no improvement. Run from 3 starting points to avoid local optima.

---

### Phase 5 — Robustness Validation

**Goal:** Verify the best candidates are not overfit. This is the critical phase — never skip it.

Run all six tests for each candidate:

#### Test 1: Sub-Period Consistency

Split the data into 3 windows. Candidate must be profitable in at least 2 of 3.

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from backtester import load_candles, BacktestConfig, run_backtest, compute_metrics
from strategies.{name}_strategy import {ClassName}

periods = [
    ('2020-01-01', '2021-12-31', 'Period 1'),
    ('2022-01-01', '2023-12-31', 'Period 2'),
    ('2024-01-01', '2025-01-01', 'Period 3'),
]
config = BacktestConfig(long_only={long_only})
strategy = {ClassName}({params})

for start, end, label in periods:
    df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='{tf}', start=start, end=end)
    signals = strategy.generate_signals(df)
    results = run_backtest(df, signals, config)
    m = compute_metrics(results)
    print(f\"{label} ({start} to {end}): Return={m['total_return_pct']:.1f}%  Sharpe={m['sharpe_ratio']:.2f}  Trades={m['total_trades']}  LongWR={m['long_win_rate_pct']:.1f}%  ShortWR={m['short_win_rate_pct']:.1f}%\")
"
```

#### Test 2: Walk-Forward Validation

**Primary method (single split):** Train on first 60%, test on remaining 40%. Out-of-sample Sharpe must be > 50% of in-sample Sharpe.

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from backtester import load_candles, BacktestConfig, run_backtest, compute_metrics
from backtester.sweep import run_sweep
from strategies.{name}_strategy import {ClassName}

df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='{tf}')
split = int(len(df) * 0.6)
df_train = df.iloc[:split]
df_test = df.iloc[split:]

# In-sample: find best params
config = BacktestConfig(long_only={long_only})
grid = {ClassName}().param_grid()
sweep = run_sweep({ClassName}, df_train, grid, config, rank_by='sharpe_ratio', top=1)
best_params = sweep['results'][0]['params']
is_sharpe = sweep['results'][0]['metrics']['sharpe_ratio']

# Out-of-sample: test those params
strategy = {ClassName}(**best_params)
signals = strategy.generate_signals(df_test)
results = run_backtest(df_test, signals, config)
oos_m = compute_metrics(results)

print(f'In-sample Sharpe:  {is_sharpe:.3f}')
print(f'Out-of-sample Sharpe: {oos_m[\"sharpe_ratio\"]:.3f}')
ratio = oos_m['sharpe_ratio'] / is_sharpe if is_sharpe != 0 else 0
print(f'OOS/IS ratio: {ratio:.1%}')
print(f'Walk-Forward Efficiency: {\"PASS\" if oos_m[\"sharpe_ratio\"] > 0 and ratio > 0.5 else \"FAIL\"}')
"
```

**Enhanced method (rolling multi-window):** When data spans 3+ years, use rolling walk-forward with multiple segments. This detects strategy decay and is more statistically robust.

**Anchored vs Rolling walk-forward:** Anchored (expanding window) uses all historical data from the origin; rolling (fixed window) discards the oldest data. **For crypto, default to rolling.** Crypto market microstructure shifts frequently (exchange fee changes, new participants, regulatory shifts, DeFi/ETF regime shifts). Anchored windows dilute the signal from the current regime with stale data. Use anchored only when the strategy exploits a structural edge unlikely to change (e.g., funding rate mechanics).

```
Rolling WF configuration (3+ years of data):
  IS window: 9 months (crypto-optimized — captures current regime without overfitting)
  OOS window: 1 month
  Step: 1 month (rolling, not anchored)
  Segments: ~24-28 for 3-year dataset
  Recency weighting: within each IS window, apply exponential decay with 120-day half-life

Rolling WF configuration (5+ years of data):
  IS window: 12-18 months
  OOS window: 3 months
  Step: 3 months
  Segments: ~12-16

For each segment:
  1. Optimize on IS window -> find best params
  2. Apply those params to OOS window -> compute OOS Sharpe
  3. Record WFER_i = OOS_Sharpe_i / IS_Sharpe_i
  4. Track whether optimal params are stable across segments

Aggregation:
  - WFER_mean = mean of all segment WFERs
  - Concatenate all OOS return series -> compute overall OOS metrics
  - Check for slope: if WFER declines across successive segments -> strategy is decaying
  - Compute 95% CI of WFER across segments: if lower bound < 0.30 -> high instability

WFER thresholds:
  >= 0.60 -> PASS
  0.40-0.60 -> MARGINAL (strategy may need simpler parameters)
  < 0.40 -> FAIL (heavy overfitting)
```

**Strategy decay detection:** If the rolling WFER is declining across segments (fit a linear regression, slope significantly negative), the strategy's edge is eroding. Flag this even if overall WFER passes.

#### Test 3: Multi-Timeframe Check

If optimized on 1h, also test on 4h. Sharpe should remain the same sign.

```bash
python3 run_backtest.py --strategy {name} --resample 4h {mode_flag} --params '{json_params}'
```

#### Test 4: Parameter Neighborhood Stability

The 8 nearest neighbors (+/-1 step per parameter) should mostly be profitable.

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from backtester import load_candles, BacktestConfig, run_backtest, compute_metrics
from strategies.{name}_strategy import {ClassName}

df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='{tf}')
config = BacktestConfig(long_only={long_only})

# Define center and step sizes
center = {best_params}
steps = {step_sizes}  # e.g. {'period': 1, 'overbought': 5, 'oversold': 5}

neighbors = []
for param, step in steps.items():
    for delta in [-step, step]:
        neighbor = dict(center)
        neighbor[param] = center[param] + delta
        neighbors.append(neighbor)

profitable = 0
for params in neighbors:
    strategy = {ClassName}(**params)
    signals = strategy.generate_signals(df)
    results = run_backtest(df, signals, config)
    m = compute_metrics(results)
    status = 'PROFIT' if m['total_return_pct'] > 0 else 'LOSS'
    profitable += 1 if status == 'PROFIT' else 0
    print(f\"{params} -> Sharpe={m['sharpe_ratio']:.2f} Return={m['total_return_pct']:.1f}% [{status}]\")

print(f'\\nNeighborhood: {profitable}/{len(neighbors)} profitable')
print('PASS' if profitable >= 5 else 'FAIL')
"
```

#### Test 5: Monte Carlo Trade Shuffling

Shuffle the order of trades to assess if the observed max drawdown was "lucky" or representative.

```bash
python3 -c "
import sys, random; sys.path.insert(0, '.')
from backtester import load_candles, BacktestConfig, run_backtest, compute_metrics
from strategies.{name}_strategy import {ClassName}
import numpy as np

df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='{tf}')
config = BacktestConfig(long_only={long_only})
strategy = {ClassName}({params})
signals = strategy.generate_signals(df)
results = run_backtest(df, signals, config)
m = compute_metrics(results)

# Extract trade PnLs
pnls = [t['pnl_pct'] / 100 for t in results['trades']]
observed_dd = m['max_drawdown_pct']

# Monte Carlo: shuffle trade order 5000 times
mc_drawdowns = []
for _ in range(5000):
    shuffled = list(pnls)
    random.shuffle(shuffled)
    equity = [1.0]
    for pnl in shuffled:
        equity.append(equity[-1] * (1 + pnl))
    eq = np.array(equity)
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    mc_drawdowns.append(dd)

mc_95 = sorted(mc_drawdowns)[int(0.05 * len(mc_drawdowns))]  # 5th percentile (worst)
print(f'Observed max drawdown: {observed_dd:.1f}%')
print(f'Monte Carlo 95th pctl worst drawdown: {mc_95:.1f}%')
print(f'Realistic worst-case drawdown: {mc_95:.1f}%')
print(f'Observed vs worst-case ratio: {observed_dd/mc_95:.2f}' if mc_95 != 0 else '')
"
```

**Use the Monte Carlo 95th percentile drawdown as the "realistic worst-case" for risk assessment.** If the observed drawdown was near the 5th percentile of the Monte Carlo distribution (i.e., unusually favorable), future drawdowns will likely be worse.

**Important:** Use **block bootstrap** (not i.i.d. bootstrap) for time series data. Trade returns may have autocorrelation and clustering that i.i.d. shuffling destroys. Block size = `sqrt(n_trades)` is a reasonable default (e.g., for 400 trades, block size ~20).

#### Test 5b: Bootstrap Confidence Intervals

Compute 95% CIs for Sharpe and max drawdown to quantify estimation uncertainty:

```python
# Block bootstrap for Sharpe CI
import numpy as np
trade_returns = [t['pnl_pct'] / 100 for t in results['trades']]
n = len(trade_returns)
block_size = max(1, int(np.sqrt(n)))
n_bootstrap = 10000
sharpe_samples = []
for _ in range(n_bootstrap):
    # Block bootstrap: sample blocks with replacement
    blocks = [trade_returns[i:i+block_size] for i in range(0, n - block_size + 1)]
    sampled_blocks = [blocks[np.random.randint(len(blocks))] for _ in range(n // block_size + 1)]
    sample = [r for block in sampled_blocks for r in block][:n]
    sr = np.mean(sample) / np.std(sample) * np.sqrt(bars_per_year / avg_bars_per_trade) if np.std(sample) > 0 else 0
    sharpe_samples.append(sr)
ci_lower = np.percentile(sharpe_samples, 2.5)
ci_upper = np.percentile(sharpe_samples, 97.5)
# If ci_lower < 0.5: strategy is not confidently viable
```

**Decision rule:** If the Sharpe 95% CI lower bound is below 0.5, the strategy should not be considered reliable for deployment. Report CIs alongside point estimates in Phase 8.

#### Test 6: Fee Sensitivity (Three-Level Test)

Test at three fee levels to understand the strategy's cost sensitivity profile:

```bash
# Level 1: Optimistic (maker fees, minimal slippage — best case)
python3 run_backtest.py --strategy {name} --resample {tf} {mode_flag} --params '{json_params}' --fee 0.0003 --slippage 0.0002

# Level 2: Realistic (taker fees + median spread + base slippage)
python3 run_backtest.py --strategy {name} --resample {tf} {mode_flag} --params '{json_params}' --fee 0.001 --slippage 0.0005

# Level 3: Pessimistic (taker fees + wide spread + 2x slippage — stress test)
python3 run_backtest.py --strategy {name} --resample {tf} {mode_flag} --params '{json_params}' --fee 0.002 --slippage 0.001
```

**Strategy must be profitable under Level 2 (realistic) to pass.** If only profitable under Level 1 (optimistic), the strategy requires maker-only execution and carries significant implementation risk.

**Breakeven fee analysis (for strategies that pass Level 2):** Sweep fee rates to find where the strategy breaks even:

```python
# Sweep fees from 0 to 50 bps to find breakeven
for fee_bps in [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
    fee = fee_bps / 10000
    # Run backtest with this fee, compute Sharpe
    # Plot Sharpe vs fee_bps
# The x-intercept (Sharpe = 0) is the breakeven fee rate
```

**Breakeven interpretation:**

| Breakeven Fee | Assessment |
|--------------|-----------|
| < 5 bps | Extremely fee-sensitive. Not viable for spot. Requires maker rebates. |
| 5-15 bps | Viable only with low-fee tiers or maker execution |
| 15-30 bps | Viable on most major exchanges at standard rates |
| > 50 bps | Robust to transaction costs — strong edge |

**Holding period cost check:** Strategies with short holding periods are disproportionately fee-sensitive. At 25 bps round-trip cost: 1-day avg hold = ~91% annual fee drag; 1-week avg hold = ~13%; 1-month avg hold = ~3%. If `avg_bars_held * bar_duration < 3 days` and breakeven fee < 15 bps, the strategy is likely not viable at retail fee tiers.

#### Test 7: Cross-Symbol Validation (Optional but Recommended)

Apply the BTC-optimized parameters **without re-tuning** to ETH and optionally SOL. This tests structural robustness — whether the signal captures a genuine market pattern vs. BTC-specific noise.

```bash
# Fetch ETH data if needed
python3 fetch_binance_klines.py --symbol ETHUSDT --interval 1m --years-back 3

# Run with BTC-optimized params on ETH
python3 run_backtest.py --strategy {name} --resample {tf} {mode_flag} --params '{json_params}' --data data/binance_ETHUSDT_1m_klines.csv
```

**Cross-symbol thresholds:**
- Positive Sharpe on 2 of 3 symbols (BTC + ETH + SOL) without re-optimization -> strong structural signal
- Positive Sharpe on BTC only -> may be BTC-specific artifact; note concern
- If re-optimization per symbol produces wildly different optimal parameters (e.g., RSI period 14 for BTC vs 7 for SOL) -> strategy is not generalizable

**Note:** Cross-symbol tests are complementary to, not a replacement for, walk-forward testing. Walk-forward tests temporal robustness; cross-symbol tests structural robustness.

#### Test 8: Combinatorial Purged Cross-Validation (CPCV) — Advanced

CPCV generates multiple out-of-sample paths from the same dataset, giving a **distribution** of OOS performance instead of a single walk-forward estimate. Use this for strategies with 3+ free parameters or when the backtest Sharpe is suspiciously high (> 3.0).

**Algorithm overview:**
1. Divide the dataset into N sequential, non-overlapping groups (preserve chronological order).
2. Select k groups as the test set. Remaining N-k groups form training.
3. Enumerate all C(N,k) combinations. For each, purge training observations whose indicator lookback overlaps with test boundaries, and embargo a gap after each test boundary.
4. Train (optimize) and evaluate on each split.
5. Stitch test segments into complete backtest paths (paths = k * C(N,k) / N).

**Recommended configurations:**

| Setting | N | k | Splits | Paths | Use Case |
|---------|---|---|--------|-------|----------|
| Quick | 6 | 2 | 15 | 5 | Initial robustness check |
| Standard | 10 | 2 | 45 | 9 | Production validation |
| Thorough | 10 | 3 | 120 | 36 | ML-heavy or high-stakes |

**Purge and embargo by timeframe:**

| Timeframe | Purge (bars) | Embargo (bars) | Rationale |
|-----------|-------------|---------------|-----------|
| 1h | 24-48 | 12-24 | 1-2 day indicator bleed-through |
| 4h | 12 | 6 | ~2 day buffer |
| 1D | 5-10 | 2-5 | ~1-2 week buffer |

**Interpretation:**
- Median OOS Sharpe > 0.5 AND 10th-percentile OOS Sharpe > 0 AND 80%+ paths profitable -> **PASS**
- Median OOS Sharpe > 0 but 10th percentile < 0 -> **MARGINAL** (wide distribution = regime-dependent)
- Median OOS Sharpe < 0 -> **FAIL**

**Probability of Backtest Overfitting (PBO):** After a parameter sweep, partition the performance matrix into S equal subsets (S=8 or S=16, must be even). For all C(S,S/2) symmetric IS/OOS splits, check if the IS-best configuration ranks below median OOS. PBO = fraction of splits where this happens.

| PBO | Interpretation |
|-----|---------------|
| < 0.15 | Low overfitting risk — selection process reliable |
| 0.15-0.30 | Moderate — proceed with caution |
| 0.30-0.50 | High — selection is near-random; reduce parameter grid |
| > 0.50 | Selection is worse than random — the sweep is overfitting |

**When CPCV is essential:** 3+ free parameters, 20+ combos tested, Sharpe > 3.0, or planning to deploy real capital. **When it's overkill:** 0-1 free parameters, pure research/exploration, or < 500 bars at target timeframe.

**Decision gate after Phase 5:**

| Outcome | Criteria | Action |
|---------|----------|--------|
| **Pass** | Profitable in 2+ sub-periods, walk-forward OOS/IS > 50%, 5+ neighbors profitable, MC drawdown reasonable, profitable at 2x fees | Proceed to Phase 6 |
| **Marginal** | Passes some but not all tests | Present with heavy caveats |
| **Fail** | Fails walk-forward OR < 3 neighbors profitable OR unprofitable at 2x fees | Reject. In loop mode: mark FAIL on Leaderboard and derive next strategy. |

---

### Phase 6 — Benchmark Comparison & Context

**Goal:** Put the strategy in context against passive investing.

**Steps:**

1. Compute buy-and-hold metrics over the same period:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from backtester import load_candles, BacktestConfig, run_backtest, compute_metrics
import pandas as pd

df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='{tf}')
# Buy-and-hold = always long
bh_signals = pd.Series(1, index=df.index, dtype='int8')
results = run_backtest(df, bh_signals, BacktestConfig(fee_rate=0, slippage_pct=0))
m = compute_metrics(results)
print(f\"Buy-and-Hold: Return={m['total_return_pct']:.1f}%  Sharpe={m['sharpe_ratio']:.2f}  MaxDD={m['max_drawdown_pct']:.1f}%\")
"
```

2. Compare:
   - **Alpha** = strategy return - buy-and-hold return
   - **Risk-adjusted**: strategy Sharpe vs buy-and-hold Sharpe
   - **Time in market**: strategy may return less but with much less exposure
   - **Drawdown advantage**: if strategy MaxDD is significantly less than B&H MaxDD, that is a meaningful risk reduction

3. Risk-of-ruin check:
   - From the strategy's trade list, find the max consecutive losses
   - Calculate: if you hit 5x max consecutive losses from starting capital, do you survive?

4. **Equity curve shape analysis** — beyond aggregate metrics:

```python
# R-squared: how linear is the equity growth? (1.0 = perfectly consistent)
import numpy as np
equity = results['equity_curve'].values
x = np.arange(len(equity))
slope, intercept = np.polyfit(x, np.log(equity), 1)
predicted = intercept + slope * x
ss_res = np.sum((np.log(equity) - predicted) ** 2)
ss_tot = np.sum((np.log(equity) - np.mean(np.log(equity))) ** 2)
r_squared = 1 - (ss_res / ss_tot)
# R² > 0.90: excellent consistency
# R² 0.75-0.90: acceptable
# R² < 0.75: concerning — regime-dependent returns

# Ulcer Index: penalizes depth AND duration of drawdowns
cummax = equity.cummax() if hasattr(equity, 'cummax') else np.maximum.accumulate(equity)
drawdown_pct = ((equity - cummax) / cummax) * 100
ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))
# UI < 5%: excellent | 5-10%: good | 10-15%: acceptable | > 15%: flag concern

# Strategy death detection: rolling 90-day Sharpe near zero for 2+ windows?
rolling_sharpe = returns.rolling(90 * bars_per_day).apply(
    lambda x: x.mean() / x.std() * np.sqrt(bars_per_year) if x.std() > 0 else 0
)
# If the most recent rolling_sharpe < 0.1: strategy may have stopped working
```

   **Equity curve shapes to watch for:**
   - **Smooth upward:** Target shape. Consistent small wins, stable slope.
   - **Step-function:** Flat for long periods, sharp jumps. Too few independent wins for statistical confidence.
   - **Plateau/death:** Curve rises then flatters. Edge no longer exists. Check when the last equity high was set.

5. **Alternative risk metrics** — these capture aspects that Sharpe misses, especially with crypto's fat-tailed returns (BTC excess kurtosis ~6-10):

```python
# CVaR (Conditional Value at Risk) at 95%: average loss in the worst 5% of days
daily_returns = results['strategy_returns']
var_95 = daily_returns.quantile(0.05)
cvar_95 = daily_returns[daily_returns <= var_95].mean()
# Interpretation: "In the worst 5% of days, average loss is X%"
# Report as annualized: daily CVaR * sqrt(bars_per_year) for scale comparison

# Tail Ratio: asymmetry of upside vs downside tails
tail_ratio = abs(daily_returns.quantile(0.95)) / abs(daily_returns.quantile(0.05))
# > 1.0: positive asymmetry (upside larger than downside) — desirable
# 0.8-1.0: near-symmetric — acceptable
# < 0.5: severe negative asymmetry — disqualifying red flag

# Gain-to-Pain Ratio (Schwager): total return / total pain
gpr = daily_returns.sum() / abs(daily_returns[daily_returns < 0].sum())
# > 1.5: very good | 1.0-1.5: good | 0.5-1.0: mediocre | < 0.5: poor
```

   **Report these alongside Sharpe/Sortino.** Sharpe is unreliable for fat-tailed distributions because variance underestimates tail risk. CVaR directly addresses this. Tail Ratio is a one-number asymmetry diagnostic. GPR normalizes by actual pain rather than variance.

---

### Phase 7 — Regime Analysis

**Goal:** Understand when the strategy works and when it doesn't.

**Steps:**

1. Generate regime labels for the full test period using at least volatility percentile method:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from backtester import load_candles, BacktestConfig, run_backtest, compute_metrics
from strategies.{name}_strategy import {ClassName}
import numpy as np

df = load_candles('data/binance_BTCUSDT_1m_klines.csv', resample='{tf}')
config = BacktestConfig(long_only={long_only})
strategy = {ClassName}({params})
signals = strategy.generate_signals(df)
results = run_backtest(df, signals, config)

# Rolling volatility regime
returns = df['close'].pct_change()
vol_20 = returns.rolling(20 * 24).std()  # adjust multiplier for timeframe
vol_pct = vol_20.rolling(252 * 24).rank(pct=True)

# Assign regimes
regime = vol_pct.copy()
regime[vol_pct < 0.33] = 'low_vol'
regime[(vol_pct >= 0.33) & (vol_pct <= 0.67)] = 'med_vol'
regime[vol_pct > 0.67] = 'high_vol'

# Per-regime metrics
for r_name in ['low_vol', 'med_vol', 'high_vol']:
    mask = regime == r_name
    if mask.sum() < 20:
        continue
    r_returns = results['strategy_returns'][mask]
    pct_time = mask.mean() * 100
    r_sharpe = r_returns.mean() / r_returns.std() * np.sqrt(8760) if r_returns.std() > 0 else 0
    r_total = ((1 + r_returns).prod() - 1) * 100
    print(f'{r_name} ({pct_time:.0f}% of time): Return={r_total:.1f}%  Sharpe≈{r_sharpe:.2f}')
"
```

2. Assess regime dependency:
   - If strategy is profitable in all 3 regimes -> strong
   - If profitable in 2/3 -> acceptable, note the weak regime
   - If profitable in only 1/3 -> regime-dependent, flag heavily
   - If unprofitable in a regime that occurs > 30% of the time -> significant concern

3. Report findings for the leaderboard `RegimeConsistency` score.

---

### Phase 8a — Performance Attribution

Before the final recommendation, decompose returns to understand where alpha comes from:

#### Alpha vs Beta Decomposition

Run OLS regression: `R_strategy(t) = alpha + beta * R_BTC(t) + epsilon(t)`

Use Newey-West standard errors (5-10 lags for daily data) — naive OLS standard errors are inflated by BTC's fat tails and heteroskedasticity. Roughly half of strategies with significant alpha under naive OLS lose significance under robust errors.

**Alpha interpretation:** Annualize daily alpha by multiplying by 365. Most systematic BTC strategies produce -5% to +15% annualized alpha. Alpha > 20% should be viewed with extreme suspicion (likely overfitting or short favorable window).

**Beta interpretation:**
- β = 0.0: Market-neutral | β = 0.3-0.7: Partially directional (typical for active strategies)
- β = 1.0: Essentially buy-and-hold | β > 1.0: Leveraged/momentum-chasing

**Significance threshold:** t_alpha > 2.0 (95% confidence). Report both naive and Newey-West t-statistics.

#### Multi-Factor Attribution

Extend to three factors to isolate residual alpha:

```
R_strategy(t) = alpha + beta_BTC * R_BTC(t) + beta_mom * MOM(t) + beta_vol * VOL(t) + epsilon(t)
```

Where MOM(t) = BTC momentum factor (long if 20-day return > 0, short otherwise), VOL(t) = daily change in 30-day realized vol.

**Interpreting factor loadings:**
- beta_mom > 0: Strategy benefits from trend continuation (trend-following DNA)
- beta_mom < 0: Strategy benefits from mean-reversion
- beta_vol > 0: Long-volatility bias | beta_vol < 0: Short-vol bias (dangerous tail risk)

**R-squared ranges:** R² < 0.10 = highly idiosyncratic (genuine alpha or noise); R² 0.10-0.40 = moderate factor exposure; R² > 0.40 = largely replicable by factor exposure (limited unique alpha).

#### Timing Skill Assessment

**Treynor-Mazuy test:** Add `gamma * (R_BTC - R_f)²` term. Gamma > 0 = timing skill; gamma < 0 = anti-timing (common in trend-followers due to entry/exit lag). Requires 2+ years of daily data for significance.

**Henriksson-Merton:** `Timing Score = P(long when BTC up) + P(flat when BTC down) - 1.0`. Score > 0.05 is noteworthy. Typical for trend-followers: -0.05 to +0.10.

#### Regime Attribution

Segment all periods by regime, compute performance in each:

**Regime Herfindahl Index (HHI):** Sum of squared contribution fractions across regimes.
- HHI < 0.25: Well-diversified across regimes (excellent)
- HHI 0.25-0.40: Moderate concentration (acceptable)
- HHI > 0.40: High concentration (fragile — flag for review)

**Critical threshold:** If >60% of cumulative return comes from a single regime, the strategy is a disguised regime bet.

#### Signal Decomposition

**Entry contribution test:** Keep original exits, randomize entries (1000 runs). If random-entry variant captures >80% of original return → exit signal does most work (typical for trend-following). **Exit contribution test:** Keep original entries, randomize exits. If random-exit variant captures >80% → entry signal does most work (typical for mean-reversion).

**Filter contribution:** For each filter, compute `Filter_contribution = Sharpe(all_filters) - Sharpe(all_except_this)`. If removing a filter changes Sharpe < 0.05 → dead weight (remove). If removing improves Sharpe → actively harmful (must remove).

#### Information Coefficient (IC) & Information Ratio (IR)

`IC = Correlation(Forecast_signal, Actual_return)` over forecast horizon.

**Expected IC for BTC strategies:**
- IC < 0.01: No predictive power | IC 0.01-0.03: Weak but exploitable at high frequency
- IC 0.03-0.07: Moderate (typical decent strategy) | IC 0.07-0.15: Strong (exceptional)
- IC > 0.15: Almost certainly overfitted

**Fundamental Law of Active Management:** `IR = IC * sqrt(Breadth)` where Breadth = effective independent trades/year.

**IR interpretation:** < 0.2 = poor; 0.2-0.5 = below average; 0.5-0.75 = average professional; > 1.0 = excellent (scrutinize).

#### Attribution Red Flags (Auto-Detect in Phase 8)

| Red Flag | Threshold | Action |
|---|---|---|
| Concentrated alpha | >50% of alpha from single month | Remove best month, recompute |
| Negative timing (gamma < 0, significant) | t > 2.0 | Structural flaw, not bad luck |
| Alpha vanishes under robust SEs | Newey-West t < 1.65 | Factor exposure, not skill |
| High factor R² | > 0.50 | Replicable by simple exposure |
| Regime concentration | HHI > 0.40 | Disguised regime bet |
| IC below threshold | < 0.01 | No measurable predictive power |
| Slippage destroys edge | Slippage > 30% of avg trade profit | Not viable live |
| Sharpe in worst regime | < -1.0 | Severe vulnerability |

#### Key Attribution Formulas Summary

| Metric | Formula | Good Range for BTC |
|---|---|---|
| Alpha (annualized) | intercept × 365 | > 5% with t > 2.0 |
| Beta | Regression slope | 0.2-0.8 (active) |
| Treynor-Mazuy gamma | Coefficient on (R_BTC)² | > 0 (rare) |
| HM Timing Score | P(up) + P(down) - 1 | > 0.05 |
| Regime HHI | Σ(contribution²) | < 0.30 |
| IC | corr(forecast, actual) | 0.03-0.07 |
| IR | IC × √(breadth) | > 0.5 |

---

### Phase 8 — Final Recommendation

Present findings in this exact format:

```
## Recommendation

**Strategy**: {name}
**Best Parameters**: {params}
**Mode**: {long-only / long+short}
**Timeframe**: {resample}

### Performance (full period)
- Return: X% | Annualized: X%
- Sharpe: X | Sortino: X
- Max Drawdown: X% | Duration: X days
- Win Rate: X% | Profit Factor: X
- Total Trades: N | Time in Market: X%
- Long trades: N (win rate X%) | Short trades: N (win rate X%)

### Statistical Validity
- Deflated Sharpe Ratio: X (N={total_combos_tested})
- Sharpe 95% CI: [X, Y] (bootstrap, lower bound must be > 0.5 for deployment)
- MinTRL estimate: X years (given strategy's skewness and kurtosis)
- Trade concentration: top 10% of trades account for X% of returns
- Max consecutive losses: N
- Trades per parameter: N (min 30 per free param for reliability)

### Equity Curve Quality
- R-squared: X (>0.90 excellent, 0.75-0.90 acceptable)
- Ulcer Index: X% (<10% good, >15% concern)
- Strategy alive: {yes/no — last equity high set within recent 20% of bars}

### Robustness Evidence
- Sub-period: {results per window}
- Walk-forward: in-sample Sharpe X, out-of-sample Sharpe X (WFE ratio: X%)
- Parameter stability: {N}/8 neighbors profitable
- Monte Carlo 95th pctl max drawdown: X% (observed: X%)
- Fee sensitivity: {profitable / unprofitable} at 2x fees
- vs Buy-and-Hold: strategy X% vs B&H X% (alpha: X%)

### Regime Analysis
- Low volatility: Sharpe X, Return X%
- Medium volatility: Sharpe X, Return X%
- High volatility: Sharpe X, Return X%
- Regime consistency: X/3 regimes profitable

### Alternative Risk Metrics
- CVaR (95%): X% ("In the worst 5% of days, avg loss is X%")
- Tail Ratio: X (>1.0 good, <0.5 disqualifying)
- Gain-to-Pain Ratio: X (>1.5 very good, <0.5 poor)

### Backtest-to-Live Expectation
- Optimization level: {minimal / moderate / heavy}
- Haircut factor: X% (see table below)
- Expected live Sharpe: X (backtested Sharpe * (1 - haircut))
- Expected live MaxDD: X% (backtested MaxDD * 1.5x stress multiplier)

### Fee Sensitivity Profile
- Breakeven fee rate: X bps (>15 bps required for spot viability)
- Level 1 (optimistic): Sharpe X, Return X%
- Level 2 (realistic): Sharpe X, Return X%
- Level 3 (pessimistic): Sharpe X, Return X%

### Caveats
- {list any concerns: low trade count, marginal walk-forward, regime-dependent, narrow parameter optimum, return concentration, high fee sensitivity, CI too wide, MinTRL exceeded, etc.}

### Reproduction Command
python3 run_backtest.py --strategy {name} --resample {tf} {mode_flag} --params '{json}'
```

### Backtest-to-Live Haircut Table

Backtested performance always degrades in live trading due to execution slippage, market impact, data snooping decay, and parameter estimation error. Apply these haircuts before making deployment decisions:

| Optimization Level | Configs Tested | Haircut | Expected Live Sharpe |
|-------------------|----------------|---------|---------------------|
| Minimal (<5 configs, OOS validated) | < 5 | 30% | Backtest * 0.70 |
| Moderate (5-20 configs, walk-forward) | 5-20 | 40-50% | Backtest * 0.50-0.60 |
| Heavy (20-100 configs, grid search) | 20-100 | 50-65% | Backtest * 0.35-0.50 |
| Optimizer-driven (no OOS) | 100+ | 65-80% | Backtest * 0.20-0.35 |

**Hard rules:**
- If haircut-adjusted Sharpe < 0.5 -> do not recommend for deployment
- If backtested Sharpe > 3.0 without live paper-trading validation -> assume overfitting regardless
- Apply a 1.5x stress multiplier to backtested MaxDD for realistic worst-case expectations
- Also compute DSR: if DSR < 0.95, the Sharpe is not statistically significant after multiple testing

**In loop mode:** This is the per-strategy report. After convergence, produce the Final Report (see below).

---

### Loop Mode — Final Report

When convergence is triggered, produce this report:

```
## Optimization Loop — Final Report

### Convergence
- Reason: {which criterion triggered}
- Total strategies tested: N
- Total parameter combinations evaluated: N_total
- Strategy families explored: {list}

### Leaderboard
{full leaderboard table with DSR column}

### Derivation Genealogy
{derivation chain with Sharpe at each step}

### Winner
{Phase 8 format for the #1 strategy}

### Runner-Up
{Phase 8 format for the #2 strategy, if it passed robustness}

### Strategy Portfolio Analysis (if 3+ strategies passed robustness)
- Pairwise return correlation matrix between top strategies
- Estimated portfolio Sharpe: SR_portfolio = SR_avg * sqrt(n / (1 + (n-1) * avg_corr))
- Strategy approval check: only add strategy if SR_candidate > avg_corr * SR_portfolio_current
- Independent strategy groups: [list clusters with corr > 0.70 within each]
- Note: on a single asset (BTC), expect 2-4 truly independent strategy archetypes max

### Deployment Readiness
- Pre-deployment checklist: {X/8 items passed}
- Recommended sizing: {quarter/half Kelly} = {X}% per trade
- At recommended sizing: CAGR ~{X}%, MaxDD ~{X}%
- Circuit breakers: Yellow at {X}% DD, Orange at {X}%, Red at {X}%
- Estimated alpha half-life: {X} (based on strategy type)
- Strategy replacement planning: consider refreshing in {X} months

### Key Insights
- {What worked: which derivation methods improved results}
- {What didn't work: which strategies/methods failed and why}
- {Market structure observations}
- {Regime insights: which regimes favored which strategy types}
- {Recommendations for further exploration beyond the loop}

### Run History
- Run report: data/runs/run-{timestamp}.json
- Knowledge base: data/runs/KNOWLEDGE.md (updated)
- Regime classified as: {label}
- New anti-patterns: {list or "none"}
```

**After producing the Final Report**, immediately execute the Capture Protocol (see Run History & Cross-Run Learning). Save the run report JSON and update KNOWLEDGE.md before presenting results to the user.

---

## Multi-Strategy Ensemble Protocol

When the Optimization Loop produces 2+ validated strategies, evaluate whether combining them improves risk-adjusted returns. All strategies trade the same underlying (BTC), so diversification comes entirely from **signal diversification** — different entry/exit logic, different timeframes, or different market views.

### Signal Blending Methods

**1. Equal-Weight Average (default starting point):**
```
composite_signal = (signal_A + signal_B + ... + signal_N) / N
```
If composite > +0.3: long. If composite < -0.3: short. Otherwise: flat. The threshold prevents acting on weak consensus.

**2. Sharpe-Weighted Average (preferred when strategies differ in quality):**
```
weight_i = OOS_Sharpe_i / sum(OOS_Sharpe_j for all j)
composite_signal = sum(weight_i * signal_i)
```
Use out-of-sample Sharpe (not in-sample) to prevent overfitting the weighting.

**3. Carver Diversification Multiplier:** When averaging signals from imperfectly correlated strategies, the combined forecast has lower variance than any individual. Rescale by the diversification multiplier to restore target risk:

| Strategies | Avg Pairwise Corr | Multiplier |
|-----------|-------------------|-----------|
| 2 | 0.0 | 1.41 |
| 2 | 0.3 | 1.26 |
| 2 | 0.5 | 1.15 |
| 3 | 0.0 | 1.73 |
| 3 | 0.3 | 1.43 |

Apply the multiplier to the blended signal to restore the intended volatility target.

### Correlation Thresholds for Ensemble Benefit

**Minimum requirement:** A new strategy must have pairwise OOS return correlation < 0.3 with all existing ensemble members to justify inclusion. At correlation > 0.3, marginal variance reduction is < 6% for adding a 4th strategy to a 3-strategy ensemble.

**Measure correlations correctly:**
- Use out-of-sample daily P&L streams (not trade returns, not in-sample)
- Minimum 100 daily observations for a reliable estimate (60 bare minimum)
- Use Spearman rank correlation for robustness to BTC's non-normal returns
- Check rolling 30-day correlation to verify the relationship is stable, not regime-dependent

**Diminishing returns by strategy count:**

| Strategies | Variance Reduction (corr=0.0) | Variance Reduction (corr=0.3) |
|-----------|------|------|
| 2 | 50% | 35% |
| 3 | 67% | 47% |
| 4 | 75% | 53% |
| 5 | 80% | 56% |

At correlation 0.3, three strategies capture ~84% of the maximum achievable reduction. For a single asset (BTC), expect 2-4 truly independent strategy archetypes maximum. Do not add a 5th strategy unless it demonstrably reduces portfolio variance.

### Best Ensemble Candidates for BTC

These four strategy types have naturally low pairwise correlation because they profit from different market behaviors:

1. **Trend-following** (EMA crossover, MACD, Donchian breakout) — profits from persistent moves
2. **Mean-reversion** (RSI, Bollinger bounce) — profits from overextension snaps
3. **Volatility breakout** (BB squeeze, ATR expansion) — profits from compression-to-expansion
4. **Carry/funding rate** (funding rate contrarian) — profits from positioning imbalances

**Natural complementarity:** Momentum and mean-reversion on BTC are effectively anti-correlated in many regimes. Post-2021, mean-reversion outperformed; pre-2021, momentum dominated. A 50/50 blend of these two types has historically achieved better risk-adjusted returns than either alone.

### Handling Conflicting Signals

When strategy A says long and strategy B says short:
- **Net signal approach (recommended for single-asset):** The weighted sum determines direction. If net signal is near zero, stay flat. There is no benefit to holding offsetting long and short positions in the same asset.
- **Veto system (conservative):** Any strategy signaling strong opposite (signal magnitude > 0.7) vetoes the entry. Good for drawdown control.
- Size the **net signal**, not individual strategy positions. This reduces transaction costs and avoids offsetting trades.

### Ensemble Validation Protocol

1. Develop each strategy independently via walk-forward validation
2. Measure pairwise correlations on out-of-sample periods only
3. Combine strategies and run ensemble backtest on fully out-of-sample data
4. Verify: ensemble Sharpe > best individual Sharpe by at least 0.2
5. Verify: max drawdown improvement > 15% vs best individual strategy
6. Stress test on known BTC crash events (March 2020, May 2021, Nov 2022, FTX collapse)
7. Check conditional correlation during stress: if all strategies lose simultaneously, the ensemble does not protect against tail events

### Regime-Conditional Deployment

Instead of switching strategies entirely based on regime (high overfitting risk), use regime as a **weight adjuster**:

```
If ADX > 25 (trending):     increase trend-following weight by 1.5x, reduce mean-reversion by 0.5x
If ADX < 20 (ranging):      increase mean-reversion weight by 1.5x, reduce trend-following by 0.5x
If 30d vol > 90th pctl:     reduce all position sizes by 50% (crisis filter)
```

This is more robust than binary switching because regime detection is always lagging and uncertain. Weight adjustment degrades gracefully when the regime detector is wrong; binary switching does not.

---

## Multi-Strategy Portfolio Construction

When the ensemble produces 2+ validated strategies across different families (e.g., trend-following + mean-reversion + carry), construct a formal portfolio with risk parity allocation, exposure limits, and portfolio-level circuit breakers. This goes beyond simple signal blending to full portfolio management.

### Risk Parity Allocation

Allocate capital inversely proportional to each strategy's realized volatility so each contributes equally to portfolio risk:

```
w_i = (1 / σ_i) / Σ(1 / σ_j)
```

Where σ_i = annualized realized volatility from the OOS period (60-day EWMA, λ=0.94).

**Example (3-Strategy BTC Portfolio):**

| Strategy | Annual Vol | Inverse Vol | Weight |
|---|---|---|---|
| 1H Trend-following | 40% | 2.50 | 8.6% |
| Daily Mean-reversion | 15% | 6.67 | 22.9% |
| Funding Rate Carry | 5% | 20.00 | 68.6% |

This is mathematically correct but carry-dominated. The interaction with regime-conditional allocation (below) rebalances toward active strategies when their expected alpha is higher.

### Net Exposure Aggregation

When multiple strategies independently signal on BTC, the portfolio's realized exposure is the weighted sum:

```
net_exposure = Σ(w_i × s_i × leverage_i)
gross_exposure = Σ(w_i × |s_i| × leverage_i)
```

**Gross Exposure Limits (BTC-specific):**

| Market Condition | Max Gross | Max Net Long | Max Net Short |
|---|---|---|---|
| Normal (realized vol <60%) | 1.5x NAV | +1.0x | -0.75x |
| Elevated vol (60-90%) | 1.0x NAV | +0.75x | -0.50x |
| Stress (>90%) | 0.5x NAV | +0.40x | -0.30x |
| Max DD alert triggered | 0.25x NAV | +0.25x | -0.15x |

**Long/short asymmetry:** Max net long > max net short because BTC has historically positive drift and short squeezes are faster and more violent than down moves.

**Scale-down enforcement:** When gross exceeds limit, apply `scale_factor = max_gross / current_gross` uniformly to all positions, preserving relative weights.

### Conditional Correlation (Normal vs. Stress)

Diversification benefit vanishes during crisis events. Plan for it:

| Strategy Pair | Normal Corr | March 2020 (Liquidity) | May 2021 (Slow DD) | Nov 2022 (FTX) |
|---|---|---|---|---|
| Trend vs. MR | -0.35 | +0.92 | +0.62 | +0.18 |
| Trend vs. Carry | +0.15 | +0.88 | +0.45 | -0.08 |
| MR vs. Carry | -0.10 | +0.95 | +0.72 | +0.55 |

**Lower-tail dependence (both strategies losing badly simultaneously):**
- Trend + MR: 0.35-0.50 (structurally opposed, but overwhelmed by extreme liquidity events)
- MR + Carry: 0.55-0.70 (both long-biased at bottoms)

**For stress testing, use a correlation matrix of +0.75 between all pairs when BTC moves >20% in 24h.**

**Portfolio vol impact:** Normal regime ~4.7% annualized. Stress regime (vols doubled, corr = +0.80): ~21.5% annualized — a 4.6x increase.

### Rebalancing Framework

**Hybrid trigger (recommended for BTC):**

| Trigger | Threshold | Frequency |
|---|---|---|
| Relative drift | 20% of target weight | Check daily |
| Time-based backup | Monthly (1st trading day) | Safety net |
| Volatility shock | Realized vol increases >50% in 10 days | Immediate |
| Portfolio drawdown >8% | Emergency rebalance | Immediate |

**Expected rebalancing frequency:** 4-8 events/year. Annual cost: $600-$1,200 on $100K (4-8% of gross alpha).

### Portfolio-Level Circuit Breakers

| Alert Level | Trigger | Action | Duration |
|---|---|---|---|
| Yellow | -6% portfolio DD from HWM | Reduce gross 20%, suspend MR entries | Until -3% from trough |
| Orange | -10% portfolio DD from HWM | All strategies half-size, 48h entry pause | Until +5% recovery |
| Red | -15% portfolio DD from HWM | Full shutdown, go to cash | Min. 10 calendar days |
| Heat 1.5x | Portfolio vol > 1.5x target | Scale highest-vol strategy -30% | Until heat < 1.2x |
| Heat 2.0x | Portfolio vol > 2.0x target | Activate Yellow protocol | Per Yellow protocol |

**Portfolio heat:** `heat = Σ(w_i × current_vol_i × |s_i|) / target_portfolio_vol`. Heat > 1.5x = consuming 50% more risk than planned.

**Priority order when signals conflict:** Red Alert > Heat 2.0 > Orange > Per-strategy breakers > Regime allocation > Risk parity > Rebalancing trigger.

### Regime-Conditional Allocation

Shift capital toward higher-alpha strategies using smooth ADX-based interpolation:

```
regime_score = tanh((ADX_14_daily - 25) / 8)    # ∈ [-1, +1]

trend_weight = 0.35 + 0.45 × regime_score
MR_weight    = 0.35 - 0.30 × regime_score
carry_weight = 0.30 - 0.15 × regime_score
```

| ADX | Regime Score | Trend | MR | Carry |
|---|---|---|---|---|
| 15 | -0.85 | 2.8% | 60.4% | 42.7% |
| 20 | -0.56 | 9.8% | 51.8% | 38.4% |
| 25 | 0.00 | 35.0% | 35.0% | 30.0% |
| 30 | +0.56 | 60.2% | 18.2% | 21.6% |
| 35 | +0.85 | 73.0% | 9.6% | 17.4% |

**Final blend:** `final_weight = α × regime_optimal + (1-α) × risk_parity`, where α = 0.5 (normal) or 0.8 (high-confidence, all indicators agree).

**ADX momentum overlay:** +2% trend weight per 1 pt/day ADX acceleration, capped ±10%.

**BTC regime persistence:** Once ADX > 30 on daily, BTC stays trending for median 22 days — justifying patience in trend-favoring allocations.

**Net benefit:** +2.5-4.0% additional risk-adjusted return minus ~1.6% transaction costs = +1.0-2.5% net improvement annually.

### Portfolio Construction Parameter Summary

| Parameter | Value | Justification |
|---|---|---|
| Vol lookback | 60-day EWMA, λ=0.94 | Reactivity-stability balance for BTC |
| Drift rebalancing trigger | 20% relative | Cost-benefit crossover |
| Time-based rebalancing | Monthly | Safety net for calm periods |
| Max gross exposure | 1.5x NAV (normal) | ~Half-Kelly for multi-strategy |
| Yellow Alert | -6% DD from HWM | Controlled deceleration |
| Red Alert | -15% DD from HWM | Capital preservation |
| Heat limit | 1.5x target vol | Forward-looking risk |
| Regime blend alpha | 0.5 / 0.8 | Preserve diversification |

---

## Decision Trees

### "Should I tune or pivot?"

```
Start -> Run broad sweep (Phase 2, both modes)
  |-- Best Sharpe > 0.3 (either mode) -> TUNE (proceed to Phase 3)
  |-- Best Sharpe 0 to 0.3 -> TUNE with caution
  |-- Best Sharpe -0.5 to 0 -> PIVOT to new strategy
  +-- Best Sharpe < -0.5 -> ABANDON immediately, PIVOT to new strategy
```

### "Which derivation method?"

```
Current best Sharpe > 0.3 but MaxDD > 30%?
  -> Method 1: Add filter (reduce drawdown)

Fine-grid improvement < 0.02 over broad?
  -> Method 2: Swap indicator (strategy plateaued)

2+ leaderboard entries with Sharpe > 0.2?
  -> Method 3: Combine strategies

Strategy untested at other resolutions?
  -> Method 4: Change timeframe

3 consecutive no-improvement derivations OR unexplored strategy family?
  -> Method 5: New concept (fresh approach from unexplored family)

High sub-period variance (strategy works great in some periods, poorly in others)?
  -> Method 6: Add regime filter

Narrow parameter optimum / cliff effects detected?
  -> Method 7: Adaptive parameters
```

### "Is this overfit?" Checklist

All twelve must be checked. If 6+ fail -> overfit. Do not recommend.

- [ ] 20+ trades in the backtest period
- [ ] At least 30 trades per free parameter (complexity check)
- [ ] Profitable in at least 2 of 3 sub-periods
- [ ] Walk-forward out-of-sample Sharpe > 0 (WFER > 0.50)
- [ ] At least 5 of 8 parameter neighbors also profitable
- [ ] Not the single best point in an otherwise losing grid (broad hill, not narrow spike)
- [ ] Deflated Sharpe Ratio > 0.90 (accounting for all combinations tested)
- [ ] Returns not concentrated in a few lucky trades (top 10% of trades < 60% of total return)
- [ ] Equity curve R-squared > 0.75 (returns are consistent, not step-function)
- [ ] Haircut-adjusted Sharpe > 0.5 (see Backtest-to-Live Haircut Table)
- [ ] Bootstrap 95% CI lower bound for Sharpe > 0.5 (estimation uncertainty acceptable)
- [ ] Profitable at realistic fee level (Level 2: taker fees + median spread + base slippage)

### "Long-only or bidirectional?"

```
Phase 2 sweep complete for both modes?
  |-- Bidirectional Sharpe > Long-only + 0.1 -> Bidirectional
  |-- Long-only Sharpe > Bidirectional + 0.1 -> Long-only
  |-- Gap <= 0.1 -> Long-only (simpler)
  +-- Both Sharpe > 0.3 -> Test both through remaining phases
```

---

## Advanced Statistical Validation

These tests supplement the existing DSR and walk-forward validation. They detect subtle issues that basic metrics miss: autocorrelation inflation, non-stationarity, trade clustering, and insufficient multiple testing correction.

### Lo's Autocorrelation-Adjusted Sharpe Ratio

Standard Sharpe is inflated when returns are serially correlated (common for trend-following strategies on BTC). Lo (2002) provides the correction:

```
adjusted_SR = observed_SR * sqrt(q / eta(q))

Where:
  q = number of return observations per year
  eta(q) = q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k
  rho_k = autocorrelation at lag k
```

**Practical shortcut:** If `rho_1` (lag-1 autocorrelation) is the dominant term:
```
eta(q) ≈ q * (1 + 2*rho_1)
adjusted_SR ≈ observed_SR / sqrt(1 + 2*rho_1)
```

**BTC typical values:** Strategy returns on 1h data often show `rho_1` = 0.05-0.15 for trend-following. This deflates Sharpe by 5-15%. Mean-reversion strategies often have `rho_1` < 0 (favorable — Sharpe is understated).

**Integration:** Run Ljung-Box test (`scipy.stats` or manually) on strategy returns at lags 1-10. If significant autocorrelation detected (p < 0.05), compute and report Lo's adjusted Sharpe alongside raw Sharpe in Phase 8.

### Augmented Dickey-Fuller (ADF) Test

Tests whether strategy returns are stationary (required for Sharpe to be meaningful):

```
H0: returns have a unit root (non-stationary)
H1: returns are stationary

ADF statistic thresholds (5% significance):
  < -3.43 (with constant+trend) → reject H0, returns ARE stationary: PASS
  > -3.43 → cannot reject H0: WARNING (returns may drift, Sharpe unreliable)
```

**BTC context:** Raw BTC log returns are stationary (ADF << -3.43). Strategy returns should also be stationary. If they are not, the strategy has a structural drift — likely regime dependency where edge appears/disappears over time.

**Integration:** Run on full-period strategy returns (should pass easily) AND on rolling 90-day windows. If any rolling window fails, the strategy's edge is time-varying — flag in Phase 7 regime analysis.

### Variance Ratio Test (Lo-MacKinlay)

Detects mean-reversion vs momentum at specific horizons:

```
VR(q) = Var(q-period returns) / (q * Var(1-period returns))

VR > 1 → momentum (returns persist) → favor trend-following
VR < 1 → mean-reversion (returns reverse) → favor mean-reversion
VR ≈ 1 → random walk (no structure) → no directional edge
```

**Test at multiple horizons:** q = 2, 5, 10, 20, 40. The horizon where VR deviates most from 1.0 indicates the optimal strategy timeframe.

**BTC empirical pattern:** VR > 1 at daily/weekly horizons (momentum), VR ≈ 1 at 1m-5m (near random walk). This is consistent with the Hurst exponent findings.

**Integration:** Compute in Phase 1 (Structural Assessment) alongside buy-and-hold. Use to confirm strategy type matches market structure at the chosen timeframe.

### Hansen's Superior Predictive Ability (SPA) Test

More rigorous than DSR for testing whether the best strategy from a sweep genuinely outperforms a benchmark:

```
H0: No strategy in the sweep outperforms the benchmark
H1: At least one strategy genuinely outperforms

Steps:
1. Compute excess returns: d_i = strategy_i_returns - benchmark_returns for all i
2. Compute test statistic: T_SPA = max_i(sqrt(n) * mean(d_i) / std(d_i))
3. Bootstrap the distribution of T_SPA under H0 (stationary bootstrap, 10,000 iterations)
4. p-value = fraction of bootstrap T_SPA values exceeding observed T_SPA

If p < 0.05: at least one strategy genuinely outperforms the benchmark
```

**Integration:** Run after Phase 5 on the sweep winner vs buy-and-hold. This is the definitive test for "does this strategy have an edge?" — more theoretically grounded than DSR for sweep validation.

### Runs Test (Trade Win/Loss Independence)

Tests whether wins and losses are independent or clustered:

```
Z < -1.96: Clustering (streaks of wins/losses) — regimes drive outcomes
Z > +1.96: Alternation (suspicious, check for look-ahead bias)
|Z| < 1.96: Independent — standard assumptions hold
```

**BTC-specific:** Trend-following strategies typically show Z = -2.0 to -4.0 (expected clustering from regime dependency). Mean-reversion shows Z = -0.5 to -2.0. Strong clustering is NOT a red flag for trend-following — it IS a red flag for strategies claiming to be regime-independent.

**Integration:** If clustering detected, use block bootstrap (block size = average streak length) for Monte Carlo simulations in Phase 5. I.i.d. Monte Carlo underestimates drawdowns by 30-50% when clustering is present.

### Harvey-Liu-Zhu (HLZ) Multiple Testing Hurdles

Minimum t-statistic required for the best strategy to be statistically significant:

| Strategies Tested (M) | t_min (Bonferroni) | t_min (BH-FDR) |
|---|---|---|
| 10 | 3.02 | 2.49 |
| 50 | 3.50 | 2.81 |
| 100 | 3.68 | 2.94 |
| 200 | 3.85 | 3.06 |
| 500 | 4.06 | 3.20 |

```
t(SR) = annualized_SR / sqrt(periods_per_year) * sqrt(n_observations)
Required: t(SR) > t_min(M)
```

**BTC advantage:** 5 years of hourly data = ~43,800 observations, which significantly lowers the required Sharpe vs traditional assets with only daily data. For 200 strategies tested on 3 years of hourly data with Bonferroni correction, the minimum annualized Sharpe is ~2.2.

**Integration:** Track cumulative M across all sweeps (broad + fine grid, both modes). Compare best strategy's t-statistic against HLZ hurdle. Report alongside DSR in the Leaderboard.

### Hurst Exponent Estimation (DFA Method)

DFA (Detrended Fluctuation Analysis) is preferred over R/S for BTC because it handles non-stationarity better and has lower bias:

```
1. Compute cumulative sum of demeaned returns
2. For each window size n (log-spaced from 10 to n/4):
   a. Divide into non-overlapping segments
   b. Linear detrend each segment
   c. Compute RMS of residuals: F(n)
3. H = slope of log(F(n)) vs log(n) regression
```

**Updated BTC Hurst table with DFA estimates and confidence intervals:**

| Timeframe | H (DFA) | 95% CI | Interpretation |
|---|---|---|---|
| 1m | 0.50 | [0.48, 0.52] | Random walk — no directional edge |
| 5m | 0.50 | [0.48, 0.53] | Random walk |
| 1h | 0.53 | [0.50, 0.56] | Slight persistence, near boundary |
| 4h | 0.57 | [0.53, 0.61] | Mild persistence — both strategies viable |
| 1D | 0.62 | [0.57, 0.67] | Moderate persistence — trend-following edge |
| 1W | 0.65 | [0.58, 0.72] | Strong persistence (wider CI due to fewer obs) |

**R/S estimates are consistently 0.02-0.03 higher** than DFA for BTC. Trust DFA as primary.

**H is time-varying:** During high-volatility regimes, H can spike to 0.7+ on daily data. During consolidation, H drops to 0.50-0.52. Use rolling 100-day window for regime-adaptive strategy selection.

### Phase 8 Statistical Validation Summary Template

Add this table to every Phase 8 recommendation:

```
### Statistical Validation Summary

| Test | Statistic | Result | Notes |
|------|-----------|--------|-------|
| Ljung-Box (m=10) | Q = X, p = Y | PASS/FAIL | rho_1 = Z |
| Lo's Adjusted Sharpe | X | — | Raw: Y, deflation: Z% |
| ADF (returns) | X, p = Y | PASS/WARN | Lags: N |
| Variance Ratio (q=5) | VR = X, p = Y | momentum/MR/RW | Strategy type match: Y/N |
| Jarque-Bera | skew=X, kurt=Y | Grade: Z | Preferred metric: Sortino/Sharpe |
| Runs Test | Z = X, p = Y | independent/clustering | MC block size: N |
| HLZ Hurdle (M=X) | t=Y vs t_min=Z | PASS/FAIL | Method: Bonferroni |
| Hansen's SPA | p = X | PASS/FAIL | vs buy-and-hold |
| Hurst (DFA) | H = X | 95% CI: [Y, Z] | Strategy match: Y/N |
```

### Non-Normality Awareness (Jarque-Bera)

BTC always rejects normality (excess kurtosis 6-10, skewness -1 to +1). The JB test p-value is uninformative — focus on the **magnitude** of skewness and kurtosis:

| Condition | Action |
|-----------|--------|
| \|Skewness\| < 0.5, Excess Kurt < 3 | Near-normal. All standard metrics valid. (Rare for BTC.) |
| \|Skewness\| 0.5-1.0, Excess Kurt 3-6 | Moderately non-normal. Prefer Sortino over Sharpe. Report CVaR. |
| \|Skewness\| > 1.0, Excess Kurt > 6 | Heavily non-normal. Sharpe unreliable. Use Sortino + CVaR + Tail Ratio. |
| Skewness < -1.5 | Extreme crash risk. Require explicit stop-loss or circuit breaker. |

---

## Walk-Forward Analysis & CPCV

Walk-forward validation is the gold standard for preventing overfitting in parameter optimization. It simulates the real experience of a trader who periodically re-optimizes on recent data and trades on unseen data. This section provides exact procedures, window configurations, and BTC-specific calibrations.

### Rolling Multi-Window Walk-Forward Procedure

1. **Define windows.** IS (in-sample) for parameter optimization, OOS (out-of-sample) for evaluation.
2. **Slice IS** via `load_candles(start, end)`, run `run_sweep()` to find optimal θ*_k.
3. **Record θ*_k** (the winning parameter set for segment k).
4. **Slice OOS**, run single backtest with θ*_k, record OOS metrics.
5. **Advance window** by one OOS length. Repeat until data exhausted.
6. **Concatenate OOS segments** for aggregate performance.

**Recommended Window Configurations (BTC):**

| Timeframe | IS Window | OOS Window | Min Segments |
|---|---|---|---|
| 4-hour (preferred) | 18 months | 3 months | 5 |
| Daily | 24 months | 6 months | 5 |
| 1-hour | 18 months | 3 months | 5 |

**Anchored vs. Rolling Decision:**
- Compute `param_drift = ||θ*_k - θ*_{k-1}|| / ||θ*_1||`. If mean > 0.30 → use rolling; if < 0.15 → use anchored.
- Pearson correlation of IS-optimal params to OOS Sharpe. If |r| > 0.5 → rolling; if |r| < 0.3 → anchored.
- Lag-1 autocorrelation of OOS Sharpe. If > 0.4 → rolling; if < 0 → anchored.

### Parameter Stability Analysis

**Parameter Concentration Ratio (PCR):** Fraction of WF segments where optimal param equals the modal value:

```
PCR_j = count(segments where θ*_{k,j} = mode_j) / N
```

| PCR | Interpretation | Action |
|---|---|---|
| > 0.70 | Stable | Lock value for production |
| 0.50 – 0.70 | Moderate | Accept with monitoring |
| 0.30 – 0.50 | Borderline | Widen sweep grid |
| < 0.30 | Unstable | Remove from optimization; use default |

**Stability lift test:** `stability_lift = OOS_Sharpe_when_mode_wins - OOS_Sharpe_when_other_wins`. If stability_lift < 0, the modal IS value does not predict OOS performance — PCR is meaningless.

**Three trajectory patterns:** (1) Gradual drift = regime tracking (acceptable, use rolling WF); (2) Random scatter = overfitting (remove parameter); (3) Bi-modal clustering = regime-conditional behavior (add explicit regime variable).

### Walk-Forward Efficiency Ratio (WFER)

The primary metric for quantifying out-of-sample retention:

```
WFER_k = Sharpe_OOS_k / Sharpe_IS_k    (per segment)
WFER_overall = geometric_mean(WFER_k)   (across all segments)
```

Use geometric mean, not arithmetic — prevents high outliers from masking consistently negative segments.

**WFER Thresholds:**

| WFER | Interpretation | Action |
|---|---|---|
| > 0.70 | Excellent | Deploy normally. Audit for look-ahead bias (suspiciously high). |
| 0.50 – 0.70 | Good | Deploy with 25% size reduction |
| 0.30 – 0.50 | Marginal | Paper trade first |
| < 0.10 | Failure | Discard strategy |

**BTC-specific baselines:** Expect WFER 0.35-0.55 for momentum, 0.50-0.65 for volatility-adjusted mean reversion. WFER > 0.70 warrants look-ahead bias audit.

**Declining WFER detection:**
```
WFER_k = β₀ + β₁ × t_k + ε_k    (OLS regression)
```
Threshold: β₁ < -0.03 per segment AND |t_stat| > 1.5 → significant edge decay. Use t = 1.5 (not 1.96) because false negatives are more costly with only 6-10 segments.

### CPCV (Combinatorial Purged Cross-Validation)

Lopez de Prado's approach for obtaining a distribution of OOS performance estimates rather than a single point:

**Procedure (single-asset BTC):**
1. Divide data into N=6 non-overlapping groups of equal length.
2. Set k=2 OOS groups per split → C(6,2) = 15 total splits.
3. Each group appears as OOS in exactly 5 of the 15 splits.
4. Apply **embargo** after each IS group to prevent leakage: `embargo_bars = ceil(max_indicator_lookback × 2.5)`. At 4h bars: EMA(200) → 500 bars (~83 days).
5. For each split: optimize on IS groups, test on OOS groups.
6. Construct non-overlapping paths (paths of 3 splits covering all 6 groups exactly once).
7. Compute path-level Sharpe for each path → distribution of 15 OOS Sharpe estimates.

**5-Criterion Robustness Check:**

| Criterion | Threshold | Purpose |
|---|---|---|
| μ_OOS > 0.50 | Mean OOS Sharpe sufficient | Overall viability |
| σ_OOS < 0.40 | OOS Sharpe not too dispersed | Consistency |
| Sharpe_p5 > 0.10 | 5th percentile positive | Worst-case tolerable |
| positive_rate > 0.75 | >75% of paths profitable | Broad viability |
| μ_OOS / IS_Sharpe > 0.40 | Sufficient OOS retention | Not overfit |

If all 5 pass, the strategy is considered robustly validated.

### Regime-Conditioned Walk-Forward

Standard WF can produce misleading results when IS and OOS windows have different regime compositions.

**Regime classifier (look-ahead-free, BTC-specific):**
- Bull: close > SMA(200) AND +5% in 20 bars
- Bear: close < SMA(200) AND -5% in 20 bars
- Sideways: all other

**Regime bias flag:** A fold is flagged when any single regime accounts for >70% of IS bars — the IS window cannot provide parameters that generalize.

**Three mitigation techniques:**
1. **Harmonic mean** of per-regime Sharpe as optimization objective (penalizes extreme failure in any regime).
2. **Regime-stratified bar weights:** `w_regime = N_IS / (3 × N_regime)` — downweights dominant regime, upweights minority.
3. **Regime-anchored window boundaries** — align IS window starts to regime transitions rather than calendar dates.

**Regime mismatch metric:** Regress `WFER_k` on `|f_bull_OOS - f_bull_IS| + |f_bear_OOS - f_bear_IS|`. If β < -0.20, regime mismatch is a primary WFER driver → implement a live regime filter.

**Extended WF segment table format:** Include columns for IS/OOS regime composition (% bull/bear/sideways) alongside the standard WFER and optimal parameters.

---

## Machine Learning Guardrails

ML can augment rule-based strategies when applied with extreme discipline. Without guardrails, ML overfits financial data far more easily than other domains due to low signal-to-noise ratios, non-stationarity, and autocorrelation. This section defines when to use ML, how to avoid its traps, and what additional validation is required.

### When ML Is Appropriate vs When It's Not

**Use ML only when ALL of the following are true:**
- Feature space is high-dimensional (>5 interacting features) with non-linear relationships
- Rule-based strategies have been exhausted first (completed the exhaustion checklist below)
- Sufficient data: ≥1,000 completed trades, ≥3 years of data, ≥50 observations per feature
- You can articulate a hypothesis for why ML should find something rules cannot

**Rule-based exhaustion checklist (complete before considering ML):**
1. Tested single-indicator strategies across multiple timeframes?
2. Tested complementary indicator combinations (trend + momentum + volatility)?
3. Tested regime-filtered versions of the above?
4. Tested order flow / microstructure signals?
5. If all produce OOS Sharpe < 0.5, ML is a reasonable next step

### The Overfitting Problem in Finance

| Factor | Image Recognition | BTC Trading |
|--------|------------------|-------------|
| Signal-to-noise ratio | ~90%+ signal | ~1-5% signal |
| Stationarity | Static | Regime changes every 6-18 months |
| Sample independence | Each image independent | Adjacent bars autocorrelated |
| Effective degrees of freedom | Millions | Hundreds (after autocorrelation) |

### Effective Sample Size (ESS)

10,000 1-minute bars ≠ 10,000 independent observations. BTC returns have autocorrelation and volatility clustering that reduce ESS dramatically:

| Raw Observations | Timeframe | Approximate ESS | Reduction |
|-----------------|-----------|-----------------|-----------|
| 2,628,000 | 1m (5 years) | ~500,000-800,000 | 3-5x |
| 43,800 | 1h (5 years) | ~15,000-25,000 | 2-3x |
| 1,825 | Daily (5 years) | ~400-600 | 3-4x |

### Model Complexity Budgets

```
Maximum features < sqrt(N_effective)
```

| Data Configuration | Approx ESS | Max Features | Max Model Parameters |
|-------------------|------------|-------------|---------------------|
| 5 years daily | ~500 | ~22 | ~50 (with regularization) |
| 5 years hourly | ~20,000 | ~141 | ~300 (with regularization) |
| 3 years daily | ~300 | ~17 | ~35 |

**Hyperparameters count as degrees of freedom too.** A grid search over 100 combos = 100 additional trials. Include all trials in DSR calculation.

### Recommended ML Approaches (Ranked by Overfitting Risk)

| Risk Level | Model | BTC Use Case | Constraints | When Appropriate |
|-----------|-------|-------------|-------------|-----------------|
| **Low** | Ridge/Lasso regression | Combine 5-15 pre-engineered features; feature selection | Built-in regularization | First ML model to try; ESS > 500 |
| **Medium** | Random Forest (limited depth) | Regime classification; non-linear interactions | max_depth=3-5, min_samples_leaf ≥ 50 | ESS > 2,000 |
| **High** | XGBoost/LightGBM | Hourly prediction with order flow + momentum + vol features | max_depth=2-3, learning_rate < 0.05, early stopping | ESS > 5,000; after simpler models tried |
| **Very High** | Neural networks / LSTM | Only with multi-year minute data + strong regularization | Dropout > 0.3, early stopping, ensemble of 3-5 | ESS > 50,000; last resort |

**Decision rule:** If ESS < 500, do not use ML. If ESS 500-2,000, use Ridge/Lasso only. If ESS 2,000-10,000, add Random Forest. If ESS > 10,000, cautiously consider gradient boosting.

### Feature Engineering Guidelines

**Best features for BTC ML (Tier 1):**
- Returns at multiple horizons (1h, 4h, 1d, 7d, 30d log returns)
- Volatility measures (rolling std 6h/24h/7d, ATR ratios, Garman-Klass)
- Volume ratios (current/rolling mean at 6h/24h, taker buy ratio)
- Momentum factors (ROC at 3/6/12/24 hours, RSI at 14/30 periods)

**Features to AVOID:**
- Raw prices (non-stationary; breaks when BTC moves to new range)
- Date/timestamp as integer (model learns "BTC went up in 2021")
- Indicators with many parameters (MACD(12,26,9) — each param is hidden optimization)
- Full-series z-scores (uses future data; use `.expanding()` instead)
- Highly correlated pairs (RSI + Stochastic; BB width + ATR)

**Feature importance stability test:** Train on each walk-forward fold separately. Compute permutation importance on test set (not training set). A feature is stable only if it ranks in top-K in ≥70% of folds. Features whose rank varies by >50% across folds → remove.

### Train/Test Contamination Prevention

**Standard k-fold cross-validation is WRONG for time series.** It inflates performance by 50-300%.

**Walk-forward validation is the correct approach:**
1. Train on initial window (e.g., months 1-12)
2. Test on next block (e.g., months 13-15)
3. Slide window forward. Repeat.
4. Minimum 3 folds for stability assessment

**Purging:** Remove training observations whose label period overlaps with the test set. If predicting 24h forward return, purge the last 24h of training before the test boundary.

**Embargoing:** Remove additional observations after the purge boundary. For hourly data: 24-48 bars. For daily: 5-10 bars.

### Common ML Traps in Crypto

| Trap | Description | Prevention |
|------|-------------|------------|
| **Survivorship bias in features** | Select best 20 of 200 features, then train/test — feature selection already used test data | Feature selection must happen INSIDE the CV loop |
| **Target leakage** | "24h rolling volume" to predict next hour — window extends into prediction period | Verify for every feature: "Would I have had this value at prediction time?" |
| **Non-stationarity** | Model trained on 2020-2022 learns relationships that break in 2024 (ETFs, microstructure changes) | Retrain monthly/quarterly; monitor prediction accuracy |
| **Regime invalidation** | Model averages across bull/bear/range, performs poorly in all | Consider regime-conditional models or regime as gating mechanism |
| **Hidden hyperparameter optimization** | "One model" with 500 hyperparameter combos = 500 effective trials | Include ALL trials (features × hyperparams × architectures) in DSR |

**Quantified risk:** With 500 trials and 3 years of daily data, the expected maximum Sharpe from pure noise is ~2.0 (Bailey & Lopez de Prado False Strategy Theorem). A Sharpe of 2.0 can emerge from nothing.

### ML-Specific Validation Requirements

Beyond standard Phase 5, ML strategies require:

| Requirement | Threshold |
|-------------|-----------|
| Walk-forward OOS Sharpe > 1.0 | Across all folds, not just the best |
| OOS/IS Sharpe ratio > 60% | ML commonly shows 90%+ IS, 20% OOS |
| Feature importance stability | Top features consistent in ≥70% of folds |
| Prediction calibration | Predicted probabilities match observed within 5% |
| DSR > 0 after ALL trials counted | Models + hyperparams + feature sets |
| ≥100 OOS trades total | Across all walk-forward folds |
| No single fold dominates returns | No fold > 50% of total OOS return |

### Integration with 8-Phase Protocol

ML strategies flow through the same 8 phases with additions:

| Phase | Additional ML Requirement |
|-------|--------------------------|
| Phase 1 | Compute ESS. If ESS < 500, abort ML. |
| Phase 2 | Walk-forward replaces grid sweep. Feature importance report per fold. |
| Phase 3 | Calibration check. Out-of-distribution detection. Feature stability (Spearman > 0.5). |
| Phase 4 | Hyperparameter budget: total trials ≤ 10x broad sweep count. |
| Phase 5 | Complexity budget verification. Model retraining stability. |
| Phase 6 | Compare vs naive forecast, vs best rule-based, vs random signal baseline. |
| Phase 8 | Add: feature importance report, calibration, complexity budget, total trials, alpha decay estimate. |

**When to add ML to the workflow:** Only after the best rule-based strategy has OOS Sharpe between 0.3 and 1.5 (below 0.3 = no edge to augment; above 1.5 = rule-based already strong), AND the user explicitly requests ML enhancement.

### ML Alpha Decay

ML strategies decay faster than rule-based in crypto:

| Strategy Type | Crypto Alpha Half-Life |
|--------------|----------------------|
| HF ML strategies | Days to weeks |
| Momentum ML | 3-6 months |
| Swing/position ML | 6-18 months |
| Macro/regime ML | 1-3 years |

**Monitor:** If rolling 60-day OOS Sharpe drops below 50% of initial OOS Sharpe, trigger retraining. If retraining fails to restore performance, retire the strategy.

---

## Proactive Analysis Suggestions

Offer these to the user when relevant:

1. **Regime-filtered backtest**: "Would you like me to test this strategy only during high-volatility regimes (ATR > 90th percentile) vs low-volatility?"

2. **Drawdown recovery analysis**: "The max drawdown lasted X days. Want me to analyze what market conditions caused it and whether an exit rule could shorten it?"

3. **Strategy combination**: "{Strategy} alone gives Sharpe X. Want me to test adding a trend filter (e.g., only take signals when price > 200 EMA)?"

4. **Alternative strategy**: When the current strategy fails, suggest alternatives based on market structure:
   - Trending market -> EMA crossover, MACD crossover, momentum factor
   - Range-bound -> RSI mean-reversion, Bollinger Band bounce
   - High volatility -> Bollinger Band squeeze/breakout, ATR breakout
   - Volatility compression -> Squeeze strategies (BB Width + MACD direction)

5. **Fee sensitivity**: "At 0.1% fees this strategy works. Want me to check what fee rate breaks it?"

6. **Equity curve analysis**: "Want me to check if returns are smooth or come from a few lucky trades?"

7. **Sub-minute exploration**: "Want me to test this strategy at sub-minute resolution (1s, 5s)? I'll fetch the data and run a focused sweep."

8. **Short-side isolation**: "The short side is dragging performance. Want me to isolate short trades and analyze what's going wrong?"

9. **Regime overlay**: "Want me to add a regime filter so the strategy only trades in favorable market conditions (e.g., only mean-revert when ADX < 20)?"

10. **Strategy decay check**: "This strategy has been optimized on historical data. Want me to check if its edge has been decaying over time using rolling Sharpe analysis?"

11. **Monte Carlo stress test**: "Want me to run 5000 Monte Carlo simulations to find the realistic worst-case drawdown for this strategy?"

12. **Cross-family exploration**: "We've only tested {family} strategies so far. BTC has documented edge in {unexplored_family} — want me to explore that?"

13. **Portfolio diversification**: "We have {N} strategies on the leaderboard. Want me to compute the pairwise return correlations and estimate the combined portfolio Sharpe? Low-correlation strategies may combine for a portfolio Sharpe of {estimate}."

14. **Cross-symbol robustness**: "This strategy was optimized on BTC. Want me to test the same parameters on ETH and SOL without re-optimization to check if the edge is structural?"

15. **Equity curve health check**: "Want me to compute the R-squared, Ulcer Index, and rolling Sharpe of the equity curve to check if returns are consistent or coming from a few outlier periods?"

16. **Position sizing translation**: "This backtest assumes 100% capital allocation. Want me to compute expected returns and drawdowns at quarter-Kelly and half-Kelly sizing for a more realistic deployment scenario?"

17. **CPCV validation**: "This strategy has {N} free parameters. Want me to run Combinatorial Purged Cross-Validation to get a distribution of out-of-sample performance across multiple paths, and compute the Probability of Backtest Overfitting?"

18. **Session-aware execution**: "This strategy trades frequently. Want me to check if restricting entries to the EU/US overlap (12:00-16:00 UTC) improves execution quality and reduces slippage costs?"

19. **Backtest-to-live degradation**: "This strategy shows Sharpe {X} in backtest. After applying a {Y}% haircut for {Z} configs tested, the expected live Sharpe is {W}. Want me to also compute CVaR, Tail Ratio, and Gain-to-Pain for a fuller risk picture?"

20. **Strategy lifecycle planning**: "This strategy has been validated. Want me to set up circuit breaker thresholds, estimate alpha decay half-life, and recommend a capital ramp-up schedule for deployment?"

21. **Ensemble construction**: "We have {N} validated strategies. Want me to compute pairwise OOS correlations and test whether blending them (Sharpe-weighted or equal-weight with Carver multiplier) produces a higher risk-adjusted return than any individual strategy?"

22. **Breakeven fee analysis**: "Want me to sweep fee rates from 0 to 50 bps to find the exact breakeven point where this strategy stops being profitable? This shows how much execution cost buffer you have."

23. **Statistical confidence check**: "This strategy has {N} trades. Want me to compute bootstrap 95% confidence intervals for the Sharpe ratio and check the Minimum Track Record Length to see if we have enough data for reliable conclusions?"

24. **Look-ahead bias audit**: "Want me to audit the signal generation code for subtle look-ahead bias? I'll check for bfill(), centered rolling windows, full-sample normalization, and resampling alignment issues."

25. **Order flow analysis**: "Your Binance data includes taker_buy_base_volume. Want me to compute CVD divergence, taker buy ratio, and VWAP deviation bands to see if order flow adds signal to this strategy?"

26. **Liquidation cascade overlay**: "Want me to build a cascade detection filter that identifies post-liquidation bounce opportunities? Historical BTC cascades of >5% show 60-70% bounce win rates."

27. **On-chain regime overlay**: "Want me to fetch MVRV Z-Score and SOPR data to determine where we are in the BTC cycle? This tells us whether to bias toward accumulation or distribution."

28. **Advanced statistical validation**: "Want me to run Lo's adjusted Sharpe, runs test, and HLZ hurdle analysis on the top strategy? This catches autocorrelation inflation and insufficient multiple testing correction that DSR may miss."

29. **Structural pattern check**: "Want me to check whether this backtest period spans a halving cycle transition, options expiry clusters, or other BTC structural events that could inflate or deflate results?"

30. **Data quality deep scan**: "Want me to run Hampel filter outlier detection, check for Binance maintenance gaps, and verify resampling alignment before trusting these results?"

31. **Hurst-guided strategy selection**: "Want me to compute rolling DFA Hurst exponent to determine whether the current market regime favors trend-following or mean-reversion at this timeframe?"

32. **Sentiment overlay**: "Want me to add a Fear & Greed Index filter to this strategy? Suppressing longs during Extreme Greed (>75) and shorts during Extreme Fear (<25) has historically improved Sharpe by 0.1-0.3."

33. **Funding rate contrarian signal**: "Want me to test a funding rate Z-score overlay? When funding is >2σ above its 60-day mean, BTC tends to correct — this could filter out bad long entries."

34. **MTF trend confirmation**: "This strategy runs on {TF}. Want me to add a daily trend filter? The QuantPedia D1H1 study showed Sharpe improvement from 0.33 to 0.80 on BTC."

35. **ML feature screening**: "This strategy has {N} parameters and {M} trades. Want me to check if a Ridge/Lasso model on the features could improve signal quality, or whether the effective sample size is too small for ML to help?"

36. **Composite sentiment scoring**: "Want me to build a composite sentiment score from Fear & Greed Index + funding rate Z-score + social volume to use as a regime overlay? The composite typically outperforms any single sentiment source."

37. **Weekend risk reduction**: "Want me to test adding a weekend position reduction filter (50% sizing Sat-Mon)? BTC weekend flash crashes occur 3-5x/year with 30-50% lower volume."

38. **Strategy catalog selection**: "We have a catalog of 17+ research-backed strategy suggestions across all families. Want me to pick the best unexplored family and build the next strategy from the catalog?"

39. **Timeframe alignment scoring**: "Want me to compute the Multi-Timeframe Harmony Index across weekly/daily/4H/1H to see how aligned the current market structure is and whether it favors high-conviction entries?"

40. **Funding rate carry backtest**: "Want me to backtest a delta-neutral cash-and-carry (spot long + perp short) to capture funding payments? Academic studies show Sharpe 3.0-6.0+ for this approach."

41. **BTC 60-day cycle analysis**: "BTC exhibits a ~60-day intermediate cycle. Want me to check if current price action aligns with this cycle and whether a cycle-aware entry could improve timing?"

42. **Triple Screen implementation**: "Want me to build an Elder Triple Screen strategy adapted for BTC? Weekly MACD for trend, daily Stochastic for pullback, 4H for entry — this framework has strong practitioner backing."

43. **ML feasibility check**: "Before attempting ML, want me to compute the effective sample size and complexity budget? If ESS < 500, ML will almost certainly overfit and rule-based strategies are the better path."

44. **Volume confirmation filter**: "Want me to add a volume confirmation filter requiring bar volume > 1.5x the session-normalized 20-bar average before taking signals? This typically reduces trades by 30-45% but improves win rate by 3-7pp and Sharpe by 0.1-0.3."

45. **Drawdown-based position reduction**: "This strategy's max drawdown is {X}%. Want me to test layered drawdown management — vol targeting at 20% annualized + drawdown-based reduction (go-flat at -35%)? Research shows 25-40% max DD reduction."

46. **MAE stop calibration**: "Want me to run Maximum Adverse Excursion analysis on this strategy's trades? MAE analysis finds the optimal stop-loss by identifying the drawdown level where trades rarely recover — typically more robust than arbitrary fixed stops."

47. **Performance attribution**: "Want me to decompose this strategy's returns into BTC beta, momentum factor, volatility factor, and residual alpha? This reveals how much return is genuine skill vs passive exposure."

48. **Mean reversion with regime filter**: "Want me to test a regime-filtered RSI mean reversion? ADX < 25 filter alone improves BTC mean reversion Sharpe by +0.3-0.7. Combined with BB confluence, win rates reach 60-68%."

49. **Session-time restriction**: "Want me to test restricting signals to 08:00-20:00 UTC weekdays only? This is one of the highest-impact, lowest-complexity enhancements — eliminates thin-market noise for +0.2-0.5 Sharpe."

50. **Equity curve filter**: "This strategy has extended drawdown periods. Want me to test an equity curve filter (30-trade MA) to reduce position sizing when the strategy is underperforming? Reduces max DD by 20-40%."

51. **Walk-forward validation**: "Want me to run rolling walk-forward analysis (18-month IS, 3-month OOS) to measure out-of-sample retention? The WFER tells us how much of the backtested Sharpe to realistically expect live."

52. **CPCV robustness check**: "Want me to run Combinatorial Purged CV (N=6, k=2) to get 15 independent OOS Sharpe estimates? This gives a distribution of expected performance, not just a single point — and catches strategies that work on only one data split."

53. **Transaction cost tier analysis**: "The current flat 15 bps may overstate your actual costs at $10K size. Want me to compute costs under Tier 2 (impact-adjusted) and compare? The minimum viable edge at $10K is ~13.4 bps — some rejected strategies may actually be viable."

54. **Funding rate impact assessment**: "This strategy holds positions overnight on perps. Want me to estimate the funding drag? At mean BTC funding (0.012%/8h), a continuously-long strategy faces ~13-20% annualized drag that erodes gross returns."

55. **Portfolio construction assessment**: "We have 2+ validated strategies. Want me to construct a risk-parity portfolio with regime-conditional allocation, exposure limits, and portfolio-level circuit breakers? Multi-strategy portfolios typically reduce max DD from 25-40% (single) to 12-20%."

56. **Strategy decay monitoring setup**: "Want me to estimate the decay rate (λ) for this strategy and set up a monitoring framework? I'll compute WFER trends, recommend decay-adjusted sizing, and establish CUSUM alarm thresholds for structural break detection."

57. **Backtest engine validation**: "Before trusting these results, want me to run the 5 synthetic test suite to validate the engine's fill model, fee accounting, and equity curve computation? This catches off-by-one shifts and fee double-counting."

58. **Cross-engine validation**: "Want me to cross-validate this result against vectorbt? If both engines produce the same equity curve (within 0.5%), we can rule out engine-specific bugs as a source of spurious performance."

59. **Parameter stability analysis**: "Want me to compute the Parameter Concentration Ratio (PCR) for each optimized parameter across walk-forward segments? PCR > 0.70 means the parameter is stable and deployable; PCR < 0.30 means it's noise."

60. **Breakeven notional analysis**: "This strategy has {X} bps gross edge. Want me to compute the breakeven position size where market impact consumes the edge? For strategies near the viability threshold, this determines the maximum deployable capital."

---

## Exit Strategy Guidance

The default signal model (forward-fill until opposite signal) provides no risk management. Consider adding exit conditions within `generate_signals()` for improved drawdown control.

### Exit Mechanisms (Priority Order)

**1. ATR Trailing Stop (Recommended for all strategies)**

```python
# Within generate_signals():
atr_vals = atr(df, period=14)
# Set stop at entry_price - multiplier * ATR (for longs)
# Typical multiplier: 2.0-3.0 (wider for trend-following, tighter for mean-reversion)
# BTC-specific: use 2.5-3.0x for 1h, 3.0-4.0x for 4h
# When price crosses stop -> set signal to 0 (exit)
```

**2. ATR Profit Target**

```python
# Take profit at entry_price + N * ATR
# Default ratio: 2.0x ATR (2:1 reward-to-risk with 1.0x stop)
# Optimize jointly: sweep profit_atr_multiple in [1.0, 1.5, 2.0, 2.5, 3.0]
```

**3. Time-Based Exit (Best for Mean-Reversion)**

```python
# Exit after N bars if no profit target or stop hit
# Mean-reversion: payoff concentrates in early bars; N = 12-24 bars at 1h
# Trend-following: longer hold; N = 48-96 bars at 1h
# Build histogram of bars_held for winners vs losers to calibrate N
```

**4. Volatility Regime Exit (Circuit Breaker)**

```python
# Force exit when current_vol / baseline_vol > 2.0
# Protects against regime changes and Black Swan events
# Baseline: rolling vol over 4x the signal period
```

### First-to-Trigger Logic

When multiple exits are active, whichever fires first closes the position:
```
exit = stop_hit OR profit_target_hit OR trailing_stop_hit OR vol_spike OR time_expired
```

### Win Rate vs Avg Win/Loss Tradeoff

Tighter exits -> higher win rate, smaller avg win. Looser exits -> lower win rate, larger avg win. The invariant is **expectancy**: `(win_rate * avg_win) - ((1 - win_rate) * avg_loss)`. With 0.15% round-trip costs, expectancy per trade must exceed ~0.20% to be viable at high frequency.

---

## Position Sizing Interpretation

The backtest engine uses all-in/all-out sizing (100% capital per signal). Real deployment uses fractional sizing. This section translates backtest results to realistic expectations.

### Kelly Criterion from Backtest Results

```
f* = win_rate - (1 - win_rate) / (avg_win / avg_loss)
```

**Example:** Win rate 55%, avg win 3.2%, avg loss 2.1% -> f* = 0.55 - 0.45/1.524 = 0.255 (risk 25.5% per trade at full Kelly).

**Never use full Kelly.** Full Kelly has ~50% probability of a 50% drawdown. Use fractional Kelly:

| Fraction | Growth Rate (% of max) | Use Case |
|----------|----------------------|----------|
| 0.50 (Half Kelly) | 75% | Standard for validated strategies |
| 0.25 (Quarter Kelly) | 44% | New/unvalidated strategies, crypto default |
| 0.10 | 19% | Initial deployment, testing |

**BTC fat-tail adjustment:** BTC excess kurtosis ~6-10 means variance underestimates risk. Apply a fat-tail discount: multiply Kelly fraction by `3 / kurtosis` (e.g., kurtosis 6 -> multiply by 0.5). For crypto, 0.20-0.25 of full Kelly is the practical recommendation.

### Scaling Backtest Results to Fractional Sizing

| Metric | At Fraction f | Notes |
|--------|---------------|-------|
| Sharpe Ratio | Unchanged | Scale-invariant (approximately) |
| CAGR | ~CAGR_backtest * f | Linear approximation for f < 0.5 |
| Max Drawdown | ~MaxDD_backtest * f | Linear approx; add safety margin |
| Worst Trade | worst_trade * f | Scales linearly |

### Volatility Targeting

Scale position size dynamically so realized portfolio vol matches a target:

```
position_fraction = target_vol / realized_vol
```

| Strategy Type | Recommended Target Vol |
|--------------|----------------------|
| BTC trend-following | 15-25% annualized |
| BTC mean-reversion | 10-15% annualized |
| Multi-strategy portfolio | 10-20% annualized |

With BTC's typical 60-80% vol, a 15% target implies ~20% position size. **Floor the vol estimate at 30%** to prevent oversizing during abnormally calm periods. Cap position at 1.5x target.

### Risk of Ruin

```
P(ruin) ≈ ((1 - f*edge) / (1 + f*edge))^(capital / risk_per_trade)
```

Target: P(ruin) < 2% for primary capital. At quarter Kelly with a modest edge, risk of ruin is typically < 1%.

### Presentation Format

When presenting Phase 8 results, include a sizing table:

```
=== Position Sizing (backtest used 100% allocation) ===
Kelly fraction estimate: f* = X (W=X%, avg_win=X%, avg_loss=X%)
Fat-tail adjusted Kelly: X (kurtosis adjustment)

Conservative (0.15 Kelly):  CAGR ~X%,  MaxDD ~X%
Moderate (0.25 Kelly):      CAGR ~X%,  MaxDD ~X%
Aggressive (0.40 Kelly):    CAGR ~X%,  MaxDD ~X%
```

---

## Dynamic Drawdown Management & Advanced Position Sizing

Beyond static Kelly-based sizing, dynamic position management adapts to equity curve state, realized volatility, and drawdown depth. These techniques reduce maximum drawdown at the cost of slightly extended recovery time — a favorable tradeoff for BTC's fat-tailed distribution.

### Anti-Martingale Position Sizing

Increases size after winning trades, decreases after losing trades. Aligns capital with favorable regime detection.

**Formula:** `position_size = base_size × (1 + equity_growth_pct / threshold)`

Where equity_growth_pct = (current_equity - starting_equity) / starting_equity × 100.

**Example:** Starting $100K, current $115K (15% growth), threshold 20 → multiplier = 1.75 (75% larger than base).

**BTC-recommended parameters:**

| Parameter | Conservative | Moderate | Aggressive |
|---|---|---|---|
| Threshold | 30 | 20 | 15 |
| Cap (max multiplier) | 1.5x | 2.0x | 3.0x |
| Floor (min multiplier) | 0.5x | 0.35x | 0.25x |

**Critical:** Always cap the multiplier (2x recommended default). BTC can reverse 20-30% in days — uncapped anti-martingale gives back all compounded gains. Combine with volatility targeting to moderate during vol spikes.

### Drawdown-Based Position Reduction

Mechanically reduce size as drawdown from equity high-water mark deepens.

**Linear reduction:** `size_multiplier = max(0, 1 - (current_DD / max_DD_threshold))`

**BTC-specific step function (recommended):**

| Drawdown Range | Position Size | Rationale |
|---|---|---|
| 0% to -15% | 100% | Normal BTC pullback zone |
| -15% to -25% | 60% | Meaningful correction |
| -25% to -35% | 30% | Potential regime change |
| Beyond -35% | 0% (flat) | Bear market confirmed |

**Why BTC thresholds are wider than equity:** BTC bull markets routinely see 20-35% drawdowns that are NOT the end of the trend (2017: six >20% DDs; 2020-21: 27%, 31%, 35% DDs followed by new ATH). Setting go-flat at -20% triggers during normal bull corrections and misses recovery.

**Backtesting evidence:** Drawdown-based position reduction reduces max DD by 25-40% and annualized return by 10-20%, improving Sharpe by ~0.05-0.15 and Calmar ratio substantially.

### Risk Budget Framework

**Daily/weekly loss budgets:**
- Daily: -1.5% of equity (reduce to 50% size). At -2.0%: go flat for day.
- Weekly (rolling 5-day): -4% (reduce 50%). At -5%: reduce to 25%.
- Monthly: -10% (reduce to minimum). At -12%: pause trading, require review.

**Rolling accumulation (preferred over calendar windows):**
- Rolling 5-day P&L < -3%: reduce 50%
- Rolling 5-day P&L < -5%: reduce to 25%
- Rolling 10-day P&L < -8%: go flat, wait for reset (rolling 5-day turns positive OR 3-day cooling)

Rolling windows avoid the "reset effect" where a large loss on the last day of the week is immediately followed by full-size trading Monday.

**Monte Carlo calibration:** Collect 200-500+ historical trades. Bootstrap/Monte Carlo 1000+ sequences. Set budgets at 95th percentile of simulated loss distributions for each window. This ensures budgets trigger only ~5% of the time during normal strategy variance.

### Volatility-Targeted Position Sizing

**Formula:** `position_size = base_size × (target_vol / realized_vol)`

**BTC-specific parameters:**

| Parameter | Conservative | Moderate | Aggressive |
|---|---|---|---|
| Target vol (annualized) | 15% | 20% | 30% |
| Vol floor (annualized) | 35% | 40% | 50% |
| Vol ceiling (annualized) | 100% | 120% | 150% |
| EWMA lambda | 0.96 | 0.94 | 0.90 |

**Vol floor rationale:** Even during quiet periods, BTC can spike in vol rapidly. Floor prevents positions from growing dangerously large during low-vol regimes. At 40% floor with 20% target: max position = 0.5x base.

**EWMA preferred over rolling window:** EWMA (lambda 0.94, ~15-day half-life) weights recent observations more heavily, capturing vol regime shifts faster than a fixed 20-day window without the "cliff effect." BTC trades 24/7 with no overnight gaps, so Parkinson (high-low) or Garman-Klass (OHLC) vol estimators are valid supplements.

**Performance evidence (Moreira & Muir, 2017):** Vol targeting improves Sharpe by 0.1-0.3, reduces max DD by 15-30%, reduces negative skewness. Improvement is largest for assets with strong vol clustering (BTC qualifies strongly).

### Maximum Adverse Excursion (MAE) Analysis

**Purpose:** Determine where to place stop-losses by analyzing the distribution of maximum unrealized loss across historical trades.

**Procedure:**
1. Record MAE for every trade (both winners and losers)
2. Plot MAE distributions separately for winners and losers
3. Compute `P(recovery | MAE = x%) = count(winners with MAE ≥ x%) / count(all trades with MAE ≥ x%)`
4. Find where recovery probability crosses 50% → theoretical optimal stop

**BTC-specific typical values (daily trend-following):**
- P(recovery | MAE = 3%) ≈ 60-70% — many winners experience 3% drawdown
- P(recovery | MAE = 7%) ≈ 40-55% — close to inflection zone
- P(recovery | MAE = 12%) ≈ 25-35% — most trades at this DD are losers
- P(recovery | MAE = 20%) ≈ 10-15% — almost certainly a losing trade

**MAE percentile stops:**
- 75th percentile of winning-trade MAE → primary stop (gives room; suitable for trend-following)
- 90th percentile → hard/emergency stop
- For BTC daily trend: approximately 8-12% (primary) and 15-20% (hard). **Must be calibrated from specific strategy data.**

### Equity Curve Trading

Apply a trend-following filter to the strategy's own equity curve. Only take full-size trades when equity is above its N-period MA.

**Graduated approach (recommended over binary on/off):**
- Equity > 5% above MA: 100% size
- Equity at MA: 70% size
- Equity 5% below MA: 40% size
- Equity 10%+ below MA: 20% size

**Lookback periods:**
- Short-term strategies (intraday): 20-30 trade lookback or 10-20 day equity MA
- Medium-term strategies: 30-50 trade lookback (most commonly effective for BTC)
- Long-term strategies: 15-25 trade lookback or 60-90 day equity MA

**Evidence:** Reduces max drawdown by 20-40% across futures strategies. Primary cost: misses 30-60% of early recovery (especially costly for BTC's V-shaped recoveries like March 2020). Graduated approach mitigates this.

### Recovery Time Analysis

| Drawdown | Recovery Time (at 40% annualized return) |
|---|---|
| 10% | ~3.2 months |
| 15% | ~4.9 months |
| 20% | ~6.7 months |
| 25% | ~8.7 months |
| 30% | ~10.7 months |

**With position reduction:** A strategy with max DD of 30% at full size might see DD reduced to ~20% with drawdown-based reduction, but recovery time extends from ~10.7 to ~14-18 months (trading at reduced size during recovery).

### Layered Risk Management Architecture

The most robust approach combines multiple overlapping systems:

1. **Volatility targeting** (base layer — always active): Adjusts for market conditions
2. **Anti-martingale scaling** (equity-aware layer): Increases with profits, decreases with losses
3. **Drawdown-based reduction** (emergency layer): Only activates beyond threshold (e.g., -15%)
4. **Risk budgets** (circuit breaker layer): Hard limits that override everything
5. **Equity curve filter** (regime layer): Detects strategy-regime misalignment

**Asymmetric rules for faster recovery:** Reduce size gradually on the way down, but restore more quickly on the way up. Example: linear reduction to 50% at -20% DD; if strategy recovers 50% of DD (from -20% to -10%), immediately restore to 80% rather than the 50% the linear rule prescribes.

**Target:** Max drawdown of 25-30% for a BTC strategy (roughly one-third to one-half of BTC buy-and-hold drawdown). Accept 6-12 month recovery from significant drawdowns.

### Drawdown Management Parameter Summary

| Parameter | Conservative | Moderate | Aggressive |
|---|---|---|---|
| DD reduction: go-flat | -40% | -35% | -25% |
| DD reduction: half-size | -25% | -20% | -15% |
| Daily loss budget | -1.0% | -1.5% | -2.0% |
| Weekly loss budget | -3.0% | -4.0% | -5.0% |
| EWMA lambda | 0.96 | 0.94 | 0.90 |
| Equity curve MA lookback | 50 trades | 30 trades | 20 trades |
| MAE stop (winning trade pctile) | 90th | 75th | 60th |

"Moderate" column = recommended starting point. Refine through strategy-specific backtesting with out-of-sample validation and Monte Carlo stress testing.

---

## Strategy Lifecycle: Deployment, Monitoring & Retirement

After the Optimization Loop produces a validated strategy, this section guides the transition from backtest to deployment and ongoing management.

### Pre-Deployment Checklist

All items must pass before going live:

- [ ] Backtest Sharpe > 1.5 (before deflation)
- [ ] DSR p-value < 0.05 (significant after multiple testing)
- [ ] Walk-forward OOS Sharpe > 50% of IS Sharpe
- [ ] At least 100 trades in backtest (200+ preferred)
- [ ] Strategy logic has clear economic rationale (not purely data-mined)
- [ ] Paper trading for minimum 100 round-trip trades before live capital
- [ ] Circuit breakers coded and tested
- [ ] Execution infrastructure tested for full paper trading period

### Paper Trading Monitoring

During paper trading, compare against backtest expectations:

| Metric | Red Flag Threshold |
|--------|-------------------|
| Win rate | > 10 percentage points below backtest |
| Avg win/loss ratio | > 20% worse than backtest |
| Sharpe ratio | < 50% of backtest Sharpe |
| Max drawdown | > 1.5x backtest max DD |
| Trade frequency | > 30% deviation from backtest |
| Fill slippage | > 2x assumed slippage |

### Capital Ramp-Up Schedule

| Period | Allocation (% of target) | Condition |
|--------|--------------------------|-----------|
| Week 1-2 | 10% | Initial deployment |
| Week 3-4 | 25% | Metrics on track |
| Month 2 | 50% | No red flags |
| Month 3+ | 75-100% | Sustained performance |

### Live Monitoring: Circuit Breakers

Set drawdown thresholds based on Monte Carlo simulation of backtest trades (95th and 99th percentile max drawdown):

| Level | Trigger | Action |
|-------|---------|--------|
| **Yellow** | DD > backtest max DD (or MC 75th pctl) | Increase monitoring frequency |
| **Orange** | DD > 1.5x backtest max DD (or MC 95th pctl) | Reduce allocation by 50%. Formal review within 48h. |
| **Red** | DD > 2x backtest max DD (or MC 99th pctl) | Halt immediately. Full post-mortem required. |

### Strategy Degradation Detection

- **Rolling 60-day Sharpe** drops below 50% of historical average for 30+ consecutive days -> warning.
- **Cumulative alpha** (strategy return - benchmark return) flattens for 30-60 days -> warning. Declines for 60+ days -> retirement signal.
- **Drawdown duration** exceeds 95th percentile of Monte Carlo-simulated drawdown durations -> strong degradation signal.
- **CUSUM chart:** Track cumulative sum of (target_daily_return - actual_daily_return). Persistent drift signals the strategy's mean return has shifted.

### "Strategy is Broken" vs "Regime Isn't Present"

| Signal | Strategy Broken | Regime Absent |
|--------|----------------|---------------|
| Trade frequency | Normal but losing | Reduced (setup doesn't appear) |
| Win rate on trades taken | Degraded (near 50%) | Maintained (historically normal) |
| Other similar strategies | May still work | Also underperforming |
| Regime indicators | Favorable regime present | Unfavorable regime confirmed |

### Retirement Criteria

- Live Sharpe < 50% of backtest Sharpe for **6+ months** with 200+ trades -> retire
- Live Sharpe 50-75% of backtest for **12+ months** -> retire
- Strategy paused and resumed 3+ times without recovery -> retire
- Alpha contribution (from regression attribution) indistinguishable from zero for 6+ months -> retire

### Alpha Decay Half-Lives

Strategy edges erode as more participants discover them. Crypto alpha typically decays 40-60% faster than traditional markets due to lower barriers to entry and faster information diffusion.

| Strategy Type | Traditional Half-Life | Crypto Estimate |
|--------------|----------------------|-----------------|
| Trend-following | 3-10 years | 1-5 years |
| Mean-reversion (daily) | 1-3 years | 6-18 months |
| Breakout | 1-3 years | 6-18 months |
| Volatility strategies | 2-5 years | 1-3 years |
| Statistical arbitrage | 6-18 months | 3-12 months |
| HFT / market making | 3-12 months | 1-6 months |

**Budget for strategy replacement:** Plan to develop 1-2 new strategies per year. Monitor rolling Sharpe slope from day one to estimate remaining lifetime.

### Strategy Decay Quantification & Monitoring

Beyond observing half-lives, quantify decay rate precisely and monitor for structural breaks vs. regime shifts.

**Decay Rate Estimation from WFER:**

```
λ_decay = -ln(WFER_k) / months_since_validation
```

Where WFER_k is the most recent walk-forward segment's efficiency ratio. Apply decay to position sizing:

```
size_t = full_size × exp(-λ × t)    subject to floor at 0.25 × full_size
```

The floor at quarter-Kelly prevents positions so small that trading costs exceed expected returns. Once the floor has been active for 3+ consecutive months without re-validation → decision point: re-validate or retire.

**Example (EMA crossover, λ=0.035/month ≈ 20-month half-life):**
- Month 0: Full Kelly = 40% → size = 40.0%
- Month 12: 40% × exp(-0.42) = 26.3%
- Month 24: 40% × exp(-0.84) = 17.2%
- Month 36: max(40% × exp(-1.26), 10.0%) = 11.3%
- Month 45: floor active → 10.0%

**Differentiated BTC Decay by Strategy Type:**

| Strategy Type | Decay Mechanism | BTC Half-Life | Reversible? |
|---|---|---|---|
| Trend-following (short-term, EMA 20/50) | Crowded signals, efficiency gains | 18-24 months | No (secular) |
| Trend-following (long-term, EMA 100/200) | Same, slower | 24-36 months | No (secular) |
| Mean reversion (intraday, <1h) | HFT competition, tighter spreads | 12-18 months | Partial |
| Mean reversion (daily) | Same competition, slower | 24-36 months | Partial |
| Momentum (ROC, Dual RSI) | Regime-dependent, crowding | 18-30 months | Regime-dependent |
| Funding carry | Capital crowding compresses carry | 6-18 months (cyclical) | Yes (cyclical) |
| Event-driven (halving, CME expiry) | Structural, front-running | 36-72 months | Mostly yes |

**For position sizing, use the upper 95% CI of λ (faster decay).** The cost of under-sizing a viable strategy < cost of over-sizing a decaying one.

### Structural Break Detection

Detect when the return-generating process has changed (not just decayed gradually):

**Chow Test:** Split at hypothesized break date, test if regression coefficients differ. Use for known events (ETF approval, exchange collapse, halving).

**Bai-Perron (Multiple Break):** Automatically detects multiple structural breaks in the return series. Apply to rolling 90-day Sharpe time series. Report break dates and confidence intervals.

**CUSUM (Cumulative Sum Control Chart):**
```
CUSUM_t = max(0, CUSUM_{t-1} + (r_t - target_mean - slack))
```
Where `target_mean` = expected daily return from decay model, `slack` = 0.5σ (prevents oversensitivity). Alarm when CUSUM > H.

**BTC-specific:** Use H = 5σ (not standard 4σ) to account for fat-tailed returns. Reset CUSUM to zero after each confirmed structural break investigation.

**Break Categories:** (1) Market structure change (new futures, ETF approval) — permanent; (2) Competition increase (more MMs, strategy crowding) — permanent; (3) Regime shift (vol collapse/expansion) — temporary; (4) Data artifacts — exclude before testing.

### Re-Validation Triggers

| Trigger | Threshold | Action |
|---|---|---|
| Performance ratio breach | Trailing 90-day Sharpe below decay-predicted ± 2 SE | Full re-validation |
| CUSUM alarm | H > 5σ | Investigation, possible halt |
| Market regime break | Bai-Perron break in BTC price process | Re-validate even if strategy on-model |
| Age trigger | 18 months (trend), 12 months (MR), 6 months (carry) | Mandatory re-validation |
| Floor active 3+ months | Decay has run its course | Retire or re-validate with fresh data |

Re-validation = fresh `run_sweep()` on recent data (last 12-18 months), evaluate winning params on fresh OOS, reset t=0 only if positive OOS expectation confirmed.

### Live Monitoring Dashboard (5 Metrics)

Begin monitoring from day 1, not month 3. Bayesian updating from backtest priors allows meaningful inference from 20-30 live trades.

**Metric 1 — Rolling 90-Day Sharpe with Significance Bands:**
- Sharpe > 1.5: On track. 0.5-1.5: Below forecast, increase monitoring. 0 to 0.5: Marginal, extra size reduction. < 0 for 60+ days: Warning, trigger review. < -0.5 for 30+ days: Critical, halt.
- Plot against decay envelope: `Sharpe_predicted = Sharpe_IS × exp(-λ × months)` ±1.5 SE bands.
- Note: 90-day window requires Sharpe > 3.31 for 5% statistical significance — you can only confirm *absence* of edge, not presence.

**Metric 2 — Rolling Alpha (90-Day OLS):**
```
r_strategy = α + β × r_BTC + ε
```
Declining alpha = edge decay even if total returns look adequate (bull market masking). If beta drifting toward 1.0: strategy becoming involuntary buy-and-hold.
- Alpha > 20% ann: Strong. 10-20%: Acceptable. 0-10%: Concerning. < 0%: Edge gone.

**Metric 3 — Rolling Information Coefficient (60-Day):**
```
IC = corr(signal_t, r_{t, t+h})
```
IC > 0.05: Meaningful. 0.02-0.05: Marginal. 0-0.02: Noise. < 0 for 30+ days: Signal flipped, halt immediately.
- BTC IC is episodically high in trends, near-zero in chop. Alarm = IC < 0.02 for 60+ consecutive days including through trending periods.

**Metric 4 — CUSUM on Strategy Returns:** As defined in structural break detection above. Continuous real-time monitoring mode.

**Metric 5 — Performance Envelope Comparison (180-Day):**
- Optimistic: Sharpe_IS × 0.80. Expected: Sharpe_IS × WFER. Pessimistic: Sharpe_IS × 0.40.
- Update envelope every 3 months with current trailing WFER observation.
- Above optimistic: Possible overfitting to current regime. Within expected: On track. Below pessimistic: Immediate re-validation.

### "Strategy Dead" vs. "Waiting for Regime"

The hardest problem in strategy management. Both look identical short-term: negative or flat returns.

**Mechanistic Priors:**
- Lean "Regime Absent" if: BTC in identifiable unfavorable regime, other strategies in same class also underperforming, signal still firing at normal frequency, no new competition.
- Lean "Structural Decay" if: Other same-class strategies performing normally, signal frequency/magnitude declined, costs increased, CUSUM alarm near underperformance onset.

**Sequential Probability Ratio Test (SPRT):**
- H0 (Regime Absent): Monthly returns ~ N(μ₀ = -0.8%/month, σ = 5%)
- H1 (Structural Decay): Monthly returns ~ N(μ₁ = -2.5%/month, σ = 5%)
- At each monthly return: `log_LR_t += log[f(r_t | H1) / f(r_t | H0)]`
- Conclude decay when log_LR > 2.20. Conclude regime absence when log_LR < -2.20. Continue observing otherwise.
- Typically requires 8-15 months to reach conclusion (effect size d = 0.34 — small-to-medium).

**Consecutive Losing Months Heuristic:**

| Consecutive Losing Months | P(Decay) | Action |
|---|---|---|
| 1-2 | 58% | Monitor |
| 3 | 62% | Size to 75% |
| 4 | 69% | Size to 50%, formal review |
| 5 | 78% | Size to 25% (floor), weekly review |
| 6 | 86% | Halt, full re-validation |
| 7+ | >92% | Presumed dead pending re-validation |

**Decision Matrix (combined signals):**

| SPRT Leans | Regime Fingerprint | CUSUM | IC | Conclusion |
|---|---|---|---|---|
| Regime absence | Unfavorable | No alarm | > 0.02 | Hold at floor sizing |
| Decay | Favorable | Alarm | < 0.02 | Confirmed decay — halt, re-validate |
| Decay | Unfavorable | No alarm | > 0.02 | Likely regime-driven — hold at 25%, re-evaluate in 3 months |
| Regime absence | Favorable | No alarm | > 0.02 | Ambiguous — investigate signal specifically |

**Governing principle:** Never conclude "waiting for regime" if IC < 0.02 for 60+ days in a favorable regime. IC is the most direct signal-quality measure.

**Statistical power reality:** At effect size d=0.34, power to detect decay is only 27% at 6 months, 46% at 12 months, 75% at 24 months. Automatic decay-adjusted sizing must do the risk management that statistical tests cannot yet perform.

### Operational Cadence

| Frequency | Actions |
|---|---|
| Daily | Update rolling Sharpe, alpha, IC, CUSUM. Flag warning zones. |
| Monthly | Update WFER. Apply decay-adjusted sizing. Run SPRT update. Review regime fingerprints. |
| Quarterly | Full performance-vs-envelope review. Update λ estimate. Assess re-validation triggers. |
| At trigger | Re-validate: fresh sweep → OOS evaluation → retire or reset t=0. |

---

## Strategy Creation Template

When creating a new strategy, use this template. **Both long and short signals must be meaningful** — do not clip or remove short signals in the strategy itself. The engine handles `long_only` mode via `BacktestConfig`.

```python
# strategies/{name}_strategy.py
import pandas as pd
from backtester.strategy import Strategy, {indicators}

class {Name}Strategy(Strategy):
    def __init__(self, {params_with_defaults}):
        self.param1 = param1
        # ...

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 1. Compute indicators
        # 2. Generate signals: 1=long, -1=short, 0=flat
        signals = pd.Series(0, index=df.index, dtype="int8")
        signals[long_condition] = 1
        signals[short_condition] = -1
        # 3. Forward-fill to hold positions until opposite signal
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")
        # NOTE: Do NOT clip signals here. BacktestConfig(long_only=True) handles it.
        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "param1": [...],
            "param2": [...],
        }
```

**Design guidelines for robust strategies:**
- Prefer fewer parameters (<=5). More parameters = more overfitting risk.
- Use round numbers in param_grid (traders cluster at round numbers, creating structural effects).
- Combine indicators from different dimensions (consult the Redundancy Matrix).
- Consider adding a regime filter if the strategy is regime-dependent.

Then run:
```bash
# Test both modes
python3 run_backtest.py --strategy {name} --resample 1h --sweep
python3 run_backtest.py --strategy {name} --resample 1h --sweep --long-only
```

---

## Rules of Engagement

1. **Use the appropriate resolution for sweeps.** Standard analysis uses 1h+. Sub-minute uses dedicated fetched data. Match grid size to resolution tier (see Multi-Resolution Protocol).

2. **Capture JSON output to a variable** when you need to process results programmatically:
   ```bash
   result=$(python3 run_backtest.py --strategy rsi --resample 1h --sweep 2>/dev/null)
   echo "$result" | python3 -c "import json, sys; data = json.load(sys.stdin); ..."
   ```

3. **Use stderr output for display.** When showing results to the user, run the command without redirecting stderr — the human-readable table goes there.

4. **Never recommend a strategy without completing Phase 5** (robustness validation).

5. **Always compare to buy-and-hold.** A strategy that returns 50% when buy-and-hold returns 80% is underperforming.

6. **Use inline Python for flexibility** — custom grids, filtering, multi-period loops, sensitivity analysis.

7. **Round numbers** to 1-2 decimal places for readability.

8. **Signal execution model:** Signal at bar `i` executes at bar `i+1` (next-bar entry). Returns are close-to-close. This is already handled by the engine — no need to shift signals manually.

9. **Always test both modes.** Never assume long-only. The Long+Short Evaluation Protocol determines the best mode empirically.

10. **Strategy signals must be bidirectional.** Never clip or remove short signals in strategy code. The engine's `BacktestConfig(long_only=True)` handles mode restriction.

11. **Sub-minute data requires explicit fetching.** Always check if the data file exists before running sub-minute backtests. Use `fetch_binance_klines.py` to acquire it.

12. **In loop mode, announce state after each iteration.** Keep the user informed of progress, leaderboard position, convergence status, and next action.

13. **Track total combinations tested.** Maintain a running count of N_total across all sweeps for DSR calculation. Report this in the final recommendation.

14. **Always report Sharpe with timeframe context.** A Sharpe of 2.0 on minute data is not the same as 2.0 on daily data. Always state the timeframe alongside the Sharpe number.

15. **Prefer plateau centers over peaks.** When selecting final parameters, choose the center of a well-performing region rather than the absolute best point. This generalizes better.

16. **Explore across strategy families.** The derivation engine should not get stuck in one family. If 3+ consecutive strategies are from the same family, force exploration of an unexplored family.

17. **Respect indicator warm-up periods.** Never evaluate signals or count returns during the warm-up window. Use `max(all indicator periods)` as the warm-up cutoff. For EMA-based indicators, use 3*N bars minimum.

18. **Validate data quality before every new dataset.** Run the Pre-Backtest Validation checks when loading data for the first time or switching symbols. Bad data produces silently wrong results.

19. **Penalize complexity.** When two strategies have similar Sharpe, prefer the one with fewer parameters. Require at least 30 trades per free parameter for statistical reliability.

20. **Consider exit mechanisms.** For strategies with max drawdown > 25% or time-in-market > 90%, evaluate whether ATR trailing stops or time-based exits could improve risk-adjusted returns. See Exit Strategy Guidance.

21. **Apply backtest-to-live haircuts.** Never present backtested Sharpe as the expected live result. Always apply the appropriate haircut from the Backtest-to-Live Haircut Table and present both numbers. If haircut-adjusted Sharpe < 0.5, do not recommend.

22. **Report alternative risk metrics.** Include CVaR (95%), Tail Ratio, and Gain-to-Pain Ratio alongside Sharpe in every Phase 8 recommendation. Sharpe alone is unreliable for BTC's fat-tailed returns (excess kurtosis ~6-10).

23. **Use session-aware execution timing.** When possible, time execution signals to the EU/US overlap (12:00-16:00 UTC) for best liquidity. Apply a 1.5x slippage multiplier for strategies that trade outside peak hours.

24. **Consider position sizing context.** All-in/all-out backtests represent the theoretical maximum. Present results at quarter-Kelly sizing for realistic deployment expectations. Include Kelly estimation and risk-of-ruin calculations in Phase 8.

25. **Plan for strategy decay.** Include alpha decay half-life estimate in the Final Report. Budget for strategy replacement (1-2 new strategies per year). Set circuit breakers from day one of live deployment.

26. **Run look-ahead bias checks on every new strategy.** Before trusting any backtest, verify the signal generation code against the Look-Ahead Bias Checklist. Pandas-specific pitfalls (bfill, center=True, full-sample normalization) are the most common silent killers.

27. **Report confidence intervals, not just point estimates.** Every Sharpe, max drawdown, and win rate should have a 95% CI via block bootstrap. If the CI is too wide to be actionable (lower bound < 0), the result needs more trades or a longer track record.

28. **Test fee sensitivity at three levels, not one.** A strategy that only works at optimistic (maker-only) fees is not viable for most traders. Level 2 (realistic) must be profitable. Report the breakeven fee rate for transparency.

29. **Evaluate ensemble benefit before recommending multiple strategies.** Only combine strategies with pairwise OOS correlation < 0.3. Verify the ensemble Sharpe exceeds the best individual Sharpe by at least 0.2. For a single asset, 2-3 strategies is the practical maximum before diminishing returns.

30. **Match parameter search method to dimensionality.** Exhaustive grid search is correct for 1-2 parameters but wasteful for 4+. Use Bayesian optimization or random search for higher-dimensional spaces. Warm-start from KNOWLEDGE.md when prior data exists.

31. **Load additional Binance columns for order flow strategies.** The CSV contains `taker_buy_base_volume`, `quote_volume`, and `trades` beyond basic OHLCV. Load these when using CVD, taker buy ratio, VPIN, or trade intensity signals. The real aggressor-classified data is strictly superior to OHLCV-only proxies (CLV, OBV).

32. **Run advanced statistical tests on every candidate.** Compute Lo's autocorrelation-adjusted Sharpe, runs test, and HLZ hurdle alongside DSR. If autocorrelation is present (Ljung-Box p < 0.05), the adjusted Sharpe is the true number — report both. If the runs test shows clustering, use block bootstrap for Monte Carlo.

33. **Shift on-chain data by at least 1 day in backtests.** On-chain metrics (MVRV, SOPR, exchange flows) are end-of-day aggregates. Using same-day values is look-ahead bias. Always apply `metric.shift(1)` before signal generation.

34. **Account for BTC structural patterns.** Check halving cycle position, options expiry calendar, and weekend regime before interpreting results. A strategy tested entirely during post-halving parabolic phase will overstate edge. Sub-period testing should span different cycle phases.

35. **Reduce weekend exposure.** Apply a 0.5x position sizing multiplier on weekends (Saturday 00:00 UTC to Monday 00:00 UTC). Weekend flash crashes occur 3-5 times per year with 30-50% lower volume — the risk/reward of holding full positions does not justify the tail risk.

36. **Use DFA for Hurst estimation, not R/S.** R/S overestimates H by 0.02-0.03 for BTC data. Report both if desired, but use DFA as the primary estimate. Always include 95% CI via block bootstrap — a point estimate of H without CI is not actionable.

37. **Add MTF trend filters before adding complexity.** A single daily trend filter on hourly signals is one of the highest-impact, lowest-risk improvements available (+0.3-0.7 Sharpe). Always use `label='right', closed='right'` when resampling for higher-TF indicators, and always `.shift(1)` after forward-filling to prevent look-ahead bias.

38. **Treat sentiment data as risk management, not alpha.** Fear & Greed Index, social media volume, and Google Trends are best used for position sizing and regime filtering, not standalone entry signals. Realistic Sharpe improvement from sentiment overlay: +0.1-0.3. Any backtest showing Sharpe > 2.0 from sentiment alone likely has look-ahead bias.

39. **Shift all alternative data by the appropriate lag.** Fear & Greed Index: +1 day. Google Trends: +7 days minimum (retroactive renormalization makes absolute levels unusable). Funding rates: +1 settlement period. On-chain data: +1 day. Never use same-day sentiment or alternative data values.

40. **Exhaust rule-based strategies before ML.** ML should augment, not replace, rule-based strategies. Compute effective sample size (ESS) first — if ESS < 500, ML will overfit. Maximum features = sqrt(ESS). Include all hyperparameter trials in DSR calculation. ML alpha decays 2-3x faster than rule-based alpha in crypto.

41. **Use the Research-Driven Strategy Catalog for Method 5 derivations.** When the loop needs a new concept, select from the catalog (Tier A first, then Tier B). This ensures each new strategy has documented evidence and a clear implementation path. Track which catalog entries have been tested.

42. **Compute the Harmony Index for trade conviction sizing.** When using multi-timeframe analysis, compute the weighted alignment score across TFs. Only take full positions when HI > 0.80. Reduce to half-size when HI 0.40-0.79. Stay flat when |HI| < 0.40. This systematically reduces exposure during conflicting-signal periods.

43. **Use session-normalized volume for anomaly detection.** Raw volume thresholds are misleading because average volume varies 3-4x across the day. Compute 30-day rolling average/std for each hour-of-day separately. Z > 2.0 = significant spike; Z < -1.5 = significant drought. This eliminates systematic W-shape bias in intraday volume.

44. **Apply drawdown-based position reduction for strategies with max DD > 20%.** Use the BTC-specific step function: 100% at 0-15% DD, 60% at 15-25%, 30% at 25-35%, flat beyond 35%. Wider thresholds than equities because BTC bull markets routinely see 20-35% pullbacks that are not the end of the trend.

45. **Run MAE analysis before finalizing stop-losses.** Compute P(recovery | MAE = x%) for all historical trades. Set primary stop at 75th percentile of winning-trade MAE, hard stop at 90th percentile. For BTC daily trend strategies, this typically yields 8-12% primary and 15-20% hard stop. Strategy-specific calibration is mandatory.

46. **Add volume confirmation to reduce false signals.** Require bar volume > 1.5x session-normalized 20-bar average before taking trend signals. Expected impact: -30-45% trade count, +3-7pp win rate, +0.1-0.3 Sharpe. Use session-normalized thresholds, not raw volume.

47. **Run performance attribution in Phase 8a before final recommendation.** Decompose returns into BTC beta + momentum factor + volatility factor + alpha. Use Newey-West standard errors. If alpha t-stat < 2.0 under robust SEs, the strategy is factor exposure, not skill. If regime HHI > 0.40, flag as regime bet.

48. **Use wider mean reversion thresholds for BTC.** Standard equity RSI 30/70 thresholds are too tight. Use 20/80 or 25/75 for BTC. Asymmetric: RSI < 25 for longs, RSI > 80-85 for shorts (accounts for structural bullish bias). Always combine with ADX < 25 regime filter — this alone adds +0.3-0.7 Sharpe.

49. **Prefer partial reversion targets over full mean reversion.** Target z-score = -1.0 (partial) instead of z = 0 (full). Partial targets reached ~70-80% of the time vs ~55-65% for full reversion. Similar risk-adjusted performance but lower time exposure and less regime-change risk. Enforce max holding period (3-7 bars on daily).

50. **Restrict trend-following strategies to active sessions.** 08:00-20:00 UTC weekdays produces +0.2-0.5 Sharpe improvement by eliminating low-quality signals during thin-market periods (20:00-04:00 UTC has 1.3-1.8x volatility-per-unit-volume). Mean-reversion strategies can trade all hours but perform best 20:00-08:00 UTC.

51. **Run walk-forward validation on every candidate before deployment.** Use 18-month IS / 3-month OOS windows, minimum 5 segments. Compute WFER (geometric mean of OOS/IS Sharpe ratios). WFER < 0.30 = discard. WFER 0.30-0.50 = paper trade only. Apply the WFER as a haircut multiplier to the backtest Sharpe for realistic expectations.

52. **Check parameter stability via PCR before locking parameters.** Compute Parameter Concentration Ratio across WF segments. If PCR < 0.30 for any key parameter, do not optimize it — use a default value. PCR > 0.70 parameters are stable and can be locked. Always compute stability lift to verify modal IS value predicts OOS performance.

53. **Use CPCV (N=6, k=2) for strategies with >3 free parameters.** Simple WF is insufficient for high-parameter strategies due to look-ahead leakage between adjacent segments. CPCV with embargo = 2.5x max lookback provides 15 independent OOS estimates. All 5 robustness criteria must pass.

54. **Apply the square root impact model when evaluating strategies near the viability threshold.** Strategies with gross edge 10-20 bps per trade may look profitable under Tier 1 flat costs but fail under Tier 2 impact-adjusted costs. Always verify viability at `impact_coefficient = 0.20` with your actual trade notional.

55. **Account for funding drag on perpetual futures strategies.** Any strategy holding through 00:00, 08:00, or 16:00 UTC settlements must include funding cost. Default 0.01%/8h = ~11% annualized. Long-biased trend strategies in bull markets face 13-20%+ annualized drag. Funding drag is not optional — it is a first-order cost.

56. **Construct multi-strategy portfolios with risk parity + regime overlay.** Never equal-weight multiple strategies. Use inverse-vol risk parity as the base, then blend with regime-conditional allocation (ADX-based tanh smoothing, α=0.5 blend). Apply portfolio-level circuit breakers (Yellow at -6% DD, Orange at -10%, Red at -15%).

57. **Monitor strategy decay from day 1 with the 5-metric dashboard.** Track rolling 90-day Sharpe, rolling alpha, rolling IC, CUSUM, and performance vs. envelope. Never wait for "enough data" — Bayesian updating from backtest priors is meaningful from 20-30 trades. IC < 0.02 for 60+ days in a favorable regime = signal has lost predictive content.

58. **Use CUSUM with BTC-adjusted H = 5σ for structural break detection.** Standard H = 4σ produces too many false alarms with BTC's fat tails. Reset CUSUM after each investigation. A CUSUM alarm + IC < 0.02 + favorable regime = confirmed structural decay — halt strategy.

59. **Distinguish "dead strategy" from "waiting for regime" using layered evidence.** No single metric is conclusive in the first 6-12 months (power < 50% at effect size d = 0.34). Layer SPRT on monthly returns, regime fingerprints, CUSUM, and rolling IC. IC is the most direct signal-quality measure — never conclude "waiting for regime" if IC < 0.02 for 60+ days in a favorable regime.

60. **Validate the backtest engine with synthetic data before trusting any strategy result.** Run the 5 synthetic test cases (perfect trend, sine wave, random walk, vol spike, flat market) whenever engine code is modified. Cross-validate against vectorbt (equity curve divergence < 0.5%). Signal values must be exactly {-1, 0, 1} with no NaN. Always pass `closed='left'` explicitly to resample calls.

---

## Sub-Hourly Trading Strategy Research & Considerations

This section documents the complete landscape of considerations for building optimal, profitable, realistic, and sustainable sub-hourly trading strategies (1m, 5m, 15m, 30m). It is based on extensive empirical testing within this project's framework, academic literature, and real-world production trading experience.

**Critical context from this project's empirical testing:**
- ATR Breakout at 1h is the ONLY walk-forward validated strategy across 13+ strategy families
- At 15min/30min, even the best strategies show zero edge after fees — tested across 8 strategy families, 3 assets, 3 fee levels
- Even at ZERO transaction costs, no OHLCV-based strategy has positive Sharpe at 15min
- No crypto pair is cointegrated (BTC/ETH, BTC/SOL, ETH/SOL all fail Engle-Granger across every time window)

---

### Market Microstructure at Sub-Hourly Timeframes

#### Order Book Dynamics & Spread Behavior

For major pairs on Binance (2025-2026 conditions):

| Pair | Typical Spread (Peak Hours) | Typical Spread (Off-Peak) | Spread During Vol Events |
|------|---------------------------|--------------------------|-------------------------|
| BTC/USDT | < 1 bps | 2-5 bps | 10-50x blowout |
| ETH/USDT | 1-3 bps | 5-10 bps | 10-30x blowout |
| SOL/USDT | 3-8 bps | 10-20 bps | 20-100x blowout |

Spreads have narrowed ~40% since 2021 due to institutional participation and professional market makers. However, spread behavior is highly non-stationary:
- **Volatility events** (liquidation cascades, news): Spreads blow out 10-50x within seconds
- **Low-volume hours** (UTC 20:00-00:00): Liquidity thins, spreads widen 2-5x
- **Funding settlement** (every 8h on perps): Temporary microstructure disruptions

#### Timeframe-Specific Microstructure Characteristics

| Timeframe | Dominant Dynamic | Signal Character | Who Profits Here |
|-----------|-----------------|-----------------|-----------------|
| 1m | Order book dynamics, queue position, latency | Almost entirely noise; signals exhausted in seconds | Co-located HFT, market makers |
| 5m | Aggregated order flow, short-term momentum/reversion | Mostly noise; sporadic short-lived signals | Professional quant firms, low-latency infra |
| 15m | Session patterns, volume profile, vol clustering | Signal starts to emerge but overwhelmed by noise + fees | Well-capitalized algo traders, maker-only execution |
| 30m | Trend/counter-trend transitions, session structure | Clearer signals but fewer; still fee-sensitive | Algorithmic traders with decent infrastructure |

#### Tick vs Bar Signals

Bar-based OHLCV signals at sub-hourly suffer from **bar aggregation artifacts**:
- A 1m candle compresses hundreds/thousands of trades into 4 prices + 1 volume number
- The "close" of a 1m candle is essentially a random sample from the last trade — no special significance
- Intra-bar volume distribution is invisible (900 of 1000 BTC in last 5 seconds looks identical to evenly distributed)
- High-low wicks can represent a single large order immediately reversed — bar makes it look like sustained action

Tick-by-tick or order book data provides fundamentally richer information but requires entirely different infrastructure (Level 2/3 feeds, event-driven architecture, real-time order book reconstruction).

#### Market Maker Adverse Selection

Professional market makers on Binance:
- Quote continuously on both sides with tight spreads
- Adjust quotes within 50-200ms of significant price moves
- Use sophisticated models to detect informed order flow (and pull quotes)

**Implication for sub-hourly strategies using market orders**: You are paying the spread to someone who has a sophisticated model of whether you are likely right or wrong. If your signal is based on publicly available OHLCV data, the market maker has already incorporated that information before you can act.

---

### Transaction Cost Reality at Sub-Hourly

#### Binance Futures Fee Schedule (2025-2026)

| Tier | 30d Volume | Maker Fee | Taker Fee | Round-Trip (Maker) | Round-Trip (Taker) |
|------|-----------|-----------|-----------|-------------------|-------------------|
| Regular | < $15M | 0.020% | 0.050% | 4 bps | 10 bps |
| VIP 1 | $15M+ | 0.016% | 0.040% | 3.2 bps | 8 bps |
| VIP 3 | $100M+ | 0.012% | 0.032% | 2.4 bps | 6.4 bps |
| VIP 9 | Top tier | 0.000% | 0.017% | 0 bps | 3.4 bps |

BNB payment provides additional 10% discount on USDT-M contracts.

#### Fee Drag Amplification — The Core Sub-Hourly Problem

This is the single most important concept. The math is relentless:

| Trades/Day | RT Fee (Taker) | Daily Drag | Annualized Drag | Assessment |
|------------|---------------|------------|-----------------|------------|
| 1 | 10 bps | 10 bps | 25% | 1h ATR Breakout territory — manageable |
| 2 | 10 bps | 20 bps | 50% | Feasible if genuine edge exists |
| 5 | 10 bps | 50 bps | 127% | Extremely difficult for retail |
| 10 | 10 bps | 100 bps | 253% | Virtually impossible at taker rates |
| 20 | 10 bps | 200 bps | 506% | Only viable at VIP9 rates |
| 5 | 4 bps (maker) | 20 bps | 50% | Plausible with limit-order-only execution |
| 10 | 4 bps (maker) | 40 bps | 100% | Still very challenging |

**Project empirical confirmation**: ATR Breakout at 15min has 530 trades over 5 years (~0.3/day) with 0-fee Sharpe of 0.29 that drops to -0.27 at 10 bps. Fee impact alone: -0.56 Sharpe. At 1h, the same strategy has 136 trades (27/year) with fee impact of only -0.14 Sharpe.

#### Slippage Modeling

Slippage comes from two sources:
1. **Market impact**: Your order consumes order book liquidity. $10K BTC market order: 0.5-2 bps. $100K: 2-5 bps. $1M+: 10+ bps.
2. **Latency slippage**: Between signal and fill, price moves. ~50ms round-trip from US cloud: 1-5 bps during volatile periods.

Aggregate slippage costs across crypto exceeded $2.7B in 2024 (+34% YoY). Retail traders experience ~40 bps more slippage than institutional traders on average. **Never apply a fixed slippage cost across all trades** — real-world costs fluctuate with volatility, liquidity, and order size.

#### Breakeven Edge Requirements by Timeframe

| Timeframe | Typical Trades/Yr | Total Cost/Trade (bps) | Min Gross Edge/Trade | Min Annual Gross Edge |
|-----------|-------------------|------------------------|---------------------|----------------------|
| 1m | 50,000+ | 12-15 (taker + slippage) | 15+ bps | 7,500%+ |
| 5m | 5,000-15,000 | 11-14 | 14+ bps | 700-2,100% |
| 15m | 1,000-5,000 | 10-13 | 13+ bps | 130-650% |
| 30m | 500-2,000 | 10-12 | 12+ bps | 60-240% |
| 1h | 100-500 | 10-11 | 11+ bps | 11-55% |

#### Funding Rate Impact at Short Holding Periods

On perps, funding rates are charged every 8h. For holds < 8h, impact is proportional:
- **1h hold**: ~1/8 of 8h rate. Typical BTC: 0.01%/8h = 0.00125%/hr. Negligible.
- **Exception**: During extreme conditions, funding spikes to 0.1-0.5%/8h, making even short holds costly on the wrong side.

For sub-hourly strategies, funding is generally minor vs fees/slippage except during extreme conditions.

---

### Signal-to-Noise Ratio Challenges

#### Why Noise Dominates at Shorter Timeframes

BTC daily standard deviation ≈ 1-3%, yielding SNR ≈ 1:30 or worse. As timeframe shrinks:
- **1h bar**: ~2-4% of daily signal but ~4% of daily noise (√24 relationship)
- **15m bar**: ~0.5% of daily signal but ~1% of daily noise
- **1m bar**: ~0.04% of daily signal but ~0.13% of daily noise

Price changes at short intervals are dominated by random microstructure noise — bid-ask bounce, random order arrivals, inventory adjustments. The "true" price signal (new information about value) is a tiny fraction of observed movement. **Approximately 90% of observed data at high frequency is essentially random noise.** This is not a measurement problem — it is a fundamental property of markets.

#### SNR Degradation Formula

```
SNR(target) / SNR(1h) ≈ sqrt(target_minutes / 60)

SNR(30min) / SNR(1h) ≈ sqrt(30/60) = 0.71
SNR(15min) / SNR(1h) ≈ sqrt(15/60) = 0.50
SNR(5min)  / SNR(1h) ≈ sqrt(5/60)  = 0.29
SNR(1min)  / SNR(1h) ≈ sqrt(1/60)  = 0.13
```

If a 1h strategy has SNR of 1.0 (barely tradeable), the equivalent at 15m is 0.50 (essentially random) and at 1m is 0.13 (pure noise). This is a mathematical fact, not a strategy design problem.

#### Minimum Signal Strength

For profitability after fees:
```
Required signal > (fees + slippage) + (noise penalty)
```

The project's empirical data confirms: **even at ZERO fees, no indicator-based strategy has positive Sharpe at 15min on BTC**. The noise penalty alone destroys the edge.

#### Statistical Power Requirements

To distinguish a real Sharpe-S strategy from noise at 95% confidence:

| Target Sharpe | Trades Needed | At 5/day | At 1/day | At 0.1/day (1h) |
|---------------|-------------|---------|---------|-----------------|
| 0.50 | ~1,600 | 320 days | 4.4 years | 44 years |
| 1.00 | ~400 | 80 days | 1.1 years | 11 years |
| 2.00 | ~100 | 20 days | 100 days | 2.7 years |

More trades from sub-hourly = better statistical power, but also more multiple testing opportunities. Standard error of Sharpe ≈ 1/√N where N is independent return periods, not trades.

---

### Data Quality Issues Specific to Sub-Hourly

#### Exchange Timestamp Precision

Binance reports millisecond timestamps, but:
- **Clock sync**: Local vs exchange clock drift: 50-500ms
- **Matching engine**: Order submit → match → report latency: 1-10ms
- **Aggregation**: Multiple fills at same price within milliseconds may report as single trade

At 1h these are irrelevant. At 1m they matter. At sub-minute they are critical.

#### Missing Data & Gaps

Minute-level crypto data has systematic issues:
- **Missing candles**: Exchange outages, API failures, extreme low-volume periods
- **Zero-volume candles**: In less liquid pairs, 1m candles with no trades (OHLC = last trade price)
- **Duplicated data**: Some providers send duplicate records corrupting indicators
- **Exchange quirks**: Different exchanges aggregate edge cases differently

#### Wash Trading Contamination

Severe and underappreciated at sub-hourly:
- Up to **95% of reported Bitcoin volume is fake** on non-regulated exchanges (Bitwise report)
- Over **70% of volume on non-regulated exchanges** identified as wash trading
- **30%+ of DEX token volume** subject to wash trading
- BTC/USDT on Binance is believed lower (surveillance in place) but not zero
- **Any volume-based signal at minute level must be tested with AND without volume confirmation**

#### Data Vendor Discrepancies

Same pair/timeframe from different sources can differ:
- OHLC prices: 1-5 bps depending on aggregation method
- Volume: 10-50% depending on trade type inclusion
- Timestamps: Off by seconds or more

#### Survivorship Bias in Exchange Selection

Backtesting on Binance data implicitly assumes Binance was always dominant. In 2020-2021, BitMEX/Huobi/OKEx had more volume. Strategies optimized for Binance microstructure may not generalize.

---

### Strategy Families That CAN Work Sub-Hourly

#### Market Making / Spread Capture

**How it works**: Post limit orders on both sides. Earn bid-ask spread when both fill. Manage inventory risk.

| Requirement | Detail |
|-------------|--------|
| Execution latency | Sub-millisecond (co-located) |
| Data | Level 2 minimum, real-time order book |
| Capital | $100K-$1M+ per pair |
| Typical Sharpe | 5-15 (at elite firms) |
| Strategy lifespan | Continuous, edge degrades with competition |

**Realistic for retail: NO.** Jump Crypto, Wintermute, Cumberland/DRW operate with co-located servers, custom FPGA hardware, and teams of quant researchers. They capture fractions of a basis point millions of times per day.

#### Cross-Exchange Arbitrage

**Current state (2025-2026)**:
- Price discrepancies of 0.5-3% still occur during volatility events
- Spreads typically last 200-800ms before arbitrage
- Operational traders average 15-25 opportunities/day, 60-65% success rate
- Average profit per successful trade: $20-35 (on modest capital)
- Monthly profit: $7,500-12,000 (requires $200K+ deployed across exchanges)

**Infrastructure reality**: A $220K infrastructure upgrade (89ms → 42ms latency) increased arb hit rate from 23% to 61%. The 47ms improvement was worth ~$180K/year in captured opportunities.

**Realistic for retail: Marginally**, with significant capital and infrastructure. Edges are shrinking rapidly.

#### Statistical Arbitrage / Pairs at Micro Level

**Academic claims**: Copula-based studies claim Sharpe 3.77 at 5min for crypto pairs.

**Project empirical reality**: Every crypto pair tested failed Engle-Granger cointegration. Best result: ETH/SOL at 5min, Sharpe 0.50 at zero fees → Sharpe -25.98 at 10 bps.

**Gap**: Papers likely used cherry-picked periods, more sophisticated methods (copula, Kalman filter) that overfit, or unrealistic cost assumptions.

#### Order Flow / Tape Reading

**Academic support**: Order flow imbalance has near-linear relationship with short-horizon price changes (Cont, Kukanov, Stoikov). But they explicitly note: **the predicted price change is well below the bid-ask spread** — the signal alone does not offer arbitrage.

**Project result**: CVD divergence at 15m WF result: WFER -7.54, OOS Sharpe -1.01. In-sample Sharpe 0.48 was pure overfit.

**Realistic for retail**: **Unlikely standalone alpha from OHLCV**. May add +0.2-0.4 Sharpe as filter for primary strategy, but genuine order flow strategies require Level 2/3 data, real-time processing, sub-second execution.

#### Volatility Breakout on News Events

Essentially what ATR Breakout does. Project data: ATR Breakout at 15min: 0-fee Sharpe 0.29, 10bps Sharpe -0.27. Continuation after breakout is not long enough at sub-hourly to overcome costs.

#### Academic Literature Summary

- Evidence of both intraday momentum and reversal in crypto
- Intraday momentum explained by late-informed investors; reversal by overreaction
- **But**: Most papers assume zero/minimal costs, or test at frequencies where cost impact isn't modeled
- Honest assessment: Prediction horizon and order book depth have greater performance impact than model complexity. Simpler models training in fraction of time are more practical for real-time use.

---

### Execution Infrastructure Requirements

#### Latency Requirements by Timeframe

| Timeframe | Acceptable Latency | Infrastructure Tier |
|-----------|-------------------|-------------------|
| 1m (HFT/MM) | < 1 ms | Co-located, FPGA, custom networking |
| 5m | < 50 ms | Cloud in exchange region (Tokyo/Singapore for Binance) |
| 15m | < 500 ms | Any cloud VPS with decent connectivity |
| 30m | < 2 seconds | Home server or basic cloud |
| 1h | < 10 seconds | Anything |

Crypto exchange order-to-execution: 20-500ms depending on infrastructure. Co-location: < 1ms.

#### Binance API Rate Limits

- **Orders**: 1,200/minute (default), 300 per 10 seconds
- **REST weight**: 2,400 weight/minute
- **WebSocket**: 5 connections/IP, 300 subscriptions/connection
- **Violation consequence**: HTTP 429 → temporary IP ban → potential permanent ban
- **During market stress**: Exchanges may further reduce limits or disable API access

For 15m strategy (5 trades/day): Irrelevant. For 1m strategy (100+ trades): Rate management is critical.

#### Order Types by Strategy

| Strategy Type | Required Orders | Fill Rate Assumption |
|--------------|----------------|---------------------|
| Taker momentum | Market / IOC | ~100% fill, pay taker fee + slippage |
| Maker mean-reversion | Limit / Post-Only | 50-80% fill (distance from mid dependent) |
| Market making | Limit / Post-Only / GTT | 30-70% fill per side, need both sides |
| Breakout | Stop-Market / Stop-Limit | Variable; stop-limit may not fill during gaps |

**Critical backtest assumption**: Most backtests assume fills at bar close / next bar open. Reality: market orders fill at current price (moves between signal and execution), limit orders may never fill. Discrepancy amplified at shorter timeframes.

#### Queue Position for Limit Orders

For maker-fee strategies, **queue position is everything**:
- Limit at best bid waits for all orders ahead before filling
- If price moves away: never fills
- If price moves through: fills but you're on wrong side (adverse selection)
- Only if price touches your level and reverses: limit order adds value

The `hftbacktest` framework (Rust/Python) implements probability-based queue position models. **Without queue position modeling, limit order backtest results are meaninglessly optimistic.**

---

### Realistic Edge Expectations

#### Achievable Sharpe by Category

| Category | Timeframe | Typical Sharpe | Infrastructure Required |
|----------|-----------|---------------|----------------------|
| Elite HFT market making | Sub-second | 5-15+ | $1M+ infra, PhD team |
| Professional cross-exchange arb | 1-30 seconds | 3-8 | $200K+ infra, multi-exchange |
| Professional stat arb | 1-15 min | 2-5 | $100K+ infra, Level 2 data |
| Sophisticated retail | 15-30 min | 0-2 (most likely 0) | VPS, API access, maker-only |
| Retail algo (proven) | 1h+ | 0.5-1.5 | Basic infrastructure |

The project's validated ATR Breakout multi-asset + vol targeting at Sharpe 1.32 is **excellent for retail algorithmic trading**.

#### Alpha Decay Rates by Frequency

| Strategy Type | Typical Lifespan | Recalibration Frequency |
|--------------|-----------------|------------------------|
| HFT market making | Days to weeks | Continuous / daily |
| Momentum / stat arb (5-15m) | 3-6 months | Weekly to monthly |
| Trend following (1h+) | Years | Quarterly to annually |

Profitable sub-hourly signals can become stale before a trade is even triggered. Strategies that once delivered returns are being arbitraged in hours.

#### Capacity Constraints

| Strategy Type | Deployable Capital | Limiting Factor |
|--------------|-------------------|----------------|
| Market making BTC/USDT | $10M-$100M+ | Being the liquidity |
| Cross-exchange arb | $200K-$2M per pair | Smaller exchange liquidity |
| Stat arb 5-15m | $50K-$500K | Market impact degrades returns |
| Breakout 15-30m | $100K-$5M | Instrument liquidity |
| ATR Breakout 1h (validated) | $1M-$10M+ | Infrequent trades, long holds |

---

### Backtesting Pitfalls Specific to Sub-Hourly

#### Look-Ahead Bias Amplification

At short timeframes, look-ahead bias becomes more pernicious:
- **Within-bar execution**: Assuming trade at bar close when close is only known after bar completes
- **Future volume**: Using current bar's volume for decisions before volume is known
- **Post-trade confirmation**: Using trade data milliseconds after signal

At 1h, these biases are relatively minor. At 1m, the entire bar's information is potentially look-ahead — close price can move 5-10 bps from any intra-bar point.

#### Fill Assumption Errors

The single most dangerous assumption in sub-hourly backtesting:
- **Market orders**: See mid-price at signal, fill at mid + half-spread + slippage (2-10 bps worse)
- **Limit orders**: Backtests assume 100% fill. Reality: 50-80% near mid, 30-50% further away. **Unfilled orders are never counted as losses but they remove the most profitable trades from the sample**
- **Stop orders**: During fast moves, fill 10-50 bps worse than stop price

#### Bar Aggregation Artifacts

When resampling 1m → 5m/15m:
- Resampled OHLC may not represent real-time experience
- Volume aggregation hides intra-bar distribution
- A 15m "clean breakout" may actually have been a whipsaw within the bar
- Indicators on resampled bars differ from tick-by-tick computation

#### Overfitting Risk With More Data Points

Counterintuitively, more sub-hourly data points **increase** overfitting risk:
- 5 years of 1m data = ~2.6M bars — enormous search space for spurious patterns
- A strategy with 20 parameters can find incredible performance that is pure noise
- More trades = more multiple testing opportunities

Mitigation: walk-forward validation, CPCV, strict parameter count limits (already in this framework).

---

### Risk Management Differences at Sub-Hourly

#### Position Sizing

- The standard 2% risk per trade is too much for high-frequency crypto. Use **0.5-1% max per trade**.
- **Dynamic sizing based on order book depth**: Thin book → reduce size. Deep book → increase.
- Kelly Criterion requires accurate win rate / W:L estimates — harder at short TFs due to noise.

#### Drawdown Dynamics

- **More frequent, shallower drawdowns**: 1m strategy might have 20 small drawdowns/day vs one large one
- **But**: Small losses compound — 20 trades losing 10 bps each = 2% daily loss
- **Flash crash risk**: Intra-bar flash crashes that risk management can't catch in time

#### Kill Switch Requirements (Mandatory)

Every sub-hourly system MUST have:

1. **Max daily loss**: Halt all trading if daily P&L < -2%
2. **Max consecutive losses**: Halt after 5-10 consecutive losers
3. **Latency circuit breaker**: If API latency exceeds threshold, halt (prevents trading on stale data)
4. **Exchange status monitor**: If exchange reports degraded performance, halt
5. **Position size hard cap**: Maximum position regardless of signal

#### Correlation With Longer-Term Strategies

Sub-hourly SHOULD be uncorrelated with 1h+ strategies for diversification. In practice:
- Correlated during volatility events (everything loses together)
- Diversification exists primarily during calm periods — when least needed
- Running both 1h ATR Breakout and 15m strategy on same instrument introduces market exposure correlation

---

### Regulatory & Practical Considerations

#### Exchange Order-to-Trade Rules

Crypto exchanges are less regulated than equities, but:
- Binance monitors "excessive API usage" — can ban accounts
- Repeated 429 errors → automated IP bans → potential permanent ban
- Some exchanges have informal cancellation rate rules
- Evolving regulatory frameworks (CLARITY Act 2025) may introduce formal rules

#### Tax Implications (US, 2025-2026)

- Centralized exchanges must report to IRS on Form 1099-DA (as of Jan 1, 2025)
- **Wash sale rule does NOT currently apply to crypto** (2025 tax year) — expected to change
- If wash sale applies: high-frequency crypto trading becomes massively more complex (30-day windows across all accounts/exchanges)
- 5,000+ trades/year = massive reporting complexity (each needs acquisition date, USD FMV, cost basis)

#### Operational Complexity

Sub-hourly requires:
- **24/7 monitoring** (crypto never closes)
- **Multiple redundancy layers** (server crash at 3 AM?)
- Regular parameter refresh
- Ongoing data quality checks
- Exchange API change management (frequent updates)

---

### What Makes Sub-Hourly Strategies Fail in Practice

#### The "Backtest Hero, Live Zero" Phenomenon

Common failure modes:

1. **Overfitting**: Perfect equity curves in backtest (200% return, < 5% DD) = almost certainly wrong
2. **Unrealistic fills**: Backtests assume stated prices; reality has slippage, partial fills, missed entries. Strategy showing 2% monthly in backtest may lose money after 0.05% slippage + 0.1% cost per trade
3. **Regime change**: Many strategies work in one regime but fail in others. Project example: CVD Divergence Sharpe 3.22 in one sub-period, -0.71 in the next
4. **Latency/execution gap**: In backtest execution is instant. Live, there's a gap that consistently works against you (the better the trade, the more others want it, the more price moves)
5. **Insufficient sample**: Testing with 20-30 trades = unreliable. Project: Vol Squeeze V2 at 30min showed Sharpe 1.00 with only 34 trades in 2 years — statistically meaningless

#### Regime Sensitivity at Short Timeframes

Sub-hourly strategies are MORE regime-sensitive than longer-term because:
- Market microstructure changes faster than macro regimes
- A new market maker entering/exiting a pair changes order book dynamics overnight
- Regulatory changes (listing/delisting, leverage limits) have immediate microstructure effects
- Liquidity fragmentation shifts between exchanges on shorter timescales

#### Competition From Professional HFT

At sub-hourly timeframes, you compete with:

| Firm | Infrastructure | Capital |
|------|---------------|---------|
| Jump Crypto | Co-located, FPGA-based | Hundreds of millions |
| Wintermute | Largest crypto market maker | Billions daily processing |
| Cumberland / DRW | Decades of HFT experience | Institutional scale |
| Tower, Citadel, Jane Street | Traditional HFT expanding to crypto | Multi-billion |

These firms have sub-millisecond latency (you have 50-500ms), Level 3 data (you have OHLCV), teams of 20-100+ researchers (you have a framework and time). **Any signal detectable from OHLCV at sub-hourly has almost certainly been found, exploited, and arbitraged away by these firms.**

---

### Minimum Viable Infrastructure for Sub-Hourly

#### For 15-30m OHLCV-Based Strategies (Most Realistic Retail Option)

| Component | Minimum | Recommended | Cost/Month |
|-----------|---------|-------------|-----------|
| Compute | 2 vCPU, 4 GB RAM VPS | 4 vCPU, 8 GB RAM VPS | $20-80 |
| Location | Any US/EU cloud | AWS Tokyo / Singapore (near Binance) | $40-120 |
| Data feed | REST API polling | WebSocket real-time candle stream | Free |
| Monitoring | Email alerts | PagerDuty / Grafana + Prometheus | $0-50 |
| Redundancy | Single server | Primary + failover in different AZ | 2x cost |
| **Total** | | | **$60-400** |

#### For 1-5m Order Flow Strategies (Professional Tier)

| Component | Minimum | Cost/Month |
|-----------|---------|-----------|
| Compute | 8+ vCPU, 32 GB RAM, NVMe SSD | $200-500 |
| Location | Singapore/Tokyo proximity hosting | $500-2,000 |
| Data feed | WebSocket full order book (Level 2) | Free-$500 |
| Monitoring | Real-time dashboard + kill switches | $100-300 |
| Redundancy | Hot standby with auto failover | 2x compute |
| **Total** | | **$1,000-5,000** |

#### For HFT / Market Making (Institutional Tier)

| Component | Minimum | Cost/Month |
|-----------|---------|-----------|
| Compute | Custom hardware, FPGA, bare metal | $5,000-20,000 |
| Location | Co-located at exchange datacenter | $2,000-10,000 |
| Data feed | Level 3 direct feed | Exchange partnership |
| Team | 2-5 quant devs + devops | $40K-$170K |
| **Total** | | **$50,000+** |

#### Cloud vs Local Execution

| Option | Latency to Binance | Suitable For | Main Risk |
|--------|-------------------|-------------|-----------|
| Cloud (AWS Singapore/Tokyo) | 20-50ms | 15m+ strategies | Provider outages |
| Proximity hosting | 1-5ms | 1-5m strategies | Single point of failure |
| Co-location | < 1ms | HFT/MM | Exchange-specific, not portable |
| Home server | 50-200ms | 30m+ only | ISP/power/hardware failure |

#### Disaster Recovery Considerations

- **Server crash**: At 1h, restart within minutes, miss at most one signal. At 1m, open positions need immediate management.
- **Exchange API down**: Need secondary exit path (manual web interface at minimum, secondary exchange ideally).
- **IP ban**: Need backup IPs, possibly VPN layer.
- **Flash crash while offline**: Hard stop losses must exist **on the exchange side** (exchange-hosted stops, not local) so they execute even if system is down.

---

### Sub-Hourly Protocol: Decision Framework

Before pursuing any sub-hourly strategy, answer these questions:

```
1. DATA SOURCE: Am I using OHLCV or order book data?
   → OHLCV only: STOP. Project data shows no viable sub-hourly OHLCV strategy exists.
   → Order book / Level 2+: PROCEED with caution.

2. EXECUTION: Can I achieve maker-only execution?
   → Taker only: Fee drag likely exceeds any edge at < 1h.
   → Maker possible: Reduces costs 60%, opens some possibility.

3. INFRASTRUCTURE: Can I deploy < 50ms from exchange?
   → > 200ms: Only 30m+ is realistic.
   → 50-200ms: 15m possible with maker execution.
   → < 50ms: 5m strategies become feasible.
   → < 1ms: Full HFT/MM spectrum available.

4. CAPITAL: How much am I deploying?
   → < $50K: Sub-hourly unlikely to justify infrastructure costs.
   → $50K-$200K: 15-30m with maker fees, tight cost control.
   → $200K+: Cross-exchange arb, stat arb become viable.
   → $1M+: Market making becomes feasible.

5. TIME COMMITMENT: Can I maintain 24/7 monitoring?
   → No: Do not run sub-hourly strategies.
   → Yes: Ensure redundancy, kill switches, disaster recovery.

6. OPPORTUNITY COST: Could this effort improve the 1h system instead?
   → Almost always yes. Consider: more assets, better execution
     (maker fees), on-chain signals at 1h, rebalancing optimization.
```

### Sub-Hourly Validation Requirements (In Addition to Standard 8-Phase Protocol)

Any sub-hourly strategy must pass ALL of the following before consideration:

1. **Fee sensitivity at 4 levels** (not 3): Optimistic maker, realistic maker, realistic taker, pessimistic taker. Must be profitable at realistic taker level.
2. **Fill rate modeling**: If using limit orders, apply 60% fill rate assumption and verify profitability with missed fills converting to next-bar market orders.
3. **Latency simulation**: Add 50-200ms random delay to signal execution (depending on target infrastructure). Re-run all metrics.
4. **Microstructure regime test**: Split data into high-spread and low-spread regimes. Strategy must be profitable in both (microstructure conditions can change overnight).
5. **Minimum 500 trades in OOS**: Sub-hourly generates more trades, so demand proportionally more OOS evidence.
6. **Alpha decay test**: Compare first-half vs second-half OOS performance. If second-half Sharpe < 50% of first-half, assume the edge is decaying.
7. **Capacity test**: Run at 2x and 5x the intended position size. If Sharpe degrades > 20% at 2x, the strategy has capacity constraints.

### Sub-Hourly Rules of Engagement

61. **Never assume sub-hourly OHLCV strategies will work.** The project has empirically proven that no OHLCV-based strategy at 15m or below has positive Sharpe after fees across BTC, ETH, or SOL. Only pursue sub-hourly with fundamentally different data (order book, Level 2+) or execution advantages (maker-only, co-location).

62. **Model fills realistically at sub-hourly.** Assume market orders fill 2-10 bps worse than signal price. Assume limit orders fill only 50-80% of the time. Apply random 50-200ms latency delay. Unfilled limit orders that would have been profitable are invisible losses — account for them.

63. **Fee drag is the binding constraint, not signal quality.** Even if you discover a sub-hourly signal with genuine edge, the fee drag from trade frequency will likely consume it. Calculate annualized fee drag FIRST, before evaluating signal quality. If drag > 100% annualized at your fee tier, the strategy needs fewer trades or lower costs.

64. **Require 4-level fee sensitivity for sub-hourly.** Standard 3-level test is insufficient. Add a fourth level: maker-only with 60% fill rate. This reflects the realistic cost floor for sub-hourly strategies. Breakeven fee rate must exceed your actual total cost (fees + spread + slippage + market impact).

65. **Test microstructure regime stability.** Split data by spread regime (high-spread days vs low-spread days), volume regime (high-vol vs low-vol sessions), and market maker activity (proxy: trade count per bar). A sub-hourly strategy that only works in one microstructure regime will fail when conditions shift — which happens frequently and without warning.

66. **Infrastructure cost must be subtracted from returns.** A sub-hourly strategy returning 50% annually but requiring $3,000/month ($36K/year) in infrastructure only breaks even at $72K capital. Include infrastructure as a fixed cost drag in the strategy evaluation. Compare net-of-infrastructure Sharpe to the 1h ATR Breakout alternative.

67. **Apply the competition test.** Before investing in sub-hourly infrastructure, ask: "Is this signal available to co-located HFT firms with Level 3 data and sub-millisecond execution?" If yes, you are competing with firms that have 100-1000x your speed and data advantage. Focus on signals that require domain knowledge, alternative data, or longer time horizons where speed advantage is irrelevant.

68. **Queue position modeling is mandatory for limit-order sub-hourly strategies.** Without it, backtest results for maker-fee strategies are meaninglessly optimistic. Use probability-based queue models (e.g., `hftbacktest` framework). At minimum, apply a 60% fill rate haircut and convert unfilled orders to next-bar market fills.

69. **Sub-hourly alpha decays 2-5x faster than hourly alpha.** Budget for strategy replacement every 3-6 months rather than annually. Build parameter refresh into the operational cadence from day one. Monitor IC (Information Coefficient) weekly — at sub-hourly, IC < 0.02 for 2 weeks (not 60 days) in a favorable regime signals decay.

70. **Always compute opportunity cost vs the validated 1h system.** Time and capital spent on sub-hourly research has an opportunity cost. The 1h multi-asset ATR Breakout (Sharpe 1.32, MaxDD -21%, +198% return) is already validated and deployable. Improvements to that system (more assets, maker execution, on-chain overlays) likely have higher expected value than finding sub-hourly edge from scratch.

---

## Quantitative Foundations for Sub-Hourly Trading

This section documents the mathematical models, statistical methods, machine learning approaches, and cutting-edge research used by professional quant firms for sub-hourly trading. It provides the theoretical toolkit that any serious sub-hourly strategy development effort must engage with.

**Full research documents** are available at:
- `.planning/research/quant-microstructure-foundations.md` — Microstructure models, order book theory, volatility estimation, stat arb math, ML/signal processing
- `.planning/research/quantitative-methods-sub-hourly-trading.md` — Execution algorithms, alpha signals, advanced statistics, portfolio construction, risk management, backtesting methodology

---

### Market Microstructure Models — Key Results

#### Kyle (1985) — Price Impact of Informed Trading

Kyle's lambda is the permanent price impact per unit of order flow:

```
lambda = (1/2) * sqrt(Sigma_0) / sigma_u

where:
  Sigma_0 = variance of asset's true value (information uncertainty)
  sigma_u = std dev of noise trader order flow (camouflage)
```

The informed trader's optimal strategy: `x = sigma_u * (v - mu) / sqrt(Sigma_0)` — trade proportionally to information edge, scaled inversely by impact. In continuous time, trading intensity `beta_t = 1/(1-t)` (more aggressive near deadline).

**Crypto implications**: 24/7 trading removes deadline urgency. Fragmented multi-exchange flow complicates inference. Cross-asset information leakage (BTC info → altcoins) is real and measurable.

#### Glosten-Milgrom (1985) — Spread as Adverse Selection

Bid-ask spread exists purely due to adverse selection:
```
ask = E[V | buy order arrived]    (Bayesian conditional expectation)
bid = E[V | sell order arrived]

Spread > 0 whenever fraction of informed traders pi > 0
```

**Key result**: Transaction prices form a martingale — you cannot predict future prices from past prices in this framework. Every trade that crosses the spread pays an "adverse selection tax."

**Crypto spread reality**: BTC/USDT ~1-2 bps (high pi from insiders on listings/upgrades); long-tail altcoins 50+ bps.

#### Almgren-Chriss (2000) — Optimal Execution

Liquidate X shares over [0, T] minimizing mean-variance cost:

```
Optimal trajectory: x*(t) = X * sinh(kappa * (T-t)) / sinh(kappa * T)

where: kappa = sqrt(lambda * sigma^2 / eta)
  lambda = risk aversion
  sigma  = volatility
  eta    = temporary impact coefficient

Limiting cases:
  Risk-neutral (lambda → 0): TWAP (constant rate)
  Infinitely risk-averse (lambda → ∞): immediate execution (block trade)
```

**Key insight**: Permanent impact cost is identical regardless of trajectory. Only temporary impact and timing risk matter for optimization.

**Crypto adaptation**: 24/7 eliminates overnight gap risk. Cross-exchange execution requires multi-venue extensions. Crypto order books are thinner → higher temporary impact coefficients.

#### Avellaneda-Stoikov (2008) — Optimal Market Making

Market maker maximizes expected utility via HJB equation. Two key results:

```
Reservation price: r = s - q * gamma * sigma^2 * (T - t)
  (shifts mid-price reference based on inventory: long → lower price to encourage sells)

Optimal spread: delta_a + delta_b = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
  Component 1: inventory risk premium (wider when more vol or time)
  Component 2: profit from order flow
```

Fill probability decays exponentially with distance from mid: `lambda(delta) = A * exp(-kappa * delta)`.

**Crypto**: Implemented in Hummingbot for automated market making on CEX/DEX.

---

### Order Book Modeling — Key Frameworks

#### Cont-Stoikov-Talreja (2010) — LOB as Markov Chain

Models the limit order book as continuous-time Markov chain with Poisson arrivals:
- Limit order arrival: rate λ per level
- Cancellation: rate θ × queue_size (proportional)
- Market order: rate μ (removes from best queue)

Computes via Laplace transforms: P(mid-price increase before decrease), P(ask fills before bid moves), P(both sides fill before price moves). Critical for estimating limit order fill probabilities.

#### Hawkes Processes — Self-Exciting Order Flow

```
Conditional intensity: lambda*(t) = mu + sum_{t_i < t} alpha * beta * exp(-beta * (t - t_i))

Branching ratio: n = alpha  (must be < 1 for stability)
  n ~ 0.6-0.8 for equities
  n ~ 0.6-0.9 for liquid crypto

Interpretation: ~60-80% of order flow is ENDOGENOUS (self-excited)
```

4-dimensional Hawkes captures: mid-price up/down jumps + buy/sell market orders. The excitation matrix encodes:
- Self-excitation (diagonal): momentum — trades beget trades
- Cross-excitation (off-diagonal): resilience — buys trigger sell limit orders

**MLE**: O(n) recursive algorithms for exponential kernels. Flash crash modeling uses near-critical processes (n → 1).

#### Square-Root Law of Market Impact (Universal Empirical Fact)

```
I(Q) ~ sigma * Y * sqrt(Q / V_daily)

where: Y ~ 1 (dimensionless), Q = metaorder size, V_daily = daily volume
```

**This is the single most robust empirical fact in microstructure** — confirmed across equities, futures, FX, and crypto. Exponent consistently ~0.5 (range 0.4-0.7). After metaorder completion: ~2/3 of peak impact is permanent ("the 2/3 rule").

---

### Volatility Microstructure — Estimation Under Noise

#### Range-Based Estimators (Work Well at 15m-1h for Crypto)

```
Parkinson:      σ²_P  = (1/(4n*ln2)) * Σ(ln(H/L))²           [5x more efficient than close-to-close]
Garman-Klass:   σ²_GK = (1/n) * Σ[0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²]
Rogers-Satchell: σ²_RS = (1/n) * Σ[ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)]   [drift-independent]
Yang-Zhang:     σ²_YZ = σ²_O + k*σ²_C + (1-k)*σ²_RS          [gold standard: min variance, drift-independent]
  k = 0.34 / (1 + (n+1)/(n-1))
```

Below 5m bars, microstructure noise becomes problematic for all estimators. For crypto (no auction effects), range estimators work well at 15m-1h.

#### The Microstructure Noise Problem

Standard realized variance at high frequency is biased upward:
```
E[RV_n] = ∫σ² ds + 2*n*ω²

Bias term 2*n*ω² DIVERGES as sampling frequency n increases!
```

**Bandi-Russell optimal sampling frequency**: `n* ~ (IV / (4*ω⁴))^(1/3)` — typically 5-15 min for equities, 1-5 min for liquid crypto.

#### Solutions: Noise-Robust Volatility Estimators

| Estimator | Formula | Rate | Key Property |
|-----------|---------|------|-------------|
| **TSRV** (Zhang-Mykland-Ait-Sahalia 2005) | `TSRV = RV_slow - (n̄/n) * RV_fast` | n^(-1/6) | Two-frequency cancellation |
| **Multi-Scale RV** (Zhang 2006) | `MSRV = Σ a_j * RV^(K_j)` | n^(-1/4) | Optimal weights across M timescales |
| **Realized Kernels** (BNHLS 2008) | `RK = Σ k(h/(H+1)) * Γ_h` | n^(-1/4) | Parzen kernel recommended |
| **Pre-Averaging** (Jacod et al. 2009) | `PAV = Σ Ȳ_i² - bias` | n^(-1/4) | Simpler implementation, same rate |

**Practical recommendation**: For this framework at sub-hourly, use Yang-Zhang for range-based estimation (15m+ bars) or TSRV for tick-level realized vol.

#### Jump Detection at High Frequency

```
Bipower Variation: BV = (π/2) * Σ|r_i| * |r_{i-1}|
  → Converges to integrated variance WITHOUT jumps (even when jumps present)
  → Jump Variation = RV - BV

Lee-Mykland (2008): L_i = |r_i| / σ̂_i
  → Under no-jump null, max(L_i) → Gumbel distribution
  → Identifies exact jump TIMING, SIZE, and DIRECTION
  → Works at 5m or finer
```

Crypto jumps: more frequent than equities (flash crashes, liquidation cascades), exhibit clustering (Hawkes-modeled arrivals), and cross-asset contagion (BTC → altcoins within seconds).

#### Rough Volatility in Crypto

BTC volatility is confirmed "rough" with Hurst exponent **H ~ 0.1-0.15** (matching equities). The RFSV model `log σ_t = ν * W_t^H + f(t)` yields better short-term vol forecasts than GARCH. However, Bitcoin shows stronger **multifractality** — H varies across timescales and regimes; a single H may be insufficient.

---

### Statistical Arbitrage — Mathematical Toolkit

#### Ornstein-Uhlenbeck Process

```
dX_t = μ*(θ - X_t)*dt + σ*dB_t

Half-life = ln(2) / μ
Stationary distribution: X ~ N(θ, σ²/(2μ))

Discrete estimation (AR(1)):
  X_{t+1} = a + b*X_t + ε
  μ = -ln(b)/Δt,  θ = a/(1-b),  half-life = -ln(2)/ln(b)
```

**Bertram (2010) optimal thresholds**: Maximize expected return per unit time using first-passage times. Under zero costs, entry/exit symmetric around mean. With costs, thresholds widen proportionally.

#### Kalman Filter for Dynamic Hedge Ratios

```
Observation: y_t = β_t * x_t + ε_t
State:       β_t = β_{t-1} + η_t       (random walk hedge ratio)

Predict: β_{t|t-1} = β_{t-1|t-1},  P_{t|t-1} = P_{t-1|t-1} + W
Update:  K_t = P_{t|t-1}*x_t / (x_t²*P_{t|t-1} + V)
         β_{t|t} = β_{t|t-1} + K_t*(y_t - β_{t|t-1}*x_t)

Trading z-score: z_t = (y_t - β_{t|t-1}*x_t) / sqrt(x_t²*P_{t|t-1} + V)
```

Advantages over rolling OLS: no window parameter, smooth adaptation, built-in uncertainty estimates, natural Bollinger bands from forecast error variance.

#### Cointegration Beyond Engle-Granger

| Test | What It Does | When to Use |
|------|-------------|-------------|
| **Johansen** | Multivariate cointegration via VECM rank | >2 variables |
| **Phillips-Ouliaris** | Correct asymptotics for residual-based tests | More reliable small samples |
| **Gregory-Hansen (1996)** | Cointegration WITH unknown structural break | When pairs may be cointegrated within regimes |
| **FCVAR** | Fractional cointegration (I(d) with 0<d<1) | "Almost cointegrated" with long memory |

**Gregory-Hansen protocol**:
1. Run standard ADF and Gregory-Hansen
2. Both reject → cointegration without break
3. Only GH rejects → cointegration WITH structural break (important!)
4. Neither rejects → no cointegration

**FCVAR (Fractional Cointegration)**: Standard tests require I(1)→I(0). FCVAR allows I(d)→I(d-b) where 0<b<d. Spreads that are I(0.3) or I(0.5) are still mean-reverting, just slowly. Studies show **fractional cointegration outperforms standard cointegration for crypto pairs trading**, with fewer trades and higher Sortino.

**Actionable for this project**: Retest BTC/ETH, BTC/SOL with Gregory-Hansen (may find regime-dependent cointegration) and FCVAR (may find "almost cointegrated" with long memory that standard tests rejected).

---

### Alpha Signal Construction at High Frequency

#### Order Flow Imbalance (Cont-Kukanov-Stoikov 2014)

```
OFI_t = (Δbid_depth * I(bid≥prev_bid) - Q_bid * I(bid<prev_bid))
      - (Δask_depth * I(ask≤prev_ask) - Q_ask * I(ask>prev_ask))

Linear model: ΔP_t = β * OFI_t + ε_t
  → β inversely proportional to market depth
  → Robust across timescales (seconds to minutes)
  → Short-term price changes are MAINLY driven by OFI, not trade size
```

OFI exhibits positive autocorrelation at short timescales in crypto — exploitable for execution improvement.

#### VPIN (Easley-Lopez de Prado-O'Hara 2012)

Volume-synchronized probability of informed trading:
```
Step 1: Partition trades into volume buckets of size V̄ (1/50 of expected daily vol)
Step 2: Bulk Volume Classification: V_buy = V̄ * Φ((P_τ - P_{τ-1})/σ_P)
Step 3: VPIN = Σ|V_sell - V_buy| / (n * V̄)

VPIN → 0: balanced flow (low toxicity)
VPIN → 1: highly one-sided (informed trading likely)
```

VPIN spikes precede volatility events and liquidation cascades. Volume clock is naturally suited to 24/7 crypto. **Controversy**: Andersen-Bondarenko (2014) showed VPIN is mechanically correlated with trading intensity, but extreme readings remain useful as regime indicators.

#### Funding Rate as Predictive Signal

```
No-arbitrage perpetual price: F_t = [κ/(κ-(r-r'))] * S_t  (Ackerer-Hugonnier-Jermann)

Predictive threshold: |z-score(funding)| > 2 → mean-reversion over 1-24 hours
Average crypto carry: ~7-8% annualized (BIS Working Paper 1087)
```

Extreme funding predicts mean-reversion but with time-varying predictability (weakens in trending markets). Kyle's lambda spikes around funding rate snapshots.

#### Price Discovery Across Venues

```
Hasbrouck Information Share (VECM-based):
  Binance perpetual futures: ~55-65% IS for BTC (dominant)
  CME futures: gaining share (institutional adoption)
  Spot markets: lead for smaller cap altcoins
  DEXs (Uniswap): ~5-10% IS (followers, not leaders)
```

Lead-lag between spot and perps provides 1-5 second alpha windows — exploitable only with low-latency infrastructure.

---

### Advanced Statistical Methods

#### Bayesian Online Changepoint Detection (Adams-MacKay 2007)

Maintains run-length posterior via message passing:
```
Growth:      p(r_t=l, x_{1:t}) = p(r_{t-1}, x_{1:t-1}) * π_{t-1}^(l) * (1-H(r_{t-1}))
Changepoint: p(r_t=0, x_{1:t}) = Σ p(r_{t-1}, x_{1:t-1}) * π_{t-1}^(l) * H(r_{t-1})

Hazard function H controls changepoint prior:
  Constant H = 1/λ → geometric prior with expected regime duration λ
```

**Fully online** (O(1) per update), produces predictive distribution weighting both "regime continues" and "regime changed" hypotheses. Superior to batch changepoint detection for real-time regime detection.

#### Copula Models for Tail Dependence

```
Sklar's theorem: F(x₁,...,x_d) = C(F₁(x₁),...,F_d(x_d))

Key copula families for crypto:
  Clayton:   C(u₁,u₂) = (u₁^{-θ} + u₂^{-θ} - 1)^{-1/θ}  → LOWER tail dependence (crash co-movement)
  Student-t: symmetric tail dependence with df ν
  Gaussian:  NO tail dependence (underestimates crash risk)

BTC-ETH: Clayton captures the ~0.85 → ~0.95+ correlation jump during crashes
```

For 3+ assets, vine copulas (C-vine, D-vine, R-vine) build high-dimensional dependence from bivariate building blocks.

#### Extreme Value Theory for Crypto Tails

```
Generalized Pareto Distribution (peaks over threshold):
  Pr(X-u ≤ y | X>u) → G(y) = 1 - (1 + ξ*y/σ_u)^{-1/ξ}

Bitcoin tail exponents: ξ ~ 2.0-2.5
  → Variance exists but kurtosis may be infinite
  → 5-sigma events happen ~100x more often than Gaussian predicts

EVT-based VaR: VaR_α = u + (σ_u/ξ)*((n/N_u*(1-α))^{-ξ} - 1)
EVT-based ES:  ES_α = VaR_α/(1-ξ) + (σ_u - ξ*u)/(1-ξ)
```

**Always use EVT or empirical distributions for crypto risk — never Gaussian assumptions.**

---

### Machine Learning: What Actually Works vs. Hype

#### DeepLOB and LOB Prediction (State of the Art 2025)

| Model | Architecture | FI-2010 F1 | Crypto F1 | Year |
|-------|-------------|-----------|----------|------|
| DeepLOB | CNN + LSTM | 83.4% | ~55-60% | 2019 |
| TransLOB | CNN + Transformer | ~84% | Similar | 2022 |
| TLOB | Dual Attention Transformer | +3.7 F1 | +1.1 F1 | 2025 |
| LiT | LOB Transformer | 63.65% (300-700ms) | N/A | 2025 |

**Critical finding**: On crypto LOB data (BTC/USDT, Bybit 100ms), **XGBoost and logistic regression matched or exceeded DeepLOB by 1-2%** once proper preprocessing was applied (Savitzky-Golay filtering). Simpler models train in minutes vs hours for neural nets.

Feature importance ranking (LOB prediction):
1. Order imbalance ratios (bid vs ask quantities) — **81.3% of selected features**
2. Cumulative depth aggregations
3. Weighted mid-price changes
4. Supply-demand asymmetries across LOB levels
5. Technical indicators (RSI, MACD, Bollinger) — **consistently ranked LOW**

Best accuracy: **72.8% binary (up/down)** at 500ms horizon with 40-level LOB. Ternary (up/flat/down) F1 ~ 0.54.

#### Meta-Learning for Regime Adaptation (X-Trend 2023)

```
Cross-attention from current market conditions to historical regime analogues:
  Query: target sequence representation
  Keys:  context set (historical regime examples)
  Values: context set trading signals

Loss: L_Joint = α * L_prediction + L_Sharpe
  where L_Sharpe = -√252 * mean(r*pos) / std(r*pos)
```

Results: +18.9% Sharpe improvement during regime shifts, 2x faster COVID recovery, 0.47 Sharpe on never-seen assets (zero-shot).

#### Foundation Models for Financial Time Series (2025-2026)

| Model | Developer | Financial Performance | Verdict |
|-------|-----------|----------------------|---------|
| TimesFM | Google | Poor zero-shot on returns | Overhyped |
| Chronos | Amazon | R² -1.37%, directional acc ~51% | Fails on returns |
| **Kronos** | Research | +93% RankIC over generic TSFMs | **Most promising** (no Sharpe yet) |

**Honest assessment**: Zero-shot foundation models perform weakly on financial returns. Fine-tuning gives limited improvement. Training from scratch on financial data shows substantial gains. **Ensemble gradient boosting (CatBoost, XGBoost, LightGBM) consistently outperforms all foundation models.**

Kronos (Aug 2025) is the most promising: pre-trained on 12B+ K-line records from 45 exchanges. No trading results published yet.

#### Reinforcement Learning: Meta-Analysis of 167 Studies (2025)

| Approach | Meta-Average Sharpe | Assessment |
|----------|-------------------|------------|
| Pure RL (model-free) | 1.35 | Decent |
| Traditional methods | 0.95 | Baseline |
| **Hybrid (RL + traditional)** | **1.57** | Best overall |

Most important finding: **"algorithm family (0.08) is least important"** for performance. **Implementation quality and domain knowledge (0.31) dominate.**

**Sim-to-real gap**: No published sim-to-real statistics exist. Most published RL trading papers would lose money in production (don't account for slippage, impact, or regime changes between train/test).

#### What Has NO Peer-Reviewed Evidence of Sub-Hourly OHLCV Profitability

No published, peer-reviewed paper demonstrates sustained profitability from standard technical indicators at sub-hourly resolution on crypto OHLCV after realistic fees. The closest positive result: **CUSUM-filtered data + triple barrier labeling + deep learning** on tick-level data (Financial Innovation, Dec 2025), but this uses tick data processed into information-driven bars, not raw OHLCV.

A separate study achieved 82.68% direction accuracy — but only on **11.99% of the market** (sits out 88% of the time using confidence thresholds). This "selective trading" approach is the most promising OHLCV-adjacent path.

---

### Optimal Execution Algorithms

#### Obizhaeva-Wang (2013) — Transient Impact

Models a block-shaped LOB where impact is transient (exponentially decaying):
```
Impact at time t: Δ_t = Σ_{s<t} q_s * Λ * exp(-ρ*(t-s))

where: Λ = impact per unit order, ρ = resilience rate (LOB recovery speed)

Optimal strategy: initial block trade + gradual execution + terminal block trade
  → Initial block pushes prices to attract new liquidity providers
```

Reduces to Almgren-Chriss when ρ→∞ (instant resilience) or ρ→0 (no resilience). In crypto, LOB resilience ρ varies by asset/time-of-day: seconds (BTC active sessions) to minutes (low-volume periods).

#### The Market Impact Game (Brunnermeier-Pedersen)

When multiple algorithms trade simultaneously:
- Known liquidation (e.g., margin call) → other traders front-run → liquidity spiral
- Nash equilibrium: predatory phase → cooperative phase depending on costs
- In crypto liquidation cascades, this is exactly what happens

**Practical implication**: Your execution interacts with other algorithms. Simple TWAP in a cascade = getting predated. Adaptive execution (RL-based) saves 10-30% vs TWAP/VWAP.

---

### Portfolio Construction for High Frequency

#### Ledoit-Wolf Shrinkage

```
Σ_shrunk = δ*F + (1-δ)*S

where: S = sample covariance, F = structured target, δ = optimal shrinkage intensity
δ* = min(Σπ_ij / (T² * Σ(s_ij - f_ij)²), 1)
```

Nonlinear shrinkage (2017, 2020) applies different shrinkage per eigenvalue via Marchenko-Pastur. With N=3 (BTC/ETH/SOL), improvement is moderate but meaningful. Essential when N>10.

#### Random Matrix Theory — Correlation Cleaning

```
Marchenko-Pastur: eigenvalues below λ₊ = σ²*(1+√q)² are NOISE (q = N/T)

Eigenvalue clipping:
1. Compute eigenvalues of sample correlation
2. Eigenvalues in [λ₋, λ₊] → noise → replace with mean
3. Rescale to maintain trace = N

Result: 20-30% lower out-of-sample portfolio risk
```

With 3 assets, correction is small. Becomes essential with 10+ assets.

#### Hierarchical Risk Parity (Lopez de Prado 2016)

```
Step 1: Distance matrix d(i,j) = √(0.5*(1-ρ_ij)) → single-linkage clustering
Step 2: Quasi-diagonalize correlation matrix by dendrogram leaves
Step 3: Recursive bisection — allocate inversely proportional to cluster variance:
  α = 1 - V₁/(V₁+V₂),  w_cluster1 = α*w_parent
```

Does NOT require matrix inversion (stable when N>T). Lower OOS variance than CLA in Monte Carlo tests. With 3 assets at 1h, reduces to inverse-variance weighting.

#### Kelly with Fat-Tail Correction

```
Full Kelly:  f* = μ/σ²  (maximizes log-wealth growth)
Half-Kelly:  f = 0.5*f*  (industry standard, dramatically reduces drawdown)

Fat-tail correction: L* = (μ-r_f)/σ² - κ_excess*(μ-r_f)²/(2*σ⁴)

For ATR Breakout (Sharpe ~1.3, σ ~60%):
  Full Kelly leverage: ~2.17x
  Half-Kelly: ~1.08x
  Fat-tail adjusted (κ~10-20): ~1.0-1.5x → confirms near-full investment appropriate
```

---

### Risk Management Mathematics

#### Conditional Drawdown at Risk (CDaR)

```
Drawdown: D(t) = max_{s≤t} W(s) - W(t)
DaR_β = inf{d : Pr(D>d) ≤ 1-β}
CDaR_β = E[D | D > DaR_β]

Optimization: min CDaR_β(w) s.t. w'μ ≥ target, Σw = 1
  → Solvable as linear program (like CVaR via Rockafellar-Uryasev)
```

More intuitive than VaR/CVaR for systematic traders — drawdowns cause shutdowns and capital withdrawals.

#### Correlation Breakdown During Stress

```
Forbes-Rigobon correction: ρ_adj = ρ̂ / √(1 + δ*(1-ρ̂²))
  where δ = (σ²_crisis/σ²_normal - 1)

BTC-ETH: ~0.85 normally → ~0.95+ during crashes (partly mechanical, partly genuine tail dependence)
```

Diversification benefit of BTC-ETH-SOL portfolio is illusory during drawdowns. Use stress-conditioned covariance matrices and Clayton copulas.

#### Crypto Tail Behavior

```
Power law: P(|r|>x) ~ x^{-α},  α ~ 2.0-2.5 for BTC
  → Variance exists, kurtosis may be infinite
  → 5σ events ~100x more frequent than Gaussian
  → GARCH persistence α+β ~ 0.90 during cascades

Truncated Levy flights: f(x) ~ |x|^{-(1+α)} * exp(-λ|x|)
  → Power-law intermediate range, finite moments via truncation
```

---

### What Top Crypto Quant Firms Are Actually Doing (2025-2026)

| Firm | Known Approach | Revenue Scale | Retail Replicability |
|------|---------------|--------------|---------------------|
| **Jump Crypto** | Ultra-low latency MM, cross-venue arb, co-located FPGA | Hundreds of millions | None |
| **Wintermute** | Global algorithmic MM, $2.24B daily volume record, OTC +313% | Billions processing | None |
| **Citadel Securities** | Entering crypto MM (Feb 2025), offshore teams, investing in Kraken | Multi-billion revenue | None |
| **Jane Street** | Crypto contributed to $17B+ H1 2025 revenue, OCaml stack | Avg comp $1.4M/employee | None |
| **Cumberland/DRW** | Institutional OTC, invested in Kraken alongside Citadel+Jane Street | Institutional scale | None |

**Alameda Research post-mortem**: The actual quant strategies were mediocre. The edge was primarily from being the house (FTX MM with privileged access, which was fraud). Nearly collapsed in 2018 from wrong XRP predictions. Cross-venue arb (kimchi premium) worked in 2017-2019 but was not sustainable.

**Job postings reveal**: C++, Rust, OCaml (Jane Street). Low-latency systems, market microstructure, order book modeling in demand. Standard ML/DL for price prediction and technical analysis **not in demand from these firms**.

**Citadel entering crypto MM means spreads will compress further** and retail trading edges will shrink.

---

### Cutting-Edge Research Verdict: What Works vs. Hype

#### ACTUALLY WORKS (Evidence-Backed)

1. **Market making with co-location** — proven by firm revenues. Not retail-accessible.
2. **Order book features for 100ms-10min prediction** — 72.8% accuracy documented. Requires L2/L3 data.
3. **CUSUM filtering + triple barrier labeling** — only published approach showing profitability from tick-level data after costs.
4. **VPIN and microstructure measures** — documented predictive power (Easley-O'Hara 2024). Requires tick data.
5. **MEV extraction on-chain** — $7.2B extracted since 2020. Requires blockchain infra.
6. **Hybrid RL + traditional methods** — meta-average Sharpe 1.57. Implementation quality > algorithm choice.
7. **Confidence-threshold selective trading** — 82.68% accuracy by trading only 12% of the time.
8. **Fractional cointegration** — outperforms standard cointegration for crypto pairs. Worth retesting.

#### MOSTLY HYPE (Overpromised, Underdelivered)

1. **Foundation models for return prediction** — zero-shot terrible; gradient boosting wins every time.
2. **Deep learning beating simple models on crypto LOB** — XGBoost matches/beats DeepLOB with proper preprocessing.
3. **Whale wallet tracking** — R² of 0.002-0.054. Noise, not signal.
4. **Social media NLP at sub-hourly** — works daily, noise at minute level.
5. **Quantum computing for portfolio optimization** — classical solvers solve everything in seconds.
6. **Transfer learning equities → crypto** — marginal gains at best.
7. **Pure RL outperforming buy-and-hold** — Stanford study: RL Sharpe 1.23 vs B&H 1.46.

#### PROMISING BUT UNPROVEN (Monitor)

1. **Kronos foundation model** — 12B+ K-line records, +93% RankIC. No trading results yet.
2. **TDA for crash detection** — peaks before crashes, could complement regime detector.
3. **Diffusion models for synthetic training data** — outperform GANs, reproduce stylized facts.
4. **Information-driven bars** (CUSUM, volume, dollar bars) — theoretically superior to time bars.
5. **Bayesian Online Changepoint Detection** — O(1) real-time regime detection, could upgrade existing detector.

---

### Practical Quant Infrastructure

#### Event-Driven Backtesting Frameworks for Sub-Hourly

| Framework | Language | Key Feature | Speed |
|-----------|----------|-------------|-------|
| **NautilusTrader** | Python/Rust | Production-grade, live+backtest, Binance/Bybit | 5M rows/sec |
| **hftbacktest** | Python/Rust | Queue position modeling, L2/L3 data | High |
| LEAN (QuantConnect) | C# | Cloud-hosted, large community | Moderate |
| VectorBT | Python | Vectorized, fast for OHLCV | Very high for bars |

For realistic sub-hourly backtesting: NautilusTrader and hftbacktest are the only frameworks that properly handle tick data, order book simulation, and queue position. The current framework is adequate for 1h+ bar-based backtesting.

#### FPGA vs GPU for Real-Time Inference

- **FPGA**: 480 nanosecond latency; 1000x faster than CPU. Used by Jump, Wintermute for order routing.
- **GPU**: Superior for training. NVIDIA A100/H100 for parallel inference where latency >1ms acceptable.
- **For retail**: Neither matters. Bottleneck is exchange API (single-digit to hundreds of ms), not compute.

---

### Quant-Informed Rules of Engagement

71. **Use Almgren-Chriss trajectory planning for execution of validated strategies.** Even the 1h ATR Breakout benefits from optimal execution during entry/exit. Estimated savings: 5-10 bps per round trip. For BTC with 27 trades/year, this is 1.35-2.7% annual improvement with zero strategy change.

72. **Apply fractional cointegration (FCVAR) before declaring pairs dead.** Standard Engle-Granger requires I(1)→I(0). FCVAR detects I(d)→I(d-b) with 0<b<d — "almost cointegrated" relationships with long memory. Retest BTC/ETH, BTC/SOL with FCVAR and Gregory-Hansen (structural breaks). The pairs may be cointegrated within regimes even if the full sample shows no cointegration.

73. **Use noise-robust volatility estimators at sub-hourly.** Standard RV diverges at high frequency. Use Yang-Zhang for 15m+ bars, TSRV or realized kernels for tick-level data. Never trust raw realized variance computed from 1m returns — the microstructure noise bias will be larger than the true volatility signal.

74. **Detect jumps before attributing performance.** Use bipower variation to separate continuous and jump components: `Jump Variation = RV - BV`. If a strategy's returns are dominated by jump capture (rare, large moves), it is an event strategy with inherently unstable performance. Lee-Mykland test identifies exact jump timing for attribution.

75. **Model order flow with Hawkes processes when using tick data.** The branching ratio n (typically 0.6-0.9 for crypto) reveals what fraction of order flow is endogenous. Near-critical n→1 signals cascade risk. Multivariate Hawkes with buy/sell market orders + mid-price jumps provides the most complete model of LOB dynamics.

76. **Use VPIN as a real-time toxicity indicator, not a standalone signal.** VPIN spikes precede volatility events and liquidation cascades. Use it as a risk management overlay: reduce position size when VPIN > 2σ above its rolling mean. Do not use as a directional signal — Andersen-Bondarenko showed it's mechanically correlated with trading intensity.

77. **Apply Bayesian Online Changepoint Detection for real-time regime switching.** BOCPD (Adams-MacKay) is O(1) per update and produces a probability distribution over "regime changed" vs "regime continues." Superior to batch methods for live trading. Set hazard function H = 1/λ with λ calibrated to expected regime duration (e.g., λ=500 for ~500-bar expected regime length at 1h).

78. **Use Clayton copulas for stress-conditioned portfolio risk.** Gaussian copulas underestimate crash co-movement. BTC-ETH correlation jumps from ~0.85 to ~0.95+ during crashes. Clayton copulas capture lower tail dependence. For portfolio VaR/CVaR, always compute under both normal and stress copula assumptions.

79. **Apply EVT (Generalized Pareto) for tail risk, never Gaussian.** Bitcoin tail exponents ξ ~ 2.0-2.5. Five-sigma events occur ~100x more than Gaussian predicts. Use Peaks-Over-Threshold with GPD for VaR and ES computation. Standard Sharpe-based risk metrics severely underestimate true risk.

80. **Prefer XGBoost over deep learning for LOB features.** Published 2025 results show XGBoost with proper preprocessing (Savitzky-Golay filtering) matches or exceeds DeepLOB/TLOB on crypto LOB data while training in minutes instead of hours. Order book features constitute 81.3% of importance — technical indicators are consistently ranked low by ML feature importance methods.

81. **If pursuing ML for sub-hourly, use CUSUM-filtered information-driven bars + triple barrier labeling.** This is the only published approach showing profitability from tick-adjacent data after costs (Financial Innovation, Dec 2025). Standard time bars at sub-hourly are dominated by noise. Information-driven bars (CUSUM, volume, dollar bars) synchronize with market activity, concentrating signal.

82. **Use the confidence-threshold selective trading approach.** The most promising sub-hourly ML result shows 82.68% accuracy — but only trades 12% of the time using a confidence threshold. A strategy that is right 83% of the time but only trades when highly confident is fundamentally different from one that trades continuously. Design strategies to express conviction, not frequency.

83. **Account for the market impact game in execution.** Your execution interacts with other algorithms. Simple TWAP during a liquidation cascade = getting predated (Brunnermeier-Pedersen). Adaptive execution (RL-based or simple state-dependent) saves 10-30% vs naive approaches. At minimum, condition execution aggressiveness on order book depth and recent volatility.

84. **Monitor the Kronos foundation model.** First financial-specific foundation model pre-trained on 12B+ K-line records from 45 exchanges. Shows +93% RankIC over generic TSFMs and 9% lower MAE in vol forecasting. No trading results published yet. If zero-shot or few-shot crypto prediction materializes, it could change the game for OHLCV-only sub-hourly. Check arxiv.org/abs/2508.02739 for updates.

85. **Cross-exchange arbitrage is effectively dead for retail.** CEX-to-CEX spreads reached zero by 2024 for major pairs. Remaining arb is on-chain MEV ($7.2B extracted since 2020, ~$300K/day on Ethereum). This requires blockchain infrastructure, not trading algorithms. Do not invest in cross-exchange arb infrastructure unless co-located with sub-ms latency.
