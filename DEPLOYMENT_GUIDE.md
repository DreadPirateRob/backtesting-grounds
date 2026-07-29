# Live Trading Deployment Guide

## Strategy: ATR Breakout (Multi-Asset Portfolio + Vol Targeting)

The only walk-forward validated strategy out of 7 tested. Deployed as an equal-weight portfolio across BTC, ETH, and SOL on the 1h timeframe with a volatility targeting overlay.

### Portfolio Performance (Walk-Forward OOS, Aug 2022 – Nov 2025)

| Metric | Raw Portfolio | + VolTarget 25% (60d) |
|--------|-------------|----------------------|
| Sharpe | 1.30 | **1.32** |
| Return | +395% | +198% |
| MaxDD | -38% | **-21%** |
| Calmar | 3.19 | **6.38** |
| WFER | 0.97 (EXCELLENT) | — |

---

## Per-Asset Parameters (WF-Validated)

| | **BTCUSDT** | **ETHUSDT** | **SOLUSDT** |
|---|---|---|---|
| `sma_period` | 10 | 10 | 20 |
| `atr_period` | 14 | 20 | 14 |
| `atr_mult` | 3.0 | 2.5 | 3.0 |
| OOS Sharpe | 0.45 | 0.59 | 1.40 |
| OOS Return | +71% | +75% | +1604% |
| MaxDD | -39% | -53% | -57% |
| WFER | 0.38 (OK) | 0.55 (GOOD) | 0.72 (EXCELLENT) |

### Vol Targeting Settings
- Target annualized vol: **25%**
- Lookback: **60 days** (1440 bars at 1h)
- Position scale clamp: **0.5x – 2.0x**
- Only resize if scale changes >10%

---

## Phase 1: Paper Trade (2-4 weeks)

1. Open 3 TradingView charts: BTCUSDT, ETHUSDT, SOLUSDT — all on **1h**
2. Add PineScript strategy (`pinescript/atr_breakout_multiasset.pine`) to each chart with asset-specific params from the table above
3. Enable "Paper Trading" mode in TradingView
4. Track for 2-4 weeks, checking that:
   - Trades trigger at the right levels (price crosses SMA ± ATR*mult band)
   - Trade frequency matches backtest (~7-16 trades per 3-month segment)
   - Vol scale reading makes sense (should be near 1.0x in normal conditions)

---

## Phase 2: Capital & Exchange Setup

### Capital allocation
- Start with capital you can afford to lose entirely
- Split equally across 3 assets (1/3 each)

| Total Capital | Per Asset | Example at $30k |
|--------------|-----------|-----------------|
| 100% | 33.3% each | $10k BTC, $10k ETH, $10k SOL |

### Exchange selection
- **Bybit** or **OKX** recommended (lower fees, good API)
- Use **linear perpetual contracts** (USDT-margined) for easy long/short flipping
- **Cross margin** mode so full allocation backs each position
- Set leverage to **2x max** (matches vol-target ceiling of 2.0x)

### Fee considerations
- Backtest assumes 10bps (0.1%) per trade
- Most exchanges offer 10bps maker/taker at base tier
- Use limit orders when possible (maker fees often 2-6bps)
- VIP tiers or holding exchange tokens can cut fees further

---

## Phase 3: Execution Method

### Option A: TradingView Alerts → Manual Execution (simplest)
1. Set up alerts on each chart: "Order fills only"
2. When alert fires (long/short flip), manually place the trade
3. Pros: No coding, full control
4. Cons: Must be available when alerts fire (could be 3am)

### Option B: TradingView Alerts → Webhook → Bot (recommended)
1. Use a webhook relay service (3Commas, Alertatron, or custom)
2. TradingView alert → webhook URL → bot places order on exchange
3. Fully automated, no manual intervention
4. Requires: TradingView Pro plan (for webhook alerts)

### Option C: Custom Python Bot (most control)
1. Run existing Python strategy code on a VPS (e.g., DigitalOcean $6/mo)
2. Fetch 1h candles via exchange API every hour
3. Compute signals, compare to current position, execute if different
4. Use the CCXT library for exchange abstraction

```
Rough loop:
  Every hour at :01 (after candle close):
    1. Fetch latest 1h candles for BTC, ETH, SOL
    2. Run AtrBreakoutStrategy.generate_signals() for each
    3. Get latest signal (last row)
    4. Compare to current exchange position
    5. If different → place market order to flip
    6. Compute vol-target scale → adjust position size
    7. Log everything
```

---

## Phase 4: Position Sizing (Vol Targeting)

**Every hour, for each asset:**
1. Compute realized vol = std(hourly log returns, last 1440 bars) × sqrt(8766)
2. Scale = 0.25 / realized_vol, clamped to [0.5, 2.0]
3. Position size = (account_equity × 0.333) / price × scale

**In practice:**
- Normal vol (~40-60% annualized): scale ≈ 0.4-0.6x → conservative sizing
- Low vol (~20-30%): scale ≈ 0.8-1.25x → normal sizing
- Very low vol (<15%): scale = 2.0x (capped) → aggressive sizing
- Crisis vol (>80%): scale = 0.5x (floored) → defensive

---

## Phase 5: Ongoing Management

### Daily checklist (5 min)
- Confirm all 3 positions are correct (long or short, right size)
- Check funding rate on perpetuals (if consistently >0.1% per 8h against position, note it)
- Verify bot is running / no missed alerts

### Monthly
- Compare live P&L vs what the backtest would have produced
- If tracking error > 2% per month, investigate (slippage? missed fills? data issues?)

### Quarterly (param refresh)
- Re-run walk-forward sweep on latest 18 months of data
- If optimal params shift, update strategy params
- In practice, params are very stable (sma=10 held across 12/13 WF segments for BTC)

---

## Phase 6: Circuit Breakers

Set these **before** going live. Non-negotiable.

| Rule | Action |
|------|--------|
| Single asset DD > 15% in a month | Reduce that asset to 50% size for 2 weeks |
| Portfolio DD > 10% in a week | Flatten all positions, pause 48h, review |
| Portfolio DD > 20% from peak | Flatten everything, full review before restarting |
| Exchange API errors > 3 in a row | Flatten, switch to manual until resolved |
| Live vs backtest divergence > 5% over 1 month | Investigate before continuing |

---

## Key Risks

1. **Funding rates** — Perpetual contracts charge/pay funding every 8h. In strong trends, funding can be 0.1-0.3% per 8h against trend followers. Not in backtest. Budget 1-3% per month for funding drag.

2. **Slippage on SOL** — Less liquid than BTC/ETH. Use limit orders or reduce size if fills are poor.

3. **Exchange risk** — Don't keep all capital on one exchange. Consider splitting across 2.

4. **Regime change** — Strategy works across bull, bear, and sideways historically, but fundamentally new regimes could break it.

5. **Correlation spike** — In crashes, BTC/ETH/SOL correlations spike to 0.9+, destroying diversification. Vol-target overlay partially handles this (scales down in high-vol).

---

## First Week Checklist

- [ ] Set up 3 TradingView charts with correct params per asset
- [ ] Paper trade for minimum 2 weeks
- [ ] Choose exchange, set up API keys
- [ ] Choose execution method (manual / webhook / bot)
- [ ] Define total capital and 1/3 per asset split
- [ ] Set circuit breakers as exchange stop-losses or bot rules
- [ ] Go live with **50% of planned capital** for first month
- [ ] Scale to full size only after 1 month of live tracking matches expectations

---

## What Failed (for reference)

| Strategy | Result | Why |
|----------|--------|-----|
| RSI+BB | FAIL | Overfit — only works in range-bound, WFER -0.87 |
| Momentum ROC | FAIL | Trend-following destroyed in chop, WFER 0.16 |
| CVD Divergence | FAIL | In-sample promise was pure overfit, WFER -7.54 |
| Donchian Channel | FAIL | 1h catastrophic, 4h barely flat with -57% DD |
| Z-Score MR | FAIL | Wildly inconsistent, -42% return |
| Bollinger Bounce | FAIL | Worst tested — 7/13 segments had zero trades |
