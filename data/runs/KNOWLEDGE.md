# KNOWLEDGE.md — BTC Strategy Optimization Knowledge Base
<!-- schema_version: 1.1 | last_updated: 2026-02-19T11:37:52Z | total_runs: 2 -->

---

## AGENT RETRIEVAL INSTRUCTIONS

Read this entire file before Phase 1 of each new Optimization Loop. Extract four things:
1. Current regime column from the Strategy-Regime Matrix — pick your seed family from the highest-Sharpe, highest-confidence cell in that column.
2. Matching rows from Robust Parameter Regions — use `center` values as Phase 4 fine-grid midpoints instead of class defaults.
3. Derivation Method Effectiveness rows for this regime — reorder your derivation queue by descending success rate.
4. Anti-Patterns for this regime — skip any flagged configuration before attempting it.

Then announce: "Historical context: [N] prior runs. Regime: [label]. Best historical family: [X] (Sharpe [Y], [confidence] confidence). Anti-patterns to avoid: [list]."

If the current regime has no data (all cells blank in that column), use similarity matching: find the closest historical regime by |B&H return difference|, load its 1-2 most similar run JSONs, and apply their guidance tentatively. Note this in your announcement.

---

## 1. STRATEGY-REGIME PERFORMANCE MATRIX

Each cell = `Sharpe | strategy_name | confidence`. Confidence tiers: **L** = 1-3 runs, **M** = 4-7 runs, **H** = 8+ runs. Blank = untested. Use the highest-Sharpe non-blank cell in the current regime column as your seed family. If all cells in that column are blank, see retrieval instructions above.

| Family            | strong_uptrend                    | mild_uptrend                  | range_bound                     | bear_market                   |
|-------------------|-----------------------------------|-------------------------------|---------------------------------|-------------------------------|
| mean_reversion    | 2.64 \| rsi_bollinger \| L        | 3.72 \| dual_rsi \| L        | 3.03 \| rsi_bollinger \| M      | 2.84 \| dual_rsi \| L        |
| trend_following   | 1.89 \| ema_crossover \| L        | 2.44 \| supertrend \| L      | -0.83 \| ema_crossover \| M     | 1.56 \| ema_crossover \| L    |
| breakout          | 2.47 \| atr_breakout \| L         | 3.52 \| atr_breakout \| L    | 2.32 \| atr_breakout \| L       | 2.00 \| atr_breakout \| L     |
| volume            | 2.31 \| cvd_divergence \| L       | 1.99 \| cvd_divergence \| L  | 3.22 \| cvd_divergence \| L     | 2.04 \| cvd_divergence \| L   |
| momentum          |                                   |                               |                                 |                               |
| volatility        |                                   |                               |                                 |                               |
| carry             |                                   |                               |                                 |                               |

**Update rule:** After each run, update the cell for (winner family, run regime). Replace the cell value only if the new Sharpe is higher than the current value OR if this run raises the confidence tier. Never lower a Sharpe entry — this matrix tracks historical bests, not most recent. Pipe characters inside cells are escaped as `\|` to preserve Markdown table structure.

---

## 2. ROBUST PARAMETER REGIONS

For each confirmed region, use `center` as the midpoint of your Phase 4 fine grid. A region is "confirmed" at confirmations >= 2. A single-run region (confirmations=1) is a starting hypothesis — use it but do not rely on it heavily.

### mean_reversion / rsi_bollinger (regime: range_bound)

| Parameter    | Center | Range      | Confirmations | Notes                                    |
|--------------|-------:|------------|:-------------:|------------------------------------------|
| rsi_period   |     28 | 21 – 35    |       1       | Higher periods reduced false signals in chop |
| overbought   |     65 | 60 – 70    |       1       | Tighter than default 70 catches more signals |
| oversold     |     32 | 28 – 38    |       1       | Slight asymmetry vs overbought is intentional |
| bb_period    |     15 | 12 – 20    |       1       | Shorter BB responds faster in ranging market |
| bb_std       |    2.0 | 1.8 – 2.2  |       1       | Standard 2.0 held up; cliff at 1.5 and 2.5  |

- **Mode:** long+short. Short side win rate 81.2% in this run — do not force long-only in range_bound.
- **Timeframe:** 1h for more trades; 4h produces higher Sharpe (3.03) but fewer trades.

### mean_reversion / dual_rsi (regime: bear_market)

| Parameter          | Center | Range      | Confirmations | Notes                                    |
|--------------------|-------:|------------|:-------------:|------------------------------------------|
| slow_period        |     60 | 30 – 120   |       1       | Only 1 bear sub-period tested            |
| fast_period        |     20 | 10 – 20    |       1       |                                          |
| slow_overbought    |     65 | 65 – 75    |       1       |                                          |
| slow_oversold      |     25 | 25 – 35    |       1       |                                          |

- **Timeframe:** 1h (Sharpe 2.84). Only 1 trade at 4h — unreliable.

### breakout / atr_breakout (regime: all — WALK-FORWARD VALIDATED, CROSS-ASSET CONFIRMED)

| Parameter    | Center | Range      | Confirmations | Notes                                    |
|--------------|-------:|------------|:-------------:|------------------------------------------|
| sma_period   |     10 | 10 – 20    |      39       | PCR 0.92 BTC/ETH 1h (modal 10), PCR 0.54 SOL (modal 20) |
| atr_period   |     20 | 10 – 20    |      26       | PCR 0.85 ETH 1h, 0.69 ETH 4h, 0.62 SOL  |
| atr_mult     |    3.0 | 2.5 – 3.0  |      36       | PCR 0.77 SOL 1h (modal 3.0), 0.54 ETH 1h (modal 2.5) |

- **BTC**: OOS Sharpe +0.45 (1h), +0.58 (4h). WFER 0.38 (ACCEPTABLE).
- **ETH 1h**: OOS Sharpe +0.59, WFER 0.55 (GOOD). sma=10 stable.
- **SOL 1h**: OOS Sharpe +1.40, WFER 0.72 (EXCELLENT). atr_mult=3.0 stable.
- **1h is the sweet spot** across all assets. 4h suffers low trade counts.
- **Sharpe 2.0+ in every regime type.** No regime conditioning needed.

**Update rule:** When a new run's best params fall within an existing range, increment confirmations. When they fall outside, expand the range boundaries and reset confirmations to 1. Remove a region entirely if 3+ consecutive runs find their best params outside it. Merge overlapping regions into one; keep at most 3 confirmed regions per strategy/regime pair.

---

## 3. DERIVATION METHOD EFFECTIVENESS

`succ/att` = successes / attempts. `avg_gain` = mean Sharpe increase on successful attempts only (n/a when 0 successes). Use this table to prioritize your derivation queue: sort descending by success rate within the current regime, then fall back to global rates. Blank avg_gain cells = no successful attempts yet.

### regime: range_bound (1 run)

| Method                  | att | succ | avg_gain | Notes                                               |
|-------------------------|----:|-----:|---------:|-----------------------------------------------------|
| M3: Combine             |   1 |    1 |    +0.52 | RSI+BB confluence over pure RSI. Single best move.  |
| M5: New concept         |   4 |    1 |    +0.23 | Most attempts failed; one momentum concept showed edge |
| M1: Add filter          |   1 |    0 |          | EMA trend filter cancelled valid reversion signals  |
| M4: Change TF           |   2 |    0 |          | 4h: trade count too low for robustness tests        |
| M2: Swap indicator      |   0 |    0 |          | Not yet attempted in this regime                    |
| M6: Add regime filter   |   0 |    0 |          | Not yet attempted in this regime                    |
| M7: Adaptive params     |   0 |    0 |          | Not yet attempted in this regime                    |

### global (all regimes, 1 run — same data as above)

| Method                  | att | succ | avg_gain |
|-------------------------|----:|-----:|---------:|
| M3: Combine             |   1 |    1 |    +0.52 |
| M5: New concept         |   4 |    1 |    +0.23 |
| M1: Add filter          |   1 |    0 |          |
| M4: Change TF           |   2 |    0 |          |
| M2: Swap indicator      |   0 |    0 |          |
| M6: Add regime filter   |   0 |    0 |          |
| M7: Adaptive params     |   0 |    0 |          |

**Update rule:** After each run, increment `att` for every method used, increment `succ` when the derived strategy improved leaderboard rank above its parent, and update `avg_gain` as a rolling mean. Add a new regime sub-table the first time a regime is seen. The global table aggregates across all regime sub-tables.

---

## 4. RUN SUMMARIES

Most recent first. Capped at 10 — remove the oldest entry when adding an 11th.

```
[voltarget-multiasset-2026-02-19]
  date:        2026-02-19
  type:        position sizing overlay on multi-asset portfolio (3 methods, 11 variants)
  result:      STAR — VolTarget 25% (60d): Sharpe 1.32 (+0.11), MaxDD -21% (from -38%), Calmar 6.38.
  key_finding: Vol targeting IMPROVED Sharpe while halving MaxDD. Sizes up in calm, down in volatile.
               Half-Kelly most conservative (-16% MaxDD). DD-scaling underperformed across all thresholds.
               60d lookback > 30d. Deployment config: VolTarget 25%, 60d lookback, clamp [0.5, 2.0].

[wf-cvd-donchian-2026-02-19]
  date:        2026-02-19
  type:        walk-forward (CVD Divergence + Donchian Channel, 1h + 4h, 13 segments each)
  result:      BOTH FAILED — CVD 1h: Sharpe -1.01, WFER -7.54. Donchian 1h: Sharpe -0.96.
  key_finding: CVD regime-sweep promise (avg 0.48) was pure overfit. Donchian 4h barely +0.20 with
               -57% MaxDD and decaying edge. No second WF-validated strategy found yet.

[multiasset-portfolio-wf-2026-02-19]
  date:        2026-02-19
  type:        multi-asset portfolio walk-forward (BTC+ETH+SOL ATR Breakout EW, 1h, 13 segments)
  result:      STAR — OOS Sharpe 1.30, WFER 0.97 (EXCELLENT), +395% ret, -38% MaxDD.
  key_finding: Diversification delta +0.48. 10/13 segments positive. Pairwise corr 0.33-0.46.

[wf-zscore-bbounce-2026-02-20]
  date:        2026-02-20
  type:        walk-forward (Z-Score MR + Bollinger Bounce, 1h + 4h)
  result:      BOTH FAILED — Z-Score: Sharpe -0.32 (1h), -0.28 (4h). BBounce: Sharpe -1.28 (1h), -0.29 (4h).
  key_finding: 7/7 non-ATR strategies now failed WF. Mean-reversion family structurally disadvantaged
               on BTC (crypto trends persistently). BBounce worst ever — 7/13 segments had 0 trades.

[wf-momentum-roc-2026-02-19]
  date:        2026-02-19
  type:        walk-forward (rolling 18mo IS / 3mo OOS, 13 segments)
  strategy:    momentum_roc | 1h + 4h
  result:      FAILED — 1h: Sharpe 0.02, WFER 0.16 (WEAK). Trend-following dead for BTC.

[crossasset-atr-2026-02-19]
  date:        2026-02-19
  type:        cross-asset walk-forward (ATR Breakout on ETH + SOL, 1h + 4h)
  result:      ETH 1h: Sharpe 0.59, WFER 0.55 (GOOD), +75% ret. SOL 1h: Sharpe 1.40, WFER 0.72 (EXCELLENT), +1604% ret.
  key_finding: ATR Breakout generalizes beyond BTC. 1h is sweet spot (4h too few trades).
               ETH sma=10 stable (PCR 0.92). SOL atr_mult=3.0 stable (PCR 0.77).
               SOL 4h REJECTED — WFER 0.18 with decaying edge (t=-2.22), MaxDD -91%.

[portfolio-wf-2026-02-19]
  date:        2026-02-19
  type:        portfolio walk-forward (ATR Breakout + RSI+BB EW, 13 segments)
  result:      FAILED — 1h: Sharpe -0.37, WFER -0.30. 4h: Sharpe -0.61, WFER -0.54.
  key_finding: RSI+BB (OOS Sharpe -0.87 to -1.07) destroys portfolio despite excellent
               negative correlation (-0.51). ATR alone: +0.50 Sharpe. DO NOT combine.

[regime-switch-v2-2026-02-19]
  date:        2026-02-19
  type:        regime detector recalibration (6 calibrations, SMA 50-200, mom 1-5%)
  result:      ALL FAILED — best regime-switch Sharpe -0.10 (SMA200). Static ATR: Sharpe 0.89, +459%.
  key_finding: More sensitive detector = worse (more transitions = more fees). Regime switching
               is fundamentally flawed for this strategy set. Static ATR Breakout dominates.

[portfolio-2026-02-19]
  date:        2026-02-19
  type:        portfolio backtest (2025 only, RSI+BB + CVD, 1h)
  result:      EW Sharpe 1.98 (66% ret, -21% DD). Risk-parity 1.84. Max-Sharpe 1.02.
  key_finding: -0.13 correlation gives 1.51 diversification ratio. EW beats optimization.

[wf-atr-breakout-2026-02-19]
  date:        2026-02-19
  type:        walk-forward (rolling 18mo IS / 3mo OOS, 13 segments)
  strategy:    atr_breakout | 1h + 4h
  result:      ROBUST — OOS Sharpe +0.45 (1h), +0.58 (4h). WFER 0.38 (ACCEPTABLE).
  key_finding: First WF-validated strategy. sma_period=10 perfectly stable (PCR 1.00).
               Works across ALL regimes without conditioning. Best foundation strategy.

[regime-sweep-2026-02-19]
  date:        2026-02-19
  type:        regime sweep (8 regimes × 7 strategies × 2 TF = 112 tasks)
  winner:      mean_reversion family dominates ALL regimes in-sample (Sharpe 2.6-3.7) — but ALL fail WF.
  key_finding: ATR breakout is the only strategy that survives walk-forward validation.
  report:      data/runs/regime_sweep_full.json
```

**Update rule:** Prepend new entries in the same code-block format. Remove the oldest when the count exceeds 10. Keep each entry under 10 lines — move detailed analysis to the JSON report. The `report` field must point to the actual JSON file path.

---

## 5. ANTI-PATTERNS

Confirmed failure patterns only. Each row requires `confirmed_across >= 2` OR a single-run result with Sharpe < -0.5 and a clear structural explanation. Capped at 15 — when adding a 16th, remove the row with the lowest `confirmed_across` count. Do not remove entries that have `confirmed_across >= 3`.

| # | Regime       | Pattern                                              | Reason                                                                                  | confirmed_across |
|---|--------------|------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------|
| 1 | range_bound  | Trend-following families (EMA, MACD) as seed         | B&H < ±20% means price chop destroys directional edge; all variants produced Sharpe < 0 | 3 sub-periods   |
| 2 | range_bound  | EMA trend filter added onto mean-reversion strategy  | Filter too restrictive — long-term EMA direction unreliable in range; cancels valid signals | 1 run         |
| 3 | all          | MACD as primary strategy                              | Negative Sharpe in 6/8 regimes at 1h. High trade count (200-800) = fee destruction      | 8 regimes       |
| 4 | range_bound  | 1m resolution for indicator-based strategies           | Same signals as 1h but with noise; fee drag kills any edge                               | 1 run           |
| 5 | all          | Supertrend with default params                         | Typically 1 trade per regime — insufficient for statistical evaluation                   | 8 regimes       |
| 6 | all          | RSI+BB without regime conditioning                    | Walk-forward WFER -0.87 across all regimes. Profitable only in range-bound OOS windows  | 13 segments     |
| 7 | range_bound  | CVD Divergence — inconsistent across sub-periods      | Sharpe 3.22 in 2024-H2 but -0.71 in 2025. Not a reliable range-bound strategy          | 3 sub-periods   |
| 8 | all          | Regime switching (any detector calibration)            | 6 calibrations tested (SMA 50-200, mom 1-5%). All negative. Transaction costs eat alpha | 8 configs       |
| 9 | all          | Max-Sharpe portfolio optimization                      | Sharpe 1.02 vs EW 1.98. Over-concentrates. Equal-weight is more robust                 | 1 run           |
| 10 | all         | RSI+BB in equal-weight portfolio with ATR Breakout    | RSI+BB OOS Sharpe -0.87 to -1.07 destroys portfolio. ATR alone Sharpe +0.50. Don't combine | 26 segments   |
| 11 | all         | ATR Breakout at 4h on altcoins (ETH/SOL)              | Too few trades (many segments <5). SOL 4h MaxDD -91%, WFER 0.18 (WEAK), decaying edge  | 26 segments     |
| 12 | all         | Momentum ROC as standalone strategy                    | WFER 0.16-0.26. Works in trends (2023-24) but destroyed in range-bound. -66% to -71% MaxDD | 26 segments     |
| 13 | all         | CVD Divergence as standalone strategy                   | WFER -7.54 (1h), -1.77 (4h). Regime-sweep promise was pure overfit. -76% to -80% MaxDD   | 26 segments     |
| 14 | all         | Donchian Channel as standalone strategy                 | 1h catastrophic (-77% ret). 4h: Sharpe +0.20, -57% DD, decaying edge. Not tradeable      | 26 segments     |
| 15 | all         | DD-based position scaling (binary 50% reduction)        | Misses rebounds. All 3 thresholds (10/15/20%) worse than raw or vol-target. Avoid.        | 3 variants      |
| 16 | all         | Z-Score MR / Bollinger Bounce as standalone             | Mean-reversion band-touch structurally fails on BTC — crypto trends too persistently      | 52 segments     |

**Update rule:** Add a new row when a pattern repeats across 2+ runs, or when a single run shows strong structural evidence (Sharpe < -0.5 with a clear cause). Increment `confirmed_across` when a subsequent run reproduces the same failure. Remove a row if a future run contradicts it — i.e., the pattern fails to hold and the formerly-flagged approach succeeds.

---

## 6. OPEN QUESTIONS

Regenerate this section completely after each run. Do not accumulate stale questions. Remove any question answered by the most recent run. Add new gaps discovered during the run.

### Priority: Deployment-ready (high impact)

- **Live trading pipeline**: PineScript export for TradingView alerts, or exchange API integration.
- **Rebalancing frequency**: Current WF uses 3-month param refresh. Test monthly vs quarterly.
- **Transaction cost sensitivity**: Test at 15bps and 20bps to confirm robustness to higher fees.

### Priority: Finding a second WF-validated strategy (low probability, medium impact)

- `carry` family — requires funding rate data (Binance API geo-blocked from current IP).
- **Multi-timeframe combinations** — combining 1h+4h ATR Breakout signals may reduce drawdowns.
- **On-chain data strategies** — fundamentally different signal source. Not in current framework.
- **Cross-exchange arb / basis trade** — requires multi-exchange data. Not in current framework.
- NOTE: 6/7 strategies in existing library have failed WF. Remaining candidates require new data sources.

### WF scorecard (7 tested, 1 passed)

- ATR Breakout: **PASS** (Sharpe 0.45-1.40 across 3 assets)
- RSI+BB: FAIL | Momentum ROC: FAIL | CVD Divergence: FAIL
- Donchian: FAIL | Z-Score MR: FAIL | Bollinger Bounce: FAIL

---

## 7. METADATA

```
schema_version:  1.1
last_updated:    2026-02-19T20:00:00Z
total_runs:      18 (2 optim + 1 regime sweep + 9 walk-forward + 2 regime-switch + 3 portfolio + 1 sizing)
data_coverage:   BTC/ETH/SOL USDT, 1m klines → 1h/4h, 2021-02-20 to 2026-02-17
regimes_seen:    strong_uptrend (2), mild_uptrend (2), range_bound (3+2), bear_market (1)
regimes_unseen:  none — all 4 regime types now have data
line_count:      ~260
```

**Schema changelog:**
- v1.0 → v1.1: Added `runner_up` field in Run Summaries. Added `confirmed_across` column in Anti-Patterns. Added per-regime sub-tables in Derivation Method Effectiveness. Replaced dash placeholders in matrix with blank cells (blank = untested, dash = ambiguous). Added Formatting Guidelines section below.

---

## FORMATTING GUIDELINES

For the agent maintaining this file — not for end users.

**Blank vs dash in matrix cells:** Blank = untested (no data). A negative Sharpe result still fills the cell (e.g., `-0.83 | ema_crossover | L`). Never use `—` or `N/A` for untested — the agent must distinguish "not tested" from "tested and failed."

**Pipe escaping:** Cells containing `|` must escape as `\|`. This applies to the Strategy-Regime Matrix only; other tables do not contain pipe characters.

**Sharpe precision:** Always two decimal places (e.g., `2.97`, `-0.83`). Three decimals only when the difference between adjacent values is < 0.01.

**Regime labels:** Always use exactly these four strings: `strong_uptrend`, `mild_uptrend`, `range_bound`, `bear_market`. Never abbreviate or vary capitalization.

**Method codes:** M1 through M7 map to: M1=Add filter, M2=Swap indicator, M3=Combine, M4=Change TF, M5=New concept, M6=Add regime filter, M7=Adaptive params. These codes must match SKILL.md exactly.

**Line budget:** Target under 200 lines when populated at steady state (10 run summaries, 15 anti-patterns, 3 confirmed parameter regions). When approaching the limit: collapse single-confirmation parameter regions into notes; prune Open Questions that have been open for 5+ runs without an attempt; drop anti-patterns with confirmed_across=1 that are older than 5 runs.

**Update atomicity:** Update all seven numbered sections plus metadata in a single write operation. Never leave the file half-updated. The `last_updated` timestamp and `total_runs` counter must reflect the completed state.
