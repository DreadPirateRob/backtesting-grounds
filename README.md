# Crypto Backtester

A vectorized backtesting framework for systematic crypto strategies, with a
library of trading strategies, walk-forward validation, regime detection, and
parameter sweeps over Binance kline data.

## Layout

| Path | Description |
|------|-------------|
| `backtester/` | Core engine: data loading, backtest engine, metrics, portfolio, walk-forward, regime detection |
| `strategies/` | Strategy implementations (each exposes `generate_signals`) |
| `pinescript/` | TradingView PineScript ports of selected strategies |
| `data/` | Market data and run outputs (large CSVs are gitignored — see below) |
| `run_*.py`, `sweep_*.py`, `finetune_*.py` | Experiment / analysis drivers |
| `SKILL.md`, `DEPLOYMENT_GUIDE.md` | Detailed usage and deployment notes |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The 1-minute kline CSVs (~400MB each) are **not** committed. Fetch them locally:

```bash
python3 fetch_binance_klines.py --symbol BTCUSDT --interval 1m --years-back 5
python3 fetch_binance_klines.py --symbol ETHUSDT --interval 1m --years-back 5
python3 fetch_binance_klines.py --symbol SOLUSDT --interval 1m --years-back 5
```

Files land in `data/binance_<SYMBOL>_1m_klines.csv`. Data is sourced from public
Binance endpoints; no API key is required.

## Running a backtest

```bash
python3 run_backtest.py --list-strategies
python3 run_backtest.py --strategy <name> --sweep --top 10
```

See `SKILL.md` for the full CLI reference and workflow.
