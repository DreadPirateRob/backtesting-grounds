#!/usr/bin/env python3
"""Run portfolio backtests combining RSI+Bollinger and CVD Divergence."""

from backtester.data_loader import load_candles
from backtester.engine import BacktestConfig
from backtester.portfolio import (
    PortfolioConfig, build_portfolio, run_portfolio_backtest, print_portfolio_report,
)
from strategies.rsi_bollinger_strategy import RsiBollingerStrategy
from strategies.cvd_divergence_strategy import CvdDivergenceStrategy


DATA_PATH = "data/binance_BTCUSDT_1m_klines.csv"


def main():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0005)

    strategies = {
        "rsi_bollinger": (
            RsiBollingerStrategy,
            {"rsi_period": 30, "overbought": 65, "oversold": 32, "bb_period": 15, "bb_std": 2.0},
        ),
        "cvd_divergence": (
            CvdDivergenceStrategy,
            {"pivot_lookback": 10, "cvd_smoothing": 5, "confirmation_bars": 2, "atr_stop_mult": 2.0},
        ),
    }

    # 2025 range-bound period at 1h
    print("Loading 2025 1h data...")
    df = load_candles(DATA_PATH, start="2025-01-01", end="2025-12-31", resample="1h",
                      extra_columns=["taker_buy_base_volume"])
    print(f"Data: {len(df)} bars\n")

    # Equal weight portfolio
    print("=" * 70)
    print("EQUAL WEIGHT PORTFOLIO")
    print("=" * 70)
    eq_result = run_portfolio_backtest(
        strategies, df, config,
        portfolio_config=PortfolioConfig(method="equal_weight", rebalance_freq="1ME"),
    )
    print_portfolio_report(eq_result)

    # Risk parity portfolio
    print("\n" + "=" * 70)
    print("RISK PARITY PORTFOLIO")
    print("=" * 70)
    rp_result = run_portfolio_backtest(
        strategies, df, config,
        portfolio_config=PortfolioConfig(method="risk_parity", rebalance_freq="1ME", lookback_bars=720),
    )
    print_portfolio_report(rp_result)

    # Max Sharpe portfolio
    print("\n" + "=" * 70)
    print("MAX SHARPE PORTFOLIO")
    print("=" * 70)
    ms_result = run_portfolio_backtest(
        strategies, df, config,
        portfolio_config=PortfolioConfig(method="max_sharpe", rebalance_freq="1ME", lookback_bars=720),
    )
    print_portfolio_report(ms_result)


if __name__ == "__main__":
    main()
