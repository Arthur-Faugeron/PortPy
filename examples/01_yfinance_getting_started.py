"""
01 - Getting started with PortPy, using real data from yfinance.

This example is the "hello world" of PortPy: it shows the full flow a new user
goes through:

    1. Acquire raw data (yfinance)
    2. Clean it into the shape PortPy expects (a DataFrame of prices, one
       column per asset, sorted DatetimeIndex, no surprises)
    3. Build a Portfolio
    4. Compute metrics - and, the feature that makes PortPy different from a
       plain numbers-out library, ask each metric to explain itself

Requires yfinance python library -> pip install yfinance

Run:
    python examples/01_yfinance_getting_started.py
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from portpy import Portfolio
from portpy import explain as portpy_explain


# For prettier console output for tables.
pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def fetch_prices(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """
    Download daily close prices for a list of tickers and return a clean wide DataFrame.

    Cleaning steps explained inline below - this is the part every PortPy user
    has to do themselves, since PortPy deliberately does not fetch or clean data.
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    # yfinance's multi-ticker download returns MultiIndex columns: (field, ticker).
    # auto_adjust=True already folds splits and dividends into the price series
    # (this is the single most important cleaning step for equities - without it,
    # a stock split shows up as a fake -50% "return").
    prices = raw["Close"]

    # Two US large-caps trade on the same NYSE/Nasdaq calendar, so in practice
    # there should be no cross-asset gaps here - but check anyway, since silently
    # trusting that is exactly the kind of assumption that breaks later.
    n_missing = prices.isna().sum()
    if n_missing.any():
        print(f"Warning: missing values found per ticker before cleaning:\n{n_missing}")
    prices = prices.dropna(how="any")

    return prices


def fetch_risk_free_rate() -> float:
    """Use the 13-week Treasury Bill yield (^IRX) as a real annual risk-free rate proxy."""
    irx = yf.download("^IRX", period="5d", auto_adjust=True, progress=False)["Close"]
    latest_yield_pct = float(irx.iloc[-1].item())
    return latest_yield_pct / 100.0


def main() -> None:
    tickers = ["AAPL", "MSFT", "GOOGL"]
    benchmark_ticker = "SPY"

    print(f"Downloading {tickers + [benchmark_ticker]} from yfinance...")
    prices = fetch_prices(tickers)
    benchmark_prices = fetch_prices([benchmark_ticker])[benchmark_ticker]
    rf = fetch_risk_free_rate()
    print(f"Using ^IRX as risk-free rate proxy: {rf:.2%} annual\n")

    portfolio = Portfolio(
        prices,
        weights={"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
        name="Tech Portfolio",
        risk_free_rate=rf,
    )
    print(portfolio)
    print(f"Weights: {portfolio.weights.to_dict()}\n")

    benchmark_returns = benchmark_prices.pct_change().dropna()

    # --- The headline metrics, each wrapped as a MetricResult ---
    print("=" * 70)
    print("HEADLINE METRICS")
    print("=" * 70)
    summary = portfolio.metrics.tearsheet_summary()
    for metric_name, result in summary.items():
        print(f"  {metric_name:22s} {float(result):>10.4f}   ({result.interpretation})")

    # --- Ask a couple of metrics to fully explain themselves ---
    print("\n" + "=" * 70)
    print("WHAT DOES 'SHARPE RATIO' ACTUALLY MEAN? (.explain() in action)")
    print("=" * 70)
    sharpe = portfolio.metrics.sharpe_ratio(as_result=True)
    sharpe.explain()

    print("\n" + "=" * 70)
    print("WHAT ABOUT MAX DRAWDOWN?")
    print("=" * 70)
    portfolio.metrics.max_drawdown(as_result=True).explain()

    # --- Compare against a benchmark ---
    print("\n" + "=" * 70)
    print(f"PORTFOLIO VS. BENCHMARK ({benchmark_ticker})")
    print("=" * 70)
    comparison = portfolio.metrics.compare_to_benchmark(benchmark=benchmark_returns)
    print(comparison)

    print("\n" + "=" * 70)
    portpy_explain("compare_to_benchmark")


if __name__ == "__main__":
    main()
