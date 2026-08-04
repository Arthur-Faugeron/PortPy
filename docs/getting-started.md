# Getting Started

## Install

```bash
pip install portpy
```

Optional extras:

```bash
pip install "portpy[viz]"      # plotly, matplotlib, seaborn (for future visualization work)
pip install "portpy[models]"   # scikit-learn, arch, hmmlearn (for future models work)
pip install "portpy[data]"     # yfinance, alpaca-py, python-dotenv (to run examples/)
pip install "portpy[all]"      # everything above
```

Core dependencies (`numpy`, `pandas`, `scipy`, `statsmodels`) are always installed. PortPy
does **not** depend on a data-fetching library — you bring your own prices.

## Your data

`Portfolio` expects a pandas `DataFrame`:

- **Index**: a sorted, duplicate-free `DatetimeIndex`.
- **Columns**: one per asset (ticker symbols, or any label you like).
- **Values**: prices by default (`input_type="prices"`), or periodic returns
  (`input_type="returns"`).

```python
import pandas as pd

prices = pd.DataFrame(
    {
        "AAPL": [180.0, 182.5, 179.0, 185.0],
        "MSFT": [410.0, 415.0, 408.0, 420.0],
    },
    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
)
```

If you're pulling real data from `yfinance` or `alpaca-py`, see
[`examples/01_yfinance_getting_started.py`](https://github.com/Arthur-Faugeron/PortPy/blob/main/examples/01_yfinance_getting_started.py)
and
[`examples/02_alpaca_multiasset_calendar.py`](https://github.com/Arthur-Faugeron/PortPy/blob/main/examples/02_alpaca_multiasset_calendar.py)
for the full fetch-and-clean flow, including calendar alignment for crypto and currency
conversion for non-USD assets.

## Build a Portfolio

```python
from portpy import Portfolio

portfolio = Portfolio(
    prices,
    weights={"AAPL": 0.6, "MSFT": 0.4},     # optional - defaults to equal weight
    name="Tech Portfolio",
    frequency=252,                          # trading periods/year, for annualization
    risk_free_rate=0.04,                    # annual rate, used as the default `rf`
)
print(portfolio)
```

## Compute metrics

Every function in `portpy.metrics` is available as a method on `.metrics`, with the
portfolio's own returns/weights/risk-free rate/frequency filled in automatically:

```python
portfolio.metrics.sharpe_ratio()                # float
portfolio.metrics.sharpe_ratio(as_result=True)  # MetricResult - carries .explain()

portfolio.metrics.max_drawdown(as_result=True).explain()
```

Get a full headline summary in one call:

```python
summary = portfolio.metrics.tearsheet_summary()
for name, result in summary.items():
    print(f"{name:22s} {float(result):>10.4f}   ({result.interpretation})")
```

Compare against a benchmark:

```python
import pandas as pd

benchmark_returns = pd.Series(...)                 # e.g. SPY daily returns, same date range
portfolio.metrics.compare_to_benchmark(benchmark=benchmark_returns)
```

## Next steps

- [The Portfolio object](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/guide/portfolio.md) — what `.metrics` auto-fills, and what it doesn't.
- [Weights & short positions](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/guide/weights.md) — building a long/short book.
- [Calendars & currencies](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/guide/calendars-and-currency.md) — combining crypto with equities,
  or assets in different currencies.
- [Explainability](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/guide/explainability.md) — how `.explain()` works.
- Run [`examples/tutorial.ipynb`](https://github.com/Arthur-Faugeron/PortPy/blob/main/examples/tutorial.ipynb)
  for an exhaustive, working tour of every metric function.
