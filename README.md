# PortPy

**Portfolio analysis that explains itself.**

PortPy is a Python library for portfolio performance measurement and risk analysis, built around one object — `Portfolio` — and one idea: every number it gives you can explain, in plain language, what it is, how to read it, and whether it's good or bad.

[![CI](https://github.com/Arthur-Faugeron/PortPy/actions/workflows/ci.yml/badge.svg)](https://github.com/Arthur-Faugeron/PortPy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

```python
import pandas as pd
from portpy import Portfolio

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)  # columns = tickers

portfolio = Portfolio(
    prices,
    weights={"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
    name="Tech Portfolio",
    risk_free_rate=0.04,
)

portfolio.metrics.sharpe_ratio(as_result=True).explain()
```

```text
sharpe_ratio (metric)
=====================

What it is:
  The most widely used risk-adjusted return measure: excess return earned per unit of total volatility taken on.

Formula:
  mean(r - rf) / std(r - rf, ddof=1) * sqrt(periods_per_year)

How to read it:
  A Sharpe of 1.0 means you earned, on average, one standard deviation of excess return for the volatility you took on.

Good vs. bad:
  Rules of thumb: <0 poor (lost money net of the risk-free rate), 0-1 sub-par, 1-2 good, 2-3 very good, >3 excellent (and worth double-checking for overfitting or a very short sample).

Caveats:
  Assumes returns are roughly symmetric - it penalizes upside volatility just as much as downside, and can be misleadingly high for strategies with rare, large negative tail events (e.g. option-selling). Pair with sortino_ratio and max_drawdown.

This result:
  sub-par
```

## Why PortPy

- **One object, one namespace.** `portfolio.metrics.<name>()` auto-fills returns, weights,
  risk-free rate, and annualization frequency from the portfolio itself — no re-threading
  the same five arguments through every call.
- **A real portfolio, not just a return stream.** Weights (negative/short included), asset-class
  tags, and explicit calendar-alignment / currency-conversion helpers for combining assets that
  don't trade on the same schedule (crypto, equities, bonds) or in the same currency.
- **Explains itself.** `.explain()` on any result — a plain-language card covering what it is,
  how to read it, good vs. bad, and known caveats.
- **Numerically validated.** Cross-checked against [`empyrical`](https://github.com/stefan-jansen/empyrical-reloaded)
  and [`quantstats`](https://github.com/ranaroussi/quantstats) on real market data — see
  `tests/validation/`.

## Install

```bash
pip install portpy-quant               # core: numpy, pandas, scipy, statsmodels
pip install "portpy-quant[viz]"        # + plotly, matplotlib, seaborn
pip install "portpy-quant[data]"       # + yfinance, alpaca-py (for the examples)
pip install "portpy-quant[all]"        # everything
```

## Documentation

- [Getting Started](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/getting-started.md)
- [User Guide](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/index.md) — the `Portfolio` object, weights & shorts, calendar/currency
  alignment, the explainability layer
- [API Reference](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/api/index.md) — every function, by module
- [Roadmap](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/roadmap.md) — what's implemented today vs. planned (`models`, `strategies`,
  `visualization` are not yet built)
- [`examples/`](https://github.com/Arthur-Faugeron/PortPy/blob/main/examples/) — runnable scripts and a full tutorial notebook exercising every
  metric against live Alpaca + Fed (FRED) data

## Status

PortPy is pre-1.0 (`Development Status :: 4 - Beta`). **Metrics and core** (calendar/currency
alignment, weights, the explainability layer) are implemented and tested. **Visualization,
models/optimization, and strategies/backtesting** are designed but not yet built — see the
[Roadmap](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/roadmap.md).

## License

[MIT](LICENSE) © Arthur Faugeron