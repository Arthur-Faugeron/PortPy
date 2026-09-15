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
  Measures excess return earned per unit of total volatility taken, using the risk-free rate as the return hurdle.

Formula:
  mean(r - rf) / std(r - rf, ddof=1) * sqrt(periods_per_year)

How to read it:
  A Sharpe of 1.0 means the strategy generated approximately one unit of excess return for each unit of volatility. Higher values indicate better risk-adjusted performance.

Good vs. bad:
  Higher is generally better. Values above 1 are commonly considered strong, but interpretation depends on the asset class, time period, and strategy complexity.

Caveats:
  Sharpe treats all volatility as bad, including upside volatility. It can also overstate strategies with asymmetric downside risk, illiquidity, or short backtests. The sqrt(periods_per_year) annualization assumes i.i.d., serially uncorrelated returns - positive autocorrelation (illiquid/infrequently-priced assets) means the annualized figure is overstated (Lo, 2002).

This result:
  very good
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
  and [`quantstats`](https://github.com/ranaroussi/quantstats) for metrics, and
  [`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt),
  [`Riskfolio-Lib`](https://github.com/dcajasn/Riskfolio-Lib), and
  [`skfolio`](https://github.com/skfolio/skfolio) for every optimizer — including
  negative-weight (short) solutions, not just long-only — on real market data, see
  `tests/validation/`.
- **Builds portfolios, not just measures them.** `portfolio.models` estimates expected
  returns/covariance, solves for weights (mean-variance, max Sharpe, risk parity, hierarchical
  risk parity, Black-Litterman, ...), and manages a book afterward (rebalancing, drift
  monitoring, before/after comparisons) — every result explains itself the same way metrics do.

## Install

```bash
pip install portpy-quant               # core: numpy, pandas, scipy, statsmodels
pip install "portpy-quant[viz]"        # + plotly, matplotlib, seaborn
pip install "portpy-quant[data]"       # + yfinance, alpaca-py (for the examples)
pip install "portpy-quant[all]"        # everything
```

## Documentation

- [Getting Started](./docs/getting-started.md)
- [User Guide](./docs/index.md) — the `Portfolio` object, weights & shorts, calendar/currency
  alignment, the explainability layer
- [API Reference](./docs/api/index.md) — every function, by module
- [Roadmap](./docs/roadmap.md) — what's implemented today vs. planned (`visualization` and
  `strategies` are not yet built)
- [`examples/`](./examples/) — runnable scripts, `metrics_debug.ipynb`/`metrics_tutorial.ipynb`
  exercising every metric against live Alpaca + Fed (FRED) data, and
  `models_debug.ipynb`/`models_tutorial.ipynb` doing the same for `.models`

## Status

PortPy is pre-1.0 (`Development Status :: 4 - Beta`). **Metrics and core** (calendar/currency
alignment, weights, the explainability layer) and **`.models`** (estimators, optimization,
construction, management) are implemented and tested. **Visualization and
strategies/backtesting** are designed but not yet built — see the [Roadmap](./docs/roadmap.md).

## License

[MIT](LICENSE) © Arthur Faugeron