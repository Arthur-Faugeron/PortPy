# PortPy

Portfolio analysis, risk, and performance metrics that explain themselves.

PortPy centers on one object, `Portfolio`, which holds your price/return data and weights and
exposes every metric through a single, self-documenting namespace:

```python
portfolio.metrics.sharpe_ratio(as_result=True).explain()
```

## Start here

<div class="grid cards" markdown>

- **[Getting Started](getting-started.md)**
  Install PortPy, build your first `Portfolio`, and compute your first metrics.

- **[The Portfolio object](guide/portfolio.md)**
  Prices vs. returns, weights, frequency, risk-free rate, and what `.metrics` auto-fills for you.

- **[Weights & short positions](guide/weights.md)**
  Long/short books, `normalize_weights`, and how negative weights flow through every metric.

- **[Calendars & currencies](guide/calendars-and-currency.md)**
  Combining 24/7 crypto with Mon-Fri equities, and converting multi-currency prices to one base.

- **[Explainability](guide/explainability.md)**
  How `.explain()` works, where the text comes from, and `MetricResult`.

- **[API Reference](api/index.md)**
  Every function, grouped by module, generated straight from the docstrings.

</div>

## Why PortPy

Two things distinguish PortPy from a plain metrics library like `empyrical` or `quantstats`
(both of which PortPy's own test suite validates its numbers against):

1. **It models an actual portfolio.** Weights — including shorts — asset-class tags, and
   explicit calendar/currency alignment are first-class, not something you build by hand
   before calling a flat function library.
2. **It teaches while it computes.** Every metric is registered with a plain-language
   explanation of what it is, how to read it, and whether a given value is good or bad —
   call `.explain()` on any result.

## What's not here yet

`portpy.visualization`, `portpy.models` (optimization, factor models, Monte Carlo), and
`portpy.strategies` (backtesting) are designed but not implemented in this release. See the
[Roadmap](roadmap.md) for what's planned, and [Architecture](architecture.md) for the full
design document.
