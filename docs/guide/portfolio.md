# The Portfolio Object

`Portfolio` is PortPy's single entry point. It holds your price data and weights, and exposes every function in `portpy.metrics` as a bound method under `.metrics`.

## Construction

```python
from portpy import Portfolio

portfolio = Portfolio(
    data,                        # DataFrame: columns = assets, index = DatetimeIndex
    input_type="prices",         # or "returns"
    weights=None,                # Series/dict/array; defaults to equal weight (1/N)
    name="My Portfolio",         # String name used in explanations and plots
    frequency=252,               # trading periods/year (365 for a 24/7 crypto calendar)
    risk_free_rate=0.0,          # annual rate
    asset_classes=None,          # optional {symbol: AssetClass} tags
)
```

Validation happens at construction time: the index must be a sorted, duplicate-free `DatetimeIndex`, and there must be at least two observations. PortPy raises rather than silently fixing malformed input - see [Calendars & currencies](./calendars-and-currency.md) for why (mixed trading calendars are the usual cause).

## Accessors

| Property / method | Returns |
|---|---|
| `.prices` | Price `DataFrame` (reconstructed from returns if `input_type="returns"`). |
| `.asset_returns(log=False)` | Per-asset return `DataFrame` - **not** weight-aggregated. |
| `.returns(period=1, log=False)` | The portfolio's own aggregated return `Series` (weights applied, held constant each period). |
| `.price_index(base=100.0)` | A synthetic portfolio value series, rebased, built by compounding `.returns()`. |
| `.asset_names`, `.num_assets` | Column labels / count. |
| `.weights` | Current weights as a `Series` (sums to 1; may contain negative entries). |

Mutators: `.set_weights(weights)` (validates and re-normalizes) and `.set_risk_free_rate(rate)`.

## The `.metrics` namespace and argument auto-fill

`portfolio.metrics.<name>(...)` looks up `<name>` in `portpy.metrics` and calls it, but first inspects its signature and fills in, **from the portfolio itself**, any parameter named:

- `returns` - `portfolio.returns()` (or `portfolio.asset_returns()` for the two functions that operate on the full multi-asset frame: `covariance_matrix`, `correlation_matrix`)
- `y` - `portfolio.returns()` (used by the regression functions)
- `prices` - `portfolio.price_index()`
- `weights` - `portfolio.weights`
- `cov_matrix` - `portfolio.metrics.covariance_matrix()`
- `rf` - `portfolio.risk_free_rate`
- `periods_per_year` - `portfolio.frequency`

This only happens **on keyword-only calls** - passing any positional argument disables auto-fill for that call entirely, and every parameter falls back to the plain function's own default instead of the portfolio's:

```python
portfolio.metrics.volatility()                             # uses portfolio.frequency (e.g. 365)
portfolio.metrics.volatility(periods_per_year=252)         # override just this one
portfolio.metrics.volatility(portfolio.returns(), True)    # positional - periods_per_year
                                                           # silently reverts to the function's
                                                           # own default of 252, NOT 365
```

That last line is the most essential: mixing positional args with an expectation of auto-fill. Stick to keyword arguments when calling `.metrics.*` and this never bites you.

Parameters that aren't on that list (`benchmark`, `beta`, `method`, `n`, `window`, ...) are never auto-filled - you always supply them explicitly:

```python
portfolio.metrics.beta(benchmark=spy_returns)
portfolio.metrics.value_at_risk(method="cornish_fisher", confidence=0.99)
```

## Standalone functions

Every function in `portpy.metrics` also works without a `Portfolio` at all - pass a plain `pandas.Series`/`DataFrame` directly:

```python
from portpy.metrics.performance import sharpe_ratio

sharpe_ratio(my_returns_series, rf=0.03, periods_per_year=252)
```

`Portfolio.metrics` is a convenience layer on top of these pure functions, not a requirement for using them.
