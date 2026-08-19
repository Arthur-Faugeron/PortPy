# Weights & Short Positions

Weights are just a `pandas.Series` indexed by asset name, summing to 1. PortPy allows negative entries by default, a negative weight is a short position, and every metric downstream (returns, risk, drawdowns, covariance decomposition, ...) treats it exactly as you'd expect: a negative-weighted asset's losses become the portfolio's gains, and vice versa.

## Building weights

```python
from portpy.core import equal_weights, normalize_weights

equal_weights(["AAPL", "MSFT", "GOOGL"])
# AAPL     0.333333
# MSFT     0.333333
# GOOGL    0.333333

normalize_weights({"AAPL": 0.6, "MSFT": 0.3, "XOM": -0.2})
# AAPL    0.857143
# MSFT    0.428571
# XOM    -0.285714
# rescaled to sum to 1 - negative entries are preserved as shorts
```

`normalize_weights` is what `Portfolio.__init__` and `.set_weights()` call internally. It accepts a `Series`, `dict`, or plain array/list (with a matching `names` list), validates there are no `NaN`s and the weights don't sum to zero, and rescales the result to sum to exactly 1.

## Long-only validation

Pass `allow_negative=False` to reject a book with shorts - useful if you're feeding weights into something that assumes long-only (e.g. a future optimizer with a long-only constraint):

```python
normalize_weights({"AAPL": 0.6, "MSFT": 0.3, "XOM": -0.2}, allow_negative=False)
# ValueError: Negative weights are not allowed here (long-only constraint).
```

## A long/short portfolio end to end

```python
from portpy import Portfolio

weights = {
    "AAPL": 0.5,
    "MSFT": 0.5,
    "XOM": -0.2,   # short - a hedge, not held long
}

portfolio = Portfolio(prices, weights=weights, name="Long/Short Tech")
portfolio.weights
# AAPL    0.625
# MSFT    0.625
# XOM    -0.250
# Name: weight, dtype: float64
```

Notice the weights were rescaled so they still sum to 1 (net exposure), while the *relative* proportions you specified (0.5 : 0.5 : -0.2) are preserved. `.returns()` then combines each asset's return by its (possibly negative) weight every period - no special-casing needed anywhere else in the library. `portpy.metrics.covariance.portfolio_variance` and others use the same signed weight vector, so risk decomposition (marginal/component contribution to risk) correctly attributes risk reduction to a short that's negatively correlated with the rest of the book.

## Changing weights later

```python
portfolio.set_weights({"AAPL": 0.7, "MSFT": 0.3})  # re-validates and re-normalizes
```

There is no rebalancing/turnover engine yet (see [Roadmap](../roadmap.md) - `portpy.strategies` isn't built). `set_weights` just replaces the static weight vector used by every subsequent `.metrics` call; see `portpy.metrics.costs` if you want to model transaction costs against an assumed weight-history `DataFrame` you construct yourself.
