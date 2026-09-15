# Construction, Optimization & Management

`portpy.models` answers three questions, in order: what should I assume about returns and risk (`.estimators`), what weights should I hold given those assumptions (`.optimization` / `.construction`), and how do I get from my current book to that target and stay honest about drift afterward (`.management`). Every call below also works as a plain function (`from portpy.models.optimization import mean_variance`) - `.models` is just the auto-filling, portfolio-bound convenience layer, the same pattern `.metrics` already uses.

## Estimating inputs

Every optimizer needs an expected-return vector (`mu`) and a covariance matrix (`Sigma`) - neither is observable, both are estimates:

```python
mu = portfolio.models.estimators.expected_returns(method="james_stein")
cov = portfolio.models.estimators.covariance(method="ledoit_wolf")
```

`method="mean_historical"` (the default) is the simplest and noisiest choice; `"james_stein"` shrinks toward the cross-sectional grand mean, and `"capm_implied"` (needs `benchmark=`) backs out returns from each asset's beta instead of its own history. For covariance, `"sample"` is the raw matrix; `"ledoit_wolf"` and `"shrinkage"` trade a little bias for much better conditioning when your history isn't long relative to the number of assets. See the [API reference](../api/models/estimators.md) for every method and its caveats.

`portpy.models.estimators.factor_models` also has `capm`, `fama_french`, `linear_regression`, `rolling_regression`, and `factor_attribution` - thin, opinionated wrappers around `statsmodels` OLS.

## Solving for weights

```python
result = portfolio.models.optimize(method="max_sharpe")
result.weights          # pandas Series, indexed by asset
result.summary()        # weights sorted by size
result.explain()        # what max_sharpe is, how to read it, caveats
```

Called bare like this, `expected_returns`/`cov_matrix` are auto-filled from the portfolio's own data (default estimator methods) - pass them explicitly to use a different estimate. `method` is one of `"mean_variance"`, `"max_sharpe"`, `"min_variance"`, `"target_return"`, `"target_volatility"`, `"black_litterman"`, `"hierarchical_risk_parity"`, `"risk_parity"`, `"risk_budgeting"`, `"maximum_diversification"` - see [API reference](../api/models/optimization.md) for what each one optimizes and needs.

`.models.build(method=...)` is the same thing starting one step earlier - it estimates `mu`/`Sigma` for you first:

```python
result = portfolio.models.build(method="mean_variance", covariance_kwargs={"method": "ledoit_wolf"})
```

`.models.efficient_frontier(n_points=50)` traces the whole frontier instead of one point - `result.weights` is the max-Sharpe point on the curve, and `result.summary()` returns the full return/volatility/weights table.

Every one of these returns a `ModelResult` - `portpy.explain`'s model-layer counterpart to `MetricResult`. `.diagnostics` carries solver internals (`success`, `objective_value`, `iterations`); always check `diagnostics["success"]` before trusting a `target_return`/`target_volatility` solve, since an infeasible target fails to converge silently rather than raising.

## Constraints

```python
from portpy.models.construction import GroupCap, TurnoverCap, WeightBounds

result = portfolio.models.optimize(
    method="mean_variance",
    constraints=[
        WeightBounds(low=0.0, high=0.40),                                  # no more than 40% in any one asset
        GroupCap(groups={"equity": ["AAPL", "MSFT", "JPM"]}, max_weight=0.60),
        TurnoverCap(max_turnover=0.15),                                    # current_weights auto-filled from the portfolio
    ],
)
```

A full-investment constraint (`sum(w) == 1`) is applied automatically unless you pass `GrossExposure` instead (for a long/short book, e.g. a 130/30 fund uses `GrossExposure(1.6)`). `hierarchical_risk_parity` only supports `WeightBounds` (applied post-hoc); every other constraint on it is dropped with a warning, since recursive bisection has no natural way to target a group cap or turnover budget.

`TurnoverCap.max_turnover` caps the raw `sum(abs(w - current_weights))` - **not** divided by
2. That makes it twice `metrics.costs.turnover_from_weights`'s "one-way turnover" convention
(selling 10% of A to buy 10% of B reads as 10% one-way turnover, but 20% under
`TurnoverCap`'s raw-sum convention) - `TurnoverCap(max_turnover=0.15)` above is a 7.5%
one-way turnover budget, not 15%.

## Managing a book afterward

An optimizer gives you a destination, not a plan to get there:

```python
plan = portfolio.models.management.rebalance(target_weights=result, method="threshold", threshold=0.03, cost_bps=8)
plan.diagnostics["trades"]           # signed per-asset trade
plan.diagnostics["turnover"]         # one-way turnover
plan.diagnostics["estimated_cost"]   # flat, linear-in-turnover cost model
```

`method="threshold"` only trades assets that have drifted past `threshold`; `"full"`/`"calendar"` trade all the way to target regardless (the two are identical here - `rebalance` is a stateless, point-in-time trade-list generator, not a scheduler).

```python
report = portfolio.models.management.monitor(limits={"max_weight": 0.40, "max_gross": 1.0})
report["ok"]           # bool
report["breaches"]     # [{"type", "asset"/"group", "value", "limit"}, ...]

comparison = portfolio.models.management.compare(other_portfolio_or_result)
comparison.diagnostics["weight_delta"]   # always available
comparison.diagnostics["metric_delta"]   # only when both sides are Portfolios with return history
```

## What's not here yet

`ModelResult.plot()` raises `NotImplementedError` until `portpy.visualization` ships - use `.summary()` and `.explain()` in the meantime. `portpy.models.simulation` (stochastic processes, Monte Carlo, ARIMA/GARCH, PCA/regime detection) is a separate, later stage - see the [Roadmap](../roadmap.md).
