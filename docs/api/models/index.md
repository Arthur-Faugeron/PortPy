# models

A namespace of namespaces over portfolio construction, optimization, and management. See [Construction, optimization & management](../../guide/models.md) for a worked walkthrough, or `examples/models_debug.ipynb`/`examples/models_tutorial.ipynb` for runnable, end-to-end tours.

| Submodule | Covers |
|---|---|
| [base](base.md) | `ModelResult`, constraint objects (`WeightBounds`, `GroupCap`, `TurnoverCap`, `NetExposure`, `GrossExposure`), and the shared multi-start SLSQP solve routine. |
| [estimators](estimators.md) | Expected returns (mean-historical/EWMA/CAPM-implied/James-Stein), covariance (sample/EWMA/shrinkage/Ledoit-Wolf/robust), and factor models (CAPM, Fama-French, rolling/linear regression, factor attribution). |
| [optimization](optimization.md) | Pure solver functions: mean-variance, max Sharpe, min variance, target return/volatility, efficient frontier, Black-Litterman, hierarchical risk parity, risk parity/budgeting, maximum diversification. |
| [construction](construction.md) | Constraint objects (re-exported from `base`), plus `optimize`/`build`/`efficient_frontier` - the recipe layer over `optimization`. |
| [management](management.md) | `rebalance` (trade lists), `monitor` (drift/limit checks), `compare` (before/after or portfolio-vs-portfolio). |

`Portfolio.models` exposes all of the above bound to a portfolio's own data, with `expected_returns`/`cov_matrix`/`current_weights`/etc. auto-filled - the same `_AutoFillNamespace` pattern `Portfolio.metrics` uses, see [The Portfolio object](../portfolio.md).

Not yet implemented: `portpy.models.simulation` (stochastic processes, Monte Carlo, time series, ML - see the [Roadmap](../../roadmap.md)).
