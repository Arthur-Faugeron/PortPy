# optimization

Pure solver functions: `(expected_returns, cov_matrix, constraints) -> weights`. Every function here is importable directly, with no dependency on `Portfolio` - see [construction](./construction.md) for the discoverable, `ModelResult`-returning dispatcher over these (`optimize`/`build`), and [estimators](./estimators.md) for computing `expected_returns`/`cov_matrix`.

::: portpy.models.optimization.solvers
    options:
      members:
        - mean_variance
        - min_variance
        - max_sharpe
        - target_return
        - target_volatility
        - efficient_frontier
        - black_litterman
        - hierarchical_risk_parity
        - risk_parity
        - risk_budgeting
        - maximum_diversification
