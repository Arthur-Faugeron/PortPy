# estimators

Inputs to construction/optimization: expected returns, covariance, and factor models. Distinct from `portpy.metrics.covariance`, which *decomposes realized risk* for a given weight vector against a given matrix - this module *produces* the matrix itself.

## Expected returns

::: portpy.models.estimators.expected_returns.expected_returns

## Covariance

::: portpy.models.estimators.covariance.covariance

## Factor models

::: portpy.models.estimators.factor_models
    options:
      members:
        - linear_regression
        - rolling_regression
        - capm
        - fama_french
        - factor_attribution
