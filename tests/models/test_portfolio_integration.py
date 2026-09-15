import pandas as pd
import pytest

from portpy import Portfolio

# `Portfolio.models` shares the `_AutoFillNamespace` mechanism with `Portfolio.metrics`
# (see portfolio.py) - these tests cover that dispatcher/autofill behavior specifically
# for the `.models` namespace, not the underlying estimator/optimizer/builder functions
# themselves (those are covered in the other tests/models/test_*.py files).


def test_portfolio_models_estimators_autofills_asset_level_returns(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    from portpy.models.estimators.expected_returns import expected_returns

    expected = expected_returns(p.asset_returns())
    result = p.models.estimators.expected_returns()
    pd.testing.assert_series_equal(result, expected)


def test_portfolio_models_estimators_covariance_autofills(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    cov = p.models.estimators.covariance()
    assert list(cov.columns) == p.asset_names


def test_portfolio_models_optimization_autofills_expected_returns_and_cov_matrix(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    # A bare keyword call with nothing supplied should still solve, by
    # computing expected_returns/cov_matrix from the portfolio's own data.
    weights = p.models.optimization.min_variance()
    assert weights.sum() == pytest.approx(1.0)


def test_portfolio_models_optimization_positional_args_disable_autofill(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    mu = p.models.estimators.expected_returns()
    # One positional arg (expected_returns) means cov_matrix is NOT auto-filled
    # even though a bare keyword call would have supplied it - same rule as .metrics.
    with pytest.raises(TypeError):
        p.models.optimization.mean_variance(mu)


def test_portfolio_models_top_level_optimize_matches_construction_optimize(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    top_level = p.models.optimize(method="min_variance")
    nested = p.models.construction.optimize(method="min_variance")
    pd.testing.assert_series_equal(top_level.weights, nested.weights)


def test_portfolio_models_build_autofills_asset_returns(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    result = p.models.build(method="hierarchical_risk_parity")
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.meta["built_from"]["n_observations"] == len(p.asset_returns())


def test_portfolio_models_efficient_frontier_autofills(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    result = p.models.efficient_frontier(n_points=8)
    assert "frontier" in result.diagnostics


def test_portfolio_models_turnover_cap_current_weights_autofills(multi_asset_prices):
    p = Portfolio(multi_asset_prices, weights={"AAPL": 0.5, "MSFT": 0.3, "TLT": 0.2})
    cap = p.models.construction.TurnoverCap(max_turnover=0.1)
    pd.testing.assert_series_equal(cap.current_weights, p.weights)


def test_portfolio_models_optimize_fills_bare_turnover_cap_built_via_plain_import(multi_asset_prices):
    # A TurnoverCap constructed via a plain top-level import (not through
    # portfolio.models.construction.TurnoverCap) has current_weights=None baked
    # in - since it's a frozen dataclass, nothing can mutate it after the fact,
    # so .models.optimize/.build/.efficient_frontier must patch the constraints
    # list itself rather than relying on construction-time auto-fill.
    from portpy.models.base import TurnoverCap

    p = Portfolio(multi_asset_prices, weights={"AAPL": 0.5, "MSFT": 0.3, "TLT": 0.2})
    bare_cap = TurnoverCap(max_turnover=0.1)
    assert bare_cap.current_weights is None

    result = p.models.optimize(method="min_variance", constraints=[bare_cap])
    assert result.weights.sum() == pytest.approx(1.0)
    # The original object passed in must be untouched (frozen, and shared by reference elsewhere).
    assert bare_cap.current_weights is None


def test_portfolio_models_build_fills_bare_turnover_cap(multi_asset_prices):
    from portpy.models.base import TurnoverCap

    p = Portfolio(multi_asset_prices)
    result = p.models.build(method="hierarchical_risk_parity", constraints=[TurnoverCap(max_turnover=0.2)])
    assert result.weights.sum() == pytest.approx(1.0)


def test_portfolio_models_optimization_namespace_fills_bare_turnover_cap(multi_asset_prices):
    from portpy.models.base import TurnoverCap

    p = Portfolio(multi_asset_prices)
    weights = p.models.optimization.min_variance(constraints=[TurnoverCap(max_turnover=0.2)])
    assert weights.sum() == pytest.approx(1.0)
