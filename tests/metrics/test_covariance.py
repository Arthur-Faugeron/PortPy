import numpy as np
import pandas as pd
import pytest

from portpy.metrics import covariance as m

@pytest.fixture
def two_asset_returns(business_day_index):
    rng = np.random.default_rng(51)
    a = rng.normal(0.0006, 0.015, len(business_day_index))
    b = rng.normal(0.0004, 0.010, len(business_day_index))
    return pd.DataFrame({"A": a, "B": b}, index=business_day_index)


def test_covariance_matrix_matches_pandas_cov(two_asset_returns):
    cov = m.covariance_matrix(two_asset_returns)
    pd.testing.assert_frame_equal(cov, two_asset_returns.cov())


def test_covariance_matrix_annualized_scales_by_periods(two_asset_returns):
    cov = m.covariance_matrix(two_asset_returns, annualized=True, periods_per_year=252)
    pd.testing.assert_frame_equal(cov, two_asset_returns.cov() * 252)


def test_correlation_matrix_diagonal_is_one(two_asset_returns):
    corr = m.correlation_matrix(two_asset_returns)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), [1.0, 1.0])


def test_portfolio_variance_matches_manual_quadratic_form(two_asset_returns):
    cov = m.covariance_matrix(two_asset_returns)
    w = pd.Series({"A": 0.6, "B": 0.4})
    expected = float(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    assert m.portfolio_variance(w, cov) == pytest.approx(expected)


def test_portfolio_volatility_is_sqrt_of_variance(two_asset_returns):
    cov = m.covariance_matrix(two_asset_returns)
    w = pd.Series({"A": 0.6, "B": 0.4})
    assert m.portfolio_volatility(w, cov) == pytest.approx(np.sqrt(m.portfolio_variance(w, cov)))


def test_diversification_ratio_at_least_one_for_imperfect_correlation(two_asset_returns):
    cov = m.covariance_matrix(two_asset_returns)
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert m.diversification_ratio(w, cov) >= 1.0


def test_diversification_ratio_equals_one_for_perfectly_correlated_assets():
    cov = pd.DataFrame({"A": [0.0004, 0.0004], "B": [0.0004, 0.0004]}, index=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert m.diversification_ratio(w, cov) == pytest.approx(1.0)


def test_component_contribution_to_risk_sums_to_portfolio_volatility(two_asset_returns):
    cov = m.covariance_matrix(two_asset_returns)
    w = pd.Series({"A": 0.6, "B": 0.4})
    cctr = m.component_contribution_to_risk(w, cov)
    assert cctr.sum() == pytest.approx(m.portfolio_volatility(w, cov))


def test_marginal_contribution_to_risk_matches_manual_formula(two_asset_returns):
    cov = m.covariance_matrix(two_asset_returns)
    w = pd.Series({"A": 0.6, "B": 0.4})
    port_vol = m.portfolio_volatility(w, cov)
    expected = (cov.to_numpy() @ w.to_numpy()) / port_vol
    np.testing.assert_allclose(m.marginal_contribution_to_risk(w, cov), expected)
