import numpy as np
import pandas as pd
import pytest

from portpy.metrics import risk as m


def test_volatility_annualized_matches_manual(normal_returns):
    std = normal_returns.std(ddof=1)
    expected = std * np.sqrt(252)
    assert m.volatility(normal_returns, annualized=True) == pytest.approx(expected)
    assert m.volatility(normal_returns, annualized=False) == pytest.approx(std)


def test_volatility_constant_series_is_zero(constant_returns):
    assert m.volatility(constant_returns) == pytest.approx(0.0, abs=1e-12)


def test_downside_deviation_ignores_upside(business_day_index):
    only_gains = pd.Series(0.01, index=business_day_index)
    assert m.downside_deviation(only_gains, mar=0.0, annualized=False) == pytest.approx(0.0)


def test_downside_deviation_matches_manual_formula():
    r = pd.Series([0.02, -0.03, 0.01, -0.05, 0.0])
    shortfall = np.minimum(r.to_numpy(), 0.0)
    expected = np.sqrt(np.mean(shortfall**2))
    assert m.downside_deviation(r, mar=0.0, annualized=False) == pytest.approx(expected)


def test_semi_variance_uses_full_sample_denominator():
    r = pd.Series([0.02, -0.03, 0.01, -0.05, 0.0])
    shortfall = np.minimum(r.to_numpy() - 0.0, 0.0)
    expected = float(np.mean(shortfall**2))  # divides by N=5, not count-below-threshold=2
    assert m.semi_variance(r, mar=0.0) == pytest.approx(expected)


def test_semi_variance_matches_downside_deviation_squared():
    r = pd.Series([0.02, -0.03, 0.01, -0.05, 0.0])
    sv = m.semi_variance(r, mar=0.0)
    dd = m.downside_deviation(r, mar=0.0, annualized=False)
    # Both divide by the full sample size N, so sqrt(semi_variance) == downside_deviation.
    assert sv == pytest.approx(dd**2)


def test_value_at_risk_historical_matches_percentile():
    r = pd.Series(np.linspace(-0.10, 0.10, 101))
    value = m.value_at_risk(r, method="historical", confidence=0.95)
    assert value == pytest.approx(np.percentile(r, 5))


def test_conditional_var_is_at_least_as_negative_as_var():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.02, 300))
    var = m.value_at_risk(r, confidence=0.95)
    cvar = m.conditional_var(r, confidence=0.95)
    assert cvar <= var


def test_value_at_risk_parametric_and_cornish_fisher_run(normal_returns):
    for method in ("historical", "parametric", "cornish_fisher"):
        value = m.value_at_risk(normal_returns, method=method)
        assert np.isfinite(value)


def test_value_at_risk_invalid_method_raises(normal_returns):
    with pytest.raises(ValueError):
        m.value_at_risk(normal_returns, method="bogus")


def test_tail_ratio_symmetric_distribution_near_one():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    assert m.tail_ratio(r) == pytest.approx(1.0, rel=0.25)


def test_skewness_and_kurtosis_of_normal_data_near_zero():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0, 0.01, 20000))
    assert m.skewness(r) == pytest.approx(0.0, abs=0.1)
    assert m.kurtosis(r) == pytest.approx(0.0, abs=0.2)


def test_ulcer_index_zero_for_monotonic_increasing_prices():
    prices = pd.Series(np.linspace(100, 200, 50))
    assert m.ulcer_index(prices) == pytest.approx(0.0, abs=1e-9)


def test_pain_index_zero_for_monotonic_increasing_prices():
    prices = pd.Series(np.linspace(100, 200, 50))
    assert m.pain_index(prices) == pytest.approx(0.0, abs=1e-9)


def test_beta_of_series_with_itself_is_one(normal_returns):
    assert m.beta(normal_returns, normal_returns) == pytest.approx(1.0)


def test_beta_matches_cov_over_var(normal_returns, normal_benchmark):
    cov = np.cov(normal_returns, normal_benchmark, ddof=1)[0, 1]
    var_b = np.var(normal_benchmark, ddof=1)
    assert m.beta(normal_returns, normal_benchmark) == pytest.approx(cov / var_b)


def test_tracking_error_zero_when_series_identical(normal_returns):
    assert m.tracking_error(normal_returns, normal_returns) == pytest.approx(0.0, abs=1e-12)
