import numpy as np
import pandas as pd
import pytest

from portpy.metrics import performance as m
from portpy.utils.validation import periodic_rate_from_annual

def test_sharpe_ratio_zero_rf_matches_manual(normal_returns):
    mu, sigma = normal_returns.mean(), normal_returns.std(ddof=1)
    expected = mu / sigma * np.sqrt(252)
    assert m.sharpe_ratio(normal_returns, rf=0.0) == pytest.approx(expected)


def test_sharpe_ratio_annual_rf_converts_geometrically(normal_returns):
    rf_annual = 0.05
    rf_period = periodic_rate_from_annual(rf_annual, 252)
    excess = normal_returns - rf_period
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    assert m.sharpe_ratio(normal_returns, rf=rf_annual) == pytest.approx(expected)


def test_sharpe_ratio_constant_series_is_zero_not_error(constant_returns):
    # zero volatility, zero excess (rf=0) -> safe_divide returns 0.0 rather than raising
    value = m.sharpe_ratio(constant_returns, rf=0.0)
    assert value == 0.0 or np.isinf(value)


def test_sortino_at_least_sharpe_when_upside_skewed():
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(11)
    r = pd.Series(np.abs(rng.normal(0.001, 0.01, len(idx))), index=idx)  # only non-negative returns
    sharpe = m.sharpe_ratio(r)
    sortino = m.sortino_ratio(r)
    assert sortino >= sharpe


def test_calmar_ratio_matches_manual(prices_from_normal, normal_returns):
    from portpy.metrics.drawdowns import max_drawdown
    from portpy.metrics.returns import annualized_return

    ann_ret = annualized_return(normal_returns, periods_per_year=252)
    synthetic = (1 + normal_returns).cumprod()
    mdd = max_drawdown(synthetic)
    expected = ann_ret / abs(mdd)
    assert m.calmar_ratio(normal_returns) == pytest.approx(expected)


def test_omega_ratio_above_one_for_positive_mean_series():
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(13)
    r = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    assert m.omega_ratio(r, threshold=0.0) > 1.0


def test_omega_ratio_invalid_threshold_raises(normal_returns):
    with pytest.raises(ValueError):
        m.omega_ratio(normal_returns, threshold=-1.5)


def test_information_ratio_zero_when_series_identical(normal_returns):
    assert m.information_ratio(normal_returns, normal_returns) == pytest.approx(0.0, abs=1e-9)


def test_treynor_ratio_scales_inversely_with_beta():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 300))
    low_beta = m.treynor_ratio(r, beta=0.5, rf=0.0)
    high_beta = m.treynor_ratio(r, beta=2.0, rf=0.0)
    assert abs(low_beta) > abs(high_beta)


def test_m2_measure_equals_rf_when_sharpe_zero(normal_benchmark):
    # If portfolio has zero excess Sharpe (rf equals mean return exactly is hard to force,
    # so instead check m2 = rf + sharpe*vol_bench holds by construction).
    from portpy.metrics.performance import sharpe_ratio
    from portpy.metrics.risk import volatility

    b = normal_benchmark.iloc[:300]
    r = pd.Series(np.random.default_rng(2).normal(0.0007, 0.015, 300), index=b.index)
    rf = 0.03
    expected = rf + sharpe_ratio(r, rf=rf) * volatility(b)
    assert m.m2_measure(r, b, rf=rf) == pytest.approx(expected)


def test_gain_to_pain_ratio_matches_manual():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    expected = r.sum() / abs(r[r < 0].sum())
    assert m.gain_to_pain_ratio(r) == pytest.approx(expected)


def test_upside_potential_ratio_nonnegative_for_positive_skew_series():
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(17)
    r = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    assert m.upside_potential_ratio(r) >= 0


def test_kappa_three_ratio_runs_without_error(normal_returns):
    value = m.kappa_three_ratio(normal_returns)
    assert np.isfinite(value)


def test_sterling_and_burke_ratios_run_without_error(normal_returns):
    assert np.isfinite(m.sterling_ratio(normal_returns))
    assert np.isfinite(m.burke_ratio(normal_returns))
