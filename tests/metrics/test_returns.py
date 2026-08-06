import numpy as np
import pandas as pd
import pytest

from portpy.explain import MetricResult
from portpy.metrics import returns as m

def test_simple_returns_matches_pct_change():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2020-01-01", periods=3))
    r = m.simple_returns(prices)
    assert len(r) == 2
    np.testing.assert_allclose(r.to_numpy(), [0.10, -0.10], atol=1e-9)


def test_log_returns_matches_manual_formula():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2020-01-01", periods=3))
    r = m.log_returns(prices)
    expected = np.log(np.array([110.0 / 100.0, 99.0 / 110.0]))
    np.testing.assert_allclose(r.to_numpy(), expected, atol=1e-9)


def test_cumulative_returns_compounds_correctly():
    r = pd.Series([0.10, -0.10, 0.05])
    cum = m.cumulative_returns(r)
    expected = (1.10 * 0.90 * 1.05) - 1.0
    assert cum.iloc[-1] == pytest.approx(expected)


def test_total_return_known_value():
    r = pd.Series([0.10, -0.10, 0.05])
    value = m.total_return(r)
    assert value == pytest.approx(1.10 * 0.90 * 1.05 - 1.0)


def test_total_return_as_result_is_metric_result():
    r = pd.Series([0.10, -0.10, 0.05])
    result = m.total_return(r, as_result=True)
    assert isinstance(result, MetricResult)
    assert isinstance(result, float)
    assert result.name == "total_return"
    assert result > 0  # float comparisons work directly


def test_annualized_return_geometric_vs_arithmetic(normal_returns):
    geo = m.annualized_return(normal_returns, geometric=True)
    arith = m.annualized_return(normal_returns, geometric=False)
    # Arithmetic annualization overstates vs. geometric for volatile series (variance drag).
    assert arith > geo


def test_annualized_return_constant_series_matches_simple_compounding(constant_returns):
    value = m.annualized_return(constant_returns, periods_per_year=252, geometric=True)
    n = len(constant_returns)
    total = 1.0005**n
    assert value == pytest.approx(total ** (252 / n) - 1.0)


def test_cagr_matches_price_ratio():
    idx = pd.bdate_range("2020-01-01", periods=253)  # 252 periods elapsed => ~1 year
    prices = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    value = m.cagr(prices, periods_per_year=252)
    n_periods = len(prices) - 1
    years = n_periods / 252
    expected = (120.0 / 100.0) ** (1.0 / years) - 1.0
    assert value == pytest.approx(expected)


def test_average_return_geometric_leq_arithmetic(normal_returns):
    arith = m.average_return(normal_returns, geometric=False)
    geo = m.average_return(normal_returns, geometric=True)
    assert geo <= arith


def test_rebased_returns_starts_at_base():
    prices = pd.Series([50.0, 55.0, 60.0])
    rebased = m.rebased_returns(prices, base=100.0)
    assert rebased.iloc[0] == pytest.approx(100.0)
    assert rebased.iloc[-1] == pytest.approx(100.0 * 60.0 / 50.0)


def test_excess_returns_with_scalar_rf():
    r = pd.Series([0.01, 0.02, -0.01])
    excess = m.excess_returns(r, 0.005)
    np.testing.assert_allclose(excess.to_numpy(), r.to_numpy() - 0.005)


def test_excess_returns_with_benchmark_series_aligns_index():
    idx = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0], index=idx)
    b = pd.Series([0.005, 0.01, 0.0, 0.02, 0.01], index=idx)
    excess = m.excess_returns(r, b)
    np.testing.assert_allclose(excess.to_numpy(), (r - b).to_numpy())


def test_prices_from_returns_recovers_original_returns_exactly():
    idx = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([0.02, -0.03, 0.01, -0.05, 0.04], index=idx)
    prices = m.prices_from_returns(r)
    assert len(prices) == len(r) + 1
    recovered = m.simple_returns(prices)
    pd.testing.assert_series_equal(recovered, r, check_names=False, check_freq=False)


def test_prices_from_returns_anchor_is_base_one_period_before_first_date():
    idx = pd.bdate_range("2020-01-08", periods=3)
    r = pd.Series([0.1, -0.1, 0.05], index=idx)
    prices = m.prices_from_returns(r, base=100.0)
    assert prices.iloc[0] == pytest.approx(100.0)
    assert prices.index[0] == idx[0] - (idx[1] - idx[0])


def test_prices_from_returns_captures_first_period_drawdown():
    # A return-reconstructed series without a proper anchor would show 0% drawdown
    # at the first point (since it'd be the series max by construction) - with the
    # anchor, a negative first return is correctly visible as a real drawdown.
    from portpy.metrics.drawdowns import max_drawdown

    idx = pd.bdate_range("2020-01-01", periods=3)
    r = pd.Series([-0.20, 0.01, 0.01], index=idx)
    prices = m.prices_from_returns(r)
    assert max_drawdown(prices) == pytest.approx(-0.20)


def test_prices_from_returns_works_on_dataframe():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [-0.01, 0.02, -0.03]}, index=idx[1:])
    prices = m.prices_from_returns(r)
    assert list(prices.columns) == ["A", "B"]
    assert len(prices) == len(r) + 1


def test_active_returns_is_alias_of_excess_returns_with_series():
    idx = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0], index=idx)
    b = pd.Series([0.005, 0.01, 0.0, 0.02, 0.01], index=idx)
    pd.testing.assert_series_equal(m.active_returns(r, b), m.excess_returns(r, b))
