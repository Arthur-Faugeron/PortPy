import numpy as np
import pandas as pd
import pytest

from portpy.metrics import rolling as m
from portpy.metrics.performance import sharpe_ratio
from portpy.metrics.risk import volatility


def test_rolling_metric_matches_direct_call_on_window(normal_returns):
    window = 60
    rolled = m.rolling_metric(normal_returns, sharpe_ratio, window=window)
    last_window = normal_returns.iloc[-window:]
    assert rolled.iloc[-1] == pytest.approx(sharpe_ratio(last_window))


def test_rolling_metric_has_nan_before_window_fills(normal_returns):
    window = 60
    rolled = m.rolling_metric(normal_returns, sharpe_ratio, window=window)
    assert rolled.iloc[: window - 1].isna().all()
    assert rolled.iloc[window - 1 :].notna().all()


def test_expanding_metric_grows_sample_each_step(normal_returns):
    expanded = m.expanding_metric(normal_returns, volatility, min_periods=10, annualized=False)
    assert expanded.iloc[9] == pytest.approx(volatility(normal_returns.iloc[:10], annualized=False))
    assert expanded.iloc[-1] == pytest.approx(volatility(normal_returns, annualized=False))


def test_rolling_sharpe_matches_generic_engine(normal_returns):
    a = m.rolling_sharpe(normal_returns, window=50, rf=0.0)
    b = m.rolling_metric(normal_returns, sharpe_ratio, window=50, rf=0.0, periods_per_year=252, annualized=True)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_rolling_volatility_matches_manual(normal_returns):
    window = 40
    expected = normal_returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
    got = m.rolling_volatility(normal_returns, window=window, annualized=True)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_rolling_beta_of_series_with_itself_is_one(normal_returns):
    rolled = m.rolling_beta(normal_returns, normal_returns, window=30)
    assert rolled.dropna().apply(lambda v: v == pytest.approx(1.0)).all()


def test_rolling_correlation_of_series_with_itself_is_one(normal_returns):
    rolled = m.rolling_correlation(normal_returns, normal_returns, window=30)
    assert rolled.dropna().apply(lambda v: v == pytest.approx(1.0, abs=1e-9)).all()
