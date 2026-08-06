import numpy as np
import pandas as pd
import pytest

from portpy.metrics import benchmarks as m

def test_correlation_of_series_with_itself_is_one(normal_returns):
    assert m.correlation(normal_returns, normal_returns) == pytest.approx(1.0)


def test_r_squared_equals_correlation_squared(normal_returns, normal_benchmark):
    corr = m.correlation(normal_returns, normal_benchmark)
    assert m.r_squared(normal_returns, normal_benchmark) == pytest.approx(corr**2)


def test_alpha_zero_when_returns_equal_beta_times_benchmark(normal_benchmark):
    beta_true = 1.3
    r = beta_true * normal_benchmark
    value = m.alpha(r, normal_benchmark, rf=0.0)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_up_capture_ratio_above_one_when_outperforming_in_up_periods():
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(31)
    b = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    r = b * 1.5  # amplified benchmark
    assert m.up_capture_ratio(r, b) > 1.0


def test_down_capture_ratio_below_one_for_defensive_series():
    idx = pd.bdate_range("2020-01-01", periods=252)
    rng = np.random.default_rng(33)
    b = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    r = b * 0.5  # dampened benchmark: loses less when b<0
    assert m.down_capture_ratio(r, b) < 1.0


def test_capture_ratio_one_when_series_identical(normal_returns):
    assert m.capture_ratio(normal_returns, normal_returns) == pytest.approx(1.0)


def test_batting_average_matches_manual(normal_returns, normal_benchmark):
    r, b = normal_returns.align(normal_benchmark, join="inner")
    expected = float((r > b).mean())
    assert m.batting_average(normal_returns, normal_benchmark) == pytest.approx(expected)
