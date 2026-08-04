import numpy as np
import pandas as pd
import pytest

from portpy.metrics import distributions as m


def test_describe_includes_skew_and_kurtosis(normal_returns):
    desc = m.describe(normal_returns)
    assert "skew" in desc.index
    assert "kurtosis" in desc.index
    assert desc["count"] == len(normal_returns)


def test_normality_test_flags_normal_data_as_normal():
    rng = np.random.default_rng(21)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    result = m.normality_test(r)
    assert result["is_normal"] is True
    assert result["p_value"] > 0.05


def test_normality_test_flags_heavy_tailed_data_as_not_normal():
    rng = np.random.default_rng(21)
    r = pd.Series(rng.standard_t(df=2, size=5000) * 0.01)
    result = m.normality_test(r)
    assert result["is_normal"] is False


def test_best_worst_periods_returns_correct_extremes():
    r = pd.Series([0.05, -0.03, 0.01, -0.10, 0.08, 0.0])
    result = m.best_worst_periods(r, n=2)
    assert list(result["best"].to_numpy()) == [0.08, 0.05]
    assert list(result["worst"].to_numpy()) == [-0.10, -0.03]


def test_win_rate_matches_fraction_positive():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    assert m.win_rate(r) == pytest.approx(2 / 5)


def test_win_loss_ratio_matches_manual():
    r = pd.Series([0.02, 0.04, -0.01, -0.03])
    expected = ((0.02 + 0.04) / 2) / ((0.01 + 0.03) / 2)
    assert m.win_loss_ratio(r) == pytest.approx(expected)


def test_positive_periods_pct_equals_win_rate(normal_returns):
    assert m.positive_periods_pct(normal_returns) == pytest.approx(m.win_rate(normal_returns))


def test_monthly_returns_table_shape_and_year_column(normal_returns):
    table = m.monthly_returns_table(normal_returns)
    assert "Year" in table.columns
    assert set(table.columns[:12]) == {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    }


def test_monthly_returns_table_year_column_matches_annual_compounding(normal_returns):
    table = m.monthly_returns_table(normal_returns)
    manual = (1 + normal_returns).groupby(normal_returns.index.year).prod() - 1
    pd.testing.assert_series_equal(table["Year"], manual, check_names=False)


def test_return_histogram_data_matches_numpy_histogram(normal_returns):
    counts, edges = m.return_histogram_data(normal_returns, bins=20)
    expected_counts, expected_edges = np.histogram(normal_returns.dropna().to_numpy(), bins=20)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(edges, expected_edges)
