import numpy as np
import pytest

from portpy.explain import MetricResult
from portpy.metrics import summary as m

def test_tearsheet_summary_returns_metric_results(normal_returns, prices_from_normal):
    result = m.tearsheet_summary(normal_returns, prices=prices_from_normal)
    expected_keys = {
        "total_return", "annualized_return", "cagr", "volatility", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "max_drawdown", "value_at_risk_95",
        "conditional_var_95", "skewness", "kurtosis", "win_rate",
    }
    assert set(result.keys()) == expected_keys
    for value in result.values():
        assert isinstance(value, MetricResult)
        assert np.isfinite(float(value))


def test_tearsheet_summary_reconstructs_prices_when_not_given(normal_returns):
    result = m.tearsheet_summary(normal_returns)
    assert np.isfinite(float(result["cagr"]))


def test_compare_to_benchmark_has_difference_column(normal_returns, normal_benchmark):
    table = m.compare_to_benchmark(normal_returns, normal_benchmark)
    assert "difference" in table.columns
    ann_row = table.loc["annualized_return"]
    assert ann_row["difference"] == pytest.approx(ann_row["portfolio"] - ann_row["benchmark"])


def test_compare_to_benchmark_relative_only_rows_have_nan_benchmark(normal_returns, normal_benchmark):
    table = m.compare_to_benchmark(normal_returns, normal_benchmark)
    for row in ["beta", "alpha", "correlation", "information_ratio"]:
        assert np.isnan(table.loc[row, "benchmark"])


def test_compare_to_benchmark_has_explain_attrs(normal_returns, normal_benchmark):
    table = m.compare_to_benchmark(normal_returns, normal_benchmark)
    assert table.attrs["portpy_explanation"] == "compare_to_benchmark"
