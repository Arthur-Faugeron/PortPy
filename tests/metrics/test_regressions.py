import numpy as np
import pandas as pd
import pytest

from portpy.metrics import regressions as m

def test_linear_regression_recovers_known_beta():
    rng = np.random.default_rng(41)
    idx = pd.bdate_range("2020-01-01", periods=500)
    x = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    noise = rng.normal(0, 0.0005, len(idx))
    y = 0.0002 + 1.7 * x + noise
    y = pd.Series(y, index=idx)

    model = m.linear_regression(y, x)
    assert model.params["x"] == pytest.approx(1.7, abs=0.05)
    assert model.params["const"] == pytest.approx(0.0002, abs=0.001)


def test_rolling_regression_returns_dataframe_with_expected_columns():
    rng = np.random.default_rng(43)
    idx = pd.bdate_range("2020-01-01", periods=300)
    x = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    y = pd.Series(0.8 * x + rng.normal(0, 0.001, len(idx)), index=idx)

    rolling = m.rolling_regression(y, x, window=60)
    assert set(rolling.columns) == {"const", "x"}
    assert rolling["x"].dropna().mean() == pytest.approx(0.8, abs=0.2)


def test_regression_summary_has_expected_columns_and_attrs():
    rng = np.random.default_rng(45)
    idx = pd.bdate_range("2020-01-01", periods=300)
    x = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    y = pd.Series(0.5 * x + rng.normal(0, 0.002, len(idx)), index=idx)

    model = m.linear_regression(y, x)
    summary = m.regression_summary(model)
    assert list(summary.columns) == ["coef", "std_err", "t_stat", "p_value", "conf_low", "conf_high"]
    assert 0.0 <= summary.attrs["r_squared"] <= 1.0
    assert summary.attrs["portpy_explanation"] == "regression_summary"
