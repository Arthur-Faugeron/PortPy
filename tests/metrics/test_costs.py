import numpy as np
import pandas as pd
import pytest

from portpy.metrics import costs as m

def test_turnover_from_weights_first_row_is_half_initial_weights():
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.6, 0.6, 0.5], "B": [0.4, 0.4, 0.5]}, index=idx)
    turnover = m.turnover_from_weights(weights)
    assert turnover.iloc[0] == pytest.approx((0.6 + 0.4) / 2)


def test_turnover_from_weights_matches_manual_diff():
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.6, 0.6, 0.5], "B": [0.4, 0.4, 0.5]}, index=idx)
    turnover = m.turnover_from_weights(weights)
    expected_second = (abs(0.6 - 0.6) + abs(0.4 - 0.4)) / 2
    expected_third = (abs(0.5 - 0.6) + abs(0.5 - 0.4)) / 2
    assert turnover.iloc[1] == pytest.approx(expected_second)
    assert turnover.iloc[2] == pytest.approx(expected_third)


def test_net_of_costs_returns_with_constant_turnover(normal_returns):
    net = m.net_of_costs_returns(normal_returns, turnover=0.1, cost_bps=10)
    expected_drag = 0.1 * (10 / 10_000)
    np.testing.assert_allclose(net.to_numpy(), (normal_returns - expected_drag).to_numpy())


def test_net_of_costs_returns_with_turnover_series(normal_returns):
    turnover = pd.Series(0.2, index=normal_returns.index)
    net = m.net_of_costs_returns(normal_returns, turnover=turnover, cost_bps=5)
    expected_drag = 0.2 * (5 / 10_000)
    np.testing.assert_allclose(net.to_numpy(), (normal_returns - expected_drag).to_numpy())


def test_net_of_costs_returns_zero_cost_is_identity(normal_returns):
    net = m.net_of_costs_returns(normal_returns, turnover=0.5, cost_bps=0.0)
    pd.testing.assert_series_equal(net, normal_returns)
