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


@pytest.fixture
def weights_history():
    idx = pd.bdate_range("2020-01-01", periods=3)
    return pd.DataFrame({"A": [0.6, 0.6, 0.5], "B": [0.4, 0.4, 0.5]}, index=idx)


def test_bid_ask_spreads_charges_half_spread_per_leg(weights_history):
    cost = m.bid_ask_spreads(weights_history, spread_bps=20.0)
    np.testing.assert_allclose(cost.to_numpy(), [0.001, 0.0, 0.0002])


def test_bid_ask_spreads_supports_per_asset_dict(weights_history):
    cost = m.bid_ask_spreads(weights_history, spread_bps={"A": 20.0, "B": 20.0})
    uniform = m.bid_ask_spreads(weights_history, spread_bps=20.0)
    np.testing.assert_allclose(cost.to_numpy(), uniform.to_numpy())


def test_bid_ask_spreads_missing_asset_in_dict_raises(weights_history):
    with pytest.raises(ValueError):
        m.bid_ask_spreads(weights_history, spread_bps={"A": 20.0})


def test_broker_commissions_no_half_spread_discount(weights_history):
    cost = m.broker_commissions(weights_history, commission_bps=15.0)
    np.testing.assert_allclose(cost.to_numpy(), [0.0015, 0.0, 0.0003])


def test_fx_spread_costs_zero_when_all_base_currency(weights_history):
    cost = m.fx_spread_costs(
        weights_history, asset_currencies={"A": "USD", "B": "USD"}, base_currency="USD", fx_spread_bps=10.0
    )
    np.testing.assert_allclose(cost.to_numpy(), [0.0, 0.0, 0.0])


def test_fx_spread_costs_only_charges_foreign_currency_assets(weights_history):
    cost = m.fx_spread_costs(
        weights_history, asset_currencies={"A": "EUR", "B": "USD"}, base_currency="USD", fx_spread_bps=10.0
    )
    np.testing.assert_allclose(cost.to_numpy(), [0.0003, 0.0, 0.00005])


def test_tax_impact_only_taxes_positive_periods():
    idx = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.Series([0.05, -0.03, 0.02], index=idx)
    net = m.tax_impact(returns, turnover=0.3, capital_gains_tax_rate=0.2)
    expected = returns - pd.Series([0.05 * 0.3 * 0.2, 0.0, 0.02 * 0.3 * 0.2], index=idx)
    np.testing.assert_allclose(net.to_numpy(), expected.to_numpy())


def test_tax_impact_dividend_drag_applies_regardless_of_sign():
    idx = pd.bdate_range("2020-01-01", periods=2)
    returns = pd.Series([0.05, -0.03], index=idx)
    net = m.tax_impact(
        returns, turnover=0.0, capital_gains_tax_rate=0.2, dividend_tax_rate=0.1, dividend_yield=0.02
    )
    np.testing.assert_allclose(net.to_numpy(), (returns - 0.002).to_numpy())


def test_tax_impact_zero_turnover_and_zero_dividend_is_identity(normal_returns):
    net = m.tax_impact(normal_returns, turnover=0.0, capital_gains_tax_rate=0.3)
    pd.testing.assert_series_equal(net, normal_returns)
