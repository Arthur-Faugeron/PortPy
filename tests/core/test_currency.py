import numpy as np
import pandas as pd
import pytest

from portpy.core.currency import calculate_fx_spread, convert_to_base_currency


@pytest.fixture
def prices_and_fx():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        {"US_STOCK": [100.0] * 5, "EU_STOCK": [50.0] * 5},
        index=idx,
    )
    fx = pd.DataFrame({"EUR": [1.10, 1.11, 1.12, 1.13, 1.14]}, index=idx)
    return prices, fx


def test_convert_to_base_currency_leaves_base_currency_untouched(prices_and_fx):
    prices, fx = prices_and_fx
    converted = convert_to_base_currency(
        prices, fx, asset_currencies={"US_STOCK": "USD", "EU_STOCK": "EUR"}, base_currency="USD"
    )
    pd.testing.assert_series_equal(converted["US_STOCK"], prices["US_STOCK"])


def test_convert_to_base_currency_applies_fx_multiplier(prices_and_fx):
    prices, fx = prices_and_fx
    converted = convert_to_base_currency(
        prices, fx, asset_currencies={"US_STOCK": "USD", "EU_STOCK": "EUR"}, base_currency="USD"
    )
    expected = prices["EU_STOCK"] * fx["EUR"]
    np.testing.assert_allclose(converted["EU_STOCK"].to_numpy(), expected.to_numpy())


def test_convert_to_base_currency_missing_fx_column_raises(prices_and_fx):
    prices, fx = prices_and_fx
    with pytest.raises(ValueError):
        convert_to_base_currency(
            prices, fx, asset_currencies={"EU_STOCK": "GBP"}, base_currency="USD"
        )


def test_convert_to_base_currency_ffills_mismatched_fx_calendar():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"EU_STOCK": [50.0] * 5}, index=idx)
    sparse_fx = pd.DataFrame({"EUR": [1.10]}, index=[idx[0]])
    converted = convert_to_base_currency(
        prices, sparse_fx, asset_currencies={"EU_STOCK": "EUR"}, base_currency="USD"
    )
    np.testing.assert_allclose(converted["EU_STOCK"].to_numpy(), 50.0 * 1.10)


def test_convert_to_base_currency_max_staleness_limits_ffill():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"EU_STOCK": [50.0] * 5}, index=idx)
    sparse_fx = pd.DataFrame({"EUR": [1.10]}, index=[idx[0]])
    converted = convert_to_base_currency(
        prices, sparse_fx, asset_currencies={"EU_STOCK": "EUR"}, base_currency="USD", max_staleness=1
    )
    assert converted["EU_STOCK"].iloc[0] == pytest.approx(55.0)
    assert converted["EU_STOCK"].iloc[1] == pytest.approx(55.0)
    assert converted["EU_STOCK"].iloc[2:].isna().all()


def test_convert_to_base_currency_unlimited_staleness_is_default():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"EU_STOCK": [50.0] * 5}, index=idx)
    sparse_fx = pd.DataFrame({"EUR": [1.10]}, index=[idx[0]])
    converted = convert_to_base_currency(
        prices, sparse_fx, asset_currencies={"EU_STOCK": "EUR"}, base_currency="USD"
    )
    assert not converted["EU_STOCK"].isna().any()


def test_calculate_fx_spread_matches_manual_formula():
    idx = pd.bdate_range("2020-01-01", periods=3)
    bid = pd.DataFrame({"EUR": [1.10, 1.10, 1.10]}, index=idx)
    ask = pd.DataFrame({"EUR": [1.12, 1.12, 1.12]}, index=idx)
    spread = calculate_fx_spread(bid, ask)
    expected = (1.12 - 1.10) / ((1.12 + 1.10) / 2.0)
    np.testing.assert_allclose(spread["EUR"].to_numpy(), expected)


def test_calculate_fx_spread_zero_when_bid_equals_ask():
    idx = pd.bdate_range("2020-01-01", periods=2)
    quotes = pd.DataFrame({"EUR": [1.10, 1.11]}, index=idx)
    spread = calculate_fx_spread(quotes, quotes)
    np.testing.assert_allclose(spread["EUR"].to_numpy(), [0.0, 0.0])


def test_calculate_fx_spread_mismatched_columns_raises():
    idx = pd.bdate_range("2020-01-01", periods=2)
    bid = pd.DataFrame({"EUR": [1.10, 1.11]}, index=idx)
    ask = pd.DataFrame({"GBP": [1.30, 1.31]}, index=idx)
    with pytest.raises(ValueError):
        calculate_fx_spread(bid, ask)
