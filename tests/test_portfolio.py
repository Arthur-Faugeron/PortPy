import numpy as np
import pandas as pd
import pytest

from portpy import Portfolio
from portpy.core.weights import equal_weights

def test_portfolio_requires_dataframe():
    with pytest.raises(TypeError):
        Portfolio(pd.Series([1.0, 2.0]))


def test_portfolio_requires_datetime_index():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError):
        Portfolio(df)


def test_portfolio_default_weights_are_equal(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    expected = equal_weights(list(multi_asset_prices.columns))
    pd.testing.assert_series_equal(p.weights, expected, check_names=False)


def test_portfolio_from_returns_reconstructs_prices(multi_asset_prices):
    from portpy.metrics.returns import simple_returns

    asset_returns = simple_returns(multi_asset_prices)
    p = Portfolio(asset_returns, input_type="returns")
    # One extra anchor row (base=1.0) before the first return date.
    assert len(p.prices) == len(asset_returns) + 1
    assert p.prices.iloc[0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    # Recomputing simple returns on the reconstructed prices must recover the
    # original returns exactly, including the first observation (the anchor
    # is what makes that possible - see prices_from_returns).
    recovered = simple_returns(p.prices)
    pd.testing.assert_frame_equal(recovered, asset_returns, check_names=False, check_freq=False)


def test_portfolio_set_weights_rejects_unknown_asset(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    with pytest.raises(ValueError):
        p.set_weights({"NOT_AN_ASSET": 1.0})


def test_portfolio_returns_is_weighted_combination(multi_asset_prices):
    p = Portfolio(multi_asset_prices, weights={"AAPL": 0.5, "MSFT": 0.3, "TLT": 0.2})
    asset_r = p.asset_returns()
    expected = (asset_r * p.weights).sum(axis=1)
    pd.testing.assert_series_equal(p.returns(), expected, check_names=False)


def test_portfolio_returns_period_compounds_over_blocks(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    daily = p.returns()
    weekly = p.returns(period=5)
    # 5-day compounded return of first block should match manual compounding.
    manual_first_block = float((1 + daily.iloc[:5]).prod() - 1)
    assert weekly.iloc[0] == pytest.approx(manual_first_block)


def test_portfolio_price_index_starts_near_base(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    idx = p.price_index(base=100.0)
    # First row is the anchor (base, one period before the first return); the
    # second row is where the first real return is actually reflected.
    assert idx.iloc[0] == pytest.approx(100.0)
    assert idx.iloc[1] == pytest.approx(100.0 * (1 + p.returns().iloc[0]))
    assert len(idx) == len(p.returns()) + 1


def test_portfolio_asset_classes_validates_unknown_assets(multi_asset_prices):
    from portpy.core.asset import AssetClass

    with pytest.raises(ValueError):
        Portfolio(multi_asset_prices, asset_classes={"BOGUS": AssetClass.EQUITY})


def test_portfolio_metrics_dispatcher_autofills_returns(multi_asset_prices):
    p = Portfolio(multi_asset_prices, risk_free_rate=0.02)
    from portpy.metrics.performance import sharpe_ratio

    expected = sharpe_ratio(p.returns(), rf=0.02, periods_per_year=252)
    assert p.metrics.sharpe_ratio() == pytest.approx(expected)


def test_portfolio_metrics_dispatcher_autofills_prices(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    from portpy.metrics.drawdowns import max_drawdown

    expected = max_drawdown(p.price_index())
    assert p.metrics.max_drawdown() == pytest.approx(expected)


def test_portfolio_metrics_dispatcher_uses_asset_level_returns_for_covariance(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    cov = p.metrics.covariance_matrix()
    assert list(cov.columns) == p.asset_names


def test_portfolio_metrics_dispatcher_autofills_weights_and_cov_matrix(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    vol = p.metrics.portfolio_volatility()
    assert np.isfinite(vol)
    assert vol == pytest.approx(p.metrics.portfolio_volatility(weights=p.weights, cov_matrix=p.metrics.covariance_matrix()))


def test_portfolio_metrics_dispatcher_requires_explicit_benchmark(multi_asset_prices, normal_benchmark):
    p = Portfolio(multi_asset_prices)
    with pytest.raises(TypeError):
        p.metrics.beta()
    value = p.metrics.beta(benchmark=normal_benchmark)
    assert np.isfinite(value)


def test_portfolio_set_risk_free_rate_updates_default_used_by_dispatcher(multi_asset_prices):
    p = Portfolio(multi_asset_prices)
    p.set_risk_free_rate(0.05)
    assert p.risk_free_rate == pytest.approx(0.05)
    from portpy.metrics.performance import sharpe_ratio

    expected = sharpe_ratio(p.returns(), rf=0.05, periods_per_year=252)
    assert p.metrics.sharpe_ratio() == pytest.approx(expected)


def test_portfolio_repr_contains_key_info(multi_asset_prices):
    p = Portfolio(multi_asset_prices, name="Test Portfolio")
    text = repr(p)
    assert "Test Portfolio" in text
    assert "assets=3" in text
