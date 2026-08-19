"""
Shared pytest fixtures: synthetic return/price series with known properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def business_day_index():
    return pd.bdate_range("2020-01-01", periods=504)


@pytest.fixture
def constant_returns(business_day_index):
    """Zero-variance return series - exercises division-by-zero edge cases."""
    return pd.Series(0.0005, index=business_day_index, name="const")


@pytest.fixture
def normal_returns(business_day_index):
    """Seeded Normal(mean=0.0006, std=0.012) daily returns - known moments for approximate checks."""
    rng = np.random.default_rng(42)
    data = rng.normal(0.0006, 0.012, len(business_day_index))
    return pd.Series(data, index=business_day_index, name="normal")


@pytest.fixture
def normal_benchmark(business_day_index):
    rng = np.random.default_rng(7)
    data = rng.normal(0.0004, 0.010, len(business_day_index))
    return pd.Series(data, index=business_day_index, name="benchmark")


@pytest.fixture
def prices_from_normal(normal_returns):
    return (1.0 + normal_returns).cumprod() * 100.0


@pytest.fixture
def short_returns():
    """Too short to compute most metrics - used to assert clear validation errors."""
    idx = pd.bdate_range("2020-01-01", periods=1)
    return pd.Series([0.01], index=idx)


@pytest.fixture
def multi_asset_prices(business_day_index):
    rng = np.random.default_rng(123)
    data = {}
    for name, mu, sigma, start in [("AAPL", 0.0006, 0.018, 150), ("MSFT", 0.0005, 0.016, 120), ("TLT", 0.0001, 0.007, 140)]:
        data[name] = start * np.exp(np.cumsum(rng.normal(mu, sigma, len(business_day_index))))
    return pd.DataFrame(data, index=business_day_index)
