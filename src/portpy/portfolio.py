"""The `Portfolio` class: PortPy's single entry point for analysis.

Holds price data and weights, and exposes every function in
:mod:`portpy.metrics` as a bound method under `.metrics`, automatically
supplying the portfolio's own returns/prices/weights/risk-free rate as
defaults wherever a metric function needs them.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from portpy import metrics as _metrics_module
from portpy.core.asset import AssetClass
from portpy.core.weights import equal_weights, normalize_weights
from portpy.metrics.returns import log_returns, prices_from_returns, simple_returns
from portpy.utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import ensure_datetime_index, ensure_min_observations

__all__ = ["Portfolio"]

_INPUT_TYPES = ("prices", "returns")

# Functions where the "returns" parameter means the full multi-asset DataFrame
# (not the portfolio's own aggregated return series).
_ASSET_LEVEL_RETURNS_FUNCS = frozenset({"covariance_matrix", "correlation_matrix"})


class _MetricsNamespace:
    """`portfolio.metrics.<name>(...)` - every function in `portpy.metrics`, bound to this portfolio.

    Any parameter named `returns`, `prices`, `y`, `weights`, or `cov_matrix` that
    the caller doesn't supply is filled in automatically from the parent
    `Portfolio` (its own aggregated returns/price index/weights/covariance
    matrix); `rf` and `periods_per_year` default to the portfolio's
    `risk_free_rate` and `frequency`. Auto-fill is skipped entirely for any call
    made with positional arguments, to avoid ambiguous double-binding - use
    keyword arguments to benefit from the defaults.
    """

    def __init__(self, portfolio: Portfolio) -> None:
        self._portfolio = portfolio

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(_metrics_module.__all__))

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        func = getattr(_metrics_module, name, None)
        if func is None or not callable(func):
            raise AttributeError(
                f"Portfolio.metrics has no {name!r}. See portpy.metrics.__all__ for the full list."
            )

        portfolio = self._portfolio
        params = inspect.signature(func).parameters

        @functools.wraps(func)
        def bound(*args, **kwargs):
            call_kwargs = dict(kwargs)
            if not args:
                if "returns" in params and "returns" not in call_kwargs:
                    call_kwargs["returns"] = (
                        portfolio.asset_returns() if name in _ASSET_LEVEL_RETURNS_FUNCS else portfolio.returns()
                    )
                if "y" in params and "y" not in call_kwargs:
                    call_kwargs["y"] = portfolio.returns()
                if "prices" in params and "prices" not in call_kwargs:
                    call_kwargs["prices"] = portfolio.price_index()
                if "weights" in params and "weights" not in call_kwargs:
                    call_kwargs["weights"] = portfolio.weights
                if "cov_matrix" in params and "cov_matrix" not in call_kwargs:
                    call_kwargs["cov_matrix"] = self.covariance_matrix()
                if "rf" in params and "rf" not in call_kwargs:
                    call_kwargs["rf"] = portfolio.risk_free_rate
                if "periods_per_year" in params and "periods_per_year" not in call_kwargs:
                    call_kwargs["periods_per_year"] = portfolio.frequency
            return func(*args, **call_kwargs)

        return bound


class Portfolio:
    """A collection of assets, their prices, and their weights - PortPy's main entry point.

    Args:
        data: Columns = asset symbols, index = a sorted, duplicate-free
            `DatetimeIndex`. See :func:`portpy.core.align_calendars` if you're
            combining assets with different trading calendars (e.g. crypto and
            equities), and :func:`portpy.core.convert_to_base_currency` if your
            assets aren't all in the same currency - both run *before*
            constructing the `Portfolio`, which never modifies your data itself.
        input_type: `"prices"` (default) or `"returns"` - what `data` contains.
        weights: Optional weights (Series/dict/array, any order); defaults to
            equal-weight (1/N) across all columns in `data`.
        name: A label used in chart titles and comparisons.
        frequency: Trading periods per year, used to annualize metrics
            (default 252 for daily equity data; use 365 for a crypto-only,
            7-day-a-week portfolio, 12 for monthly data, etc.).
        risk_free_rate: Annual risk-free rate, used as the default `rf` in
            `.metrics` calls.
        asset_classes: Optional `{symbol: AssetClass}` tags, purely for
            reporting/grouping - see `portpy.core.AssetClass`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_type: str = "prices",
        weights: pd.Series | np.ndarray | dict | None = None,
        name: str = "My Portfolio",
        frequency: int = TRADING_DAYS_PER_YEAR,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        asset_classes: dict[str, AssetClass] | None = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Portfolio expects a pandas DataFrame (columns = asset symbols), got {type(data).__name__}."
            )
        if data.shape[1] == 0:
            raise ValueError("data has no asset columns.")
        ensure_datetime_index(data, name="data")
        ensure_min_observations(data, 2, "data")
        if input_type not in _INPUT_TYPES:
            raise ValueError(f"input_type must be one of {_INPUT_TYPES}, got {input_type!r}.")

        if input_type == "prices":
            self._prices = data.astype(float).copy()
        else:
            self._prices = prices_from_returns(data.astype(float))

        self._asset_names = list(data.columns)
        self.name = name
        self.frequency = int(frequency)
        self.risk_free_rate = float(risk_free_rate)

        if asset_classes:
            unknown = set(asset_classes) - set(self._asset_names)
            if unknown:
                raise ValueError(f"asset_classes references unknown assets: {sorted(unknown)}")
        self.asset_classes: dict[str, AssetClass] = dict(asset_classes or {})

        self.set_weights(weights if weights is not None else equal_weights(self._asset_names))
        self.metrics = _MetricsNamespace(self)

    # -- Basic accessors -----------------------------------------------------

    @property
    def prices(self) -> pd.DataFrame:
        """Asset price DataFrame (reconstructed from returns at construction time if needed)."""
        return self._prices.copy()

    @property
    def asset_names(self) -> list[str]:
        return list(self._asset_names)

    @property
    def num_assets(self) -> int:
        return len(self._asset_names)

    @property
    def weights(self) -> pd.Series:
        return self._weights.copy()

    def set_weights(self, weights: pd.Series | np.ndarray | dict) -> None:
        """Validate, normalize (sum to 1), and apply new portfolio weights."""
        self._weights = normalize_weights(weights, names=self._asset_names)

    def set_risk_free_rate(self, rate: float) -> None:
        """Set the annual risk-free rate used as the default `rf` in `.metrics` calls.

        A common choice is the annualized return of a cash-like proxy (e.g. a
        T-Bill ETF) over the same period as your analysis.
        """
        self.risk_free_rate = float(rate)

    def asset_returns(self, log: bool = False) -> pd.DataFrame:
        """Per-asset simple (or log) returns - the multi-asset DataFrame, not the portfolio aggregate."""
        return log_returns(self._prices) if log else simple_returns(self._prices)

    def returns(self, period: int = 1, log: bool = False) -> pd.Series:
        """The portfolio's own return series: asset returns combined by current weights.

        Assumes weights are held constant each period (i.e. rebalanced back to
        target every period) - the standard simplifying assumption for a
        buy-and-hold-with-fixed-weights analysis. For turnover/rebalancing
        effects, see :mod:`portpy.strategies`.

        Args:
            period: Compound returns over non-overlapping blocks of this many
                periods (e.g. `period=21` on daily data for a rough monthly
                series) instead of period-over-period.
            log: Return log returns instead of simple returns.
        """
        asset_r = self.asset_returns(log=False)
        port_r = (asset_r * self._weights).sum(axis=1)
        port_r.name = self.name

        if period > 1:
            block = np.arange(len(port_r)) // period
            compounded = (1.0 + port_r).groupby(block).prod() - 1.0
            block_end_dates = port_r.index.to_series().groupby(block).last()
            compounded.index = block_end_dates.to_numpy()
            port_r = compounded

        if log:
            port_r = np.log1p(port_r)
        return port_r.rename(self.name)

    def price_index(self, base: float = 100.0) -> pd.Series:
        """Synthetic aggregated portfolio value, rebased to `base`, built by compounding `.returns()`.

        Anchored one period before the first return (see `prices_from_returns`), so
        a drawdown/CAGR computed on this index correctly reflects the very first
        period's move instead of silently treating it as the starting point.
        """
        r = self.returns()
        return prices_from_returns(r, base=base).rename(self.name)

    def __repr__(self) -> str:
        return (
            f"Portfolio(name={self.name!r}, assets={self.num_assets}, "
            f"n_obs={len(self._prices)}, frequency={self.frequency})"
        )
