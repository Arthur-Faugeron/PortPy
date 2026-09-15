"""
The Portfolio class: PortPy's single entry point for analysis.

Holds price data and weights, and exposes every function in
:mod:portpy.metrics as a bound method under .metrics`, automatically
supplying the portfolio's own returns/prices/weights/risk-free rate as
defaults wherever a metric function needs them. `.models` extends the same
auto-fill pattern to portpy.models' estimators/optimization/construction/
management namespaces.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from portpy import metrics as _metrics_module
from portpy import models as _models_module
from portpy.core.asset import AssetClass
from portpy.core.weights import equal_weights, normalize_weights
from portpy.metrics.returns import log_returns, prices_from_returns, simple_returns
from portpy.utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import ensure_datetime_index, ensure_min_observations

__all__ = ["Portfolio"]

_INPUT_TYPES = ("prices", "returns")


class _AutoFillNamespace:
    """
    <namespace>.<name>(...) - every function in a plain module (`self._module`),
    bound to a parent Portfolio.

    Shared by `.metrics` and every `.models.*` namespace: any parameter the
    caller doesn't supply, and whose name is one this namespace knows how to
    fill (see `_autofill_kwargs`), is auto-filled from the parent Portfolio's
    own state. Auto-fill is skipped entirely for any call made with positional
    arguments, to avoid ambiguous double-binding - use keyword arguments to
    benefit from the defaults.
    """

    _module: Any

    def __init__(self, portfolio: Portfolio) -> None:
        self._portfolio = portfolio

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._module.__all__))

    def _autofill_kwargs(self, name: str, params: Mapping[str, inspect.Parameter]) -> dict[str, Any]:
        """
        Subclasses return `{param_name: value}` for every parameter this
        namespace knows how to fill from the parent Portfolio - only entries
        whose key is actually in `params` end up applied.
        """
        return {}

    def _postprocess_kwargs(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Subclasses may repair fields *inside* an already-provided argument
        (e.g. completing a partially-specified constraint object) - unlike
        `_autofill_kwargs`, this runs even on positional-arg calls, since it
        isn't filling in a missing top-level parameter.
        """
        return call_kwargs

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        func = getattr(self._module, name, None)
        if func is None or not callable(func):
            raise AttributeError(
                f"{type(self).__name__} has no {name!r}. See {self._module.__name__}.__all__ for the full list."
            )
        params = inspect.signature(func).parameters

        @functools.wraps(func)
        def bound(*args, **kwargs):
            call_kwargs = dict(kwargs)
            if not args:
                for key, value in self._autofill_kwargs(name, params).items():
                    if key in params and key not in call_kwargs:
                        call_kwargs[key] = value
            call_kwargs = self._postprocess_kwargs(call_kwargs)
            return func(*args, **call_kwargs)

        return bound


# Functions where the "returns" parameter means the full multi-asset DataFrame
# (not the portfolio's own aggregated return series).
_ASSET_LEVEL_RETURNS_FUNCS = frozenset({"covariance_matrix", "correlation_matrix"})


class _MetricsNamespace(_AutoFillNamespace):
    """
    portfolio.metrics.<name>(...) - every function in portpy.metrics, bound to this portfolio.

    Any parameter named returns, prices, y, weights, or cov_matrix that
    the caller doesn't supply is filled in automatically from the parent
    Portfolio (its own aggregated returns/price index/weights/covariance
    matrix); rf and periods_per_year default to the portfolio's
    risk_free_rate and frequency.
    """

    _module = _metrics_module

    def _autofill_kwargs(self, name: str, params: Mapping[str, inspect.Parameter]) -> dict[str, Any]:
        portfolio = self._portfolio
        kwargs: dict[str, Any] = {}
        if "returns" in params:
            kwargs["returns"] = portfolio.asset_returns() if name in _ASSET_LEVEL_RETURNS_FUNCS else portfolio.returns()
        if "y" in params:
            kwargs["y"] = portfolio.returns()
        if "prices" in params:
            kwargs["prices"] = portfolio.price_index()
        if "weights" in params:
            kwargs["weights"] = portfolio.weights
        if "cov_matrix" in params:
            kwargs["cov_matrix"] = self.covariance_matrix()
        if "rf" in params:
            kwargs["rf"] = portfolio.risk_free_rate
        if "periods_per_year" in params:
            kwargs["periods_per_year"] = portfolio.frequency
        return kwargs


def _fill_turnover_cap_current_weights(constraints: Any, portfolio: Portfolio) -> Any:
    """
    Replace any `TurnoverCap` in `constraints` that has `current_weights=None`
    with a copy carrying `portfolio`'s own current weights.

    `TurnoverCap` is a frozen dataclass, so a bare `TurnoverCap(max_turnover=...)`
    built via a plain top-level import (rather than through
    `portfolio.models.construction.TurnoverCap(...)`) has no way to pick up a
    default afterward on its own - every `.models` entry point that accepts
    `constraints=` runs its list through this first, so which import path
    built the object doesn't matter.
    """
    if not constraints:
        return constraints
    from dataclasses import replace

    from portpy.models.base import TurnoverCap

    return [
        replace(c, current_weights=portfolio.weights) if isinstance(c, TurnoverCap) and c.current_weights is None else c
        for c in constraints
    ]


class _ModelsAutoFillNamespace(_AutoFillNamespace):
    """
    Shared auto-fill logic for every `portpy.models.*` sub-namespace.

    `y` means the full multi-asset return DataFrame for functions listed in
    `_asset_level_y_funcs` (subclass-configurable - e.g. estimators.expected_returns
    needs the whole panel), and the portfolio's own aggregated return series otherwise
    (e.g. estimators.capm's `y` is the thing being regressed, one series).
    """

    _asset_level_y_funcs: frozenset[str] = frozenset()

    def _autofill_kwargs(self, name: str, params: Mapping[str, inspect.Parameter]) -> dict[str, Any]:
        portfolio = self._portfolio
        kwargs: dict[str, Any] = {}
        if "y" in params:
            kwargs["y"] = portfolio.asset_returns() if name in self._asset_level_y_funcs else portfolio.returns()
        if "expected_returns" in params:
            kwargs["expected_returns"] = portfolio.models.estimators.expected_returns()
        if "cov_matrix" in params:
            kwargs["cov_matrix"] = portfolio.models.estimators.covariance()
        if "market_weights" in params:
            kwargs["market_weights"] = portfolio.weights
        if "current_weights" in params:
            kwargs["current_weights"] = portfolio.weights
        if "rf" in params:
            kwargs["rf"] = portfolio.risk_free_rate
        if "periods_per_year" in params:
            kwargs["periods_per_year"] = portfolio.frequency
        return kwargs

    def _postprocess_kwargs(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "constraints" in call_kwargs:
            call_kwargs["constraints"] = _fill_turnover_cap_current_weights(call_kwargs["constraints"], self._portfolio)
        return call_kwargs


class _EstimatorsNamespace(_ModelsAutoFillNamespace):
    """portfolio.models.estimators.<name>(...) - see portpy.models.estimators."""

    _module = _models_module.estimators
    _asset_level_y_funcs = frozenset({"expected_returns", "covariance"})


class _OptimizationNamespace(_ModelsAutoFillNamespace):
    """portfolio.models.optimization.<name>(...) - see portpy.models.optimization."""

    _module = _models_module.optimization


class _ConstructionNamespace(_ModelsAutoFillNamespace):
    """portfolio.models.construction.<name>(...) - see portpy.models.construction."""

    _module = _models_module.construction
    _asset_level_y_funcs = frozenset({"build"})


class _ManagementNamespace(_ModelsAutoFillNamespace):
    """portfolio.models.management.<name>(...) - see portpy.models.management."""

    _module = _models_module.management

    def compare(self, other: Any) -> Any:
        """
        Compare this portfolio against `other` (a Portfolio, ModelResult, or weight Series).

        A hand-written override (rather than the generic auto-fill dispatch)
        so the common case is a single positional argument - see
        `portpy.models.management.compare` for the underlying pure function
        and what its result carries.
        """
        return _models_module.management.compare(self._portfolio, other)


class _ModelsNamespace:
    """
    portfolio.models - a namespace of namespaces over portpy.models, each
    reusing `_ModelsAutoFillNamespace` so `.estimators`/`.optimization`/
    `.construction`/`.management` all auto-fill from this portfolio's own data.

    `.optimize`/`.build`/`.efficient_frontier` are also exposed directly here
    (in addition to `.construction.optimize`/`.construction.build`/
    `.construction.efficient_frontier`, the same functions) since they're the
    two or three calls most users reach for first.
    """

    def __init__(self, portfolio: Portfolio) -> None:
        self._portfolio = portfolio
        self.estimators = _EstimatorsNamespace(portfolio)
        self.optimization = _OptimizationNamespace(portfolio)
        self.construction = _ConstructionNamespace(portfolio)
        self.management = _ManagementNamespace(portfolio)

    def optimize(
        self,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        method: str = "mean_variance",
        constraints: list[Any] | None = None,
        **kwargs: Any,
    ):
        """See `portpy.models.construction.optimize` - expected_returns/cov_matrix default to this portfolio's own estimates."""
        if expected_returns is None:
            expected_returns = self.estimators.expected_returns()
        if cov_matrix is None:
            cov_matrix = self.estimators.covariance()
        constraints = _fill_turnover_cap_current_weights(constraints, self._portfolio)
        return _models_module.construction.optimize(
            expected_returns=expected_returns, cov_matrix=cov_matrix, method=method, constraints=constraints, **kwargs
        )

    def build(
        self,
        y: pd.DataFrame | None = None,
        method: str = "mean_variance",
        constraints: list[Any] | None = None,
        expected_returns_kwargs: dict[str, Any] | None = None,
        covariance_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """See `portpy.models.construction.build` - y defaults to this portfolio's own asset returns."""
        if y is None:
            y = self._portfolio.asset_returns()
        constraints = _fill_turnover_cap_current_weights(constraints, self._portfolio)
        return _models_module.construction.build(
            y=y,
            method=method,
            constraints=constraints,
            expected_returns_kwargs=expected_returns_kwargs,
            covariance_kwargs=covariance_kwargs,
            **kwargs,
        )

    def efficient_frontier(
        self,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        n_points: int = 50,
        constraints: list[Any] | None = None,
        min_return: float | None = None,
        max_return: float | None = None,
    ):
        """See `portpy.models.construction.efficient_frontier` - expected_returns/cov_matrix default to this portfolio's own estimates."""
        if expected_returns is None:
            expected_returns = self.estimators.expected_returns()
        if cov_matrix is None:
            cov_matrix = self.estimators.covariance()
        constraints = _fill_turnover_cap_current_weights(constraints, self._portfolio)
        return _models_module.construction.efficient_frontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            n_points=n_points,
            constraints=constraints,
            min_return=min_return,
            max_return=max_return,
        )

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__()) | {"estimators", "optimization", "construction", "management", "optimize", "build", "efficient_frontier"}
        )


class Portfolio:
    """
    A collection of assets, their prices, and their weights, PortPy's main entry point.

    Args:
        data: Columns = asset symbols, index = a sorted, duplicate-free
            DatetimeIndex. See :func:portpy.core.align_calendars if you're
            combining assets with different trading calendars (e.g. crypto and
            equities), and :func:portpy.core.convert_to_base_currency if your
            assets aren't all in the same currency - both run *before*
            constructing the Portfolio, which never modifies your data itself.
        input_type: "prices" (default) or "returns" - what data contains.
        weights: Optional weights (Series/dict/array, any order); defaults to
            equal-weight (1/N) across all columns in data.
        name: A label used in chart titles and comparisons.
        frequency: Trading periods per year, used to annualize metrics
            (default 252 for daily equity data; use 365 for a crypto-only,
            7-day-a-week portfolio, 12 for monthly data, etc.).
        risk_free_rate: Annual risk-free rate, used as the default rf in
            .metrics calls.
        asset_classes: Optional {symbol: AssetClass} tags, purely for
            reporting/grouping - see portpy.core.AssetClass.
        base_currency: Optional 3-letter currency code (e.g. "USD"), purely
            informational/for reporting - like asset_classes, it's never used
            to convert anything. `data` must already be in a single currency
            by the time it reaches Portfolio; run
            portpy.core.convert_to_base_currency yourself first if it isn't,
            then pass the currency you converted to here as a label.
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
        base_currency: str | None = None,
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
        self.base_currency: str | None = base_currency

        self.set_weights(weights if weights is not None else equal_weights(self._asset_names))
        self.metrics = _MetricsNamespace(self)
        self.models = _ModelsNamespace(self)

    # -- Basic accessors -----------------------------------------------------

    @property
    def prices(self) -> pd.DataFrame:
        """
        Asset price DataFrame (reconstructed from returns at construction time if needed).
        """
        return self._prices.copy()

    @property
    def asset_names(self) -> list[str]:
        """
        The portfolio's asset symbols, in column order.
        """
        return list(self._asset_names)

    @property
    def num_assets(self) -> int:
        """
        The number of assets in the portfolio.
        """
        return len(self._asset_names)

    @property
    def weights(self) -> pd.Series:
        """
        The current per-asset weights, indexed by asset name.
        """
        return self._weights.copy()

    def set_weights(self, weights: pd.Series | np.ndarray | dict) -> None:
        """
        Validate, normalize (sum to 1), and apply new portfolio weights.

        Args:
            weights: A Series/dict keyed by asset name, or a plain array aligned
                with `asset_names`.
        """
        self._weights = normalize_weights(weights, names=self._asset_names)

    def set_risk_free_rate(self, rate: float) -> None:
        """
        Set the annual risk-free rate used as the default rf in .metrics calls.

        Args:
            rate: The new annual risk-free rate (e.g. 0.04 for 4%/year).
        """
        self.risk_free_rate = float(rate)

    def asset_returns(self, log: bool = False) -> pd.DataFrame:
        """
        Per-asset simple (or log) returns - the multi-asset DataFrame, not the portfolio aggregate.

        Args:
            log: If True, return log returns instead of simple returns.

        Returns:
            A DataFrame of per-period returns, one column per asset.
        """
        return log_returns(self._prices) if log else simple_returns(self._prices)

    def returns(self, period: int = 1, log: bool = False) -> pd.Series:
        """
        The portfolio's own return series: asset returns combined by current weights.

        Assumes weights are held constant each period (rebalanced back to target
        every period). For turnover/rebalancing effects, see :mod:portpy.models.management
        and (eventually) :mod:portpy.strategies.

        Args:
            period: Compound returns over non-overlapping blocks of this many
                periods (e.g. period=21 on daily data for a rough monthly
                series) instead of period-over-period.
            log: If True, return log returns instead of simple returns.

        Returns:
            The portfolio's return series, named after the portfolio.
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
        """
        Synthetic aggregated portfolio value, rebased to base, built by compounding .returns().

        Args:
            base: The starting value of the index.

        Returns:
            A synthetic price series, named after the portfolio, anchored one
            period before the first return.
        """
        r = self.returns()
        return prices_from_returns(r, base=base).rename(self.name)

    def __repr__(self) -> str:
        return (
            f"Portfolio(name={self.name!r}, assets={self.num_assets}, "
            f"n_obs={len(self._prices)}, frequency={self.frequency})"
        )
