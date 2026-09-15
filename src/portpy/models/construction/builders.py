"""
The recipe layer users hit most often: dispatch to a named optimizer
(`optimize`), or estimate expected_returns/cov_matrix from raw returns first
and then dispatch (`build`) - both wrapped into a ModelResult. `efficient_frontier`
lives here too (not in `optimize`'s dispatch table - see its own docstring)
since it needs the same ModelResult-wrapping treatment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from inspect import signature
from typing import Any

import pandas as pd

from portpy.models import optimization as _opt
from portpy.models.base import ModelResult
from portpy.models.estimators.covariance import covariance as _estimate_covariance
from portpy.models.estimators.expected_returns import expected_returns as _estimate_expected_returns

__all__ = ["optimize", "build", "efficient_frontier"]

_SOLVERS: dict[str, Callable[..., pd.Series]] = {
    "mean_variance": _opt.mean_variance,
    "max_sharpe": _opt.max_sharpe,
    "min_variance": _opt.min_variance,
    "target_return": _opt.target_return,
    "target_volatility": _opt.target_volatility,
    "black_litterman": _opt.black_litterman,
    "hierarchical_risk_parity": _opt.hierarchical_risk_parity,
    "risk_parity": _opt.risk_parity,
    "risk_budgeting": _opt.risk_budgeting,
    "maximum_diversification": _opt.maximum_diversification,
}

# Methods whose solver function takes an `expected_returns` argument - everything
# else (min_variance, HRP, risk_parity/budgeting, maximum_diversification, and
# black_litterman with its own market_weights/views inputs) is return-agnostic.
_NEEDS_EXPECTED_RETURNS = {"mean_variance", "max_sharpe", "target_return", "target_volatility"}


def optimize(
    expected_returns: pd.Series | None = None,
    cov_matrix: pd.DataFrame | None = None,
    method: str = "mean_variance",
    constraints: Sequence[Any] | None = None,
    **kwargs: Any,
) -> ModelResult:
    """
    Dispatch to a named solver in `portpy.models.optimization`, wrapping its
    result into a `ModelResult` with solver diagnostics attached.

    Args:
        expected_returns: Annualized expected return per asset. Required by
            "mean_variance", "max_sharpe", "target_return", and "target_volatility";
            ignored (and not required) by every other method.
        cov_matrix: Annualized asset covariance matrix. Required by every method.
        method: One of "mean_variance", "max_sharpe", "min_variance",
            "target_return", "target_volatility", "black_litterman",
            "hierarchical_risk_parity", "risk_parity", "risk_budgeting",
            "maximum_diversification".
        constraints: Forwarded to the underlying solver - see `portpy.models.base`.
        **kwargs: Method-specific keyword arguments, e.g. `risk_aversion=` for
            "mean_variance"/"black_litterman", `target=` for "target_return"/
            "target_volatility", `rf=` for "max_sharpe", `market_weights=`/`views=`/
            `tau=`/`omega=` for "black_litterman", `budget=` for "risk_budgeting",
            `linkage_method=` for "hierarchical_risk_parity".

    Returns:
        A `ModelResult` named after `method`. `.diagnostics` always has
        `"method"`, plus (for every method except `hierarchical_risk_parity`,
        which isn't numerically solved) `"success"`, `"objective_value"`,
        `"iterations"`, `"message"`, and `"n_restarts"` from the solve.

    Raises:
        ValueError: If `method` is unrecognized, `cov_matrix` is missing, or
            `expected_returns` is missing for a method that requires it.
    """
    if method not in _SOLVERS:
        raise ValueError(f"method must be one of {sorted(_SOLVERS)}, got {method!r}.")
    if cov_matrix is None:
        raise ValueError("cov_matrix is required.")
    if method in _NEEDS_EXPECTED_RETURNS and expected_returns is None:
        raise ValueError(f"method={method!r} requires expected_returns.")

    solver = _SOLVERS[method]
    params = signature(solver).parameters

    call_kwargs: dict[str, Any] = dict(kwargs)
    if "expected_returns" in params:
        call_kwargs.setdefault("expected_returns", expected_returns)
    if "cov_matrix" in params:
        call_kwargs.setdefault("cov_matrix", cov_matrix)
    if "constraints" in params:
        call_kwargs.setdefault("constraints", constraints)

    weights = solver(**call_kwargs)

    diagnostics: dict[str, Any] = {"method": method}
    diagnostics.update(getattr(weights, "attrs", {}).get("solve_diagnostics", {}))

    return ModelResult(name=method, weights=weights, diagnostics=diagnostics, meta={"constraints": list(constraints or [])})


def build(
    y: pd.DataFrame | None = None,
    method: str = "mean_variance",
    constraints: Sequence[Any] | None = None,
    expected_returns_kwargs: dict[str, Any] | None = None,
    covariance_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelResult:
    """
    The recipe most users actually want: estimate `expected_returns`/`cov_matrix`
    from raw returns via `portpy.models.estimators`, then `optimize()`.

    Args:
        y: Per-asset periodic returns (the full multi-asset DataFrame). Required
            explicitly here; auto-filled from the portfolio when called via
            `portfolio.models.build(...)`.
        method: See `optimize`.
        constraints: See `optimize`.
        expected_returns_kwargs: Extra keyword arguments for
            `estimators.expected_returns` (e.g. `{"method": "james_stein"}`).
            Skipped entirely for methods that don't need expected_returns
            (see `optimize`), so this can be left as `{}`/`None` for those.
        covariance_kwargs: Extra keyword arguments for `estimators.covariance`
            (e.g. `{"method": "ledoit_wolf"}`).
        **kwargs: Forwarded to `optimize` (and from there to the chosen solver).

    Returns:
        A `ModelResult` exactly as `optimize` returns, with `.meta["built_from"]`
        additionally recording the estimator kwargs and data shape used.

    Raises:
        ValueError: If `y` is missing.
    """
    if y is None:
        raise ValueError(
            "y (the multi-asset return DataFrame) is required - pass it explicitly, "
            "or call via portfolio.models.build(...) so it's auto-filled."
        )

    er_kwargs = dict(expected_returns_kwargs or {})
    cov_kwargs = dict(covariance_kwargs or {})

    mu = _estimate_expected_returns(y, **er_kwargs) if method in _NEEDS_EXPECTED_RETURNS else None
    cov = _estimate_covariance(y, **cov_kwargs)

    result = optimize(expected_returns=mu, cov_matrix=cov, method=method, constraints=constraints, **kwargs)
    result.meta["built_from"] = {
        "expected_returns_kwargs": er_kwargs,
        "covariance_kwargs": cov_kwargs,
        "n_assets": cov.shape[0],
        "n_observations": len(y),
    }
    return result


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    constraints: Sequence[Any] | None = None,
    min_return: float | None = None,
    max_return: float | None = None,
) -> ModelResult:
    """
    Trace the efficient frontier and wrap it into a `ModelResult` - kept as its
    own call rather than folded into `optimize()`/`build()`, since it returns a
    curve of portfolios, not one.

    Args:
        expected_returns: Annualized expected return per asset.
        cov_matrix: Annualized asset covariance matrix.
        n_points: Number of frontier points to compute.
        constraints: Forwarded to every point's solve - see `portpy.models.base`.
        min_return: See `portpy.models.optimization.efficient_frontier`.
        max_return: See `portpy.models.optimization.efficient_frontier`.

    Returns:
        A `ModelResult` named "efficient_frontier". `.weights` is the max-Sharpe
        point on the curve; the full curve is `.diagnostics["frontier"]` (also
        what `.summary()` returns for this result).

    Raises:
        RuntimeError: If not even one grid point was solvable.
    """
    frontier = _opt.efficient_frontier(
        expected_returns, cov_matrix, n_points=n_points, constraints=constraints, min_return=min_return, max_return=max_return
    )
    if frontier.empty:
        raise RuntimeError("efficient_frontier: no grid point was solvable - check that constraints leave a feasible region.")

    asset_names = list(cov_matrix.columns)
    best_idx = frontier["sharpe"].idxmax()
    max_sharpe_weights = pd.Series(
        frontier.loc[best_idx, asset_names].to_numpy(dtype=float), index=asset_names, name="weight"
    )
    diagnostics = {
        "frontier": frontier,
        "n_points_requested": n_points,
        "n_points_solved": len(frontier),
        "max_sharpe_index": int(best_idx),
    }
    return ModelResult(
        name="efficient_frontier", weights=max_sharpe_weights, diagnostics=diagnostics, meta={"constraints": list(constraints or [])}
    )
