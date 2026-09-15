"""
Shared plumbing for :mod:`portpy.models`: the result type every builder/optimizer/
rebalancer returns, the constraint objects solvers understand, and the numerical
solve routine (multi-start SLSQP) every function in :mod:`portpy.models.optimization`
is built on.

PortPy's `models` extra deliberately stays scipy-only (no cvxpy/cvxopt) - every
solver here is a general nonlinear program handed to `scipy.optimize.minimize`
(SLSQP), not a specialized convex QP/SOCP solve. That trades a little numerical
polish on the classic convex cases (mean-variance, min-variance) for one solver
path that also handles the genuinely non-convex ones (max_sharpe's ratio
objective, maximum_diversification) and arbitrary constraint combinations
without a second code path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize

__all__ = [
    "ModelResult",
    "WeightBounds",
    "GroupCap",
    "TurnoverCap",
    "NetExposure",
    "GrossExposure",
    "solve_weights",
]

Constraint = "WeightBounds | GroupCap | TurnoverCap | NetExposure | GrossExposure"


@dataclass(frozen=True)
class WeightBounds:
    """
    Per-asset weight bounds. `low`/`high` apply to every asset unless overridden
    in `per_asset`.

    Args:
        low: Minimum weight for any asset not listed in `per_asset` (default
            0.0, i.e. long-only). Pass a negative value to allow shorting.
        high: Maximum weight for any asset not listed in `per_asset`.
        per_asset: Optional `{asset_name: (low, high)}` overrides for specific assets.
    """

    low: float = 0.0
    high: float = 1.0
    per_asset: dict[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class GroupCap:
    """
    Caps (and optionally floors) total exposure to named groups of assets -
    e.g. sectors or `core.asset.AssetClass` buckets.

    Args:
        groups: `{group_name: [asset_names]}`. Assets not listed in any group
            are unconstrained by this object.
        max_weight: Maximum summed weight per group - either one number applied
            to every group, or a `{group_name: cap}` dict.
        min_weight: Optional minimum summed weight per group, same shape as `max_weight`.
    """

    groups: dict[str, list[str]]
    max_weight: float | dict[str, float]
    min_weight: float | dict[str, float] | None = None


@dataclass(frozen=True)
class TurnoverCap:
    """
    Caps how far the solved weights may move away from a starting weight vector.

    Args:
        max_turnover: Maximum allowed `sum(abs(w - current_weights))` - note this
            is the raw sum, NOT divided by 2. That makes it twice
            `metrics.costs.turnover_from_weights`'s "one-way turnover" convention
            (which halves the same sum, so that selling 10% of A to buy 10% of B
            reads as "10% of the book traded" rather than "20%") - the two
            conventions are both individually well-defined but not
            interchangeable: a `TurnoverCap(max_turnover=0.10)` allows exactly
            the same trades as a one-way-turnover budget of 5%, not 10%.
        current_weights: Starting weights. When left `None`, `.models` callers
            auto-fill it from the portfolio's current weights; calling the
            optimization functions directly requires setting it explicitly.
    """

    max_turnover: float
    current_weights: pd.Series | None = None


@dataclass(frozen=True)
class NetExposure:
    """
    `sum(w) == target` - the usual full-investment constraint (target=1.0).

    Applied automatically with `target=1.0` when a constraint list has neither
    this nor `GrossExposure`, so every solver is fully-invested by default.
    """

    target: float = 1.0


@dataclass(frozen=True)
class GrossExposure:
    """
    `sum(abs(w)) == target` - caps total (long + short) exposure, the natural
    constraint for a long/short book (e.g. a 130/30 fund uses target=1.6).
    """

    target: float = 1.0


@dataclass
class ModelResult:
    """
    The structured result of any `.models` builder/optimizer/rebalancer call -
    the model-layer counterpart to :class:`~portpy.explain.MetricResult`.

    Behaves like a plain container (not a float): the number you usually want
    is a whole weight vector, not a scalar. Explains itself the same way
    `MetricResult` does, via the `_portpy_explain_name`/`_portpy_explain_value`
    hook in `portpy.explain.explain` - `_portpy_explain_value` is set to the
    whole result (not a bare float), since a model's `interpret` reads
    weights/diagnostics, not a single number.

    Attributes:
        name: Registry key this result explains itself with (e.g. "mean_variance").
        weights: Solved (or otherwise resulting) allocation, indexed by asset.
            For `efficient_frontier`, this is the max-Sharpe point on the curve -
            see `diagnostics["frontier"]` for the full curve.
        diagnostics: Solver/estimation internals - status, objective value,
            iterations, inputs used. Shape varies by model; see each model's
            registered Explanation for what to expect.
        meta: Free-form extra context (e.g. the method/constraints used to build this).
    """

    name: str
    weights: pd.Series
    diagnostics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._portpy_explain_name = self.name
        self._portpy_explain_value = self

    def explain(self, print_it: bool = True) -> str:
        """
        Print (by default) and return the full explanation for this result's model type.

        Args:
            print_it: If True (default), also print() the rendered text.

        Returns:
            The rendered explanation text.
        """
        from portpy.explain import explain as _explain

        return _explain(self, print_it=print_it)

    def summary(self) -> pd.DataFrame:
        """
        A tabular view: weights sorted by absolute size (or, for `efficient_frontier`
        results, the full return/volatility/weights curve).

        Returns:
            A DataFrame carrying `.attrs["portpy_explanation"] = self.name`, so
            `portpy.explain()` also works on the returned table.
        """
        if "frontier" in self.diagnostics:
            frontier = self.diagnostics["frontier"].copy()
            frontier.attrs["portpy_explanation"] = self.name
            return frontier

        df = self.weights.rename("weight").to_frame()
        df = df.reindex(df["weight"].abs().sort_values(ascending=False).index)
        df.attrs["portpy_explanation"] = self.name
        return df

    def plot(self, **kwargs: Any):
        """
        Render this result via `portpy.visualization` (not yet implemented).

        Raises:
            NotImplementedError: Always, until Stage 2 (`.visualization`) ships.
        """
        raise NotImplementedError(
            "ModelResult.plot() delegates to portpy.visualization, which hasn't "
            "shipped yet (it's next, right after .models). Use .summary() or "
            ".explain() in the meantime."
        )

    def compare(self, other: ModelResult | pd.Series, **kwargs: Any) -> ModelResult:
        """
        Compare this result's weights against another result/portfolio.

        Args:
            other: Another `ModelResult`, a `Portfolio`, or a raw weight `Series`.
            **kwargs: Forwarded to `portpy.models.management.compare`.

        Returns:
            A `ModelResult` named "compare" - see `models.management.compare`.
        """
        from portpy.models.management.rebalancing import compare as _compare

        return _compare(self, other, **kwargs)

    def __repr__(self) -> str:
        n_active = int((self.weights.abs() > 1e-8).sum())
        status = self.diagnostics.get("success")
        status_str = "" if status is None else f", converged={status}"
        return f"ModelResult(name={self.name!r}, n_assets={len(self.weights)}, n_active={n_active}{status_str})"


def _ensure_exposure_constraint(constraints: Sequence[Any]) -> list[Any]:
    """Append a full-investment NetExposure(1.0) if the caller supplied neither exposure kind."""
    out = list(constraints)
    if not any(isinstance(c, (NetExposure, GrossExposure)) for c in out):
        out.append(NetExposure(1.0))
    return out


def resolve_bounds(constraints: Sequence[Any], asset_names: Sequence[str]) -> list[tuple[float, float]]:
    """
    Translate any `WeightBounds` in `constraints` into a per-asset `(low, high)` list.

    Args:
        constraints: Constraint objects (only `WeightBounds` entries matter here).
        asset_names: Asset order to produce bounds in.

    Returns:
        One `(low, high)` tuple per asset, defaulting to `(0.0, 1.0)` (long-only)
        when no `WeightBounds` is present.

    Raises:
        ValueError: If `WeightBounds.per_asset` references an asset not in `asset_names`.
    """
    n = len(asset_names)
    bounds = [(0.0, 1.0)] * n
    index = {name: i for i, name in enumerate(asset_names)}
    for c in constraints:
        if isinstance(c, WeightBounds):
            bounds = [(c.low, c.high)] * n
            if c.per_asset:
                for asset, (lo, hi) in c.per_asset.items():
                    if asset not in index:
                        raise ValueError(f"WeightBounds.per_asset references unknown asset {asset!r}.")
                    bounds[index[asset]] = (lo, hi)
    return bounds


def resolve_scipy_constraints(constraints: Sequence[Any], asset_names: Sequence[str]) -> list[dict[str, Any]]:
    """
    Translate constraint objects into `scipy.optimize.minimize`'s constraint-dict format.

    Args:
        constraints: Constraint objects to translate (`WeightBounds` is handled
            separately, via `resolve_bounds`, and is ignored here).
        asset_names: Asset order matching the decision vector `w`.

    Returns:
        A list of `{"type": "eq"|"ineq", "fun": callable}` dicts, SLSQP's expected shape.

    Raises:
        ValueError: If a `TurnoverCap` has no `current_weights`, or a `GroupCap`
            references an unknown asset.
    """
    index = {name: i for i, name in enumerate(asset_names)}
    cons: list[dict[str, Any]] = []

    for c in constraints:
        if isinstance(c, NetExposure):
            cons.append({"type": "eq", "fun": lambda w, t=c.target: float(np.sum(w) - t)})
        elif isinstance(c, GrossExposure):
            cons.append({"type": "eq", "fun": lambda w, t=c.target: float(np.sum(np.abs(w)) - t)})
        elif isinstance(c, GroupCap):
            for group_name, members in c.groups.items():
                unknown = [m for m in members if m not in index]
                if unknown:
                    raise ValueError(f"GroupCap group {group_name!r} references unknown assets: {unknown}")
                member_idx = np.array([index[m] for m in members], dtype=int)
                max_w = c.max_weight if isinstance(c.max_weight, (int, float)) else c.max_weight[group_name]
                cons.append({"type": "ineq", "fun": lambda w, mi=member_idx, mx=max_w: float(mx - np.sum(w[mi]))})
                if c.min_weight is not None:
                    min_w = c.min_weight if isinstance(c.min_weight, (int, float)) else c.min_weight[group_name]
                    cons.append({"type": "ineq", "fun": lambda w, mi=member_idx, mn=min_w: float(np.sum(w[mi]) - mn)})
        elif isinstance(c, TurnoverCap):
            if c.current_weights is None:
                raise ValueError(
                    "TurnoverCap.current_weights is required - pass it explicitly, or use "
                    "portfolio.models.* so it's auto-filled from the portfolio's own weights."
                )
            cw = c.current_weights.reindex(asset_names).fillna(0.0).to_numpy()
            cons.append(
                {"type": "ineq", "fun": lambda w, cw=cw, mx=c.max_turnover: float(mx - np.sum(np.abs(w - cw)))}
            )
        elif isinstance(c, WeightBounds):
            continue
        else:
            raise TypeError(f"Unsupported constraint type: {type(c).__name__}")
    return cons


def solve_weights(
    objective: Callable[[np.ndarray], float],
    asset_names: Sequence[str],
    constraints: Sequence[Any] | None = None,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
    n_restarts: int = 4,
    seed: int = 7,
    maxiter: int = 1000,
    extra_scipy_constraints: Sequence[dict[str, Any]] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Minimize `objective(w)` over portfolio weights via multi-start SLSQP.

    A full-investment constraint (`NetExposure(1.0)`) is applied automatically
    if `constraints` contains neither `NetExposure` nor `GrossExposure`. SLSQP's
    result is sensitive to its starting point, so this tries the equal-weight
    portfolio plus `n_restarts - 1` random feasible-ish (Dirichlet, clipped to
    bounds) starts and keeps the best converged solution (or, if none converge,
    the lowest-objective attempt, flagged via `diagnostics["success"] = False`).

    Args:
        objective: Function of a weight array (in `asset_names` order) to minimize.
        asset_names: Asset order for the decision vector and the returned Series.
        constraints: Constraint objects (see `WeightBounds`, `GroupCap`,
            `TurnoverCap`, `NetExposure`, `GrossExposure`).
        jac: Optional analytic gradient of `objective`, same signature.
        n_restarts: Number of SLSQP starting points to try.
        seed: RNG seed for the random restarts (deterministic across calls).
        maxiter: Max iterations per SLSQP attempt.
        extra_scipy_constraints: Additional raw `scipy.optimize.minimize`-style
            constraint dicts, for a one-off constraint specific to a single
            solver (e.g. `target_return`'s "hit this exact return") that isn't
            one of the general, reusable `Constraint` types above.

    Returns:
        `(weights, diagnostics)` - `weights` is a Series indexed by `asset_names`;
        `diagnostics` carries `success`, `objective_value`, `iterations`,
        `message`, and `n_restarts`.
    """
    asset_names = list(asset_names)
    n = len(asset_names)
    all_constraints = _ensure_exposure_constraint(constraints or [])
    bounds = resolve_bounds(all_constraints, asset_names)
    scipy_cons = resolve_scipy_constraints(all_constraints, asset_names) + list(extra_scipy_constraints or [])

    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)

    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1.0 / n)]
    for _ in range(max(0, n_restarts - 1)):
        raw = rng.dirichlet(np.ones(n))
        starts.append(np.clip(lows + raw * (highs - lows), lows, highs))

    results = []
    for x0 in starts:
        res = sp_optimize.minimize(
            objective,
            x0,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_cons,
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        results.append(res)

    converged = [r for r in results if r.success]
    pool = converged if converged else results
    best = min(pool, key=lambda r: float(r.fun))

    weights = pd.Series(best.x, index=asset_names, name="weight")
    diagnostics = {
        "success": bool(best.success),
        "objective_value": float(best.fun),
        "iterations": int(best.nit),
        "message": str(best.message),
        "n_restarts": n_restarts,
    }
    return weights, diagnostics
