"""
Pure portfolio optimization solvers: (expected_returns, cov_matrix, constraints) -> weights.

Every function here is importable directly for functional/advanced use - none
of them depend on Portfolio or portpy.models.estimators, you bring your own
mu/Sigma. All of them (except hierarchical_risk_parity, which needs no
numerical solver) go through models.base.solve_weights - multi-start SLSQP;
see that module's docstring for why PortPy stays scipy-only here instead of
adding a convex-QP dependency like cvxpy.

Nothing here checks that `expected_returns` and `cov_matrix` share a time
scale. Every docstring below says "annualized" because that's what
`portpy.models.estimators` produces by default and what `rf`/`risk_aversion`
defaults assume, but the solvers themselves are scale-agnostic: pass a
periodic mu next to an annualized Sigma (or vice versa) and you get a
solvable, plausible-looking, and financially meaningless result - no
exception, no warning. Get both inputs from `portpy.models.estimators` (or
otherwise put both on the same footing yourself) before optimizing.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from portpy.explain import Explanation, register
from portpy.metrics.covariance import diversification_ratio as _diversification_ratio
from portpy.models.base import WeightBounds, solve_weights
from portpy.utils.validation import ensure_min_observations, safe_divide

__all__ = [
    "mean_variance",
    "min_variance",
    "max_sharpe",
    "target_return",
    "target_volatility",
    "efficient_frontier",
    "black_litterman",
    "hierarchical_risk_parity",
    "risk_parity",
    "risk_budgeting",
    "maximum_diversification",
]


def _asset_names(cov_matrix: pd.DataFrame) -> list[str]:
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError(f"cov_matrix must be a pandas DataFrame (asset names come from its columns), got {type(cov_matrix).__name__}.")
    ensure_min_observations(cov_matrix, 1, "cov_matrix")
    return list(cov_matrix.columns)


def _cov_array(cov_matrix: pd.DataFrame, names: list[str]) -> np.ndarray:
    return cov_matrix.reindex(index=names, columns=names).to_numpy()


def _mu_array(expected_returns: pd.Series, names: list[str]) -> np.ndarray:
    aligned = expected_returns.reindex(names)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        raise ValueError(f"expected_returns is missing values for assets in cov_matrix: {missing}")
    return aligned.to_numpy()


def mean_variance(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: Sequence[Any] | None = None,
    risk_aversion: float = 1.0,
) -> pd.Series:
    """
    Classic Markowitz utility maximization: `maximize w'mu - 0.5*risk_aversion*w'Sigma*w`.

    Args:
        expected_returns: Annualized expected return per asset.
        cov_matrix: Annualized asset covariance matrix (its columns define asset order).
        constraints: See `portpy.models.base` (`WeightBounds`, `GroupCap`,
            `TurnoverCap`, `NetExposure`/`GrossExposure`). Full investment
            (`NetExposure(1.0)`) is assumed unless overridden.
        risk_aversion: Higher values weight the variance penalty more heavily,
            pulling the solution toward lower risk (and lower return).

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns.
    """
    names = _asset_names(cov_matrix)
    mu = _mu_array(expected_returns, names)
    cov = _cov_array(cov_matrix, names)

    def objective(w: np.ndarray) -> float:
        return float(-(w @ mu) + 0.5 * risk_aversion * (w @ cov @ w))

    def jac(w: np.ndarray) -> np.ndarray:
        return -mu + risk_aversion * (cov @ w)

    weights, diagnostics = solve_weights(objective, names, constraints, jac=jac)
    weights.attrs["solve_diagnostics"] = diagnostics
    return weights


def min_variance(cov_matrix: pd.DataFrame, constraints: Sequence[Any] | None = None) -> pd.Series:
    """
    Minimize portfolio variance: `minimize w'Sigma*w`. Needs no return forecast at all.

    Args:
        cov_matrix: Annualized asset covariance matrix.
        constraints: See `mean_variance`.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns.
    """
    names = _asset_names(cov_matrix)
    cov = _cov_array(cov_matrix, names)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    weights, diagnostics = solve_weights(objective, names, constraints, jac=jac)
    weights.attrs["solve_diagnostics"] = diagnostics
    return weights


def max_sharpe(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: Sequence[Any] | None = None,
    rf: float = 0.0,
) -> pd.Series:
    """
    Maximize the (annualized) Sharpe ratio: `maximize (w'mu - rf) / sqrt(w'Sigma*w)`.

    Args:
        expected_returns: Annualized expected return per asset.
        cov_matrix: Annualized asset covariance matrix.
        constraints: See `mean_variance`.
        rf: Annual risk-free rate, in the same (annualized) units as `expected_returns`.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns.
    """
    names = _asset_names(cov_matrix)
    mu = _mu_array(expected_returns, names)
    cov = _cov_array(cov_matrix, names)

    def objective(w: np.ndarray) -> float:
        vol = np.sqrt(max(float(w @ cov @ w), 1e-16))
        return float(-(float(w @ mu) - rf) / vol)

    weights, diagnostics = solve_weights(objective, names, constraints)
    weights.attrs["solve_diagnostics"] = diagnostics
    return weights


def target_return(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target: float,
    constraints: Sequence[Any] | None = None,
) -> pd.Series:
    """
    Minimize variance subject to hitting an exact expected return: `w'mu == target`.

    Note this is an equality, not "at least target" - a target below the
    global minimum-variance portfolio's own return still pins you to that
    lower, needlessly higher-variance point (some other libraries, e.g.
    PyPortfolioOpt's `efficient_return`, use a `>=` inequality instead, which
    substitutes the min-variance portfolio once its own return already clears
    the bar). The equality is what lets `efficient_frontier` sweep a full,
    non-degenerate curve by calling this at every grid point.

    Args:
        expected_returns: Annualized expected return per asset.
        cov_matrix: Annualized asset covariance matrix.
        target: Required annualized portfolio return.
        constraints: See `mean_variance`.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns. If `target` is
        infeasible given the other constraints, the SLSQP solve won't fully
        converge - check `.models.optimize(...).diagnostics["success"]`.
    """
    names = _asset_names(cov_matrix)
    mu = _mu_array(expected_returns, names)
    cov = _cov_array(cov_matrix, names)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    return_constraint = {"type": "eq", "fun": lambda w: float(w @ mu - target)}
    weights, diagnostics = solve_weights(objective, names, constraints, jac=jac, extra_scipy_constraints=[return_constraint])
    weights.attrs["solve_diagnostics"] = diagnostics
    return weights


def target_volatility(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target: float,
    constraints: Sequence[Any] | None = None,
) -> pd.Series:
    """
    Maximize expected return subject to a volatility cap: `w'Sigma*w <= target^2`.

    Args:
        expected_returns: Annualized expected return per asset.
        cov_matrix: Annualized asset covariance matrix.
        target: Maximum allowed annualized volatility.
        constraints: See `mean_variance`.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns. The cap binds
        (volatility == target) whenever the unconstrained-by-risk optimum
        would exceed it, which is the usual case.
    """
    names = _asset_names(cov_matrix)
    mu = _mu_array(expected_returns, names)
    cov = _cov_array(cov_matrix, names)

    def objective(w: np.ndarray) -> float:
        return float(-(w @ mu))

    def jac(w: np.ndarray) -> np.ndarray:
        return -mu

    vol_constraint = {"type": "ineq", "fun": lambda w: float(target**2 - w @ cov @ w)}
    weights, diagnostics = solve_weights(objective, names, constraints, jac=jac, extra_scipy_constraints=[vol_constraint])
    weights.attrs["solve_diagnostics"] = diagnostics
    return weights


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    constraints: Sequence[Any] | None = None,
    min_return: float | None = None,
    max_return: float | None = None,
) -> pd.DataFrame:
    """
    Trace the efficient frontier: the min-variance solution at each of `n_points`
    target returns spanning `[min_return, max_return]`.

    Args:
        expected_returns: Annualized expected return per asset.
        cov_matrix: Annualized asset covariance matrix.
        n_points: Number of frontier points to compute.
        constraints: See `mean_variance`, applied at every point.
        min_return: Lower end of the return grid. Defaults to the global
            minimum-variance portfolio's own return.
        max_return: Upper end of the return grid. Defaults to the single
            highest per-asset expected return (only exactly reachable
            long-only/unconstrained; with custom bounds it may not be, in
            which case that point's row is simply dropped - see below). This
            default is a long-only ceiling: it does not account for leverage
            or short-selling. If `constraints` includes a `WeightBounds` with
            `high > 1` or `low < 0`, the true reachable frontier extends past
            the single highest per-asset return - pass `max_return` explicitly
            to trace that far, since the default won't do it for you.

    Returns:
        A DataFrame with one row per (successfully solved) grid point:
        columns "target_return", "return", "volatility", "sharpe" (rf=0),
        plus one column per asset holding that point's weight. Points where
        `target_return` turned out infeasible for the given `constraints`
        are silently dropped, so the result can have fewer than `n_points` rows.
    """
    names = _asset_names(cov_matrix)
    mu = expected_returns.reindex(names)
    cov_np = _cov_array(cov_matrix, names)

    if min_return is None:
        mv_weights = min_variance(cov_matrix, constraints)
        min_return = float(mv_weights.to_numpy() @ mu.to_numpy())
    if max_return is None:
        max_return = float(mu.max())
    if max_return <= min_return:
        max_return = min_return + abs(min_return) * 0.5 + 1e-6

    rows = []
    for target in np.linspace(min_return, max_return, n_points):
        w = target_return(expected_returns, cov_matrix, target=float(target), constraints=constraints)
        w_np = w.reindex(names).to_numpy()
        port_ret = float(w_np @ mu.to_numpy())
        port_vol = float(np.sqrt(max(w_np @ cov_np @ w_np, 0.0)))
        row = {"target_return": float(target), "return": port_ret, "volatility": port_vol, "sharpe": safe_divide(port_ret, port_vol)}
        row.update(dict(zip(names, w_np, strict=True)))
        rows.append(row)

    return pd.DataFrame(rows)


def black_litterman(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | tuple[pd.DataFrame, pd.Series],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    omega: pd.DataFrame | None = None,
    constraints: Sequence[Any] | None = None,
) -> pd.Series:
    """
    Black-Litterman: blend market-implied equilibrium returns with explicit
    views, then mean-variance optimize the posterior estimates.

    Args:
        cov_matrix: Annualized asset covariance matrix.
        market_weights: Market-capitalization (or other equilibrium) weights,
            used to back out the implied equilibrium return `pi = risk_aversion * Sigma @ w_mkt`.
        views: Either a simple absolute-view dict (`{"AAPL": 0.12}` means "I
            expect AAPL to return 12%/yr"), or an explicit `(P, Q)` pair for
            relative/multi-asset views - `P` a DataFrame (one row per view,
            columns = a subset of `cov_matrix`'s assets, entries = that view's
            picking weights) and `Q` a Series of view returns, one per row of `P`.
            Every view value (dict values, or `Q`) must be annualized, on the
            same scale as `cov_matrix` (and therefore `pi`) - nothing checks
            this, so a periodic-scale view number silently distorts the
            posterior instead of raising.
        tau: Scales the uncertainty of the equilibrium prior (`tau * Sigma`).
            Smaller values trust the equilibrium more; larger values let views
            dominate faster.
        risk_aversion: Used both to back out `pi` from `market_weights` and as
            the final mean-variance step's risk aversion.
        omega: View uncertainty matrix. Defaults to He & Litterman's usual
            proportional choice, `diag(P @ (tau*Sigma) @ P')` - views implied
            by noisier picking rows get proportionally less confidence automatically.
        constraints: See `mean_variance`, applied to the final posterior optimization.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns.
    """
    names = _asset_names(cov_matrix)
    cov = _cov_array(cov_matrix, names)
    w_mkt = market_weights.reindex(names).fillna(0.0).to_numpy()

    pi = risk_aversion * (cov @ w_mkt)

    if isinstance(views, dict):
        assets = list(views.keys())
        index = {name: i for i, name in enumerate(names)}
        unknown = [a for a in assets if a not in index]
        if unknown:
            raise ValueError(f"views references unknown asset(s): {unknown}")
        p_matrix = np.zeros((len(assets), len(names)))
        for row, asset in enumerate(assets):
            p_matrix[row, index[asset]] = 1.0
        q_vector = np.array([views[a] for a in assets], dtype=float)
    else:
        p_df, q_series = views
        p_matrix = p_df.reindex(columns=names).fillna(0.0).to_numpy()
        q_vector = q_series.to_numpy(dtype=float)

    tau_cov = tau * cov
    if omega is None:
        omega_matrix = np.diag(np.diag(p_matrix @ tau_cov @ p_matrix.T))
        if np.allclose(omega_matrix, 0.0):
            omega_matrix = omega_matrix + np.eye(omega_matrix.shape[0]) * 1e-8
    else:
        omega_matrix = omega.to_numpy() if isinstance(omega, pd.DataFrame) else np.asarray(omega, dtype=float)

    tau_cov_inv = np.linalg.pinv(tau_cov)
    omega_inv = np.linalg.pinv(omega_matrix)
    posterior_precision = tau_cov_inv + p_matrix.T @ omega_inv @ p_matrix
    posterior_cov_of_mean = np.linalg.pinv(posterior_precision)
    posterior_mu = posterior_cov_of_mean @ (tau_cov_inv @ pi + p_matrix.T @ omega_inv @ q_vector)

    # Deliberately optimize on the ORIGINAL cov, not cov + posterior_cov_of_mean:
    # this is what makes "no views" reproduce market_weights exactly at the same
    # risk_aversion used to build pi (posterior_mu collapses to pi, and pi was
    # built from cov, not an inflated posterior). Some expositions add
    # posterior_cov_of_mean back in for the final step (folding return-estimation
    # uncertainty into the risk term); PortPy follows the more common convention
    # (matching PyPortfolioOpt's default) that keeps this recovery property intact.
    mu_series = pd.Series(posterior_mu, index=names)
    cov_df = pd.DataFrame(cov, index=names, columns=names)
    return mean_variance(mu_series, cov_df, constraints=constraints, risk_aversion=risk_aversion)


def hierarchical_risk_parity(
    cov_matrix: pd.DataFrame,
    constraints: Sequence[Any] | None = None,
    linkage_method: str = "single",
) -> pd.Series:
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016): cluster assets by
    correlation, then recursively split inverse-variance weight down the tree
    instead of inverting the covariance matrix directly.

    No numerical optimizer is involved - HRP is a deterministic, closed-form
    algorithm, which is also why it tolerates a poorly-conditioned (e.g.
    near-singular) covariance matrix far better than the matrix-inversion-based
    solvers above.

    Args:
        cov_matrix: Asset covariance matrix (annualization doesn't affect the
            result - HRP only uses correlations and *relative* variances).
        constraints: Only `WeightBounds` is supported (applied post-hoc: clip
            to bounds, then redistribute any surplus/deficit through the
            assets not already pinned at a bound, so the result still sums to
            1 without re-violating the bound). Any other constraint type is
            ignored with a `UserWarning`, since recursive bisection has no
            natural way to target an exact group cap, turnover budget, or
            leverage level.
        linkage_method: `scipy.cluster.hierarchy.linkage` method (default
            "single", matching the original paper).

    Returns:
        Weights, indexed by `cov_matrix`'s columns, long-only and summing to 1.
    """
    names = _asset_names(cov_matrix)
    cov_np = _cov_array(cov_matrix, names)
    n = len(names)

    std = np.sqrt(np.diag(cov_np))
    corr = cov_np / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    if n > 2:
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        link = hierarchy.linkage(squareform(dist, checks=False), method=linkage_method)
        sort_ix = hierarchy.leaves_list(link).tolist()
    else:
        sort_ix = list(range(n))

    def cluster_variance(idxs: list[int]) -> float:
        sub = cov_np[np.ix_(idxs, idxs)]
        inv_diag = 1.0 / np.diag(sub)
        ivp = inv_diag / inv_diag.sum()
        return float(ivp @ sub @ ivp)

    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = cluster_variance(left)
            var_right = cluster_variance(right)
            alpha = 1.0 - safe_divide(var_left, var_left + var_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha

    weights = w.sort_index()
    weights.index = names
    weights = weights.rename("weight")

    all_constraints = list(constraints or [])
    bounds_constraint = next((c for c in all_constraints if isinstance(c, WeightBounds)), None)
    unsupported = [c for c in all_constraints if not isinstance(c, WeightBounds)]
    if unsupported:
        warnings.warn(
            "hierarchical_risk_parity only supports WeightBounds; ignoring "
            f"{[type(c).__name__ for c in unsupported]} - recursive bisection has no "
            "natural way to enforce group/turnover/exposure constraints.",
            stacklevel=2,
        )
    if bounds_constraint is not None:
        weights = _clip_and_redistribute(weights, bounds_constraint)

    return weights


def _clip_and_redistribute(weights: pd.Series, bounds: WeightBounds) -> pd.Series:
    """
    Clip `weights` to `bounds` and redistribute the resulting surplus/deficit
    among the assets *not* pinned at a bound, so the result still sums to 1
    without re-violating the bound that redistribution just applied.

    A naive clip-then-uniformly-rescale (dividing every weight by the clipped
    sum) can push already-clipped assets back past their own cap whenever
    they're a large share of the clipped total - this instead redistributes
    only through assets with room left, iterating (at most once per asset)
    until nothing more needs to move or every asset is pinned.
    """
    names = list(weights.index)
    lows = pd.Series(bounds.low, index=names, dtype=float)
    highs = pd.Series(bounds.high, index=names, dtype=float)
    if bounds.per_asset:
        for asset, (lo, hi) in bounds.per_asset.items():
            lows[asset] = lo
            highs[asset] = hi

    w = weights.reindex(names).astype(float)
    free = pd.Series(True, index=names)

    for _ in range(len(names) + 1):
        w = w.clip(lower=lows, upper=highs)
        residual = 1.0 - float(w.sum())
        if abs(residual) < 1e-10 or not free.any():
            break
        pinned_now = free & (
            (np.isclose(w, lows) & (residual < 0)) | (np.isclose(w, highs) & (residual > 0))
        )
        free = free & ~pinned_now
        if not free.any():
            break
        free_total = float(w[free].sum())
        if free_total > 1e-12:
            w[free] = w[free] * (1.0 + residual / free_total)
        else:
            w[free] = w[free] + residual / int(free.sum())

    if abs(1.0 - float(w.sum())) > 1e-6:
        warnings.warn(
            "hierarchical_risk_parity: WeightBounds is infeasible for a fully-invested "
            f"book (sum(w)={float(w.sum()):.4f} != 1) - widen low/high or per_asset.",
            stacklevel=3,
        )
    return w.rename("weight")


def risk_budgeting(
    cov_matrix: pd.DataFrame,
    budget: pd.Series,
    constraints: Sequence[Any] | None = None,
) -> pd.Series:
    """
    Solve for weights whose risk contributions match a target budget:
    `weight_i * MCTR_i approx= budget_i * portfolio_variance` for every asset.

    Args:
        cov_matrix: Annualized asset covariance matrix.
        budget: Target risk-contribution share per asset (renormalized to sum
            to 1 internally, so relative sizes are what matter).
        constraints: See `mean_variance`. Risk budgeting is only well-posed
            long-only - overriding `WeightBounds` to allow negative weights
            makes "risk contribution" sign-ambiguous.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns.
    """
    names = _asset_names(cov_matrix)
    cov = _cov_array(cov_matrix, names)
    b = budget.reindex(names).to_numpy(dtype=float)
    b = b / b.sum()

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ cov @ w)
        if port_var <= 1e-16:
            return 0.0
        risk_contribution = w * (cov @ w)
        target = b * port_var
        return float(np.sum((risk_contribution - target) ** 2))

    weights, diagnostics = solve_weights(objective, names, constraints)
    weights.attrs["solve_diagnostics"] = diagnostics
    return weights


def risk_parity(cov_matrix: pd.DataFrame, constraints: Sequence[Any] | None = None) -> pd.Series:
    """
    Equal Risk Contribution: every asset contributes the same share of total
    portfolio risk. A special case of `risk_budgeting` with an equal budget.

    Args:
        cov_matrix: Annualized asset covariance matrix.
        constraints: See `risk_budgeting`.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns.
    """
    names = _asset_names(cov_matrix)
    return risk_budgeting(cov_matrix, pd.Series(1.0, index=names), constraints=constraints)


def maximum_diversification(cov_matrix: pd.DataFrame, constraints: Sequence[Any] | None = None) -> pd.Series:
    """
    Maximize `metrics.covariance.diversification_ratio` directly: `maximize (w'sigma) / sqrt(w'Sigma*w)`.

    Args:
        cov_matrix: Annualized asset covariance matrix.
        constraints: See `mean_variance`.

    Returns:
        Solved weights, indexed by `cov_matrix`'s columns.
    """
    names = _asset_names(cov_matrix)

    def objective(w: np.ndarray) -> float:
        dr = _diversification_ratio(pd.Series(w, index=names), cov_matrix)
        return float(-dr) if np.isfinite(dr) else 0.0

    weights, diagnostics = solve_weights(objective, names, constraints)
    weights.attrs["solve_diagnostics"] = diagnostics
    return weights


_MODEL_INTERPRET = lambda r: (  # noqa: E731
    f"HHI={float((r.weights ** 2).sum()):.3f} ({int((r.weights.abs() > 0.01).sum())} active positions), "
    f"objective={r.diagnostics.get('objective_value', float('nan')):.4g}, "
    f"converged={r.diagnostics.get('success', 'n/a')}"
)

register(
    Explanation(
        name="mean_variance",
        category="model",
        summary="Markowitz's original portfolio problem: pick weights that maximize expected return net of a risk penalty on variance.",
        formula="maximize w'mu - 0.5*risk_aversion*w'Sigma*w, s.t. sum(w)=1 (+ any extra constraints)",
        how_to_read="Higher risk_aversion pulls the solution toward the global minimum-variance portfolio; lower risk_aversion pulls it toward concentrating in the highest-mu assets.",
        good_vs_bad="A well-diversified solution (moderate HHI, several active positions) suggests mu/Sigma disagree enough to justify holding multiple assets; a solution collapsed into 1-2 assets often means mu is dominating an under-diversified Sigma estimate (Michaud's 'error maximization').",
        caveats="Extremely sensitive to the expected_returns input - see expected_returns' own caveats. risk_aversion has no universal 'correct' value; it must be calibrated to the investor, not treated as a free hyperparameter to tune for a nicer-looking frontier. Solved via multi-start SLSQP (see portpy.models.base), not a specialized convex QP solver - verified against PyPortfolioOpt/Riskfolio-Lib in the cross-validation tests, but don't expect bit-for-bit identical weights on every problem.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="min_variance",
        category="model",
        summary="The single portfolio on the efficient frontier with the lowest possible variance, ignoring expected returns entirely.",
        formula="minimize w'Sigma*w, s.t. sum(w)=1 (+ any extra constraints)",
        how_to_read="This is the leftmost point of the efficient frontier - every other frontier portfolio has equal or higher risk.",
        good_vs_bad="Appropriate when you have little confidence in your return forecasts (a common, defensible choice, since Sigma is estimated far more reliably than mu) or explicitly want the lowest-risk fully-invested portfolio.",
        caveats="Says nothing about return - the min-variance portfolio can have a very low or even negative expected return. Still inherits covariance estimation-risk caveats from whatever cov_matrix estimator you used.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="max_sharpe",
        category="model",
        summary="The portfolio on the efficient frontier with the highest ratio of expected excess return to volatility - the tangency portfolio.",
        formula="maximize (w'mu - rf) / sqrt(w'Sigma*w), s.t. sum(w)=1 (+ any extra constraints)",
        how_to_read="Geometrically, this is where a line from the risk-free rate is tangent to the efficient frontier - combining it with cash/leverage traces out the entire capital allocation line.",
        good_vs_bad="A high in-sample Sharpe here is close to guaranteed by construction (it's literally what's being maximized) - it says little about out-of-sample performance. Judge the *inputs* (mu/Sigma quality), not the achieved objective value.",
        caveats="The Sharpe-ratio objective is non-convex in general, unlike mean_variance/min_variance - the multi-start solve in portpy.models.base exists specifically to guard against local optima here. The single most estimation-error-sensitive optimizer in this module, since it inherits both the mu-sensitivity of mean_variance and a ratio objective that can be unstable when volatility is small.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="target_return",
        category="model",
        summary="The lowest-variance portfolio that achieves an exact target expected return - one specific point on the efficient frontier.",
        formula="minimize w'Sigma*w, s.t. w'mu = target, sum(w)=1 (+ any extra constraints)",
        how_to_read="Use this when you have a required return (e.g. a liability or spending target) rather than a risk-aversion preference.",
        good_vs_bad="diagnostics['success']=False after solving means `target` was infeasible given the other constraints (e.g. above the return of the single best asset, or below what's reachable with a WeightBounds floor) - treat the returned weights as unreliable in that case, not as a valid frontier point.",
        caveats="Infeasible targets don't raise an error - they just fail to converge silently unless you check diagnostics. Same mu-sensitivity caveat as mean_variance. Uses an EQUALITY constraint (w'mu = target exactly), not 'at least target' - if you ask for a target below the global minimum-variance portfolio's own return, you still get pinned to that lower return (needlessly, since the min-variance portfolio already clears it) rather than the tool silently handing you the better, lower-variance min-variance portfolio instead. Some other libraries (e.g. PyPortfolioOpt's efficient_return) use return >= target instead, which does substitute the min-variance portfolio in that case - the two conventions genuinely disagree below the min-variance return and only agree above it.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="target_volatility",
        category="model",
        summary="The highest-expected-return portfolio subject to a volatility cap - the frontier point at a chosen risk level.",
        formula="maximize w'mu, s.t. w'Sigma*w <= target^2, sum(w)=1 (+ any extra constraints)",
        how_to_read="Use this when you have a risk budget (e.g. 'no more than 12% annualized volatility') rather than a risk-aversion preference.",
        good_vs_bad="The cap almost always binds exactly (realized volatility == target) whenever target is above the global minimum-variance portfolio's volatility; if target is below it, the problem is infeasible and diagnostics['success'] will read False.",
        caveats="Same mu-sensitivity caveat as mean_variance; the cap constraint is convex (a quadratic inequality) but the overall multi-start SLSQP path is still shared with the non-convex solvers, so treat convergence as 'likely, verify via diagnostics' rather than guaranteed.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="efficient_frontier",
        category="model",
        summary="Traces the whole efficient frontier - the best (lowest-variance) portfolio achievable at each of a grid of target returns - rather than picking one point.",
        formula="Repeated target_return solves across a grid from the min-variance portfolio's return up to the highest single-asset return.",
        how_to_read="A ModelResult here has .weights set to the max-Sharpe point on the curve for convenience; the full curve (return/volatility/sharpe/weights per grid point) lives in .diagnostics['frontier'] and .summary().",
        good_vs_bad="A smooth, monotonically-increasing volatility-vs-return curve is the healthy case. Gaps (missing rows) mean some target returns were infeasible under the given constraints - fewer points than requested is a signal, not a bug.",
        caveats="Every point inherits mean_variance's mu/Sigma sensitivity independently - the frontier's *shape* (not just its level) can shift substantially if mu is re-estimated. n_points controls resolution only, not solve quality at any single point. The default return grid stops at the single highest per-asset expected return - a long-only ceiling that understates the true frontier's extent whenever constraints allow leverage or short-selling; pass max_return explicitly in that case.",
        interpret=lambda r: f"{len(r.diagnostics.get('frontier', r.weights.to_frame()))} points, return range handled internally - see .summary()",
    )
)

register(
    Explanation(
        name="black_litterman",
        category="model",
        summary="Blends market-implied equilibrium returns with your own explicit views (with confidence levels), producing a posterior return estimate that's typically far more stable than plugging historical means straight into mean-variance.",
        formula="pi = risk_aversion*Sigma@w_mkt; posterior_mu = [(tau*Sigma)^-1 + P'Omega^-1P]^-1 [(tau*Sigma)^-1 pi + P'Omega^-1 Q]; then mean_variance(posterior_mu, Sigma, risk_aversion)",
        how_to_read="Without any views (an empty views dict), posterior_mu collapses back to pi exactly, which reproduces market_weights exactly at the same risk_aversion used to build pi - views only pull the result away from the market portfolio in proportion to how confident (via Omega) they are.",
        good_vs_bad="A result that stays close to market_weights when your views are weak/uncertain, and moves further only for high-confidence views, is Black-Litterman behaving as designed - the opposite (wild swings from a single low-confidence view) suggests Omega is mis-specified (too small = overconfident).",
        caveats="Only as good as market_weights (the equilibrium anchor) and tau (how much you trust that anchor vs. your views) - both are modeling choices, not estimated from data. The default 'proportional' Omega is a common convenience, not the only valid choice; badly-scaled views dict entries (e.g. mismatched units) will silently distort the posterior rather than raising an error.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="hierarchical_risk_parity",
        category="model",
        summary="Lopez de Prado's (2016) HRP: clusters assets by correlation structure, then allocates risk recursively down the resulting tree instead of inverting the full covariance matrix.",
        formula="quasi-diagonalize Sigma via hierarchical clustering on correlation distance sqrt(0.5*(1-corr)); recursively split inverse-variance weight between the two halves of each cluster",
        how_to_read="No risk_aversion or expected_returns to tune - the only real lever is linkage_method, which changes how assets get clustered.",
        good_vs_bad="Tends to spread weight more evenly across correlated groups than mean-variance-based methods, and is far less sensitive to small changes in the covariance matrix (no matrix inversion) - the tradeoff is that it optimizes nothing explicitly, so it can be dominated in-sample by a well-specified mean-variance solve.",
        caveats="Ignores expected returns entirely by design - it is a risk-allocation heuristic, not a return-seeking optimizer. Only WeightBounds constraints are supported (applied post-hoc via clip-and-redistribute); every other constraint type is dropped with a warning. Results depend on the clustering linkage_method and can change noticeably (not just numerically, but which assets end up in which sub-cluster) with small covariance perturbations near a clustering boundary.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="risk_parity",
        category="model",
        summary="Equal Risk Contribution: allocates so that every asset contributes the same share of total portfolio volatility, rather than the same dollar weight.",
        formula="solve for w such that w_i * (Sigma@w)_i is equal across all i, s.t. sum(w)=1, w>=0",
        how_to_read="A low-volatility asset ends up with a *larger* weight than a high-volatility one, specifically so their risk contributions match - don't compare the weights to a cap-weighted or equal-weight benchmark and expect them to look similar.",
        good_vs_bad="Nearly-equal component_contribution_to_risk values (see metrics.covariance) across assets confirms the solve converged correctly; large residual dispersion after solving means diagnostics['success'] is probably False.",
        caveats="Long-only by construction - risk contribution sign becomes ambiguous with shorts, so allowing negative weights via WeightBounds breaks the interpretation even though the solver won't stop you. Ignores expected returns entirely, like hierarchical_risk_parity. A special case of risk_budgeting with an equal budget; same solver caveats apply.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="risk_budgeting",
        category="model",
        summary="Generalizes risk_parity to an explicit, non-equal target split of total portfolio risk across assets.",
        formula="solve for w such that w_i * (Sigma@w)_i / portfolio_variance is proportional to budget_i, s.t. sum(w)=1, w>=0",
        how_to_read="budget is renormalized to sum to 1 internally, so only relative sizes matter - budget=[2,1,1] means the first asset should carry twice the risk share of each of the other two.",
        good_vs_bad="Realized risk contributions (metrics.covariance.component_contribution_to_risk) close to the requested budget shares confirm convergence; large deviations mean check diagnostics['success'].",
        caveats="Same long-only-by-construction and expected-returns-agnostic caveats as risk_parity. An extreme budget concentrated in one asset can push the solve toward that asset's corner solution, which may not converge cleanly under tight WeightBounds.",
        interpret=_MODEL_INTERPRET,
    )
)

register(
    Explanation(
        name="maximum_diversification",
        category="model",
        summary="Choueifaty & Coignard's (2008) Most Diversified Portfolio: maximizes metrics.covariance.diversification_ratio directly.",
        formula="maximize (w'sigma) / sqrt(w'Sigma*w), s.t. sum(w)=1 (+ any extra constraints)",
        how_to_read="The result is, by construction, the fully-invested portfolio with the highest possible diversification_ratio given Sigma - call metrics.covariance.diversification_ratio on the resulting weights to see the achieved value.",
        good_vs_bad="A diversification_ratio well above 1 (see that metric's own card) confirms the optimizer found real diversification benefit in the correlation structure; a value near 1 means the assets are too correlated for this method to add much over equal-risk alternatives.",
        caveats="Ignores expected returns entirely, like hierarchical_risk_parity/risk_parity. The diversification_ratio objective's own caveat applies directly: its clean '>=1, higher is better' interpretation assumes long-only weights - allowing shorts via WeightBounds breaks that guarantee even though the solver will still run.",
        interpret=_MODEL_INTERPRET,
    )
)
