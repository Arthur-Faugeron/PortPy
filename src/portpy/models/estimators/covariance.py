"""
Estimating the covariance matrix (Sigma) that feeds portpy.models.optimization.

Distinct from `portpy.metrics.covariance`: that module *decomposes realized
risk* for a given weight vector against a given matrix (portfolio_variance,
marginal_contribution_to_risk, ...). This module *produces* the matrix itself,
via several alternative estimators - the two compose (estimate here, decompose
there).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, register
from portpy.utils.constants import TRADING_DAYS_PER_YEAR
from portpy.utils.validation import ensure_min_observations

__all__ = ["covariance"]

_METHODS = ("sample", "ewma", "shrinkage", "ledoit_wolf", "robust")


def covariance(
    y: pd.DataFrame,
    method: str = "sample",
    span: int | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annualize: bool = True,
    shrinkage_intensity: float = 0.2,
) -> pd.DataFrame:
    """
    Estimate an (by default annualized) asset covariance matrix.

    Args:
        y: Per-asset periodic returns - the full multi-asset DataFrame (matches
            `Portfolio.asset_returns()`).
        method: One of:
            - "sample": plain sample covariance (`y.cov()`, ddof=1). Same
              formula as `metrics.covariance.covariance_matrix`, just annualized
              by default here since optimizers expect it paired with an
              annualized `expected_returns`.
            - "ewma": RiskMetrics-style exponentially-weighted covariance -
              recent co-movements dominate, controlled by `span`.
            - "shrinkage": manual-intensity linear shrinkage of the sample
              matrix toward a diagonal target that keeps each asset's own
              sample variance and zeroes every cross-correlation, at a
              user-chosen `shrinkage_intensity`. This is a different target
              from the "shrink every variance toward the cross-sectional
              average" convention some other libraries use for their own
              "shrinkage" method (e.g. PyPortfolioOpt's `shrunk_covariance`,
              which shrinks toward a *scaled-identity* matrix - constant
              variance across assets - instead of a diagonal-of-sample
              matrix) - the two do not, in general, agree at the same
              intensity, and neither is "more correct" than the other.
            - "ledoit_wolf": Ledoit-Wolf (2004) automatic-intensity shrinkage
              toward a scaled-identity target (`sklearn.covariance.LedoitWolf`) -
              the shrinkage intensity is estimated from the data, not chosen by hand.
            - "robust": Minimum Covariance Determinant (`sklearn.covariance.MinCovDet`) -
              a high-breakdown-point estimator resistant to outlier periods (e.g. crashes).
              `MinCovDet` fits the raw MCD subset covariance and then applies a
              statistical consistency correction (reweighting using the
              subset's own outlier threshold) on top of it, so this is not
              numerically identical to a "raw" MCD covariance computed some
              other way (e.g. `sklearn.covariance.fast_mcd` called directly,
              which some other libraries wrap instead - see the cross-validation
              notebook for the resulting gap on real data). Needs enough
              observations relative to the asset count for the MCD subset
              search to be well-posed; with too few rows relative to columns,
              `MinCovDet` raises rather than silently degrading.
        span: EWMA decay in periods, used by "ewma" (default 60 if omitted).
        periods_per_year: Periods per year, used when `annualize` is True.
        annualize: If True (default), scale the matrix by `periods_per_year`
            (assumes i.i.d., serially uncorrelated returns - same caveat as
            `metrics.covariance.covariance_matrix`).
        shrinkage_intensity: Blend weight on the diagonal target for
            "shrinkage", in [0, 1] (0 = pure sample covariance, 1 = pure diagonal).

    Returns:
        The (optionally annualized) covariance matrix, indexed and columned by asset.

    Raises:
        ValueError: If `method` is unrecognized or `shrinkage_intensity` is
            outside [0, 1].
    """
    ensure_min_observations(y, 2, "y")
    scale = periods_per_year if annualize else 1.0

    if method == "sample":
        return y.cov() * scale

    if method == "ewma":
        window = 60 if span is None else span
        lam = 1.0 - 2.0 / (window + 1.0)
        arr = y.to_numpy(dtype=float)
        arr = arr - np.nanmean(arr, axis=0)
        n, k = arr.shape
        weights = lam ** np.arange(n - 1, -1, -1)
        weights = weights / weights.sum()
        weighted = arr * weights[:, None]
        cov = weighted.T @ arr
        return pd.DataFrame(cov * scale, index=y.columns, columns=y.columns)

    if method == "shrinkage":
        if not (0.0 <= shrinkage_intensity <= 1.0):
            raise ValueError(f"shrinkage_intensity must be in [0, 1], got {shrinkage_intensity}.")
        sample = y.cov()
        target = pd.DataFrame(np.diag(np.diag(sample.to_numpy())), index=sample.index, columns=sample.columns)
        blended = shrinkage_intensity * target + (1.0 - shrinkage_intensity) * sample
        return blended * scale

    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        fitted = LedoitWolf().fit(y.dropna().to_numpy())
        return pd.DataFrame(fitted.covariance_ * scale, index=y.columns, columns=y.columns)

    if method == "robust":
        from sklearn.covariance import MinCovDet

        fitted = MinCovDet(random_state=0).fit(y.dropna().to_numpy())
        return pd.DataFrame(fitted.covariance_ * scale, index=y.columns, columns=y.columns)

    raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")


register(
    Explanation(
        name="covariance",
        category="function",
        summary="Estimates the asset return covariance matrix (Sigma) that every optimizer in .models.optimization uses to measure risk and diversification.",
        formula="sample: cov(y) | ewma: exponentially-weighted cov, decay from span | shrinkage: (1-a)*sample + a*diag(sample) | ledoit_wolf: automatic-intensity shrinkage to scaled identity | robust: Minimum Covariance Determinant",
        how_to_read="A symmetric, positive-(semi)definite matrix; diagonal = each asset's own variance, off-diagonal = co-movement. Feed it straight into .models.optimization.* as `cov_matrix=`.",
        good_vs_bad="No universal 'good value' - judge the *method* choice by sample size relative to asset count. With few observations relative to the number of assets, prefer ledoit_wolf/shrinkage over sample (which becomes poorly conditioned and can even be singular).",
        caveats="annualize=True's linear scaling by periods_per_year assumes i.i.d., serially uncorrelated returns, same as metrics.covariance_matrix. 'sample' and 'ewma' have no shrinkage and are poorly conditioned when the history isn't much longer than the asset count - the exact estimation-risk problem 'shrinkage'/'ledoit_wolf' exist to address. 'shrinkage' targets a diagonal-of-sample matrix (each asset keeps its own variance), not the scaled-identity target some other libraries default to under the same name - don't expect the two to match at the same intensity. 'robust' downweights whole outlier *periods* (e.g. crash days), which is a feature for correlation stability but means a genuine regime shift can be partly discounted as 'outliers'; it also applies MinCovDet's consistency-reweighting step, so it isn't a 'raw' MCD estimate either.",
    )
)
