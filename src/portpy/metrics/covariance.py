"""Covariance/correlation matrices and portfolio-level risk decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, MetricResult, register
from portpy.utils.constants import TRADING_DAYS_PER_YEAR

__all__ = [
    "covariance_matrix",
    "correlation_matrix",
    "portfolio_variance",
    "portfolio_volatility",
    "diversification_ratio",
    "marginal_contribution_to_risk",
    "component_contribution_to_risk",
]


def covariance_matrix(
    returns: pd.DataFrame, annualized: bool = False, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Sample covariance matrix of asset returns (ddof=1), optionally annualized."""
    cov = returns.cov()
    return cov * periods_per_year if annualized else cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix of asset returns."""
    return returns.corr()


def _align_weights(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    if isinstance(weights, pd.Series) and isinstance(cov_matrix, pd.DataFrame):
        weights = weights.reindex(cov_matrix.columns)
        if weights.isna().any():
            raise ValueError("weights do not cover all assets in cov_matrix's columns.")
        return weights.to_numpy()
    return np.asarray(weights, dtype=float)


def portfolio_variance(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame, as_result: bool = False) -> float | MetricResult:
    """Portfolio return variance: `w^T @ Cov @ w`."""
    w = _align_weights(weights, cov_matrix)
    cov = cov_matrix.to_numpy() if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)
    value = float(w @ cov @ w)
    return MetricResult(value, "portfolio_variance") if as_result else value


def portfolio_volatility(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame, as_result: bool = False) -> float | MetricResult:
    """Portfolio return standard deviation: `sqrt(w^T @ Cov @ w)`."""
    value = float(np.sqrt(portfolio_variance(weights, cov_matrix)))
    return MetricResult(value, "portfolio_volatility") if as_result else value


def diversification_ratio(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame, as_result: bool = False) -> float | MetricResult:
    """Weighted-average standalone volatility divided by actual portfolio volatility - how much diversification is buying you."""
    w = _align_weights(weights, cov_matrix)
    cov = cov_matrix.to_numpy() if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)
    asset_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(w @ asset_vols)
    port_vol = float(np.sqrt(w @ cov @ w))
    value = weighted_avg_vol / port_vol if port_vol > 1e-15 else np.nan
    return MetricResult(value, "diversification_ratio") if as_result else value


def marginal_contribution_to_risk(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    """Marginal Contribution to Risk (MCTR): how much portfolio volatility changes per unit change in each weight.

    `MCTR_i = (Cov @ w)_i / portfolio_volatility`
    """
    w = _align_weights(weights, cov_matrix)
    cov = cov_matrix.to_numpy() if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol < 1e-15:
        return np.zeros_like(w)
    return (cov @ w) / port_vol


def component_contribution_to_risk(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    """Component Contribution to Risk (CCTR): each asset's slice of total portfolio volatility.

    `CCTR_i = w_i * MCTR_i`, and `sum(CCTR) == portfolio_volatility` exactly (divide by
    portfolio_volatility for a percentage breakdown that sums to 100%).
    """
    w = _align_weights(weights, cov_matrix)
    mctr = marginal_contribution_to_risk(w, cov_matrix)
    return w * mctr


register(
    Explanation(
        name="portfolio_variance",
        category="metric",
        summary="The portfolio's overall return variance, accounting for every asset's individual variance *and* how they move together (covariance).",
        formula="w^T @ Cov @ w",
        how_to_read="In squared-return units - take the square root (portfolio_volatility) for something directly interpretable.",
        good_vs_bad="Lower is 'less risky' in absolute terms, but always weigh against expected return.",
    )
)

register(
    Explanation(
        name="portfolio_volatility",
        category="metric",
        summary="The portfolio's overall annualized-or-not standard deviation, combining every asset's volatility and their pairwise correlations.",
        formula="sqrt(w^T @ Cov @ w)",
        how_to_read="Directly comparable to a single asset's volatility metric.",
        good_vs_bad="Should almost always be lower than the weighted average of the individual assets' volatilities - if it isn't, check for a data or weights error, since that would imply *negative* diversification.",
        interpret=lambda v: f"{v:.2%}",
    )
)

register(
    Explanation(
        name="diversification_ratio",
        category="metric",
        summary="How much risk-reduction the portfolio is getting from combining imperfectly-correlated assets, versus holding them separately.",
        formula="(weighted average of individual asset volatilities) / portfolio_volatility",
        how_to_read="A ratio of 1.5 means the portfolio's actual volatility is 1/1.5 = 67% of what you'd get from the assets' volatilities alone (i.e. diversification cut risk by about a third).",
        good_vs_bad="Higher is better (more diversification benefit); exactly 1.0 means zero diversification benefit (assets are perfectly correlated, or there's only one asset).",
        interpret=lambda v: f"{v:.2f}x - diversification is reducing risk to about {1/v:.0%} of the undiversified level" if v > 0 else "n/a",
    )
)
