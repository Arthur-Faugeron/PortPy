"""
Covariance/correlation matrices and portfolio-level risk decomposition.
"""

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
    """
    Sample covariance matrix of asset returns (ddof=1), optionally annualized.
    """
    cov = returns.cov()
    return cov * periods_per_year if annualized else cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlation matrix of asset returns.
    """
    return returns.corr()


def _align_weights(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    if isinstance(weights, pd.Series) and isinstance(cov_matrix, pd.DataFrame):
        weights = weights.reindex(cov_matrix.columns)
        if weights.isna().any():
            raise ValueError("weights do not cover all assets in cov_matrix's columns.")
        return weights.to_numpy()
    return np.asarray(weights, dtype=float)


def portfolio_variance(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame, as_result: bool = False) -> float | MetricResult:
    """
    Portfolio return variance.
    """
    w = _align_weights(weights, cov_matrix)
    cov = cov_matrix.to_numpy() if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)
    value = float(w @ cov @ w)
    return MetricResult(value, "portfolio_variance") if as_result else value


def portfolio_volatility(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame, as_result: bool = False) -> float | MetricResult:
    """Portfolio return standard deviation: sqrt(w^T @ Cov @ w)."""
    value = float(np.sqrt(portfolio_variance(weights, cov_matrix)))
    return MetricResult(value, "portfolio_volatility") if as_result else value


def diversification_ratio(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame, as_result: bool = False) -> float | MetricResult:
    """
    Weighted-average standalone volatility divided by actual portfolio volatility. How much diversification is buying you.
    """
    w = _align_weights(weights, cov_matrix)
    cov = cov_matrix.to_numpy() if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)
    asset_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(w @ asset_vols)
    port_vol = float(np.sqrt(w @ cov @ w))
    value = weighted_avg_vol / port_vol if port_vol > 1e-15 else np.nan
    return MetricResult(value, "diversification_ratio") if as_result else value


def marginal_contribution_to_risk(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    Marginal Contribution to Risk (MCTR): how much portfolio volatility changes per unit change in each weight.
    """
    w = _align_weights(weights, cov_matrix)
    cov = cov_matrix.to_numpy() if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol < 1e-15:
        return np.zeros_like(w)
    return (cov @ w) / port_vol


def component_contribution_to_risk(weights: pd.Series | np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    Component Contribution to Risk (CCTR): each asset's slice of total portfolio volatility.

    CCTR_i = w_i * MCTR_i, and sum(CCTR) == portfolio_volatility exactly (divide by
    portfolio_volatility for a percentage breakdown that sums to 100%).
    """
    w = _align_weights(weights, cov_matrix)
    mctr = marginal_contribution_to_risk(w, cov_matrix)
    return w * mctr


register(
    Explanation(
        name="portfolio_variance",
        category="metric",
        summary="The total portfolio return variance, measuring how much portfolio returns fluctuate based on individual asset variances and how assets move together.",
        formula="w @ Cov @ w",
        how_to_read="Variance is expressed in squared return units. It is mainly useful as an intermediate calculation because volatility is easier to interpret.",
        good_vs_bad="Lower variance means lower absolute portfolio risk, but it should always be evaluated together with expected return and investment objectives.",
        caveats="Variance is sensitive to the covariance estimate. Short datasets or unstable correlations can produce unreliable risk estimates.",
    )
)

register(
    Explanation(
        name="portfolio_volatility",
        category="metric",
        summary="The portfolio's total return volatility, representing the expected dispersion of portfolio returns around their average return.",
        formula="sqrt(w @ Cov @ w)",
        how_to_read="Expressed as a percentage, volatility can be directly compared with the volatility of individual assets or benchmarks.",
        good_vs_bad="Lower volatility generally means lower risk, but higher volatility can be acceptable when compensated by higher expected returns.",
        caveats="Historical volatility does not predict future volatility. Market regimes, correlations, and liquidity conditions can change.",
        interpret=lambda v: f"{v:.2%}",
    )
)

register(
    Explanation(
        name="diversification_ratio",
        category="metric",
        summary="Measures how much diversification benefit a portfolio receives from combining assets with different volatility and correlation characteristics.",
        formula="(sum(weight_i * asset_volatility_i)) / portfolio_volatility",
        how_to_read="A value above 1 means the combined portfolio is less risky than the weighted average standalone asset risks.",
        good_vs_bad="Higher values indicate stronger diversification benefits. A value close to 1 indicates little or no diversification advantage.",
        caveats="The ratio depends on the quality of the covariance matrix. Poor correlation estimates can overstate diversification benefits.",
        interpret=lambda v: f"{v:.2f}x diversification benefit" if v > 0 else "n/a",
    )
)

register(
    Explanation(
        name="marginal_contribution_to_risk",
        category="metric",
        summary="Measures how much portfolio volatility changes when the allocation to one asset changes slightly.",
        formula="(Cov @ w) / portfolio_volatility",
        how_to_read="Each value represents the incremental volatility impact of increasing one asset's portfolio weight.",
        good_vs_bad="Assets with lower or negative marginal risk contribution can help reduce portfolio risk. Large positive values indicate assets driving portfolio volatility.",
        caveats="Marginal risk contribution depends on current portfolio weights and correlations. It is not the same as standalone asset volatility.",
        interpret=lambda v: f"{v:.2%}",
    )
)

register(
    Explanation(
        name="component_contribution_to_risk",
        category="metric",
        summary="Breaks down total portfolio volatility into the amount of risk contributed by each individual asset position.",
        formula="weight_i * marginal_contribution_to_risk_i",
        how_to_read="Each component shows the portion of total portfolio volatility attributable to one asset. The components add up to total portfolio volatility.",
        good_vs_bad="A balanced risk contribution means no single asset dominates portfolio risk. Concentrated contributions indicate hidden risk concentration.",
        caveats="Risk contribution can be misleading when weights are negative, such as in leveraged or hedged portfolios.",
        interpret=lambda v: f"{v:.2%}",
    )
)

register(
    Explanation(
        name="covariance_matrix",
        category="metric",
        summary="A matrix showing how asset returns move together, including both individual asset volatility and relationships between assets.",
        formula="covariance(returns_i, returns_j)",
        how_to_read="Diagonal values represent individual asset variance. Off-diagonal values represent how two assets move together.",
        good_vs_bad="A stable covariance matrix improves portfolio risk estimation. Unstable covariance estimates can lead to poor portfolio decisions.",
        caveats="Covariance estimates are highly dependent on the sample period, market regime, and data frequency.",
    )
)

register(
    Explanation(
        name="correlation_matrix",
        category="metric",
        summary="A matrix showing the strength and direction of relationships between asset returns, independent of their individual volatility levels.",
        formula="correlation(returns_i, returns_j)",
        how_to_read="Values range from -1 to 1. Positive values indicate assets tend to move together, negative values indicate opposite movement.",
        good_vs_bad="Lower correlations between assets generally improve diversification potential.",
        caveats="Correlation is not constant. Assets that appear diversified historically can become highly correlated during market stress.",
    )
)