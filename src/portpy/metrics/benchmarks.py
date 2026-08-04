"""Benchmark-relative metrics: alpha, correlation, R-squared, capture ratios, batting average."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, MetricResult, register
from portpy.metrics.returns import annualized_return
from portpy.metrics.risk import beta as _beta
from portpy.utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import align_pair, ensure_min_observations, periodic_rate_from_annual

__all__ = [
    "alpha",
    "correlation",
    "r_squared",
    "up_capture_ratio",
    "down_capture_ratio",
    "capture_ratio",
    "batting_average",
]


def alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """Jensen's alpha: annualized excess return unexplained by exposure (beta) to the benchmark."""
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    rf_period = periodic_rate_from_annual(rf, periods_per_year)
    beta_value = _beta(r, b)
    alpha_period = float((r - rf_period).mean() - beta_value * (b - rf_period).mean())
    value = (1.0 + alpha_period) ** periods_per_year - 1.0
    return MetricResult(value, "alpha", unit="%") if as_result else value


def correlation(returns: pd.Series, benchmark: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Pearson correlation coefficient between `returns` and `benchmark`."""
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    value = float(r.corr(b))
    return MetricResult(value, "correlation") if as_result else value


def r_squared(returns: pd.Series, benchmark: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Fraction of the portfolio's variance explained by the benchmark: `correlation(r, b)^2`."""
    value = float(correlation(returns, benchmark) ** 2)
    return MetricResult(value, "r_squared") if as_result else value


def _capture(returns: pd.Series, benchmark: pd.Series, periods_per_year: int) -> float:
    ar = annualized_return(returns, periods_per_year=periods_per_year, geometric=True)
    ab = annualized_return(benchmark, periods_per_year=periods_per_year, geometric=True)
    return ar / ab if ab != 0 else np.nan


def up_capture_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """Ratio of annualized returns during periods when the benchmark was positive."""
    r, b = align_pair(returns, benchmark)
    mask = b > 0
    value = _capture(r[mask], b[mask], periods_per_year) if mask.sum() >= 2 else np.nan
    return MetricResult(value, "up_capture_ratio") if as_result else value


def down_capture_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """Ratio of annualized returns during periods when the benchmark was negative."""
    r, b = align_pair(returns, benchmark)
    mask = b < 0
    value = _capture(r[mask], b[mask], periods_per_year) if mask.sum() >= 2 else np.nan
    return MetricResult(value, "down_capture_ratio") if as_result else value


def capture_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """Overall ratio of annualized returns, unconditional on the benchmark's sign."""
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    value = _capture(r, b, periods_per_year)
    return MetricResult(value, "capture_ratio") if as_result else value


def batting_average(returns: pd.Series, benchmark: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Fraction of periods in which `returns` beat `benchmark`."""
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 1, "returns")
    value = float((r > b).mean())
    return MetricResult(value, "batting_average", unit="%") if as_result else value


register(
    Explanation(
        name="alpha",
        category="metric",
        summary="The annualized return left over after accounting for what the benchmark's movements (scaled by beta) would predict - 'skill' beyond just riding the market.",
        formula="(1 + mean(r - rf) - beta*mean(b - rf))^periods_per_year - 1",
        how_to_read="Positive alpha means the portfolio outperformed what its market exposure alone would predict; negative means it underperformed even after accounting for beta.",
        good_vs_bad="Positive and statistically/practically meaningful (not just a few lucky periods) is good. A small alpha over a short sample is easily noise.",
        caveats="Only as good as the beta/benchmark choice - a mismatched benchmark can make skill look like alpha or vice versa.",
        interpret=lambda v: f"{v:+.1%}/yr" + (" (outperforming what beta alone predicts)" if v > 0 else " (underperforming what beta alone predicts)"),
    )
)

register(
    Explanation(
        name="correlation",
        category="metric",
        summary="How closely the portfolio's returns move together with the benchmark's, linearly, period to period.",
        formula="Pearson correlation coefficient",
        how_to_read="+1 = move in perfect lockstep, -1 = move in perfect opposition, 0 = no linear relationship.",
        good_vs_bad="Depends on intent: high correlation to a benchmark you're trying to track is good; high correlation to something you're trying to diversify away from is bad.",
        interpret=lambda v: f"{v:+.2f}" + (" (strong co-movement)" if abs(v) > 0.7 else " (weak/moderate co-movement)"),
    )
)

register(
    Explanation(
        name="r_squared",
        category="metric",
        summary="The fraction of the portfolio's return variance that can be statistically 'explained' by the benchmark's returns.",
        formula="correlation(r, b)^2",
        how_to_read="0.80 means 80% of the portfolio's variance moves together with the benchmark; the remaining 20% is idiosyncratic.",
        good_vs_bad="For an index-tracking product, higher is better (tight tracking). For an actively differentiated strategy, a lower R-squared can be a feature, not a bug - it means the strategy is doing something different from the benchmark.",
        interpret=lambda v: f"{v:.0%} of variance explained by the benchmark",
    )
)

register(
    Explanation(
        name="up_capture_ratio",
        category="metric",
        summary="How much of the benchmark's annualized gains the portfolio captured, restricted to periods when the benchmark was up.",
        formula="annualized_return(r | b>0) / annualized_return(b | b>0)",
        how_to_read="120% means the portfolio gained 20% more than the benchmark, on average, during up periods.",
        good_vs_bad="Above 100% is desirable (capturing more upside than the benchmark).",
        interpret=lambda v: f"{v:.0%} of benchmark upside captured",
    )
)

register(
    Explanation(
        name="down_capture_ratio",
        category="metric",
        summary="How much of the benchmark's annualized losses the portfolio also experienced, restricted to periods when the benchmark was down.",
        formula="annualized_return(r | b<0) / annualized_return(b | b<0)",
        how_to_read="70% means the portfolio only lost 70% as much as the benchmark, on average, during down periods (a good defensive sign).",
        good_vs_bad="Below 100% is desirable (losing less than the benchmark on the way down). The ideal combination is up_capture > 100% and down_capture < 100%.",
        interpret=lambda v: f"{v:.0%} of benchmark downside captured" + (" (defensive)" if v < 1 else " (amplifies benchmark's downside)"),
    )
)

register(
    Explanation(
        name="capture_ratio",
        category="metric",
        summary="The overall ratio of annualized returns between the portfolio and benchmark, without conditioning on the benchmark's direction.",
        formula="annualized_return(r) / annualized_return(b)",
        how_to_read="Above 1.0 means the portfolio outgrew the benchmark overall, below 1.0 means it lagged.",
        good_vs_bad="Higher is better, but always look at up_capture_ratio and down_capture_ratio too - the same overall ratio can hide very different (defensive vs. aggressive) return profiles.",
        interpret=lambda v: f"{v:.2f}x the benchmark's annualized return",
    )
)

register(
    Explanation(
        name="batting_average",
        category="metric",
        summary="The fraction of periods in which the portfolio beat the benchmark, regardless of by how much.",
        formula="count(r > b) / count(r)",
        how_to_read="0.55 means the portfolio outperformed the benchmark in 55% of periods.",
        good_vs_bad="Above 50% is a simple sign of consistent relative outperformance, but a low batting average with a few huge wins can still beat the benchmark overall - pair with capture_ratio.",
        interpret=lambda v: f"beat the benchmark in {v:.0%} of periods",
    )
)
