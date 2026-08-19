"""
Benchmark-relative metrics: alpha, correlation, R-squared, capture ratios, batting average.
"""

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
    """
    Jensen's alpha: the portfolio's annualized excess return unexplained by
    its beta exposure to the benchmark.

    Args:
        returns: Periodic portfolio returns.
        benchmark: Periodic benchmark returns, aligned to `returns`.
        rf: Annual risk-free rate used to compute excess returns.
        periods_per_year: Number of return periods per year, used for
            annualization.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        Annualized alpha, or a `MetricResult` wrapping it.
    """
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    rf_period = periodic_rate_from_annual(rf, periods_per_year)
    beta_value = _beta(r, b)
    alpha_period = float((r - rf_period).mean() - beta_value * (b - rf_period).mean())
    value = (1.0 + alpha_period) ** periods_per_year - 1.0
    return MetricResult(value, "alpha", unit="%") if as_result else value


def correlation(returns: pd.Series, benchmark: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Pearson correlation coefficient between returns and benchmark.

    Args:
        returns: Periodic portfolio returns.
        benchmark: Periodic benchmark returns, aligned to `returns`.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The correlation coefficient, or a `MetricResult` wrapping it.
    """
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    value = float(r.corr(b))
    return MetricResult(value, "correlation") if as_result else value


def r_squared(returns: pd.Series, benchmark: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Fraction of the portfolio's return variance explained by the benchmark.

    Args:
        returns: Periodic portfolio returns.
        benchmark: Periodic benchmark returns, aligned to `returns`.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        R-squared (correlation squared), or a `MetricResult` wrapping it.
    """
    value = float(correlation(returns, benchmark) ** 2)
    return MetricResult(value, "r_squared") if as_result else value


def _capture(returns: pd.Series, benchmark: pd.Series, periods_per_year: int) -> float:
    """
    Ratio of annualized (geometric) portfolio return to annualized benchmark
    return, over the full contiguous series.

    Returns:
        The ratio, or NaN if the annualized benchmark return is zero.
    """
    ar = annualized_return(returns, periods_per_year=periods_per_year, geometric=True)
    ab = annualized_return(benchmark, periods_per_year=periods_per_year, geometric=True)
    return ar / ab if ab != 0 else np.nan


def _capture_uncontiguous(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    Ratio of compounded (but not annualized) sub-period returns. Used for
    up/down capture, where masking to up-only or down-only periods makes the
    sub-period non-contiguous, so annualization does not apply.

    Returns:
        The ratio, or NaN if the compounded benchmark return is zero.
    """
    total_r = float((1.0 + returns).prod() - 1.0)
    total_b = float((1.0 + benchmark).prod() - 1.0)
    return total_r / total_b if total_b != 0 else np.nan


def up_capture_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Ratio of compounded returns during periods when the benchmark was positive.

    Args:
        returns: Periodic portfolio returns.
        benchmark: Periodic benchmark returns, aligned to `returns`.
        periods_per_year: Unused; accepted only for signature consistency
            with `down_capture_ratio`/`capture_ratio`.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The up-capture ratio, or a `MetricResult` wrapping it. NaN if fewer
        than 2 periods have a positive benchmark return.
    """
    r, b = align_pair(returns, benchmark)
    mask = b > 0
    value = _capture_uncontiguous(r[mask], b[mask]) if mask.sum() >= 2 else np.nan
    return MetricResult(value, "up_capture_ratio") if as_result else value


def down_capture_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Ratio of compounded returns during periods when the benchmark was negative.

    Args:
        returns: Periodic portfolio returns.
        benchmark: Periodic benchmark returns, aligned to `returns`.
        periods_per_year: Unused; accepted only for signature consistency
            with `up_capture_ratio`/`capture_ratio`.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The down-capture ratio, or a `MetricResult` wrapping it. NaN if fewer
        than 2 periods have a negative benchmark return.
    """
    r, b = align_pair(returns, benchmark)
    mask = b < 0
    value = _capture_uncontiguous(r[mask], b[mask]) if mask.sum() >= 2 else np.nan
    return MetricResult(value, "down_capture_ratio") if as_result else value


def capture_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Ratio of the portfolio's annualized return to the benchmark's annualized
    return, over the full contiguous series (unconditional on the
    benchmark's sign).

    Args:
        returns: Periodic portfolio returns.
        benchmark: Periodic benchmark returns, aligned to `returns`.
        periods_per_year: Number of return periods per year, used for
            annualization.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The capture ratio, or a `MetricResult` wrapping it.
    """
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    value = _capture(r, b, periods_per_year)
    return MetricResult(value, "capture_ratio") if as_result else value


def batting_average(returns: pd.Series, benchmark: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Fraction of periods in which the portfolio's return strictly exceeded
    the benchmark's return.

    Args:
        returns: Periodic portfolio returns.
        benchmark: Periodic benchmark returns, aligned to `returns`.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The batting average, or a `MetricResult` wrapping it.
    """
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 1, "returns")
    value = float((r > b).mean())
    return MetricResult(value, "batting_average", unit="%") if as_result else value


register(
    Explanation(
        name="alpha",
        category="metric",
        summary="The annualized excess return generated by the portfolio after accounting for the return expected from its benchmark exposure through beta.",
        formula="(1 + mean(excess_return) - beta * mean(benchmark_excess_return)) ** periods_per_year - 1",
        how_to_read="Positive alpha means the portfolio produced returns beyond what its benchmark exposure would explain. Negative alpha means underperformance after adjusting for market exposure.",
        good_vs_bad="Positive and persistent alpha is desirable because it indicates value added beyond benchmark exposure.",
        caveats="Alpha depends heavily on the chosen benchmark and beta estimate. A poor benchmark can make unrelated returns appear as alpha. The per-period CAPM residual is annualized by geometric compounding, a defensible but not the only convention - see the docstring.",
        interpret=lambda v: f"{v:+.1%}/yr" + (" (positive excess performance)" if v > 0 else " (negative excess performance)"),
    )
)

register(
    Explanation(
        name="correlation",
        category="metric",
        summary="Measures the linear relationship between portfolio returns and benchmark returns, showing how closely they move together.",
        formula="covariance(returns, benchmark) / (std(returns) * std(benchmark))",
        how_to_read="Values close to 1 indicate the portfolio usually moves with the benchmark. Values close to -1 indicate opposite movement. Values near 0 indicate weak linear relationship.",
        good_vs_bad="High correlation is desirable for benchmark-tracking strategies. Lower correlation is preferable when seeking diversification from the benchmark.",
        caveats="Correlation only measures linear relationships and can change significantly during different market environments.",
        interpret=lambda v: f"{v:+.2f}" + (" (strong relationship)" if abs(v) > 0.7 else " (weak/moderate relationship)"),
    )
)

register(
    Explanation(
        name="r_squared",
        category="metric",
        summary="Measures how much of the portfolio return variability can be statistically explained by benchmark movements.",
        formula="correlation(returns, benchmark) ** 2",
        how_to_read="A value of 0.80 means 80% of return variation is associated with benchmark movements, while 20% comes from other sources.",
        good_vs_bad="Higher values indicate stronger benchmark dependence. This is useful for index tracking but may be undesirable for actively differentiated strategies.",
        caveats="R-squared does not measure whether returns are positive or negative. A portfolio can have high R-squared and still perform poorly. Single-benchmark only.",
        interpret=lambda v: f"{v:.0%} of variance explained by benchmark",
    )
)

register(
    Explanation(
        name="up_capture_ratio",
        category="metric",
        summary="Measures how much of the benchmark's positive performance the portfolio captures during periods when the benchmark rises.",
        formula="compounded_return(portfolio_returns_when_benchmark_positive) / compounded_return(benchmark_returns_when_positive)  [not re-annualized]",
        how_to_read="A value above 1 means the portfolio gains more than the benchmark during positive benchmark periods.",
        good_vs_bad="Above 1 is generally desirable because it indicates stronger participation in market gains.",
        caveats=(
            "Requires enough positive benchmark periods. Results can be distorted by a small number of "
            "strong market moves. The up-day subsample is non-contiguous, so this compares compounded "
            "sub-period returns directly rather than re-annualizing them (re-annualizing a filtered, "
            "non-contiguous subsample is not statistically valid - there's no real 'years elapsed' to "
            "raise a growth factor to). Note this means the value can differ from empyrical's up_capture, "
            "which does re-annualize the subsample via annual_return() - a known, documented methodology "
            "difference, not a bug."
        ),
        interpret=lambda v: f"{v:.0%} of benchmark upside captured",
    )
)

register(
    Explanation(
        name="down_capture_ratio",
        category="metric",
        summary="Measures how much of the benchmark's negative performance the portfolio experiences during periods when the benchmark falls.",
        formula="compounded_return(portfolio_returns_when_benchmark_negative) / compounded_return(benchmark_returns_when_negative)  [not re-annualized]",
        how_to_read="A value below 1 means the portfolio loses less than the benchmark during declining periods.",
        good_vs_bad="Below 1 is generally desirable because it indicates downside protection.",
        caveats=(
            "The ratio can behave unexpectedly when benchmark losses are small or when the sample "
            "contains few negative periods. Compares compounded sub-period returns directly rather than "
            "re-annualizing them - see up_capture_ratio's caveat for why, and for the resulting known "
            "difference from empyrical's down_capture."
        ),
        interpret=lambda v: f"{v:.0%} of benchmark downside captured" + (" (defensive)" if v < 1 else " (higher downside exposure)"),
    )
)

register(
    Explanation(
        name="capture_ratio",
        category="metric",
        summary="Compares the portfolio's overall annualized return against the benchmark's annualized return without separating positive and negative periods.",
        formula="annualized_return(portfolio) / annualized_return(benchmark)",
        how_to_read="A value above 1 means the portfolio produced more annualized return than the benchmark.",
        good_vs_bad="Higher values indicate stronger relative performance, but they should be interpreted together with up and down capture ratios.",
        caveats="A single ratio hides the path taken to achieve returns. Two portfolios can have the same capture ratio with very different risk profiles.",
        interpret=lambda v: f"{v:.2f}x benchmark return",
    )
)

register(
    Explanation(
        name="batting_average",
        category="metric",
        summary="Measures the percentage of periods where the portfolio outperformed the benchmark.",
        formula="count(portfolio_return > benchmark_return) / count(periods)",
        how_to_read="A value of 0.55 means the portfolio beat the benchmark in 55% of observed periods.",
        good_vs_bad="A higher batting average indicates more frequent relative wins, but it does not measure the size of those wins or losses.",
        caveats="A portfolio can have a low batting average and still outperform if a small number of gains are large enough. Combine with return-based metrics. Uses a strict >, so exact ties don't count as a win.",
        interpret=lambda v: f"outperformed benchmark in {v:.0%} of periods",
    )
)