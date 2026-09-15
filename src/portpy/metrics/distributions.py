"""
Descriptive statistics and shape of the return distribution.
"""

from __future__ import annotations

import calendar as _calendar

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from portpy.explain import Explanation, MetricResult, register
from portpy.utils.validation import ensure_min_observations, safe_divide

__all__ = [
    "describe",
    "normality_test",
    "best_worst_periods",
    "win_rate",
    "win_loss_ratio",
    "positive_periods_pct",
    "monthly_returns_table",
    "return_histogram_data",
]


def describe(returns: pd.Series) -> pd.Series:
    """
    Standard descriptive statistics (count, mean, std, min/max, quartiles)
    plus skew and kurtosis, using pandas' bias-adjusted Fisher-Pearson
    sample estimators.

    Args:
        returns: Periodic returns.

    Returns:
        A Series indexed by statistic name (count, mean, std, min, 25%,
        50%, 75%, max, skew, kurtosis).
    """
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    base = r.describe()
    extra = pd.Series({"skew": r.skew(), "kurtosis": r.kurtosis()})
    return pd.concat([base, extra])


def normality_test(returns: pd.Series) -> dict:
    """
    Jarque-Bera test of whether returns look Normally distributed.

    Args:
        returns: Periodic returns. Requires at least 8 observations.

    Returns:
        A dict with "statistic" (the JB test statistic), "p_value", and
        "is_normal" (True if p_value > 0.05).
    """
    r = returns.dropna()
    ensure_min_observations(r, 8, "returns")
    stat, p_value = scipy_stats.jarque_bera(r)
    result: dict[str, object] = {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": bool(p_value > 0.05),
    }
    result["_portpy_explain_name"] = "normality_test"
    return result


def best_worst_periods(returns: pd.Series, n: int = 5) -> dict:
    """
    The n best and n worst individual periods, each as a Series sorted from
    most extreme to least extreme.

    Args:
        returns: Periodic returns.
        n: Number of best/worst periods to return.

    Returns:
        A dict with keys "best" and "worst", each a Series of the top-n
        returns.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    return {
        "best": r.sort_values(ascending=False).head(n),
        "worst": r.sort_values(ascending=True).head(n),
    }


def win_rate(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Fraction of periods with a strictly positive return.

    Args:
        returns: Periodic returns.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The win rate, or a `MetricResult` wrapping it.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    value = float((r > 0).mean())
    return MetricResult(value, "win_rate", unit="%") if as_result else value


def win_loss_ratio(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Average winning period's return divided by the average (absolute)
    losing period's return. Zero-return periods count toward neither bucket.

    Args:
        returns: Periodic returns.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The win/loss ratio, or a `MetricResult` wrapping it. 0.0 if there
        are no wins or no losses.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    wins = r[r > 0]
    losses = r[r < 0]
    value = safe_divide(float(wins.mean()) if len(wins) else 0.0, float(-losses.mean()) if len(losses) else 0.0)
    return MetricResult(value, "win_loss_ratio") if as_result else value


def positive_periods_pct(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Alias of `win_rate`, kept for API compatibility with the common naming
    convention.

    Args:
        returns: Periodic returns.
        as_result: If True, return a `MetricResult` instead of a plain float.

    Returns:
        The fraction of positive periods, or a `MetricResult` wrapping it.
    """
    r = returns.dropna()
    value = win_rate(r, as_result=False)
    return MetricResult(value, "positive_periods_pct", unit="%") if as_result else value


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Calendar table of compounded returns: rows are years, columns are
    Jan-Dec, plus a Year total column.

    Args:
        returns: Periodic returns, finer than monthly (e.g. daily/weekly).

    Returns:
        A DataFrame indexed by year with one column per calendar month
        (compounded return) plus a "Year" column (compounded annual return).
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")

    monthly = (1.0 + r).resample("ME").prod() - 1.0
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    table = df.pivot(index="year", columns="month", values="ret")
    table = table.reindex(columns=range(1, 13))
    table.columns = [_calendar.month_abbr[m] for m in range(1, 13)]

    annual = (1.0 + r).groupby(r.index.year).prod() - 1.0
    table["Year"] = annual
    table.index.name = "year"
    return table


def return_histogram_data(returns: pd.Series, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Histogram of the return distribution, computed via `numpy.histogram`.

    Args:
        returns: Periodic returns.
        bins: Number of histogram bins.

    Returns:
        A tuple of (counts, bin_edges).
    """
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    return np.histogram(r.to_numpy(), bins=bins)


register(
    Explanation(
        name="describe",
        category="metric",
        summary="A descriptive summary of the return distribution, including central tendency, dispersion, and tail behavior.",
        formula="pandas.Series.describe() + skewness/kurtosis",
        how_to_read="Use this as a quick sanity check for the return sample: typical values, spread, and extreme observations.",
        good_vs_bad="A well-behaved distribution should show reasonable center and spread, but there is no universal 'good' threshold for descriptive stats alone.",
        caveats="skew/kurtosis are pandas' bias-adjusted Fisher-Pearson sample estimators (G1/G2), not plain population moment-ratio formulas - see portpy.metrics.risk.skewness/kurtosis for the exact formulas and a fuller caveat.",
    )
)

register(
    Explanation(
        name="win_rate",
        category="metric",
        summary="The fraction of periods (days/weeks/months) with a positive return.",
        formula="count(r > 0) / count(r)",
        how_to_read="0.55 means 55% of periods were profitable.",
        good_vs_bad=(
            "Above 50% is intuitively 'good', but a high win rate with small wins and rare huge "
            "losses can still be a bad strategy overall (see win_loss_ratio) - never judge win rate alone."
        ),
        caveats="Exact-zero-return periods count against win_rate (the check is strict > 0) but are excluded from both buckets in win_loss_ratio, so the two aren't perfectly complementary once zero-return periods exist.",
        interpret=lambda v: f"{v:.1%} of periods were positive",
    )
)

register(
    Explanation(
        name="win_loss_ratio",
        category="metric",
        summary="How big the average win is relative to the average loss, in magnitude.",
        formula="mean(r for r > 0) / |mean(r for r < 0)|",
        how_to_read="A ratio of 2.0 means winning periods are, on average, twice as large as losing periods.",
        good_vs_bad=(
            "Read together with win_rate: a strategy can be profitable with a win rate below 50% if "
            "this ratio is high enough (small frequent losses, larger rare wins), and vice versa."
        ),
        caveats="Exact-zero-return periods are excluded from both the win and loss buckets, unlike win_rate where they count against it - not perfectly complementary once zeros are present. Returns a degenerate 0.0 (via safe_divide) when there are zero wins or zero losses; that's not a literal 'bad' ratio.",
        interpret=lambda v: f"{v:.2f}" + (" (average win smaller than average loss - relies on high win rate)" if v < 1 else " (average win bigger than average loss)"),
    )
)

register(
    Explanation(
        name="positive_periods_pct",
        category="metric",
        summary="Same as win_rate: the fraction of periods with a positive return.",
        how_to_read="See win_rate.",
        good_vs_bad="See win_rate.",
        interpret=lambda v: f"{v:.1%} of periods were positive",
    )
)

register(
    Explanation(
        name="normality_test",
        category="metric",
        summary="Jarque-Bera test for whether the return distribution's skewness and kurtosis match a Normal distribution.",
        formula="statistic based on sample skewness and excess kurtosis; p-value from a chi-squared(2) reference distribution",
        how_to_read="A small p-value (conventionally < 0.05) means the data significantly deviates from Normal - which is the norm, not the exception, for real returns.",
        good_vs_bad=(
            "Neither outcome is 'good' or 'bad' on its own - it's a diagnostic. Rejecting normality "
            "is a signal to prefer historical/Cornish-Fisher VaR over parametric (Normal-assumption) VaR."
        ),
        caveats=(
            "The minimum-observations floor (8) is far below any meaningful sample size for this test "
            "(textbook guidance is more like N=30-50, and even that's shaky for real financial returns). "
            "Also, failing to reject normality is not evidence of normality - real daily returns almost "
            "always reject normality at scale, which is itself expected, not a data problem."
        ),
        interpret=lambda d: (
            f"p={d['p_value']:.4f} -> "
            + ("reject Normality (fat tails/skew likely - use historical or Cornish-Fisher VaR)"
               if not d["is_normal"] else "fail to reject Normality (Gaussian risk models are more defensible here)")
        ),
    )
)

register(
    Explanation(
        name="best_worst_periods",
        category="metric",
        summary="The single best and worst individual return observations in the sample.",
        how_to_read="Useful for a gut check: do the extremes correspond to known market events, or do they look like data errors?",
        good_vs_bad="Not a graded metric - a diagnostic/sanity-check tool.",
    )
)

register(
    Explanation(
        name="monthly_returns_table",
        category="function",
        summary="Compounded return for every calendar month, laid out as a year x month grid (the data behind a monthly-returns heatmap).",
        how_to_read="Each cell is that month's compounded return; the 'Year' column is the full calendar year's compounded return. Read row-by-row to spot seasonal patterns or bad years; column-by-column to spot a consistently weak/strong calendar month.",
        good_vs_bad="More green (positive) than red (negative) cells, with the Year column trending positive, is the visual 'good' pattern.",
        caveats="Assumes input finer than monthly (daily/weekly). Passing already-monthly (or lower-frequency) returns won't error, but produces a distorted/degenerate table.",
    )
)

register(
    Explanation(
        name="return_histogram_data",
        category="function",
        summary="The raw bin counts and edges describing the shape of the return distribution.",
        how_to_read="A tall, narrow, symmetric hump centered near (or slightly above) zero is 'textbook'. Look for a long left tail (crash risk) or a bimodal shape (regime-switching behavior).",
        good_vs_bad="Not graded directly - use skewness/kurtosis/normality_test for quantitative judgments about the shape shown here.",
    )
)
