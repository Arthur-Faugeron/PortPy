"""Descriptive statistics and shape of the return distribution."""

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
    """Standard descriptive statistics (count, mean, std, min/max, quartiles) plus skew and kurtosis."""
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    base = r.describe()
    extra = pd.Series({"skew": r.skew(), "kurtosis": r.kurtosis()})
    return pd.concat([base, extra])


def normality_test(returns: pd.Series) -> dict:
    """Jarque-Bera test of whether returns look Normally distributed."""
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
    """The `n` best and `n` worst individual periods, each as a Series sorted from most extreme."""
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    return {
        "best": r.sort_values(ascending=False).head(n),
        "worst": r.sort_values(ascending=True).head(n),
    }


def win_rate(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Fraction of periods with a strictly positive return."""
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    value = float((r > 0).mean())
    return MetricResult(value, "win_rate", unit="%") if as_result else value


def win_loss_ratio(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Average winning period divided by the average (absolute) losing period."""
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    wins = r[r > 0]
    losses = r[r < 0]
    value = safe_divide(float(wins.mean()) if len(wins) else 0.0, float(-losses.mean()) if len(losses) else 0.0)
    return MetricResult(value, "win_loss_ratio") if as_result else value


def positive_periods_pct(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Alias of :func:`win_rate`, kept for API compatibility with the common naming convention."""
    r = returns.dropna()
    value = win_rate(r, as_result=False)
    return MetricResult(value, "positive_periods_pct", unit="%") if as_result else value


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Calendar table of compounded returns: rows = years, columns = Jan..Dec, plus a Year total column."""
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
    """Histogram of the return distribution: `(counts, bin_edges)`, straight from `numpy.histogram`."""
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    return np.histogram(r.to_numpy(), bins=bins)


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
        category="chart",
        summary="Compounded return for every calendar month, laid out as a year x month grid (the data behind a monthly-returns heatmap).",
        how_to_read="Each cell is that month's compounded return; the 'Year' column is the full calendar year's compounded return. Read row-by-row to spot seasonal patterns or bad years; column-by-column to spot a consistently weak/strong calendar month.",
        good_vs_bad="More green (positive) than red (negative) cells, with the Year column trending positive, is the visual 'good' pattern.",
    )
)

register(
    Explanation(
        name="return_histogram_data",
        category="chart",
        summary="The raw bin counts and edges describing the shape of the return distribution.",
        how_to_read="A tall, narrow, symmetric hump centered near (or slightly above) zero is 'textbook'. Look for a long left tail (crash risk) or a bimodal shape (regime-switching behavior).",
        good_vs_bad="Not graded directly - use skewness/kurtosis/normality_test for quantitative judgments about the shape shown here.",
    )
)
