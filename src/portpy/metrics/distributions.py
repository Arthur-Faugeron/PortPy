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
    Standard descriptive statistics (count, mean, std, min/max, quartiles) plus skew and kurtosis.
    """
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    base = r.describe()
    extra = pd.Series({"skew": r.skew(), "kurtosis": r.kurtosis()})
    return pd.concat([base, extra])


def normality_test(returns: pd.Series) -> dict:
    """
    Jarque-Bera test of whether returns look Normally distributed.
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
    The n best and n worst individual periods, each as a Series sorted from most extreme.
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
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    value = float((r > 0).mean())
    return MetricResult(value, "win_rate", unit="%") if as_result else value


def win_loss_ratio(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Average winning period divided by the average (absolute) losing period.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    wins = r[r > 0]
    losses = r[r < 0]
    value = safe_divide(float(wins.mean()) if len(wins) else 0.0, float(-losses.mean()) if len(losses) else 0.0)
    return MetricResult(value, "win_loss_ratio") if as_result else value


def positive_periods_pct(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Alias of :func:win_rate, kept for API compatibility with the common naming convention.
    """
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
    """
    Histogram of the return distribution: (counts, bin_edges), straight from numpy.histogram.
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



register(
    Explanation(
        name="alpha",
        category="metric",
        summary="The annualized excess return generated by the portfolio after accounting for the return expected from its benchmark exposure through beta.",
        formula="(1 + mean(excess_return) - beta * mean(benchmark_excess_return)) ** periods_per_year - 1",
        how_to_read="Positive alpha means the portfolio produced returns beyond what its benchmark exposure would explain. Negative alpha means underperformance after adjusting for market exposure.",
        good_vs_bad="Positive and persistent alpha is desirable because it indicates value added beyond benchmark exposure.",
        caveats="Alpha depends heavily on the chosen benchmark and beta estimate. A poor benchmark can make unrelated returns appear as alpha.",
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
        caveats="R-squared does not measure whether returns are positive or negative. A portfolio can have high R-squared and still perform poorly.",
        interpret=lambda v: f"{v:.0%} of variance explained by benchmark",
    )
)

register(
    Explanation(
        name="up_capture_ratio",
        category="metric",
        summary="Measures how much of the benchmark's positive performance the portfolio captures during periods when the benchmark rises.",
        formula="annualized_return(portfolio_returns_when_benchmark_positive) / annualized_return(benchmark_returns_when_positive)",
        how_to_read="A value above 1 means the portfolio gains more than the benchmark during positive benchmark periods.",
        good_vs_bad="Above 1 is generally desirable because it indicates stronger participation in market gains.",
        caveats="Requires enough positive benchmark periods. Results can be distorted by a small number of strong market moves.",
        interpret=lambda v: f"{v:.0%} of benchmark upside captured",
    )
)

register(
    Explanation(
        name="down_capture_ratio",
        category="metric",
        summary="Measures how much of the benchmark's negative performance the portfolio experiences during periods when the benchmark falls.",
        formula="annualized_return(portfolio_returns_when_benchmark_negative) / annualized_return(benchmark_returns_when_negative)",
        how_to_read="A value below 1 means the portfolio loses less than the benchmark during declining periods.",
        good_vs_bad="Below 1 is generally desirable because it indicates downside protection.",
        caveats="The ratio can behave unexpectedly when benchmark losses are small or when the sample contains few negative periods.",
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
        caveats="A portfolio can have a low batting average and still outperform if a small number of gains are large enough. Combine with return-based metrics.",
        interpret=lambda v: f"outperformed benchmark in {v:.0%} of periods",
    )
)
