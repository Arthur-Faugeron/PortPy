"""
Risk (dispersion / tail-risk) metrics.

rf, mar (minimum acceptable return), and similar rate parameters are
annual rates throughout PortPy, de-annualized internally via geometric
compounding (utils.validation.periodic_rate_from_annual). This differs from
empyrical, whose equivalent parameters (risk_free, required_return) are
already-periodic rates. See the validation tests for a worked comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from portpy.explain import Explanation, MetricResult, register
from portpy.utils.constants import DEFAULT_CONFIDENCE_LEVEL, DEFAULT_MAR, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import (
    align_pair,
    ensure_min_observations,
    periodic_rate_from_annual,
    validate_confidence,
)

__all__ = [
    "variance",
    "volatility",
    "downside_deviation",
    "semi_variance",
    "value_at_risk",
    "conditional_var",
    "tail_ratio",
    "skewness",
    "kurtosis",
    "ulcer_index",
    "pain_index",
    "beta",
    "tracking_error",
]

_VAR_METHODS = ("historical", "parametric", "cornish_fisher")


def variance(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Sample variance of per-period returns (ddof=1, not annualized).

    Args:
        returns: Per-period returns.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The sample variance, in squared-return units.
    """
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    value = float(r.var(ddof=1))
    return MetricResult(value, "variance") if as_result else value


def volatility(
    returns: pd.Series,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Standard deviation of returns (ddof=1), annualized by sqrt(periods_per_year) by default.

    Args:
        returns: Per-period returns.
        annualized: If True (default), scale by sqrt(periods_per_year).
        periods_per_year: Number of periods in a year, used to annualize.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The (optionally annualized) standard deviation of returns.
    """
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    value = float(r.std(ddof=1))
    if annualized:
        value *= np.sqrt(periods_per_year)
    return MetricResult(value, "volatility", unit="%/yr" if annualized else "%") if as_result else value


def downside_deviation(
    returns: pd.Series,
    mar: float = DEFAULT_MAR,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Root-mean-square of returns falling short of `mar`, dividing by the full sample size N.

    Args:
        returns: Per-period returns.
        mar: Minimum acceptable return, as an annual rate. Defaults to
            DEFAULT_MAR.
        annualized: If True (default), scale by sqrt(periods_per_year).
        periods_per_year: Number of periods in a year, used to de-annualize
            `mar` and to annualize the result.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The (optionally annualized) downside deviation.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    mar_period = periodic_rate_from_annual(mar, periods_per_year)
    shortfall = np.minimum(r - mar_period, 0.0)
    value = float(np.sqrt(np.mean(np.square(shortfall))))
    if annualized:
        value *= np.sqrt(periods_per_year)
    return MetricResult(value, "downside_deviation", unit="%/yr" if annualized else "%") if as_result else value


def semi_variance(
    returns: pd.Series,
    mar: float = DEFAULT_MAR,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Mean squared shortfall below `mar`, dividing by the full sample size N.

    Args:
        returns: Per-period returns.
        mar: Minimum acceptable return, as an annual rate. Defaults to
            DEFAULT_MAR.
        periods_per_year: Number of periods in a year, used to de-annualize `mar`.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The mean squared shortfall, in squared-return units.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    mar_period = periodic_rate_from_annual(mar, periods_per_year)
    shortfall = np.minimum(r - mar_period, 0.0)
    value = float(np.mean(np.square(shortfall)))
    return MetricResult(value, "semi_variance") if as_result else value


def value_at_risk(
    returns: pd.Series,
    method: str = "historical",
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Per-period loss threshold exceeded only (1 - confidence) of the time.

    Reported in return units and signed like a return (a negative number
    means a loss); e.g. -0.03 at 95% confidence means "there's a 5% chance of
    losing more than 3% in a single period."

    Args:
        returns: Per-period returns.
        method: One of "historical" (empirical percentile), "parametric"
            (assumes Normally-distributed returns), or "cornish_fisher"
            (parametric VaR adjusted for the sample's skewness and kurtosis).
        confidence: Confidence level in (0, 1), e.g. 0.95. Defaults to
            DEFAULT_CONFIDENCE_LEVEL.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The per-period VaR at the given confidence level.

    Raises:
        ValueError: If `method` is not one of the recognized methods.
    """
    validate_confidence(confidence)
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    alpha = 1.0 - confidence

    if method == "historical":
        value = float(np.percentile(r, 100 * alpha))
    elif method == "parametric":
        z = scipy_stats.norm.ppf(alpha)
        value = float(r.mean() + z * r.std(ddof=1))
    elif method == "cornish_fisher":
        z = scipy_stats.norm.ppf(alpha)
        s = scipy_stats.skew(r)
        k = scipy_stats.kurtosis(r)  # excess kurtosis
        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )
        value = float(r.mean() + z_cf * r.std(ddof=1))
    else:
        raise ValueError(f"method must be one of {_VAR_METHODS}, got {method!r}.")

    return MetricResult(value, "value_at_risk", unit="%") if as_result else value


def conditional_var(
    returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Conditional VaR / Expected Shortfall: the average loss in the worst
    (1 - confidence) fraction of periods.

    Always at least as severe as value_at_risk at the same confidence. It
    answers "given that we're in the bad tail, how bad is it on average?"
    rather than just where the tail starts.

    Args:
        returns: Per-period returns.
        confidence: Confidence level in (0, 1), e.g. 0.95. Defaults to
            DEFAULT_CONFIDENCE_LEVEL.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The average per-period loss in the tail beyond the VaR cutoff.
    """
    validate_confidence(confidence)
    r = returns.dropna().to_numpy()
    ensure_min_observations(r, 2, "returns")
    alpha = 1.0 - confidence
    cutoff_index = int((len(r) - 1) * alpha)
    worst = np.partition(r, cutoff_index)[: cutoff_index + 1]
    value = float(np.mean(worst))
    return MetricResult(value, "conditional_var", unit="%") if as_result else value


def tail_ratio(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Ratio of the size of the right tail (95th percentile) to the left tail (5th percentile).

    Args:
        returns: Per-period returns.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The tail ratio.
    """
    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    value = float(np.abs(np.percentile(r, 95)) / np.abs(np.percentile(r, 5)))
    return MetricResult(value, "tail_ratio") if as_result else value


def skewness(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Sample skewness (adjusted Fisher-Pearson) of the return distribution.

    Args:
        returns: Per-period returns.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The sample skewness.
    """
    r = returns.dropna()
    ensure_min_observations(r, 3, "returns")
    value = float(r.skew())
    return MetricResult(value, "skewness") if as_result else value


def kurtosis(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Sample excess kurtosis (0 = Normal-like tails; positive = fatter tails than Normal).

    Args:
        returns: Per-period returns.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The sample excess kurtosis.
    """
    r = returns.dropna()
    ensure_min_observations(r, 4, "returns")
    value = float(r.kurtosis())
    return MetricResult(value, "kurtosis") if as_result else value


def ulcer_index(prices: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Root-mean-square of percentage drawdowns. Penalizes both deep and prolonged drawdowns.

    Args:
        prices: Price series.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The ulcer index.
    """
    from portpy.metrics.drawdowns import drawdown_series

    dd = drawdown_series(prices)
    value = float(np.sqrt(np.mean(np.square(dd))))
    return MetricResult(value, "ulcer_index") if as_result else value


def pain_index(prices: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Mean absolute drawdown across the whole period (a.k.a. average drawdown).

    Args:
        prices: Price series.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The pain index.
    """
    from portpy.metrics.drawdowns import drawdown_series

    dd = drawdown_series(prices)
    value = float(np.mean(np.abs(dd)))
    return MetricResult(value, "pain_index") if as_result else value


def beta(returns: pd.Series, benchmark: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Sensitivity of returns to a benchmark.

    Formula: Cov(r, b) / Var(b)

    Args:
        returns: Per-period returns.
        benchmark: Per-period benchmark returns.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The beta coefficient.
    """
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    cov = float(np.cov(r, b, ddof=1)[0, 1])
    var_b = float(np.var(b, ddof=1))
    value = cov / var_b if var_b > 1e-30 else np.nan
    return MetricResult(value, "beta") if as_result else value


def tracking_error(
    returns: pd.Series,
    benchmark: pd.Series,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Standard deviation of the return difference (returns - benchmark), i.e. active risk.

    Args:
        returns: Per-period returns.
        benchmark: Per-period benchmark returns.
        annualized: If True (default), scale by sqrt(periods_per_year).
        periods_per_year: Number of periods in a year, used to annualize.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The (optionally annualized) tracking error.
    """
    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    diff = r - b
    value = float(diff.std(ddof=1))
    if annualized:
        value *= np.sqrt(periods_per_year)
    return MetricResult(value, "tracking_error") if as_result else value


register(
    Explanation(
        name="variance",
        category="metric",
        summary="The average squared deviation of returns from their mean - the textbook measure of dispersion.",
        formula="mean((r - mean(r))^2), ddof=1",
        how_to_read="In squared-return units, so it's hard to interpret directly - use volatility (its square root) instead for anything intuitive.",
        good_vs_bad="Lower means more consistent returns; there's no universal good/bad cutoff on its own.",
        interpret=lambda v: f"{v:.6f} (in squared-return units; sqrt gives volatility = {np.sqrt(v):.2%})",
    )
)

register(
    Explanation(
        name="volatility",
        category="metric",
        summary="The standard deviation of returns - the most common measure of how much an asset's returns bounce around.",
        formula="std(r, ddof=1) * sqrt(periods_per_year)  [if annualized]",
        how_to_read="Annualized volatility of 0.20 means returns typically swing about +/-20%/year around the average.",
        good_vs_bad=(
            "Lower is 'safer' in the sense of smoother returns, but volatility alone says nothing "
            "about direction - a fast-rising asset can be just as volatile as a falling one. Always "
            "read it alongside the return."
        ),
        caveats=(
            "Penalizes upside and downside swings equally, unlike downside_deviation or semi_variance. "
            "The sqrt(periods_per_year) annualization also assumes i.i.d., serially uncorrelated returns; "
            "positive autocorrelation (e.g. illiquid or infrequently-priced assets) means the naive "
            "annualized figure understates true risk (Lo, 2002). The same caveat applies to sharpe_ratio, "
            "sortino_ratio, information_ratio, and tracking_error, which all use the same scaling."
        ),
        interpret=lambda v: f"{v:.1%}/yr" + (" (high - equity-like or more volatile)" if v > 0.25 else (" (low - bond-like)" if v < 0.08 else " (moderate)")),
    )
)

register(
    Explanation(
        name="downside_deviation",
        category="metric",
        summary="Like volatility, but only counts returns that fall short of a minimum acceptable return (MAR) - upside swings aren't penalized.",
        formula="sqrt(mean(min(r - mar, 0)^2)) * sqrt(periods_per_year)  [if annualized]",
        how_to_read="Same units and scale as volatility, but will always be <= volatility for the same series since only bad periods count.",
        good_vs_bad="Lower is better. Compare it to volatility: a downside deviation much smaller than volatility means most of the swings are on the upside.",
        caveats="Divides by the full sample size N (Sortino's original convention). sqrt(semi_variance(...)) uses the same convention and matches this function exactly at annualized=False.",
        interpret=lambda v: f"{v:.1%}/yr of downside risk",
    )
)

register(
    Explanation(
        name="semi_variance",
        category="metric",
        summary="Mean squared shortfall below a minimum acceptable return (MAR), averaged over the full sample - Markowitz's original semi-variance.",
        formula="mean(min(r - mar, 0)^2)  [averaged over all N periods, not just the shortfalls]",
        how_to_read="In squared-return units; compare across portfolios rather than reading in isolation. sqrt(semi_variance(r, mar, annualized=False)) equals downside_deviation(r, mar, annualized=False) exactly.",
        good_vs_bad="Lower is better (less/milder downside).",
        caveats="If nothing fell below the MAR, this is exactly 0 - always check how many periods were below threshold before trusting a 0.",
        interpret=lambda v: f"{v:.6f} squared-return units below the MAR",
    )
)

register(
    Explanation(
        name="value_at_risk",
        category="metric",
        summary="The loss threshold you'd only expect to breach a small fraction of the time (e.g. 5% of periods at 95% confidence).",
        formula="historical: percentile(r, 100*(1-confidence))  |  parametric: mean + z*std  |  cornish_fisher: mean + z_cf*std (z adjusted for sample skew/kurtosis)",
        how_to_read="A 95% VaR of -0.03 means: in 95% of periods, you did NOT lose more than 3%. It does NOT say anything about how bad the worst 5% get.",
        good_vs_bad="Closer to zero (less negative) is better/safer. Only comparable across series measured at the same confidence level and frequency.",
        caveats=(
            "VaR ignores everything beyond the threshold - two portfolios can have identical VaR "
            "and wildly different tail risk. Use conditional_var (CVaR) alongside it for that reason. "
            "parametric assumes Normally-distributed returns (understates fat tails); cornish_fisher "
            "corrects for sample skew/kurtosis but is a local approximation that degrades - and can "
            "become non-monotonic - at high skew/kurtosis or very extreme confidence levels. All three "
            "methods are single-period estimates; none scale to a multi-period holding horizon."
        ),
        interpret=lambda v: f"{v:.2%} - expect a worse-than-this loss only in the excluded tail probability",
    )
)

register(
    Explanation(
        name="conditional_var",
        category="metric",
        summary="Also called Expected Shortfall (ES/CVaR): the average loss given that you're already in the bad tail defined by VaR.",
        formula="mean(r for r <= VaR cutoff)",
        how_to_read="A 95% CVaR of -0.05 means: on the worst 5% of periods, the average loss was 5%.",
        good_vs_bad="Closer to zero is better/safer. Always at least as negative as VaR at the same confidence - a large gap between VaR and CVaR signals a fat, dangerous tail.",
        caveats="Still a historical/empirical estimate - a tail event worse than anything in the sample isn't reflected.",
        interpret=lambda v: f"{v:.2%} average loss in the worst-case tail",
    )
)

register(
    Explanation(
        name="tail_ratio",
        category="metric",
        summary="Compares the size of big gains (95th percentile) to big losses (5th percentile).",
        formula="|percentile(r, 95)| / |percentile(r, 5)|",
        how_to_read="A ratio of 1.0 means big gains and big losses are similarly sized; below 1.0 means losses in the tail are bigger than gains.",
        good_vs_bad="Above 1.0 is generally favorable (upside tail bigger than downside tail); well below 1.0 (e.g. 0.25) signals a strategy prone to occasional large losses.",
        interpret=lambda v: f"{v:.2f}" + (" (fat downside tail relative to upside)" if v < 0.8 else (" (upside tail bigger than downside)" if v > 1.2 else " (roughly symmetric tails)")),
    )
)

register(
    Explanation(
        name="skewness",
        category="metric",
        summary="Measures asymmetry in the return distribution: whether extreme moves tend to be on the upside or downside.",
        formula="adjusted Fisher-Pearson sample skewness",
        how_to_read="Positive skew = occasional large gains with frequent small losses; negative skew = occasional large losses with frequent small gains.",
        good_vs_bad="Positive is generally considered more desirable for a long-only investor (limited, frequent losses; occasional big wins), though many trend/momentum strategies deliberately run negative skew in exchange for a higher hit rate.",
        caveats="Very sensitive to sample size and outliers in short return histories. This is the bias-adjusted sample estimator, not the raw population moment ratio - the two only converge asymptotically, so treat it as noisy at typical financial sample sizes (same caveat applies to kurtosis).",
        interpret=lambda v: f"{v:+.2f}" + (" (negative skew - watch for rare large losses)" if v < -0.5 else (" (positive skew - rare large gains)" if v > 0.5 else " (roughly symmetric)")),
    )
)

register(
    Explanation(
        name="kurtosis",
        category="metric",
        summary="Measures how fat the tails of the return distribution are relative to a Normal distribution.",
        formula="excess kurtosis = kurtosis - 3 (so 0 = Normal-like)",
        how_to_read="Positive excess kurtosis means extreme returns (both directions) are more frequent than a Normal distribution would predict.",
        good_vs_bad="There's no 'good' side - high kurtosis just means fatter tails, i.e. VaR/volatility computed assuming Normality will understate real tail risk.",
        caveats="Financial returns are almost always leptokurtic (excess kurtosis > 0) - don't be alarmed by a positive number, but do combine with cornish_fisher VaR instead of parametric VaR when it's large. Like skewness, this is a bias-adjusted sample estimator, not the raw population moment ratio - noisy at typical financial sample sizes.",
        interpret=lambda v: f"{v:+.2f}" + (" (fat tails - Normal-based risk estimates will understate real risk)" if v > 1 else " (close to Normal)"),
    )
)

register(
    Explanation(
        name="ulcer_index",
        category="metric",
        summary="A drawdown-based risk measure that penalizes both the depth and the duration of drawdowns (unlike volatility, which ignores duration).",
        formula="sqrt(mean(drawdown_pct^2)) over the whole price history",
        how_to_read="Higher values mean drawdowns were deeper and/or the portfolio spent more time underwater.",
        good_vs_bad="Lower is better. Useful for comparing strategies with similar volatility but very different drawdown experience (a choppy-but-quick-to-recover series scores lower than a slow bleed of the same depth).",
        caveats="Time-weights every period underwater, unlike average_drawdown's unweighted per-episode mean. Inherits drawdown_series's sample-window caveat.",
        interpret=lambda v: f"{v:.2%}",
    )
)

register(
    Explanation(
        name="pain_index",
        category="metric",
        summary="The average drawdown magnitude across the entire history - literally 'how much pain, on average, was this investor in?'",
        formula="mean(|drawdown_pct|) over the whole price history",
        how_to_read="A pain index of 0.05 means the portfolio was, on average, 5% below its running peak.",
        good_vs_bad="Lower is better. Often paired with return to form a 'pain ratio' analogous to Calmar but using average rather than max drawdown.",
        caveats="Time-weights every period underwater, unlike average_drawdown's unweighted per-episode mean. Inherits drawdown_series's sample-window caveat.",
        interpret=lambda v: f"{v:.2%} average distance below peak",
    )
)

register(
    Explanation(
        name="beta",
        category="metric",
        summary="How much the portfolio tends to move for every 1-unit move in a benchmark - its sensitivity to market/benchmark risk.",
        formula="Cov(returns, benchmark) / Var(benchmark)",
        how_to_read="Beta of 1.2 means the portfolio historically moved about 20% more than the benchmark, in the same direction, on average.",
        good_vs_bad=(
            "Neither high nor low beta is inherently 'good' - it depends on your goal. Beta < 1 "
            "suggests a defensive profile; beta > 1 an aggressive one; beta near 0 suggests low "
            "correlation to that particular benchmark (not necessarily low risk overall)."
        ),
        caveats="A single historical beta assumes a stable linear relationship - it can shift substantially in market stress (beta instability). No Dimson adjustment for thin/non-synchronous trading (biases beta toward zero) and no Blume adjustment (empirical mean-reversion toward 1 out of sample); also sensitive to the chosen return frequency and lookback window.",
        interpret=lambda v: f"{v:.2f}" + (" (more volatile than the benchmark)" if v > 1.1 else (" (more defensive than the benchmark)" if v < 0.9 else " (moves roughly in line with the benchmark)")),
    )
)

register(
    Explanation(
        name="tracking_error",
        category="metric",
        summary="How much the portfolio's returns deviate, period to period, from a benchmark - the volatility of the difference, not of the portfolio itself.",
        formula="std(returns - benchmark, ddof=1) * sqrt(periods_per_year)  [if annualized]",
        how_to_read="A tracking error of 0.03/yr means the portfolio's return typically differs from the benchmark's by about 3 percentage points per year.",
        good_vs_bad=(
            "For an index-tracking strategy, lower is better (tighter tracking). For an active "
            "strategy seeking to beat a benchmark, some tracking error is expected and even "
            "desirable - judge it together with information_ratio, not alone."
        ),
        interpret=lambda v: f"{v:.2%}/yr",
    )
)
