"""
Risk-adjusted performance ratios.

As in the rest of PortPy, rf/mar/threshold parameters are annual rates,
de-annualized internally via geometric compounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, MetricResult, register
from portpy.metrics.drawdowns import max_drawdown, top_n_drawdowns
from portpy.metrics.returns import annualized_return, prices_from_returns
from portpy.metrics.risk import downside_deviation, volatility
from portpy.utils.constants import DEFAULT_MAR, DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import align_pair, ensure_min_observations, periodic_rate_from_annual, safe_divide

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "information_ratio",
    "treynor_ratio",
    "m2_measure",
    "sterling_ratio",
    "burke_ratio",
    "gain_to_pain_ratio",
    "kappa_three_ratio",
    "upside_potential_ratio",
]


def sharpe_ratio(
    returns: pd.Series,
    rf: float = DEFAULT_RISK_FREE_RATE,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Mean excess return over its own standard deviation - the classic
    risk-adjusted return measure.

    Args:
        returns: Periodic return series.
        rf: Annual risk-free rate, de-annualized internally to a constant
            per-period rate.
        annualized: If True, scale the result by sqrt(periods_per_year).
        periods_per_year: Number of periods per year, used for annualization.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Sharpe ratio, or a MetricResult wrapping it.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    rf_period = periodic_rate_from_annual(rf, periods_per_year)
    excess = r - rf_period
    value = safe_divide(float(excess.mean()), float(excess.std(ddof=1)))
    if annualized:
        value *= np.sqrt(periods_per_year)
    return MetricResult(value, "sharpe_ratio") if as_result else value


def sortino_ratio(
    returns: pd.Series,
    mar: float = DEFAULT_MAR,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Like Sharpe, but penalizes only downside deviation below `mar`, not
    total volatility.

    Args:
        returns: Periodic return series.
        mar: Annual minimum acceptable return, de-annualized internally.
        annualized: If True, scale the result by sqrt(periods_per_year).
        periods_per_year: Number of periods per year, used for annualization.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Sortino ratio, or a MetricResult wrapping it.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    mar_period = periodic_rate_from_annual(mar, periods_per_year)
    dd_period = downside_deviation(r, mar=mar, annualized=False, periods_per_year=periods_per_year)
    value = safe_divide(float((r - mar_period).mean()), float(dd_period))
    if annualized:
        value *= np.sqrt(periods_per_year)
    return MetricResult(value, "sortino_ratio") if as_result else value


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Annualized return divided by the absolute worst drawdown - reward per
    unit of worst-case pain.

    Args:
        returns: Periodic return series.
        periods_per_year: Number of periods per year, used for annualization.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Calmar ratio, or NaN if the series had no drawdown.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    ann_ret = annualized_return(r, periods_per_year=periods_per_year, geometric=True)
    synthetic_prices = prices_from_returns(r)
    mdd = max_drawdown(synthetic_prices)
    value = ann_ret / abs(mdd) if mdd < 0 else np.nan
    return MetricResult(value, "calmar_ratio") if as_result else value


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Ratio of the probability-weighted sum of gains to losses above/below a
    threshold return.

    Args:
        returns: Periodic return series.
        threshold: Annual required return; must be > -1.
        periods_per_year: Number of periods per year, used to de-annualize
            `threshold`.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Omega ratio, or NaN if there were no losses below the threshold.

    Raises:
        ValueError: If `threshold` is <= -1.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    if threshold <= -1:
        raise ValueError("threshold must be > -1 (it represents an annual required return).")
    thresh_period = periodic_rate_from_annual(threshold, periods_per_year)
    diff = r - thresh_period
    numer = float(diff[diff > 0].sum())
    denom = float(-diff[diff < 0].sum())
    value = numer / denom if denom > 0 else np.nan
    return MetricResult(value, "omega_ratio") if as_result else value


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Active return over tracking error - how consistently a strategy beats
    its benchmark.

    Args:
        returns: Periodic return series.
        benchmark: Periodic benchmark return series, aligned to `returns`.
        annualized: If True, scale the result by sqrt(periods_per_year).
        periods_per_year: Number of periods per year, used for annualization.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The information ratio, or a MetricResult wrapping it.
    """

    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    diff = r - b
    value = safe_divide(float(diff.mean()), float(diff.std(ddof=1)))
    if annualized:
        value *= np.sqrt(periods_per_year)
    return MetricResult(value, "information_ratio") if as_result else value


def treynor_ratio(
    returns: pd.Series,
    beta: float,
    rf: float = DEFAULT_RISK_FREE_RATE,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Excess return per unit of systematic (beta) risk, rather than per unit
    of total volatility.

    Args:
        returns: Periodic return series.
        beta: Pre-computed beta of `returns` against the relevant benchmark
            (see `portpy.metrics.risk.beta`).
        rf: Annual risk-free rate, de-annualized internally to a constant
            per-period rate.
        annualized: If True, scale the result by periods_per_year (linear,
            not sqrt, scaling).
        periods_per_year: Number of periods per year, used for annualization.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Treynor ratio, or a MetricResult wrapping it.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    rf_period = periodic_rate_from_annual(rf, periods_per_year)
    mean_excess = float((r - rf_period).mean())
    value = safe_divide(mean_excess, beta)
    if annualized:
        value *= periods_per_year
    return MetricResult(value, "treynor_ratio") if as_result else value


def m2_measure(
    returns: pd.Series,
    benchmark: pd.Series,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Modigliani risk-adjusted performance: the return the portfolio would
    have earned levered/delevered to match the benchmark's volatility.
    Expressed in the same units as an annual return, unlike Sharpe.

    Args:
        returns: Periodic return series.
        benchmark: Periodic benchmark return series, aligned to `returns`.
        rf: Annual risk-free rate, used as-is (not de-annualized).
        periods_per_year: Number of periods per year, used for annualization.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The M2 measure, expressed as an annualized return.
    """

    r, b = align_pair(returns, benchmark)
    ensure_min_observations(r, 2, "returns")
    sharpe_p = sharpe_ratio(r, rf=rf, annualized=True, periods_per_year=periods_per_year)
    vol_b = volatility(b, annualized=True, periods_per_year=periods_per_year)
    value = rf + float(sharpe_p) * float(vol_b)
    return MetricResult(value, "m2_measure", unit="%") if as_result else value


def sterling_ratio(
    returns: pd.Series,
    n: int = 5,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    drawdown_adjustment: float = 0.0,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Annualized return divided by the average magnitude of the n worst
    drawdowns.

    Uses a simplified, widely-used variant (average of the n largest
    historical drawdowns) rather than the original definition's
    calendar-year drawdowns.

    Args:
        returns: Periodic return series.
        n: Number of worst drawdown episodes to average.
        periods_per_year: Number of periods per year, used for annualization.
        drawdown_adjustment: Added to the averaged drawdown denominator before
            dividing. Defaults to 0.0; the classic convention uses 0.10.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Sterling ratio, or NaN if there were no drawdown episodes.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    ann_ret = annualized_return(r, periods_per_year=periods_per_year, geometric=True)
    synthetic_prices = prices_from_returns(r)
    worst = top_n_drawdowns(synthetic_prices, n=n)
    if worst.empty:
        return MetricResult(np.nan, "sterling_ratio") if as_result else np.nan
    avg_dd = float(worst["depth"].abs().mean()) + drawdown_adjustment
    value = ann_ret / avg_dd if avg_dd > 0 else np.nan
    return MetricResult(value, "sterling_ratio") if as_result else value


def burke_ratio(
    returns: pd.Series,
    n: int = 5,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Excess annualized return over the root-sum-square of the n worst
    drawdowns.

    Similar in spirit to sterling_ratio, but squares the drawdowns before
    combining them, penalizing a single deep drawdown more than several
    moderate ones of the same total size.

    Args:
        returns: Periodic return series.
        n: Number of worst drawdown episodes to include.
        rf: Annual risk-free rate, used as-is (not de-annualized).
        periods_per_year: Number of periods per year, used for annualization.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Burke ratio, or NaN if there were no drawdown episodes.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    ann_ret = annualized_return(r, periods_per_year=periods_per_year, geometric=True)
    synthetic_prices = prices_from_returns(r)
    worst = top_n_drawdowns(synthetic_prices, n=n)
    if worst.empty:
        return MetricResult(np.nan, "burke_ratio") if as_result else np.nan
    rss = float(np.sqrt(np.sum(np.square(worst["depth"]))))
    value = (ann_ret - rf) / rss if rss > 0 else np.nan
    return MetricResult(value, "burke_ratio") if as_result else value


def gain_to_pain_ratio(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Sum of all returns divided by the sum of absolute losses.

    Args:
        returns: Periodic return series.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The gain-to-pain ratio.
    """

    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    losses = float(-r[r < 0].sum())
    value = safe_divide(float(r.sum()), losses)
    return MetricResult(value, "gain_to_pain_ratio") if as_result else value


def kappa_three_ratio(
    returns: pd.Series,
    mar: float = DEFAULT_MAR,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Like Sortino, but penalizes shortfalls using a third-order (cubed) lower
    partial moment - more sensitive to a handful of severe shortfalls than
    Sortino's squared penalty.

    Args:
        returns: Periodic return series.
        mar: Annual minimum acceptable return, de-annualized internally.
        periods_per_year: Number of periods per year, used to de-annualize `mar`.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The Kappa-3 ratio (unannualized, unlike sibling sortino_ratio), or a
        MetricResult wrapping it.
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    mar_period = periodic_rate_from_annual(mar, periods_per_year)
    shortfall = np.maximum(mar_period - r, 0.0)
    lpm3 = float(np.mean(np.power(shortfall, 3)))
    denom = lpm3 ** (1.0 / 3.0) if lpm3 > 0 else 0.0
    value = safe_divide(float((r - mar_period).mean()), denom)
    return MetricResult(value, "kappa_three_ratio") if as_result else value


def upside_potential_ratio(
    returns: pd.Series,
    mar: float = DEFAULT_MAR,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Average upside above `mar` divided by downside deviation below `mar` -
    rewards asymmetric upside.

    Args:
        returns: Periodic return series.
        mar: Annual minimum acceptable return, de-annualized internally.
        periods_per_year: Number of periods per year, used to de-annualize `mar`.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The upside potential ratio (unannualized, unlike sibling sortino_ratio).
    """

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    mar_period = periodic_rate_from_annual(mar, periods_per_year)
    upside = float(np.mean(np.maximum(r - mar_period, 0.0)))
    dd_period = downside_deviation(r, mar=mar, annualized=False, periods_per_year=periods_per_year)
    value = safe_divide(upside, float(dd_period))
    return MetricResult(value, "upside_potential_ratio") if as_result else value


register(
    Explanation(
        name="sharpe_ratio",
        category="metric",
        summary="Measures excess return earned per unit of total volatility taken, using the risk-free rate as the return hurdle.",
        formula="mean(r - rf) / std(r - rf, ddof=1) * sqrt(periods_per_year)",
        how_to_read="A Sharpe of 1.0 means the strategy generated approximately one unit of excess return for each unit of volatility. Higher values indicate better risk-adjusted performance.",
        good_vs_bad="Higher is generally better. Values above 1 are commonly considered strong, but interpretation depends on the asset class, time period, and strategy complexity.",
        caveats="Sharpe treats all volatility as bad, including upside volatility. It can also overstate strategies with asymmetric downside risk, illiquidity, or short backtests. The sqrt(periods_per_year) annualization assumes i.i.d., serially uncorrelated returns - positive autocorrelation (illiquid/infrequently-priced assets) means the annualized figure is overstated (Lo, 2002).",
        interpret=lambda v: (
            "poor (negative excess return)" if v < 0
            else "acceptable" if v < 1
            else "good" if v < 2
            else "very good" if v < 3
            else "excellent (verify robustness)"
        ),
    )
)

register(
    Explanation(
        name="sortino_ratio",
        category="metric",
        summary="Measures excess return relative to downside risk only, ignoring upside volatility.",
        formula="mean(r - mar) / downside_deviation(r, mar) * sqrt(periods_per_year)",
        how_to_read="Higher values indicate that returns are being achieved with less harmful downside variation. Compare only when MAR assumptions are consistent.",
        good_vs_bad="Higher is generally better. A large gap between Sortino and Sharpe suggests that volatility is mostly coming from positive returns rather than losses.",
        caveats="Sensitive to the chosen MAR. Different MAR assumptions can produce significantly different results. Same sqrt(periods_per_year) i.i.d. annualization caveat as sharpe_ratio.",
        interpret=lambda v: (
            "poor" if v < 0 else "acceptable" if v < 1 else "good" if v < 2 else "very good" if v < 3 else "excellent"
        ),
    )
)

register(
    Explanation(
        name="calmar_ratio",
        category="metric",
        summary="Measures annualized return relative to the portfolio's largest historical drawdown.",
        formula="annualized_return / abs(max_drawdown)",
        how_to_read="A Calmar of 2 means the annualized return is roughly twice the size of the worst observed drawdown.",
        good_vs_bad="Higher values indicate more return generated relative to worst historical loss. Values above 3 are generally considered strong, but depend on the sample period.",
        caveats="Highly dependent on the worst historical event in the sample. Short histories may underestimate future drawdown risk. The classic convention (Young, 1991) uses a fixed trailing 36-month window; this implementation uses whatever history is passed in.",
        interpret=lambda v: "weak" if v < 1 else "solid" if v < 3 else "excellent",
    )
)

register(
    Explanation(
        name="omega_ratio",
        category="metric",
        summary="Compares the probability-weighted magnitude of gains and losses relative to a chosen return threshold.",
        formula="sum(gains above threshold) / sum(abs(losses below threshold))",
        how_to_read="A value above 1 means gains above the threshold outweigh losses below it. A value below 1 means losses dominate.",
        good_vs_bad="Higher is better because it captures the full return distribution rather than only mean and volatility.",
        caveats="Highly dependent on the selected threshold. Results can change substantially when the target return changes. Unlike Sharpe/Sortino, there's no valid way to rescale Omega across sampling frequencies - only compare values computed at the same frequency.",
        interpret=lambda v: f"{v:.2f}" + (" (losses dominate)" if v < 1 else " (gains dominate)"),
    )
)

register(
    Explanation(
        name="information_ratio",
        category="metric",
        summary="Measures active return generated over a benchmark relative to the consistency of that outperformance.",
        formula="mean(returns - benchmark) / std(returns - benchmark, ddof=1) * sqrt(periods_per_year)",
        how_to_read="Higher values indicate more consistent benchmark outperformance. A value near zero means active returns are not reliably different from the benchmark.",
        good_vs_bad="Higher is better. Sustained values above 1 are uncommon and indicate strong active management consistency.",
        caveats="Depends heavily on benchmark selection. A poor benchmark can make the ratio misleading. Same sqrt(periods_per_year) i.i.d. annualization caveat as sharpe_ratio applies to the tracking-error denominator. quantstats' information_ratio() never annualizes - the two agree exactly at annualized=False and differ by exactly sqrt(periods_per_year) at the default annualized=True; not a formula disagreement, just a different default time-scale.",
        interpret=lambda v: "negative active performance" if v < 0 else "weak" if v < 0.5 else "good" if v < 1 else "excellent",
    )
)

register(
    Explanation(
        name="treynor_ratio",
        category="metric",
        summary="Measures excess return earned per unit of systematic market risk measured by beta.",
        formula="mean(r - rf) / beta * periods_per_year",
        how_to_read="Higher values indicate more return generated for each unit of market exposure. It is mainly useful when comparing diversified portfolios with similar benchmarks.",
        good_vs_bad="Higher is better, but only meaningful when beta is stable and accurately estimated.",
        caveats="Ignores idiosyncratic risk. Results become unstable when beta is close to zero or changes significantly over time - no sanity check is applied, so a near-zero or negative beta can explode or sign-flip the ratio. Uses linear (not sqrt) annualization, a third convention distinct from Sharpe/Sortino's sqrt-scaling and Alpha's geometric compounding. quantstats' treynor_ratio() divides the *total cumulative* return over the whole sample by beta instead of an annualized mean return - a different metric definition, not a rescaling; expect no fixed conversion factor between the two, especially on multi-year samples.",
        interpret=lambda v: f"{v:+.3f} excess return per beta unit",
    )
)

register(
    Explanation(
        name="m2_measure",
        category="metric",
        summary="Converts Sharpe ratio into a return percentage by adjusting the portfolio to the benchmark volatility level.",
        formula="rf + sharpe_ratio(returns) * volatility(benchmark)",
        how_to_read="Can be compared directly with benchmark returns. A higher M2 indicates better risk-adjusted performance after matching volatility.",
        good_vs_bad="Higher than the benchmark return indicates superior risk-adjusted performance.",
        caveats="Assumes volatility is the relevant risk measure and depends on the benchmark used for comparison. rf is used as an annual rate directly here, unlike sharpe_ratio/treynor_ratio which convert it to periodic first.",
        interpret=lambda v: f"{v:+.1%}/yr risk-adjusted return",
    )
)

register(
    Explanation(
        name="sterling_ratio",
        category="metric",
        summary="Measures annualized return relative to the average magnitude of the largest historical drawdowns.",
        formula="annualized_return / (mean(abs(n worst drawdowns)) + drawdown_adjustment)",
        how_to_read="Higher values indicate more return generated relative to repeated severe drawdown events.",
        good_vs_bad="Higher is better. It is less dominated by a single worst event than Calmar ratio.",
        caveats="drawdown_adjustment defaults to 0.0 here; the classic Deane Sterling Jones convention adds a flat 10 percentage points (drawdown_adjustment=0.10) to keep the ratio from exploding when drawdowns are small - pass that explicitly to match a vendor using the classic convention.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="burke_ratio",
        category="metric",
        summary="Measures excess return relative to the combined impact of the largest drawdowns, giving more weight to extreme losses.",
        formula="(annualized_return - rf) / sqrt(sum(worst_drawdowns^2))",
        how_to_read="Higher values indicate stronger return generation after accounting for severe drawdown history.",
        good_vs_bad="Higher is better. Lower values indicate that drawdown severity consumes more of the portfolio's return.",
        caveats="Sensitive to the selected number of drawdowns included and the historical sample. This is the standard (non-modified) Burke Ratio; a 'Modified Burke Ratio' variant divides by sqrt(sum(D_i^2)/n) instead - confirm the convention before comparing across tools.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="gain_to_pain_ratio",
        category="metric",
        summary="Compares cumulative gains against cumulative losses, focusing on the balance between positive and negative returns.",
        formula="sum(positive returns) / sum(abs(negative returns))",
        how_to_read="Values above 1 indicate that gains outweigh losses. Higher values indicate a more favorable return distribution.",
        good_vs_bad="Higher is better, but it does not account for volatility, drawdown depth, or the path taken to achieve returns.",
        caveats="Ignores compounding effects and timing of returns. Two strategies with the same ratio may have very different risk profiles. Schwager's original convention (and common '>1.5 is good' thresholds) is benchmarked on monthly returns - resample to monthly before comparing a daily-computed value to published thresholds.",
        interpret=lambda v: "poor" if v < 0.5 else "acceptable" if v < 1.5 else "strong",
    )
)

register(
    Explanation(
        name="kappa_three_ratio",
        category="metric",
        summary="Measures return relative to downside risk using a third-order penalty for returns below the minimum acceptable return.",
        formula="mean(r - mar) / mean(max(mar - r, 0)^3)^(1/3)",
        how_to_read="Higher values indicate better compensation for severe downside outcomes. It penalizes extreme losses more than Sortino ratio.",
        good_vs_bad="Higher is better when comparing strategies using the same MAR assumption.",
        caveats="More sensitive to extreme observations than Sortino, which can make it unstable with short samples. Unlike sibling sortino_ratio, has no `annualized` toggle - mar is accepted as an annual rate but the output stays period-scale.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="upside_potential_ratio",
        category="metric",
        summary="Measures average upside above the MAR relative to downside deviation below the MAR.",
        formula="mean(max(r - mar, 0)) / downside_deviation(r, mar)",
        how_to_read="Higher values indicate that upside potential is large relative to downside risk.",
        good_vs_bad="Higher is better, especially for strategies targeting asymmetric return profiles.",
        caveats="Depends on MAR selection and does not measure drawdown behavior directly. Same as kappa_three_ratio: no `annualized` toggle, unlike sibling sortino_ratio.",
        interpret=lambda v: f"{v:.2f}",
    )
)