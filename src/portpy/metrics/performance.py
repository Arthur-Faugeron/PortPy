"""Risk-adjusted performance ratios.

As in the rest of PortPy, `rf`/`mar`/`threshold` parameters are **annual** rates,
de-annualized internally via geometric compounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, MetricResult, register
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
    """Mean excess return over its own standard deviation - the classic risk-adjusted return."""
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
    """Like Sharpe, but only penalizes downside deviation below `mar`, not total volatility."""
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
    """Annualized return divided by the absolute worst drawdown - reward per unit of worst-case pain."""
    from portpy.metrics.drawdowns import max_drawdown

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
    """Ratio of the probability-weighted sum of gains to losses above/below a threshold."""
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
    """Active return over tracking error - how consistently a strategy beats its benchmark."""
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
    """Excess return per unit of *systematic* (beta) risk, rather than per unit of total volatility.

    Args:
        beta: Pre-computed beta of `returns` against the relevant benchmark - see
            :func:`portpy.metrics.risk.beta`.
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
    """Modigliani risk-adjusted performance: the return the portfolio *would* have earned levered/delevered to the benchmark's volatility.

    Expressed in the same units as an annual return, so it's directly comparable
    to the benchmark's own annualized return - unlike Sharpe, which is unitless.
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
    as_result: bool = False,
) -> float | MetricResult:
    """Annualized return divided by the average magnitude of the `n` worst drawdowns.

    This implementation uses a simplified, widely-used variant (average of the
    `n` largest historical drawdowns) rather than the original definition's
    calendar-year drawdowns with a 10-point adjustment - see caveats.
    """
    from portpy.metrics.drawdowns import top_n_drawdowns

    r = returns.dropna()
    ensure_min_observations(r, 2, "returns")
    ann_ret = annualized_return(r, periods_per_year=periods_per_year, geometric=True)
    synthetic_prices = prices_from_returns(r)
    worst = top_n_drawdowns(synthetic_prices, n=n)
    if worst.empty:
        return MetricResult(np.nan, "sterling_ratio") if as_result else np.nan
    avg_dd = float(worst["depth"].abs().mean())
    value = ann_ret / avg_dd if avg_dd > 0 else np.nan
    return MetricResult(value, "sterling_ratio") if as_result else value


def burke_ratio(
    returns: pd.Series,
    n: int = 5,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """Excess annualized return over the root-sum-square of the `n` worst drawdowns.

    Similar in spirit to :func:`sterling_ratio`, but squares the drawdowns before
    combining them, so it penalizes a single very deep drawdown more than several
    moderate ones of the same total size.
    """
    from portpy.metrics.drawdowns import top_n_drawdowns

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
    """Sum of all returns divided by the sum of absolute losses - a Schwager favorite for its simplicity."""
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
    """Like Sortino, but penalizes shortfalls below `mar` using a third-order (cubed) lower partial moment.

    More sensitive to a handful of severe shortfalls than Sortino's second-order
    (squared) penalty.
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
    """Average upside above `mar` divided by downside deviation below `mar` - rewards asymmetric upside."""
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
        summary="The most widely used risk-adjusted return measure: excess return earned per unit of total volatility taken on.",
        formula="mean(r - rf) / std(r - rf, ddof=1) * sqrt(periods_per_year)",
        how_to_read="A Sharpe of 1.0 means you earned, on average, one standard deviation of excess return for the volatility you took on.",
        good_vs_bad="Rules of thumb: <0 poor (lost money net of the risk-free rate), 0-1 sub-par, 1-2 good, 2-3 very good, >3 excellent (and worth double-checking for overfitting or a very short sample).",
        caveats=(
            "Assumes returns are roughly symmetric - it penalizes upside volatility just as much as "
            "downside, and can be misleadingly high for strategies with rare, large negative tail "
            "events (e.g. option-selling). Pair with sortino_ratio and max_drawdown."
        ),
        interpret=lambda v: (
            "poor (losing to the risk-free rate)" if v < 0
            else "sub-par" if v < 1
            else "good" if v < 2
            else "very good" if v < 3
            else "excellent (sanity-check for overfitting/short sample)"
        ),
    )
)

register(
    Explanation(
        name="sortino_ratio",
        category="metric",
        summary="Sharpe's more forgiving cousin: only volatility *below* a minimum acceptable return counts against the strategy.",
        formula="mean(r - mar) / downside_deviation(r, mar) * sqrt(periods_per_year)",
        how_to_read="Will always be >= Sharpe for the same series (since the denominator only counts bad volatility). A big gap between Sortino and Sharpe means the strategy's volatility is mostly on the upside.",
        good_vs_bad="Same rough scale as Sharpe (>1 good, >2 very good), but not directly comparable across strategies with different MAR choices.",
        interpret=lambda v: (
            "poor" if v < 0 else "sub-par" if v < 1 else "good" if v < 2 else "very good" if v < 3 else "excellent"
        ),
    )
)

register(
    Explanation(
        name="calmar_ratio",
        category="metric",
        summary="Annualized return relative to the single worst drawdown ever experienced - a 'worst case reward-to-pain' ratio.",
        formula="annualized_return / abs(max_drawdown)",
        how_to_read="A Calmar of 2.0 means the strategy earns roughly 2x its worst-ever peak-to-trough loss, per year.",
        good_vs_bad="Rules of thumb: <1 weak (the worst drawdown roughly erased a year's gains), 1-3 solid, >3 excellent (very fast recovery relative to drawdown depth).",
        caveats="Extremely sensitive to a single historical event - a short backtest without a real crash may show a great Calmar that won't survive the next one.",
        interpret=lambda v: "weak" if v < 1 else "solid" if v < 3 else "excellent",
    )
)

register(
    Explanation(
        name="omega_ratio",
        category="metric",
        summary="Ratio of total gains to total losses relative to a threshold, using the *entire* return distribution rather than just its mean and variance.",
        formula="sum(gains above threshold) / sum(|losses below threshold|)",
        how_to_read="Omega of 1.0 means gains and losses around the threshold exactly balance. Above 1.0 favors the strategy.",
        good_vs_bad="Higher is better; unlike Sharpe/Sortino it captures skewness and fat tails directly since it uses the full distribution, not just mean/variance.",
        interpret=lambda v: f"{v:.2f}" + (" (losses dominate around the threshold)" if v < 1 else " (gains dominate around the threshold)"),
    )
)

register(
    Explanation(
        name="information_ratio",
        category="metric",
        summary="Active return over a benchmark, scaled by how consistent (low tracking error) that outperformance is.",
        formula="mean(returns - benchmark) / std(returns - benchmark, ddof=1) * sqrt(periods_per_year)",
        how_to_read="An IR of 0.5 is considered decent for active management; 1.0+ is very good and rare to sustain.",
        good_vs_bad="Higher is better - it rewards *consistent* outperformance, not just occasional lucky big wins.",
        interpret=lambda v: "underperforming/inconsistent" if v < 0 else "modest" if v < 0.5 else "good" if v < 1 else "excellent (rare to sustain)",
    )
)

register(
    Explanation(
        name="treynor_ratio",
        category="metric",
        summary="Excess return per unit of market (systematic/beta) risk, instead of per unit of total volatility like Sharpe.",
        formula="mean(r - rf) / beta * periods_per_year",
        how_to_read="Only meaningful when comparing portfolios that share a similar, well-diversified benchmark exposure - beta ignores idiosyncratic risk entirely.",
        good_vs_bad="Higher is better; a low or negative beta near zero makes this ratio unstable/uninformative, so check beta's magnitude first.",
        caveats="Garbage in, garbage out: the ratio is only as good as the beta estimate you feed it.",
        interpret=lambda v: f"{v:+.3f} excess return per unit of beta",
    )
)

register(
    Explanation(
        name="m2_measure",
        category="metric",
        summary="Modigliani-Modigliani (M2): re-expresses Sharpe ratio as an annualized return, by imagining the portfolio levered/delevered to match the benchmark's volatility.",
        formula="rf + sharpe_ratio(returns) * volatility(benchmark)",
        how_to_read="Directly comparable to the benchmark's own annualized return - e.g. an M2 of 12% vs. a benchmark return of 9% means the portfolio would have beaten the benchmark by 3 points/year at matched risk.",
        good_vs_bad="Higher than the benchmark's own return is good - it means better risk-adjusted performance, expressed in return terms rather than an abstract ratio.",
        interpret=lambda v: f"{v:+.1%}/yr risk-matched return",
    )
)

register(
    Explanation(
        name="sterling_ratio",
        category="metric",
        summary="Annualized return relative to the average size of the worst few drawdowns, rather than just the single worst one (Calmar).",
        formula="annualized_return / mean(|n worst drawdowns|)",
        how_to_read="Smooths out Calmar's sensitivity to one single event by averaging several bad episodes.",
        good_vs_bad="Higher is better; same rough scale as Calmar.",
        caveats="This package uses a simplified variant (average of the n largest historical drawdowns); some sources define Sterling using calendar-year drawdowns with a fixed 10-point adjustment instead.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="burke_ratio",
        category="metric",
        summary="Like Sterling, but combines the worst drawdowns via root-sum-of-squares instead of a plain average, penalizing one severe drawdown more than several mild ones.",
        formula="(annualized_return - rf) / sqrt(sum(worst_drawdowns^2))",
        how_to_read="Lower than Sterling for the same data if the worst drawdown is much bigger than the others (squaring emphasizes it).",
        good_vs_bad="Higher is better.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="gain_to_pain_ratio",
        category="metric",
        summary="Jack Schwager's simplest 'is this worth it' ratio: total gains vs. total losses, no annualization or volatility involved.",
        formula="sum(r) / sum(|r| for r < 0)",
        how_to_read="A ratio of 1.0 means cumulative gains exactly offset cumulative losses (roughly breakeven journey, even if the endpoint is positive due to compounding).",
        good_vs_bad="Schwager's own rule of thumb: above 1.5 is quite good for most strategies; below 0.5 suggests a rough ride for the return achieved.",
        interpret=lambda v: "rough ride for the return achieved" if v < 0.5 else "acceptable" if v < 1.5 else "quite good",
    )
)

register(
    Explanation(
        name="kappa_three_ratio",
        category="metric",
        summary="A generalization of Sortino that penalizes shortfalls below the MAR using a cubed (third-order) penalty instead of squared.",
        formula="mean(r - mar) / (mean(max(mar - r, 0)^3))^(1/3)",
        how_to_read="More sensitive than Sortino to a small number of severe shortfalls, less sensitive to many small ones.",
        good_vs_bad="Higher is better; compare only across strategies measured with the same MAR.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="upside_potential_ratio",
        category="metric",
        summary="Average upside above the MAR divided by downside risk below it - rewards strategies with a favorable up/down shape, not just a high average.",
        formula="mean(max(r - mar, 0)) / downside_deviation(r, mar)",
        how_to_read="Higher values mean the strategy captures meaningfully more upside than the downside risk it takes on.",
        good_vs_bad="Higher is better; typically read alongside Sortino since they share the same downside-deviation denominator.",
        interpret=lambda v: f"{v:.2f}",
    )
)
