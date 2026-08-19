"""
Return transformations and return-based summary statistics.

Convention: every scalar function accepts as_result=True to get back a
portpy.explain.MetricResult (a float that also knows how to explain
itself) instead of a plain float. Annualization everywhere in PortPy uses the
period-count convention (years = n_periods / periods_per_year), matching
empyrical/quantstats defaults, rather than actual elapsed calendar time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, MetricResult, register
from portpy.utils.constants import TRADING_DAYS_PER_YEAR
from portpy.utils.validation import ensure_min_observations

__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "prices_from_returns",
    "total_return",
    "annualized_return",
    "cagr",
    "average_return",
    "rebased_returns",
    "excess_returns",
    "active_returns",
]


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Period-over-period simple (arithmetic) returns.

    Args:
        prices: Price series or DataFrame of price columns.

    Returns:
        Simple returns aligned to `prices`, with the first period dropped.
    """
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Period-over-period continuously-compounded (log) returns.

    Args:
        prices: Price series or DataFrame of price columns.

    Returns:
        Log returns aligned to `prices`, with the first period dropped.
    """
    return np.log(prices / prices.shift(1)).iloc[1:]


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Compounded cumulative return series from a series of simple returns.

    Args:
        returns: Simple per-period returns.

    Returns:
        Cumulative return at each period, e.g. 0.25 for a 25% total gain.
    """
    return (1.0 + returns).cumprod() - 1.0  # type: ignore[attr-defined]


def prices_from_returns(returns: pd.Series | pd.DataFrame, base: float = 1.0) -> pd.Series | pd.DataFrame:
    """
    Reconstruct a price-like series from simple returns, anchored at `base`
    one period before the first return.

    Args:
        returns: Simple per-period returns.
        base: Starting price level. Defaults to 1.0.

    Returns:
        A price series (or DataFrame) one period longer than `returns`, whose
        first value is `base` and which compounds `returns` from there.
    """
    idx = returns.index
    if isinstance(idx, pd.DatetimeIndex):
        step = (idx[1] - idx[0]) if len(idx) >= 2 else pd.Timedelta(days=1)
        anchor_label = idx[0] - step
    else:
        anchor_label = -1

    compounded = base * (1.0 + returns).cumprod()  # type: ignore[attr-defined]
    if isinstance(returns, pd.DataFrame):
        anchor = pd.DataFrame([[base] * returns.shape[1]], index=[anchor_label], columns=returns.columns)
    else:
        anchor = pd.Series([base], index=[anchor_label], name=returns.name)
    return pd.concat([anchor, compounded])


def rebased_returns(prices: pd.Series | pd.DataFrame, base: float = 100.0) -> pd.Series | pd.DataFrame:
    """
    Rescale a price series so it starts at `base`, preserving percentage changes.

    Args:
        prices: Price series or DataFrame of price columns.
        base: Starting value for the rescaled series. Defaults to 100.

    Returns:
        Rescaled price series, same shape as `prices`.
    """
    return prices / prices.iloc[0] * base


def total_return(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Total compounded return over the full return series.

    Args:
        returns: Per-period simple returns.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The total compounded return, e.g. 0.35 for a 35% gain.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    value = float((1.0 + r).prod() - 1.0)
    return MetricResult(value, "total_return", unit="%") if as_result else value


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    geometric: bool = True,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Annualized return computed from a return series.

    Args:
        returns: Per-period simple returns.
        periods_per_year: Number of periods in a year, used to annualize.
        geometric: If True (default), compounds the total return and raises
            it to periods_per_year / n_periods (the CAGR of the series). If
            False, scales the arithmetic mean return by periods_per_year.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The annualized return. When geometric and the compounded growth
        factor is non-positive (a total loss or worse), returns -1.0.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    n = len(r)
    if geometric:
        total = float((1.0 + r).prod())
        years = n / periods_per_year
        if years <= 0:
            value = np.nan
        elif total <= 0.0:
            value = -1.0
        else:
            value = total ** (1.0 / years) - 1.0
    else:
        value = float(r.mean()) * periods_per_year
    return MetricResult(value, "annualized_return", unit="%") if as_result else float(value)


def cagr(
    prices: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """
    Compound Annual Growth Rate, computed directly from a price series.

    Args:
        prices: Price series spanning the measurement period.
        periods_per_year: Number of periods in a year, used to annualize.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The CAGR. Returns -1.0 if the start/end price ratio is non-positive,
        rather than raising.
    """
    p = prices.dropna()
    ensure_min_observations(p, 2, "prices")
    n_periods = len(p) - 1
    years = n_periods / periods_per_year
    total = float(p.iloc[-1] / p.iloc[0])
    if years <= 0:
        value = np.nan
    elif total <= 0.0:
        value = -1.0
    else:
        value = total ** (1.0 / years) - 1.0
    return MetricResult(value, "cagr", unit="%") if as_result else float(value)


def average_return(returns: pd.Series, geometric: bool = False, as_result: bool = False) -> float | MetricResult:
    """
    Average return per period, arithmetic (default) or geometric.

    Args:
        returns: Per-period simple returns.
        geometric: If True, computes exp(mean(log1p(returns))) - 1 instead of
            the arithmetic mean. Defaults to arithmetic.
        as_result: If True, return a MetricResult instead of a plain float.

    Returns:
        The average per-period return.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    if geometric:
        value = float(np.exp(np.log1p(r).mean()) - 1.0)
    else:
        value = float(r.mean())
    return MetricResult(value, "average_return", unit="%") if as_result else value


def excess_returns(returns: pd.Series, benchmark_or_rf: pd.Series | float) -> pd.Series:
    """
    Per-period return in excess of a benchmark series or a constant rate.

    Args:
        returns: Per-period simple returns.
        benchmark_or_rf: A benchmark return series (aligned by date) or a
            constant per-period rate to subtract.

    Returns:
        Per-period excess returns.
    """
    if isinstance(benchmark_or_rf, pd.Series):
        joined = pd.concat([returns.rename("r"), benchmark_or_rf.rename("b")], axis=1, join="inner").dropna()
        return joined["r"] - joined["b"]
    return returns - benchmark_or_rf


def active_returns(returns: pd.Series, benchmark: pd.Series) -> pd.Series:
    """
    Per-period return relative to a benchmark series.

    Args:
        returns: Per-period simple returns.
        benchmark: Benchmark per-period simple returns.

    Returns:
        Per-period active (portfolio minus benchmark) returns.
    """
    return excess_returns(returns, benchmark)


register(
    Explanation(
        name="simple_returns",
        category="function",
        summary=("The arithmetic percentage change in price from one period to the next. This is the standard return representation used for portfolio calculations."),
        formula="P_t / P_(t-1) - 1",
        how_to_read=("A value of 0.01 means the asset gained 1% during that period. A value of -0.02 means it lost 2%."),
        good_vs_bad=("Higher returns are generally desirable, but simple returns only describe performance, not the amount of risk required to achieve it."),
        caveats=("Simple returns are additive across assets (weight-average them for a portfolio return) but not additive through time - use log_returns for that, and don't mix the two conventions."),
        interpret=lambda v: f"{v:+.2%} period return",
    )
)

register(
    Explanation(
        name="log_returns",
        category="function",
        summary=("The continuously compounded return calculated from the logarithmic change in price. Commonly used in quantitative finance and statistical modeling."),
        formula="ln(P_t / P_(t-1))",
        how_to_read=("A value of 0.02 represents approximately a 2% continuously compounded return for the period."),
        good_vs_bad=("Useful for quantitative analysis because log returns are additive across time and often behave better in mathematical models."),
        caveats=("Log returns are not identical to simple returns, especially during large price movements, and are not additive across assets - convert to simple returns before weight-averaging a portfolio."),
        interpret=lambda v: f"{v:+.2%} log return",
    )
)

register(
    Explanation(
        name="cumulative_returns",
        category="function",
        summary=("The compounded growth of an investment over a sequence of returns, showing how wealth evolves through time."),
        formula="prod(1 + r_t) - 1",
        how_to_read=("A value of 0.25 means an initial investment increased by 25% over the entire measured period."),
        good_vs_bad=("Higher cumulative return indicates stronger absolute performance, but it must be evaluated alongside volatility, drawdown, and investment horizon."),
        caveats=("Cumulative return ignores the path taken. Two investments can have the same ending return but very different risk experiences. Assumes simple returns - feeding log returns in silently produces a wrong number."),
        interpret=lambda v: f"{v:+.1%} cumulative return",
    )
)

register(
    Explanation(
        name="prices_from_returns",
        category="function",
        summary=("Transforms a return series into a synthetic price/equity curve by compounding returns from a chosen starting value."),
        formula="base * prod(1 + r_t)",
        how_to_read=("A resulting value of 1.20 means a starting investment of 1.00 grew to 1.20 after applying the return sequence."),
        good_vs_bad=("Useful for creating equity curves from return data and enabling price-based analysis such as drawdowns and CAGR."),
        caveats=("This does not recreate real market prices. It only reconstructs the growth path implied by the returns, and assumes those returns are simple (not log) returns."),
        interpret=lambda v: f"{v:.2f} reconstructed price level",
    )
)

register(
    Explanation(
        name="total_return",
        category="metric",
        summary=("The total compounded gain or loss achieved over the entire investment period without converting it into an annual rate."),
        formula="prod(1 + r_t) - 1",
        how_to_read=("A value of 0.35 means an investment gained 35% from start to finish."),
        good_vs_bad=("Useful for measuring absolute performance, but comparisons should use the same time period and similar risk exposure."),
        caveats=("Not annualized. The same total return can represent very different performance depending on whether it occurred over months or years."),
        interpret=lambda v: (f"{v:+.1%} total return" + (" (loss)" if v < 0 else "")),
    )
)

register(
    Explanation(
        name="annualized_return",
        category="metric",
        summary=("The return converted into an annual growth rate, allowing comparison "
            "between investments with different measurement periods."),
        formula=("geometric: prod(1+r)^(periods_per_year/n) - 1 | "
            "arithmetic: mean(r) * periods_per_year"),
        how_to_read=("A value of 0.12 means the investment produced an equivalent annualized "
            "return of 12%."),
        good_vs_bad=("Higher annualized returns are generally preferable, but they should always "
            "be evaluated together with volatility, drawdown, and consistency."),
        caveats=("Geometric annualization is preferred because it accounts for compounding. "
            "Arithmetic annualization can overstate expected growth in volatile series. "
            "The geometric branch needs a strictly positive compounded growth factor - a "
            "leveraged/short series that compounds to a total loss or worse returns -1.0 "
            "rather than a fractional power of a non-positive number."),
        interpret=lambda v: f"{v:+.1%} annualized return",
    )
)

register(
    Explanation(
        name="cagr",
        category="metric",
        summary=("The constant annual growth rate required for an investment to move from its starting value to its ending value."),
        formula="(P_end / P_start)^(1/years) - 1",
        how_to_read=("A CAGR of 0.10 means the investment grew as if it compounded at 10% per year."),
        good_vs_bad=("Higher CAGR indicates stronger long-term growth, but it does not describe volatility or the path taken."),
        caveats=("CAGR ignores intermediate fluctuations. Two investments with identical CAGR can have completely different risk profiles. "
            "Needs a strictly positive start/end price ratio - a non-positive ratio returns -1.0 rather than a fractional power of a non-positive number."),
        interpret=lambda v: f"{v:+.1%}/year compounded",
    )
)

register(
    Explanation(
        name="average_return",
        category="metric",
        summary=("The average return per observation, calculated either arithmetically or geometrically depending on the selected method."),
        formula=("arithmetic: mean(r) | geometric: exp(mean(ln(1+r))) - 1"),
        how_to_read=("This is the average return per period (for example daily), not an annual performance measure."),
        good_vs_bad=("Useful for understanding return behavior, but should not be used alone because it ignores dispersion and downside risk."),
        caveats=("Arithmetic averages can overstate realized growth when returns are volatile. Geometric averages better represent compounded wealth growth."),
        interpret=lambda v: f"{v:+.3%} average period return",
    )
)

register(
    Explanation(
        name="rebased_returns",
        category="function",
        summary=("Rescales a price series to a common starting value while preserving all relative price movements."),
        formula="price / first_price * base",
        how_to_read=("With a base of 100, a value of 150 means the investment increased by 50% from the starting point."),
        good_vs_bad=("Useful for visual comparison of assets with different price levels."),
        caveats=("Rebasing changes only the displayed scale. It does not change returns, risk, or performance statistics. "
            "Despite living in the returns module, this takes and returns a PRICE series, not a return series - see prices_from_returns for the returns-to-prices direction."),
        interpret=lambda v: f"{v:.2f} rebased value",
    )
)

register(
    Explanation(
        name="excess_returns",
        category="function",
        summary=("The return earned above a benchmark return or risk-free rate during the same period."),
        formula="portfolio return - benchmark or risk-free return",
        how_to_read=("A value of 0.03 means the portfolio outperformed the reference by 3% during that period."),
        good_vs_bad=("Positive excess return indicates outperformance relative to the chosen reference. Negative values indicate underperformance."),
        caveats=("Excess return does not account for risk taken. A higher excess return may simply come from accepting higher volatility."),
        interpret=lambda v: f"{v:+.2%} excess return",
    )
)

register(
    Explanation(
        name="active_returns",
        category="function",
        summary=("The return difference between a portfolio and its benchmark, representing the performance generated by active decisions."),
        formula="portfolio return - benchmark return",
        how_to_read=("A value of 0.01 means the portfolio exceeded the benchmark by 1% during that period."),
        good_vs_bad=("Positive active returns indicate benchmark outperformance. They should be evaluated with tracking error and information ratio."),
        caveats=("Active return alone does not measure consistency. A portfolio can have high active returns but poor risk-adjusted performance."),
        interpret=lambda v: f"{v:+.2%} active return",
    )
)
