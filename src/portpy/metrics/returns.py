"""Return transformations and return-based summary statistics.

Convention: every scalar function accepts `as_result=True` to get back a
:class:`~portpy.explain.MetricResult` (a float that also knows how to explain
itself) instead of a plain float. Annualization everywhere in PortPy uses the
period-count convention (``years = n_periods / periods_per_year``), matching
`empyrical`/`quantstats` defaults, rather than actual elapsed calendar time.
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
    """Period-over-period simple (arithmetic) returns: ``p_t / p_{t-1} - 1``."""
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Period-over-period log returns: ``ln(p_t / p_{t-1})``."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compounded cumulative return series: ``(1 + r).cumprod() - 1``."""
    return (1.0 + returns).cumprod() - 1.0  # type: ignore[attr-defined]


def prices_from_returns(returns: pd.Series | pd.DataFrame, base: float = 1.0) -> pd.Series | pd.DataFrame:
    """Reconstruct a price-like series from returns, anchored at `base` one period before the first return.

    This matches the structure of a real price series (whose first observation is
    a plain starting price, untouched by any return) - so
    ``simple_returns(prices_from_returns(r))`` recovers `r` exactly, including its
    first observation. Feeding a bare `(1 + r).cumprod()` into a drawdown/CAGR
    function instead silently drops the first period's return and can understate
    a drawdown that started immediately.
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
    """Rescale a price series so it starts at `base` (default 100), preserving pct changes."""
    return prices / prices.iloc[0] * base


def total_return(returns: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Compounded total return over the whole period: ``prod(1 + r) - 1``."""
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
    """Annualized return.

    Args:
        geometric: If True (default), compounds the total return and raises it
            to `periods_per_year / n_periods` (this is the CAGR of the return
            series). If False, simply scales the arithmetic mean return by
            `periods_per_year` - simpler, but ignores compounding.
    """
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    n = len(r)
    if geometric:
        total = float((1.0 + r).prod())
        years = n / periods_per_year
        value = total ** (1.0 / years) - 1.0 if years > 0 else np.nan
    else:
        value = float(r.mean()) * periods_per_year
    return MetricResult(value, "annualized_return", unit="%") if as_result else float(value)


def cagr(
    prices: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    as_result: bool = False,
) -> float | MetricResult:
    """Compound Annual Growth Rate, computed directly from a price series."""
    p = prices.dropna()
    ensure_min_observations(p, 2, "prices")
    n_periods = len(p) - 1
    years = n_periods / periods_per_year
    total = float(p.iloc[-1] / p.iloc[0])
    value = total ** (1.0 / years) - 1.0 if years > 0 else np.nan
    return MetricResult(value, "cagr", unit="%") if as_result else float(value)


def average_return(returns: pd.Series, geometric: bool = False, as_result: bool = False) -> float | MetricResult:
    """Average per-period return: arithmetic mean, or geometric mean if `geometric=True`."""
    r = returns.dropna()
    ensure_min_observations(r, 1, "returns")
    if geometric:
        value = float(np.exp(np.log1p(r).mean()) - 1.0)
    else:
        value = float(r.mean())
    return MetricResult(value, "average_return", unit="%") if as_result else value


def excess_returns(returns: pd.Series, benchmark_or_rf: pd.Series | float) -> pd.Series:
    """Per-period excess return: `returns` minus a benchmark series or a constant per-period rate."""
    if isinstance(benchmark_or_rf, pd.Series):
        joined = pd.concat([returns.rename("r"), benchmark_or_rf.rename("b")], axis=1, join="inner").dropna()
        return joined["r"] - joined["b"]
    return returns - benchmark_or_rf


def active_returns(returns: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Alias of :func:`excess_returns` restricted to a benchmark series (as opposed to a rate)."""
    return excess_returns(returns, benchmark)


register(
    Explanation(
        name="total_return",
        category="metric",
        summary=(
            "The compounded percentage gain or loss over the entire period, ignoring time - "
            "'if I put money in on day 1 and took it out on the last day, what's my total gain?'"
        ),
        formula="prod(1 + r_t) - 1",
        how_to_read="Read directly as a percentage: 0.35 means +35% over the whole period.",
        good_vs_bad=(
            "There's no universal 'good' threshold - it depends entirely on the length of the "
            "period and the asset class. Compare it to a relevant benchmark over the *same* "
            "period rather than judging it in isolation."
        ),
        caveats="Not annualized, so a 35% return over 1 year and 35% over 10 years look identical here.",
        interpret=lambda v: (
            f"{v:+.1%} total over the period"
            + (" (a net loss)" if v < 0 else "")
        ),
    )
)

register(
    Explanation(
        name="annualized_return",
        category="metric",
        summary="Total return rescaled to a 'per year' basis so periods of different lengths become comparable.",
        formula="geometric: (prod(1+r))^(periods_per_year/n) - 1  |  arithmetic: mean(r) * periods_per_year",
        how_to_read="A value of 0.10 means the return compounded (or averaged) to roughly +10%/year.",
        good_vs_bad=(
            "Compare to a benchmark (e.g. equities ~7-10%/yr historically before inflation) and to "
            "the asset's own volatility - a high annualized return with extreme volatility isn't "
            "automatically 'better' than a lower, steadier one."
        ),
        caveats=(
            "Geometric annualization is the academically correct way to compare returns of "
            "different lengths; arithmetic annualization overstates returns for volatile series "
            "(variance drag)."
        ),
        interpret=lambda v: f"{v:+.1%} per year" + (" (losing money annually)" if v < 0 else ""),
    )
)

register(
    Explanation(
        name="cagr",
        category="metric",
        summary=(
            "Compound Annual Growth Rate: the constant annual growth rate that would take the "
            "starting price to the ending price, smoothing out all the volatility in between."
        ),
        formula="(P_end / P_start)^(1/years) - 1",
        how_to_read="0.12 means the investment grew as if it compounded at +12%/year, every year.",
        good_vs_bad=(
            "Higher is better, all else equal, but always pair it with a risk metric (volatility, "
            "max drawdown) - a high CAGR achieved with wild swings may not be worth the ride."
        ),
        caveats=(
            "CAGR only looks at the start and end price - a portfolio that went up 200% and back "
            "down can have the same CAGR as one that grew steadily, if the start/end prices match."
        ),
        interpret=lambda v: f"{v:+.1%}/year compounded",
    )
)

register(
    Explanation(
        name="average_return",
        category="metric",
        summary="The typical per-period return - arithmetic mean by default, or geometric mean if requested.",
        formula="arithmetic: mean(r)  |  geometric: exp(mean(ln(1+r))) - 1",
        how_to_read="This is a *per-period* number (e.g. per day), not annualized - multiply/compound it yourself.",
        good_vs_bad="Depends on frequency and asset class; mostly useful for comparing two series measured over the same period length.",
        caveats=(
            "The arithmetic mean overstates what you'd actually earn compounding through volatile "
            "returns; the geometric mean matches realized compounded growth and is usually the "
            "more honest number."
        ),
        interpret=lambda v: f"{v:+.3%} per period on average",
    )
)
