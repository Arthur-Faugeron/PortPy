"""Rolling and expanding-window metric engines.

`rolling_metric` / `expanding_metric` are the generic engines the more specific
helpers below are built on - use them directly to roll *any* PortPy metric
function (or your own) over a window.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from portpy.explain import Explanation, register
from portpy.utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import align_pair

__all__ = [
    "rolling_metric",
    "rolling_sharpe",
    "rolling_volatility",
    "rolling_beta",
    "rolling_correlation",
    "expanding_metric",
]


def rolling_metric(
    returns: pd.Series,
    func: Callable[..., float],
    window: int,
    min_periods: int | None = None,
    **kwargs,
) -> pd.Series:
    """Apply any Series -> float metric function over a rolling window.

    Args:
        func: A callable taking a `pd.Series` window (plus `**kwargs`) and
            returning a float - any PortPy metric function works directly
            (pass `as_result=False`, the default, since MetricResult inside a
            rolling apply adds no value).
        window: Window size, in periods.
        min_periods: Minimum observations required to produce a value;
            defaults to `window` (no partial windows).
    """
    return returns.rolling(window, min_periods=min_periods or window).apply(
        lambda x: func(pd.Series(x, copy=False), **kwargs), raw=False
    )


def expanding_metric(
    returns: pd.Series,
    func: Callable[..., float],
    min_periods: int = 2,
    **kwargs,
) -> pd.Series:
    """Apply any Series -> float metric function over an expanding (growing) window."""
    return returns.expanding(min_periods=min_periods).apply(
        lambda x: func(pd.Series(x, copy=False), **kwargs), raw=False
    )


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Sharpe ratio recomputed over a trailing window - shows whether risk-adjusted performance is stable or decaying."""
    from portpy.metrics.performance import sharpe_ratio

    return rolling_metric(
        returns, sharpe_ratio, window, rf=rf, periods_per_year=periods_per_year, annualized=True
    ).rename("rolling_sharpe")


def rolling_volatility(
    returns: pd.Series,
    window: int,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Standard deviation recomputed over a trailing window - reveals volatility clustering/regime shifts."""
    vol = returns.rolling(window, min_periods=window).std(ddof=1)
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
    return vol.rename("rolling_volatility")


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    """Beta recomputed over a trailing window - reveals whether market sensitivity is stable over time."""
    r, b = align_pair(returns, benchmark)
    cov = r.rolling(window, min_periods=window).cov(b)
    var = b.rolling(window, min_periods=window).var(ddof=1)
    return (cov / var).rename("rolling_beta")


def rolling_correlation(returns_a: pd.Series, returns_b: pd.Series, window: int) -> pd.Series:
    """Pearson correlation recomputed over a trailing window."""
    a, b = align_pair(returns_a, returns_b, "series_a", "series_b")
    return a.rolling(window, min_periods=window).corr(b).rename("rolling_correlation")


register(
    Explanation(
        name="rolling_sharpe",
        category="chart",
        summary="Sharpe ratio recomputed on a trailing window (e.g. 6 or 12 months) at every point in time, instead of once over the whole history.",
        how_to_read="A flat, high line means consistently good risk-adjusted performance. A line that decays toward zero or goes negative flags performance that hasn't held up recently, even if the full-history Sharpe still looks good.",
        good_vs_bad="Prefer strategies whose rolling Sharpe stays consistently positive and doesn't show a strong recent downtrend.",
        caveats="Short windows are noisy; a single bad week can swing a 3-month rolling Sharpe a lot. Use a window long enough to contain multiple return cycles.",
    )
)

register(
    Explanation(
        name="rolling_volatility",
        category="chart",
        summary="Annualized volatility recomputed on a trailing window - the standard way to see volatility clustering (calm periods vs. turbulent ones).",
        how_to_read="Spikes usually coincide with market stress; a rising baseline over time suggests the strategy/asset is structurally getting riskier.",
        good_vs_bad="Lower and more stable is generally more comfortable to hold, though it depends entirely on the strategy's goals (a vol-targeting strategy *should* show a flat line by design).",
    )
)

register(
    Explanation(
        name="rolling_beta",
        category="chart",
        summary="Beta to a benchmark recomputed on a trailing window - shows whether market sensitivity is stable or shifts over time.",
        how_to_read="A beta that jumps around a lot (e.g. from 0.5 to 1.5) means the portfolio's relationship to the benchmark is unstable - risk models calibrated on the full-history beta may be unreliable going forward.",
        good_vs_bad="Stability is usually more important than the specific level - a portfolio with a steady beta of 0.8 is easier to risk-manage than one oscillating between 0.3 and 1.3.",
    )
)

register(
    Explanation(
        name="rolling_correlation",
        category="chart",
        summary="Correlation between two return series recomputed on a trailing window.",
        how_to_read="Values near +1 mean the series move together; near -1 they move oppositely; near 0 they're roughly independent over that window.",
        good_vs_bad="For diversification purposes, lower (or more negative) is usually better between assets in the same portfolio. Watch for correlations that spike toward +1 during market stress ('correlations go to 1 in a crisis').",
    )
)
