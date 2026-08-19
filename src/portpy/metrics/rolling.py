"""
Rolling and expanding-window metric engines.

rolling_metric / expanding_metric are the generic engines the more specific
helpers below are built on - use them directly to roll any PortPy metric
function (or your own) over a window.
"""

from __future__ import annotations

from collections.abc import Callable

from typing import Any

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
    **kwargs: Any,
) -> pd.Series:
    """
    Apply any Series -> float metric function over a rolling window.

    Args:
        returns: Per-period returns.
        func: A callable taking a pd.Series window (plus **kwargs) and
            returning a float - any PortPy metric function works directly
            (pass as_result=False, the default).
        window: Window size, in periods.
        min_periods: Minimum observations required to produce a value.
            Defaults to `window` (no partial windows).
        **kwargs: Additional keyword arguments passed through to `func`.

    Returns:
        A Series of `func` evaluated on each rolling window, aligned to `returns`.
    """
    return returns.rolling(window, min_periods=min_periods or window).apply(
        lambda x: func(pd.Series(x, copy=False), **kwargs), raw=False
    )


def expanding_metric(
    returns: pd.Series,
    func: Callable[..., float],
    min_periods: int = 2,
    **kwargs: Any,
) -> pd.Series:
    """
    Apply any Series -> float metric function over an expanding (growing) window.

    Args:
        returns: Per-period returns.
        func: A callable taking a pd.Series window (plus **kwargs) and
            returning a float.
        min_periods: Minimum observations required to produce a value.
            Defaults to 2.
        **kwargs: Additional keyword arguments passed through to `func`.

    Returns:
        A Series of `func` evaluated on each expanding window, aligned to `returns`.
    """
    return returns.expanding(min_periods=min_periods).apply(
        lambda x: func(pd.Series(x, copy=False), **kwargs), raw=False
    )


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Sharpe ratio recomputed over a trailing window.

    Shows whether risk-adjusted performance is stable or decaying.

    Args:
        returns: Per-period returns.
        window: Window size, in periods.
        rf: Risk-free rate, as an annual rate. Defaults to DEFAULT_RISK_FREE_RATE.
        periods_per_year: Number of periods in a year, used to annualize.

    Returns:
        A Series of rolling Sharpe ratios, named "rolling_sharpe".
    """
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
    """
    Standard deviation recomputed over a trailing window; reveals volatility
    clustering and regime shifts.

    Args:
        returns: Per-period returns.
        window: Window size, in periods.
        annualized: If True (default), scale by sqrt(periods_per_year).
        periods_per_year: Number of periods in a year, used to annualize.

    Returns:
        A Series of rolling volatility, named "rolling_volatility".
    """
    vol = returns.rolling(window, min_periods=window).std(ddof=1)
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
    return vol.rename("rolling_volatility")


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    """
    Beta recomputed over a trailing window; reveals whether market sensitivity
    is stable over time.

    Args:
        returns: Per-period returns.
        benchmark: Per-period benchmark returns.
        window: Window size, in periods.

    Returns:
        A Series of rolling beta, named "rolling_beta".
    """
    r, b = align_pair(returns, benchmark)
    cov = r.rolling(window, min_periods=window).cov(b)
    var = b.rolling(window, min_periods=window).var(ddof=1)
    return (cov / var).rename("rolling_beta")


def rolling_correlation(returns_a: pd.Series, returns_b: pd.Series, window: int) -> pd.Series:
    """
    Pearson correlation recomputed over a trailing window.

    Args:
        returns_a: First per-period return series.
        returns_b: Second per-period return series.
        window: Window size, in periods.

    Returns:
        A Series of rolling correlation, named "rolling_correlation".
    """
    a, b = align_pair(returns_a, returns_b, "series_a", "series_b")
    return a.rolling(window, min_periods=window).corr(b).rename("rolling_correlation")


register(
    Explanation(
        name="rolling_sharpe",
        category="chart",
        summary="Sharpe ratio recalculated over a moving historical window to show whether risk-adjusted performance is stable, improving, or deteriorating through time.",
        formula="(average_return - risk_free_rate) / standard_deviation_of_returns",
        how_to_read="A consistently positive value indicates returns have compensated for risk during that period. Declining values indicate weakening risk-adjusted performance.",
        good_vs_bad="Prefer strategies with stable positive rolling Sharpe values rather than short periods of unusually high performance.",
        caveats="Rolling Sharpe is sensitive to the chosen window size. Short windows react quickly but are noisy, while long windows are more stable but slower to detect changes. Values are also autocorrelated by construction, since consecutive windows share most of their observations - don't treat the rolling series as independent draws for further statistical analysis.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="rolling_volatility",
        category="chart",
        summary="Volatility recalculated over a moving historical window to identify changes in risk levels, volatility clustering, and market regimes.",
        formula="rolling_standard_deviation(returns) * sqrt(periods_per_year)",
        how_to_read="Higher values indicate periods where returns are fluctuating more. Sudden increases often correspond to market stress or uncertainty.",
        good_vs_bad="Stable volatility is usually easier to manage. Rapid increases suggest rising portfolio risk and possible regime changes.",
        caveats="Volatility measures past variability, not future risk. The selected rolling window strongly affects how quickly changes appear. Values are also autocorrelated by construction, since consecutive windows share most of their observations - don't treat the rolling series as independent draws for further statistical analysis.",
        interpret=lambda v: f"{v:.2%}",
    )
)

register(
    Explanation(
        name="rolling_beta",
        category="chart",
        summary="Beta recalculated over a moving historical window to show how portfolio sensitivity to a benchmark changes over time.",
        formula="covariance(asset_returns, benchmark_returns) / variance(benchmark_returns)",
        how_to_read="Values above 1 indicate the portfolio tends to amplify benchmark movements. Values below 1 indicate lower sensitivity.",
        good_vs_bad="Stable beta values make risk management easier. Large shifts indicate changing market exposure or unstable relationships.",
        caveats="Beta depends on the selected benchmark and rolling window. Historical relationships may break during market regime changes. Values are also autocorrelated by construction, since consecutive windows share most of their observations - don't treat the rolling series as independent draws for further statistical analysis.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="rolling_correlation",
        category="chart",
        summary="Correlation between two return series recalculated over a moving historical window to show whether their relationship changes through time.",
        formula="covariance(asset_a, asset_b) / (volatility_a * volatility_b)",
        how_to_read="Values near 1 indicate the assets move together. Values near -1 indicate opposite movement. Values near 0 indicate weak relationship.",
        good_vs_bad="Lower correlation between portfolio assets generally improves diversification. Rising correlation can reduce diversification benefits.",
        caveats="Correlation is dynamic and can increase sharply during market stress, reducing the protection expected from diversification. Values are also autocorrelated by construction, since consecutive windows share most of their observations - don't treat the rolling series as independent draws for further statistical analysis.",
        interpret=lambda v: f"{v:.2f}",
    )
)

register(
    Explanation(
        name="rolling_metric",
        category="function",
        summary="Generic engine that applies any metric function repeatedly over a moving historical window.",
        formula="metric(window_returns_t)",
        how_to_read="Each output value represents the metric calculated using only the observations inside the current rolling window.",
        good_vs_bad="Useful for detecting changes in strategy behavior, risk, and performance characteristics over time.",
        caveats="The usefulness depends on the metric being applied and the chosen window size. Short windows increase noise while long windows reduce responsiveness. It also re-runs the full Python callable per window via .apply(..., raw=False), which is slower than the vectorized rolling_sharpe/rolling_volatility/rolling_beta/rolling_correlation when a dedicated version of the metric exists - prefer those. Output is autocorrelated by construction since consecutive windows overlap, so don't treat it as independent observations.",
    )
)

register(
    Explanation(
        name="expanding_metric",
        category="function",
        summary="Generic engine that applies any metric function using an expanding dataset that starts small and grows as new observations arrive.",
        formula="metric(all_returns_available_until_t)",
        how_to_read="The metric starts with limited information and becomes more stable as more observations accumulate.",
        good_vs_bad="Useful for tracking how estimates converge over time and how early performance compares with long-term history.",
        caveats="Early values can be unreliable because they are calculated from very few observations. Like rolling_metric, it also re-runs the full Python callable per window via .apply(..., raw=False), which is slow over long histories, and its output is autocorrelated by construction since windows overlap heavily.",
    )
)