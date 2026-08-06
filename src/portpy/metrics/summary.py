"""
One-shot aggregate summaries built out of the other metric modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, register
from portpy.metrics import benchmarks as _benchmarks
from portpy.metrics import distributions as _distributions
from portpy.metrics import drawdowns as _drawdowns
from portpy.metrics import performance as _performance
from portpy.metrics import returns as _returns
from portpy.metrics import risk as _risk
from portpy.utils.constants import DEFAULT_CONFIDENCE_LEVEL, DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import align_pair

__all__ = ["tearsheet_summary", "compare_to_benchmark"]


def tearsheet_summary(
    returns: pd.Series,
    prices: pd.Series | None = None,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Compute the key headline metrics in one call, each as a self-explaining MetricResult.
    """
    r = returns.dropna()
    price_series = prices if prices is not None else _returns.prices_from_returns(r)

    return {
        "total_return": _returns.total_return(r, as_result=True),
        "annualized_return": _returns.annualized_return(r, periods_per_year=periods_per_year, as_result=True),
        "cagr": _returns.cagr(price_series, periods_per_year=periods_per_year, as_result=True),
        "volatility": _risk.volatility(r, periods_per_year=periods_per_year, as_result=True),
        "sharpe_ratio": _performance.sharpe_ratio(r, rf=rf, periods_per_year=periods_per_year, as_result=True),
        "sortino_ratio": _performance.sortino_ratio(r, periods_per_year=periods_per_year, as_result=True),
        "calmar_ratio": _performance.calmar_ratio(r, periods_per_year=periods_per_year, as_result=True),
        "max_drawdown": _drawdowns.max_drawdown(price_series, as_result=True),
        "value_at_risk_95": _risk.value_at_risk(r, confidence=DEFAULT_CONFIDENCE_LEVEL, as_result=True),
        "conditional_var_95": _risk.conditional_var(r, confidence=DEFAULT_CONFIDENCE_LEVEL, as_result=True),
        "skewness": _risk.skewness(r, as_result=True),
        "kurtosis": _risk.kurtosis(r, as_result=True),
        "win_rate": _distributions.win_rate(r, as_result=True),
    }


def compare_to_benchmark(
    returns: pd.Series,
    benchmark: pd.Series,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Side-by-side portfolio-vs-benchmark table: absolute metrics for both, plus relative metrics.
    """
    r, b = align_pair(returns, benchmark)
    synthetic_r = _returns.prices_from_returns(r)
    synthetic_b = _returns.prices_from_returns(b)

    rows: dict[str, dict[str, float]] = {
        "annualized_return": {
            "portfolio": _returns.annualized_return(r, periods_per_year=periods_per_year),
            "benchmark": _returns.annualized_return(b, periods_per_year=periods_per_year),
        },
        "volatility": {
            "portfolio": _risk.volatility(r, periods_per_year=periods_per_year),
            "benchmark": _risk.volatility(b, periods_per_year=periods_per_year),
        },
        "sharpe_ratio": {
            "portfolio": _performance.sharpe_ratio(r, rf=rf, periods_per_year=periods_per_year),
            "benchmark": _performance.sharpe_ratio(b, rf=rf, periods_per_year=periods_per_year),
        },
        "max_drawdown": {
            "portfolio": _drawdowns.max_drawdown(synthetic_r),
            "benchmark": _drawdowns.max_drawdown(synthetic_b),
        },
        "beta": {"portfolio": _risk.beta(r, b), "benchmark": np.nan},
        "alpha": {"portfolio": _benchmarks.alpha(r, b, rf=rf, periods_per_year=periods_per_year), "benchmark": np.nan},
        "correlation": {"portfolio": _benchmarks.correlation(r, b), "benchmark": np.nan},
        "information_ratio": {
            "portfolio": _performance.information_ratio(r, b, periods_per_year=periods_per_year),
            "benchmark": np.nan,
        },
        "up_capture_ratio": {
            "portfolio": _benchmarks.up_capture_ratio(r, b, periods_per_year=periods_per_year),
            "benchmark": np.nan,
        },
        "down_capture_ratio": {
            "portfolio": _benchmarks.down_capture_ratio(r, b, periods_per_year=periods_per_year),
            "benchmark": np.nan,
        },
        "batting_average": {"portfolio": _benchmarks.batting_average(r, b), "benchmark": np.nan},
    }

    df = pd.DataFrame(rows).T
    df["difference"] = df["portfolio"] - df["benchmark"]
    df.attrs["portpy_explanation"] = "compare_to_benchmark"
    return df


register(
    Explanation(
        name="tearsheet_summary",
        category="metric",
        summary="A complete portfolio overview combining return, risk, drawdown, distribution, and risk-adjusted performance metrics into a single analysis output.",
        formula="summary = {performance_metrics + risk_metrics + drawdown_metrics + distribution_metrics}",
        how_to_read="Review the metrics together rather than individually. Returns describe reward, volatility and drawdowns describe risk, and ratios describe the efficiency of the return generated.",
        good_vs_bad="A strong portfolio typically shows competitive returns, controlled volatility, limited drawdowns, favorable risk-adjusted ratios, and consistent return behavior.",
        caveats="The summary does not rank portfolios automatically. Different strategies optimize different combinations of return, risk, liquidity, and drawdown characteristics.",
        interpret=lambda v: "Portfolio performance and risk summary",
    )
)

register(
    Explanation(
        name="compare_to_benchmark",
        category="metric",
        summary="A structured comparison between a portfolio and a benchmark showing absolute performance, risk characteristics, and benchmark-relative statistics.",
        formula="comparison = portfolio_metrics - benchmark_metrics + relative_metrics",
        how_to_read="Positive differences generally indicate portfolio outperformance for return-based metrics. For risk metrics, the preferred direction depends on the objective, such as lower volatility or smaller drawdown.",
        good_vs_bad="A favorable comparison usually combines higher risk-adjusted returns, lower downside risk, positive alpha, strong information ratio, and appropriate benchmark exposure.",
        caveats="The quality of the comparison depends on benchmark selection. A poorly chosen benchmark can make relative performance conclusions misleading.",
        interpret=lambda v: "Portfolio versus benchmark comparison table",
    )
)