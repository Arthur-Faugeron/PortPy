"""Optional transaction-cost modeling.

PortPy assumes frictionless trading by default. These helpers let you layer
transaction costs back in, either from an explicit weight-history (to compute
real turnover) or from an assumed constant per-period turnover.
"""

from __future__ import annotations

import pandas as pd

from portpy.explain import Explanation, register
from portpy.utils.validation import ensure_min_observations

__all__ = ["turnover_from_weights", "net_of_costs_returns"]


def turnover_from_weights(weights_history: pd.DataFrame) -> pd.Series:
    """Per-period portfolio turnover: half the sum of absolute weight changes (the fraction of the book traded).

    The first row is treated as building the initial portfolio from cash, so its
    turnover equals the sum of the initial (absolute) weights, halved.
    """
    ensure_min_observations(weights_history, 1, "weights_history")
    turnover = weights_history.diff().abs().sum(axis=1) / 2.0
    turnover.iloc[0] = float(weights_history.iloc[0].abs().sum()) / 2.0
    return turnover.rename("turnover")


def net_of_costs_returns(
    returns: pd.Series,
    turnover: pd.Series | float,
    cost_bps: float,
) -> pd.Series:
    """Deduct a transaction-cost drag from `returns`, given per-period turnover and a cost in basis points.

    Args:
        turnover: Either a per-period turnover Series (see `turnover_from_weights`,
            aligned to `returns.index`) or a single constant turnover fraction
            applied to every period.
        cost_bps: Round-trip cost per unit of notional traded, in basis points
            (e.g. 10 = 0.10% per unit turnover).
    """
    cost_rate = cost_bps / 10_000.0
    if isinstance(turnover, pd.Series):
        cost_drag = turnover.reindex(returns.index).fillna(0.0) * cost_rate
    else:
        cost_drag = turnover * cost_rate
    return returns - cost_drag


register(
    Explanation(
        name="turnover_from_weights",
        category="metric",
        summary="How much of the portfolio was bought/sold each period, as a fraction of its value - the raw input needed to estimate transaction costs.",
        formula="sum(|w_t - w_(t-1)|) / 2",
        how_to_read="A turnover of 0.20 means 20% of the portfolio's value was traded that period (e.g. selling 10% of one asset and buying 10% of another).",
        good_vs_bad="Lower turnover means lower cost drag, all else equal - but some strategies (momentum, mean-reversion) inherently need higher turnover to work.",
    )
)

register(
    Explanation(
        name="net_of_costs_returns",
        category="metric",
        summary="Returns after subtracting an estimated transaction-cost drag, so you can see performance net of realistic trading friction instead of a frictionless ideal.",
        formula="returns - turnover * (cost_bps / 10,000)",
        how_to_read="Compare cumulative net-of-cost returns to the frictionless version - a strategy that looks great gross but falls apart net of costs is a common backtest trap (especially for high-turnover strategies).",
        good_vs_bad="The smaller the gap between gross and net performance, the more robust the strategy is to real-world execution costs.",
    )
)
