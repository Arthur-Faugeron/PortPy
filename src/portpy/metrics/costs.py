"""
Optional transaction-cost modeling.

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
    """
    Per-period portfolio turnover: half the sum of absolute weight changes,
    i.e. the fraction of the book traded each period.

    The first row is treated as building the initial portfolio from cash, so
    its turnover equals half the sum of the initial absolute weights.

    Args:
        weights_history: Portfolio weights per period, indexed by date with
            one column per asset.

    Returns:
        Per-period turnover as a Series named "turnover", aligned to
        `weights_history.index`.
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
    """
    Deduct a transaction-cost drag from `returns`, given per-period turnover
    and a cost in basis points. Uses a flat, linear-in-turnover cost model.

    Args:
        returns: Periodic gross returns.
        turnover: Either a per-period turnover Series (see
            `turnover_from_weights`, aligned to `returns.index`) or a single
            constant turnover fraction applied to every period.
        cost_bps: Round-trip cost per unit of notional traded, in basis
            points (e.g. 10 = 0.10% per unit turnover).

    Returns:
        `returns` net of the estimated transaction-cost drag.
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
        category="function",
        summary="Measures how much of the portfolio is traded between periods by analyzing changes in portfolio allocation weights.",
        formula="0.5 * sum(abs(current_weight - previous_weight))",
        how_to_read="A turnover value of 0.20 means that 20% of portfolio value was exchanged during that period. The metric reflects portfolio trading activity rather than investment performance.",
        good_vs_bad="Lower turnover usually means lower implementation costs and less trading drag. Higher turnover can be justified when the strategy generates enough excess return to compensate for additional costs.",
        caveats="Weight-based turnover does not estimate actual execution costs. It ignores market impact, spreads, commissions, taxes, leverage changes, and liquidity constraints. Treating period 0 as building the position from cash is a real modeling choice that can double-count for a book that already held a position before the sample window - overwrite turnover.iloc[0] if you don't want that phantom charge. Also assumes a fully-invested book; with cash drag, sum(|delta w|)/2 no longer exactly equals dollars-traded/portfolio-value.",
        interpret=lambda v: f"{v:.2%} traded during period",
    )
)

register(
    Explanation(
        name="net_of_costs_returns",
        category="function",
        summary="Adjusts portfolio returns by subtracting estimated transaction costs to show performance after implementation expenses.",
        formula="gross_return - turnover * (cost_bps / 10000)",
        how_to_read="Compare net returns with gross returns to understand how much performance is consumed by trading costs.",
        good_vs_bad="A small difference between gross and net returns indicates a strategy is more robust to implementation costs. A large difference suggests trading activity significantly reduces realized performance.",
        caveats="The estimate assumes a simplified constant transaction cost. Real trading costs vary with liquidity, order size, market conditions, execution method, and asset characteristics. This is a flat, linear-in-turnover model - it does not capture market impact (super-linear in trade size relative to average daily volume), bid-ask asymmetry, or intraperiod execution-timing mismatch; treat it as a first-order approximation only.",
        interpret=lambda v: f"{v:.2%} return after estimated costs",
    )
)
