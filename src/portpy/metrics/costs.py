"""
Optional transaction-cost modeling.

PortPy assumes frictionless trading by default. These helpers let you layer
transaction costs back in from an explicit weight-history: an aggregate flat
rate (`net_of_costs_returns`, given turnover from `turnover_from_weights`),
or itemized by source (`bid_ask_spreads`, `broker_commissions`,
`fx_spread_costs`, `tax_impact`) - the itemized functions all return a
per-period return-drag Series in the same units, so they compose by plain
addition: `total_drag = bid_ask_spreads(...) + broker_commissions(...) + ...`,
then `returns - total_drag`.

Every function here is a flat, per-period approximation, not a trade-level
execution or accounting engine - see each function's own caveats for exactly
what it ignores.
"""

from __future__ import annotations

import pandas as pd

from portpy.explain import Explanation, register
from portpy.utils.validation import ensure_min_observations

__all__ = [
    "turnover_from_weights",
    "net_of_costs_returns",
    "bid_ask_spreads",
    "broker_commissions",
    "fx_spread_costs",
    "tax_impact",
]


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
        return (returns - cost_drag).rename(returns.name)
    return returns - turnover * cost_rate


def _per_asset_rate(rate: float | dict[str, float] | pd.Series, columns: pd.Index) -> pd.Series:
    """Broadcast a scalar/dict/Series bps-rate input to one value per column, in bps."""
    if isinstance(rate, (int, float)):
        return pd.Series(float(rate), index=columns)
    aligned = pd.Series(rate).reindex(columns)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        raise ValueError(f"rate is missing a value for asset(s): {missing}")
    return aligned.astype(float)


def _per_asset_trade_fractions(weights_history: pd.DataFrame) -> pd.DataFrame:
    """Per-asset, per-period one-leg trade size |delta w_i,t| - the first row is a full buy from cash."""
    ensure_min_observations(weights_history, 1, "weights_history")
    trades = weights_history.diff().abs()
    trades.iloc[0] = weights_history.iloc[0].abs()
    return trades


def bid_ask_spreads(
    weights_history: pd.DataFrame,
    spread_bps: float | dict[str, float] | pd.Series,
) -> pd.Series:
    """
    Per-period cost of crossing the bid-ask spread, from an explicit weight history.

    A market order pays roughly half the quoted spread away from the mid
    price on each leg (buy or sell) - this charges that half-spread on every
    asset's traded notional each period, then sums across assets.

    Args:
        weights_history: Portfolio weights per period, indexed by date with
            one column per asset - same shape `turnover_from_weights` takes,
            but per-asset trades matter here (not just their net sum), since
            spreads can differ by asset.
        spread_bps: Quoted round-trip bid-ask spread, in basis points, either
            one number applied to every asset or a `{asset: bps}` dict/Series
            for per-asset (or, via a dict comprehension over your own
            region/asset-class mapping, per-region) spreads.

    Returns:
        Per-period cost drag (a positive return-rate reduction) as a Series
        named "bid_ask_spread_cost", aligned to `weights_history.index`. The
        first row charges the full initial allocation, matching
        `turnover_from_weights`' "built from cash" convention.
    """
    trades = _per_asset_trade_fractions(weights_history)
    rate = _per_asset_rate(spread_bps, weights_history.columns) / 10_000.0
    cost = (trades * rate) / 2.0
    return cost.sum(axis=1).rename("bid_ask_spread_cost")


def broker_commissions(
    weights_history: pd.DataFrame,
    commission_bps: float | dict[str, float] | pd.Series,
) -> pd.Series:
    """
    Per-period broker commission cost, proportional to traded notional.

    Args:
        weights_history: See `bid_ask_spreads`.
        commission_bps: Commission rate in basis points of traded notional,
            either one number for every asset or a `{asset: bps}` dict/Series.
            PortPy works in weights/returns, not share counts or dollar NAV,
            so a fixed-currency-amount-per-trade commission schedule (e.g.
            "$1 per trade" or "$0.005/share") can't be expressed here directly -
            convert it to an equivalent bps-of-notional rate for your typical
            trade size first if that's your actual fee schedule.

    Returns:
        Per-period cost drag as a Series named "broker_commission_cost",
        aligned to `weights_history.index`.
    """
    trades = _per_asset_trade_fractions(weights_history)
    rate = _per_asset_rate(commission_bps, weights_history.columns) / 10_000.0
    cost = trades * rate
    return cost.sum(axis=1).rename("broker_commission_cost")


def fx_spread_costs(
    weights_history: pd.DataFrame,
    asset_currencies: dict[str, str],
    base_currency: str,
    fx_spread_bps: float | dict[str, float] | pd.Series,
) -> pd.Series:
    """
    Per-period cost of crossing the FX bid-ask spread when trading assets
    denominated in a currency other than `base_currency`.

    Every trade in a foreign-currency asset implicitly converts currency too
    (buying a EUR-denominated stock from a USD book means buying EUR first) -
    this charges the same half-spread convention as `bid_ask_spreads`, but
    only on assets whose currency differs from `base_currency`, using an FX
    spread rather than that asset's own equity/bond spread.

    Args:
        weights_history: See `bid_ask_spreads`.
        asset_currencies: Mapping of asset symbol -> currency code. Assets
            already in `base_currency` (or missing from this dict) incur no
            FX spread cost.
        base_currency: The portfolio's base currency code.
        fx_spread_bps: Quoted FX bid-ask spread, in basis points, either one
            number for every non-base currency or a `{currency: bps}`
            dict/Series (not per-asset - see `calculate_fx_spread` to derive
            this from real bid/ask FX quotes, multiplying its fractional
            output by 10,000).

    Returns:
        Per-period cost drag as a Series named "fx_spread_cost", aligned to
        `weights_history.index`. All-zero if every asset is already in
        `base_currency`.
    """
    trades = _per_asset_trade_fractions(weights_history)
    foreign_assets = [a for a in weights_history.columns if asset_currencies.get(a, base_currency) != base_currency]
    if not foreign_assets:
        return pd.Series(0.0, index=weights_history.index, name="fx_spread_cost")

    currencies_needed = {asset_currencies[a] for a in foreign_assets}
    rate_by_currency = _per_asset_rate(fx_spread_bps, pd.Index(sorted(currencies_needed))) / 10_000.0

    cost = pd.Series(0.0, index=weights_history.index)
    for asset in foreign_assets:
        cost = cost + trades[asset] * rate_by_currency[asset_currencies[asset]] / 2.0
    return cost.rename("fx_spread_cost")


def tax_impact(
    returns: pd.Series,
    turnover: pd.Series | float,
    capital_gains_tax_rate: float,
    dividend_tax_rate: float = 0.0,
    dividend_yield: float | pd.Series = 0.0,
) -> pd.Series:
    """
    Estimate a per-period tax drag on realized capital gains plus dividend income.

    This is a first-order, turnover-proxy approximation, not a lot-level tax
    engine (see caveats) - it assumes the *fraction of the book turned over*
    each period approximates the fraction of that period's gain that gets
    realized (and therefore taxed) rather than deferred by staying invested,
    and taxes any assumed dividend yield separately (dividends are taxed
    whether or not you sell).

    Args:
        returns: Periodic gross returns.
        turnover: Either a per-period turnover Series (see
            `turnover_from_weights`) or a single constant turnover fraction -
            same convention as `net_of_costs_returns`.
        capital_gains_tax_rate: Tax rate applied to the realized-gain estimate
            in periods with a positive return (e.g. 0.15 for a 15% long-term
            US federal rate, 0.37 for short-term-as-ordinary-income, or a
            blended local+foreign effective rate you've computed yourself).
            Losses are never taxed here (no loss-harvesting credit either -
            see caveats).
        dividend_tax_rate: Tax rate applied to `dividend_yield`, independent
            of realized capital gains. Defaults to 0 (no dividend tax modeled).
        dividend_yield: Assumed per-period dividend yield, as a return
            fraction - a single constant or a Series aligned to `returns.index`.
            Defaults to 0. This is a return you assume is *already included*
            in `returns` and are separately taxing, not an addition to it.

    Returns:
        `returns` net of the estimated tax drag.

    This is a deliberately simplified, single-effective-rate approximation:
    it has no per-lot cost basis, no distinction between short-term and
    long-term holding periods beyond whatever single `capital_gains_tax_rate`
    you pass in, no loss carryforward or tax-loss harvesting, no wash-sale
    rule, and no jurisdiction-specific treatment of local vs. foreign
    withholding beyond a blended rate you supply yourself. Using turnover as
    a realization proxy is a common simplifying assumption in tax-drag
    literature, not an actual accounting rule - a buy-and-hold book with zero
    turnover pays zero estimated capital-gains tax here even though the
    portfolio's value has genuinely appreciated (correctly reflecting that
    unrealized gains aren't taxed yet, but only if you also track cost basis
    yourself when gains are eventually realized outside this function).
    """
    turnover_series = turnover if isinstance(turnover, pd.Series) else pd.Series(turnover, index=returns.index)
    turnover_series = turnover_series.reindex(returns.index).fillna(0.0).clip(upper=1.0)

    realized_gain = returns.clip(lower=0.0) * turnover_series
    capital_gains_drag = realized_gain * capital_gains_tax_rate

    dividend_series = dividend_yield if isinstance(dividend_yield, pd.Series) else pd.Series(dividend_yield, index=returns.index)
    dividend_series = dividend_series.reindex(returns.index).fillna(0.0)
    dividend_drag = dividend_series * dividend_tax_rate

    return (returns - capital_gains_drag - dividend_drag).rename(returns.name)


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
        caveats="The estimate assumes a simplified constant transaction cost. Real trading costs vary with liquidity, order size, market conditions, execution method, and asset characteristics. This is a flat, linear-in-turnover model - it does not capture market impact (super-linear in trade size relative to average daily volume), bid-ask asymmetry, or intraperiod execution-timing mismatch; treat it as a first-order approximation only. For itemized costs by source, see bid_ask_spreads/broker_commissions/fx_spread_costs/tax_impact, which compose by addition instead of one blended cost_bps.",
        interpret=lambda v: f"{v:.2%} return after estimated costs",
    )
)

register(
    Explanation(
        name="bid_ask_spreads",
        category="function",
        summary="Estimates the cost of crossing the bid-ask spread on every trade, from an explicit per-asset weight history.",
        formula="sum_i(|delta w_i,t| * spread_bps_i / 10000 / 2), first period charges the full initial allocation",
        how_to_read="A per-period cost drag in return units, same convention as net_of_costs_returns' cost_drag - subtract it from returns directly, or add it to other itemized cost Series first.",
        good_vs_bad="Lower spread cost for the same amount of rebalancing activity means either tighter-spread (typically larger, more liquid) assets or per-asset spread assumptions that may be too optimistic - check the spread_bps you supplied against real quoted spreads for the actual instruments.",
        caveats="Uses quoted top-of-book spread and a flat half-spread-per-leg convention - it says nothing about market impact from order size relative to average daily volume, and a large order can walk through multiple price levels, paying meaningfully more than half the quoted spread. spread_bps is an input you supply, not observed or estimated from data; garbage in, garbage out.",
        interpret=lambda v: f"{v:.4%} cost drag" if hasattr(v, "__float__") else f"mean {v.mean():.4%} cost drag/period",
    )
)

register(
    Explanation(
        name="broker_commissions",
        category="function",
        summary="Estimates broker commission cost, proportional to traded notional, from an explicit per-asset weight history.",
        formula="sum_i(|delta w_i,t| * commission_bps_i / 10000), first period charges the full initial allocation",
        how_to_read="A per-period cost drag in return units - compose with bid_ask_spreads/fx_spread_costs/tax_impact by addition before subtracting from returns.",
        good_vs_bad="No universal 'good value' - judge commission_bps against your actual broker's fee schedule, converted to a bps-of-notional rate for your typical trade size.",
        caveats="Only expresses commissions as a rate on traded notional - a fixed-currency-per-trade or per-share commission schedule needs converting to an equivalent bps rate yourself, since this package works in weights/returns, not share counts or dollar NAV. Ignores minimum-ticket fees, which make small rebalances proportionally far more expensive than this linear model implies.",
        interpret=lambda v: f"{v:.4%} cost drag" if hasattr(v, "__float__") else f"mean {v.mean():.4%} cost drag/period",
    )
)

register(
    Explanation(
        name="fx_spread_costs",
        category="function",
        summary="Estimates the FX bid-ask spread cost incurred when trading assets denominated in a currency other than the portfolio's base currency.",
        formula="sum_{i: ccy(i)!=base}(|delta w_i,t| * fx_spread_bps_ccy(i) / 10000 / 2)",
        how_to_read="Zero whenever every asset shares the portfolio's base_currency; otherwise a per-period cost drag in return units, same composition convention as bid_ask_spreads.",
        good_vs_bad="A materially non-zero fx_spread_cost is the price of holding foreign-currency assets and rebalancing them - not itself good or bad, but worth comparing against the diversification benefit those assets provide.",
        caveats="fx_spread_bps is keyed by currency, not asset - use calculate_fx_spread (times 10,000) if you have real bid/ask FX quotes instead of an assumption. Ignores that FX spreads themselves vary by trade size and time of day (thinner outside major-session overlap hours), and assumes every currency conversion happens at the same spread regardless of counterparty/venue.",
        interpret=lambda v: f"{v:.4%} cost drag" if hasattr(v, "__float__") else f"mean {v.mean():.4%} cost drag/period",
    )
)

register(
    Explanation(
        name="tax_impact",
        category="function",
        summary="Estimates a tax drag on returns from realized capital gains (proxied by turnover) plus a dividend tax on an assumed dividend yield.",
        formula="returns - clip(returns, min=0)*turnover*capital_gains_tax_rate - dividend_yield*dividend_tax_rate",
        how_to_read="A widening gap between gross and tax-adjusted returns over a high-turnover period means the estimated realized-gains tax is doing real work - compare against net_of_costs_returns' transaction-cost drag to see which friction dominates.",
        good_vs_bad="Lower estimated tax drag for the same investment thesis (e.g. via lower turnover, tax-advantaged account assumptions, or long-term vs. short-term rate choice) is generically 'better' after-tax, all else equal - but don't tune capital_gains_tax_rate downward just to make a backtest look better than your actual tax situation.",
        caveats="A first-order, single-effective-rate approximation, not a lot-level tax engine: no cost basis, no short/long-term distinction beyond whichever single rate you pass in, no loss carryforward or tax-loss harvesting, no wash-sale rule, and no jurisdiction-specific local-vs-foreign withholding beyond a blended rate you supply. Using turnover as a realization proxy is a common simplifying assumption in the tax-drag literature, not an accounting rule - a zero-turnover book pays zero estimated tax here regardless of how much it has genuinely appreciated, which is only correct if you separately track and tax the eventual realization yourself.",
        interpret=lambda v: f"{v:.2%} return after estimated tax" if hasattr(v, "__float__") else f"mean {v.mean():.4%} return/period after estimated tax",
    )
)
