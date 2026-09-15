"""
Optional currency-conversion helpers.

By default, PortPy assumes every asset in your DataFrame is denominated in the
same currency and never converts anything. If you do have dated FX-rate
series, use `convert_to_base_currency` before constructing a Portfolio to
bring everything into one base currency, and `calculate_fx_spread` to turn a
pair of bid/ask FX quotes into a spread-cost estimate for
`metrics.costs.fx_spread_costs`.
"""

from __future__ import annotations

import warnings

import pandas as pd


def convert_to_base_currency(
    prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    asset_currencies: dict[str, str],
    base_currency: str,
    max_staleness: int | None = None,
) -> pd.DataFrame:
    """
    Convert a multi-asset price DataFrame into a single base currency.

    Args:
        prices: Columns = asset symbols, index = dates, values = prices in each
            asset's native currency.
        fx_rates: Columns = 3-letter currency codes, index = dates, values = units
            of base_currency per 1 unit of that currency - a direct quote (e.g.
            an "EUR" column of 1.08 means 1 EUR = 1.08 of base_currency; invert
            an indirect vendor quote with 1 / rate before passing it in).
            Reindexed to prices.index and forward-filled, so it doesn't need to
            share the exact same calendar.
        asset_currencies: Mapping of asset symbol -> currency code. Assets already
            in base_currency (or missing from this dict) are left untouched.
        base_currency: The target currency code (e.g. "USD").
        max_staleness: If given, caps how many consecutive rows a forward-filled
            FX rate may be carried for (`DataFrame.ffill(limit=max_staleness)`)
            before it's treated as missing again (`NaN`, which then propagates
            into the converted price - a visible failure rather than a silently
            stale rate). `None` (default) keeps the old unlimited-ffill behavior.

    Returns:
        A new DataFrame, same shape as prices, denominated in base_currency.

    Raises:
        ValueError: If an asset's currency has no matching column in fx_rates.

    fx_rates must hold direct quotes (units of base_currency per unit of
    local currency). Most vendor data quotes pairs the other way round (e.g.
    USDJPY = JPY per USD, an indirect quote from a USD-base perspective) -
    feeding that in as-is does not raise, it silently multiplies prices by
    roughly 100-150x too much or too little; invert with 1 / rate if your feed
    is indirect. There's still no triangulation - if you have USD/EUR and
    USD/JPY but need EUR/JPY directly, this raises rather than cross-computing
    it. `max_staleness` addresses the other historical gap (an unlimited ffill
    silently carrying forward a months-old rate) but only if you set it - it
    stays unlimited by default for backward compatibility.
    """
    converted = prices.copy()
    fx_aligned = fx_rates.reindex(prices.index).ffill(limit=max_staleness)

    for asset, ccy in asset_currencies.items():
        if asset not in converted.columns or ccy == base_currency:
            continue
        if ccy not in fx_aligned.columns:
            raise ValueError(
                f"No FX rate column found for currency {ccy!r} (needed by asset {asset!r}). "
                f"fx_rates columns: {list(fx_rates.columns)}"
            )
        converted[asset] = converted[asset] * fx_aligned[ccy]

    return converted


def calculate_fx_spread(bid_rates: pd.DataFrame, ask_rates: pd.DataFrame) -> pd.DataFrame:
    """
    FX bid-ask spread as a fraction of the mid rate, per currency per date.

    Args:
        bid_rates: Columns = 3-letter currency codes, index = dates, values =
            the rate you'd receive selling that currency (units of base
            currency per unit of local currency, same direct-quote convention
            as `convert_to_base_currency`'s `fx_rates`).
        ask_rates: Same shape/columns/convention as `bid_rates` - the rate
            you'd pay buying that currency. Must be columned identically to
            `bid_rates` (same currency set); reindexed to `bid_rates.index`.

    Returns:
        `(ask - bid) / mid` per currency per date, where `mid = (ask+bid)/2` -
        a dimensionless spread fraction (0.001 = a 10 bps quoted spread),
        feed straight into `metrics.costs.fx_spread_costs` as `fx_spread_bps`
        after multiplying by 10,000.

    Raises:
        ValueError: If `bid_rates` and `ask_rates` don't share the same columns.

    This is the *quoted* spread implied by two rate series you already have -
    it doesn't fetch or estimate anything itself, and it says nothing about
    the cost of trading through more than the top of book (a large order can
    walk through several price levels, paying more than half this spread).
    Treat it as a lower bound on realistic FX conversion cost, not the whole story.
    """
    if set(bid_rates.columns) != set(ask_rates.columns):
        raise ValueError(
            f"bid_rates and ask_rates must have the same currency columns, "
            f"got {sorted(bid_rates.columns)} vs {sorted(ask_rates.columns)}."
        )
    ask_aligned = ask_rates.reindex(index=bid_rates.index, columns=bid_rates.columns)
    mid = (ask_aligned + bid_rates) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spread = (ask_aligned - bid_rates) / mid
    return spread
