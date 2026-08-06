"""
Optional currency-conversion helper.

By default, PortPy assumes every asset in your DataFrame is denominated in the
same currency and never converts anything. If you *do* have a dated FX-rate
series, use `convert_to_base_currency` before constructing a `Portfolio` to bring
everything into one base currency.
"""

from __future__ import annotations

import pandas as pd


def convert_to_base_currency(
    prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    asset_currencies: dict[str, str],
    base_currency: str,
) -> pd.DataFrame:
    """
    Convert a multi-asset price DataFrame into a single base currency.

    Args:
        prices: Columns = asset symbols, index = dates, values = prices in each
            asset's *native* currency.
        fx_rates: Columns = 3-letter currency codes, index = dates, values = units
            of `base_currency` per 1 unit of that currency (a direct multiplier -
            e.g. an "EUR" column of 1.08 means 1 EUR = 1.08 of `base_currency`).
            Reindexed to `prices.index` and forward-filled, so it doesn't need to
            share the exact same calendar.
        asset_currencies: Mapping of asset symbol -> currency code. Assets already
            in `base_currency` (or missing from this dict) are left untouched.
        base_currency: The target currency code (e.g. "USD").

    Returns:
        A new DataFrame, same shape as `prices`, denominated in `base_currency`.
    """
    converted = prices.copy()
    fx_aligned = fx_rates.reindex(prices.index).ffill()

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
