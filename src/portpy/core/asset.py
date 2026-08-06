"""
Lightweight asset tagging, used only for reporting/grouping, but never required.
"""

from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    """
    Optional tag for an asset's broad class.

    Passing these to :class:`~portpy.portfolio.Portfolio` (via ``asset_classes=``)
    is purely informational - it powers grouping in tearsheets and reminds you which
    assets trade around the clock (crypto) vs. only on business days (equities,
    bonds), but PortPy never uses it to silently reindex or fill your data.
    """

    EQUITY = "equity"
    CRYPTO = "crypto"
    BOND = "bond"
    COMMODITY = "commodity"
    FX = "fx"
    REAL_ESTATE = "real_estate"
    OTHER = "other"


"""
Asset classes that typically trade 24/7 - useful when choosing a calendar-alignment method.
"""
ALWAYS_ON_CLASSES = frozenset({AssetClass.CRYPTO, AssetClass.FX})
