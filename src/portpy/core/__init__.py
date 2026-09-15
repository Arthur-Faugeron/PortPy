"""
Core building blocks: asset tagging, weights, calendar alignment, currency conversion.
"""

from portpy.core.asset import ALWAYS_ON_CLASSES, AssetClass
from portpy.core.calendar import align_calendars, calendar_coverage_report, detect_frequency
from portpy.core.currency import calculate_fx_spread, convert_to_base_currency
from portpy.core.weights import equal_weights, normalize_weights

__all__ = [
    "AssetClass",
    "ALWAYS_ON_CLASSES",
    "align_calendars",
    "calendar_coverage_report",
    "detect_frequency",
    "convert_to_base_currency",
    "calculate_fx_spread",
    "equal_weights",
    "normalize_weights",
]
