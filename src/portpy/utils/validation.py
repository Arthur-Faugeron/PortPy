"""Input-validation helpers shared by :mod:`portpy.portfolio` and the metric functions.

PortPy never silently fixes malformed input (wrong index type, duplicate dates,
unsorted data) - these helpers raise clear errors instead, because silently
reordering or deduplicating financial time series can hide real data-quality bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_datetime_index(data: pd.Series | pd.DataFrame, name: str = "data") -> pd.Series | pd.DataFrame:
    """Raise if `data` isn't indexed by a sorted, duplicate-free DatetimeIndex."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            f"{name} must be indexed by a pandas DatetimeIndex, got {type(data.index).__name__}. "
            "Use `df.index = pd.to_datetime(df.index)` before passing data to PortPy."
        )
    if data.index.has_duplicates:
        dupes = data.index[data.index.duplicated()].unique()
        preview = list(dupes[:5])
        raise ValueError(
            f"{name} has duplicate dates: {preview}{' ...' if len(dupes) > 5 else ''}. "
            "PortPy does not silently drop duplicates - deduplicate explicitly first."
        )
    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted ascending. Call `.sort_index()` first.")
    return data


def ensure_min_observations(data: pd.Series | pd.DataFrame, min_obs: int = 2, name: str = "data") -> None:
    n = len(data)
    if n < min_obs:
        raise ValueError(f"{name} needs at least {min_obs} observation(s), got {n}.")


def validate_confidence(confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")


def to_series(data: pd.Series | pd.DataFrame, column: str | None = None, name: str = "data") -> pd.Series:
    """Coerce a single-column DataFrame (or an explicit column) down to a Series."""
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        if column is not None:
            return data[column]
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError(
            f"{name} is a multi-column DataFrame; pass a Series or specify `column=`."
        )
    raise TypeError(f"{name} must be a pandas Series or DataFrame, got {type(data).__name__}.")


def align_pair(a: pd.Series, b: pd.Series, name_a: str = "returns", name_b: str = "benchmark") -> tuple[pd.Series, pd.Series]:
    """Inner-join two return series on their index and drop rows where either is NaN.

    Raises if the overlap is empty - a common silent bug when comparing series that
    don't actually share any dates (e.g. different calendars or date ranges).
    """
    joined = pd.concat([a.rename(name_a), b.rename(name_b)], axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError(
            f"{name_a} and {name_b} have no overlapping, non-NaN dates. "
            "Check that both series share the same date range and calendar."
        )
    return joined[name_a], joined[name_b]


def periodic_rate_from_annual(annual_rate: float, periods_per_year: int) -> float:
    """Convert an annual rate to a per-period rate via geometric (compounding) de-annualization."""
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning +/-inf (or 0 if numerator is also 0) instead of raising on a zero denominator."""
    if abs(denominator) < 1e-15:
        if abs(numerator) < 1e-15:
            return 0.0
        return float(np.sign(numerator) * np.inf)
    return numerator / denominator
