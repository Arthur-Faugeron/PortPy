"""
Portfolio weight validation and normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weights(names: list[str]) -> pd.Series:
    """
    Build a 1/N weight vector for the given asset names.
    """
    n = len(names)
    if n == 0:
        raise ValueError("Cannot build weights for zero assets.")
    return pd.Series(np.full(n, 1.0 / n), index=list(names), name="weight")


def normalize_weights(
    weights: pd.Series | np.ndarray | dict | list,
    names: list[str] | None = None,
    allow_negative: bool = True,
) -> pd.Series:
    """
    Validate a weights input and rescale it so it sums to 1.

    Args:
        weights: A Series/dict keyed by asset name, or a plain array/list aligned
            with `names`.
        names: Required asset order when `weights` isn't already labeled; also used
            to check that every asset has a weight.
        allow_negative: If False, raise when any weight is negative (e.g. for
            long-only optimizers).

    Returns:
        A Series of weights summing to 1.0, indexed by `names` (in that order) when
        `names` is given.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)
    elif isinstance(weights, pd.Series):
        weights = weights.astype(float)
    elif isinstance(weights, (list, tuple, np.ndarray)):
        if names is None:
            raise ValueError("`names` is required when `weights` is a list/array.")
        arr = np.asarray(weights, dtype=float)
        if len(arr) != len(names):
            raise ValueError(f"weights has {len(arr)} entries but {len(names)} names were given.")
        weights = pd.Series(arr, index=list(names))
    else:
        raise TypeError(f"Unsupported weights type: {type(weights).__name__}")

    if names is not None:
        missing = set(names) - set(weights.index)
        if missing:
            raise ValueError(f"Weights are missing for assets: {sorted(missing)}")
        weights = weights.reindex(list(names))

    if weights.isna().any():
        raise ValueError("Weights contain NaN values.")
    if not allow_negative and (weights < 0).any():
        raise ValueError("Negative weights are not allowed here (long-only constraint).")

    total = weights.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Weights sum to zero; cannot normalize.")
    return (weights / total).rename("weight")
