"""
Portfolio weight validation and normalization.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def equal_weights(names: list[str]) -> pd.Series:
    """
    Build a 1/N weight vector for the given asset names.

    Args:
        names: The asset names to weight equally.

    Returns:
        A Series of weights indexed by `names`, each equal to 1/len(names).

    Raises:
        ValueError: If `names` is empty.
    """
    n = len(names)
    if n == 0:
        raise ValueError("Cannot build weights for zero assets.")
    return pd.Series(np.full(n, 1.0 / n), index=list(names), name="weight")


def normalize_weights(
    weights: pd.Series | np.ndarray | dict | list,
    names: list[str] | None = None,
    allow_negative: bool = True,
    method: Literal["net", "gross"] = "net",
) -> pd.Series:
    """
    Validate a weights input and rescale it to a unit-exposure book.

    Args:
        weights: A Series/dict keyed by asset name, or a plain array/list aligned
            with `names`.
        names: Required asset order when `weights` isn't already labeled; also used
            to check that every asset has a weight.
        allow_negative: If False, raise when any weight is negative (e.g. for
            long-only optimizers).
        method: "net" divides by `weights.sum()` (the usual convention). "gross"
            divides by `weights.abs().sum()` instead, for long/short books.

    Returns:
        A Series of weights indexed by `names` (in that order) when `names` is
        given. Sums to 1.0 under "net"; under "gross", `.abs().sum()` is 1.0.

    Raises:
        TypeError: If `weights` isn't a Series, dict, list, tuple, or ndarray.
        ValueError: If `names` is required but missing, if any asset is missing
            a weight, if any weight is NaN, if `allow_negative` is False and a
            weight is negative, or if the relevant exposure (net or gross) is zero.
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

    if method == "gross":
        denom = weights.abs().sum()
    elif method == "net":
        denom = weights.sum()
    else:
        raise ValueError(f"method must be 'net' or 'gross', got {method!r}.")
    if np.isclose(denom, 0.0):
        raise ValueError(f"Weights' {method} exposure is zero; cannot normalize.")
    return (weights / denom).rename("weight")
