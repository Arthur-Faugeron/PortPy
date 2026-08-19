import numpy as np
import pandas as pd
import pytest

from portpy.core.weights import equal_weights, normalize_weights


def test_equal_weights_sums_to_one():
    w = equal_weights(["A", "B", "C"])
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w.to_numpy(), 1 / 3)


def test_equal_weights_empty_raises():
    with pytest.raises(ValueError):
        equal_weights([])


def test_normalize_weights_from_dict_rescales_to_one():
    w = normalize_weights({"A": 2, "B": 2}, names=["A", "B"])
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(0.5)


def test_normalize_weights_from_list_requires_names():
    with pytest.raises(ValueError):
        normalize_weights([0.5, 0.5])


def test_normalize_weights_missing_asset_raises():
    with pytest.raises(ValueError):
        normalize_weights({"A": 1.0}, names=["A", "B"])


def test_normalize_weights_nan_raises():
    with pytest.raises(ValueError):
        normalize_weights(pd.Series({"A": np.nan, "B": 1.0}), names=["A", "B"])


def test_normalize_weights_zero_sum_raises():
    with pytest.raises(ValueError):
        normalize_weights({"A": 1.0, "B": -1.0}, names=["A", "B"])


def test_normalize_weights_disallow_negative():
    with pytest.raises(ValueError):
        normalize_weights({"A": -0.5, "B": 1.5}, names=["A", "B"], allow_negative=False)


def test_normalize_weights_reorders_to_match_names():
    w = normalize_weights({"B": 1, "A": 1}, names=["A", "B"])
    assert list(w.index) == ["A", "B"]


def test_normalize_weights_default_method_is_net_and_unchanged():
    w = normalize_weights({"A": 2, "B": 2}, names=["A", "B"])
    w_explicit = normalize_weights({"A": 2, "B": 2}, names=["A", "B"], method="net")
    pd.testing.assert_series_equal(w, w_explicit)


def test_normalize_weights_gross_normalizes_by_abs_sum_for_long_short_book():
    # Near dollar-neutral: net exposure is a small fraction of gross exposure.
    w = normalize_weights({"A": 1.0, "B": -0.98}, names=["A", "B"], method="gross")
    assert w.abs().sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(1.0 / 1.98)
    assert w["B"] == pytest.approx(-0.98 / 1.98)


def test_normalize_weights_gross_vs_net_differ_for_long_short_book():
    net = normalize_weights({"A": 1.0, "B": -0.98}, names=["A", "B"], method="net")
    gross = normalize_weights({"A": 1.0, "B": -0.98}, names=["A", "B"], method="gross")
    # Net exposure (0.02) blows the net-normalized weights up to a much larger magnitude.
    assert abs(net["A"]) > abs(gross["A"])


def test_normalize_weights_gross_zero_raises():
    with pytest.raises(ValueError):
        normalize_weights({"A": 0.0}, names=["A"], method="gross")


def test_normalize_weights_invalid_method_raises():
    with pytest.raises(ValueError):
        normalize_weights({"A": 1.0, "B": 1.0}, names=["A", "B"], method="bogus")
