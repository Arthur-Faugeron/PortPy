import numpy as np
import pandas as pd
import pytest

from portpy.models.base import (
    GrossExposure,
    GroupCap,
    ModelResult,
    NetExposure,
    TurnoverCap,
    WeightBounds,
    resolve_bounds,
    resolve_scipy_constraints,
    solve_weights,
)

NAMES = ["A", "B", "C"]


def test_resolve_bounds_defaults_to_long_only():
    bounds = resolve_bounds([], NAMES)
    assert bounds == [(0.0, 1.0)] * 3


def test_resolve_bounds_applies_weight_bounds():
    bounds = resolve_bounds([WeightBounds(low=-0.5, high=0.5)], NAMES)
    assert bounds == [(-0.5, 0.5)] * 3


def test_resolve_bounds_per_asset_override():
    bounds = resolve_bounds([WeightBounds(low=0.0, high=1.0, per_asset={"B": (-0.2, 0.2)})], NAMES)
    assert bounds == [(0.0, 1.0), (-0.2, 0.2), (0.0, 1.0)]


def test_resolve_bounds_per_asset_unknown_raises():
    with pytest.raises(ValueError):
        resolve_bounds([WeightBounds(per_asset={"ZZZ": (0.0, 1.0)})], NAMES)


def test_resolve_scipy_constraints_net_exposure():
    cons = resolve_scipy_constraints([NetExposure(1.0)], NAMES)
    assert len(cons) == 1
    assert cons[0]["type"] == "eq"
    assert cons[0]["fun"](np.array([0.3, 0.3, 0.4])) == pytest.approx(0.0)
    assert cons[0]["fun"](np.array([0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_resolve_scipy_constraints_gross_exposure():
    cons = resolve_scipy_constraints([GrossExposure(1.6)], NAMES)
    assert cons[0]["fun"](np.array([0.8, -0.3, -0.5])) == pytest.approx(1.6 - 1.6)


def test_resolve_scipy_constraints_group_cap():
    cons = resolve_scipy_constraints([GroupCap(groups={"g1": ["A", "B"]}, max_weight=0.6)], NAMES)
    assert len(cons) == 1
    assert cons[0]["type"] == "ineq"
    # Within cap: 0.3 + 0.2 = 0.5 <= 0.6 -> positive slack.
    assert cons[0]["fun"](np.array([0.3, 0.2, 0.5])) == pytest.approx(0.1)


def test_resolve_scipy_constraints_group_cap_with_min():
    cons = resolve_scipy_constraints([GroupCap(groups={"g1": ["A"]}, max_weight=0.9, min_weight=0.1)], NAMES)
    assert len(cons) == 2  # one for max, one for min


def test_resolve_scipy_constraints_group_cap_unknown_asset_raises():
    with pytest.raises(ValueError):
        resolve_scipy_constraints([GroupCap(groups={"g1": ["ZZZ"]}, max_weight=0.5)], NAMES)


def test_resolve_scipy_constraints_turnover_cap_requires_current_weights():
    with pytest.raises(ValueError):
        resolve_scipy_constraints([TurnoverCap(max_turnover=0.1)], NAMES)


def test_resolve_scipy_constraints_turnover_cap():
    current = pd.Series([1 / 3, 1 / 3, 1 / 3], index=NAMES)
    cons = resolve_scipy_constraints([TurnoverCap(max_turnover=0.2, current_weights=current)], NAMES)
    # No turnover at all from current weights -> full slack of 0.2.
    assert cons[0]["fun"](current.to_numpy()) == pytest.approx(0.2)


def test_resolve_scipy_constraints_weight_bounds_produces_no_scipy_constraint():
    cons = resolve_scipy_constraints([WeightBounds(0.0, 1.0)], NAMES)
    assert cons == []


def test_resolve_scipy_constraints_unsupported_type_raises():
    with pytest.raises(TypeError):
        resolve_scipy_constraints(["not a constraint"], NAMES)


def test_solve_weights_minimizes_variance_and_sums_to_one():
    cov = np.diag([0.04, 0.09, 0.01])

    def objective(w):
        return float(w @ cov @ w)

    weights, diagnostics = solve_weights(objective, NAMES, constraints=None)
    assert weights.sum() == pytest.approx(1.0)
    # Lowest-variance asset (C, var=0.01) should dominate the min-variance solution.
    assert weights["C"] > weights["A"]
    assert weights["C"] > weights["B"]
    assert diagnostics["success"] is True
    assert "objective_value" in diagnostics


def test_solve_weights_defaults_to_full_investment_when_no_exposure_constraint():
    def objective(w):
        return float(np.sum(w**2))

    weights, _ = solve_weights(objective, NAMES)
    assert weights.sum() == pytest.approx(1.0)


def test_solve_weights_respects_gross_exposure_instead_of_net():
    def objective(w):
        return float(np.sum(w**2))

    weights, _ = solve_weights(objective, NAMES, constraints=[GrossExposure(1.0), WeightBounds(low=-1.0, high=1.0)])
    assert weights.abs().sum() == pytest.approx(1.0)


def test_solve_weights_respects_bounds():
    def objective(w):
        return float(-(w[0]))  # push asset A as high as possible

    weights, _ = solve_weights(objective, NAMES, constraints=[WeightBounds(low=0.0, high=0.4)])
    assert weights["A"] <= 0.4 + 1e-6
    assert (weights >= -1e-6).all()


def test_solve_weights_is_deterministic_across_calls():
    def objective(w):
        return float(w @ np.diag([0.1, 0.2, 0.3]) @ w)

    w1, _ = solve_weights(objective, NAMES)
    w2, _ = solve_weights(objective, NAMES)
    pd.testing.assert_series_equal(w1, w2)


class TestModelResult:
    def test_sets_explain_hooks(self):
        result = ModelResult(name="mean_variance", weights=pd.Series({"A": 0.6, "B": 0.4}))
        assert result._portpy_explain_name == "mean_variance"
        assert result._portpy_explain_value is result

    def test_explain_dispatches_through_registry(self, capsys):
        result = ModelResult(name="mean_variance", weights=pd.Series({"A": 0.6, "B": 0.4}), diagnostics={"objective_value": -0.1, "success": True})
        text = result.explain(print_it=False)
        assert "mean_variance" in text
        assert "This result:" in text

    def test_summary_returns_sorted_weight_table(self):
        result = ModelResult(name="mean_variance", weights=pd.Series({"A": 0.2, "B": 0.7, "C": 0.1}))
        summary = result.summary()
        assert list(summary.index) == ["B", "A", "C"]
        assert summary.attrs["portpy_explanation"] == "mean_variance"

    def test_summary_returns_frontier_when_present(self):
        frontier = pd.DataFrame({"return": [0.05, 0.1], "volatility": [0.1, 0.2]})
        result = ModelResult(name="efficient_frontier", weights=pd.Series({"A": 1.0}), diagnostics={"frontier": frontier})
        pd.testing.assert_frame_equal(result.summary().drop(columns=[], errors="ignore"), frontier, check_like=False)

    def test_plot_raises_not_implemented(self):
        result = ModelResult(name="mean_variance", weights=pd.Series({"A": 1.0}))
        with pytest.raises(NotImplementedError):
            result.plot()

    def test_compare_delegates_to_management_compare(self):
        a = ModelResult(name="mean_variance", weights=pd.Series({"A": 0.5, "B": 0.5}))
        b = pd.Series({"A": 0.6, "B": 0.4})
        comparison = a.compare(b)
        assert comparison.name == "compare"
        assert comparison.diagnostics["mode"] == "weights_only"

    def test_repr_contains_key_info(self):
        result = ModelResult(
            name="mean_variance",
            weights=pd.Series({"A": 0.0, "B": 1.0}),
            diagnostics={"success": True},
        )
        text = repr(result)
        assert "mean_variance" in text
        assert "n_active=1" in text
        assert "converged=True" in text
