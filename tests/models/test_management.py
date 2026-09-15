import pandas as pd
import pytest

from portpy import Portfolio
from portpy.models.base import ModelResult
from portpy.models.management import compare, monitor, rebalance

NAMES = ["A", "B", "C"]


class TestRebalance:
    def test_threshold_leaves_small_drift_untouched(self):
        current = pd.Series({"A": 0.34, "B": 0.33, "C": 0.33})
        target = pd.Series({"A": 0.35, "B": 0.32, "C": 0.33})
        result = rebalance(target, current, method="threshold", threshold=0.05)
        pd.testing.assert_series_equal(result.weights.sort_index(), current.rename("weight").sort_index())
        assert result.diagnostics["n_traded"] == 0

    def test_threshold_trades_assets_past_the_threshold(self):
        current = pd.Series({"A": 0.10, "B": 0.45, "C": 0.45})
        target = pd.Series({"A": 0.40, "B": 0.30, "C": 0.30})
        result = rebalance(target, current, method="threshold", threshold=0.05)
        assert result.diagnostics["n_traded"] == 3
        assert result.weights.sum() == pytest.approx(1.0)

    def test_full_and_calendar_trade_all_the_way_to_target(self):
        current = pd.Series({"A": 0.34, "B": 0.33, "C": 0.33})
        target = pd.Series({"A": 0.60, "B": 0.20, "C": 0.20})
        full_result = rebalance(target, current, method="full")
        calendar_result = rebalance(target, current, method="calendar")
        pd.testing.assert_series_equal(full_result.weights, calendar_result.weights)
        pd.testing.assert_series_equal(full_result.weights.sort_index(), target.rename("weight").sort_index())

    def test_accepts_model_result_as_target(self):
        current = pd.Series({"A": 0.5, "B": 0.5})
        target_result = ModelResult(name="mean_variance", weights=pd.Series({"A": 0.2, "B": 0.8}))
        result = rebalance(target_result, current, method="full")
        assert result.weights["B"] == pytest.approx(0.8)

    def test_turnover_matches_half_sum_absolute_trades(self):
        current = pd.Series({"A": 0.5, "B": 0.5})
        target = pd.Series({"A": 0.2, "B": 0.8})
        result = rebalance(target, current, method="full")
        expected_turnover = float((target - current).abs().sum()) / 2.0
        assert result.diagnostics["turnover"] == pytest.approx(expected_turnover)

    def test_estimated_cost_scales_with_cost_bps(self):
        current = pd.Series({"A": 0.5, "B": 0.5})
        target = pd.Series({"A": 0.0, "B": 1.0})
        free = rebalance(target, current, method="full", cost_bps=0.0)
        costly = rebalance(target, current, method="full", cost_bps=50.0)
        assert free.diagnostics["estimated_cost"] == 0.0
        assert costly.diagnostics["estimated_cost"] > 0.0

    def test_unknown_method_raises(self):
        current = pd.Series({"A": 1.0})
        target = pd.Series({"A": 1.0})
        with pytest.raises(ValueError):
            rebalance(target, current, method="bogus")

    def test_result_weights_sum_to_one_even_with_mismatched_asset_sets(self):
        current = pd.Series({"A": 0.5, "B": 0.5})
        target = pd.Series({"A": 0.5, "C": 0.5})  # B drops out, C is new
        result = rebalance(target, current, method="full")
        assert result.weights.sum() == pytest.approx(1.0)


class TestMonitor:
    def test_no_breaches_when_within_limits(self):
        weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        report = monitor(weights, {"max_weight": 0.5})
        assert report["ok"] is True
        assert report["breaches"] == []

    def test_flags_max_weight_breach(self):
        weights = pd.Series({"A": 0.7, "B": 0.2, "C": 0.1})
        report = monitor(weights, {"max_weight": 0.5})
        assert report["ok"] is False
        assert report["breaches"][0]["type"] == "max_weight"
        assert report["breaches"][0]["asset"] == "A"

    def test_flags_min_weight_breach(self):
        weights = pd.Series({"A": -0.1, "B": 0.6, "C": 0.5})
        report = monitor(weights, {"min_weight": 0.0})
        assert any(b["type"] == "min_weight" and b["asset"] == "A" for b in report["breaches"])

    def test_flags_group_cap_breach(self):
        weights = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
        limits = {"groups": {"tech": ["A", "B"]}, "max_group": {"tech": 0.5}}
        report = monitor(weights, limits)
        assert report["breaches"][0]["type"] == "max_group"
        assert report["breaches"][0]["group"] == "tech"

    def test_flags_max_gross_breach_for_leveraged_book(self):
        weights = pd.Series({"A": 1.2, "B": -0.5})
        report = monitor(weights, {"max_gross": 1.5})
        assert report["breaches"][0]["type"] == "max_gross"

    def test_no_limits_means_no_breaches(self):
        weights = pd.Series({"A": 5.0})  # absurd, but nothing was asked to be checked
        report = monitor(weights, {})
        assert report["ok"] is True


class TestCompare:
    def test_weights_only_mode_for_bare_series(self):
        a = pd.Series({"A": 0.5, "B": 0.5})
        b = pd.Series({"A": 0.7, "B": 0.3})
        result = compare(a, b)
        assert result.diagnostics["mode"] == "weights_only"
        assert "metric_delta" not in result.diagnostics

    def test_returns_mode_for_two_portfolios(self, multi_asset_prices):
        a = Portfolio(multi_asset_prices, name="A")
        b = Portfolio(multi_asset_prices, weights={"AAPL": 0.6, "MSFT": 0.2, "TLT": 0.2}, name="B")
        result = compare(a, b)
        assert result.diagnostics["mode"] == "returns"
        assert isinstance(result.diagnostics["metric_delta"], pd.Series)
        assert len(result.diagnostics["metric_delta"]) > 0

    def test_hhi_increases_when_more_concentrated(self):
        a = pd.Series({"A": 0.34, "B": 0.33, "C": 0.33})
        b = pd.Series({"A": 1.0, "B": 0.0, "C": 0.0})
        result = compare(a, b)
        assert result.diagnostics["hhi_after"] > result.diagnostics["hhi_before"]

    def test_turnover_zero_for_identical_weights(self):
        a = pd.Series({"A": 0.5, "B": 0.5})
        result = compare(a, a.copy())
        assert result.diagnostics["turnover"] == pytest.approx(0.0)

    def test_accepts_model_result_on_either_side(self):
        a = pd.Series({"A": 0.5, "B": 0.5})
        b = ModelResult(name="mean_variance", weights=pd.Series({"A": 0.8, "B": 0.2}))
        result = compare(a, b)
        assert result.weights["A"] == pytest.approx(0.8)


class TestPortfolioManagementNamespace:
    def test_compare_single_positional_argument(self, multi_asset_prices):
        a = Portfolio(multi_asset_prices, name="A")
        b = Portfolio(multi_asset_prices, weights={"AAPL": 0.6, "MSFT": 0.2, "TLT": 0.2}, name="B")
        result = a.models.management.compare(b)
        assert result.diagnostics["mode"] == "returns"

    def test_rebalance_autofills_current_weights_from_portfolio(self, multi_asset_prices):
        p = Portfolio(multi_asset_prices, name="P")
        target = pd.Series({"AAPL": 0.6, "MSFT": 0.2, "TLT": 0.2})
        result = p.models.management.rebalance(target_weights=target, threshold=0.01)
        assert result.weights.sum() == pytest.approx(1.0)

    def test_monitor_autofills_current_weights_from_portfolio(self, multi_asset_prices):
        p = Portfolio(multi_asset_prices, name="P")
        report = p.models.management.monitor(limits={"max_weight": 0.9})
        assert report["ok"] is True
