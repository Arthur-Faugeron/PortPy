import numpy as np
import pandas as pd
import pytest

from portpy.models.base import ModelResult, WeightBounds
from portpy.models.construction import build, efficient_frontier, optimize

NAMES = ["A", "B", "C"]


@pytest.fixture
def mu():
    return pd.Series([0.10, 0.06, 0.04], index=NAMES)


@pytest.fixture
def cov():
    sigma = np.array([0.20, 0.12, 0.06])
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    return pd.DataFrame(np.outer(sigma, sigma) * corr, index=NAMES, columns=NAMES)


class TestOptimize:
    def test_returns_model_result_named_after_method(self, mu, cov):
        result = optimize(mu, cov, method="mean_variance")
        assert isinstance(result, ModelResult)
        assert result.name == "mean_variance"

    def test_carries_solver_diagnostics(self, mu, cov):
        result = optimize(mu, cov, method="max_sharpe")
        assert result.diagnostics["method"] == "max_sharpe"
        assert "success" in result.diagnostics
        assert "objective_value" in result.diagnostics

    def test_min_variance_does_not_require_expected_returns(self, cov):
        result = optimize(cov_matrix=cov, method="min_variance")
        assert result.weights.sum() == pytest.approx(1.0)

    def test_missing_expected_returns_for_mean_variance_raises(self, cov):
        with pytest.raises(ValueError):
            optimize(cov_matrix=cov, method="mean_variance")

    def test_missing_cov_matrix_raises(self, mu):
        with pytest.raises(ValueError):
            optimize(expected_returns=mu, method="mean_variance")

    def test_unknown_method_raises(self, mu, cov):
        with pytest.raises(ValueError):
            optimize(mu, cov, method="not_a_real_method")

    def test_forwards_method_specific_kwargs(self, mu, cov):
        target = 0.06
        result = optimize(mu, cov, method="target_return", target=target)
        realized = float(result.weights.to_numpy() @ mu.to_numpy())
        assert realized == pytest.approx(target, abs=1e-4)

    def test_forwards_constraints(self, mu, cov):
        result = optimize(mu, cov, method="mean_variance", constraints=[WeightBounds(low=0.0, high=0.5)])
        assert (result.weights <= 0.5 + 1e-6).all()

    def test_hierarchical_risk_parity_has_no_solver_diagnostics_but_still_works(self, cov):
        result = optimize(cov_matrix=cov, method="hierarchical_risk_parity")
        assert result.weights.sum() == pytest.approx(1.0)
        assert result.diagnostics["method"] == "hierarchical_risk_parity"


class TestBuild:
    @pytest.fixture
    def asset_returns(self, business_day_index):
        rng = np.random.default_rng(21)
        mu_daily = np.array([0.0006, 0.0004, 0.0002])
        sigma_daily = np.array([0.018, 0.012, 0.006])
        corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
        cov_daily = np.outer(sigma_daily, sigma_daily) * corr
        data = rng.multivariate_normal(mu_daily, cov_daily, len(business_day_index))
        return pd.DataFrame(data, index=business_day_index, columns=NAMES)

    def test_estimates_then_optimizes(self, asset_returns):
        result = build(y=asset_returns, method="mean_variance")
        assert isinstance(result, ModelResult)
        assert result.weights.sum() == pytest.approx(1.0)

    def test_records_what_it_was_built_from(self, asset_returns):
        result = build(y=asset_returns, method="min_variance", covariance_kwargs={"method": "ledoit_wolf"})
        assert result.meta["built_from"]["covariance_kwargs"] == {"method": "ledoit_wolf"}
        assert result.meta["built_from"]["n_assets"] == 3
        assert result.meta["built_from"]["n_observations"] == len(asset_returns)

    def test_missing_y_raises(self):
        with pytest.raises(ValueError):
            build(method="mean_variance")

    def test_passes_expected_returns_kwargs(self, asset_returns):
        result = build(y=asset_returns, method="mean_variance", expected_returns_kwargs={"method": "james_stein"})
        assert result.weights.sum() == pytest.approx(1.0)

    def test_skips_expected_returns_estimation_for_return_agnostic_methods(self, asset_returns, monkeypatch):
        import portpy.models.construction.builders as builders_module

        calls = []
        original = builders_module._estimate_expected_returns

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(builders_module, "_estimate_expected_returns", spy)
        build(y=asset_returns, method="hierarchical_risk_parity")
        assert calls == []


class TestEfficientFrontierBuilder:
    def test_returns_model_result_with_frontier_diagnostic(self, mu, cov):
        result = efficient_frontier(mu, cov, n_points=10)
        assert isinstance(result, ModelResult)
        assert "frontier" in result.diagnostics
        assert len(result.diagnostics["frontier"]) <= 10

    def test_weights_is_the_max_sharpe_point(self, mu, cov):
        result = efficient_frontier(mu, cov, n_points=15)
        frontier = result.diagnostics["frontier"]
        best_row = frontier.loc[frontier["sharpe"].idxmax()]
        for name in NAMES:
            assert result.weights[name] == pytest.approx(best_row[name])

    def test_summary_returns_the_frontier_table(self, mu, cov):
        result = efficient_frontier(mu, cov, n_points=8)
        pd.testing.assert_frame_equal(result.summary().drop(columns=[]), result.diagnostics["frontier"])
