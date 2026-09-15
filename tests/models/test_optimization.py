import numpy as np
import pandas as pd
import pytest

from portpy.metrics.covariance import diversification_ratio, portfolio_volatility
from portpy.models.base import GroupCap, NetExposure, WeightBounds
from portpy.models.optimization import (
    black_litterman,
    efficient_frontier,
    hierarchical_risk_parity,
    max_sharpe,
    maximum_diversification,
    mean_variance,
    min_variance,
    risk_budgeting,
    risk_parity,
    target_return,
    target_volatility,
)

NAMES = ["A", "B", "C", "D"]


@pytest.fixture
def mu():
    return pd.Series([0.08, 0.06, 0.03, 0.05], index=NAMES)


@pytest.fixture
def cov():
    sigma = np.array([0.20, 0.15, 0.05, 0.10])
    corr = np.array(
        [
            [1.0, 0.6, 0.0, 0.2],
            [0.6, 1.0, 0.1, 0.3],
            [0.0, 0.1, 1.0, 0.05],
            [0.2, 0.3, 0.05, 1.0],
        ]
    )
    matrix = np.outer(sigma, sigma) * corr
    return pd.DataFrame(matrix, index=NAMES, columns=NAMES)


class TestMeanVariance:
    def test_sums_to_one_and_long_only_by_default(self, mu, cov):
        w = mean_variance(mu, cov)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-6).all()

    def test_higher_risk_aversion_reduces_volatility(self, mu, cov):
        low_ra = mean_variance(mu, cov, risk_aversion=0.5)
        high_ra = mean_variance(mu, cov, risk_aversion=20.0)
        vol_low = portfolio_volatility(low_ra, cov)
        vol_high = portfolio_volatility(high_ra, cov)
        assert vol_high <= vol_low + 1e-6

    def test_respects_weight_bounds(self, mu, cov):
        w = mean_variance(mu, cov, constraints=[WeightBounds(low=0.0, high=0.3)])
        assert (w <= 0.3 + 1e-6).all()

    def test_respects_group_cap(self, mu, cov):
        w = mean_variance(mu, cov, constraints=[GroupCap(groups={"AB": ["A", "B"]}, max_weight=0.4)])
        assert w[["A", "B"]].sum() <= 0.4 + 1e-4

    def test_allows_shorting_with_wide_bounds(self, mu, cov):
        w = mean_variance(mu, cov, constraints=[WeightBounds(low=-1.0, high=1.0)], risk_aversion=0.01)
        assert w.sum() == pytest.approx(1.0)


class TestMinVariance:
    def test_achieves_lower_or_equal_variance_than_equal_weight(self, cov):
        w = min_variance(cov)
        equal_weight = pd.Series(0.25, index=NAMES)
        assert portfolio_volatility(w, cov) <= portfolio_volatility(equal_weight, cov) + 1e-9

    def test_needs_no_expected_returns_argument(self, cov):
        # Purely a signature/behavior check: min_variance shouldn't require mu at all.
        w = min_variance(cov)
        assert w.sum() == pytest.approx(1.0)


class TestMaxSharpe:
    def test_beats_min_variance_sharpe(self, mu, cov):
        w_sharpe = max_sharpe(mu, cov, rf=0.0)
        w_minvar = min_variance(cov)

        def sharpe(w):
            ret = float(w.to_numpy() @ mu.to_numpy())
            vol = portfolio_volatility(w, cov)
            return ret / vol

        assert sharpe(w_sharpe) >= sharpe(w_minvar) - 1e-6


class TestTargetReturn:
    def test_hits_target_return_exactly(self, mu, cov):
        target = float(mu.mean())
        w = target_return(mu, cov, target=target)
        realized = float(w.to_numpy() @ mu.to_numpy())
        assert realized == pytest.approx(target, abs=1e-4)

    def test_infeasible_target_flags_nonconvergence(self, mu, cov):
        too_high = float(mu.max()) + 1.0
        w = target_return(mu, cov, target=too_high)
        assert w.attrs["solve_diagnostics"]["success"] is False


class TestTargetVolatility:
    def test_hits_feasible_target_volatility(self, mu, cov):
        min_vol = portfolio_volatility(min_variance(cov), cov)
        target = min_vol + 0.02
        w = target_volatility(mu, cov, target=target)
        realized = portfolio_volatility(w, cov)
        assert realized == pytest.approx(target, abs=1e-3)

    def test_infeasible_target_below_min_variance_flags_nonconvergence(self, mu, cov):
        min_vol = portfolio_volatility(min_variance(cov), cov)
        w = target_volatility(mu, cov, target=min_vol * 0.1)
        assert w.attrs["solve_diagnostics"]["success"] is False


class TestRiskParityAndBudgeting:
    def test_risk_parity_equalizes_risk_contributions(self, cov):
        w = risk_parity(cov)
        cov_np = cov.to_numpy()
        w_np = w.reindex(NAMES).to_numpy()
        mctr = cov_np @ w_np
        risk_contrib = w_np * mctr
        assert np.allclose(risk_contrib, risk_contrib.mean(), rtol=0.05)

    def test_risk_parity_is_long_only(self, cov):
        w = risk_parity(cov)
        assert (w >= -1e-6).all()

    def test_risk_budgeting_matches_requested_ratio(self, cov):
        budget = pd.Series([2.0, 1.0, 1.0, 1.0], index=NAMES)
        w = risk_budgeting(cov, budget)
        cov_np = cov.to_numpy()
        w_np = w.reindex(NAMES).to_numpy()
        risk_contrib = w_np * (cov_np @ w_np)
        # Asset A should carry roughly 2x the risk contribution of B/C/D.
        assert risk_contrib[0] == pytest.approx(2 * risk_contrib[1], rel=0.15)

    def test_risk_parity_is_risk_budgeting_with_equal_budget(self, cov):
        w_parity = risk_parity(cov)
        w_budget = risk_budgeting(cov, pd.Series(1.0, index=NAMES))
        pd.testing.assert_series_equal(w_parity, w_budget, check_exact=False, atol=1e-6)


class TestHierarchicalRiskParity:
    def test_sums_to_one_and_long_only(self, cov):
        w = hierarchical_risk_parity(cov)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()

    def test_two_asset_case_does_not_crash(self):
        cov2 = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["X", "Y"], columns=["X", "Y"])
        w = hierarchical_risk_parity(cov2)
        assert w.sum() == pytest.approx(1.0)

    def test_ignores_unsupported_constraints_with_warning(self, cov):
        with pytest.warns(UserWarning):
            w = hierarchical_risk_parity(cov, constraints=[NetExposure(1.0)])
        assert w.sum() == pytest.approx(1.0)

    def test_weight_bounds_applied_post_hoc(self, cov):
        w = hierarchical_risk_parity(cov, constraints=[WeightBounds(low=0.0, high=0.3)])
        assert (w <= 0.3 + 1e-6).all()
        assert w.sum() == pytest.approx(1.0)


class TestMaximumDiversification:
    def test_beats_or_matches_equal_weight_diversification_ratio(self, cov):
        w = maximum_diversification(cov)
        equal_weight = pd.Series(0.25, index=NAMES)
        assert diversification_ratio(w, cov) >= diversification_ratio(equal_weight, cov) - 1e-6

    def test_sums_to_one(self, cov):
        w = maximum_diversification(cov)
        assert w.sum() == pytest.approx(1.0)


class TestBlackLitterman:
    def test_no_views_collapses_to_market_weights(self, cov):
        market_weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=NAMES)
        # An empty views dict + tiny risk_aversion mismatch aside, feeding the
        # implied equilibrium mu straight back into mean_variance at the same
        # risk_aversion used to construct pi should reproduce market_weights.
        w = black_litterman(cov, market_weights, views={}, risk_aversion=2.5)
        pd.testing.assert_series_equal(w, market_weights.rename("weight"), check_exact=False, atol=1e-3)

    def test_bullish_view_increases_that_assets_weight(self, cov):
        market_weights = pd.Series(0.25, index=NAMES)
        baseline = black_litterman(cov, market_weights, views={}, risk_aversion=2.5)
        bullish = black_litterman(cov, market_weights, views={"C": 0.30}, risk_aversion=2.5)
        assert bullish["C"] > baseline["C"]

    def test_explicit_pq_views_tuple(self, cov):
        market_weights = pd.Series(0.25, index=NAMES)
        p_matrix = pd.DataFrame([[1.0, -1.0, 0.0, 0.0]], columns=NAMES)
        q_vector = pd.Series([0.05])
        w = black_litterman(cov, market_weights, views=(p_matrix, q_vector))
        assert w.sum() == pytest.approx(1.0)

    def test_unknown_view_asset_raises(self, cov):
        market_weights = pd.Series(0.25, index=NAMES)
        with pytest.raises(ValueError):
            black_litterman(cov, market_weights, views={"ZZZ": 0.1})


class TestEfficientFrontier:
    def test_returns_dataframe_with_expected_columns(self, mu, cov):
        frontier = efficient_frontier(mu, cov, n_points=15)
        for col in ("target_return", "return", "volatility", "sharpe", *NAMES):
            assert col in frontier.columns

    def test_volatility_is_nondecreasing_along_the_frontier(self, mu, cov):
        frontier = efficient_frontier(mu, cov, n_points=20)
        vols = frontier["volatility"].to_numpy()
        assert np.all(np.diff(vols) >= -1e-6)

    def test_row_weights_sum_to_one(self, mu, cov):
        frontier = efficient_frontier(mu, cov, n_points=10)
        row_sums = frontier[NAMES].sum(axis=1)
        np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-4)

    def test_first_point_matches_global_min_variance(self, mu, cov):
        frontier = efficient_frontier(mu, cov, n_points=10)
        min_var_vol = portfolio_volatility(min_variance(cov), cov)
        assert frontier["volatility"].iloc[0] == pytest.approx(min_var_vol, abs=1e-3)
