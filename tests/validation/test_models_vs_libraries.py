"""Cross-validation of portpy.models against PyPortfolioOpt, Riskfolio-Lib, skfolio,
and independent closed-form checks.

Mirrors tests/validation/test_vs_empyrical_quantstats.py's approach for
Stage 1's metrics: synthetic-data tests always run; real-data tests
(@pytest.mark.network) are opt-in. Every optimizer/estimator gets checked
against at least one independent implementation before being considered done
(planning.md Section 7).

Two things are called out explicitly rather than silently tolerated:

- `hierarchical_risk_parity` matches Riskfolio-Lib's HCPortfolio HRP only with
  `leaf_order=False` - Riskfolio applies scipy's *optimal leaf ordering* on top
  of the raw dendrogram by default, a documented, optional refinement PortPy's
  implementation deliberately doesn't apply (matching Lopez de Prado's original
  formulation, which uses the raw linkage order).
- PyPortfolioOpt's own `HRPOpt.optimize()` raises `AttributeError` against the
  scipy version this project pins (`scipy.cluster.hierarchy` dropped the
  private `_LINKAGE_METHODS` attribute PyPortfolioOpt 1.6.0 reads) - an
  upstream incompatibility, not something to work around here. Riskfolio-Lib's
  HCPortfolio is used as the HRP reference instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pypfopt = pytest.importorskip("pypfopt")
riskfolio = pytest.importorskip("riskfolio")
skfolio = pytest.importorskip("skfolio")

from pypfopt import EfficientFrontier  # noqa: E402
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns  # noqa: E402
from pypfopt.risk_models import CovarianceShrinkage  # noqa: E402
from skfolio import RiskMeasure  # noqa: E402
from skfolio.optimization import MeanRisk, ObjectiveFunction  # noqa: E402
from skfolio.optimization import RiskBudgeting as SkRiskBudgeting  # noqa: E402

from portpy.metrics.returns import prices_from_returns  # noqa: E402
from portpy.metrics.risk import beta as metrics_beta  # noqa: E402
from portpy.models import optimization as opt  # noqa: E402
from portpy.models.base import WeightBounds  # noqa: E402
from portpy.models.estimators.covariance import covariance as pp_covariance  # noqa: E402
from portpy.models.estimators.factor_models import capm  # noqa: E402

ABS_TOL = 1e-4
NAMES = ["A", "B", "C", "D", "E"]


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(2024)
    idx = pd.bdate_range("2019-01-01", periods=750)
    mu_true = np.array([0.0006, 0.0004, 0.0002, 0.0003, 0.0005])
    sigma_true = np.array([0.020, 0.015, 0.008, 0.012, 0.022])
    corr = np.array(
        [
            [1.0, 0.5, 0.1, 0.2, 0.3],
            [0.5, 1.0, 0.1, 0.3, 0.2],
            [0.1, 0.1, 1.0, 0.05, 0.0],
            [0.2, 0.3, 0.05, 1.0, 0.1],
            [0.3, 0.2, 0.0, 0.1, 1.0],
        ]
    )
    cov_daily = np.outer(sigma_true, sigma_true) * corr
    return pd.DataFrame(rng.multivariate_normal(mu_true, cov_daily, len(idx)), index=idx, columns=NAMES)


@pytest.fixture
def synthetic_prices(synthetic_returns):
    # prices_from_returns anchors an extra base row before the first return, so
    # simple_returns(synthetic_prices) round-trips synthetic_returns exactly -
    # a hand-rolled cumprod without that anchor silently drops the first
    # observation on the way back through .pct_change(), which is enough to
    # measurably shift a shrinkage estimator over a few hundred rows.
    return prices_from_returns(synthetic_returns, base=100.0)


def _mu_cov(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    return returns.mean() * 252, returns.cov() * 252


def _assert_weights_close(portpy_w: pd.Series, reference: dict[str, float] | pd.Series, abs_tol: float = ABS_TOL) -> None:
    ref = pd.Series(reference)
    portpy_w = portpy_w.reindex(ref.index)
    np.testing.assert_allclose(portpy_w.to_numpy(), ref.to_numpy(), atol=abs_tol)


class TestMeanVarianceFamilyVsPyPortfolioOpt:
    """mean_variance/max_sharpe/min_variance/target_return/target_volatility are
    all classic convex (or, for max_sharpe, quasi-convex) QPs that PyPortfolioOpt
    solves with cvxpy - an entirely different solver path than PortPy's SLSQP,
    so close agreement is a real check of the objective/constraint formulation,
    not just numerical coincidence."""

    def test_min_variance(self, synthetic_returns):
        mu, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.min_variance(cov)
        w_ref = EfficientFrontier(mu, cov).min_volatility()
        _assert_weights_close(w_portpy, w_ref)

    def test_max_sharpe(self, synthetic_returns):
        mu, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.max_sharpe(mu, cov, rf=0.0)
        w_ref = EfficientFrontier(mu, cov).max_sharpe(risk_free_rate=0.0)
        _assert_weights_close(w_portpy, w_ref, abs_tol=1e-3)

    def test_target_return(self, synthetic_returns):
        # PyPortfolioOpt's efficient_return() constrains return >= target (an
        # inequality - see its docstring/source), not == target: whenever the
        # unconstrained global min-variance portfolio's own return already
        # clears the bar, that inequality is non-binding and efficient_return
        # just hands back the min-variance portfolio, which needn't have
        # return anywhere near `target`. PortPy's target_return deliberately
        # uses an equality instead (the exact point on the frontier at that
        # return - required for efficient_frontier() to sweep a full curve,
        # since sweeping ">=" targets below the min-variance return would
        # collapse to the same portfolio for every one of them). The two
        # semantics necessarily agree only once `target` is above the global
        # min-variance return, which is where this test deliberately sits.
        mu, cov = _mu_cov(synthetic_returns)
        min_var_return = float(opt.min_variance(cov).to_numpy() @ mu.to_numpy())
        target = min_var_return + 0.03
        w_portpy = opt.target_return(mu, cov, target=target)
        w_ref = EfficientFrontier(mu, cov).efficient_return(target_return=target)
        _assert_weights_close(w_portpy, w_ref)

    def test_target_volatility(self, synthetic_returns):
        mu, cov = _mu_cov(synthetic_returns)
        target_vol = 0.15
        w_portpy = opt.target_volatility(mu, cov, target=target_vol)
        w_ref = EfficientFrontier(mu, cov).efficient_risk(target_volatility=target_vol)
        _assert_weights_close(w_portpy, w_ref)

    def test_mean_variance_matches_max_quadratic_utility(self, synthetic_returns):
        # PortPy's mean_variance is exactly PyPortfolioOpt's max_quadratic_utility -
        # same objective (w'mu - 0.5*risk_aversion*w'Sigma*w), same constraint (sum=1).
        mu, cov = _mu_cov(synthetic_returns)
        risk_aversion = 2.0
        w_portpy = opt.mean_variance(mu, cov, risk_aversion=risk_aversion)
        w_ref = EfficientFrontier(mu, cov).max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=False)
        _assert_weights_close(w_portpy, w_ref)


class TestRiskBasedMethodsVsRiskfolioLib:
    def test_risk_parity(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.risk_parity(cov)

        port = riskfolio.Portfolio(returns=synthetic_returns)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w_ref = port.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)["weights"]
        _assert_weights_close(w_portpy, w_ref, abs_tol=1e-3)

    def test_risk_budgeting_with_unequal_budget(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        budget = pd.Series({"A": 2.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0})
        w_portpy = opt.risk_budgeting(cov, budget)

        port = riskfolio.Portfolio(returns=synthetic_returns)
        port.assets_stats(method_mu="hist", method_cov="hist")
        b_vec = (budget / budget.sum()).reindex(NAMES).to_numpy().reshape(-1, 1)
        w_ref = port.rp_optimization(model="Classic", rm="MV", rf=0, b=b_vec, hist=True)["weights"]
        _assert_weights_close(w_portpy, w_ref, abs_tol=1e-3)

    def test_hierarchical_risk_parity_matches_without_optimal_leaf_ordering(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.hierarchical_risk_parity(cov, linkage_method="single")

        hc = riskfolio.HCPortfolio(returns=synthetic_returns)
        w_ref = hc.optimization(
            model="HRP", codependence="pearson", rm="MV", rf=0, linkage="single", leaf_order=False
        )["weights"]
        _assert_weights_close(w_portpy, w_ref, abs_tol=1e-6)

    def test_hierarchical_risk_parity_disagrees_with_optimal_leaf_ordering_enabled(self, synthetic_returns):
        # Documents *why* the above test pins leaf_order=False - with Riskfolio's
        # default (leaf_order=True), the two implementations should NOT match,
        # since they're then answering a subtly different clustering question.
        _, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.hierarchical_risk_parity(cov, linkage_method="single")

        hc = riskfolio.HCPortfolio(returns=synthetic_returns)
        w_ref = hc.optimization(model="HRP", codependence="pearson", rm="MV", rf=0, linkage="single", leaf_order=True)[
            "weights"
        ]
        with pytest.raises(AssertionError):
            _assert_weights_close(w_portpy, w_ref, abs_tol=1e-4)


class TestBlackLittermanVsPyPortfolioOpt:
    def test_implied_equilibrium_returns_match(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        market_weights = pd.Series(1 / len(NAMES), index=NAMES)
        risk_aversion = 2.5

        pi_ref = market_implied_prior_returns(market_weights, risk_aversion, cov, risk_free_rate=0.0)
        pi_portpy = risk_aversion * (cov.to_numpy() @ market_weights.to_numpy())
        np.testing.assert_allclose(pi_portpy, pi_ref.reindex(NAMES).to_numpy(), atol=1e-6)

    def test_posterior_returns_match_with_one_absolute_view(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        market_weights = pd.Series(1 / len(NAMES), index=NAMES)
        risk_aversion = 2.5
        pi = market_implied_prior_returns(market_weights, risk_aversion, cov, risk_free_rate=0.0)

        bl_ref = BlackLittermanModel(cov, pi=pi, absolute_views={"A": 0.15}, risk_aversion=risk_aversion)
        posterior_ref = bl_ref.bl_returns()

        # Reproduce PortPy's internal posterior_mu formula directly (the public
        # black_litterman() only returns final weights, not the intermediate mu).
        cov_np = cov.to_numpy()
        p_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        q_vector = np.array([0.15])
        tau_cov = 0.05 * cov_np
        omega = np.diag(np.diag(p_matrix @ tau_cov @ p_matrix.T))
        tau_cov_inv = np.linalg.pinv(tau_cov)
        omega_inv = np.linalg.pinv(omega)
        precision = tau_cov_inv + p_matrix.T @ omega_inv @ p_matrix
        posterior_cov_of_mean = np.linalg.pinv(precision)
        posterior_mu = posterior_cov_of_mean @ (tau_cov_inv @ pi.to_numpy() + p_matrix.T @ omega_inv @ q_vector)

        np.testing.assert_allclose(posterior_mu, posterior_ref.reindex(NAMES).to_numpy(), atol=1e-6)

    def test_no_views_reproduces_market_weights(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        market_weights = pd.Series(1 / len(NAMES), index=NAMES)
        w = opt.black_litterman(cov, market_weights, views={}, risk_aversion=2.5)
        np.testing.assert_allclose(w.reindex(NAMES).to_numpy(), market_weights.to_numpy(), atol=1e-3)


class TestCovarianceEstimatorsVsPyPortfolioOpt:
    def test_ledoit_wolf_matches_independent_implementation(self, synthetic_returns, synthetic_prices):
        cov_portpy = pp_covariance(synthetic_returns, method="ledoit_wolf", periods_per_year=252)
        cov_ref = CovarianceShrinkage(synthetic_prices, frequency=252).ledoit_wolf()
        np.testing.assert_allclose(cov_portpy.reindex(index=NAMES, columns=NAMES).to_numpy(), cov_ref.reindex(index=NAMES, columns=NAMES).to_numpy(), atol=1e-6)


class TestFactorModelsVsIndependentOLS:
    """No third-party library dependency needed here - a plain numpy least-squares
    solve is itself an independent implementation of OLS, distinct from the
    statsmodels machinery factor_models.py is built on."""

    def test_capm_matches_manual_ols_closed_form(self):
        rng = np.random.default_rng(11)
        idx = pd.bdate_range("2020-01-01", periods=600)
        market = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
        y = 0.0001 + 1.3 * market + rng.normal(0, 0.002, len(idx))

        model = capm(y, market, rf=0.0)

        design = np.column_stack([np.ones(len(idx)), market.to_numpy()])
        manual_beta = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]

        assert model.params["const"] == pytest.approx(manual_beta[0], abs=1e-9)
        assert model.params["market"] == pytest.approx(manual_beta[1], abs=1e-9)

    def test_capm_beta_matches_metrics_risk_beta_at_zero_rf(self, synthetic_returns):
        y = synthetic_returns["A"]
        benchmark = synthetic_returns["B"]  # any second series stands in for a "benchmark" here
        model = capm(y, benchmark, rf=0.0)
        assert model.params["market"] == pytest.approx(metrics_beta(y, benchmark), abs=1e-8)


class TestMaximumDiversificationLocalOptimality:
    """No mainstream library exposes Choueifaty's Most Diversified Portfolio as a
    one-line call, so this checks local optimality directly instead: perturb the
    solution in every coordinate direction and confirm none of the (projected
    back to sum=1) neighbors achieve a higher diversification_ratio."""

    def test_no_nearby_feasible_point_diversifies_better(self, synthetic_returns):
        from portpy.metrics.covariance import diversification_ratio

        _, cov = _mu_cov(synthetic_returns)
        w = opt.maximum_diversification(cov)
        baseline = diversification_ratio(w, cov)

        rng = np.random.default_rng(5)
        n = len(w)
        for _ in range(200):
            perturbation = rng.normal(0, 0.01, n)
            candidate = (w.to_numpy() + perturbation).clip(min=0.0)
            candidate = candidate / candidate.sum()
            candidate_series = pd.Series(candidate, index=w.index)
            assert diversification_ratio(candidate_series, cov) <= baseline + 1e-6


class TestSkfolioThirdLibrary:
    """A third, independent cross-check (planning.md Section 7's "at least 3 other
    libraries" bar) - skfolio solves via cvxpy+CLARABEL, a different convex-optimization
    stack than both PyPortfolioOpt's cvxpy formulation and Riskfolio-Lib's own solver.
    Fit directly on periodic returns (skfolio's default EmpiricalPrior estimates the same
    sample mean/covariance PortPy's mean_historical/sample methods do) - min_variance and
    risk_parity are scale-invariant to the annualization convention, and max_sharpe's
    ratio is invariant too since mu and Sigma are both annualized the same way, so this is
    a like-for-like comparison despite skfolio never seeing PortPy's annualized mu/cov."""

    def test_min_variance(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.min_variance(cov)

        mr = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.VARIANCE)
        mr.fit(synthetic_returns)
        w_ref = pd.Series(mr.weights_, index=synthetic_returns.columns)
        _assert_weights_close(w_portpy, w_ref)

    def test_max_sharpe(self, synthetic_returns):
        mu, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.max_sharpe(mu, cov, rf=0.0)

        mr = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.VARIANCE, risk_free_rate=0.0)
        mr.fit(synthetic_returns)
        w_ref = pd.Series(mr.weights_, index=synthetic_returns.columns)
        _assert_weights_close(w_portpy, w_ref)

    def test_risk_parity(self, synthetic_returns):
        _, cov = _mu_cov(synthetic_returns)
        w_portpy = opt.risk_parity(cov)

        rb = SkRiskBudgeting(risk_measure=RiskMeasure.VARIANCE)
        rb.fit(synthetic_returns)
        w_ref = pd.Series(rb.weights_, index=synthetic_returns.columns)
        _assert_weights_close(w_portpy, w_ref)


class TestNegativeWeightsShortPositions:
    """Every other weight-based test above uses the default long-only WeightBounds.
    max_sharpe with WeightBounds(low<0) genuinely wants to short the negative-mu asset
    here (not just avoid it) - checked against both PyPortfolioOpt and skfolio to confirm
    the negative weight itself, not just its magnitude, matches."""

    def test_max_sharpe_with_shorts_vs_pypfopt_and_skfolio(self, synthetic_returns):
        mu, cov = _mu_cov(synthetic_returns)
        wb = WeightBounds(low=-0.3, high=1.0)
        w_portpy = opt.max_sharpe(mu, cov, rf=0.0, constraints=[wb])
        assert (w_portpy < -1e-6).any(), "expected a genuine short position in this example"

        w_ref_pypfopt = EfficientFrontier(mu, cov, weight_bounds=(-0.3, 1.0)).max_sharpe(risk_free_rate=0.0)
        _assert_weights_close(w_portpy, w_ref_pypfopt)

        mr = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.VARIANCE,
            min_weights=-0.3, max_weights=1.0, risk_free_rate=0.0,
        )
        mr.fit(synthetic_returns)
        w_ref_skfolio = pd.Series(mr.weights_, index=synthetic_returns.columns)
        _assert_weights_close(w_portpy, w_ref_skfolio)


@pytest.mark.network
class TestRealDataVsPyPortfolioOpt:
    """Downloads real AAPL/MSFT/JPM/TLT/GLD data via yfinance - the case that
    actually matters, run locally with: pytest -m network tests/validation"""

    def test_min_variance_on_real_data(self):
        yf = pytest.importorskip("yfinance")

        tickers = ["AAPL", "MSFT", "JPM", "TLT", "GLD"]
        prices = yf.download(tickers, period="3y", auto_adjust=True, progress=False)["Close"].dropna()
        returns = prices.pct_change().dropna()
        mu, cov = _mu_cov(returns)

        w_portpy = opt.min_variance(cov)
        w_ref = EfficientFrontier(mu, cov).min_volatility()
        _assert_weights_close(w_portpy, w_ref, abs_tol=1e-3)
