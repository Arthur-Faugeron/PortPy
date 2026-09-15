import numpy as np
import pandas as pd
import pytest

from portpy.models.estimators.covariance import covariance
from portpy.models.estimators.expected_returns import expected_returns
from portpy.models.estimators.factor_models import (
    capm,
    factor_attribution,
    fama_french,
    linear_regression,
    rolling_regression,
)


@pytest.fixture
def asset_returns(business_day_index):
    rng = np.random.default_rng(11)
    n = len(business_day_index)
    mu = np.array([0.0006, 0.0003, 0.0001])
    sigma = np.array([0.015, 0.012, 0.006])
    corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.1], [0.0, 0.1, 1.0]])
    cov = np.outer(sigma, sigma) * corr
    data = rng.multivariate_normal(mu, cov, n)
    return pd.DataFrame(data, index=business_day_index, columns=["A", "B", "C"])


class TestExpectedReturns:
    def test_mean_historical_matches_manual_annualization(self, asset_returns):
        result = expected_returns(asset_returns, method="mean_historical", periods_per_year=252)
        expected = asset_returns.mean() * 252
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_mean_historical_geometric_differs_from_arithmetic(self, asset_returns):
        arithmetic = expected_returns(asset_returns, method="mean_historical", geometric=False)
        geometric = expected_returns(asset_returns, method="mean_historical", geometric=True)
        assert not np.allclose(arithmetic.to_numpy(), geometric.to_numpy())

    def test_ewma_weights_recent_periods_more(self, business_day_index):
        # A return series with a clear regime shift: flat then trending up.
        n = len(business_day_index)
        flat = np.zeros(n // 2)
        trending = np.full(n - n // 2, 0.002)
        series = pd.DataFrame({"A": np.concatenate([flat, trending])}, index=business_day_index)
        ewma_mu = expected_returns(series, method="ewma", span=20)
        historical_mu = expected_returns(series, method="mean_historical")
        # EWMA should weight the recent uptrend more heavily than the flat plain average.
        assert ewma_mu["A"] > historical_mu["A"]

    def test_capm_implied_requires_benchmark(self, asset_returns):
        with pytest.raises(ValueError):
            expected_returns(asset_returns, method="capm_implied")

    def test_capm_implied_zero_beta_gives_rf(self, business_day_index):
        rng = np.random.default_rng(3)
        market = pd.Series(rng.normal(0.0005, 0.01, len(business_day_index)), index=business_day_index)
        uncorrelated = pd.DataFrame({"Z": rng.normal(0.0002, 0.005, len(business_day_index))}, index=business_day_index)
        result = expected_returns(uncorrelated, method="capm_implied", benchmark=market, rf=0.02)
        assert result["Z"] == pytest.approx(0.02, abs=0.02)  # near-zero beta -> near rf

    def test_james_stein_shrinks_toward_grand_mean(self, asset_returns):
        historical = expected_returns(asset_returns, method="mean_historical")
        shrunk = expected_returns(asset_returns, method="james_stein")
        grand_mean = historical.mean()
        # Shrunk estimates should be strictly between the raw estimate and the grand mean.
        for col in asset_returns.columns:
            lo, hi = sorted([historical[col], grand_mean])
            assert lo - 1e-9 <= shrunk[col] <= hi + 1e-9

    def test_james_stein_explicit_scalar_prior(self, asset_returns):
        shrunk = expected_returns(asset_returns, method="james_stein", prior=0.05)
        historical = expected_returns(asset_returns, method="mean_historical")
        for col in asset_returns.columns:
            lo, hi = sorted([historical[col], 0.05])
            assert lo - 1e-9 <= shrunk[col] <= hi + 1e-9

    def test_james_stein_shrinkage_intensity_is_computed_at_periodic_scale(self, asset_returns):
        # Regression test for a units bug: mahalanobis distance must be computed on
        # PERIODIC mu/cov (matching t_obs, the periodic observation count), not on
        # already-annualized mu/cov - annualizing first scales mahalanobis up by roughly
        # periods_per_year, which collapsed phi toward 0 for any daily dataset regardless
        # of the assets' true dispersion. With annualization applied correctly, three
        # noticeably-correlated assets over ~2 years of daily data should show a real
        # (not negligible) shrinkage intensity.
        historical = expected_returns(asset_returns, method="mean_historical")
        shrunk = expected_returns(asset_returns, method="james_stein")
        grand_mean = historical.mean()
        implied_phi = 1.0 - float((shrunk - grand_mean).abs().sum() / (historical - grand_mean).abs().sum())
        assert implied_phi > 0.3, f"james_stein shrinkage intensity collapsed to {implied_phi:.4f} - mahalanobis/T scale mismatch?"

    def test_james_stein_shrinkage_intensity_is_independent_of_periods_per_year(self, asset_returns):
        # phi is a property of the periodic data (t_obs periodic observations, periodic
        # mahalanobis) - annualizing convention (252 vs. 365) must not change how hard the
        # estimate gets shrunk, only the final scale of the shrunk number.
        shrunk_252 = expected_returns(asset_returns, method="james_stein", periods_per_year=252)
        shrunk_365 = expected_returns(asset_returns, method="james_stein", periods_per_year=365)
        np.testing.assert_allclose((shrunk_252 * 365 / 252).to_numpy(), shrunk_365.to_numpy(), rtol=1e-9)

    def test_unknown_method_raises(self, asset_returns):
        with pytest.raises(ValueError):
            expected_returns(asset_returns, method="bogus")


class TestCovariance:
    def test_sample_matches_pandas_cov(self, asset_returns):
        result = covariance(asset_returns, method="sample", annualize=False)
        pd.testing.assert_frame_equal(result, asset_returns.cov())

    def test_annualize_scales_linearly(self, asset_returns):
        raw = covariance(asset_returns, method="sample", annualize=False)
        annualized = covariance(asset_returns, method="sample", annualize=True, periods_per_year=252)
        pd.testing.assert_frame_equal(annualized, raw * 252)

    @pytest.mark.parametrize("method", ["sample", "ewma", "shrinkage", "ledoit_wolf", "robust"])
    def test_every_method_is_positive_semidefinite(self, asset_returns, method):
        cov = covariance(asset_returns, method=method)
        eigvals = np.linalg.eigvalsh(cov.to_numpy())
        assert eigvals.min() > -1e-8

    @pytest.mark.parametrize("method", ["sample", "ewma", "shrinkage", "ledoit_wolf", "robust"])
    def test_every_method_is_symmetric(self, asset_returns, method):
        cov = covariance(asset_returns, method=method)
        np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)

    def test_shrinkage_intensity_zero_equals_sample(self, asset_returns):
        shrunk = covariance(asset_returns, method="shrinkage", shrinkage_intensity=0.0, annualize=False)
        sample = covariance(asset_returns, method="sample", annualize=False)
        pd.testing.assert_frame_equal(shrunk, sample)

    def test_shrinkage_intensity_one_is_pure_diagonal(self, asset_returns):
        shrunk = covariance(asset_returns, method="shrinkage", shrinkage_intensity=1.0, annualize=False)
        off_diagonal = shrunk.to_numpy() - np.diag(np.diag(shrunk.to_numpy()))
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-12)

    def test_invalid_shrinkage_intensity_raises(self, asset_returns):
        with pytest.raises(ValueError):
            covariance(asset_returns, method="shrinkage", shrinkage_intensity=1.5)

    def test_unknown_method_raises(self, asset_returns):
        with pytest.raises(ValueError):
            covariance(asset_returns, method="bogus")


class TestFactorModels:
    def test_capm_beta_matches_metrics_risk_beta_at_zero_rf(self, asset_returns, normal_benchmark):
        from portpy.metrics.risk import beta as metrics_beta

        y = asset_returns["A"].reindex(normal_benchmark.index).dropna()
        b = normal_benchmark.reindex(y.index)
        model = capm(y, b, rf=0.0)
        assert model.params["market"] == pytest.approx(metrics_beta(y, b), abs=1e-8)

    def test_capm_zero_correlation_gives_near_zero_beta(self, business_day_index):
        rng = np.random.default_rng(5)
        y = pd.Series(rng.normal(0, 0.01, len(business_day_index)), index=business_day_index)
        b = pd.Series(rng.normal(0, 0.01, len(business_day_index)), index=business_day_index)
        model = capm(y, b)
        assert abs(model.params["market"]) < 0.2

    def test_linear_regression_recovers_known_coefficients(self, business_day_index):
        rng = np.random.default_rng(1)
        x = pd.DataFrame({"f1": rng.normal(0, 1, len(business_day_index))}, index=business_day_index)
        y = 2.0 + 3.0 * x["f1"] + rng.normal(0, 1e-6, len(business_day_index))
        model = linear_regression(y, x)
        assert model.params["const"] == pytest.approx(2.0, abs=1e-3)
        assert model.params["f1"] == pytest.approx(3.0, abs=1e-3)

    def test_rolling_regression_shape_and_leading_nans(self, business_day_index):
        rng = np.random.default_rng(2)
        x = pd.DataFrame({"f1": rng.normal(0, 1, len(business_day_index))}, index=business_day_index)
        y = 1.0 + 0.5 * x["f1"] + rng.normal(0, 0.1, len(business_day_index))
        window = 60
        rolled = rolling_regression(y, x, window=window)
        assert rolled.shape == (len(business_day_index), 2)
        assert rolled.iloc[: window - 1].isna().all().all()
        assert rolled.iloc[window:].notna().all().all()

    def test_fama_french_validates_canonical_columns(self, asset_returns):
        bad_factors = pd.DataFrame({"foo": np.zeros(len(asset_returns))}, index=asset_returns.index)
        with pytest.raises(ValueError):
            fama_french(asset_returns["A"], bad_factors, version=3)

    def test_fama_french_subtracts_rf_column_when_present(self, business_day_index):
        rng = np.random.default_rng(4)
        n = len(business_day_index)
        rf = pd.Series(0.0001, index=business_day_index)
        mkt_rf = pd.Series(rng.normal(0.0005, 0.01, n), index=business_day_index)
        smb = pd.Series(rng.normal(0, 0.005, n), index=business_day_index)
        hml = pd.Series(rng.normal(0, 0.005, n), index=business_day_index)
        y = rf + 1.0 * mkt_rf + rng.normal(0, 1e-6, n)
        factors = pd.DataFrame({"Mkt-RF": mkt_rf, "SMB": smb, "HML": hml, "RF": rf})
        model = fama_french(y, factors, version=3)
        assert model.params["Mkt-RF"] == pytest.approx(1.0, abs=1e-2)

    def test_fama_french_skips_validation_for_noncanonical_version(self, asset_returns):
        factors = pd.DataFrame({"custom_factor": np.random.default_rng(0).normal(0, 0.01, len(asset_returns))}, index=asset_returns.index)
        model = fama_french(asset_returns["A"], factors, version=99)
        assert "custom_factor" in model.params.index

    def test_factor_attribution_contributions_sum_to_mean_return(self, business_day_index):
        rng = np.random.default_rng(6)
        n = len(business_day_index)
        f1 = pd.Series(rng.normal(0.001, 0.01, n), index=business_day_index)
        f2 = pd.Series(rng.normal(-0.0005, 0.01, n), index=business_day_index)
        y = 0.5 * f1 + 1.5 * f2 + rng.normal(0, 0.001, n)
        factors = pd.DataFrame({"f1": f1, "f2": f2})
        attribution = factor_attribution(y, factors)
        assert attribution["contribution"].sum() == pytest.approx(y.mean(), abs=1e-6)
