"""Cross-validation of PortPy's metrics against empyrical and quantstats.

Where the three libraries share the same methodology (annualization by period
count, rf=0 for a clean comparison), results should match to numerical
precision. Where PortPy deliberately uses a different, documented convention
(e.g. cagr's calendar-based competitors, or annual vs. per-period rf), that
difference is asserted explicitly instead of silently ignored - see the
docstring on each test for the specific methodological note.

Synthetic-data tests always run. Real-data tests (@pytest.mark.network)
download from yfinance and are excluded by default in CI (see pyproject's
-m "not network"), but are the ones that matter most for the "does this hold
up on real market data" question - run them locally with:
    pytest -m network tests/validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

empyrical = pytest.importorskip("empyrical")
qs = pytest.importorskip("quantstats.stats")

from portpy.metrics import benchmarks as bench  # noqa: E402
from portpy.metrics import drawdowns as dd  # noqa: E402
from portpy.metrics import performance as perf  # noqa: E402
from portpy.metrics import returns as rts  # noqa: E402
from portpy.metrics import risk  # noqa: E402

ABS_TOL = 1e-6


def _run_full_comparison(r: pd.Series, b: pd.Series) -> None:
    r = r.dropna()
    b = b.reindex(r.index).dropna()
    r = r.reindex(b.index).dropna()
    b = b.reindex(r.index)

    assert perf.sharpe_ratio(r, rf=0.0) == pytest.approx(empyrical.sharpe_ratio(r, risk_free=0.0), abs=ABS_TOL)
    assert perf.sharpe_ratio(r, rf=0.0) == pytest.approx(float(qs.sharpe(r, rf=0.0)), abs=ABS_TOL)

    assert risk.volatility(r, annualized=True) == pytest.approx(empyrical.annual_volatility(r), abs=ABS_TOL)

    assert rts.annualized_return(r, geometric=True) == pytest.approx(empyrical.annual_return(r), abs=ABS_TOL)

    synthetic_prices = rts.prices_from_returns(r)
    assert dd.max_drawdown(synthetic_prices) == pytest.approx(empyrical.max_drawdown(r), abs=ABS_TOL)

    assert risk.value_at_risk(r, method="historical", confidence=0.95) == pytest.approx(
        empyrical.value_at_risk(r, cutoff=0.05), abs=ABS_TOL
    )
    assert risk.conditional_var(r, confidence=0.95) == pytest.approx(
        empyrical.conditional_value_at_risk(r, cutoff=0.05), abs=ABS_TOL
    )
    assert risk.tail_ratio(r) == pytest.approx(empyrical.tail_ratio(r), abs=ABS_TOL)

    assert risk.skewness(r) == pytest.approx(float(qs.skew(r)), abs=ABS_TOL)
    assert risk.kurtosis(r) == pytest.approx(float(qs.kurtosis(r)), abs=ABS_TOL)

    assert perf.sortino_ratio(r, mar=0.0) == pytest.approx(
        empyrical.sortino_ratio(r, required_return=0.0), abs=ABS_TOL
    )

    assert perf.calmar_ratio(r) == pytest.approx(empyrical.calmar_ratio(r), abs=ABS_TOL)

    assert perf.omega_ratio(r, threshold=0.0) == pytest.approx(
        empyrical.omega_ratio(r, risk_free=0.0, required_return=0.0), abs=1e-4
    )

    ep_alpha, ep_beta = empyrical.alpha_beta(r, b, risk_free=0.0)
    assert risk.beta(r, b) == pytest.approx(ep_beta, abs=ABS_TOL)
    assert bench.alpha(r, b, rf=0.0) == pytest.approx(ep_alpha, abs=1e-4)

    assert bench.correlation(r, b) == pytest.approx(float(r.corr(b)), abs=ABS_TOL)

    # --- Documented methodology differences: same ballpark, not exact ---

    # empyrical's up_capture/down_capture re-annualize the filtered, non-contiguous
    # up-day/down-day subsample via its own capture() = annual_return(masked) /
    # annual_return(masked_factor) - i.e. it raises a compounded growth factor to
    # periods_per_year/n where n is the (small, non-contiguous) count of up/down
    # days, which is not a statistically valid use of geometric annualization (see
    # corrections.md finding #4). portpy's up_capture_ratio/down_capture_ratio were
    # fixed to compare compounded sub-period returns directly instead, without
    # re-annualizing. Because empyrical's re-annualization exponent (1/years) can be
    # large when few up/down days exist in the sample, the two methodologies can
    # diverge substantially (observed up to ~3x on real crypto data in this
    # validation harness) - same sign and rough ballpark, not point-for-point.
    ep_up = empyrical.up_capture(r, b)
    ep_down = empyrical.down_capture(r, b)
    pp_up = bench.up_capture_ratio(r, b)
    pp_down = bench.down_capture_ratio(r, b)
    assert np.sign(pp_up) == np.sign(ep_up)
    assert np.sign(pp_down) == np.sign(ep_down)
    assert pp_up == pytest.approx(ep_up, rel=1.0)
    assert pp_down == pytest.approx(ep_down, rel=1.0)

    # portpy's cagr uses period-count years (n/periods_per_year, matching
    # empyrical's own annual_return); quantstats' cagr uses actual elapsed
    # calendar days instead. Both are legitimate, common conventions - they
    # should agree closely on daily data with few gaps, but not bit-for-bit.
    portpy_cagr = rts.cagr(synthetic_prices)
    qs_cagr = float(qs.cagr(r))
    assert portpy_cagr == pytest.approx(qs_cagr, rel=0.05)

    # quantstats' information_ratio() never annualizes - it's exactly
    # mean(diff)/std(diff). portpy's default (annualized=True) differs by
    # precisely sqrt(periods_per_year); at annualized=False the two formulas
    # are identical and should match to numerical precision.
    qs_ir = float(qs.information_ratio(r, b))
    assert perf.information_ratio(r, b, annualized=False) == pytest.approx(qs_ir, abs=ABS_TOL)
    assert perf.information_ratio(r, b, annualized=True) == pytest.approx(
        qs_ir * np.sqrt(252), abs=ABS_TOL
    )

    # quantstats' treynor_ratio() divides the TOTAL cumulative return over the
    # whole sample (unannualized, not divided by elapsed years) by beta, not a
    # mean-per-period return scaled by periods_per_year - a genuinely different
    # metric definition (not a rescaling of portpy's own treynor_ratio), so
    # there's no fixed conversion factor between the two on a multi-year
    # sample. This asserts our documented understanding of quantstats' formula
    # against its actual output, using portpy's own primitives, rather than
    # against portpy's treynor_ratio.
    beta_v = risk.beta(r, b)
    qs_treynor = float(qs.treynor_ratio(r, b, rf=0.0))
    assert qs_treynor == pytest.approx(rts.total_return(r) / beta_v, rel=1e-4)

    # portpy's rf/mar are ANNUAL rates (de-annualized internally); at rf=0
    # this is moot (0 annual == 0 per-period), which is exactly why the sharpe/
    # sortino checks above use rf=0 as the apples-to-apples case. A non-zero-rf
    # check demonstrating the conversion explicitly:
    from portpy.utils.validation import periodic_rate_from_annual

    rf_annual = 0.05
    rf_period = periodic_rate_from_annual(rf_annual, 252)
    assert perf.sharpe_ratio(r, rf=rf_annual) == pytest.approx(
        empyrical.sharpe_ratio(r, risk_free=rf_period), abs=ABS_TOL
    )


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(2024)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    r = pd.Series(rng.normal(0.0006, 0.013, len(idx)), index=idx, name="strategy")
    b = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx, name="benchmark")
    return r, b


def test_synthetic_normal_returns_match_empyrical_and_quantstats(synthetic_data):
    r, b = synthetic_data
    _run_full_comparison(r, b)


def test_synthetic_fat_tailed_returns_match_empyrical_and_quantstats():
    """Repeat with Student-t (fat-tailed) returns - realistic non-Normal shape."""
    rng = np.random.default_rng(99)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    r = pd.Series(0.0004 + rng.standard_t(df=4, size=len(idx)) * 0.008, index=idx)
    b = pd.Series(0.0003 + rng.standard_t(df=5, size=len(idx)) * 0.007, index=idx)
    _run_full_comparison(r, b)


@pytest.mark.network
def test_real_yfinance_data_matches_empyrical_and_quantstats():
    """Downloads real SPY (benchmark) and AAPL (asset) daily data via yfinance."""
    yf = pytest.importorskip("yfinance")

    aapl = yf.download("AAPL", period="3y", progress=False, auto_adjust=True)["Close"].iloc[:, 0]
    spy = yf.download("SPY", period="3y", progress=False, auto_adjust=True)["Close"].iloc[:, 0]

    r = rts.simple_returns(aapl)
    b = rts.simple_returns(spy)
    _run_full_comparison(r, b)


@pytest.mark.network
def test_real_yfinance_crypto_matches_empyrical_and_quantstats():
    """Crypto trades 7 days/week - a real stress test of the annualization/frequency assumptions."""
    yf = pytest.importorskip("yfinance")

    btc = yf.download("BTC-USD", period="2y", progress=False, auto_adjust=True)["Close"].iloc[:, 0]
    eth = yf.download("ETH-USD", period="2y", progress=False, auto_adjust=True)["Close"].iloc[:, 0]

    r = rts.simple_returns(btc)
    b = rts.simple_returns(eth)
    _run_full_comparison(r, b)
