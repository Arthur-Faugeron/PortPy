import numpy as np
import pandas as pd
import pytest

from portpy.core.calendar import align_calendars, calendar_coverage_report, detect_frequency

def test_detect_frequency_business_days():
    idx = pd.bdate_range("2020-01-01", periods=30)
    assert detect_frequency(idx) == "daily-5"


def test_detect_frequency_calendar_days_includes_weekends():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    assert detect_frequency(idx) == "daily-7"


def test_detect_frequency_monthly():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    assert detect_frequency(idx) == "monthly"


def test_detect_frequency_too_short_is_unknown():
    idx = pd.bdate_range("2020-01-01", periods=2)
    assert detect_frequency(idx) == "unknown"


@pytest.fixture
def mismatched_calendars():
    crypto_idx = pd.date_range("2020-01-01", periods=10, freq="D")
    equity_idx = pd.bdate_range("2020-01-01", periods=10)
    crypto = pd.Series(np.linspace(100, 110, len(crypto_idx)), index=crypto_idx)
    equity = pd.Series(np.linspace(50, 55, len(equity_idx)), index=equity_idx)
    return {"BTC": crypto, "AAPL": equity}


def test_align_calendars_intersection_has_no_nans(mismatched_calendars):
    aligned = align_calendars(mismatched_calendars, method="intersection")
    assert not aligned.isna().any().any()
    # Only business days present in both should survive.
    assert all(d.weekday() < 5 for d in aligned.index)


def test_align_calendars_ffill_union_has_no_nans(mismatched_calendars):
    aligned = align_calendars(mismatched_calendars, method="ffill_union")
    assert not aligned.isna().any().any()
    # Union calendar includes weekends (from crypto's 7-day calendar).
    assert any(d.weekday() >= 5 for d in aligned.index)


def test_align_calendars_business_days_has_no_nans(mismatched_calendars):
    aligned = align_calendars(mismatched_calendars, method="business_days")
    assert not aligned.isna().any().any()
    assert all(d.weekday() < 5 for d in aligned.index)


def test_align_calendars_invalid_method_raises(mismatched_calendars):
    with pytest.raises(ValueError):
        align_calendars(mismatched_calendars, method="bogus")


def test_align_calendars_accepts_dataframe_directly(mismatched_calendars):
    combined = pd.concat(mismatched_calendars, axis=1)
    aligned = align_calendars(combined, method="intersection")
    assert not aligned.isna().any().any()


def test_calendar_coverage_report_reports_per_asset_stats(mismatched_calendars):
    combined = pd.concat(mismatched_calendars, axis=1)
    report = calendar_coverage_report(combined)
    assert set(report.index) == {"BTC", "AAPL"}
    assert "frequency" in report.columns
    assert "n_observations" in report.columns
