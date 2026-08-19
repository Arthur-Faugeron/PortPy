import numpy as np
import pandas as pd
import pytest

from portpy.metrics import drawdowns as m


@pytest.fixture
def v_shaped_prices():
    # Peak of 100, trough of 70 (-30%), recovers to 110, then a second smaller dip.
    idx = pd.bdate_range("2020-01-01", periods=10)
    values = [100, 90, 80, 70, 85, 100, 110, 105, 108, 112]
    return pd.Series(values, index=idx, dtype=float)


def test_drawdown_series_never_positive(v_shaped_prices):
    dd = m.drawdown_series(v_shaped_prices)
    assert (dd <= 1e-12).all()
    assert dd.iloc[0] == pytest.approx(0.0)


def test_max_drawdown_matches_manual_calc(v_shaped_prices):
    value = m.max_drawdown(v_shaped_prices)
    assert value == pytest.approx(70.0 / 100.0 - 1.0)


def test_max_drawdown_zero_for_monotonic_series():
    prices = pd.Series(np.linspace(100, 200, 20))
    assert m.max_drawdown(prices) == pytest.approx(0.0)


def test_drawdown_duration_resets_at_new_highs(v_shaped_prices):
    duration = m.drawdown_duration(v_shaped_prices)
    # New highs at positions 0 and 6 (110) and 9 (112) reset duration to 0.
    assert duration.iloc[0] == 0
    assert duration.iloc[6] == 0
    assert duration.iloc[9] == 0
    assert duration.iloc[3] == 3  # third consecutive period under the initial peak


def test_top_n_drawdowns_sorted_worst_first(v_shaped_prices):
    episodes = m.top_n_drawdowns(v_shaped_prices, n=5)
    assert list(episodes["depth"]) == sorted(episodes["depth"])
    assert episodes.iloc[0]["depth"] == pytest.approx(-0.30)


def test_top_n_drawdowns_marks_unrecovered_episode_at_series_end():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.Series([100, 90, 80, 70, 60], index=idx, dtype=float)
    episodes = m.top_n_drawdowns(prices, n=5)
    assert len(episodes) == 1
    assert not episodes.iloc[0]["recovered"]
    assert pd.isna(episodes.iloc[0]["end"])


def test_time_to_recovery_none_if_worst_drawdown_unrecovered():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.Series([100, 90, 80, 70, 60], index=idx, dtype=float)
    assert m.time_to_recovery(prices) is None


def test_time_to_recovery_counts_periods_to_recover(v_shaped_prices):
    ttr = m.time_to_recovery(v_shaped_prices)
    assert ttr == pytest.approx(5.0)  # peak at pos 0 (100), back to 100 at pos 5


def test_time_to_recovery_as_result_is_metric_result(v_shaped_prices):
    from portpy.explain import MetricResult

    result = m.time_to_recovery(v_shaped_prices, as_result=True)
    assert isinstance(result, MetricResult)
    assert result.name == "time_to_recovery"
    assert result == pytest.approx(5.0)


def test_time_to_recovery_unrecovered_returns_none_even_with_as_result():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.Series([100, 90, 80, 70, 60], index=idx, dtype=float)
    assert m.time_to_recovery(prices, as_result=True) is None


def test_average_drawdown_is_mean_of_episode_depths(v_shaped_prices):
    episodes = m.top_n_drawdowns(v_shaped_prices, n=100)
    expected = episodes["depth"].mean()
    assert m.average_drawdown(v_shaped_prices) == pytest.approx(expected)


def test_average_drawdown_zero_when_no_drawdowns():
    prices = pd.Series(np.linspace(100, 200, 10))
    assert m.average_drawdown(prices) == pytest.approx(0.0)


def test_drawdown_at_risk_matches_percentile_of_drawdown_series(v_shaped_prices):
    dd = m.drawdown_series(v_shaped_prices)
    expected = np.percentile(dd, 5)
    assert m.drawdown_at_risk(v_shaped_prices, confidence=0.95) == pytest.approx(expected)


def test_conditional_drawdown_at_risk_at_least_as_severe_as_drawdown_at_risk(v_shaped_prices):
    dar = m.drawdown_at_risk(v_shaped_prices, confidence=0.95)
    cdar = m.conditional_drawdown_at_risk(v_shaped_prices, confidence=0.95)
    assert cdar <= dar


def test_conditional_drawdown_at_risk_matches_manual_tail_mean():
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(rng.normal(0.0002, 0.01, len(idx)), index=idx)
    prices = (1.0 + r).cumprod()
    dd = m.drawdown_series(prices).to_numpy()
    confidence = 0.95
    alpha = 1.0 - confidence
    cutoff_index = int((len(dd) - 1) * alpha)
    expected = float(np.mean(np.partition(dd, cutoff_index)[: cutoff_index + 1]))
    assert m.conditional_drawdown_at_risk(prices, confidence=confidence) == pytest.approx(expected)


def test_conditional_drawdown_at_risk_as_result_is_metric_result(v_shaped_prices):
    from portpy.explain import MetricResult

    result = m.conditional_drawdown_at_risk(v_shaped_prices, confidence=0.95, as_result=True)
    assert isinstance(result, MetricResult)
    assert result.name == "conditional_drawdown_at_risk"
