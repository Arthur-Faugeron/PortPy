"""
Drawdown analysis: depth, duration, recovery, and tail risk of peak-to-trough declines.

All functions here take a price series (or any cumulative-value series, e.g.
a synthetic index built from (1 + returns).cumprod()) - not a returns series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, MetricResult, register
from portpy.utils.constants import DEFAULT_CONFIDENCE_LEVEL
from portpy.utils.validation import ensure_min_observations, validate_confidence

__all__ = [
    "drawdown_series",
    "max_drawdown",
    "drawdown_duration",
    "time_to_recovery",
    "top_n_drawdowns",
    "average_drawdown",
    "drawdown_at_risk",
]


def drawdown_series(prices: pd.Series) -> pd.Series:
    """
    Percentage decline from the running peak at every point in time (always <= 0).
    """
    p = prices.dropna()
    running_max = p.cummax()
    dd = p / running_max - 1.0
    return dd.rename("drawdown")


def _drawdown_episodes(prices: pd.Series) -> pd.DataFrame:
    """
    Identify each distinct peak -> trough -> (recovery|ongoing) episode.
    """
    p = prices.dropna()
    dd = drawdown_series(p)
    underwater = dd < 0
    n = len(p)
    episodes: list[dict] = []
    in_dd = False
    seg_start_pos = 0

    for pos in range(n):
        if underwater.iloc[pos] and not in_dd:
            in_dd = True
            seg_start_pos = pos - 1 if pos > 0 else pos
        if in_dd and (not underwater.iloc[pos] or pos == n - 1):
            recovered = not underwater.iloc[pos]
            seg = dd.iloc[seg_start_pos : pos + 1]
            trough_rel = int(np.argmin(seg.values))
            trough_pos = seg_start_pos + trough_rel
            episodes.append(
                {
                    "start": p.index[seg_start_pos],
                    "trough": p.index[trough_pos],
                    "end": p.index[pos] if recovered else pd.NaT,
                    "depth": float(dd.iloc[trough_pos]),
                    "duration_to_trough": trough_pos - seg_start_pos,
                    "duration_to_recovery": (pos - seg_start_pos) if recovered else np.nan,
                    "recovered": recovered,
                }
            )
            in_dd = False

    return pd.DataFrame(
        episodes,
        columns=["start", "trough", "end", "depth", "duration_to_trough", "duration_to_recovery", "recovered"],
    )


def max_drawdown(prices: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    The single deepest peak-to-trough decline (a negative number, or 0.0 if the series never fell).
    """
    p = prices.dropna()
    ensure_min_observations(p, 1, "prices")
    value = float(drawdown_series(p).min())
    return MetricResult(value, "max_drawdown", unit="%") if as_result else value


def drawdown_duration(prices: pd.Series) -> pd.Series:
    """
    At every point in time, how many periods have passed since the last new high.
    """
    p = prices.dropna()
    n = len(p)
    positions = np.arange(n)
    running_max = p.cummax()
    at_peak = p.to_numpy() >= running_max.to_numpy()
    last_peak_pos = np.where(at_peak, positions, -1)
    last_peak_pos = np.maximum.accumulate(last_peak_pos)
    duration = positions - last_peak_pos
    return pd.Series(duration, index=p.index, name="drawdown_duration")


def top_n_drawdowns(prices: pd.Series, n: int = 5) -> pd.DataFrame:
    """
    The n deepest drawdown episodes, sorted worst-first, with start/trough/end/depth/duration.
    """
    p = prices.dropna()
    ensure_min_observations(p, 2, "prices")
    episodes = _drawdown_episodes(p)
    if episodes.empty:
        return episodes
    return episodes.sort_values("depth").head(n).reset_index(drop=True)


def time_to_recovery(prices: pd.Series) -> float | None:
    """
    Periods it took to recover from the single worst drawdown, or None if it hasn't recovered yet.
    """
    p = prices.dropna()
    episodes = _drawdown_episodes(p)
    if episodes.empty:
        return None
    worst = episodes.sort_values("depth").iloc[0]
    return float(worst["duration_to_recovery"]) if worst["recovered"] else None


def average_drawdown(prices: pd.Series, as_result: bool = False) -> float | MetricResult:
    """
    Mean depth across all distinct drawdown episodes (not a continuous time-average see pain_index for that).
    """
    p = prices.dropna()
    episodes = _drawdown_episodes(p)
    value = float(episodes["depth"].mean()) if not episodes.empty else 0.0
    return MetricResult(value, "average_drawdown", unit="%") if as_result else value


def drawdown_at_risk(
    prices: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    as_result: bool = False,
) -> float | MetricResult:
    """
    VaR applied to the drawdown series: the drawdown level breached only 1. Confidence of the time.
    """
    validate_confidence(confidence)
    p = prices.dropna()
    ensure_min_observations(p, 2, "prices")
    dd = drawdown_series(p)
    value = float(np.percentile(dd, 100 * (1.0 - confidence)))
    return MetricResult(value, "drawdown_at_risk", unit="%") if as_result else value


register(
    Explanation(
        name="drawdown_series",
        category="metric",
        summary="The percentage decline of a portfolio value from its highest previous level at each point in time.",
        formula="current_value / running_max_value - 1",
        how_to_read="Values are always zero or negative. A value of -0.20 means the portfolio is 20% below its previous peak.",
        good_vs_bad="Smaller and shorter drawdowns indicate better capital preservation. The series is best analyzed together with drawdown duration and recovery metrics.",
        caveats="Drawdown measures losses from previous peaks but does not show the path taken, volatility before the decline, or recovery speed.",
        interpret=lambda v: f"{v:.2%} below previous peak",
    )
)

register(
    Explanation(
        name="max_drawdown",
        category="metric",
        summary="The largest observed loss from a historical peak to the following lowest point before a recovery or the end of the sample.",
        formula="minimum(drawdown_series)",
        how_to_read="Reported as a negative percentage. A value of -0.30 means the portfolio lost 30% from its previous peak at its worst point.",
        good_vs_bad="A smaller absolute drawdown is generally preferable because large losses require disproportionately larger gains to recover.",
        caveats="Max drawdown only captures the worst event. It does not measure frequency, duration, or how quickly losses were recovered.",
        interpret=lambda v: (
            "mild" if v > -0.10 else "moderate" if v > -0.25 else "severe" if v > -0.40 else "extreme"
        ),
    )
)

register(
    Explanation(
        name="drawdown_duration",
        category="metric",
        summary="Tracks how long the portfolio has remained below its previous high after entering a drawdown period.",
        formula="current_period - last_peak_period",
        how_to_read="Higher values indicate longer recovery periods. The value resets when a new portfolio high is reached.",
        good_vs_bad="Shorter drawdown durations are generally preferable because investors recover losses faster.",
        caveats="Duration does not measure the size of the loss. A portfolio can have a long shallow drawdown or a short severe drawdown.",
        interpret=lambda v: f"{v:.0f} periods since previous high",
    )
)

register(
    Explanation(
        name="time_to_recovery",
        category="metric",
        summary="Measures the number of periods required for the portfolio to recover from its largest drawdown back to its previous peak.",
        formula="recovery_date - drawdown_start_date",
        how_to_read="The value represents recovery time in the same frequency as the input data. None means the portfolio has not recovered by the end of the sample.",
        good_vs_bad="Shorter recovery periods indicate faster capital recovery. Long unrecovered periods increase investor risk and behavioral pressure.",
        caveats="Recovery time depends on the observation period. A recent unresolved drawdown may appear worse simply because insufficient recovery time has passed.",
        interpret=lambda v: "not recovered" if v is None else f"{v:.0f} periods to recover",
    )
)

register(
    Explanation(
        name="top_n_drawdowns",
        category="metric",
        summary="A ranked table of the largest drawdown episodes, including their timing, depth, and recovery characteristics.",
        formula="sort(drawdown_episodes, by=depth)",
        how_to_read="The table identifies the worst historical loss periods and whether each drawdown eventually recovered.",
        good_vs_bad="Fewer severe drawdowns and faster recoveries generally indicate stronger downside resilience.",
        caveats="Historical drawdown episodes may not represent future losses, especially when market conditions change.",
    )
)

register(
    Explanation(
        name="average_drawdown",
        category="metric",
        summary="The average depth of completed drawdown episodes, showing the typical size of portfolio declines.",
        formula="mean(drawdown_episode_depths)",
        how_to_read="A value of -0.05 means the average drawdown episode reduced portfolio value by approximately 5%.",
        good_vs_bad="Values closer to zero indicate smaller typical losses. Compare with max_drawdown to identify whether risk is dominated by rare extreme events.",
        caveats="Average drawdown ignores how long each episode lasts. A portfolio can have small but very persistent drawdowns.",
        interpret=lambda v: f"{v:.2%} average drawdown depth",
    )
)

register(
    Explanation(
        name="drawdown_at_risk",
        category="metric",
        summary="A downside risk measure applying a percentile calculation to historical drawdown levels instead of returns.",
        formula="percentile(drawdown_series, 100 * (1 - confidence))",
        how_to_read="A value of -0.20 at 95% confidence means drawdowns exceeded 20% only during the worst 5% of observed periods.",
        good_vs_bad="Values closer to zero indicate lower historical drawdown severity.",
        caveats="Like other percentile-based risk measures, this depends on the historical sample and may underestimate future extreme events.",
        interpret=lambda v: f"{v:.2%} drawdown threshold",
    )
)
