"""Drawdown analysis: depth, duration, recovery, and tail risk of peak-to-trough declines.

All functions here take a **price** series (or any cumulative-value series, e.g.
a synthetic index built from `(1 + returns).cumprod()`) - not a returns series.
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
    """Percentage decline from the running peak at every point in time (always <= 0)."""
    p = prices.dropna()
    running_max = p.cummax()
    dd = p / running_max - 1.0
    return dd.rename("drawdown")


def _drawdown_episodes(prices: pd.Series) -> pd.DataFrame:
    """Identify each distinct peak -> trough -> (recovery|ongoing) episode."""
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
    """The single deepest peak-to-trough decline (a negative number, or 0.0 if the series never fell)."""
    p = prices.dropna()
    ensure_min_observations(p, 1, "prices")
    value = float(drawdown_series(p).min())
    return MetricResult(value, "max_drawdown", unit="%") if as_result else value


def drawdown_duration(prices: pd.Series) -> pd.Series:
    """At every point in time, how many periods have passed since the last new high."""
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
    """The `n` deepest drawdown episodes, sorted worst-first, with start/trough/end/depth/duration."""
    p = prices.dropna()
    ensure_min_observations(p, 2, "prices")
    episodes = _drawdown_episodes(p)
    if episodes.empty:
        return episodes
    return episodes.sort_values("depth").head(n).reset_index(drop=True)


def time_to_recovery(prices: pd.Series) -> float | None:
    """Periods it took to recover from the single worst drawdown, or None if it hasn't recovered yet."""
    p = prices.dropna()
    episodes = _drawdown_episodes(p)
    if episodes.empty:
        return None
    worst = episodes.sort_values("depth").iloc[0]
    return float(worst["duration_to_recovery"]) if worst["recovered"] else None


def average_drawdown(prices: pd.Series, as_result: bool = False) -> float | MetricResult:
    """Mean depth across all distinct drawdown episodes (not a continuous time-average - see pain_index for that)."""
    p = prices.dropna()
    episodes = _drawdown_episodes(p)
    value = float(episodes["depth"].mean()) if not episodes.empty else 0.0
    return MetricResult(value, "average_drawdown", unit="%") if as_result else value


def drawdown_at_risk(
    prices: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    as_result: bool = False,
) -> float | MetricResult:
    """VaR applied to the drawdown series: the drawdown level breached only `1 - confidence` of the time."""
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
        summary="The running percentage decline from the highest value seen so far, at every point in time.",
        formula="price_t / max(price_0..t) - 1",
        how_to_read="Always <= 0. A drawdown chart that spends most of its time near 0 and dips briefly is healthier than one that lingers deeply negative.",
        good_vs_bad="Shallower and shorter dips are better. Use top_n_drawdowns and drawdown_duration to see individual episodes rather than reading the raw series alone.",
    )
)

register(
    Explanation(
        name="max_drawdown",
        category="metric",
        summary="The single worst peak-to-trough loss the portfolio ever experienced.",
        formula="min(drawdown_series)",
        how_to_read="Reported as a negative percentage: -0.30 means the portfolio was, at its worst point, 30% below its prior peak.",
        good_vs_bad="Rules of thumb: shallower than -10% very mild, -10% to -25% typical for equities, worse than -40% severe (comparable to major bear markets) - context (asset class, leverage) matters a lot.",
        caveats="A single number - it tells you nothing about how long the drawdown lasted or whether it recovered. Pair with drawdown_duration and time_to_recovery.",
        interpret=lambda v: (
            "mild" if v > -0.10 else "typical for a risky asset" if v > -0.25 else "severe" if v > -0.40 else "extreme"
        ),
    )
)

register(
    Explanation(
        name="drawdown_duration",
        category="metric",
        summary="A running counter of how many periods have elapsed since the portfolio last hit a new high.",
        how_to_read="Rises steadily while underwater and resets to 0 the instant a new high is made. Long stretches of high values mean the portfolio spent a long time recovering, even if the drawdown itself wasn't very deep.",
        good_vs_bad="Shorter is better - a portfolio that's frequently at new highs (duration resets often) is more comfortable to hold even at similar drawdown depth.",
    )
)

register(
    Explanation(
        name="time_to_recovery",
        category="metric",
        summary="How many periods it took the portfolio to climb back to its prior peak after the single worst drawdown.",
        how_to_read="Expressed in the same frequency as your data (e.g. trading days). None means the portfolio never recovered by the end of the sample.",
        good_vs_bad="Shorter is better. `None` is a serious red flag if the sample is recent - it means the worst drawdown is still open.",
        interpret=lambda v: "still underwater as of the end of the sample" if v is None else f"{v:.0f} periods to recover",
    )
)

register(
    Explanation(
        name="top_n_drawdowns",
        category="metric",
        summary="A table of the worst individual drawdown episodes, each with its start, trough, end (if recovered), depth, and duration.",
        how_to_read="Sorted worst-first by depth. `recovered=False` rows are still open as of the last observation.",
        good_vs_bad="Fewer, shallower, faster-recovering episodes are better. Watch especially for unrecovered episodes near the end of the sample.",
    )
)

register(
    Explanation(
        name="average_drawdown",
        category="metric",
        summary="The mean depth across every distinct drawdown episode (peak-to-trough dip), treating each episode once regardless of how long it lasted.",
        formula="mean(depth of each drawdown episode)",
        how_to_read="A milder number than max_drawdown by construction - it answers 'how bad is a *typical* drawdown', not 'how bad was the worst one'.",
        good_vs_bad="Closer to zero is better. Compare to max_drawdown: a small gap means drawdowns are fairly uniform in severity; a big gap means one outlier event dominates.",
        caveats="Different from pain_index, which time-averages the drawdown series continuously (so it also reflects how long each episode lasted, not just its depth).",
        interpret=lambda v: f"{v:.2%} typical episode depth",
    )
)

register(
    Explanation(
        name="drawdown_at_risk",
        category="metric",
        summary="Value-at-Risk applied to drawdown depth instead of returns: the drawdown level only breached `1 - confidence` of the time.",
        formula="percentile(drawdown_series, 100*(1-confidence))",
        how_to_read="A 95% drawdown-at-risk of -0.18 means the portfolio was deeper than 18% underwater only 5% of the time in this sample.",
        good_vs_bad="Closer to zero is better/safer.",
        interpret=lambda v: f"{v:.1%}",
    )
)
