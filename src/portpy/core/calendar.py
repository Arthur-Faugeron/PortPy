"""
Calendar-alignment helpers for combining assets with different trading calendars.

PortPy's Portfolio never aligns or fills data for you (see the package README) -
if you build a DataFrame from, say, Bitcoin (trades every day) and a stock (trades
Mon-Fri), the stock's columns will have NaNs on weekends. Call one of these
functions before constructing a Portfolio to resolve that deliberately, rather
than accidentally.
"""

from __future__ import annotations

import pandas as pd

_METHODS = ("intersection", "ffill_union", "business_days")


def detect_frequency(index: pd.DatetimeIndex) -> str:
    """
    Roughly classify a DatetimeIndex's spacing (informational only).

    Returns one of: "daily-7" (includes weekends, e.g. crypto), "daily-5" (business
    days, e.g. equities/bonds), "weekly", "monthly", "irregular", or "unknown" (too
    few points to tell).

    Daily-or-lower-frequency only: spacing is bucketed on .dt.days, which
    truncates to whole days, so any sub-daily/intraday index has dt.days == 0
    between consecutive timestamps and gets mislabeled "daily-5"/"daily-7" instead
    of a genuine intraday frequency. Don't rely on this for intraday data - the
    same applies to calendar_coverage_report, which calls it.
    """
    if len(index) < 3:
        return "unknown"
    diffs = pd.Series(index).diff().dropna().dt.days
    median = diffs.median()
    if median <= 1.5:
        weekdays = pd.Index(index).dayofweek
        has_weekend = bool(((weekdays == 5) | (weekdays == 6)).any())
        return "daily-7" if has_weekend else "daily-5"
    if median <= 4:
        return "weekly"
    if median <= 10:
        return "biweekly"
    if median <= 35:
        return "monthly"
    return "irregular"


def align_calendars(
    prices: dict[str, pd.Series] | pd.DataFrame,
    method: str = "intersection",
) -> pd.DataFrame:
    """
    Combine assets that trade on different calendars into one aligned DataFrame.

    Args:
        prices: Either a dict of {asset_name: price_series} or an already-
            combined DataFrame (columns = assets) that may contain NaNs from
            mismatched calendars.
        method: One of:

            - "intersection": keep only dates where every asset has a price
              (drops all weekend/holiday rows if any asset is business-days-only).
              Use this to analyze on the slowest asset's calendar (e.g. equities).
            - "ffill_union": take the union of all dates, forward-fill gaps
              (e.g. carries Friday's stock close through the weekend), then drop
              any remaining leading NaNs. Use this to keep crypto's 24/7 calendar
              while still pricing equities every day.
            - "business_days": reindex everything onto a Mon-Fri business-day
              calendar, forward-filling gaps, then drop remaining leading NaNs.
              Use this to standardize on the equity/bond calendar even if the
              combined frame currently has 7-day-a-week rows.

    Returns:
        A DataFrame with no NaNs, ready to pass to
        :class:~portpy.portfolio.Portfolio.

    "ffill_union" and "business_days" both forward-fill prices on days an
    asset didn't trade, which is equivalent to inserting a fabricated 0% return
    for that asset on those days - this mechanically drags down measured
    correlation/covariance (and beta) between assets on different calendars, a
    stale-price effect related to non-synchronous-trading bias. "business_days"
    additionally reindexes to pd.bdate_range (Mon-Fri) before forward-filling,
    so for a continuously-traded asset (e.g. crypto) it silently discards real
    weekend price action rather than just filling a gap. "intersection" avoids
    both problems (at the cost of a shorter sample) and is generally the safer
    default for correlation/beta/covariance work across heterogeneous calendars.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")

    if isinstance(prices, dict):
        df = pd.concat(prices, axis=1)
    else:
        df = prices.copy()
    df = df.sort_index()

    if method == "intersection":
        return df.dropna(how="any")
    if method == "ffill_union":
        return df.ffill().dropna(how="any")
    # business_days
    bdays = pd.bdate_range(df.index.min(), df.index.max())
    return df.reindex(bdays).ffill().dropna(how="any")


def calendar_coverage_report(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset diagnostic: detected frequency, date range, and NaN count.

    Run this before choosing an align_calendars method - it tells you which
    assets are actually mismatched and by how much.

    Args:
        prices: Columns = asset symbols, index = dates.

    Returns:
        A DataFrame indexed by asset, with columns "frequency", "first_date",
        "last_date", "n_observations", and "n_missing_in_frame".
    """
    rows = []
    for col in prices.columns:
        s = prices[col].dropna()
        rows.append(
            {
                "asset": col,
                "frequency": detect_frequency(pd.DatetimeIndex(s.index)),
                "first_date": s.index.min() if len(s) else pd.NaT,
                "last_date": s.index.max() if len(s) else pd.NaT,
                "n_observations": len(s),
                "n_missing_in_frame": int(prices[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("asset")
