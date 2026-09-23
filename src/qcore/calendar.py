"""Month-boundary helpers for truncated price histories.

A completed month is identified by a later month in the data. The final row
is retained only when no later weekday remains in its month, with a warning.
This weekday heuristic is not a complete exchange calendar: month-ending
holidays can move the final session earlier. A mid-month trailing row is
excluded to avoid treating a partial month as a rebalance date.
"""

import warnings

import pandas as pd


def confirmed_month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month in `index`; the trailing month's row
    is dropped unless it passes the business-day month-end heuristic."""
    s = index.to_series()
    last = s.groupby(index.to_period("M")).max()
    if len(last) == 0:
        return pd.DatetimeIndex([])
    final = last.iloc[-1]
    if (final + pd.offsets.BDay(1)).month != final.month:
        warnings.warn(
            f"treating final data date {final.date()} as a month-end by "
            "weekday heuristic - verify against the NYSE calendar (a "
            "month-ending holiday shifts the true last trading day earlier)",
            stacklevel=2)
        return pd.DatetimeIndex(last.values)
    return pd.DatetimeIndex(last.iloc[:-1].values)


def in_complete_month(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean per row: this row's month has a confirmed month-end, so
    'trading days left in the month' is knowable from the data alone."""
    ok = set(confirmed_month_ends(index).to_period("M"))
    return pd.Series(index.to_period("M").isin(list(ok)), index=index)
