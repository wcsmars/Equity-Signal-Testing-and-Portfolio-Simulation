"""Month-boundary helpers for truncated price histories.

A month contributes a decision only when its final NYSE session is present;
a missing session never moves a rebalance to an earlier row. The final row
is retained only when no NYSE trading day remains in its month under the
approximate holiday rules in qcore.quality (regular holidays such as Good
Friday and Memorial Day, plus listed special closures). A warning asks for
verification when that decision depends on a holiday rule; unscheduled
closures cannot be anticipated. A trailing row with trading days left in its
month is excluded to avoid treating a partial month as a rebalance date.
"""

import warnings

import pandas as pd

from .quality import nyse_bdays


def confirmed_month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month in `index`; the trailing month's row
    is dropped unless no NYSE trading day remains in its month. Missing
    historical month-ends raise, preserving calendar-month signal lags."""
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans:
        raise ValueError("calendar requires a DatetimeIndex without NaT")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("calendar dates must be unique and sorted")
    s = index.to_series()
    last = s.groupby(index.to_period("M")).max()
    # A later month's row does not prove that an earlier month's final
    # session is present: never move a rebalance earlier across a data gap.
    if len(last) == 0:
        return pd.DatetimeIndex([])
    sessions = nyse_bdays(index[0].to_period("M").start_time,
                         index[-1] + pd.offsets.MonthEnd(0))
    expected = sessions.to_series().groupby(sessions.to_period("M")).max()
    complete = last.eq(expected.reindex(last.index))
    prior_months = expected.index[:-1]
    missing = prior_months[~complete.reindex(prior_months, fill_value=False)]
    if len(missing):
        # Dropping an interior month silently turns shift(12) into a
        # 13-month (or longer) signal in monthly strategies.
        raise ValueError(f"missing or invalid historical month-end session(s): {list(map(str, missing))}")
    historical = last.iloc[:-1]
    if not complete.iloc[-1]:
        return pd.DatetimeIndex(historical)
    final = last.iloc[-1]
    rest = pd.bdate_range(final + pd.Timedelta(days=1), final + pd.offsets.MonthEnd(0))
    if len(rest) and len(nyse_bdays(rest[0], rest[-1])):
        return pd.DatetimeIndex(historical)
    if len(rest):
        warnings.warn(
            f"treating final data date {final.date()} as a month-end: the "
            f"remaining weekday(s) {', '.join(str(d.date()) for d in rest)} are "
            "NYSE holidays under qcore.quality's rules - verify against the "
            "published NYSE calendar",
            stacklevel=2)
    return pd.DatetimeIndex(pd.concat([historical, last.iloc[-1:]]))


def in_complete_month(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean per row: this row's month has a confirmed month-end, so
    'trading days left in the month' is knowable from the data alone."""
    ok = set(confirmed_month_ends(index).to_period("M"))
    return pd.Series(index.to_period("M").isin(list(ok)), index=index)
