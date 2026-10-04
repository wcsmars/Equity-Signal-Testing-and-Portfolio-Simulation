"""The NYSE trading calendar against the closures that actually happened.

qcore.quality.nyse_bdays decides which days count as sessions: month-end
decision dates, trading-day counts around the turn of the month, the live
session check and the calendar-gap data check all read it. Its holiday rules
are compared here with a hard-coded record, not with themselves: every
full-day closure on a weekday from 2000 to 2025 as it occurred (the dates on
which no US equity traded), and the exchange's published holiday schedule
for 2026 and 2027. A rule that is dropped, added or observed on the wrong
day changes at least one year below.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qcore.quality import (  # noqa: E402
    SPECIAL_CLOSURES, nyse_bdays, nyse_holidays)

# year -> (sessions held, weekday closures as MM-DD). Holidays that fell on
# a weekend and were not observed (New Year's Day 2000, 2005, 2011, 2022)
# simply do not appear.
CALENDAR = {
    2000: (252, "01-17 02-21 04-21 05-29 07-04 09-04 11-23 12-25"),
    2001: (248, "01-01 01-15 02-19 04-13 05-28 07-04 09-03 "
                "09-11 09-12 09-13 09-14 11-22 12-25"),
    2002: (252, "01-01 01-21 02-18 03-29 05-27 07-04 09-02 11-28 12-25"),
    2003: (252, "01-01 01-20 02-17 04-18 05-26 07-04 09-01 11-27 12-25"),
    2004: (252, "01-01 01-19 02-16 04-09 05-31 06-11 07-05 09-06 11-25 12-24"),
    2005: (252, "01-17 02-21 03-25 05-30 07-04 09-05 11-24 12-26"),
    2006: (251, "01-02 01-16 02-20 04-14 05-29 07-04 09-04 11-23 12-25"),
    2007: (251, "01-01 01-02 01-15 02-19 04-06 05-28 07-04 09-03 11-22 12-25"),
    2008: (253, "01-01 01-21 02-18 03-21 05-26 07-04 09-01 11-27 12-25"),
    2009: (252, "01-01 01-19 02-16 04-10 05-25 07-03 09-07 11-26 12-25"),
    2010: (252, "01-01 01-18 02-15 04-02 05-31 07-05 09-06 11-25 12-24"),
    2011: (252, "01-17 02-21 04-22 05-30 07-04 09-05 11-24 12-26"),
    2012: (250, "01-02 01-16 02-20 04-06 05-28 07-04 09-03 "
                "10-29 10-30 11-22 12-25"),
    2013: (252, "01-01 01-21 02-18 03-29 05-27 07-04 09-02 11-28 12-25"),
    2014: (252, "01-01 01-20 02-17 04-18 05-26 07-04 09-01 11-27 12-25"),
    2015: (252, "01-01 01-19 02-16 04-03 05-25 07-03 09-07 11-26 12-25"),
    2016: (252, "01-01 01-18 02-15 03-25 05-30 07-04 09-05 11-24 12-26"),
    2017: (251, "01-02 01-16 02-20 04-14 05-29 07-04 09-04 11-23 12-25"),
    2018: (251, "01-01 01-15 02-19 03-30 05-28 07-04 09-03 11-22 12-05 12-25"),
    2019: (252, "01-01 01-21 02-18 04-19 05-27 07-04 09-02 11-28 12-25"),
    2020: (253, "01-01 01-20 02-17 04-10 05-25 07-03 09-07 11-26 12-25"),
    2021: (252, "01-01 01-18 02-15 04-02 05-31 07-05 09-06 11-25 12-24"),
    2022: (251, "01-17 02-21 04-15 05-30 06-20 07-04 09-05 11-24 12-26"),
    2023: (250, "01-02 01-16 02-20 04-07 05-29 06-19 07-04 09-04 11-23 12-25"),
    2024: (252, "01-01 01-15 02-19 03-29 05-27 06-19 07-04 09-02 11-28 12-25"),
    2025: (250, "01-01 01-09 01-20 02-17 04-18 05-26 06-19 07-04 09-01 "
                "11-27 12-25"),
    # published schedule
    2026: (251, "01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25"),
    2027: (251, "01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24"),
}

# Days that look like holidays to a careless rule but were ordinary or
# shortened SESSIONS.
OPEN = [
    "2004-12-31", "2010-12-31", "2021-12-31",  # Jan 1 on a Saturday is not
    "2027-12-31",                              # observed on the Friday
    "2020-06-19", "2021-06-18",   # before Juneteenth was an exchange holiday
    "2024-10-14", "2024-11-11",   # Columbus / Veterans Day: bond market only
    "2023-11-24", "2024-07-03", "2024-12-24",  # early closes are sessions
    "2001-09-10", "2001-09-17",   # either side of the September 2001 closure
    "2012-10-26", "2012-10-31",   # either side of the storm closure
    "2018-12-04", "2018-12-06", "2025-01-08", "2025-01-10",
]

# Full-day closures that follow no holiday rule.
UNSCHEDULED = [
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",
    "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30",
    "2018-12-05", "2025-01-09",
]


def _closed(year: int) -> set[str]:
    days = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
    held = nyse_bdays(f"{year}-01-01", f"{year}-12-31")
    return {d.strftime("%m-%d") for d in days.difference(held)}


@pytest.mark.parametrize("year", sorted(CALENDAR))
def test_weekday_closures_match_the_record(year):
    sessions, closures = CALENDAR[year]
    expected = set(closures.split())
    got = _closed(year)
    assert got == expected, (
        f"{year}: rules close {sorted(got - expected)} which traded, and "
        f"keep open {sorted(expected - got)} which did not")
    assert len(nyse_bdays(f"{year}-01-01", f"{year}-12-31")) == sessions


def test_the_record_itself_is_consistent():
    # sessions = weekdays - closures, and every closure is a weekday: guards
    # the table against a typo, independently of the rules under test
    for year, (sessions, closures) in CALENDAR.items():
        days = pd.DatetimeIndex([f"{year}-{d}" for d in closures.split()])
        assert (days.dayofweek < 5).all(), year
        assert days.is_unique and days.is_monotonic_increasing, year
        weekdays = len(pd.bdate_range(f"{year}-01-01", f"{year}-12-31"))
        assert weekdays - len(days) == sessions, year


def test_named_sessions_are_open():
    held = nyse_bdays("2000-01-01", "2027-12-31")
    assert [d for d in OPEN if pd.Timestamp(d) not in held] == []


def test_unscheduled_closures_are_closed_and_listed_separately():
    held = nyse_bdays("2000-01-01", "2027-12-31")
    assert [d for d in UNSCHEDULED if pd.Timestamp(d) in held] == []
    assert sorted(str(d.date()) for d in SPECIAL_CLOSURES) == UNSCHEDULED
    regular = nyse_holidays("2000-01-01", "2027-12-31")
    assert len(regular.intersection(SPECIAL_CLOSURES)) == 0, \
        "a closure that follows a rule belongs in the rules"
    listed = {f"{y}-{d}" for y, (_, c) in CALENDAR.items() for d in c.split()}
    assert {str(d.date()) for d in regular if d.dayofweek < 5} \
        == listed - set(UNSCHEDULED)


def test_range_ends_are_inclusive_and_partial_ranges_agree():
    week = nyse_bdays("2024-03-25", "2024-04-01")  # Good Friday on 03-29
    assert [str(d.date()) for d in week] == [
        "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01"]
    assert len(nyse_bdays("2024-03-29", "2024-03-29")) == 0
    assert len(nyse_bdays("2024-03-28", "2024-03-28")) == 1
    whole = nyse_bdays("2000-01-01", "2027-12-31")
    assert len(whole) == sum(s for s, _ in CALENDAR.values())
    assert whole.is_unique and whole.is_monotonic_increasing
    assert (whole.dayofweek < 5).all()
