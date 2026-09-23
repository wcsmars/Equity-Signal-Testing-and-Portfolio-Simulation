"""Exact positional math for WalkForwardSplitter on a tiny 30-day index."""

import pandas as pd
import pytest

from alpha_lab.backtest.walkforward import WalkForwardSplitter
from alpha_lab.config.schema import WalkForwardConfig
from alpha_lab.core.errors import ConfigError

DATES = pd.bdate_range("2020-01-01", periods=30)


def _cfg(**overrides):
    base = dict(scheme="rolling", train_days=10, test_days=5, purge_days=2, embargo_days=0)
    base.update(overrides)
    return WalkForwardConfig(**base)


def _pos(ts):
    return int(DATES.get_loc(pd.Timestamp(ts)))


def _positions(window):
    return (
        _pos(window.train_start),
        _pos(window.train_end),
        _pos(window.test_start),
        _pos(window.test_end),
    )


def test_rolling_windows_exact_positions():
    # gap = 2, p0 = 12; blocks at p = 12, 17, 22, 27 (last truncated to 3 days)
    windows = WalkForwardSplitter(_cfg()).split(DATES)
    assert [_positions(w) for w in windows] == [
        (0, 9, 12, 16),
        (5, 14, 17, 21),
        (10, 19, 22, 26),
        (15, 24, 27, 29),
    ]


def test_expanding_train_start_pinned_to_zero():
    windows = WalkForwardSplitter(_cfg(scheme="expanding")).split(DATES)
    assert [_pos(w.train_start) for w in windows] == [0, 0, 0, 0]
    # everything else identical to rolling
    assert [(_pos(w.train_end), _pos(w.test_start), _pos(w.test_end)) for w in windows] == [
        (9, 12, 16),
        (14, 17, 21),
        (19, 22, 26),
        (24, 27, 29),
    ]


def test_purge_gap_exactly_respected_and_no_overlap():
    gap = 2
    windows = WalkForwardSplitter(_cfg()).split(DATES)
    for w in windows:
        # positional distance train_end -> test_start is exactly gap + 1
        assert _pos(w.test_start) - _pos(w.train_end) == gap + 1
        # train never touches [test_start - gap, ...)
        assert _pos(w.train_end) < _pos(w.test_start) - gap
        assert w.train_start <= w.train_end < w.test_start <= w.test_end


def test_test_blocks_contiguous_and_cover_to_the_end():
    windows = WalkForwardSplitter(_cfg()).split(DATES)
    covered = []
    for w in windows:
        covered.extend(range(_pos(w.test_start), _pos(w.test_end) + 1))
    # contiguous, non-overlapping, from p0 through the last date
    assert covered == list(range(12, 30))


def test_truncated_final_window_kept():
    windows = WalkForwardSplitter(_cfg()).split(DATES)
    last = windows[-1]
    assert _pos(last.test_start) == 27
    assert _pos(last.test_end) == 29  # only 3 of 5 test days exist


def test_single_day_final_window_kept():
    # p0 = 12, test_days = 17 -> second block starts at 29, one day long
    windows = WalkForwardSplitter(_cfg(test_days=17)).split(DATES)
    assert len(windows) == 2
    assert _pos(windows[-1].test_start) == _pos(windows[-1].test_end) == 29


def test_config_error_when_dates_too_short():
    # needs train_days + gap + 1 = 13 dates
    short = pd.bdate_range("2020-01-01", periods=12)
    with pytest.raises(ConfigError):
        WalkForwardSplitter(_cfg()).split(short)
    # 13 dates is exactly enough: one single-day test window
    just_enough = pd.bdate_range("2020-01-01", periods=13)
    windows = WalkForwardSplitter(_cfg()).split(just_enough)
    assert len(windows) == 1
    assert windows[0].test_start == windows[0].test_end == just_enough[12]


def test_embargo_adds_to_the_gap():
    windows = WalkForwardSplitter(_cfg(embargo_days=1)).split(DATES)
    # gap = 3, p0 = 13; blocks at p = 13, 18, 23, 28
    assert [_positions(w) for w in windows] == [
        (0, 9, 13, 17),
        (5, 14, 18, 22),
        (10, 19, 23, 27),
        (15, 24, 28, 29),
    ]
    for w in windows:
        assert _pos(w.test_start) - _pos(w.train_end) == 3 + 1


def test_timestamps_come_from_the_supplied_index():
    windows = WalkForwardSplitter(_cfg()).split(DATES)
    for w in windows:
        for ts in (w.train_start, w.train_end, w.test_start, w.test_end):
            assert ts in DATES
