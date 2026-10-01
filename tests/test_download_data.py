"""Offline tests for the downloader and the cache loader (no network)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import download_data
import qcore.data
from download_data import split_adjusted_panels

CACHE_FILES = ["adj_close", "close", "open", "high", "low", "volume", "indices", "coverage"]


def _response(n=6, tickers=("AAA", "BBB")):
    """A yfinance-shaped auto_adjust=False frame: (field, ticker) columns."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(0)
    fields = {}
    for t in tickers:
        close = 100 + rng.standard_normal(n).cumsum()
        factor = np.linspace(0.97, 1.0, n)  # dividend adjustment, 1 on the last bar
        fields[("Close", t)] = close
        fields[("Adj Close", t)] = close * factor
        fields[("Open", t)] = close * 0.99
        fields[("High", t)] = close * 1.01
        fields[("Low", t)] = close * 0.98
        fields[("Volume", t)] = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(fields, index=dates)


def _index_response(n=6):
    """A yfinance-shaped auto_adjust=True index frame; Open differs from Close."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(1)
    fields = {}
    for t in download_data.INDEX_UNIVERSE:
        level = 10 + rng.random(n)
        fields[("Close", t)] = level
        fields[("Open", t)] = level + 1.0
    return pd.DataFrame(fields, index=dates)


def _fake_vendor(equity, index, calls=None):
    def fetch(tickers, **kwargs):
        if calls is not None:
            calls.append((list(tickers), kwargs))
        return index if list(tickers) == list(download_data.INDEX_UNIVERSE) else equity
    return fetch


def _refresh(monkeypatch, tmp_path, equity, index=None, **kwargs):
    """Run main() against a fake vendor, with tmp_path as the cache."""
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(qcore.data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "ETF_UNIVERSE", ["AAA"])
    monkeypatch.setattr(download_data, "STOCK_UNIVERSE", ["BBB"])
    monkeypatch.setattr(download_data, "OPERATIONAL_UNIVERSE", [])
    index = _index_response(len(equity)) if index is None else index
    monkeypatch.setattr(download_data.yf, "download", _fake_vendor(equity, index))
    download_data.main(**kwargs)


def _snapshot(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def _read(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _assert_same_panel(written, expected):
    assert list(written.columns) == list(expected.columns)
    assert list(written.index) == list(expected.index)
    np.testing.assert_allclose(written.to_numpy(dtype=float), expected.to_numpy(dtype=float),
                               rtol=1e-12, equal_nan=True)


def test_panels_follow_yfinance_auto_adjust_arithmetic():
    raw = _response()
    out = split_adjusted_panels(raw)
    ratio = raw["Adj Close"] / raw["Close"]
    pd.testing.assert_frame_equal(out["close"], raw["Adj Close"])
    pd.testing.assert_frame_equal(out["raw_close"], raw["Close"])
    for field in ("Open", "High", "Low"):
        pd.testing.assert_frame_equal(out[field.lower()], raw[field] * ratio)
    pd.testing.assert_frame_equal(out["volume"], raw["Volume"])


def test_download_batch_makes_one_unadjusted_request(monkeypatch):
    # raw and adjusted closes must come from ONE response: two separate
    # downloads during a session price the in-progress bar seconds apart,
    # so the latest adj/raw factor is no longer 1 and the DQ anchor fails
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append(kwargs)
        return _response()

    monkeypatch.setattr(download_data.yf, "download", fake_download)
    out = download_data.download_batch(["AAA", "BBB"])
    assert len(calls) == 1 and calls[0]["auto_adjust"] is False
    assert (out["close"].iloc[-1] / out["raw_close"].iloc[-1] == 1.0).all()


def test_failed_ticker_download_is_rejected(monkeypatch):
    raw = _response()
    raw.loc[:, pd.IndexSlice[:, "BBB"]] = np.nan
    monkeypatch.setattr(download_data.yf, "download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="no observations"):
        download_data.download_batch(["AAA", "BBB"])


def test_index_failure_preserves_every_existing_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "ETF_UNIVERSE", ["AAA"])
    monkeypatch.setattr(download_data, "OPERATIONAL_UNIVERSE", [])
    monkeypatch.setattr(download_data, "STOCK_UNIVERSE", ["BBB"])
    names = ["adj_close", "close", "open", "high", "low", "volume", "indices", "coverage"]
    for name in names:
        (tmp_path / f"{name}.csv").write_text(f"original {name}")
    calls = 0

    def fetch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("index vendor outage")
        return _response()

    monkeypatch.setattr(download_data.yf, "download", fetch)
    with pytest.raises(RuntimeError, match="outage"):
        download_data.main()
    for name in names:
        assert (tmp_path / f"{name}.csv").read_text() == f"original {name}"


def test_cache_replacement_rolls_back_partial_write(monkeypatch, tmp_path):
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    (tmp_path / "a.csv").write_text("original A")
    (tmp_path / "b.csv").write_text("original B")
    actual_replace = download_data.os.replace
    calls = 0

    def replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk error")
        return actual_replace(source, target)

    monkeypatch.setattr(download_data.os, "replace", replace)
    with pytest.raises(OSError, match="disk error"):
        download_data._write_cache({"a": _response(), "b": _response()})
    assert (tmp_path / "a.csv").read_text() == "original A"
    assert (tmp_path / "b.csv").read_text() == "original B"
    assert not list(tmp_path.glob(".refresh-*"))  # a clean rollback leaves no staging


# ------------------------------------------------------------ cache replacement
def _failing_replace(monkeypatch, fail_on, error=OSError, after_rename=()):
    """Make os.replace raise on the given call numbers (1-based). Calls listed
    in after_rename perform the rename first and raise afterwards."""
    actual_replace = download_data.os.replace
    calls = 0

    def replace(source, target):
        nonlocal calls
        calls += 1
        if calls in after_rename:
            actual_replace(source, target)
        if calls in fail_on:
            raise error(f"disk error {calls}")
        return actual_replace(source, target)

    monkeypatch.setattr(download_data.os, "replace", replace)


def test_cache_replacement_removes_new_files_on_rollback(monkeypatch, tmp_path):
    # "a" did not exist before the refresh: a failed refresh must not leave a
    # fresh a.csv beside the restored old b.csv (a mixed-vintage cache)
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    (tmp_path / "b.csv").write_text("original B")
    _failing_replace(monkeypatch, fail_on={2})
    with pytest.raises(OSError, match="disk error 2"):
        download_data._write_cache({"a": _response(), "b": _response()})
    assert [p.name for p in tmp_path.iterdir()] == ["b.csv"]  # no a.csv, no staging dir
    assert (tmp_path / "b.csv").read_text() == "original B"


def test_cache_replacement_writes_every_file_and_leaves_no_staging(monkeypatch, tmp_path):
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    (tmp_path / "a.csv").write_text("original A")
    frame = _response()
    download_data._write_cache({"a": frame, "b": frame})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.csv", "b.csv"]
    for name in ("a", "b"):
        assert (tmp_path / f"{name}.csv").read_bytes() == frame.to_csv().encode()


def test_interrupt_during_replacement_restores_the_cache(monkeypatch, tmp_path):
    # Ctrl-C is not an Exception: the rollback must still run, then re-raise it
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    (tmp_path / "a.csv").write_text("original A")
    (tmp_path / "b.csv").write_text("original B")
    _failing_replace(monkeypatch, fail_on={2}, error=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        download_data._write_cache({"a": _response(), "b": _response()})
    assert _snapshot(tmp_path) == {"a.csv": b"original A", "b.csv": b"original B"}


def test_failure_right_after_a_rename_still_restores_that_file(monkeypatch, tmp_path):
    # the interruption lands after b.csv was replaced but before the loop
    # moved on: b must be restored as well, not left as the new generation
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    (tmp_path / "a.csv").write_text("original A")
    (tmp_path / "b.csv").write_text("original B")
    _failing_replace(monkeypatch, fail_on={2}, after_rename={2})
    with pytest.raises(OSError, match="disk error 2"):
        download_data._write_cache({"a": _response(), "b": _response()})
    assert _snapshot(tmp_path) == {"a.csv": b"original A", "b.csv": b"original B"}


def test_failed_rollback_keeps_the_backups_and_restores_the_rest(monkeypatch, tmp_path):
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.csv").write_text(f"original {name}")
    # call 3 promotes c and fails; call 4 restores a and fails as well
    _failing_replace(monkeypatch, fail_on={3, 4})
    frames = {name: _response() for name in ("a", "b", "c")}
    with pytest.raises(RuntimeError, match=r"could not restore \['a'\]") as failure:
        download_data._write_cache(frames)
    assert isinstance(failure.value.__cause__, OSError)
    assert "disk error 3" in str(failure.value.__cause__)
    # one failed restore must not strand the files after it
    assert (tmp_path / "b.csv").read_text() == "original b"
    assert (tmp_path / "c.csv").read_text() == "original c"
    assert (tmp_path / "a.csv").read_text() != "original a"  # the new generation
    # ... and the only copy of the original a.csv is still on disk, and named
    stages = list(tmp_path.glob(".refresh-*"))
    assert len(stages) == 1 and str(stages[0]) in str(failure.value)
    assert (stages[0] / "a.backup").read_text() == "original a"


def test_leftover_staging_folder_is_reported_and_not_deleted(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    stale = tmp_path / ".refresh-interrupted"
    stale.mkdir()
    (stale / "a.backup").write_text("original A")
    download_data._write_cache({"a": _response()})
    assert "interrupted refresh" in capsys.readouterr().out
    assert (stale / "a.backup").read_text() == "original A"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".refresh-interrupted", "a.csv"]


# ------------------------------------------------------------- full refresh
def test_successful_refresh_writes_the_documented_cache(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(qcore.data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "ETF_UNIVERSE", ["AAA"])
    monkeypatch.setattr(download_data, "STOCK_UNIVERSE", ["BBB"])
    # the operational universe is deliberately NOT patched: the cash-parking
    # quote must be downloaded yet stay out of the research price panel
    assert download_data.OPERATIONAL_UNIVERSE == ["SGOV"]
    tickers = ["AAA", "BBB", "SGOV"]
    raw = _response(n=8, tickers=tickers)
    dates = raw.index
    prices = ["Adj Close", "Close", "Open", "High", "Low"]
    raw.loc[dates[:2], pd.IndexSlice[:, "BBB"]] = np.nan      # late starter
    raw.loc[dates[3], pd.IndexSlice["Open", :]] = np.nan      # one field misses a whole row
    raw.loc[dates[-1], pd.IndexSlice[prices, :]] = np.nan     # volume-only row: no close
    raw.loc[dates[0], pd.IndexSlice["Volume", "AAA"]] = np.nan  # coverage is about closes
    index = _index_response(n=8)
    index.loc[dates[5]] = np.nan                              # no index printed that day
    calls = []
    monkeypatch.setattr(download_data.yf, "download", _fake_vendor(raw, index, calls))

    download_data.main()

    assert [c[0] for c in calls] == [tickers, download_data.INDEX_UNIVERSE]
    assert calls[0][1]["auto_adjust"] is False and calls[1][1]["auto_adjust"] is True
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{n}.csv" for n in CACHE_FILES)
    assert str(tmp_path) in capsys.readouterr().out  # the operator sees which cache was written

    kept = dates[:-1]  # the calendar of the adjusted close
    ratio = raw["Adj Close"] / raw["Close"]
    expected = {
        "adj_close": raw["Adj Close"], "close": raw["Close"],
        "open": raw["Open"] * ratio, "high": raw["High"] * ratio, "low": raw["Low"] * ratio,
        "volume": raw["Volume"],
    }
    for name, frame in expected.items():
        _assert_same_panel(_read(tmp_path / f"{name}.csv"), frame.reindex(kept))
    assert _read(tmp_path / "open.csv").loc[dates[3]].isna().all()
    _assert_same_panel(_read(tmp_path / "indices.csv"), index["Close"].drop(dates[5]))

    coverage = pd.read_csv(tmp_path / "coverage.csv", index_col=0, parse_dates=["first", "last"])
    assert list(coverage.columns) == ["first", "last", "rows"]
    assert list(coverage.index) == tickers
    assert list(coverage["first"]) == [dates[0], dates[2], dates[0]]
    assert list(coverage["last"]) == [dates[-2]] * 3
    assert list(coverage["rows"]) == [7, 5, 7]

    # reader side: every panel reloads on the downloaded calendar, research
    # prices exclude the operational quote, and the adjusted/raw pair yields
    # non-negative, somewhere-positive dividends
    for name in expected:
        assert qcore.data.load(name).index.equals(pd.DatetimeIndex(kept)), name
    assert list(qcore.data.load_prices().columns) == ["AAA", "BBB"]
    assert list(qcore.data.load("adj_close").columns) == tickers
    assert list(qcore.data.load_indices().columns) == download_data.INDEX_UNIVERSE
    assert (qcore.data.dividend_yields() > 0).any().all()


@pytest.mark.parametrize("breakage, message", [
    ("missing", "indices: incomplete or empty ticker response"),
    ("unobserved", "indices: at least one ticker has no observations"),
    ("infinite", "indices: invalid numeric values"),
    ("duplicate date", "indices: invalid/duplicate dates"),
])
def test_invalid_index_panel_aborts_before_any_file_is_replaced(monkeypatch, tmp_path, breakage, message):
    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "ETF_UNIVERSE", ["AAA"])
    monkeypatch.setattr(download_data, "OPERATIONAL_UNIVERSE", [])
    monkeypatch.setattr(download_data, "STOCK_UNIVERSE", ["BBB"])
    for name in CACHE_FILES:
        (tmp_path / f"{name}.csv").write_text(f"original {name}")
    index = _index_response()
    victim = ("Close", download_data.INDEX_UNIVERSE[1])
    if breakage == "missing":
        index = index.drop(columns=[victim])
    elif breakage == "unobserved":
        index[victim] = np.nan
    elif breakage == "infinite":
        index.iloc[2, index.columns.get_loc(victim)] = np.inf
    else:
        index = pd.concat([index, index.iloc[[-1]]])
    monkeypatch.setattr(download_data.yf, "download", _fake_vendor(_response(), index))
    with pytest.raises(ValueError, match=message):
        download_data.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{n}.csv" for n in CACHE_FILES)
    for name in CACHE_FILES:
        assert (tmp_path / f"{name}.csv").read_text() == f"original {name}"


def test_stale_index_panel_aborts_refresh_and_keeps_the_cache(monkeypatch, tmp_path):
    allowed = download_data.INDEX_MAX_LAG_SESSIONS
    equity = _response(n=allowed + 7)
    (tmp_path / "adj_close.csv").write_text("original adj_close")
    with pytest.raises(ValueError, match=rf"indices: panel ends 2024-01-09, {allowed + 1} sessions"):
        _refresh(monkeypatch, tmp_path, equity, index=_index_response(n=6))
    assert _snapshot(tmp_path) == {"adj_close.csv": b"original adj_close"}
    # routine lag (a day or two of late index prints) is not a failure
    _refresh(monkeypatch, tmp_path, equity, index=_index_response(n=7))
    assert len(_read(tmp_path / "indices.csv")) == 7
    assert len(_read(tmp_path / "adj_close.csv")) == allowed + 7


# ------------------------------------------------------ download validation
def _panel():
    return _response()["Close"]


def _with_index(frame, index):
    return frame.set_axis(index, axis=0)


def _poke(frame, value):
    out = frame.copy()
    out.iloc[2, 0] = value
    return out


VALIDATION_CASES = {
    "empty": (lambda f: f.iloc[:0], "close", True, "incomplete or empty"),
    "missing ticker": (lambda f: f[["AAA"]], "close", True, "incomplete or empty"),
    "extra ticker": (lambda f: f.assign(CCC=1.0), "close", True, "incomplete or empty"),
    "duplicate ticker": (lambda f: pd.concat([f, f[["BBB"]]], axis=1), "close", True,
                         "incomplete or empty"),
    "not dates": (lambda f: f.reset_index(drop=True), "close", True, "invalid/duplicate dates"),
    "missing date": (lambda f: _with_index(f, [pd.NaT, *f.index[1:]]), "close", True,
                     "invalid/duplicate dates"),
    "duplicate date": (lambda f: _with_index(f, [f.index[0], *f.index[:-1]]), "close", True,
                       "invalid/duplicate dates"),
    "unsorted dates": (lambda f: f.iloc[::-1], "close", True, "unsorted dates"),
    "exchange timezone": (lambda f: _with_index(f, f.index.tz_localize("America/New_York")),
                          "close", True, "timezone-naive calendar days"),
    "utc timezone": (lambda f: _with_index(f, f.index.tz_localize("UTC")), "indices", False,
                     "timezone-naive calendar days"),
    "intraday stamps": (lambda f: _with_index(f, f.index + pd.Timedelta(16, unit="h")),
                        "close", True, "timezone-naive calendar days"),
    "infinite price": (lambda f: _poke(f, np.inf), "close", True, "invalid numeric values"),
    "negative infinite index": (lambda f: _poke(f, -np.inf), "indices", False,
                                "invalid numeric values"),
    "infinite volume": (lambda f: _poke(f, np.inf), "volume", False, "invalid numeric values"),
    "zero price": (lambda f: _poke(f, 0.0), "close", True, "invalid numeric values"),
    "negative price": (lambda f: _poke(f, -1.0), "low", True, "invalid numeric values"),
    "negative volume": (lambda f: _poke(f, -1.0), "volume", False, "volume: negative values"),
    "unobserved ticker": (lambda f: f.assign(BBB=np.nan), "close", True, "no observations"),
}


@pytest.mark.parametrize("case", sorted(VALIDATION_CASES))
def test_validate_download_rejects_malformed_panels(case):
    breaker, label, positive, message = VALIDATION_CASES[case]
    good = _panel()
    download_data._validate_download(good, ["AAA", "BBB"], label, positive=positive)
    panel = breaker(good)  # built outside the raises block: a helper error must not pass
    with pytest.raises(ValueError, match=message):
        download_data._validate_download(panel, ["AAA", "BBB"], label, positive=positive)


def test_validate_download_accepts_gaps_zero_volume_and_nonpositive_index_levels():
    panel = _panel()
    panel.iloc[0, 1] = np.nan  # a late starter is not a failed download
    download_data._validate_download(panel, ["AAA", "BBB"], "close", positive=True)
    volume = _response()["Volume"]
    volume.iloc[2, 0] = 0.0   # a halted session reports zero volume
    download_data._validate_download(volume, ["AAA", "BBB"], "volume")
    levels = _panel()
    levels.iloc[2, 0] = 0.0    # a bill yield can be zero or negative
    levels.iloc[3, 0] = -0.05
    download_data._validate_download(levels, ["AAA", "BBB"], "indices")


@pytest.mark.parametrize("field", ["Adj Close", "Close", "Open", "High", "Low"])
def test_download_batch_rejects_nonpositive_prices(monkeypatch, field):
    raw = _response()
    raw.iloc[2, raw.columns.get_loc((field, "AAA"))] = -1.0
    monkeypatch.setattr(download_data.yf, "download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="invalid numeric values"):
        download_data.download_batch(["AAA", "BBB"])


def test_download_batch_accepts_zero_volume_and_rejects_negative_volume(monkeypatch):
    raw = _response()
    raw.iloc[2, raw.columns.get_loc(("Volume", "AAA"))] = 0.0
    monkeypatch.setattr(download_data.yf, "download", lambda *args, **kwargs: raw)
    assert download_data.download_batch(["AAA", "BBB"])["volume"].iloc[2, 0] == 0.0
    raw.iloc[2, raw.columns.get_loc(("Volume", "AAA"))] = -1.0
    with pytest.raises(ValueError, match="volume: negative values"):
        download_data.download_batch(["AAA", "BBB"])


@pytest.mark.parametrize("shift", ["America/New_York", "UTC", "16h"])
def test_timezone_aware_or_intraday_vendor_dates_never_reach_the_cache(monkeypatch, tmp_path, shift):
    raw = _response()
    raw.index = (raw.index + pd.Timedelta(16, unit="h") if shift == "16h"
                 else raw.index.tz_localize(shift))
    with pytest.raises(ValueError, match="timezone-naive calendar days"):
        _refresh(monkeypatch, tmp_path, raw)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------- history lost by a refresh
def _lose(raw, rows, ticker, fields=slice(None)):
    out = raw.copy()
    out.loc[out.index[rows], pd.IndexSlice[fields, ticker]] = np.nan
    return out


# name: (equity response, index response or None for a complete one, refusal)
LOSSES = {
    "later start": (lambda: (_lose(_response(), slice(0, 4), "AAA"), None),
                    r"adj_close\.csv AAA: 4 of 6 cached observations missing "
                    r"\(2024-01-02 \.\. 2024-01-05\)"),
    "earlier end": (lambda: (_lose(_response(), slice(4, 6), "BBB"), None),
                    r"adj_close\.csv BBB: 2 of 6 cached observations missing "
                    r"\(2024-01-08 \.\. 2024-01-09\)"),
    # more rows than before, yet one cached bar has gone
    "hole while growing": (lambda: (_lose(_response(8), [3], "BBB"), None),
                           r"adj_close\.csv BBB: 1 of 6 cached observations missing "
                           r"\(2024-01-05 \.\. 2024-01-05\)"),
    "whole response starts later": (lambda: (_response(8).iloc[3:], _index_response(8).iloc[3:]),
                                    r"adj_close\.csv AAA: 3 of 6 cached observations missing"),
    "one field only": (lambda: (_lose(_response(), [2], "AAA", fields="Volume"), None),
                       r"volume\.csv AAA: 1 of 6 cached observations missing"),
    "index series": (lambda: (_response(), _lose(_index_response(), slice(0, 3), "^IRX")),
                     r"indices\.csv \^IRX: 3 of 6 cached observations missing "
                     r"\(2024-01-02 \.\. 2024-01-04\)"),
}


@pytest.mark.parametrize("case", sorted(LOSSES))
def test_refresh_that_loses_cached_history_is_refused_and_cache_untouched(monkeypatch, tmp_path, case):
    build, message = LOSSES[case]
    _refresh(monkeypatch, tmp_path, _response())
    before = _snapshot(tmp_path)
    assert sorted(before) == sorted(f"{n}.csv" for n in CACHE_FILES)
    equity, index = build()
    with pytest.raises(ValueError, match=message) as refusal:
        _refresh(monkeypatch, tmp_path, equity, index=index)
    assert "refresh refused" in str(refusal.value) and "--allow-shrink" in str(refusal.value)
    assert _snapshot(tmp_path) == before


def test_allow_shrink_accepts_the_shorter_history(monkeypatch, tmp_path, capsys):
    _refresh(monkeypatch, tmp_path, _response())
    _refresh(monkeypatch, tmp_path, _lose(_response(), slice(0, 4), "AAA"), allow_shrink=True)
    coverage = pd.read_csv(tmp_path / "coverage.csv", index_col=0)
    assert list(coverage["rows"]) == [2, 6]
    out = capsys.readouterr().out
    assert "WARNING: --allow-shrink" in out and "adj_close.csv AAA: 4 of 6" in out


def test_refresh_that_extends_and_revises_history_is_accepted(monkeypatch, tmp_path):
    _refresh(monkeypatch, tmp_path, _response())
    old = _read(tmp_path / "adj_close.csv")
    longer = _response(8)
    longer.loc[:, pd.IndexSlice[["Adj Close", "Close", "Open", "High", "Low"], :]] *= 1.01
    _refresh(monkeypatch, tmp_path, longer)  # same dates, revised values, two new bars
    new = _read(tmp_path / "adj_close.csv")
    assert len(new) == 8 and not np.allclose(new.loc[old.index], old)


def test_ticker_removed_from_the_universe_is_not_lost_history(monkeypatch, tmp_path):
    _refresh(monkeypatch, tmp_path, _response())
    monkeypatch.setattr(download_data, "STOCK_UNIVERSE", [])
    monkeypatch.setattr(download_data.yf, "download",
                        _fake_vendor(_response(tickers=("AAA",)), _index_response()))
    download_data.main()
    assert list(_read(tmp_path / "adj_close.csv").columns) == ["AAA"]


@pytest.mark.parametrize("damage", ["unparseable date", "empty file"])
def test_existing_cache_that_cannot_be_compared_blocks_the_refresh(monkeypatch, tmp_path, damage):
    _refresh(monkeypatch, tmp_path, _response())
    victim = tmp_path / "close.csv"
    victim.write_text("" if damage == "empty file"
                      else victim.read_text().replace("2024-01-04", "not-a-date"))
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match=r"close\.csv: existing file cannot be compared"):
        _refresh(monkeypatch, tmp_path, _response())
    assert _snapshot(tmp_path) == before
    _refresh(monkeypatch, tmp_path, _response(), allow_shrink=True)  # the repair path
    assert len(_read(victim)) == 6


# ------------------------------------------------------------- cache loader
def _cache_csv(index):
    return pd.DataFrame({"AAA": np.arange(1.0, len(index) + 1)}, index=index).to_csv()


_DATES = pd.bdate_range("2024-10-28", periods=8)  # spans a daylight-saving change
DAMAGED = {
    "duplicate date": (_cache_csv(_DATES[[0, 1, 1, 2]]), "dates must be unique and increasing"),
    "unsorted dates": (_cache_csv(_DATES[::-1]), "dates must be unique and increasing"),
    "unparseable date": (_cache_csv(_DATES).replace("2024-10-30", "not-a-date"),
                         "timezone-naive calendar dates"),
    "blank date": (_cache_csv(_DATES).replace("2024-10-30,", ",", 1),
                   "timezone-naive calendar dates"),
    "exchange timezone": (_cache_csv(_DATES.tz_localize("America/New_York")),
                          "timezone-naive calendar dates"),
    "utc timezone": (_cache_csv(_DATES.tz_localize("UTC")), "timezone-naive calendar dates"),
    "intraday stamps": (_cache_csv(_DATES + pd.Timedelta(16, unit="h")),
                        "timezone-naive calendar dates"),
    "header only": ("Date,AAA\n", "no rows"),
}


@pytest.mark.parametrize("case", sorted(DAMAGED))
def test_loader_rejects_a_damaged_date_column(monkeypatch, tmp_path, case):
    text, message = DAMAGED[case]
    monkeypatch.setattr(qcore.data, "DATA_DIR", tmp_path)
    (tmp_path / "adj_close.csv").write_text(_cache_csv(_DATES))
    assert qcore.data.load_prices().index.equals(_DATES)  # the undamaged file loads
    (tmp_path / "adj_close.csv").write_text(text)
    with pytest.raises(ValueError, match=rf"adj_close\.csv: .*{message}"):
        qcore.data.load_prices()


def test_lenient_loader_returns_duplicate_unsorted_and_empty_files(monkeypatch, tmp_path):
    # a data-quality check wants to report these rows itself
    monkeypatch.setattr(qcore.data, "DATA_DIR", tmp_path)
    for case in ("duplicate date", "unsorted dates"):
        (tmp_path / "close.csv").write_text(DAMAGED[case][0])
        frame = qcore.data.load("close", strict=False)
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert not (frame.index.is_unique and frame.index.is_monotonic_increasing)
    (tmp_path / "close.csv").write_text(DAMAGED["header only"][0])
    empty = qcore.data.load("close", strict=False)
    assert isinstance(empty.index, pd.DatetimeIndex) and empty.empty
    assert list(empty.columns) == ["AAA"]
    # dates that are not dates are never returned, strict or not
    (tmp_path / "close.csv").write_text(DAMAGED["unparseable date"][0])
    with pytest.raises(ValueError, match="timezone-naive calendar dates"):
        qcore.data.load("close", strict=False)


def test_missing_cache_file_names_the_file_and_the_downloader(monkeypatch, tmp_path):
    monkeypatch.setattr(qcore.data, "DATA_DIR", tmp_path)
    missing = re.escape(str(tmp_path / "adj_close.csv"))
    # FileNotFoundError specifically: the engine's fallbacks catch that type
    with pytest.raises(FileNotFoundError, match=missing + r".*python src/download_data\.py"):
        qcore.data.load_prices()
    with pytest.raises(FileNotFoundError, match="indices.csv"):
        qcore.data.load_indices()
    with pytest.raises(FileNotFoundError, match="adj_close.csv"):
        qcore.data.dividend_yields()


def test_data_directory_override_is_opt_in(tmp_path):
    default = Path(qcore.data.__file__).resolve().parents[2] / "data"
    assert qcore.data.DEFAULT_DATA_DIR == default
    assert qcore.data._data_dir({}) == default
    assert qcore.data._data_dir({"QCORE_DATA_DIR": ""}) == default
    override = qcore.data._data_dir({"QCORE_DATA_DIR": str(tmp_path / "live")})
    assert override == (tmp_path / "live").resolve()
    relative = qcore.data._data_dir({"QCORE_DATA_DIR": "live_cache"})
    assert relative.is_absolute() and relative.name == "live_cache"
    # the downloader writes where the loader reads
    assert download_data.DATA_DIR == qcore.data.DATA_DIR == qcore.data._data_dir(os.environ)


@pytest.mark.parametrize("value", [None, "", "live cache"])
def test_environment_override_sets_the_cache_directory(tmp_path, value):
    src = Path(qcore.data.__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "QCORE_DATA_DIR"}
    env.update(PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1")
    if value is not None:
        env["QCORE_DATA_DIR"] = str(tmp_path / value) if value else ""
    done = subprocess.run([sys.executable, "-c", "import qcore.data; print(qcore.data.DATA_DIR)"],
                          env=env, capture_output=True, text=True, check=True)
    expected = (tmp_path / value).resolve() if value else src.parent / "data"
    assert done.stdout.strip() == str(expected)


def test_dividend_noise_floor_decides_which_differences_count_as_payouts(monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    raw = pd.DataFrame(100.0 * np.exp(rng.normal(0.0, 0.01, (400, 2)).cumsum(axis=0)),
                       index=dates, columns=["AAA", "BBB"])
    ex_rows, paid = [60, 120, 180, 240, 300, 360], 0.004
    factor = pd.DataFrame(1.0, index=dates, columns=raw.columns)
    for row in ex_rows:  # back-adjust every close before each ex-date
        factor.iloc[:row, 0] *= 1.0 - paid
    # vendor rounding: symmetric, up to about 2.2e-6 in the return difference
    adjusted = raw * factor * (1.0 + rng.uniform(-1.1e-6, 1.1e-6, size=raw.shape))
    monkeypatch.setattr(qcore.data, "load", lambda name: adjusted if name == "adj_close" else raw)
    expected = pd.DataFrame(False, index=dates, columns=raw.columns)
    expected.iloc[ex_rows, 0] = True

    # the shipped floor keeps every payout, and also the positive half of the
    # noise above it; changing the value moves saved results, so it is pinned
    assert qcore.data.DIVIDEND_NOISE_FLOOR == 1e-6
    kept = qcore.data.dividend_yields()
    assert (kept >= 0).all().all() and (kept[expected] > 0).sum().sum() == len(ex_rows)
    spurious = (kept > 0) & ~expected
    assert spurious.sum().sum() > 10 and kept[spurious].max().max() < 2.5e-6

    monkeypatch.setattr(qcore.data, "DIVIDEND_NOISE_FLOOR", 5e-6)
    clean = qcore.data.dividend_yields()
    pd.testing.assert_frame_equal(clean > 0, expected)
    assert clean.iloc[ex_rows, 0].to_numpy() == pytest.approx(paid, rel=0.02)
