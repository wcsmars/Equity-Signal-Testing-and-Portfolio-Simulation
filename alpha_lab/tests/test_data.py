"""Tests for the data layer: sources, validation, snapshot store."""

import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config.schema import DataConfig, SyntheticConfig
from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.data.sources import SOURCES, CSVSource, SyntheticSource, source_from_config
from alpha_lab.data.store import MarketDataStore
from alpha_lab.data.synthetic import make_market
from alpha_lab.data.validation import validate_market
from alpha_lab.core.types import MarketData


def assert_panel_close(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Same axes (names ignored) and values equal to float tolerance."""
    assert actual.index.equals(expected.index)
    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=1e-12, equal_nan=True
    )


def write_wide_dir(directory, data: MarketData, universe_as_int: bool = False):
    """Dump a MarketData as a wide-format CSV directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("close", "open", "high", "low", "volume", "unadjusted_close"):
        frame = getattr(data, name)
        if frame is not None:
            frame.to_csv(directory / f"{name}.csv", index_label="date")
    if data.universe is not None:
        universe = data.universe.astype(int) if universe_as_int else data.universe
        universe.to_csv(directory / "universe.csv", index_label="date")
    return directory


# -- sources --------------------------------------------------------------


def test_registry_names():
    assert "synthetic" in SOURCES
    assert "csv" in SOURCES


@pytest.mark.parametrize("field", ["close", "open", "high", "low", "volume", "unadjusted_close"])
@pytest.mark.parametrize("bad", [np.inf, -np.inf, "invalid"])
def test_market_rejects_non_numeric_or_infinite_data(field, bad):
    dates = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    panel = pd.DataFrame({"A": [10.0, bad, 12.0]}, index=dates)
    with pytest.raises(DataError, match="numeric|infinite"):
        MarketData(**({"close": panel} if field == "close" else {"close": close, field: panel}))


@pytest.mark.parametrize("field", ["open", "high", "low", "unadjusted_close"])
def test_market_rejects_nonpositive_optional_prices(field):
    dates = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    with pytest.raises(DataError, match="non-positive"):
        MarketData(close=close, **{field: close * 0.0})


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_market_rejects_nonpositive_close(bad):
    dates = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(DataError, match="non-positive close"):
        MarketData(pd.DataFrame({"A": [10.0, bad, 12.0]}, index=dates))
    # volume is the one numeric field where zero is a legal value
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    assert MarketData(close, volume=close * 0.0).volume.eq(0.0).all().all()


def _panel(index, columns=("A", "B")):
    return pd.DataFrame(
        np.arange(1.0, 1.0 + len(index) * len(columns)).reshape(len(index), len(columns)),
        index=index,
        columns=list(columns),
    )


def test_market_rejects_malformed_date_index():
    dates = pd.bdate_range("2024-01-01", periods=4)
    assert MarketData(_panel(dates)).dates.equals(dates)
    with pytest.raises(DataError, match="must be a DatetimeIndex"):
        MarketData(_panel(pd.RangeIndex(4)))
    with pytest.raises(DataError, match="must be a DatetimeIndex"):
        MarketData(_panel(dates.strftime("%Y-%m-%d")))
    with pytest.raises(DataError, match="tz-naive"):
        MarketData(_panel(dates.tz_localize("UTC")))
    with pytest.raises(DataError, match="missing dates"):
        MarketData(_panel(pd.DatetimeIndex([dates[0], pd.NaT, dates[2], dates[3]])))
    with pytest.raises(DataError, match="ascending"):
        MarketData(_panel(dates[::-1]))
    with pytest.raises(DataError, match="ascending"):
        MarketData(_panel(dates[[0, 2, 1, 3]]))
    with pytest.raises(DataError, match="duplicate dates"):
        MarketData(_panel(dates[[0, 1, 1, 3]]))
    with pytest.raises(DataError, match="must be a DataFrame"):
        MarketData(_panel(dates)["A"])


def test_market_rejects_duplicate_tickers():
    dates = pd.bdate_range("2024-01-01", periods=4)
    with pytest.raises(DataError, match="duplicate tickers"):
        MarketData(_panel(dates, columns=("A", "A")))


@pytest.mark.parametrize("field", ["open", "high", "low", "volume", "unadjusted_close", "universe"])
def test_market_constructor_demands_exact_alignment(field):
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = _panel(dates)
    full = close.notna() if field == "universe" else close.copy()
    assert getattr(MarketData(close, **{field: full}), field).shape == close.shape
    for misaligned in (full.iloc[:-1], full[["A"]], full[["B", "A"]], full.iloc[::-1]):
        with pytest.raises(DataError, match=f"{field} is not aligned with close"):
            MarketData(close, **{field: misaligned})


def test_market_universe_must_be_boolean():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = _panel(dates)
    with pytest.raises(DataError, match="universe must be boolean"):
        MarketData(close, universe=close.notna().astype(int))


def test_from_frames_fills_a_partial_universe_with_false_without_warnings():
    # the documented use: a universe covering fewer dates and tickers than
    # close. Filling the reindexed (object) frame used to raise a pandas
    # FutureWarning on every such load.
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = _panel(dates, columns=("A", "B", "C"))
    partial = pd.DataFrame(True, index=dates[2:], columns=["A", "B"])
    partial.iloc[0, 1] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        data = MarketData.from_frames(close, universe=partial)
    assert all(dt == bool for dt in data.universe.dtypes)
    assert data.universe.index.equals(dates) and list(data.universe.columns) == ["A", "B", "C"]
    expected = np.zeros((6, 3), dtype=bool)
    expected[2:, 0] = True
    expected[3:, 1] = True
    assert (data.universe.to_numpy() == expected).all()


@pytest.mark.parametrize("kind", ["bool", "object"])
def test_from_frames_universe_keeps_canonical_boolean_bytes(kind):
    # the panel bytes feed content hashes (feature cache identity). A
    # vectorised `== True` returns 0xFE for True on NumPy 1.24.0, so the
    # alignment must not build the panel from such a comparison.
    dates = pd.bdate_range("2024-01-01", periods=200)
    close = _panel(dates)
    member = pd.DataFrame({"A": [True, False, True, True] * 50, "B": [False, True] * 100}, index=dates)
    universe = member.astype(object) if kind == "object" else member
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        full = MarketData.from_frames(close, universe=universe).universe
        part = MarketData.from_frames(close, universe=universe.iloc[40:]).universe
    assert (full.to_numpy() == member.to_numpy()).all()
    assert not part.iloc[:40].any().any() and (part.iloc[40:].to_numpy() == member.iloc[40:].to_numpy()).all()
    for panel in (full, part):
        assert set(np.unique(panel.to_numpy().view(np.uint8)).tolist()) <= {0, 1}


@pytest.mark.parametrize("kind", ["object_nan", "object_none", "nullable", "numpy_bool"])
def test_from_frames_treats_missing_universe_cells_as_false(kind):
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = _panel(dates, columns=("A",))
    if kind == "nullable":
        universe = pd.DataFrame({"A": pd.array([True, None, False, True], dtype="boolean")}, index=dates)
    elif kind == "numpy_bool":
        universe = pd.DataFrame(
            {"A": [np.bool_(True), np.nan, np.bool_(False), np.bool_(True)]}, index=dates, dtype=object
        )
    else:
        gap = np.nan if kind == "object_nan" else None
        universe = pd.DataFrame({"A": [True, gap, False, True]}, index=dates, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        data = MarketData.from_frames(close, universe=universe.iloc[:3])
    assert data.universe["A"].tolist() == [True, False, False, False]
    assert data.universe["A"].dtype == bool


@pytest.mark.parametrize("field", ["open", "high", "low", "volume", "unadjusted_close", "universe"])
def test_from_frames_names_the_field_with_duplicate_labels(field):
    # reindexing a frame with a repeated date used to escape as pandas'
    # "cannot reindex on an axis with duplicate labels" ValueError
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = _panel(dates)
    frame = close.notna() if field == "universe" else close.copy()
    with pytest.raises(DataError, match=f"{field} has duplicate dates"):
        MarketData.from_frames(close, **{field: frame.iloc[[0, 1, 1, 3]]})
    with pytest.raises(DataError, match=f"{field} has duplicate tickers"):
        MarketData.from_frames(close, **{field: frame[["A", "A"]]})
    with pytest.raises(DataError, match=f"{field} must be a DataFrame"):
        MarketData.from_frames(close, **{field: frame["A"]})
    with pytest.raises(DataError, match="unknown MarketData fields"):
        MarketData.from_frames(close, vwap=close)


@pytest.mark.parametrize("how", ["slice_until", "slice_range"])
def test_slices_are_copies_so_writes_do_not_reach_the_parent(how):
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = _panel(dates)
    parent = MarketData.from_frames(
        close, open=close.copy(), high=close.copy(), low=close.copy(),
        volume=close.copy(), unadjusted_close=close.copy(), universe=close.notna(),
    )
    before = {name: getattr(parent, name).copy() for name in
              ("close", "open", "high", "low", "volume", "unadjusted_close", "universe")}
    part = parent.slice_until(dates[3]) if how == "slice_until" else parent.slice_range(dates[1], dates[4])
    assert len(part.dates) == 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the chained-write warning is not the point
        for name in before:
            getattr(part, name).iloc[1, 0] = False if name == "universe" else 999.0
    for name, frame in before.items():
        pd.testing.assert_frame_equal(getattr(parent, name), frame)


def test_market_does_not_cast_text_false_to_universe_membership():
    dates = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    with pytest.raises(DataError, match="boolean"):
        MarketData.from_frames(close, universe=pd.DataFrame("False", index=dates, columns=["A"]))


def test_synthetic_source_matches_make_market():
    kwargs = dict(n_assets=6, n_days=120, seed=3, split_asset=False, universe_churn=False)
    loaded = SyntheticSource(**kwargs).load()
    expected = make_market(**kwargs)
    assert_panel_close(loaded.close, expected.close)
    assert_panel_close(loaded.volume, expected.volume)
    assert_panel_close(loaded.unadjusted_close, expected.unadjusted_close)
    assert loaded.universe.equals(expected.universe)
    # loading twice is deterministic
    again = SyntheticSource(**kwargs).load()
    assert_panel_close(again.close, expected.close)


def test_wide_csv_round_trip(tmp_path, market_simple):
    directory = write_wide_dir(tmp_path / "wide", market_simple, universe_as_int=True)
    loaded = CSVSource(directory, format="wide").load()
    for name in ("close", "open", "high", "low", "volume", "unadjusted_close"):
        assert_panel_close(getattr(loaded, name), getattr(market_simple, name))
    assert all(dt == bool for dt in loaded.universe.dtypes)
    assert (loaded.universe.to_numpy() == market_simple.universe.to_numpy()).all()


def test_synthetic_generator_keeps_shifted_draws_out_of_compiled_code(monkeypatch):
    """rng.normal(loc, scale) and rng.uniform(low, high) compute loc + scale * z
    inside the generator, where some NumPy builds fuse the multiply and the add
    and return a different last bit. make_market applies every shift itself."""
    calls = []
    real_default_rng = np.random.default_rng

    class Recorder:
        def __init__(self, seed):
            self._rng = real_default_rng(seed)

        def __getattr__(self, name):
            method = getattr(self._rng, name)

            def call(*args, **kwargs):
                calls.append((name, args, kwargs))
                return method(*args, **kwargs)

            return call

    plain = make_market(n_assets=8, n_days=200, seed=7)
    monkeypatch.setattr(np.random, "default_rng", Recorder)
    recorded = make_market(n_assets=8, n_days=200, seed=7)
    assert recorded.close.equals(plain.close) and recorded.volume.equals(plain.volume)

    used = {name for name, _, _ in calls}
    assert "random" in used and "standard_normal" in used
    assert "uniform" not in used
    for name, args, kwargs in calls:
        if name in ("normal", "lognormal"):
            assert not kwargs and args[0] == 0.0, f"{name} called with a non-zero location"


def test_synthetic_default_panel_values_are_pinned():
    # guards the random stream (draw order and distributions) behind the
    # published example; the tolerance leaves room for another platform's
    # math library, which can move the last bits
    data = make_market(n_assets=20, n_days=1512, seed=7)
    np.testing.assert_allclose(
        data.close.iloc[0, :3], [68.56501556993888, 109.43804107590209, 47.996778949927375], rtol=1e-9
    )
    np.testing.assert_allclose(
        data.close.iloc[-1, :3], [40.03474894281373, 61.29746190264328, 9.278302392991979], rtol=1e-9
    )
    np.testing.assert_allclose(
        [data.open.iat[700, 5], data.high.iat[700, 5], data.low.iat[700, 5]],
        [15.629053837186618, 15.65476868241063, 15.256506708138025],
        rtol=1e-9,
    )
    np.testing.assert_allclose(data.volume.iloc[0, :3], [1534482.0, 9157356.0, 19911122.0], rtol=1e-9)
    np.testing.assert_allclose(data.unadjusted_close.iat[0, 0], 274.2600622797555, rtol=1e-9)
    assert int(data.universe.to_numpy().sum()) == 29559
    assert int(data.close.isna().to_numpy().sum()) == 681
    assert validate_market(data).issues == []


def test_csv_values_round_trip_exactly(tmp_path):
    # 17-significant-digit text needs the round-trip float parser; the default
    # one is off by one unit in the last place on a share of the cells
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    close = pd.DataFrame(100.0 * np.exp(rng.normal(0.0, 0.01, (300, 4)).cumsum(axis=0)),
                         index=dates, columns=list("ABCD"))
    directory = tmp_path / "wide"
    directory.mkdir()
    close.to_csv(directory / "close.csv", index_label="date")
    wide = CSVSource(directory).load().close
    assert (wide.to_numpy() == close.to_numpy()).all()

    long_file = tmp_path / "long.csv"
    close.rename_axis(index="date", columns="ticker").stack().rename("close").reset_index().to_csv(
        long_file, index=False
    )
    long = CSVSource(long_file, format="long").load().close
    assert (long.to_numpy() == close.to_numpy()).all()


def _write_csv_layouts(tmp_path, date_cells):
    """The same three-ticker rows as a wide directory and as a long file."""
    wide = tmp_path / "wide"
    wide.mkdir(exist_ok=True)
    (wide / "close.csv").write_text(
        "date,A,B\n" + "".join(f"{d},{100 + i},{50 + i}\n" for i, d in enumerate(date_cells))
    )
    long = tmp_path / "long.csv"
    long.write_text(
        "date,ticker,close\n"
        + "".join(f"{d},{t},{base + i}\n" for i, d in enumerate(date_cells) for t, base in (("A", 100), ("B", 50)))
    )
    return wide, long


@pytest.mark.parametrize(
    "date_cells",
    [
        # day-first file: read cell by cell this became 1 Oct, 1 Nov, 1 Dec,
        # 15 Jan, 16 Jan and sorting then scrambled the price history
        ["10/01/2024", "11/01/2024", "12/01/2024", "15/01/2024", "16/01/2024"],
        ["01/02/2024", "02/02/2024", "05/02/2024"],   # every day <= 12: silently month-first
        ["01/16/2024", "01/17/2024"],                 # consistent month-first
        ["10 Jan 2024", "11 Jan 2024"],
        ["2024-01-10", "notadate"],
        ["2024", "2025"],
        ["2024-01", "2024-02"],
        ["2024/01/10", "2024/01/11"],
        ["2024-1-5", "2024-1-8"],
        ["2024-01-10T00:00:00+00:00", "2024-01-11T00:00:00+00:00"],
    ],
)
def test_csv_rejects_non_iso_dates_in_both_layouts(tmp_path, date_cells):
    wide, long = _write_csv_layouts(tmp_path, date_cells)
    with pytest.raises(DataError, match=r"close\.csv: date in data row \d+ .* is not an ISO date"):
        CSVSource(wide).load()
    with pytest.raises(DataError, match=r"long\.csv: date in data row \d+ .* is not an ISO date"):
        CSVSource(long, format="long").load()


def test_csv_reports_blank_and_impossible_dates(tmp_path):
    wide, long = _write_csv_layouts(tmp_path, ["2024-01-10", ""])
    with pytest.raises(DataError, match=r"close\.csv: date in data row 2 is blank"):
        CSVSource(wide).load()
    with pytest.raises(DataError, match=r"long\.csv: date in data row 3 is blank"):
        CSVSource(long, format="long").load()
    wide, long = _write_csv_layouts(tmp_path, ["2024-01-10", "2024-02-30"])
    with pytest.raises(DataError, match=r"close\.csv: could not parse dates"):
        CSVSource(wide).load()
    with pytest.raises(DataError, match=r"long\.csv: could not parse dates"):
        CSVSource(long, format="long").load()


def test_wide_csv_rejects_non_iso_dates_in_optional_panels(tmp_path):
    wide, _ = _write_csv_layouts(tmp_path, ["2024-01-10", "2024-01-11"])
    (wide / "volume.csv").write_text("date,A,B\n10/01/2024,1,2\n11/01/2024,3,4\n")
    with pytest.raises(DataError, match=r"volume\.csv: date in data row 1 .* is not an ISO date"):
        CSVSource(wide).load()


@pytest.mark.parametrize(
    "date_cells",
    [
        ["2024-01-12", "2024-01-11", "2024-01-10"],
        ["2024-01-12 00:00:00", "2024-01-11 00:00:00", "2024-01-10 00:00:00"],
        ["2024-01-12T00:00:00", "2024-01-11", "2024-01-10T00:00"],
        ["20240112", "20240111", "20240110"],
    ],
)
def test_csv_accepts_iso_dates_in_both_layouts(tmp_path, date_cells):
    # rows are written newest first: the loader sorts, values travel with dates
    wide, long = _write_csv_layouts(tmp_path, date_cells)
    expected = pd.DatetimeIndex(["2024-01-10", "2024-01-11", "2024-01-12"])
    for source in (CSVSource(wide), CSVSource(long, format="long")):
        close = source.load().close
        assert close.index.equals(expected)
        assert close["A"].tolist() == [102.0, 101.0, 100.0]
        assert close["B"].tolist() == [52.0, 51.0, 50.0]


def test_csv_date_cells_may_be_padded_with_whitespace(tmp_path):
    # padding is not a question of format (fixed-width exports have it); the
    # cell is read just as strictly once the padding is dropped
    wide, long = _write_csv_layouts(tmp_path, ["2024-01-10 ", " 2024-01-11", "20240112\t"])
    expected = pd.DatetimeIndex(["2024-01-10", "2024-01-11", "2024-01-12"])
    for source in (CSVSource(wide), CSVSource(long, format="long")):
        close = source.load().close
        assert close.index.equals(expected)
        assert close["A"].tolist() == [100.0, 101.0, 102.0]
    wide, long = _write_csv_layouts(tmp_path, ["2024-01-10", "10/01/2024 "])
    with pytest.raises(DataError) as caught:
        CSVSource(wide).load()
    # the whole message: file, row, the cell as written, and what is accepted
    assert str(caught.value) == (
        f"{wide / 'close.csv'}: date in data row 2 '10/01/2024 ' is not an ISO date; "
        "use YYYY-MM-DD (optionally with a time, no UTC offset) or YYYYMMDD"
    )
    wide, long = _write_csv_layouts(tmp_path, ["2024-01-10", "   "])
    with pytest.raises(DataError, match=r"long\.csv: date in data row 3 '   ' is not an ISO date"):
        CSVSource(long, format="long").load()


def test_iso_date_pattern_keeps_both_alternatives_inside_added_anchors():
    # pandas matches Arrow-backed strings by wrapping the pattern in ^...$,
    # in some releases without grouping it first. A bare top-level `a|b`
    # would then accept "<ISO date><anything>" and "<anything><8 digits>".
    import re

    from alpha_lab.core.types import ISO_DATE_PATTERN

    anchored = re.compile("^" + ISO_DATE_PATTERN + "$")
    for good in ("2024-01-10", "2024-01-10 16:00", "2024-01-10T16:00:00.5", "20240110"):
        assert anchored.match(good) and re.fullmatch(ISO_DATE_PATTERN, good), good
    for bad in ("2024-01-10T00:00:00+00:00", "2024-01-10Z", "2024-01-10 x", "x20240110", "10/01/20240110"):
        assert not anchored.match(bad) and not re.fullmatch(ISO_DATE_PATTERN, bad), bad


def test_csv_dates_stay_strict_with_arrow_backed_strings(tmp_path):
    # the string type pandas 3 uses when pyarrow is installed; on pandas 2 the
    # same type is switched on by the option below
    pytest.importorskip("pyarrow")
    with pd.option_context("future.infer_string", True):
        wide, long = _write_csv_layouts(tmp_path, ["2024-01-10T00:00:00+00:00", "2024-01-11T00:00:00+00:00"])
        with pytest.raises(DataError, match=r"close\.csv: date in data row 1 .* is not an ISO date"):
            CSVSource(wide).load()
        with pytest.raises(DataError, match=r"long\.csv: date in data row 1 .* is not an ISO date"):
            CSVSource(long, format="long").load()
        wide, long = _write_csv_layouts(tmp_path, ["2024-01-10", "20240111"])
        expected = pd.DatetimeIndex(["2024-01-10", "2024-01-11"])
        assert CSVSource(wide).load().close.index.equals(expected)
        assert CSVSource(long, format="long").load().close.index.equals(expected)


@pytest.mark.parametrize(
    "header, message",
    [
        ("date,AAPL,MSFT,AAPL", r"repeated column headers \['AAPL'\]"),   # read_csv: 'AAPL.1'
        ("date,A,B,", "blank column header"),                              # read_csv: 'Unnamed: 3'
        ("date,A, ,B", "blank column header"),
        ("date,A,B,date", r"repeated column headers \['date'\]"),
    ],
)
def test_wide_csv_rejects_repeated_or_blank_ticker_headers(tmp_path, header, message):
    directory = tmp_path / "wide"
    directory.mkdir()
    width = header.count(",")
    (directory / "close.csv").write_text(
        header + "\n" + "".join(f"2024-01-1{i}," + ",".join(["100"] * width) + "\n" for i in range(3))
    )
    with pytest.raises(DataError, match=rf"close\.csv: {message}"):
        CSVSource(directory).load()


def test_wide_csv_needs_a_date_column(tmp_path):
    directory = tmp_path / "wide"
    directory.mkdir()
    (directory / "close.csv").write_text("day,A\n2024-01-10,100\n")
    with pytest.raises(DataError, match="first column must be 'date'"):
        CSVSource(directory).load()
    (directory / "close.csv").write_text("")
    with pytest.raises(DataError, match="first column must be 'date'"):
        CSVSource(directory).load()


@pytest.mark.parametrize(
    "field, content, axis",
    [
        ("volume", "date,a,b\n2024-01-10,1,2\n2024-01-11,3,4\n", "tickers"),    # header case
        ("universe", "date,C\n2024-01-10,1\n2024-01-11,1\n", "tickers"),
        ("open", "date,A,B\n2023-01-10,1,2\n2023-01-11,3,4\n", "dates"),        # other year
        ("unadjusted_close", "date,A,B\n", "dates"),                            # header only
    ],
)
def test_wide_csv_rejects_optional_panel_that_misses_close_entirely(tmp_path, field, content, axis):
    # aligned to close, such a panel became all-NaN (or an all-False universe)
    # and the quality report still said "no issues found"
    wide, _ = _write_csv_layouts(tmp_path, ["2024-01-10", "2024-01-11"])
    (wide / f"{field}.csv").write_text(content)
    with pytest.raises(DataError, match=rf"{field}\.csv: shares no {axis} with close\.csv"):
        CSVSource(wide).load()


def test_wide_csv_partial_optional_panel_still_aligns(tmp_path):
    wide, _ = _write_csv_layouts(tmp_path, ["2024-01-10", "2024-01-11", "2024-01-12"])
    (wide / "volume.csv").write_text("date,A\n2024-01-11,7\n2024-01-12,8\n")
    (wide / "universe.csv").write_text("date,A\n2024-01-11,1\n2024-01-12,0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        data = CSVSource(wide).load()
    assert data.volume["A"].tolist()[1:] == [7.0, 8.0] and np.isnan(data.volume["A"].iloc[0])
    assert data.volume["B"].isna().all()
    assert data.universe["A"].tolist() == [False, True, False]
    assert not data.universe["B"].any()
    # the ticker without any volume is reported by the quality pass
    missing = [i for i in validate_market(data).issues if i.code == "missing_field"]
    assert [(i.ticker, i.severity) for i in missing] == [("B", "warning")]
    assert "volume has no values while close has 3 prices" in missing[0].message


def test_long_csv_rejects_unknown_and_missing_columns(tmp_path):
    file = tmp_path / "long.csv"
    file.write_text("date,ticker,close,vwap\n2024-01-10,A,100,99\n")
    with pytest.raises(DataError, match=r"unknown long csv columns \['vwap'\]"):
        CSVSource(file, format="long").load()
    file.write_text("date,ticker,price\n2024-01-10,A,100\n")
    with pytest.raises(DataError, match=r"long csv missing columns \['close'\]"):
        CSVSource(file, format="long").load()
    file.write_text("ticker,close\nA,100\n")
    with pytest.raises(DataError, match=r"long csv missing columns \['date'\]"):
        CSVSource(file, format="long").load()
    file.write_text("date,ticker,close\n2024-01-10,A,100\n2024-01-10,A,101\n")
    with pytest.raises(DataError, match="cannot pivot 'close'"):
        CSVSource(file, format="long").load()


@pytest.mark.parametrize("cell, message", [("2", "is not 0/1"), ("0.5", "is not 0/1"), ("maybe", "is not boolean-like")])
def test_universe_csv_rejects_values_that_are_not_boolean(tmp_path, cell, message):
    wide, _ = _write_csv_layouts(tmp_path, ["2024-01-10", "2024-01-11"])
    (wide / "universe.csv").write_text(f"date,A,B\n2024-01-10,1,0\n2024-01-11,{cell},1\n")
    with pytest.raises(DataError, match=message):
        CSVSource(wide).load()


def test_universe_csv_accepts_the_usual_boolean_spellings(tmp_path):
    wide, _ = _write_csv_layouts(tmp_path, ["2024-01-10", "2024-01-11", "2024-01-12"])
    (wide / "universe.csv").write_text("date,A,B\n2024-01-10,TRUE,no\n2024-01-11,1,\n2024-01-12,f,Yes\n")
    universe = CSVSource(wide).load().universe
    assert universe["A"].tolist() == [True, True, False]
    assert universe["B"].tolist() == [False, False, True]


@pytest.mark.parametrize("bound", ["start", "end"])
@pytest.mark.parametrize("bad", ["notadate", "10/01/2024"])
def test_csv_source_rejects_bad_date_bounds_at_construction(tmp_path, bound, bad):
    with pytest.raises(ConfigError, match=f"csv source {bound} must be an ISO date"):
        CSVSource(tmp_path, **{bound: bad})
    with pytest.raises(ConfigError, match="unknown csv format"):
        CSVSource(tmp_path, format="tall")


def test_source_registry_rejects_double_registration_and_unknown_names():
    from alpha_lab.core.registry import Registry

    registry = Registry("widget")

    @registry.register("one")
    class One:
        def __init__(self, size: int = 1) -> None:
            self.size = size

    with pytest.raises(ConfigError, match="widget 'one' registered twice"):
        registry.register("one")(One)
    assert registry.names() == ["one"] and "one" in registry and "two" not in registry
    assert registry.create("one", size=3).size == 3
    with pytest.raises(ConfigError, match=r"unknown widget 'two'; available: \['one'\]"):
        registry.get("two")
    with pytest.raises(ConfigError, match="bad params for widget 'one'"):
        registry.create("one", colour="red")
    with pytest.raises(ConfigError, match="data_source 'csv' registered twice"):
        SOURCES.register("csv")(CSVSource)


def test_wide_csv_requires_close(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    with pytest.raises(DataError, match="close.csv"):
        CSVSource(directory, format="wide").load()


def test_long_csv_round_trip(tmp_path, market_simple):
    records = []
    for ticker in market_simple.tickers:
        for date in market_simple.dates:
            records.append(
                (
                    date,
                    ticker,
                    market_simple.close.at[date, ticker],
                    market_simple.volume.at[date, ticker],
                )
            )
    long_df = pd.DataFrame(records, columns=["date", "ticker", "close", "volume"])
    # shuffle rows: the source must not rely on input ordering
    long_df = long_df.sample(frac=1.0, random_state=0)
    file = tmp_path / "long.csv"
    long_df.to_csv(file, index=False)

    loaded = CSVSource(file, format="long").load()
    assert_panel_close(loaded.close, market_simple.close)
    assert_panel_close(loaded.volume, market_simple.volume)
    assert loaded.open is None and loaded.universe is None


def test_long_csv_keeps_numeric_and_na_like_tickers_as_strings(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=5)
    rows = [(d, t, 100.0 + i) for i, d in enumerate(dates) for t in ("10001", "NA", "AAPL")]
    file = tmp_path / "long.csv"
    pd.DataFrame(rows, columns=["date", "ticker", "close"]).to_csv(file, index=False)
    loaded = CSVSource(file, format="long", tickers=["10001", "NA"]).load()
    assert list(loaded.close.columns) == ["10001", "NA"]
    assert loaded.close.notna().all().all()


def test_tickers_start_end_filtering(tmp_path, market_simple):
    directory = write_wide_dir(tmp_path / "wide", market_simple)
    dates = market_simple.dates
    source = CSVSource(
        directory,
        format="wide",
        tickers=["SYM01", "SYM03"],
        start=str(dates[100].date()),
        end=str(dates[200].date()),
    )
    loaded = source.load()
    assert loaded.tickers == ["SYM01", "SYM03"]
    assert loaded.dates[0] == dates[100]
    assert loaded.dates[-1] == dates[200]
    assert len(loaded.dates) == 101
    expected = market_simple.close.loc[dates[100] : dates[200], ["SYM01", "SYM03"]]
    assert_panel_close(loaded.close, expected)
    # optional panels are subset alongside close
    assert loaded.volume.shape == loaded.close.shape


def test_unknown_ticker_raises(tmp_path, market_simple):
    directory = write_wide_dir(tmp_path / "wide", market_simple)
    with pytest.raises(DataError, match="SYMXX"):
        CSVSource(directory, format="wide", tickers=["SYM01", "SYMXX"]).load()


def test_source_from_config_synthetic():
    cfg = DataConfig(source="synthetic", synthetic=SyntheticConfig(n_assets=4, n_days=60, seed=1))
    source = source_from_config(cfg)
    assert isinstance(source, SyntheticSource)
    data = source.load()
    assert data.close.shape == (60, 4)
    assert_panel_close(data.close, make_market(n_assets=4, n_days=60, seed=1).close)


def test_source_from_config_csv(tmp_path, market_simple):
    directory = write_wide_dir(tmp_path / "wide", market_simple)
    cfg = DataConfig(source="csv", path=str(directory), tickers=["SYM00"])
    source = source_from_config(cfg)
    assert isinstance(source, CSVSource)
    assert source.load().tickers == ["SYM00"]


# -- validation -------------------------------------------------------------


def test_validate_clean_market(market_simple):
    report = validate_market(market_simple)
    assert report.ok
    assert report.errors() == []
    assert report.issues == []
    assert "no issues" in report.summary()


def test_validate_flags_planted_anomalies():
    base = make_market(n_assets=6, n_days=250, seed=5, split_asset=False, universe_churn=False)
    close = base.close.copy()
    volume = base.volume.copy()
    tickers = base.tickers

    # unadjusted-split fingerprint on tickers[2]: -60% level shift + volume spike
    close.iloc[100:, 2] = close.iloc[100:, 2] * 0.4
    volume.iloc[100, 2] = float(volume.iloc[79:100, 2].median()) * 10.0
    # 12 identical consecutive closes on tickers[3]
    close.iloc[50:62, 3] = close.iat[50, 3]
    # universe member with NaN close on tickers[4]
    close.iloc[150, 4] = np.nan

    planted = MarketData.from_frames(close, volume=volume, universe=base.universe)
    report = validate_market(planted)

    codes = {issue.code for issue in report.issues}
    assert {
        "extreme_move",
        "possible_unadjusted_split",
        "stale_price",
        "member_without_price",
    } <= codes

    def issues_for(code):
        return [issue for issue in report.issues if issue.code == code]

    assert any(
        i.ticker == tickers[2] and i.date == close.index[100] and i.severity == "warning"
        for i in issues_for("extreme_move")
    )
    assert any(
        i.ticker == tickers[2] and i.date == close.index[100]
        for i in issues_for("possible_unadjusted_split")
    )
    assert any(i.ticker == tickers[3] for i in issues_for("stale_price"))
    assert any(
        i.ticker == tickers[4] and i.date == close.index[150]
        for i in issues_for("member_without_price")
    )
    # only warnings were planted, so the report is still "ok"
    assert report.ok
    assert len(report.summary().splitlines()) == len(report.issues) + 1


def test_validate_negative_volume_and_error_escalation():
    base = make_market(n_assets=6, n_days=60, seed=9, split_asset=False, universe_churn=False)
    close = base.close.copy()
    volume = base.volume.copy()
    close.iloc[30:, 1] = close.iloc[30:, 1] * 2.2  # +120% jump: error-severity move
    volume.iloc[20, 0] = -500.0
    report = validate_market(MarketData.from_frames(close, volume=volume))
    assert not report.ok
    codes = {(i.code, i.severity) for i in report.errors()}
    assert ("negative_volume", "error") in codes
    assert ("extreme_move", "error") in codes


def _clean_market(n_days=60):
    return make_market(n_assets=4, n_days=n_days, seed=9, split_asset=False, universe_churn=False)


def _issues(report, code):
    return [issue for issue in report.issues if issue.code == code]


def test_price_gap_needs_more_than_three_missing_days_inside_listed_life():
    base = _clean_market()
    close = base.close.copy()
    tickers = base.tickers
    close.iloc[10:13, 0] = np.nan  # 3-day gap: tolerated
    close.iloc[20:24, 1] = np.nan  # 4-day gap: reported
    close.iloc[:6, 2] = np.nan     # not yet listed
    close.iloc[-6:, 3] = np.nan    # delisted
    gaps = _issues(validate_market(MarketData(close)), "price_gap")
    assert [(i.ticker, i.date, i.severity) for i in gaps] == [(tickers[1], close.index[20], "warning")]
    assert "4-day NaN gap" in gaps[0].message


@pytest.mark.parametrize("identical, flagged", [(9, False), (10, True), (11, True)])
def test_stale_price_counts_identical_consecutive_closes(identical, flagged):
    base = _clean_market()
    close = base.close.copy()
    close.iloc[20:20 + identical, 1] = close.iat[20, 1]
    data = MarketData(close)
    stale = _issues(validate_market(data), "stale_price")
    assert bool(stale) == flagged
    if flagged:
        assert [(i.ticker, i.date) for i in stale] == [(base.tickers[1], close.index[20])]
        assert f"{identical} identical consecutive closes" in stale[0].message
    # the threshold is the stale_days argument, inclusive
    assert bool(_issues(validate_market(data, stale_days=identical), "stale_price"))
    assert not _issues(validate_market(data, stale_days=identical + 1), "stale_price")


def test_volume_spike_rule_compares_with_prior_days_only():
    base = _clean_market()
    close, volume = base.close.copy(), base.volume.copy()
    # crash and spike on row 4: four earlier volumes, one short of the five
    # the trailing median needs, so the day's own volume must not be counted
    close.iloc[4:, 0] = close.iloc[4:, 0] * 0.4
    volume.iloc[4, 0] = volume.iloc[:4, 0].max() * 50.0
    # the same on row 5 of another ticker: five earlier volumes
    close.iloc[5:, 1] = close.iloc[5:, 1] * 0.4
    volume.iloc[5, 1] = volume.iloc[:5, 1].max() * 50.0
    report = validate_market(MarketData.from_frames(close, volume=volume))
    splits = _issues(report, "possible_unadjusted_split")
    assert [(i.ticker, i.date) for i in splits] == [(base.tickers[1], close.index[5])]
    prior_median = float(volume.iloc[:5, 1].median())
    assert f"> 3x trailing median {prior_median:,.0f}" in splits[0].message
    # both crashes are still extreme moves
    assert {(i.ticker, i.date) for i in _issues(report, "extreme_move")} == {
        (base.tickers[0], close.index[4]), (base.tickers[1], close.index[5]),
    }


@pytest.mark.parametrize(
    "label, ratio",
    [("2-for-1", 1 / 2), ("3-for-2", 2 / 3), ("3-for-1", 1 / 3), ("4-for-1", 1 / 4),
     ("5-for-1", 1 / 5), ("10-for-1", 1 / 10), ("1-for-2", 2.0), ("1-for-10", 10.0)],
)
@pytest.mark.parametrize("with_volume", [False, True])
def test_split_ratio_rule_flags_common_unadjusted_splits(label, ratio, with_volume):
    # A 2-for-1 split gives about -50% (above -50% on an up day) and roughly
    # doubles raw volume, so neither the 50% move rule nor the volume-spike
    # rule fires; a 3-for-2 split (-33%) could never fire either.
    base = _clean_market()
    close, volume = base.close.copy(), base.volume.copy()
    day_move = 1.004  # the split day's own return
    close.iloc[30:, 2] = close.iloc[30:, 2] * (ratio * day_move * close.iat[29, 2] / close.iat[30, 2])
    volume.iloc[30:, 2] = volume.iloc[30:, 2] / ratio
    data = MarketData.from_frames(close, volume=volume) if with_volume else MarketData(close)
    splits = _issues(validate_market(data), "possible_unadjusted_split")
    assert [(i.ticker, i.date, i.severity) for i in splits] == [
        (base.tickers[2], close.index[30], "warning")
    ]
    assert f"{ratio * day_move - 1.0:+.1%}" in splits[0].message
    if with_volume and ratio < 0.5:
        # a crash with more than triple volume is the older volume-spike rule
        assert "> 3x trailing median" in splits[0].message
    else:
        assert f"is close to a {label} split ratio" in splits[0].message


def test_split_ratio_rule_tolerance():
    base = _clean_market()
    close = base.close.copy()
    close.iloc[30:, 2] = close.iloc[30:, 2] * (0.5 * 1.03 * close.iat[29, 2] / close.iat[30, 2])
    data = MarketData(close)
    assert not _issues(validate_market(data), "possible_unadjusted_split")
    assert len(_issues(validate_market(data, split_tolerance=0.05), "possible_unadjusted_split")) == 1
    # an ordinary panel has no return near any split ratio
    assert not _issues(validate_market(base), "possible_unadjusted_split")


def test_contradictory_price_bars_are_reported():
    base = _clean_market()
    tickers, dates = base.tickers, base.dates
    open_, high, low = base.open.copy(), base.high.copy(), base.low.copy()
    high.iloc[10, 0] = low.iat[10, 0] * 0.9          # high below low
    open_.iloc[20, 1] = high.iat[20, 1] * 1.05       # open above high
    low.iloc[30, 2] = base.close.iat[30, 2] * 1.01   # low above close
    low.iloc[40, 2] = base.close.iat[40, 2] * 1.01   # and once more, later
    high.iloc[50:, 1] = np.nan                       # no high: not a bar to judge
    data = MarketData.from_frames(base.close, open=open_, high=high, low=low)
    report = validate_market(data)
    bars = _issues(report, "ohlc_inconsistent")
    # one warning per ticker and kind, dated at the first such bar
    found = {(i.ticker, i.date, i.message.split(";")[0]) for i in bars}
    assert len(found) == len(bars)
    assert (tickers[0], dates[10], "high below low on 1 of 60 bars") in found
    assert (tickers[1], dates[20], "open above high on 1 of 50 bars") in found
    assert (tickers[2], dates[30], "close below low on 2 of 60 bars") in found
    assert {(t, d) for t, d, _ in found} == {(tickers[0], dates[10]), (tickers[1], dates[20]), (tickers[2], dates[30])}
    # the message carries the first bar's two prices
    first = next(i for i in bars if i.ticker == tickers[2] and i.message.startswith("close below low"))
    assert first.message == (
        f"close below low on 2 of 60 bars; first: close {base.close.iat[30, 2]:g}"
        f" below low {low.iat[30, 2]:g}"
    )
    assert report.ok  # warnings only
    # a flat bar (open = high = low = close) is consistent
    flat = MarketData.from_frames(base.close, open=base.close, high=base.close, low=base.close)
    assert validate_market(flat).issues == []
    # every bar inverted: each ticker is reported once per kind with the full
    # count (12 lines here), not once per bar (720 lines)
    inverted = MarketData.from_frames(base.close, high=base.close * 0.5, low=base.close * 2.0)
    flood = _issues(validate_market(inverted), "ohlc_inconsistent")
    assert sorted((i.ticker, i.date, i.message.split(";")[0]) for i in flood) == sorted(
        (ticker, dates[0], f"{kind} on 60 of 60 bars")
        for ticker in tickers
        for kind in ("high below low", "close above high", "close below low")
    )


def test_missing_field_and_zero_volume_are_reported_per_ticker():
    base = _clean_market()
    tickers, dates = base.tickers, base.dates
    close = base.close.copy()
    close[tickers[3]] = np.nan                      # never priced: nothing to report
    close.iloc[-4:, 0] = np.nan                     # first ticker delisted near the end
    volume = base.volume[tickers[:1]].copy()        # volume for the first ticker only
    volume.iloc[5:8, 0] = 0.0
    volume.iloc[-4:, 0] = 0.0                       # no trading once unpriced: not counted
    data = MarketData.from_frames(close, volume=volume, unadjusted_close=close[tickers[:2]])
    report = validate_market(data)
    missing = {(i.ticker, i.message) for i in _issues(report, "missing_field")}
    assert missing == {
        (tickers[1], "volume has no values while close has 60 prices"),
        (tickers[2], "volume has no values while close has 60 prices"),
        (tickers[2], "unadjusted_close has no values while close has 60 prices"),
    }
    assert all(i.date is None and i.severity == "warning" for i in _issues(report, "missing_field"))
    zero = _issues(report, "zero_volume")
    assert [(i.ticker, i.date, i.message) for i in zero] == [
        (tickers[0], dates[5], "volume is zero on 3 of 56 priced days")
    ]
    assert report.ok
    assert len(report.summary().splitlines()) == len(report.issues) + 1
    # all-zero volume is reported for every ticker
    silent = MarketData.from_frames(base.close, volume=base.volume * 0.0)
    assert [i.ticker for i in _issues(validate_market(silent), "zero_volume")] == tickers


# -- store --------------------------------------------------------------


def test_store_round_trip(tmp_path, market_simple):
    store = MarketDataStore(tmp_path / "store")
    path = store.save(market_simple, "panel", snapshot="20240101-000000")
    assert path == tmp_path / "store" / "panel" / "20240101-000000"

    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["n_rows"] == len(market_simple.dates)
    assert manifest["n_cols"] == len(market_simple.tickers)
    assert manifest["tickers"] == market_simple.tickers
    assert manifest["start"] == market_simple.dates[0].isoformat()
    assert manifest["end"] == market_simple.dates[-1].isoformat()
    assert set(manifest["fields"]) == {
        "close", "open", "high", "low", "volume", "unadjusted_close", "universe",
    }
    assert manifest["close_sha256"] == hashlib.sha256((path / "close.csv").read_bytes()).hexdigest()

    loaded = store.load("panel", "20240101-000000")
    for name in ("close", "open", "high", "low", "volume", "unadjusted_close"):
        assert_panel_close(getattr(loaded, name), getattr(market_simple, name))
    assert all(dt == bool for dt in loaded.universe.dtypes)
    assert (loaded.universe.to_numpy() == market_simple.universe.to_numpy()).all()


def test_store_latest_snapshot_and_listing(tmp_path, market_simple):
    store = MarketDataStore(tmp_path / "store")
    store.save(market_simple, "panel", snapshot="20240101-000000")
    trimmed = market_simple.slice_range(end=market_simple.dates[99])
    store.save(trimmed, "panel", snapshot="20240202-000000")

    assert store.list_snapshots("panel") == ["20240101-000000", "20240202-000000"]
    latest = store.load("panel")  # snapshot=None -> latest by sorted name
    assert len(latest.dates) == 100
    assert latest.dates[-1] == market_simple.dates[99]


def test_store_default_snapshot_name(tmp_path, market_simple):
    store = MarketDataStore(tmp_path / "store")
    store.save(market_simple, "auto")
    snapshots = store.list_snapshots("auto")
    assert len(snapshots) == 1
    assert len(snapshots[0]) == len("20240101-000000")
    assert len(store.load("auto").dates) == len(market_simple.dates)


def test_store_missing_raises(tmp_path, market_simple):
    store = MarketDataStore(tmp_path / "store")
    with pytest.raises(DataError, match="nope"):
        store.load("nope")
    with pytest.raises(DataError, match="nope"):
        store.list_snapshots("nope")
    store.save(market_simple, "panel", snapshot="20240101-000000")
    with pytest.raises(DataError, match="missing-snap"):
        store.load("panel", "missing-snap")


def test_store_rejects_overwrite_without_changing_existing_snapshot(tmp_path, market_simple):
    store = MarketDataStore(tmp_path)
    path = store.save(market_simple, "panel", "fixed")
    original = (path / "close.csv").read_bytes()
    with pytest.raises(DataError, match="already exists"):
        store.save(market_simple.slice_until(market_simple.dates[10]), "panel", "fixed")
    assert (path / "close.csv").read_bytes() == original


def test_failed_snapshot_save_never_becomes_latest(tmp_path, market_simple, monkeypatch):
    from pathlib import Path

    store = MarketDataStore(tmp_path)
    original = store.save(market_simple, "panel", "20240101")
    write_bytes = Path.write_bytes

    def fail_volume(path, content):
        if path.name == "volume.csv":
            raise OSError("simulated disk failure")
        return write_bytes(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_volume)
    with pytest.raises(DataError, match="could not save"):
        store.save(market_simple, "panel", "20240201")
    # A crash (which cannot run cleanup) may also leave a pending directory.
    (original.parent / ".pending-abandoned").mkdir()
    assert store.list_snapshots("panel") == ["20240101"]
    pd.testing.assert_frame_equal(store.load("panel").close, market_simple.close, check_freq=False, check_names=False)


@pytest.mark.parametrize("field", ["close", "volume", "unadjusted_close", "universe"])
def test_store_verifies_all_input_field_checksums(tmp_path, market_simple, field):
    store = MarketDataStore(tmp_path)
    path = store.save(market_simple, "panel", "fixed")
    file = path / f"{field}.csv"
    file.write_bytes(file.read_bytes() + b"\n")
    with pytest.raises(DataError, match="checksum"):
        store.load("panel", "fixed")


def test_store_preserves_integer_security_identifiers_and_exact_prices(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=3)
    data = MarketData(pd.DataFrame({10001: [0.12345678901234567, 0.12345678901234568, 1000.1234567890123]}, index=dates))
    store = MarketDataStore(tmp_path)
    store.save(data, "panel", "fixed")
    loaded = store.load("panel", "fixed")
    pd.testing.assert_frame_equal(loaded.close, data.close, check_freq=False, check_names=False, check_exact=True)


def test_legacy_snapshot_parses_false_universe_cells(tmp_path):
    directory = tmp_path / "panel" / "old"
    directory.mkdir(parents=True)
    (directory / "close.csv").write_text("date,A\n2024-01-01,100\n2024-01-02,101\n")
    (directory / "universe.csv").write_text("date,A\n2024-01-01,false\n2024-01-02,\n")
    data = MarketDataStore(tmp_path).load("panel", "old")
    assert not data.universe.any().any()


@pytest.mark.parametrize("bad", ["../x", "a/b", ".pending-x", ".", "..", "/abs", "x/"])
def test_store_rejects_ids_that_are_not_single_path_components(tmp_path, market_simple, bad):
    store = MarketDataStore(tmp_path / "store")
    store.save(market_simple, "panel", "s1")
    message = "single non-empty path components"
    with pytest.raises(DataError, match=message):
        store.save(market_simple, bad, "s2")
    with pytest.raises(DataError, match=message):
        store.save(market_simple, "panel", bad)
    with pytest.raises(DataError, match=message):
        store.load(bad)
    with pytest.raises(DataError, match=message):
        store.load("panel", bad)
    with pytest.raises(DataError, match=message):
        store.list_snapshots(bad)
    # nothing was written outside the one legitimate snapshot
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store"]
    assert sorted(p.name for p in (tmp_path / "store").iterdir()) == ["panel"]
    assert store.list_snapshots("panel") == ["s1"]


@pytest.mark.parametrize("bad", ["", None, 5])
def test_store_rejects_empty_or_non_string_dataset_names(tmp_path, market_simple, bad):
    store = MarketDataStore(tmp_path / "store")
    with pytest.raises(DataError, match="single non-empty path components"):
        store.save(market_simple, bad, "s1")
    with pytest.raises(DataError, match="single non-empty path components"):
        store.list_snapshots(bad)
    assert not (tmp_path / "store").exists()


def test_store_rejects_manifest_that_disagrees_with_the_files(tmp_path, market_simple):
    store = MarketDataStore(tmp_path)
    path = store.save(market_simple, "panel", "fixed")
    manifest = json.loads((path / "manifest.json").read_text())

    def load_with(**changes):
        (path / "manifest.json").write_text(json.dumps({**manifest, **changes}))
        return store.load("panel", "fixed")

    assert len(load_with().dates) == manifest["n_rows"]
    for changes, message in (
        ({"n_rows": manifest["n_rows"] + 1}, "shape does not match manifest"),
        ({"n_cols": manifest["n_cols"] - 1}, "shape does not match manifest"),
        ({"fields": ["close", "vwap"]}, "invalid snapshot fields"),
        ({"fields": ["close", "close"]}, "invalid snapshot fields"),
        ({"fields": "close"}, "invalid snapshot fields"),
        ({"fields": ["volume"]}, "has no close.csv"),
        ({"tickers": manifest["tickers"][::-1]}, "ticker mismatch"),
        ({"field_sha256": "not-a-mapping"}, "invalid snapshot checksums"),
    ):
        with pytest.raises(DataError, match=message):
            load_with(**changes)
    (path / "manifest.json").write_text("{not json")
    with pytest.raises(DataError, match="invalid snapshot manifest"):
        store.load("panel", "fixed")
    (path / "manifest.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "fields"}))
    with pytest.raises(DataError, match="invalid snapshot manifest"):
        store.load("panel", "fixed")
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "volume.csv").unlink()
    with pytest.raises(DataError, match="snapshot is missing volume.csv"):
        store.load("panel", "fixed")


def test_store_does_not_replace_a_snapshot_published_while_staging(tmp_path, market_simple, monkeypatch):
    from pathlib import Path

    store = MarketDataStore(tmp_path)
    target = tmp_path / "panel" / "fixed"
    write_text = Path.write_text

    def racing_write(path, content, *args, **kwargs):
        if path.name == "manifest.json":
            target.mkdir()  # another writer publishes the same id first
        return write_text(path, content, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", racing_write)
    with pytest.raises(DataError, match="snapshot already exists"):
        store.save(market_simple, "panel", "fixed")
    # the other writer's (here empty) directory is untouched, no staging left
    assert list(target.iterdir()) == []
    assert [p.name for p in target.parent.iterdir()] == ["fixed"]


def test_store_rejects_an_empty_panel(tmp_path, market_simple):
    # a header-only close.csv has no dates to parse back, so the snapshot
    # used to save and then fail to load
    store = MarketDataStore(tmp_path / "store")
    empty = market_simple.slice_until("2000-01-01")
    assert len(empty.dates) == 0
    with pytest.raises(DataError, match="cannot snapshot an empty panel"):
        store.save(empty, "panel", "s1")
    assert not (tmp_path / "store").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_published_snapshot_takes_its_dataset_directory_permissions(tmp_path, market_simple):
    # staging happens in a temporary directory created with mode 0700; the
    # published snapshot must not keep that mode
    store = MarketDataStore(tmp_path / "store")
    first = store.save(market_simple, "panel", "s1")

    def mode(path):
        return path.stat().st_mode & 0o777

    assert mode(first) == mode(first.parent)
    for wanted in (0o750, 0o755):
        first.parent.chmod(wanted)
        published = store.save(market_simple, "panel", f"s{wanted:o}")
        assert mode(published) == wanted
        assert len(store.load("panel", f"s{wanted:o}").dates) == len(market_simple.dates)
