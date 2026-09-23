"""Tests for the data layer: sources, validation, snapshot store."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config.schema import DataConfig, SyntheticConfig
from alpha_lab.core.errors import DataError
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
