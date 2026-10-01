"""Offline tests for the downloader's panel construction (no network)."""

import numpy as np
import pandas as pd

import download_data
from download_data import split_adjusted_panels


def _response(n=6):
    """A yfinance-shaped auto_adjust=False frame: (field, ticker) columns."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(0)
    fields = {}
    for t in ("AAA", "BBB"):
        close = 100 + rng.standard_normal(n).cumsum()
        factor = np.linspace(0.97, 1.0, n)  # dividend adjustment, 1 on the last bar
        fields[("Close", t)] = close
        fields[("Adj Close", t)] = close * factor
        fields[("Open", t)] = close * 0.99
        fields[("High", t)] = close * 1.01
        fields[("Low", t)] = close * 0.98
        fields[("Volume", t)] = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(fields, index=dates)


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
    import pytest
    raw = _response()
    raw.loc[:, pd.IndexSlice[:, "BBB"]] = np.nan
    monkeypatch.setattr(download_data.yf, "download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="no observations"):
        download_data.download_batch(["AAA", "BBB"])


def test_index_failure_preserves_every_existing_cache_file(monkeypatch, tmp_path):
    import pytest
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
    import pytest
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
