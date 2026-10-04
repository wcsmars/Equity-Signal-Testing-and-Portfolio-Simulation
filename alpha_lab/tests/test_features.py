"""Feature library and FeatureStore tests.

Covers exact math on the clean panel, truncation invariance for every
registered feature on the messy panel (split + universe churn), NaN
discipline, RSI bounds, and store memoization / disk cache / fingerprint.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import FeatureSpec
from alpha_lab.core.registry import Registry
from alpha_lab.core.types import MarketData
from alpha_lab.data.synthetic import make_market
from alpha_lab.features.library import (
    FEATURES,
    ADVDollars,
    Momentum,
    RSI,
    RealizedVol,
    WindowReturn,
    ZScoreReturn,
)
from alpha_lab.features.store import _FINGERPRINT_FIELDS, FeatureStore, fingerprint

ALL_FEATURE_NAMES = sorted(FEATURES.names())


# --------------------------------------------------------------------------
# exact math on the clean panel
# --------------------------------------------------------------------------

def test_registry_has_exact_contract_names():
    assert ALL_FEATURE_NAMES == [
        "adv_dollars", "momentum", "realized_vol", "returns", "rsi", "zscore_return",
    ]


def test_returns_exact_cells(market_simple):
    c = market_simple.close
    for window in (1, 5):
        panel = WindowReturn(window=window).compute(market_simple)
        for row, col in [(50, "SYM01"), (200, "SYM03"), (399, "SYM05")]:
            expected = c[col].iloc[row] / c[col].iloc[row - window] - 1.0
            assert panel[col].iloc[row] == pytest.approx(expected, rel=1e-12)
    pd.testing.assert_frame_equal(
        WindowReturn(window=1).compute(market_simple), c / c.shift(1) - 1.0
    )


def test_momentum_exact_cells(market_simple):
    c = market_simple.close
    panel = Momentum().compute(market_simple)  # window=252, skip=21
    for row, col in [(260, "SYM00"), (300, "SYM02"), (399, "SYM04")]:
        expected = c[col].iloc[row - 21] / c[col].iloc[row - 252] - 1.0
        assert panel[col].iloc[row] == pytest.approx(expected, rel=1e-12)
    # first date with full lookback is row 252; row 251 must be NaN
    assert np.isnan(panel["SYM00"].iloc[251])
    assert not np.isnan(panel["SYM00"].iloc[252])
    pd.testing.assert_frame_equal(panel, c.shift(21) / c.shift(252) - 1.0)


def test_momentum_skip_zero_matches_window_return(market_simple):
    pd.testing.assert_frame_equal(
        Momentum(window=63, skip=0).compute(market_simple),
        WindowReturn(window=63).compute(market_simple),
    )


def test_momentum_param_validation():
    with pytest.raises(ConfigError):
        Momentum(window=21, skip=21)
    with pytest.raises(ConfigError):
        Momentum(window=10, skip=20)
    with pytest.raises(ConfigError):
        Momentum(window=252, skip=-1)


def test_realized_vol_exact(market_simple):
    r = market_simple.close.pct_change(fill_method=None)
    expected = r.rolling(63, min_periods=63).std() * np.sqrt(252)
    pd.testing.assert_frame_equal(RealizedVol().compute(market_simple), expected)


def test_zscore_return_exact(market_simple):
    r = market_simple.close.pct_change(fill_method=None)
    mean = r.rolling(126, min_periods=126).mean()
    std = r.rolling(126, min_periods=126).std()
    pd.testing.assert_frame_equal(ZScoreReturn().compute(market_simple), (r - mean) / std)


def test_adv_uses_unadjusted_close(market):
    panel = ADVDollars().compute(market)
    expected = (market.volume * market.unadjusted_close).rolling(21, min_periods=10).mean()
    pd.testing.assert_frame_equal(panel, expected)
    # SYM00 has a 4:1 split at ~60% of the sample: pre-split, ADV from the
    # adjusted close would be 4x too small.
    wrong = (market.volume * market.close).rolling(21, min_periods=10).mean()
    t = market.dates[100]
    assert panel.loc[t, "SYM00"] == pytest.approx(4.0 * wrong.loc[t, "SYM00"], rel=1e-9)


def test_adv_falls_back_to_close_without_unadjusted(market_simple):
    data = MarketData.from_frames(market_simple.close, volume=market_simple.volume)
    panel = ADVDollars().compute(data)
    expected = (market_simple.volume * market_simple.close).rolling(21, min_periods=10).mean()
    pd.testing.assert_frame_equal(panel, expected)


def test_adv_requires_volume(market_simple):
    data = MarketData.from_frames(market_simple.close)
    with pytest.raises(DataError):
        ADVDollars().compute(data)


def test_rsi_exact(market_simple):
    delta = market_simple.close.diff()
    gain, loss = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
    g = gain.rolling(14, min_periods=14).mean()
    expected = 100.0 * g / (g + loss.rolling(14, min_periods=14).mean())
    pd.testing.assert_frame_equal(RSI(window=14).compute(market_simple), expected)


def test_rsi_edge_cases():
    dates = pd.bdate_range("2020-01-02", periods=8)
    close = pd.DataFrame(
        {"UP": np.arange(8.0) + 10, "DOWN": 20 - np.arange(8.0), "FLAT": 15.0}, index=dates
    )
    last = RSI(window=4).compute(MarketData.from_frames(close)).iloc[-1]
    assert last["UP"] == 100.0 and last["DOWN"] == 0.0 and last["FLAT"] == 50.0


def test_rsi_bounded(market_simple):
    panel = RSI().compute(market_simple)
    vals = panel.to_numpy().ravel()
    vals = vals[~np.isnan(vals)]
    assert len(vals) > 1000
    assert (vals >= 0.0).all()
    assert (vals <= 100.0).all()


# --------------------------------------------------------------------------
# structural invariants for every registered feature (default params)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_FEATURE_NAMES)
def test_alignment_to_close(market, name):
    panel = FEATURES.create(name).compute(market)
    assert panel.index.equals(market.close.index)
    assert panel.columns.equals(market.close.columns)


@pytest.mark.parametrize("name", ALL_FEATURE_NAMES)
def test_truncation_invariance(market, name):
    """CONVENTIONS.md rule 1: compute(slice_until(t)).loc[t] == compute(full).loc[t].

    Sampled dates straddle the split (~row 453) and the delisting (~row 604).
    """
    feature = FEATURES.create(name)
    full = feature.compute(market)
    for pos in (280, 420, 470, 620, 755):
        t = market.dates[pos]
        truncated = feature.compute(market.slice_until(t))
        assert truncated.index[-1] == t
        pd.testing.assert_series_equal(truncated.loc[t], full.loc[t], check_names=False)


@pytest.mark.parametrize("name", ALL_FEATURE_NAMES)
def test_nan_before_universe_entry(market, name):
    """Entrant (last ticker) has NaN prices pre-entry -> features must be NaN."""
    entrant = market.tickers[-1]
    first_valid = market.close[entrant].first_valid_index()
    pre_entry = market.dates[market.dates < first_valid]
    assert len(pre_entry) > 30  # sanity: fixture really has churn
    panel = FEATURES.create(name).compute(market)
    assert panel.loc[pre_entry, entrant].isna().all()


@pytest.mark.parametrize("name", ALL_FEATURE_NAMES)
def test_declared_lookback_positive(name):
    feature = FEATURES.create(name)
    assert feature.lookback >= 1
    assert feature.name == name


@pytest.mark.parametrize(
    "feature",
    [
        WindowReturn(window=1),
        WindowReturn(window=3),
        Momentum(window=5, skip=1),
        RealizedVol(window=4),
        ZScoreReturn(window=3),
        RSI(window=4),
    ],
    ids=lambda f: f"{f.name}-lb{f.lookback}",
)
def test_lookback_bars_suffice_for_first_value(feature, market_simple):
    """``lookback`` bars of history must yield a defined (non-NaN) final row,
    and one bar fewer must not — regression for the returns/momentum
    off-by-one (close.shift(window) needs a bar at t-window: window+1 closes)."""
    lb = feature.lookback
    enough = market_simple._iloc(slice(0, lb))
    assert feature.compute(enough).iloc[-1].notna().all()
    too_few = market_simple._iloc(slice(0, lb - 1))
    assert feature.compute(too_few).isna().all().all()


def test_adv_short_window_computes(market_simple):
    """Regression: window < 5 used to crash inside pandas (min_periods > window)."""
    for window in (1, 2, 4):
        panel = ADVDollars(window=window).compute(market_simple)
        expected = (
            (market_simple.volume * market_simple.unadjusted_close)
            .rolling(window, min_periods=window)
            .mean()
        )
        pd.testing.assert_frame_equal(panel, expected)


# --------------------------------------------------------------------------
# FeatureStore
# --------------------------------------------------------------------------

def _counting_registry():
    """Fresh registry whose 'returns' counts compute() invocations."""
    calls = {"n": 0}

    class CountingReturn(WindowReturn):
        def compute(self, data):
            calls["n"] += 1
            return super().compute(data)

    registry = Registry("feature")
    registry.register("returns")(CountingReturn)
    return registry, calls


def test_store_memoizes_in_memory(market_simple):
    registry, calls = _counting_registry()
    store = FeatureStore(registry=registry)
    spec = FeatureSpec.make("returns", window=1)
    first = store.get(spec, market_simple)
    second = store.get(spec, market_simple)
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second)


def test_store_recomputes_for_different_data():
    data7 = make_market(n_assets=6, n_days=150, seed=7)
    data8 = make_market(n_assets=6, n_days=150, seed=8)
    registry, calls = _counting_registry()
    store = FeatureStore(registry=registry)
    spec = FeatureSpec.make("returns", window=1)
    store.get(spec, data7)
    store.get(spec, data8)
    store.get(spec, data7)  # memoized
    assert calls["n"] == 2


def test_store_compute_all_dedupes_stable_order(market_simple):
    registry, calls = _counting_registry()
    store = FeatureStore(registry=registry)
    spec1 = FeatureSpec.make("returns", window=1)
    spec5 = FeatureSpec.make("returns", window=5)
    out = store.compute_all([spec1, spec5, FeatureSpec.make("returns", window=1)], market_simple)
    assert list(out) == [spec1.key, spec5.key]
    assert calls["n"] == 2
    pd.testing.assert_frame_equal(out[spec1.key], store.get(spec1, market_simple))


def test_store_unknown_feature_is_config_error(market_simple):
    with pytest.raises(ConfigError):
        FeatureStore().get(FeatureSpec.make("does_not_exist"), market_simple)


def test_feature_cache_does_not_confuse_numeric_and_text_parameters(market_simple):
    store = FeatureStore()
    store.get(FeatureSpec.make("returns", window=1), market_simple)
    # A textual '1' used to share the cache key and skip parameter validation.
    with pytest.raises(ConfigError, match="integer"):
        store.get(FeatureSpec.make("returns", window="1"), market_simple)


def test_store_disk_cache_roundtrip(tmp_path, market_simple):
    registry1, calls1 = _counting_registry()
    store1 = FeatureStore(registry=registry1, cache_dir=tmp_path)
    spec = FeatureSpec.make("returns", window=1)
    computed = store1.get(spec, market_simple)
    assert calls1["n"] == 1
    assert list(tmp_path.iterdir())  # something was persisted

    # a fresh store (empty memory) must satisfy the get from disk
    registry2, calls2 = _counting_registry()
    store2 = FeatureStore(registry=registry2, cache_dir=tmp_path)
    loaded = store2.get(spec, market_simple)
    assert calls2["n"] == 0
    pd.testing.assert_frame_equal(loaded, computed, check_freq=False)


def test_store_disk_cache_csv_fallback(tmp_path, market_simple, monkeypatch):
    def no_pyarrow(self, *args, **kwargs):
        raise ImportError("pyarrow unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_pyarrow)
    registry1, _ = _counting_registry()
    store1 = FeatureStore(registry=registry1, cache_dir=tmp_path)
    spec = FeatureSpec.make("returns", window=1)
    computed = store1.get(spec, market_simple)
    assert list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.parquet"))

    registry2, calls2 = _counting_registry()
    store2 = FeatureStore(registry=registry2, cache_dir=tmp_path)
    loaded = store2.get(spec, market_simple)
    assert calls2["n"] == 0
    pd.testing.assert_frame_equal(loaded, computed, check_freq=False)


def test_store_csv_fallback_keeps_integer_ticker_labels(tmp_path, market_simple, monkeypatch):
    # The csv round trip turns integer labels into strings; a cache hit must
    # still align to data.close, bit-exactly, or signals silently go flat.
    def no_pyarrow(self, *args, **kwargs):
        raise ImportError("pyarrow unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_pyarrow)
    ids = list(range(10001, 10001 + market_simple.close.shape[1]))
    fields = ("close", "open", "high", "low", "volume", "unadjusted_close", "universe")
    data = MarketData(**{
        f: None if getattr(market_simple, f) is None else getattr(market_simple, f).set_axis(ids, axis=1)
        for f in fields
    })
    spec = FeatureSpec.make("returns", window=1)
    registry1, _ = _counting_registry()
    computed = FeatureStore(registry=registry1, cache_dir=tmp_path).get(spec, data)
    registry2, calls2 = _counting_registry()
    loaded = FeatureStore(registry=registry2, cache_dir=tmp_path).get(spec, data)
    assert calls2["n"] == 0
    assert loaded.columns.equals(data.close.columns)
    pd.testing.assert_frame_equal(loaded, computed, check_freq=False, check_exact=True)


def test_store_ignores_misaligned_disk_panel(tmp_path, market_simple):
    spec = FeatureSpec.make("returns", window=1)
    registry1, _ = _counting_registry()
    store = FeatureStore(registry=registry1, cache_dir=tmp_path)
    store.get(spec, market_simple)
    [path] = list(tmp_path.iterdir())
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0, parse_dates=True)
    frame = frame.iloc[:-1]  # a stale/corrupt cache file
    if path.suffix == ".parquet":
        frame.to_parquet(path)
    else:
        frame.to_csv(path)
    registry2, calls2 = _counting_registry()
    loaded = FeatureStore(registry=registry2, cache_dir=tmp_path).get(spec, market_simple)
    assert calls2["n"] == 1  # recomputed, not served misaligned
    assert loaded.index.equals(market_simple.close.index)
    assert list(tmp_path.iterdir()) == [path]  # same entry, replaced in place


def test_disk_cache_misses_when_the_feature_source_changes(tmp_path, market_simple, monkeypatch):
    # the disk key carries a hash of the feature's source text: an edited
    # implementation must not be served the panel its old code wrote
    import inspect

    spec = FeatureSpec.make("returns", window=1)
    registry1, _ = _counting_registry()
    FeatureStore(registry=registry1, cache_dir=tmp_path).get(spec, market_simple)

    real_getsource = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj: real_getsource(obj) + "\n# edited\n")
    registry2, calls2 = _counting_registry()
    FeatureStore(registry=registry2, cache_dir=tmp_path).get(spec, market_simple)
    assert calls2["n"] == 1
    assert len(list(tmp_path.iterdir())) == 2

    # source that cannot be read (a feature defined interactively) falls
    # back to the qualified name and the code as loaded: still cached,
    # under its own entry
    def no_source(obj):
        raise OSError("source code not available")

    monkeypatch.setattr(inspect, "getsource", no_source)
    registry3, calls3 = _counting_registry()
    FeatureStore(registry=registry3, cache_dir=tmp_path).get(spec, market_simple)
    registry4, calls4 = _counting_registry()
    FeatureStore(registry=registry4, cache_dir=tmp_path).get(spec, market_simple)
    assert (calls3["n"], calls4["n"]) == (1, 0)
    assert len(list(tmp_path.iterdir())) == 3

    # ... and the name alone still keeps two implementations apart
    class HundredFold(WindowReturn):
        def compute(self, data):
            return 100.0 * super().compute(data)

    registry5 = Registry("feature")
    registry5.register("returns")(HundredFold)
    other = FeatureStore(registry=registry5, cache_dir=tmp_path).get(spec, market_simple)
    pd.testing.assert_frame_equal(other, 100.0 * market_simple.returns())


def _no_pyarrow(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("pyarrow unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", unavailable)
    monkeypatch.setattr(pd, "read_parquet", unavailable)


def _disk_formats():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return ["csv"]
    return ["parquet", "csv"]


def test_disk_cache_separates_parameterisations_of_one_feature(tmp_path, market_simple):
    # the disk key must carry the params, not only the feature name
    one, five = FeatureSpec.make("returns", window=1), FeatureSpec.make("returns", window=5)
    FeatureStore(cache_dir=tmp_path).get(one, market_simple)  # persist window=1 only
    got = FeatureStore(cache_dir=tmp_path).get(five, market_simple)  # fresh memory
    expected = market_simple.close / market_simple.close.shift(5) - 1.0
    pd.testing.assert_frame_equal(got, expected, check_freq=False)
    assert len(list(tmp_path.iterdir())) == 2
    back = FeatureStore(cache_dir=tmp_path).get(one, market_simple)
    pd.testing.assert_frame_equal(back, market_simple.returns(), check_freq=False)


@pytest.mark.parametrize("fmt", _disk_formats())
@pytest.mark.parametrize("damage", ["truncated", "empty"])
def test_store_recomputes_and_replaces_unreadable_disk_panel(
    tmp_path, market_simple, monkeypatch, fmt, damage
):
    # an interrupted write used to leave a file that crashed every later run
    if fmt == "csv":
        _no_pyarrow(monkeypatch)
    spec = FeatureSpec.make("returns", window=1)
    registry1, _ = _counting_registry()
    computed = FeatureStore(registry=registry1, cache_dir=tmp_path).get(spec, market_simple)
    [path] = list(tmp_path.iterdir())
    assert path.suffix == f".{fmt}"
    blob = path.read_bytes()
    path.write_bytes(b"" if damage == "empty" else blob[: len(blob) // 2])

    registry2, calls2 = _counting_registry()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = FeatureStore(registry=registry2, cache_dir=tmp_path).get(spec, market_simple)
    assert calls2["n"] == 1  # a miss, not a crash
    pd.testing.assert_frame_equal(loaded, computed)
    # half a csv still parses (as a shorter, misaligned panel: a silent
    # miss); the other three cases cannot be parsed at all and say so
    if not (fmt == "csv" and damage == "truncated"):
        assert any("unreadable" in str(w.message) for w in caught)
    # ... and the bad file was replaced by a good one, served from then on
    assert list(tmp_path.iterdir()) == [path]
    registry3, calls3 = _counting_registry()
    again = FeatureStore(registry=registry3, cache_dir=tmp_path).get(spec, market_simple)
    assert calls3["n"] == 0
    pd.testing.assert_frame_equal(again, computed, check_freq=False)


@pytest.mark.parametrize("fmt", _disk_formats())
def test_store_failed_disk_write_leaves_no_file(tmp_path, market_simple, monkeypatch, fmt):
    # the panel is written to a temporary name and renamed: a write that dies
    # half way must leave neither a partial panel nor the temporary behind
    if fmt == "csv":
        _no_pyarrow(monkeypatch)
    method = "to_parquet" if fmt == "parquet" else "to_csv"

    def dies_half_way(self, path, *args, **kwargs):
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise OSError("disk full")

    spec = FeatureSpec.make("returns", window=1)
    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, method, dies_half_way)
        with pytest.raises(OSError, match="disk full"):
            FeatureStore(cache_dir=tmp_path).get(spec, market_simple)
    assert list(tmp_path.iterdir()) == []

    # a normal write leaves exactly the final file, no temporary
    FeatureStore(cache_dir=tmp_path).get(spec, market_simple)
    [path] = list(tmp_path.iterdir())
    assert path.suffix == f".{fmt}" and not path.name.startswith(".")


def test_store_validates_spec_before_serving_from_disk(tmp_path, market_simple):
    spec = FeatureSpec.make("momentum", window=20, skip=5)
    FeatureStore(cache_dir=tmp_path).get(spec, market_simple)
    # a cached panel must not answer for a name this store's registry lacks
    with pytest.raises(ConfigError, match="unknown feature"):
        FeatureStore(registry=Registry("feature"), cache_dir=tmp_path).get(spec, market_simple)


def test_disk_cache_misses_when_the_implementation_differs(tmp_path, market_simple):
    # another implementation registered under the same name and params must
    # not be served the panel the first implementation persisted
    spec = FeatureSpec.make("returns", window=1)
    original = FeatureStore(cache_dir=tmp_path).get(spec, market_simple)

    class HundredFold(WindowReturn):
        def compute(self, data):
            return 100.0 * super().compute(data)

    registry = Registry("feature")
    registry.register("returns")(HundredFold)
    other = FeatureStore(registry=registry, cache_dir=tmp_path).get(spec, market_simple)
    pd.testing.assert_frame_equal(other, 100.0 * original)
    assert len(list(tmp_path.iterdir())) == 2
    # each implementation still finds its own entry
    back = FeatureStore(cache_dir=tmp_path).get(spec, market_simple)
    pd.testing.assert_frame_equal(back, original, check_freq=False)


def _sourceless_registry(scaled, calls):
    """A 'returns' feature whose class has no source text to read, as when
    it is defined in an interactive session."""
    namespace = {"WindowReturn": WindowReturn, "calls": calls, "__name__": "interactive_session"}
    exec(
        "class Scaled(WindowReturn):\n"
        "    def compute(self, data):\n"
        "        calls.append(1)\n"
        f"        return {scaled} super().compute(data)\n",
        namespace,
    )
    registry = Registry("feature")
    registry.register("returns")(namespace["Scaled"])
    return registry


def test_disk_cache_separates_versions_of_a_class_without_source(tmp_path, market_simple):
    # versions of an interactively defined class share a qualified name and
    # have no source text: the code they loaded must tell them apart
    spec = FeatureSpec.make("returns", window=1)
    calls = []

    def get(scaled):
        store = FeatureStore(registry=_sourceless_registry(scaled, calls), cache_dir=tmp_path)
        return store.get(spec, market_simple)

    one = get("1.0 *")
    two = get("2.0 *")  # another constant
    pd.testing.assert_frame_equal(two, 2.0 * one)
    shifted = get("1.0 +")  # same constants and names, another operation
    pd.testing.assert_frame_equal(shifted, 1.0 + one)
    assert len(calls) == 3 and len(list(tmp_path.iterdir())) == 3
    # the same code defined again is a hit
    pd.testing.assert_frame_equal(get("2.0 *"), two, check_freq=False)
    assert len(calls) == 3 and len(list(tmp_path.iterdir())) == 3


def test_disk_cache_keeps_stale_loaded_code_apart_from_its_edited_file(
    tmp_path, market_simple, monkeypatch
):
    # The source text is read from the file when a panel is requested. A
    # session that imported the feature BEFORE its file was edited still
    # runs the old code: that panel must not land under the key the edited
    # code will look up once it is loaded.
    import importlib.util
    import linecache
    import sys

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    path = tmp_path / "edited_feature.py"
    text = (
        "from alpha_lab.features.library import WindowReturn\n\n\n"
        "class Scaled(WindowReturn):\n"
        "    def compute(self, data):\n"
        "        return 1.0 * super().compute(data)\n"
    )

    def load():
        module_spec = importlib.util.spec_from_file_location("edited_feature", path)
        module = importlib.util.module_from_spec(module_spec)
        monkeypatch.setitem(sys.modules, "edited_feature", module)
        module_spec.loader.exec_module(module)
        registry = Registry("feature")
        registry.register("returns")(module.Scaled)
        return registry

    path.write_text(text)
    old_session = load()
    path.write_text(text.replace("1.0 *", "20.0 *"))  # edited on disk, not reloaded
    linecache.clearcache()
    spec = FeatureSpec.make("returns", window=1)
    cache = tmp_path / "cache"
    stale = FeatureStore(registry=old_session, cache_dir=cache).get(spec, market_simple)
    pd.testing.assert_frame_equal(stale, market_simple.returns())  # the old code ran

    fresh = FeatureStore(registry=load(), cache_dir=cache).get(spec, market_simple)
    pd.testing.assert_frame_equal(fresh, 20.0 * market_simple.returns())
    assert len(list(cache.iterdir())) == 2


def test_code_token_does_not_depend_on_the_hash_seed():
    # A set literal compiles to a frozenset constant whose repr order follows
    # PYTHONHASHSEED. The disk key must be the same in every process, or the
    # cache would never be hit again.
    import os
    import subprocess
    import sys
    from pathlib import Path

    from alpha_lab.features import store

    code = (
        "from alpha_lab.core.interfaces import Feature\n"
        "from alpha_lab.features.store import _code_token\n"
        "class Tagged(Feature):\n"
        "    name = 'tagged'\n"
        "    def compute(self, data):\n"
        "        return data.close if self.name in {'alpha', 'beta', 'gamma', 'delta'} else None\n"
        "print(_code_token(Tagged()))\n"
    )
    package_root = str(Path(store.__file__).resolve().parents[2])
    tokens = []
    for seed in ("1", "2", "3", "4"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        tokens.append(proc.stdout.strip().splitlines()[-1])
    assert tokens[0].startswith("__main__.Tagged:") and len(set(tokens)) == 1, tokens


def test_fingerprint_distinguishes_seeds():
    data7 = make_market(n_assets=6, n_days=150, seed=7)
    data8 = make_market(n_assets=6, n_days=150, seed=8)
    data7_again = make_market(n_assets=6, n_days=150, seed=7)
    assert fingerprint(data7) != fingerprint(data8)
    assert fingerprint(data7) == fingerprint(data7_again)


def test_fingerprint_distinguishes_truncation(market_simple):
    t = market_simple.dates[300]
    assert fingerprint(market_simple) != fingerprint(market_simple.slice_until(t))


@pytest.mark.parametrize("field", ["close", "open", "high", "low", "volume", "unadjusted_close", "universe"])
def test_fingerprint_sees_historical_corrections_between_sampled_rows(market_simple, field):
    import copy

    corrected = copy.deepcopy(market_simple)
    panel = getattr(corrected, field)
    panel.iloc[13, 0] = not panel.iloc[13, 0] if field == "universe" else panel.iloc[13, 0] * 1.01
    assert fingerprint(corrected) != fingerprint(market_simple)


def test_fingerprint_separates_missing_from_zero(market_simple):
    # volume may legitimately be 0.0; a missing cell is a different dataset
    import copy

    zero, missing = copy.deepcopy(market_simple), copy.deepcopy(market_simple)
    zero.volume.iloc[13, 0] = 0.0
    missing.volume.iloc[13, 0] = np.nan
    assert fingerprint(zero) != fingerprint(missing)


def test_fingerprint_sees_ticker_labels(market_simple):
    fields = [f for f in _FINGERPRINT_FIELDS if getattr(market_simple, f) is not None]
    renamed = [f"X{i}" for i in range(market_simple.close.shape[1])]
    relabelled = MarketData(**{f: getattr(market_simple, f).set_axis(renamed, axis=1) for f in fields})
    assert fingerprint(relabelled) != fingerprint(market_simple)


def test_fingerprint_covers_every_market_data_field():
    # a field added to MarketData must be added to the fingerprint too
    import dataclasses

    assert sorted(_FINGERPRINT_FIELDS) == sorted(f.name for f in dataclasses.fields(MarketData))
    assert len(set(_FINGERPRINT_FIELDS)) == len(_FINGERPRINT_FIELDS)


def test_fingerprint_sees_interior_date_changes(market_simple):
    import copy

    revised = copy.deepcopy(market_simple)
    dates = list(revised.dates)
    dates[13] += pd.Timedelta(1, unit="h")
    for field in ("close", "open", "high", "low", "volume", "unadjusted_close", "universe"):
        getattr(revised, field).index = pd.DatetimeIndex(dates)
    assert fingerprint(revised) != fingerprint(market_simple)


def test_memory_and_disk_cache_recompute_after_interior_price_correction(tmp_path, market_simple):
    import copy

    corrected = copy.deepcopy(market_simple)
    corrected.close.iloc[13, 0] *= 1.1
    spec = FeatureSpec.make("returns", window=1)
    store = FeatureStore(cache_dir=tmp_path)
    original = store.get(spec, market_simple)
    memory = store.get(spec, corrected)
    disk = FeatureStore(cache_dir=tmp_path).get(spec, corrected)
    expected = corrected.returns()
    assert memory.iloc[13, 0] != original.iloc[13, 0]
    pd.testing.assert_frame_equal(memory, expected)
    pd.testing.assert_frame_equal(disk, expected, check_freq=False)


def test_fingerprint_sees_every_feature_input_field():
    """Regression: two MarketData sharing a close panel but differing in
    volume / unadjusted_close / universe (content OR presence) must not
    collide — a collision makes the store serve e.g. adv_dollars computed
    from the other dataset's volume."""
    dates = pd.bdate_range("2020-01-02", periods=60)
    close = pd.DataFrame({"A": 100.0, "B": 50.0}, index=dates)
    vol_a = pd.DataFrame({"A": 1e6, "B": 1e6}, index=dates)
    vol_b = pd.DataFrame({"A": 37.0, "B": 37.0}, index=dates)
    base = MarketData.from_frames(close.copy(), volume=vol_a)

    diff_volume = MarketData.from_frames(close.copy(), volume=vol_b)
    diff_unadj = MarketData.from_frames(close.copy(), volume=vol_a, unadjusted_close=close * 4.0)
    no_volume = MarketData.from_frames(close.copy())
    universe = close.notna()
    universe.iloc[:30, 0] = False
    diff_universe = MarketData.from_frames(close.copy(), volume=vol_a, universe=universe)

    fp = fingerprint(base)
    assert fp != fingerprint(diff_volume)
    assert fp != fingerprint(diff_unadj)
    assert fp != fingerprint(no_volume)
    assert fp != fingerprint(diff_universe)
    # and identical content still collides deliberately (that's the memo hit)
    assert fp == fingerprint(MarketData.from_frames(close.copy(), volume=vol_a.copy()))


def test_store_recomputes_when_only_volume_differs():
    """Regression: the store must not serve a stale adv_dollars panel for a
    dataset that shares close but has different volume."""
    dates = pd.bdate_range("2020-01-02", periods=60)
    close = pd.DataFrame({"A": 100.0, "B": 50.0}, index=dates)
    data_a = MarketData.from_frames(close.copy(), volume=pd.DataFrame({"A": 1e6, "B": 1e6}, index=dates))
    data_b = MarketData.from_frames(close.copy(), volume=pd.DataFrame({"A": 37.0, "B": 37.0}, index=dates))

    store = FeatureStore()
    spec = FeatureSpec.make("adv_dollars", window=10)
    panel_a = store.get(spec, data_a)
    panel_b = store.get(spec, data_b)
    assert panel_a is not panel_b
    assert panel_a.iloc[-1, 0] == pytest.approx(1e6 * 100.0)
    assert panel_b.iloc[-1, 0] == pytest.approx(37.0 * 100.0)
