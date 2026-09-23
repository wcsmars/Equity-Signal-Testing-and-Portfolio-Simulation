"""Feature library and FeatureStore tests.

Covers exact math on the clean panel, truncation invariance for every
registered feature on the messy panel (split + universe churn), NaN
discipline, RSI bounds, and store memoization / disk cache / fingerprint.
"""

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
from alpha_lab.features.store import FeatureStore, fingerprint

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


def test_fingerprint_distinguishes_seeds():
    data7 = make_market(n_assets=6, n_days=150, seed=7)
    data8 = make_market(n_assets=6, n_days=150, seed=8)
    data7_again = make_market(n_assets=6, n_days=150, seed=7)
    assert fingerprint(data7) != fingerprint(data8)
    assert fingerprint(data7) == fingerprint(data7_again)


def test_fingerprint_distinguishes_truncation(market_simple):
    t = market_simple.dates[300]
    assert fingerprint(market_simple) != fingerprint(market_simple.slice_until(t))


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
