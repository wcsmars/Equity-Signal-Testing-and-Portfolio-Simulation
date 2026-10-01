"""Tests for the core contracts: feature-spec keys, result persistence."""

import json

import numpy as np
import pandas as pd
import pytest

from alpha_lab.core.errors import DataError
from alpha_lab.core.interfaces import FeatureSpec
from alpha_lab.core.results import BacktestResult, WalkForwardWindow


# -- FeatureSpec keys --------------------------------------------------------


@pytest.mark.parametrize(
    "params, key",
    [
        ({}, "momentum()"),
        ({"window": 252, "skip": 21}, "momentum(skip=21,window=252)"),
        ({"window": 63}, "momentum(window=63)"),
        ({"alpha": 0.5, "label": "x", "flag": True, "none": None},
         "momentum(alpha=0.5,flag=True,label='x',none=None)"),
        ({"windows": [5, 21], "nested": {"b": 2, "a": [1.5]}},
         "momentum(nested=(('a', (1.5,)), ('b', 2)),windows=(5, 21))"),
    ],
)
def test_feature_spec_key_for_plain_python_params(params, key):
    # the key names the panel in the feature set and keys the disk cache, so
    # these strings must not drift
    spec = FeatureSpec.make("momentum", **params)
    assert spec.key == key
    assert hash(spec) == hash(FeatureSpec.make("momentum", **params))


@pytest.mark.parametrize(
    "numpy_params, python_params",
    [
        ({"window": np.int64(252), "skip": np.int32(21)}, {"window": 252, "skip": 21}),
        ({"alpha": np.float64(0.5)}, {"alpha": 0.5}),
        ({"alpha": np.float32(0.5)}, {"alpha": 0.5}),
        ({"flag": np.bool_(True)}, {"flag": True}),
        ({"label": np.str_("x")}, {"label": "x"}),
        ({"windows": [np.int64(5), np.int64(21)]}, {"windows": [5, 21]}),
        ({"nested": {"a": np.int64(1), "b": (np.float64(2.5),)}}, {"nested": {"a": 1, "b": (2.5,)}}),
        ({"window": np.arange(5, 6)[0]}, {"window": 5}),
    ],
)
def test_feature_spec_key_is_the_same_for_numpy_scalars(numpy_params, python_params):
    # repr(np.int64(5)) is '5' on NumPy 1 and 'np.int64(5)' on NumPy 2; a key
    # built from it would differ between the two and from the key a signal
    # builds after coercing its own parameters to Python numbers
    from_numpy = FeatureSpec.make("returns", **numpy_params)
    from_python = FeatureSpec.make("returns", **python_params)
    assert from_numpy.key == from_python.key
    assert "np." not in from_numpy.key and "numpy" not in from_numpy.key
    assert from_numpy == from_python and hash(from_numpy) == hash(from_python)
    for value in from_numpy.as_dict.values():
        assert not isinstance(value, np.generic)


def test_numpy_integer_spec_is_found_by_a_signal_lookup():
    from alpha_lab.data.synthetic import make_market
    from alpha_lab.features import FeatureStore
    from alpha_lab.signals.momentum import CrossSectionalMomentum

    data = make_market(n_assets=6, n_days=120, seed=5, split_asset=False, universe_churn=False)
    specs = [FeatureSpec.make("momentum", window=window, skip=np.int64(5)) for window in np.arange(21, 64, 42)]
    features = FeatureStore().compute_all(specs, data)
    assert list(features) == ["momentum(skip=5,window=21)", "momentum(skip=5,window=63)"]
    signal = CrossSectionalMomentum(window=np.int64(63), skip=np.int64(5))
    assert signal.score(features, data).shape == data.close.shape


# -- BacktestResult persistence ------------------------------------------------


def _result(n_days=400, scores=True):
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2015-01-02", periods=n_days)

    def series():
        return pd.Series(rng.normal(0.0, 0.01, n_days), index=index)

    def panel():
        return pd.DataFrame(rng.normal(0.0, 0.1, (n_days, 4)), index=index, columns=list("ABCD"))

    return BacktestResult(
        gross_returns=series(),
        costs=series().abs(),
        net_returns=series(),
        turnover=series().abs(),
        holdings=panel(),
        target_weights=panel(),
        scores=panel() if scores else None,
        windows=[
            WalkForwardWindow(index[0], index[99], index[105], index[199]),
            WalkForwardWindow(index[100], index[199], index[205], index[299]),
        ],
        config={"experiment": {"name": "run"}},
        meta={"mode": "walkforward", "execution_lag": 2},
    )


def test_backtest_result_round_trip_is_exact(tmp_path):
    # the default CSV float parser is off by one unit in the last place on a
    # large share of 17-digit values; metrics recomputed from a reloaded run
    # then differ from the stored ones
    result = _result()
    loaded = BacktestResult.load(result.save(tmp_path / "result"))
    for name in ("gross_returns", "costs", "net_returns", "turnover"):
        saved, back = getattr(result, name), getattr(loaded, name)
        assert (back.to_numpy() == saved.to_numpy()).all(), name
        assert back.index.equals(saved.index)
    for name in ("holdings", "target_weights", "scores"):
        saved, back = getattr(result, name), getattr(loaded, name)
        assert (back.to_numpy() == saved.to_numpy()).all(), name
        assert back.index.equals(saved.index) and list(back.columns) == list(saved.columns)
    assert loaded.windows == result.windows
    assert loaded.config == result.config and loaded.meta == result.meta
    assert (loaded.equity_curve().to_numpy() == result.equity_curve().to_numpy()).all()


def test_saving_without_scores_removes_a_stale_scores_file(tmp_path):
    out = _result(scores=True).save(tmp_path / "result")
    assert (out / "scores.csv").exists()
    _result(scores=False).save(out)
    assert not (out / "scores.csv").exists()
    assert BacktestResult.load(out).scores is None
    # and the first save into a fresh directory needs no file to remove
    fresh = _result(scores=False).save(tmp_path / "fresh")
    assert BacktestResult.load(fresh).scores is None


def test_backtest_result_load_needs_a_saved_result(tmp_path):
    with pytest.raises(DataError, match="no backtest result"):
        BacktestResult.load(tmp_path)
    out = _result(n_days=300).save(tmp_path / "result")
    meta = json.loads((out / "meta.json").read_text())
    assert [WalkForwardWindow.from_dict(w) for w in meta["windows"]] == _result(n_days=300).windows
    (out / "meta.json").unlink()
    bare = BacktestResult.load(out)
    assert bare.windows is None and bare.config is None and bare.meta == {}
