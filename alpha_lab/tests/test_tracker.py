"""Tests for alpha_lab.experiments.tracker (all IO under tmp_path)."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import alpha_lab.experiments.tracker as tracker_mod
from alpha_lab.config.loader import config_from_dict, config_hash
from alpha_lab.core.errors import ExperimentError
from alpha_lab.core.results import BacktestResult
from alpha_lab.experiments.tracker import ExperimentTracker, slug

METRICS = {
    "sharpe_net": 1.23,
    "ann_return_net": 0.10,
    "max_drawdown": -0.05,
    "n_days": 30,
    "extra_detail": "kept only in metrics.json",
}


def _small_result(seed: int = 0) -> BacktestResult:
    """A tiny but structurally complete BacktestResult from seeded series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=30)
    tickers = ["AAA", "BBB"]
    gross = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
    costs = pd.Series(rng.uniform(0.0, 0.0005, len(dates)), index=dates)
    weights = pd.DataFrame(
        rng.uniform(-0.5, 0.5, (len(dates), len(tickers))), index=dates, columns=tickers
    )
    return BacktestResult(
        gross_returns=gross,
        costs=costs,
        net_returns=gross - costs,
        turnover=weights.diff().abs().sum(axis=1),
        holdings=weights.shift(1).fillna(0.0),
        target_weights=weights,
    )


def test_slug():
    assert slug("My Run! v2") == "my-run-v2"
    assert slug("ok-name_9") == "ok-name_9"
    assert slug("") == "run"


def test_log_run_round_trip(tmp_path):
    cfg = config_from_dict({})
    result = _small_result()
    tracker = ExperimentTracker(tmp_path / "runs")

    rec = tracker.log_run(cfg, result, METRICS, name="My Run!")

    assert rec.config_hash == config_hash(cfg)
    assert rec.run_id.endswith("_my-run-")  # slugged name is the id suffix
    assert rec.metrics == METRICS
    for artifact in ("config.yaml", "metrics.json", "env.json"):
        assert (rec.path / artifact).exists()
    env = json.loads((rec.path / "env.json").read_text())
    assert set(env) >= {"python", "numpy", "pandas", "platform", "alpha_lab"}

    loaded = tracker.load_run(rec.run_id)
    assert loaded["path"] == rec.path
    assert loaded["metrics"] == METRICS
    assert loaded["config"]["experiment"]["name"] == "run"
    pd.testing.assert_series_equal(
        loaded["result"].net_returns,
        result.net_returns,
        check_names=False,
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        loaded["result"].target_weights,
        result.target_weights,
        check_names=False,
        check_freq=False,
    )


def test_two_runs_registry_and_list(tmp_path):
    cfg = config_from_dict({})
    tracker = ExperimentTracker(tmp_path)
    tracker.log_run(cfg, _small_result(1), {"sharpe_net": 1.0}, name="alpha")
    tracker.log_run(cfg, _small_result(2), {"sharpe_net": 2.0}, name="beta")

    lines = [l for l in (tmp_path / "registry.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 2

    frame = tracker.list_runs()
    assert len(frame) == 2
    assert set(frame["name"]) == {"alpha", "beta"}
    assert sorted(frame["sharpe_net"]) == [1.0, 2.0]
    # keys absent from metrics land as null in the index
    assert frame["max_drawdown"].isna().all()


def test_list_runs_no_registry_file(tmp_path):
    frame = ExperimentTracker(tmp_path / "empty").list_runs()
    assert frame.empty
    assert "run_id" in frame.columns and "sharpe_net" in frame.columns


def test_run_id_collision_gets_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker_mod, "_utcnow", lambda: datetime(2026, 1, 2, 3, 4, 5))
    cfg = config_from_dict({})
    tracker = ExperimentTracker(tmp_path)

    first = tracker.log_run(cfg, _small_result(1), {}, name="dup")
    second = tracker.log_run(cfg, _small_result(2), {}, name="dup")
    third = tracker.log_run(cfg, _small_result(3), {}, name="dup")

    expected_base = "20260102-030405_" + config_hash(cfg)[:8] + "_dup"
    assert first.run_id == expected_base
    assert second.run_id == expected_base + "-2"
    assert third.run_id == expected_base + "-3"
    # all three are independently loadable
    assert tracker.load_run(third.run_id)["path"] == third.path


def test_corrupt_registry_line_is_skipped(tmp_path):
    cfg = config_from_dict({})
    tracker = ExperimentTracker(tmp_path)
    good = tracker.log_run(cfg, _small_result(1), {"sharpe_net": 0.5}, name="good")
    with (tmp_path / "registry.jsonl").open("a") as fh:
        fh.write("{not valid json at all\n")

    with pytest.warns(UserWarning, match="corrupt"):
        frame = tracker.list_runs()
    assert list(frame["run_id"]) == [good.run_id]


def test_compare_pulls_from_metrics_json(tmp_path):
    cfg = config_from_dict({})
    tracker = ExperimentTracker(tmp_path)
    a = tracker.log_run(cfg, _small_result(1), {"sharpe_net": 1.0, "max_drawdown": -0.1}, name="a")
    b = tracker.log_run(cfg, _small_result(2), {"sharpe_net": 2.0, "ann_return_net": 0.2}, name="b")

    frame = tracker.compare([a.run_id, b.run_id], keys=("sharpe_net", "max_drawdown"))
    assert list(frame.columns) == ["sharpe_net", "max_drawdown"]
    assert list(frame.index) == [a.run_id, b.run_id]
    assert frame.loc[a.run_id, "sharpe_net"] == 1.0
    assert frame.loc[a.run_id, "max_drawdown"] == -0.1
    assert frame.loc[b.run_id, "sharpe_net"] == 2.0
    assert pd.isna(frame.loc[b.run_id, "max_drawdown"])

    with pytest.raises(ExperimentError):
        tracker.compare([a.run_id, "no-such-run"])


def test_load_run_missing_raises(tmp_path):
    with pytest.raises(ExperimentError):
        ExperimentTracker(tmp_path).load_run("20990101-000000_deadbeef_ghost")
