"""Tests for alpha_lab.experiments.tracker (all IO under tmp_path)."""

import json
import re
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import alpha_lab.experiments.tracker as tracker_mod
from alpha_lab.config.loader import config_from_dict, config_hash, load_config
from alpha_lab.core.errors import ExperimentError
from alpha_lab.core.results import BacktestResult, WalkForwardWindow
from alpha_lab.experiments.tracker import ExperimentTracker, slug
from alpha_lab.risk.metrics import summary

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


def _strict_loads(text: str):
    """json.loads that rejects the non-standard NaN / Infinity tokens."""
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=reject)


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
    assert set(env) >= {"python", "numpy", "pandas", "matplotlib", "yaml", "platform", "alpha_lab"}
    assert set(env) <= {"python", "numpy", "pandas", "matplotlib", "yaml", "platform", "alpha_lab",
                        "git_revision"}
    if "git_revision" in env:  # best effort: absent outside a git checkout
        assert re.fullmatch(r"[0-9a-f]{40,64}(\+dirty)?", env["git_revision"])

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


def test_every_field_of_a_result_survives_a_logged_run(tmp_path):
    """Holdings differ from targets and the result carries scores, windows,
    config and meta: a report rebuilt from disk must match the original."""
    idx = pd.bdate_range("2021-01-04", periods=40)
    rng = np.random.default_rng(1)
    target = pd.DataFrame(rng.normal(0.0, 0.1, (40, 2)), index=idx, columns=["A", "B"])
    holdings = target.shift(2).fillna(0.0)
    gross = pd.Series(rng.normal(0.001, 0.01, 40), index=idx)
    costs = pd.Series(0.0002, index=idx)
    result = BacktestResult(
        gross_returns=gross, costs=costs, net_returns=gross - costs,
        turnover=pd.Series(0.1, index=idx), holdings=holdings, target_weights=target,
        scores=target * 2.0, windows=[WalkForwardWindow(idx[0], idx[9], idx[12], idx[-1])],
        config={"backtest": {"execution_lag": 2}},
        meta={"mode": "walkforward", "execution_lag": 2, "portfolio_value": 250000.0},
    )
    tracker = ExperimentTracker(tmp_path)
    rec = tracker.log_run(config_from_dict({}), result, summary(result))
    loaded = tracker.load_run(rec.run_id)["result"]

    kw = dict(check_names=False, check_freq=False)
    pd.testing.assert_frame_equal(loaded.holdings, holdings, **kw)
    pd.testing.assert_frame_equal(loaded.target_weights, target, **kw)
    assert not loaded.holdings.equals(loaded.target_weights)
    pd.testing.assert_frame_equal(loaded.scores, target * 2.0, **kw)
    for name in ("gross_returns", "costs", "net_returns", "turnover"):
        pd.testing.assert_series_equal(getattr(loaded, name), getattr(result, name), **kw)
    assert loaded.windows == result.windows
    assert loaded.meta == result.meta and loaded.config == result.config
    # the trimmed evaluation period is rebuilt from the stored windows and mode
    assert summary(loaded) == pytest.approx(summary(result), nan_ok=True)
    assert summary(loaded)["n_days"] == 40 - 12
    assert tracker.load_run(rec.run_id)["metrics"]["n_days"] == 28


def test_non_finite_metrics_are_stored_as_strict_json(tmp_path):
    metrics = {
        "sharpe_net": float("nan"),
        "ann_return_net": np.float64("inf"),
        "max_drawdown": -np.inf,
        "n_days": np.int64(30),
        "calmar": np.float32("nan"),
        "nested": {"values": [1.0, float("nan"), np.float64("nan")], "array": np.array([np.nan, 2.0])},
        "finite": 0.25,
    }
    tracker = ExperimentTracker(tmp_path)
    rec = tracker.log_run(config_from_dict({}), _small_result(), metrics, name="flat")

    raw = (rec.path / "metrics.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    stored = _strict_loads(raw)  # used to hold the bare token NaN
    assert stored == {
        "sharpe_net": None, "ann_return_net": None, "max_drawdown": None, "n_days": 30,
        "calmar": None, "nested": {"values": [1.0, None, None], "array": [None, 2.0]},
        "finite": 0.25,
    }
    [line] = (tmp_path / "registry.jsonl").read_text().splitlines()
    index = _strict_loads(line)
    assert index["sharpe_net"] is None and index["max_drawdown"] is None and index["n_days"] == 30

    # the caller's dict is not rewritten, and readers see "missing"
    assert np.isnan(rec.metrics["sharpe_net"])
    assert tracker.list_runs()["sharpe_net"].isna().all()
    assert tracker.compare([rec.run_id], keys=("sharpe_net", "finite")).isna().iloc[0].tolist() == [True, False]
    assert tracker.load_run(rec.run_id)["metrics"]["sharpe_net"] is None


def test_finite_metrics_are_stored_verbatim(tmp_path):
    rec = ExperimentTracker(tmp_path).log_run(config_from_dict({}), _small_result(), METRICS)
    assert (rec.path / "metrics.json").read_text() == json.dumps(METRICS, indent=2)


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


def test_run_logged_after_a_truncated_registry_line_is_not_lost(tmp_path):
    """An interrupted write leaves a partial last line WITHOUT a newline; the
    next record used to be glued onto it and discarded with it."""
    cfg = config_from_dict({})
    tracker = ExperimentTracker(tmp_path)
    tracker.log_run(cfg, _small_result(1), {"sharpe_net": 0.5}, name="first")
    with (tmp_path / "registry.jsonl").open("a") as fh:
        fh.write('{"run_id": "2026-crashed", "ts": "2026')

    tracker.log_run(cfg, _small_result(2), {"sharpe_net": 0.7}, name="second")
    tracker.log_run(cfg, _small_result(3), {"sharpe_net": 0.9}, name="third")

    with pytest.warns(UserWarning, match="corrupt line 2"):
        frame = tracker.list_runs()
    assert list(frame["name"]) == ["first", "second", "third"]
    # no blank line is inserted when the file already ends with a newline
    assert (tmp_path / "registry.jsonl").read_text().count("\n") == 4


def test_failed_save_leaves_no_run_directory_and_no_registry_line(tmp_path, monkeypatch):
    cfg = config_from_dict({})
    tracker = ExperimentTracker(tmp_path / "runs")

    def disk_full(self, path):
        raise OSError("no space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(BacktestResult, "save", disk_full)
        with pytest.raises(OSError, match="no space left"):
            tracker.log_run(cfg, _small_result(), METRICS, name="doomed")
    # used to leave config.yaml, metrics.json and env.json behind
    assert [p.name for p in (tmp_path / "runs").iterdir()] == []
    assert tracker.list_runs().empty

    good = tracker.log_run(cfg, _small_result(), METRICS, name="fine")
    assert list(tracker.list_runs()["run_id"]) == [good.run_id]


def test_stored_config_holds_no_absolute_path(tmp_path):
    """load_config resolves locations; stored copies must not carry the
    user's directories, yet must load back to the same experiment."""
    project = tmp_path / "home" / "someone" / "project"
    (project / "configs").mkdir(parents=True)
    cfg_file = project / "configs" / "run.yaml"
    cfg_file.write_text(
        "experiment: {name: paths, runs_dir: ../runs, feature_cache_dir: ../cache}\n"
        "data: {source: csv, path: ../panel}\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.experiment.runs_dir == str((project / "runs").resolve())  # absolute in memory
    result = _small_result()
    result.config = cfg.to_dict()

    rec = ExperimentTracker(cfg.experiment.runs_dir).log_run(cfg, result, METRICS)

    for stored in ("config.yaml", "result/meta.json"):
        text = (rec.path / stored).read_text()
        assert str(tmp_path) not in text and "someone" not in text, stored
    loaded = ExperimentTracker(cfg.experiment.runs_dir).load_run(rec.run_id)
    assert loaded["config"]["experiment"] == {
        "name": "paths", "runs_dir": "..", "feature_cache_dir": "../../cache",
    }
    assert loaded["config"]["data"]["path"] == "../../panel"
    assert loaded["result"].config["experiment"]["runs_dir"] == ".."
    # the caller's objects keep their resolved paths
    assert result.config["experiment"]["runs_dir"] == cfg.experiment.runs_dir
    # reading the stored file resolves to the same locations and identity
    again = load_config(rec.path / "config.yaml")
    assert again.to_dict() == cfg.to_dict()
    assert config_hash(again) == rec.config_hash


def test_relative_and_unset_locations_are_stored_as_given(tmp_path):
    cfg = config_from_dict({"experiment": {"runs_dir": "my_runs"}})
    rec = ExperimentTracker(tmp_path).log_run(cfg, _small_result(), METRICS)
    stored = ExperimentTracker(tmp_path).load_run(rec.run_id)["config"]
    assert stored["experiment"]["runs_dir"] == "my_runs"
    assert stored["experiment"]["feature_cache_dir"] is None and stored["data"]["path"] is None
    assert stored == cfg.to_dict()


def test_git_revision_identifies_the_checkout(tmp_path):
    # a directory outside any work tree has no revision
    plain = tmp_path / "plain"
    plain.mkdir()
    assert tracker_mod._git_revision(plain) is None
    assert tracker_mod._git_revision(tmp_path / "missing") is None
    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    def git(*args):
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
             "-c", "commit.gpgsign=false", *args],
            check=True, capture_output=True,
        )

    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n")
    git("init", "-q")
    # inside a work tree, but the package itself is not tracked there
    assert tracker_mod._git_revision(package) is None
    git("add", "pkg/__init__.py")
    git("commit", "-q", "-m", "one")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert tracker_mod._git_revision(package) == head
    (package / "__init__.py").write_text("x = 2\n")
    assert tracker_mod._git_revision(package) == head + "+dirty"


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
