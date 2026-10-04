"""Tests for the two command-line scripts (scripts/run_backtest.py and
scripts/make_report.py), run in-process on small synthetic panels under
tmp_path, plus two checks of what the scripts depend on: the library's data
package must not be git-ignored, and the published example report must match
what the default configuration produces."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from alpha_lab.core.errors import AlphaLabError, ConfigError, ExperimentError
from alpha_lab.core.results import BacktestResult
from alpha_lab.data.synthetic import make_market
from alpha_lab.risk import metrics as m

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_script("run_backtest")


@pytest.fixture(scope="module")
def reporter():
    return _load_script("make_report")


def _strict_loads(text: str):
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=reject)


RUN_CONFIG = """
experiment: {name: runner_test, runs_dir: runs}
data:
  source: synthetic
  synthetic: {n_assets: 8, n_days: 220, seed: 3, universe_churn: false}
features:
  - {name: realized_vol, params: {window: 21}}
signal: {name: xs_momentum, params: {window: 21, skip: 1, min_names: 4}}
portfolio: {quantile: 0.25, max_weight: 1.0}
costs: {model: realistic, portfolio_value: 250000}
backtest:
  execution_lag: 2
  walkforward: {train_days: 60, test_days: 30, purge_days: 2}
report: {formats: [html, md]}
"""


def _only_run(runs_dir: Path) -> Path:
    [run_dir] = [p for p in runs_dir.iterdir() if p.is_dir()]
    return run_dir


@pytest.fixture(scope="module")
def tracked_run(runner, tmp_path_factory):
    """One complete default run: (config directory, run directory, stdout)."""
    base = tmp_path_factory.mktemp("cli")
    (base / "run.yaml").write_text(RUN_CONFIG)
    with pytest.MonkeyPatch.context() as patch:
        lines = []
        patch.setattr("builtins.print", lambda *a, **k: lines.append((a, k)))
        code = runner.main(["--config", str(base / "run.yaml")])
    assert code == 0
    out = "\n".join(" ".join(map(str, a)) for a, k in lines if k.get("file") is None)
    return base, _only_run(base / "runs"), out


def _copy_run(tracked_run, tmp_path) -> Path:
    """A private copy of the module's run directory for tests that modify it."""
    _, run_dir, _ = tracked_run
    return Path(shutil.copytree(run_dir, tmp_path / "runs" / run_dir.name))


# --------------------------------------------------------------------------
# run_backtest.py helpers
# --------------------------------------------------------------------------

def test_parse_overrides_coerces_yaml_scalars(runner):
    assert runner.parse_overrides(
        ["a.b=1", "c=null", "d=2.5", "e=true", "f=text", " g =x", "h={window: 5}", "i=a=b"]
    ) == {"a.b": 1, "c": None, "d": 2.5, "e": True, "f": "text", "g": "x",
          "h": {"window": 5}, "i": "a=b"}
    assert runner.parse_overrides(["bad=[unclosed"]) == {"bad": "[unclosed"}
    for malformed in ("novalue", "=1"):
        with pytest.raises(SystemExit, match="KEY=VALUE"):
            runner.parse_overrides([malformed])


def test_format_metrics_table(runner):
    table = runner.format_metrics_table(
        {"sharpe_net": 1.23456, "n_days": 1003, "mode": "walkforward", "not_shown": 1.0,
         "dsr": float("nan"), "n_trials": 12}
    )
    assert table.splitlines() == [
        "  sharpe_net  1.2346",
        "  dsr         nan",
        "  n_trials    12",
        "  n_days      1003",
        "  mode        walkforward",
    ]


@pytest.mark.parametrize("text,value", [("1", 1), ("12", 12), ("12.0", 12), ("2.5", 2.5)])
def test_trial_count_argument(runner, text, value):
    parsed = runner.build_parser().parse_args(["--config", "x", "--n-trials", text]).n_trials
    assert parsed == value and type(parsed) is type(value)


@pytest.mark.parametrize("text", ["0", "-3", "0.5", "nan", "inf", "many"])
def test_trial_count_argument_rejects_unusable_values(runner, text, capsys):
    with pytest.raises(SystemExit) as exit_info:
        runner.build_parser().parse_args(["--config", "x", "--n-trials", text])
    assert exit_info.value.code == 2
    assert "expected a number >= 1" in capsys.readouterr().err


# --------------------------------------------------------------------------
# run_backtest.py end to end
# --------------------------------------------------------------------------

def test_runner_persists_config_metrics_result_and_report(tracked_run):
    base, run_dir, out = tracked_run
    assert run_dir.parent == base / "runs"          # runs_dir is relative to the config file
    assert f"run path: {run_dir}" in out
    assert out.index("run_id: ") < out.index("report [html]: ")  # run pointer comes first
    for line in ("market data OK", f"report [md]: {run_dir / 'report' / 'report.md'}", "metrics:"):
        assert line in out

    metrics = _strict_loads((run_dir / "metrics.json").read_text())
    assert metrics["mode"] == "walkforward" and metrics["n_windows"] >= 3
    # default: one trial, nothing to deflate by
    assert metrics["n_trials"] == 1 and metrics["dsr"] == metrics["psr"]

    meta = json.loads((run_dir / "result" / "meta.json").read_text())
    assert meta["meta"]["portfolio_value"] == 250000.0   # costs.portfolio_value reaches the engine
    assert meta["meta"]["execution_lag"] == 2
    assert meta["meta"]["validation"] == {"errors": 0, "warnings": 0, "forced": False}
    assert "never_traded" not in meta["meta"]
    assert meta["config"]["costs"]["portfolio_value"] == 250000.0
    assert meta["config"]["signal"]["params"] == {"window": 21, "skip": 1, "min_names": 4}
    assert meta["config"]["features"] == [{"name": "realized_vol", "params": {"window": 21}}]

    assert sorted(p.name for p in (run_dir / "report").iterdir()) == [
        "equity.png", "report.html", "report.md", "rolling_sharpe.png", "turnover_costs.png",
    ]
    assert (run_dir / "report" / "report.md").read_text().startswith("# runner_test\n")
    assert (run_dir / "config.yaml").exists() and (run_dir / "env.json").exists()
    assert len((base / "runs" / "registry.jsonl").read_text().splitlines()) == 1

    # the stored summary is the summary of the stored result
    loaded = BacktestResult.load(run_dir / "result")
    assert m.summary(loaded) == pytest.approx(metrics, nan_ok=True)


def test_stored_run_does_not_name_the_directory_it_was_made_in(tracked_run):
    base, run_dir, _ = tracked_run
    for stored in ("config.yaml", "result/meta.json", "metrics.json", "report/report.md"):
        assert str(base) not in (run_dir / stored).read_text(), stored
    meta = json.loads((run_dir / "result" / "meta.json").read_text())
    assert meta["config"]["experiment"]["runs_dir"] == ".."


def test_n_trials_deflates_the_sharpe_ratio(runner, tracked_run, tmp_path, capsys):
    base, default_run, _ = tracked_run
    code = runner.main(["--config", str(base / "run.yaml"), "--runs-dir", str(tmp_path / "out"),
                        "--no-report", "--n-trials", "12"])
    assert code == 0
    out = capsys.readouterr().out
    run_dir = _only_run(tmp_path / "out")                 # --runs-dir overrides the config
    assert not (run_dir / "report").exists()              # --no-report
    assert "report [" not in out
    metrics = json.loads((run_dir / "metrics.json").read_text())
    baseline = json.loads((default_run / "metrics.json").read_text())
    assert metrics["n_trials"] == 12
    assert metrics["psr"] == baseline["psr"] and metrics["sharpe_net"] == baseline["sharpe_net"]
    net = BacktestResult.load(run_dir / "result").net_returns.loc[metrics["start"]:]
    assert metrics["dsr"] == pytest.approx(m.dsr(net, 12), rel=1e-12)
    assert metrics["dsr"] < metrics["psr"]
    assert "  n_trials        12" in out and "  dsr " in out


def test_unknown_extra_feature_is_an_error(runner, tmp_path):
    config = tmp_path / "bad_feature.yaml"
    config.write_text(RUN_CONFIG.replace("realized_vol", "no_such_feature"))
    with pytest.raises(AlphaLabError, match="no_such_feature"):
        runner.main(["--config", str(config)])
    assert not (tmp_path / "runs").exists()


def test_validation_errors_stop_the_run_unless_forced(runner, tmp_path, capsys):
    # a tripled price is a >100% one-day move: a validation ERROR
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    market = make_market(n_assets=8, n_days=220, seed=3, split_asset=False, universe_churn=False)
    close = market.close.copy()
    close.iloc[100:, 0] *= 3.0
    close.to_csv(data_dir / "close.csv", index_label="date")
    config = tmp_path / "csv.yaml"
    config.write_text(
        RUN_CONFIG.replace(
            "source: synthetic\n  synthetic: {n_assets: 8, n_days: 220, seed: 3, universe_churn: false}",
            "source: csv\n  path: csv",
        ).replace("model: realistic", "model: fixed_bps").replace("[html, md]", "[md]")
    )

    assert runner.main(["--config", str(config)]) == 2
    captured = capsys.readouterr()
    assert "rerun with --force" in captured.err
    assert "run_id" not in captured.out
    assert not (tmp_path / "runs").exists()               # nothing persisted

    assert runner.main(["--config", str(config), "--force"]) == 0
    assert "--force given" in capsys.readouterr().err
    run_dir = _only_run(tmp_path / "runs")
    validation = json.loads((run_dir / "result" / "meta.json").read_text())["meta"]["validation"]
    assert validation["forced"] is True and validation["errors"] >= 1
    # a forced run is recognisable from its report, and from a rebuilt one
    report = (run_dir / "report" / "report.md").read_text()
    assert f"validation reported {validation['errors']} error(s) and the run was forced" in report
    # the csv location is stored relative to the run, not as an absolute path
    assert str(tmp_path) not in (run_dir / "config.yaml").read_text()


def test_run_that_never_trades_warns_and_stores_strict_json(runner, tracked_run, tmp_path, capsys):
    base, _, _ = tracked_run
    code = runner.main(["--config", str(base / "run.yaml"), "--runs-dir", str(tmp_path / "out"),
                        "--override", "signal.params={window: 5000}", "--override", "report.formats=[md]"])
    assert code == 0
    captured = capsys.readouterr()
    assert "warning: no position is held on any date" in captured.err
    run_dir = _only_run(tmp_path / "out")
    metrics = _strict_loads((run_dir / "metrics.json").read_text())   # NaN used to be written bare
    assert metrics["sharpe_net"] is None and metrics["turnover_ann"] == 0.0
    [line] = (tmp_path / "out" / "registry.jsonl").read_text().splitlines()
    assert _strict_loads(line)["sharpe_net"] is None
    assert json.loads((run_dir / "result" / "meta.json").read_text())["meta"]["never_traded"] is True
    assert "No position is held on any date" in (run_dir / "report" / "report.md").read_text()


def test_normal_run_prints_no_warning(runner, tracked_run, tmp_path, capsys):
    base, _, _ = tracked_run
    assert runner.main(["--config", str(base / "run.yaml"), "--runs-dir", str(tmp_path / "out"),
                        "--no-report"]) == 0
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------
# make_report.py
# --------------------------------------------------------------------------

def test_make_report_rebuilds_the_same_report(reporter, tracked_run, tmp_path, capsys):
    run_dir = _copy_run(tracked_run, tmp_path)
    original = (run_dir / "report" / "report.md").read_text()
    shutil.rmtree(run_dir / "report")

    assert reporter.main(["--run", str(run_dir)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""                              # stored metrics agree: no note
    assert f"report [html]: {run_dir / 'report' / 'report.html'}" in captured.out
    assert (run_dir / "report" / "report.md").read_text() == original


def test_make_report_recomputes_stale_metrics_and_says_so(reporter, tracked_run, tmp_path, capsys):
    """A run logged by older code can hold metrics for a different period than
    the charts the current code draws; the table must describe the charts."""
    run_dir = _copy_run(tracked_run, tmp_path)
    stored = json.loads((run_dir / "metrics.json").read_text())
    stale = dict(stored, n_days=220, hit_rate=0.123456, old_key=1.0)
    del stale["calmar"]
    (run_dir / "metrics.json").write_text(json.dumps(stale))

    assert reporter.main(["--run", str(run_dir), "--formats", "md"]) == 0
    captured = capsys.readouterr()
    assert "differ from metrics.json (hit_rate, n_days, old_key, calmar)" in captured.err
    report = (run_dir / "report" / "report.md").read_text()
    assert f"| n_days | {stored['n_days']} |" in report and "0.1235" not in report
    assert "old_key" not in report and "| calmar |" in report
    assert json.loads((run_dir / "metrics.json").read_text()) == stale   # the record is untouched
    # --formats md: the html of the earlier build is not left next to the new md
    assert not (run_dir / "report" / "report.html").exists()

    assert reporter.main(["--run", str(run_dir), "--formats", "md", "--keep-stored-metrics"]) == 0
    assert capsys.readouterr().err == ""
    kept = (run_dir / "report" / "report.md").read_text()
    assert "| n_days | 220 |" in kept and "| hit_rate | 0.1235 |" in kept and "| old_key | 1 |" in kept


def test_make_report_keeps_the_trial_count_of_the_run(runner, reporter, tracked_run, tmp_path, capsys):
    base, _, _ = tracked_run
    runner.main(["--config", str(base / "run.yaml"), "--runs-dir", str(tmp_path / "out"),
                 "--no-report", "--n-trials", "12"])
    run_dir = _only_run(tmp_path / "out")
    capsys.readouterr()
    assert reporter.main(["--run", str(run_dir), "--formats", "md"]) == 0
    assert capsys.readouterr().err == ""                   # recomputed with n_trials 12: no change
    assert "| n_trials | 12 |" in (run_dir / "report" / "report.md").read_text()


def test_make_report_accepts_a_run_without_metrics_or_config(reporter, tracked_run, tmp_path, capsys):
    run_dir = _copy_run(tracked_run, tmp_path)
    (run_dir / "metrics.json").unlink()
    (run_dir / "config.yaml").unlink()
    assert reporter.main(["--run", str(run_dir)]) == 0
    report = (run_dir / "report" / "report.md").read_text()
    assert report.startswith(f"# {run_dir.name}\n") and "| sharpe_net |" in report
    with pytest.raises(ExperimentError, match="no metrics.json"):
        reporter.main(["--run", str(run_dir), "--keep-stored-metrics"])


@pytest.mark.parametrize(
    "name,content,message",
    [
        ("metrics.json", "{not json", "is unreadable"),
        ("metrics.json", "[1, 2]", "metrics.json does not hold a mapping"),
        ("config.yaml", "- a\n- b\n", "config.yaml does not hold a mapping"),
        ("config.yaml", "a: [unclosed", "is unreadable"),
        ("result/series.csv", "date,gross\n2020-01-02,0.1\n", "is unreadable"),
    ],
)
def test_make_report_reports_a_corrupt_run_as_an_error(reporter, tracked_run, tmp_path, name, content, message):
    run_dir = _copy_run(tracked_run, tmp_path)
    (run_dir / name).write_text(content)
    # json/yaml/pandas exceptions used to escape the scripts' error handler
    with pytest.raises(ExperimentError, match=message):
        reporter.main(["--run", str(run_dir)])


def test_make_report_tolerates_odd_config_sections(reporter, tracked_run, tmp_path):
    run_dir = _copy_run(tracked_run, tmp_path)
    (run_dir / "config.yaml").write_text("report: [html]\nexperiment: named\n")
    assert reporter.main(["--run", str(run_dir)]) == 0     # falls back to html + md, directory name
    assert (run_dir / "report" / "report.html").exists()


def test_make_report_missing_run_and_missing_result(reporter, tmp_path):
    with pytest.raises(ConfigError, match="no run directory"):
        reporter.main(["--run", str(tmp_path / "nope")])
    with pytest.raises(AlphaLabError, match="no backtest result"):
        reporter.main(["--run", str(tmp_path)])


def test_scripts_exit_1_with_a_one_line_error(tmp_path):
    """The __main__ guards turn library errors into `error: ...` and status 1."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for script, args, text in (
        ("run_backtest.py", ["--config", str(tmp_path / "missing.yaml")], "error: config file not found"),
        ("make_report.py", ["--run", str(tmp_path / "missing")], "error: no run directory"),
    ):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 1, proc.stderr
        assert proc.stderr.startswith(text) and "Traceback" not in proc.stderr


def test_scripts_report_a_file_system_error_in_one_line(tracked_run, tmp_path):
    """A runs or report directory that cannot be created used to end in a
    traceback; it is reported like a configuration error."""
    base, _, _ = tracked_run
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    run_dir = _copy_run(tracked_run, tmp_path)
    shutil.rmtree(run_dir / "report")
    (run_dir / "report").write_text("a file where the report directory is needed")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for script, args, path in (
        ("run_backtest.py", ["--config", str(base / "run.yaml"), "--runs-dir", str(blocker / "runs")], blocker),
        ("make_report.py", ["--run", str(run_dir)], run_dir / "report"),
    ):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        [line] = [text for text in proc.stderr.splitlines() if text.startswith("error: ")]
        assert line == proc.stderr.splitlines()[-1] and path.name in line
        assert "run_id" not in proc.stdout and "report [" not in proc.stdout
        assert path.read_text().startswith("a file where")     # nothing was replaced


# --------------------------------------------------------------------------
# what the scripts depend on
# --------------------------------------------------------------------------

def test_library_data_package_is_not_git_ignored():
    """An unanchored `data/` ignore pattern (meant for local data files) also
    matches the library package alpha_lab/data, so a clone cannot import it."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(PROJECT), "check-ignore", "-q", "alpha_lab/data/__init__.py"],
            capture_output=True,
        )
    except OSError:
        pytest.skip("git is not installed")
    if proc.returncode not in (0, 1):
        pytest.skip("not inside a git work tree")
    assert proc.returncode == 1, "alpha_lab/data/ (library code) is git-ignored"
    # the local-data and run-output folders next to the package stay ignored
    for local in ("data/prices.csv", "runs/registry.jsonl"):
        ignored = subprocess.run(["git", "-C", str(PROJECT), "check-ignore", "-q", local])
        assert ignored.returncode == 0, local


def _markdown_tables(text: str) -> dict[str, list[list[str]]]:
    """{section heading: table rows} of a tearsheet in Markdown."""
    tables: dict[str, list[list[str]]] = {}
    heading = ""
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("# ")
        elif line.startswith("|") and not line.startswith("| ---"):
            tables.setdefault(heading, []).append([cell.strip() for cell in line.strip("|").split("|")])
    return tables


def _published_example() -> Path | None:
    for root in (PROJECT.parent, PROJECT.parent / "public"):
        path = root / "examples" / "synthetic_report.md"
        if path.exists():
            return path
    return None


def test_published_example_matches_the_default_configuration(runner, tmp_path):
    """The sample report shown in the repository must be what the default
    configuration produces with the current code."""
    example = _published_example()
    if example is None:
        pytest.skip("no published example next to this project")
    assert runner.main(["--config", str(PROJECT / "configs" / "base.yaml"),
                        "--runs-dir", str(tmp_path / "runs"), "--override", "report.formats=[md]"]) == 0
    fresh_text = (_only_run(tmp_path / "runs") / "report" / "report.md").read_text(encoding="utf-8")
    published_text = example.read_text(encoding="utf-8")
    fresh, published = _markdown_tables(fresh_text), _markdown_tables(published_text)

    assert fresh_text.splitlines()[:3] == published_text.splitlines()[:3]      # title and period note
    assert list(fresh) == list(published)
    assert fresh["Walk-forward windows"] == published["Walk-forward windows"]
    for section, tolerance in (("Key metrics", 2e-3), ("Monthly net returns", None)):
        assert len(fresh[section]) == len(published[section])
        for new, old in zip(fresh[section], published[section]):
            assert len(new) == len(old) and new[0] == old[0], (section, new, old)
            for a, b in zip(new[1:], old[1:]):
                if a == b:
                    continue
                # allow a last-digit rounding difference between platforms, nothing more
                x, y = float(a.rstrip("%")), float(b.rstrip("%"))
                assert a.endswith("%") == b.endswith("%") and "." in a + b, (section, new[0], a, b)
                limit = tolerance * abs(y) if tolerance else 0.011
                assert abs(x - y) <= limit, (section, new[0], a, b)
