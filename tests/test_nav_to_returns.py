"""NAV log -> monitor returns file: the arithmetic, every refusal, the
append-only rule, the exit statuses and the hand-off to scripts/monitor.py.

Made-up NAV logs only; no data cache and no saved result is read.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nav_to_returns.py"
HEADER = "date,nav,flow"
# no flow, a deposit and a withdrawal, each a +1% / +1% / -1% session by construction
LOG = ["2024-01-02,100000,", "2024-01-03,101000,", "2024-01-04,107060,5000",
       "2024-01-05,99000,-7060"]


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def nav():
    return module("scripts/nav_to_returns.py", "nav_to_returns_test")


def _log(tmp_path, rows, header=HEADER, name="nav.csv"):
    path = tmp_path / name
    path.write_text(header + "\n" + "".join(row + "\n" for row in rows))
    return path


def _run(nav, capsys, *argv):
    code = nav.main([str(arg) for arg in argv])
    return code, capsys.readouterr().out


def _returns(path):
    frame = pd.read_csv(path, index_col=0)
    return {day: value for day, value in frame["ENSEMBLE"].items()}


# ------------------------------------------------------------- arithmetic
def test_returns_take_flows_from_the_start_of_the_session(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    code, shown = _run(nav, capsys, _log(tmp_path, LOG), "--out", out)
    assert code == 0, shown
    assert out.read_text().splitlines() == [
        "Date,ENSEMBLE",
        "2024-01-03,0.010000000000",     # no flow: 101,000 / 100,000 - 1
        "2024-01-04,0.010000000000",     # deposit: 107,060 / (101,000 + 5,000) - 1
        "2024-01-05,-0.010000000000",    # withdrawal: 99,000 / (107,060 - 7,060) - 1
    ]
    # an end-of-day convention would book the deposit as profit or loss: it gives another number
    assert (107060 - 5000) / 101000 - 1 == pytest.approx(0.0105, abs=1e-4)
    assert "1 deposit(s) $5,000.00, 1 withdrawal(s) $7,060.00" in shown
    assert "wrote" in shown and "3 row(s), 2024-01-03 .. 2024-01-05" in shown


def test_return_formula_by_hand_on_uneven_numbers(nav, tmp_path):
    log = nav.read_nav_log(_log(tmp_path, ["2024-01-02,250000.00,", "2024-01-03,251234.56,",
                                           "2024-01-04,262000.00,10000", "2024-01-05,240500.00,-20000"]))
    implied = nav.daily_returns(log)
    expected = [251234.56 / 250000 - 1, 262000 / 261234.56 - 1, 240500 / 242000 - 1]
    assert list(implied.index.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04", "2024-01-05"]
    assert implied.tolist() == pytest.approx(expected, abs=1e-15)


def test_a_flow_on_the_first_row_is_inside_the_starting_nav(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    rows = ["2024-01-02,100000,100000", *LOG[1:]]
    code, shown = _run(nav, capsys, _log(tmp_path, rows), "--out", out)
    assert code == 0 and "first-row flow 100,000.00 is part of the starting nav" in shown
    assert _returns(out)["2024-01-03"] == pytest.approx(0.01, abs=1e-12)


def test_columns_may_come_in_any_order_and_blank_lines_are_skipped(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    log = tmp_path / "nav.csv"
    log.write_text("flow, nav ,date\n,100000,2024-01-02\n\n,101000,2024-01-03\n")
    assert _run(nav, capsys, log, "--out", out)[0] == 0
    assert out.read_text().splitlines()[1] == "2024-01-03,0.010000000000"


def test_one_row_is_a_starting_point_and_writes_nothing(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    code, shown = _run(nav, capsys, _log(tmp_path, LOG[:1]), "--out", out)
    assert code == 0 and "no return yet" in shown and not out.exists()


# --------------------------------------------------------------- refusals
@pytest.mark.parametrize("header,rows,message", [
    ("date,nav", LOG, "header must be exactly date,nav,flow"),
    ("date,nav,flow,note", [row + "," for row in LOG], "header must be exactly date,nav,flow"),
    ("date,nav,nav", LOG, "header must be exactly date,nav,flow"),
    (HEADER, [], "NAV log has no rows"),
    (HEADER, ["2024-01-02,100000", "2024-01-03,101000,"], "line 2: 2 cells, expected 3"),
    (HEADER, ["01/02/2024,100000,", "01/03/2024,101000,"], "date '01/02/2024' is not written YYYY-MM-DD"),
    (HEADER, ["2024-1-2,100000,", "2024-01-03,101000,"], "date '2024-1-2' is not written YYYY-MM-DD"),
    (HEADER, ["2024-02-28,100000,", "2024-02-30,101000,"], "'2024-02-30' is not a calendar date"),
    (HEADER, ["2024-01-03,100000,", "2024-01-02,101000,"],
     "strictly increasing: 2024-01-02 follows 2024-01-03"),
    (HEADER, ["2024-01-02,100000,", "2024-01-02,101000,"], "strictly increasing"),
    (HEADER, ["2024-01-05,100000,", "2024-01-06,101000,"],
     "1 row(s) on days the NYSE was closed, first 2024-01-06"),          # a Saturday
    (HEADER, ["2024-01-12,100000,", "2024-01-15,101000,", "2024-01-16,101000,"],
     "first 2024-01-15"),                                               # Martin Luther King Jr. Day
    (HEADER, ["2024-01-02,100000,", "2024-01-03,100500,", "2024-01-08,101000,"],
     "missing 2 NYSE session(s) between its first and last row, first 2024-01-04"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,,"], "line 3: nav is blank"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,abc,"], "nav 'abc' is not a number"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,nan,"], "nav 'nan' is not finite"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,inf,"], "nav 'inf' is not finite"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,0,"], "nav 0 is not positive"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,-5,"], "nav -5 is not positive"),
    (HEADER, ["2024-01-02,100000,", '2024-01-03,"101,000",'], "nav '101,000' is not a number"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,101000,x"], "flow 'x' is not a number"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,101000,inf"], "flow 'inf' is not finite"),
    (HEADER, ["2024-01-02,100000,", "2024-01-03,10,-100000"],
     "on 2024-01-03 the previous nav plus the flow is 0.00, not positive"),
])
def test_an_invalid_nav_log_is_refused_and_nothing_is_written(nav, tmp_path, capsys, header, rows,
                                                               message):
    out = tmp_path / "returns.csv"
    code, shown = _run(nav, capsys, _log(tmp_path, rows, header), "--out", out)
    assert code == nav.EXIT_REFUSED == 3
    assert shown.startswith("nav_to_returns REFUSED, nothing written: ") and message in shown
    assert not out.exists()


def test_unusable_paths_and_flags_are_refused(nav, tmp_path, capsys):
    log = _log(tmp_path, LOG)
    for argv, message in [
        ([tmp_path / "absent.csv", "--out", tmp_path / "r.csv"], "NAV log not found"),
        ([log, "--out", log], "--out names the NAV log itself"),
        ([log, "--out", tmp_path / "no_folder" / "r.csv"], "output folder does not exist"),
        ([log, "--out", tmp_path], "--out is not a file"),
        ([log, "--out", tmp_path / "r.csv", "--reb"], "unrecognized arguments: --reb"),   # no abbreviations
        ([log, "--ou", tmp_path / "r.csv"], "unrecognized arguments"),
        ([], "the following arguments are required: nav_log"),
    ]:
        code, shown = _run(nav, capsys, *argv)
        assert code == 3 and message in shown, (argv, shown)
    assert log.read_text().startswith(HEADER)          # the log itself is never written
    assert sorted(p.name for p in tmp_path.iterdir()) == ["nav.csv"]


def test_the_default_output_is_the_monitors_live_returns_file(nav):
    assert nav.DEFAULT_OUT == ROOT / "state" / "live_returns.csv"
    assert nav.OUT_HEADER == ["Date", "ENSEMBLE"]


# ------------------------------------------------------------ append-only
def test_new_rows_are_appended_and_written_rows_never_change(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    assert _run(nav, capsys, _log(tmp_path, LOG[:2]), "--out", out)[0] == 0
    first = out.read_bytes()
    assert first.decode().splitlines() == ["Date,ENSEMBLE", "2024-01-03,0.010000000000"]
    code, shown = _run(nav, capsys, _log(tmp_path, LOG), "--out", out)
    assert code == 0 and "appended 2 row(s)" in shown and "2024-01-04 .. 2024-01-05" in shown
    after = out.read_bytes()
    assert after.startswith(first) and after.decode().count("\n") == 4
    # the same log again: nothing to append, nothing rewritten
    stamp = out.stat().st_mtime_ns
    code, shown = _run(nav, capsys, _log(tmp_path, LOG), "--out", out)
    assert code == 0 and "is up to date: 3 row(s), nothing to append" in shown
    assert out.read_bytes() == after and out.stat().st_mtime_ns == stamp


def test_rows_written_by_hand_are_kept_byte_for_byte_when_within_tolerance(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    out.write_bytes(b"Date,ENSEMBLE\r\n2024-01-03,0.01\r\n2024-01-04,0.0100000000001")  # no final newline
    code, shown = _run(nav, capsys, _log(tmp_path, LOG), "--out", out)
    assert code == 0 and "appended 1 row(s)" in shown
    assert out.read_bytes() == (b"Date,ENSEMBLE\r\n2024-01-03,0.01\r\n2024-01-04,0.0100000000001\r\n"
                                b"2024-01-05,-0.010000000000\r\n")


@pytest.mark.parametrize("existing,message", [
    ("Date,ENSEMBLE\n2024-01-03,0.010000000000\n2024-01-04,0.009000000000\n",
     "on 2024-01-04 it holds 0.009000000000, the NAV log implies 0.010000000000 (tolerance 1e-10)"),
    ("Date,ENSEMBLE\n2024-01-03,0.010000001\n", "on 2024-01-03 it holds"),
    ("Date,ENSEMBLE\n2024-01-04,0.010000000000\n",
     "its row 1 is dated 2024-01-04, where the NAV log implies 2024-01-03"),
    ("Date,ENSEMBLE\n2024-01-03,0.01\n2024-01-04,0.01\n2024-01-05,-0.01\n2024-01-08,0.0\n",
     "it has a row for 2024-01-08, but the NAV log implies returns only up to 2024-01-05"),
    ("Date,ENSEMBLE,SPY\n2024-01-03,0.01,0.02\n", "has header Date,ENSEMBLE,SPY, not Date,ENSEMBLE"),
    ("date,ENSEMBLE\n2024-01-03,0.01\n", "has header date,ENSEMBLE"),
    ("Date,ENSEMBLE\n03/01/2024,0.01\n", "is not written YYYY-MM-DD"),
    ("Date,ENSEMBLE\n2024-01-03,abc\n", "return 'abc' is not a number"),
    ("", "is empty; --rebuild writes it from the NAV log"),
])
def test_a_returns_file_that_disagrees_with_the_log_is_refused_until_rebuilt(
        nav, tmp_path, capsys, existing, message):
    out = tmp_path / "returns.csv"
    out.write_text(existing)
    log = _log(tmp_path, LOG)
    code, shown = _run(nav, capsys, log, "--out", out)
    assert code == 3 and message in shown and "--rebuild" in shown
    assert out.read_text() == existing                  # untouched
    code, shown = _run(nav, capsys, log, "--out", out, "--rebuild")
    assert code == 0 and "rebuilt" in shown and "--rebuild replaced the previous file" in shown
    assert list(_returns(out)) == ["2024-01-03", "2024-01-04", "2024-01-05"]


def test_rebuild_from_a_one_row_log_leaves_only_the_header(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    assert _run(nav, capsys, _log(tmp_path, LOG), "--out", out)[0] == 0
    code, shown = _run(nav, capsys, _log(tmp_path, LOG[:1]), "--out", out)
    assert code == 3 and ("it has a row for 2024-01-03, but the NAV log implies returns only up "
                          "to no date (one NAV row)") in shown
    code, shown = _run(nav, capsys, _log(tmp_path, LOG[:1]), "--out", out, "--rebuild")
    assert code == 0 and "0 row(s) (one NAV row: no return yet)" in shown
    assert out.read_text() == "Date,ENSEMBLE\n"


def test_a_corrected_old_nav_is_named_not_silently_rewritten(nav, tmp_path, capsys):
    out = tmp_path / "returns.csv"
    assert _run(nav, capsys, _log(tmp_path, LOG), "--out", out)[0] == 0
    before = out.read_bytes()
    corrected = [LOG[0], "2024-01-03,101500,", *LOG[2:], "2024-01-08,99500,"]
    code, shown = _run(nav, capsys, _log(tmp_path, corrected), "--out", out)
    assert code == 3 and "disagrees with the NAV log: on 2024-01-03 it holds" in shown
    assert out.read_bytes() == before


def test_a_failed_write_leaves_the_old_file_and_no_temporary_file(nav, tmp_path, capsys, monkeypatch):
    out = tmp_path / "returns.csv"
    assert _run(nav, capsys, _log(tmp_path, LOG[:2]), "--out", out)[0] == 0
    before = out.read_bytes()

    def full_disk(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(nav.os, "replace", full_disk)
    code, shown = _run(nav, capsys, _log(tmp_path, LOG), "--out", out)
    assert code == 3 and "nothing written" in shown and "No space left on device" in shown
    assert out.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["nav.csv", "returns.csv"]


def test_an_unexpected_error_is_an_internal_error_not_success(nav, tmp_path, capsys, monkeypatch):
    def boom(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(nav, "daily_returns", boom)
    out = tmp_path / "returns.csv"
    code, shown = _run(nav, capsys, _log(tmp_path, LOG), "--out", out)
    assert code == nav.EXIT_CRASH == 4 and "nav_to_returns CRASHED: RuntimeError: boom" in shown
    assert not out.exists()


# ------------------------------------------------ processes and the monitor
def _process(*argv, env=None, script=SCRIPT):
    return subprocess.run([sys.executable, str(script), *map(str, argv)], capture_output=True,
                          text=True, env=env or dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


def test_process_exit_statuses_are_never_review_or_kill(tmp_path):
    out = tmp_path / "returns.csv"
    done = _process(_log(tmp_path, LOG), "--out", out)
    assert done.returncode == 0 and "Traceback" not in done.stderr
    bad = _log(tmp_path, ["2024-01-02,100000,", "2024-01-06,101000,"], name="bad.csv")
    done = _process(bad, "--out", tmp_path / "other.csv")
    assert done.returncode == 3 and "REFUSED" in done.stdout and "Traceback" not in done.stderr
    done = _process()                                    # argparse alone would exit 2 (KILL)
    assert done.returncode == 3 and "Traceback" not in done.stderr


@pytest.mark.parametrize("broken", ["pandas", "qcore"])
def test_a_failed_import_at_start_up_is_an_internal_error(tmp_path, broken):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, project / "scripts")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if broken == "pandas":
        shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (tmp_path / "site" / "pandas").mkdir(parents=True)
        (tmp_path / "site" / "pandas" / "__init__.py").write_text(
            "raise ImportError('pandas is broken here')\n")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path / "site"),
                                                          env.get("PYTHONPATH")]))
    out = tmp_path / "returns.csv"
    done = _process(_log(tmp_path, LOG), "--out", out, env=env,
                    script=project / "scripts" / "nav_to_returns.py")
    assert done.returncode == 4 and "Traceback" not in done.stderr and not out.exists()
    assert done.stdout.splitlines() == [done.stdout.strip()]           # one line
    assert done.stdout.startswith("nav_to_returns: start-up failed, cannot import a dependency (")


def test_the_monitor_accepts_the_file_it_produces(nav, tmp_path, capsys, monkeypatch):
    out = tmp_path / "returns.csv"
    assert _run(nav, capsys, _log(tmp_path, LOG), "--out", out)[0] == 0
    monitor = module("scripts/monitor.py", "monitor_nav_test")
    frame = monitor.read_returns(out)
    assert list(frame.columns) == ["ENSEMBLE"] and len(frame) == 3
    assert all(len(part) == 0 for part in monitor.session_mismatch(frame.index))
    monkeypatch.setattr(sys, "argv", ["monitor.py", str(out), "--as-of", "2024-01-05"])
    monitor.main()                                       # status 0 returns without exiting
    shown = capsys.readouterr().out
    assert "judged: 3 of 3 session(s), 2024-01-03 .. 2024-01-05" in shown
    assert "-1.0% (kill < -15%)" in shown and shown.rstrip().endswith("overall: ok")
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "monitor.py"), str(out),
                           "--as-of", "2024-01-05"], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0 and done.stdout.rstrip().endswith("overall: ok")
