"""Monitor inputs: an unusable file, source or date is a refusal (exit 3),
never an ok, a REVIEW (1), a KILL (2) or a bare traceback."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import warnings

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/monitor.py"


def _monitor(monkeypatch, today=None):
    spec = importlib.util.spec_from_file_location("monitor_input_test", SCRIPT)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monkeypatch.setattr(monitor, "cash_benchmark", lambda idx: pd.Series(0.0, index=idx))
    monkeypatch.setattr(monitor, "benchmark_span", lambda: "patched")
    if today is not None:
        monkeypatch.setattr(monitor, "current_ny_date", lambda: pd.Timestamp(today))
    return monitor


def _run(monitor, monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["monitor.py", *map(str, argv)])
    try:
        monitor.main()
        code = 0
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out


def _write(path, index, values):
    frame = pd.DataFrame({"ENSEMBLE": values}, index=index)
    frame.index.name = "Date"
    frame.to_csv(path)
    return path


@pytest.mark.parametrize("dates", [
    ["not-a-date", "2024-01-03"],
    ["", "2024-01-03"],
    ["2024-01-03", "2024-01-02"],
    ["2024-01-02", "2024-01-02"],
    ["2024-01-02T12:00:00", "2024-01-03T12:00:00"],
    ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
    ["2024-01-02T00:00:00+01:00", "2024-01-03T00:00:00+02:00"],
    ["01/02/2024", "01/03/2024"],      # would be guessed month-first
    ["13/02/2024", "14/02/2024"],      # would be guessed day-first, with a warning
    ["20240102", "20240103"],
    ["2024-01-02", "9999-01-03"],
])
def test_invalid_monitor_dates_exit_as_failure(tmp_path, monkeypatch, capsys, dates):
    monitor = _monitor(monkeypatch, "2024-01-03")
    returns = tmp_path / "returns.csv"
    returns.write_text("Date,ENSEMBLE\n" + "".join(f"{date},0\n" for date in dates))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no format guessing, so nothing to warn about
        code, out = _run(monitor, monkeypatch, capsys, returns)
    assert code == monitor.EXIT_INVALID_INPUT == 3
    assert "invalid monitor input" in out
    assert "overall" not in out


def test_monitor_has_no_default_source(tmp_path, monkeypatch, capsys):
    """A bare `monitor.py` used to judge the frozen backtest and print ok."""
    monitor = _monitor(monkeypatch, "2026-10-01")
    backtest = tmp_path / "sleeve_returns.csv"
    backtest.write_text("Date,ENSEMBLE\n2024-01-02,0.01\n2024-01-03,0.01\n")
    monkeypatch.setattr(monitor, "BACKTEST_RETURNS", backtest)
    code, out = _run(monitor, monkeypatch, capsys)
    assert code == 3 and "required" in out and "overall" not in out
    code, out = _run(monitor, monkeypatch, capsys, "--backtest")
    assert code == 0
    assert "mode: BACKTEST calibration" in out and "NOT the live account" in out
    assert out.rstrip().endswith("overall: ok")


def test_live_drawdown_is_seen_and_backtest_cannot_mask_it(tmp_path, monkeypatch, capsys):
    monitor = _monitor(monkeypatch, "2024-01-04")
    live = tmp_path / "live.csv"
    live.write_text("Date,ENSEMBLE\n2024-01-02,-0.2\n2024-01-03,0\n")
    code, out = _run(monitor, monkeypatch, capsys, live)
    assert code == 2 and "mode: LIVE account" in out and out.rstrip().endswith("overall: KILL")
    code, out = _run(monitor, monkeypatch, capsys, live, "--backtest")
    assert code == 3 and "not allowed" in out and "overall" not in out


def test_backtest_mode_reports_the_end_state_unless_since_is_given(tmp_path, monkeypatch, capsys):
    monitor = _monitor(monkeypatch, "2026-10-01")
    backtest = tmp_path / "sleeve_returns.csv"
    backtest.write_text("Date,ENSEMBLE\n2024-01-02,-0.2\n2024-01-03,0.3\n2024-01-04,0\n")
    monkeypatch.setattr(monitor, "BACKTEST_RETURNS", backtest)
    code, out = _run(monitor, monkeypatch, capsys, "--backtest")
    assert code == 0 and "judged: 1 of 3 session(s), 2024-01-04 .. 2024-01-04" in out
    code, out = _run(monitor, monkeypatch, capsys, "--backtest", "--since", "2024-01-02")
    assert code == 2 and "first 2024-01-02, worst -20.0% on 2024-01-02" in out
    code, out = _run(monitor, monkeypatch, capsys, "--backtest", "--as-of", "2024-01-04")
    assert code == 3 and "overall" not in out


@pytest.mark.parametrize("today, code, text", [
    ("2024-01-03", 0, "0 completed session(s)"),   # same day: the close just printed
    ("2024-01-10", 0, "4 completed session(s)"),   # 4, 5, 8, 9 January
    ("2024-01-11", 0, "5 completed session(s)"),   # exactly at the limit
    ("2024-01-12", 3, "stale live file"),          # six sessions behind
    ("2024-01-02", 3, "dated after"),              # last row in the future
])
def test_live_file_freshness_gate(tmp_path, monkeypatch, capsys, today, code, text):
    monitor = _monitor(monkeypatch, today)
    assert monitor.MAX_STALE_SESSIONS == 5
    live = tmp_path / "live.csv"
    live.write_text("Date,ENSEMBLE\n2024-01-02,0.001\n2024-01-03,0.001\n")
    got, out = _run(monitor, monkeypatch, capsys, live)
    assert got == code and text in out
    assert ("overall" in out) == (code == 0)
    assert (f"checked against {today} New York" in out) == (code == 0)
    assert "offline replay" not in out  # judged against the clock, not a stated date


def test_as_of_replays_an_old_live_file_and_staleness_skips_closed_days(tmp_path, monkeypatch, capsys):
    monitor = _monitor(monkeypatch, "2026-10-01")
    live = tmp_path / "live.csv"
    live.write_text("Date,ENSEMBLE\n2024-01-11,0.001\n2024-01-12,0.001\n")
    assert _run(monitor, monkeypatch, capsys, live)[0] == 3
    code, out = _run(monitor, monkeypatch, capsys, live, "--as-of", "2024-01-16")
    assert code == 0 and "0 completed session(s)" in out  # weekend + MLK day
    # The freshness gate was answered by the stated date; the output says so.
    assert "checked against --as-of 2024-01-16, an offline replay and not today's date" in out
    code, out = _run(monitor, monkeypatch, capsys, live, "--as-of", "junk")
    assert code == 3 and "--as-of must be a YYYY-MM-DD date" in out
    behind = monitor.sessions_behind
    assert behind(pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-18")) == 2
    assert behind(pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-12")) == 0
    assert behind(pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-10")) == 0


@pytest.mark.parametrize("since, text", [
    ("2024-01-13", "--since 2024-01-13 is after the last row 2024-01-12"),
    ("soon", "--since must be a YYYY-MM-DD date"),
])
def test_since_must_be_a_date_inside_the_file(tmp_path, monkeypatch, capsys, since, text):
    monitor = _monitor(monkeypatch, "2024-01-12")
    live = tmp_path / "live.csv"
    live.write_text("Date,ENSEMBLE\n2024-01-11,0.001\n2024-01-12,0.001\n")
    code, out = _run(monitor, monkeypatch, capsys, live, "--since", since)
    assert code == 3 and text in out and "overall" not in out


def test_monitor_rejects_calendar_day_rows(tmp_path, monkeypatch, capsys):
    monitor = _monitor(monkeypatch, "2024-01-11")
    live = _write(tmp_path / "live.csv", pd.date_range("2024-01-02", periods=10, freq="D"), 0.0)
    for extra in ([], ["--allow-gaps"]):  # closed-day rows are never accepted
        code, out = _run(monitor, monkeypatch, capsys, live, *extra)
        assert code == 3
        assert "2 row(s) on non-NYSE-session dates: 2024-01-06, 2024-01-07" in out
        assert "overall" not in out


def test_monitor_rejects_holiday_row(tmp_path, monkeypatch, capsys):
    monitor = _monitor(monkeypatch, "2024-01-17")
    live = _write(tmp_path / "live.csv", pd.bdate_range("2024-01-12", "2024-01-17"), 0.0)
    code, out = _run(monitor, monkeypatch, capsys, live)
    assert code == 3 and "non-NYSE-session dates: 2024-01-15" in out  # Martin Luther King Jr. Day


def test_monitor_rejects_missing_sessions_unless_allowed(tmp_path, monkeypatch, capsys):
    monitor = _monitor(monkeypatch, "2024-03-28")
    index = monitor.nyse_bdays("2024-01-02", "2024-03-28")
    returns = pd.Series(0.0, index=index)
    crash = pd.bdate_range("2024-02-05", "2024-02-09")
    returns[crash] = -0.05
    complete = _write(tmp_path / "complete.csv", index, returns)
    code, out = _run(monitor, monkeypatch, capsys, complete)
    assert code == 2 and out.rstrip().endswith("overall: KILL")
    # The same account with the crash week left out of the file must not pass.
    holed = _write(tmp_path / "holed.csv", index.difference(crash), 0.0)
    code, out = _run(monitor, monkeypatch, capsys, holed)
    assert code == 3 and "5 NYSE session(s) missing" in out and "2024-02-05" in out
    assert "overall" not in out
    code, out = _run(monitor, monkeypatch, capsys, holed, "--allow-gaps")
    assert code == 0 and "warning: 5 NYSE session(s) missing" in out
    assert out.rstrip().endswith("overall: ok")


def test_backtest_without_a_saved_stream_names_the_command_that_creates_it(tmp_path, monkeypatch, capsys):
    """A checkout that has not built the ensemble yet has no backtest stream."""
    monitor = _monitor(monkeypatch, "2026-10-01")
    monkeypatch.setattr(monitor, "BACKTEST_RETURNS", tmp_path / "results" / "sleeve_returns.csv")
    for extra in ([], ["--since", "2024-01-02"]):
        code, out = _run(monitor, monkeypatch, capsys, "--backtest", *extra)
        assert code == monitor.EXIT_INVALID_INPUT == 3
        assert out.splitlines() == [
            "invalid monitor input: results/sleeve_returns.csv not found: --backtest judges the "
            "blended backtest stream, which the ensemble build saves. Create it first: "
            "python src/ensemble.py"]


def _process(*argv):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-W", "error::UserWarning", str(SCRIPT), *map(str, argv)],
                          capture_output=True, text=True, env=env)


def test_process_exit_status_separates_refusal_from_verdicts(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Date,ENSEMBLE\nnot-a-date,0\n2024-01-03,0\n")
    done = _process(bad)
    assert done.returncode == 3 and "invalid monitor input" in done.stdout
    assert "Traceback" not in done.stderr
    done = _process()  # a usage error is argparse status 2 by default, which would read as KILL
    assert done.returncode == 3 and "overall" not in done.stdout
    done = _process(tmp_path / "absent.csv")
    assert done.returncode == 3 and "overall" not in done.stdout
    live = tmp_path / "live.csv"
    live.write_text("Date,ENSEMBLE\n2024-01-02,0.001\n2024-01-03,0.001\n")
    done = _process(live, "--as-of", "2024-01-03")
    assert done.returncode == 0 and done.stdout.rstrip().endswith("overall: ok")
    live.write_text("Date,ENSEMBLE\n2024-01-02,-0.2\n2024-01-03,0\n")
    done = _process(live, "--as-of", "2024-01-03")
    assert done.returncode == 2 and done.stdout.rstrip().endswith("overall: KILL")
