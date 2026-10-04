"""scripts/data_quality.py as a pipeline gate, run the way a pipeline runs it.

The script's contract is its exit code (0 clean, 1 WARN, 2 FAIL or any
failure to run), the JSON report and what it prints when it cannot start.
Every case below is a real subprocess run against a small synthetic cache
for the registered universe, written to a temporary directory and selected
through QCORE_DATA_DIR. Each run names its own --out file and either
--strict or its own --known file, so nothing under the project's data/ or
results/ is read or written (the module checks that at the end). The one run
that names no --out, to see where the report goes by default, is launched
from a copy of the script under a throwaway project root.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import qcore.data as qdata  # noqa: E402
from qcore.quality import nyse_bdays  # noqa: E402

SCRIPT = ROOT / "scripts" / "data_quality.py"
DEFAULT_REPORT = ROOT / "results" / "data_quality.json"
PANELS = ["open", "high", "low", "adj_close", "close", "volume"]
TICKERS = list(dict.fromkeys(qdata.ETF_UNIVERSE + qdata.STOCK_UNIVERSE))
QUIET, LATE, DROPPED = TICKERS[0], TICKERS[1], TICKERS[2:4]
LATE_ROWS, DUP_ROW, NO_VOLUME = 300, 100, slice(400, 403)


def clean_frames() -> dict[str, pd.DataFrame]:
    """A cache the gate has nothing to say about: every registered ticker
    from the first row, quarterly dividends, raw close split-adjusted at
    source (the premise PriceBundle.load() assumes), all five index series."""
    idx = nyse_bdays("2020-01-02", "2022-03-31")
    n = len(idx)
    mkt = np.clip(np.random.default_rng(7).normal(0.0, 0.010, n), -0.03, 0.03)
    cols = {name: {} for name in PANELS}
    for k, t in enumerate(TICKERS):
        rng = np.random.default_rng(1000 + k)
        r = 0.9 * mkt + np.clip(rng.normal(0.0003, 0.008, n), -0.04, 0.04)
        adj = 100.0 * np.cumprod(1.0 + r)
        factor = np.ones(n)
        for ex in range(63, n, 63):
            factor[:ex] *= 0.995
        op = adj * (1.0 + rng.uniform(-0.004, 0.004, n))
        cols["adj_close"][t], cols["close"][t], cols["open"][t] = \
            adj, adj / factor, op
        cols["high"][t] = np.maximum(op, adj) * (1.0 + rng.uniform(0.0, 0.004, n))
        cols["low"][t] = np.minimum(op, adj) * (1.0 - rng.uniform(0.0, 0.004, n))
        cols["volume"][t] = rng.integers(100_000, 5_000_000, n).astype(float)
    frames = {name: pd.DataFrame(c, index=idx) for name, c in cols.items()}
    rng, tick = np.random.default_rng(3), np.arange(n)
    spx = np.clip(np.random.default_rng(11).normal(0.0, 0.004, n), -0.01, 0.01)
    series = {"^VIX": 18 + np.cumsum(rng.normal(0, 0.3, n)).clip(-10, 30),
              "^VIX3M": 20 + np.cumsum(rng.normal(0, 0.2, n)).clip(-10, 30),
              "^IRX": 1.5 + 0.01 * (tick % 7),
              "^GSPC": 3000.0 * np.cumprod(1.0 + spx),
              "^TNX": 4.0 + 0.01 * (tick % 5)}
    frames["indices"] = pd.DataFrame(
        {name: series[name] for name in qdata.INDEX_UNIVERSE}, index=idx)
    return frames


def coverage(frames) -> pd.DataFrame:
    """What the downloader writes as coverage.csv beside the panels."""
    px = frames["adj_close"]
    return pd.DataFrame({"first": px.apply(lambda s: s.first_valid_index()),
                         "last": px.apply(lambda s: s.last_valid_index()),
                         "rows": px.count()})


def write_cache(directory: Path, frames, **changed) -> Path:
    directory.mkdir(parents=True)
    for name, frame in {**frames, **changed}.items():
        frame.to_csv(directory / f"{name}.csv",
                     index_label="Ticker" if name == "coverage" else None)
    return directory


def edited(frames, names, change) -> dict[str, pd.DataFrame]:
    out = {}
    for name in names:
        out[name] = frames[name].copy()
        change(out[name])
    return out


def run_gate(cache: Path, out: Path, *argv) -> SimpleNamespace:
    env = dict(os.environ, QCORE_DATA_DIR=str(cache), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), *argv],
                          env=env, capture_output=True, text=True, timeout=300)
    report = json.loads(out.read_text()) if out.exists() else None
    return SimpleNamespace(code=done.returncode, out=done.stdout,
                           err=done.stderr, report=report, cache=cache)


def run_copy_without_out(cache: Path, project: Path) -> SimpleNamespace:
    """The gate with no --out, run from a copy of the script under `project`:
    whichever default it picks, it cannot be this project's results/."""
    script = project / "scripts" / SCRIPT.name
    script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script)
    path = os.pathsep.join(
        p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, QCORE_DATA_DIR=str(cache), PYTHONPATH=path,
               PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(script), "--strict"],
                          env=env, capture_output=True, text=True, timeout=300)
    return SimpleNamespace(code=done.returncode, out=done.stdout,
                           err=done.stderr, cache=cache, project=project)


def _fingerprint(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Every gate run of this module, launched together (each takes a few
    seconds: interpreter start plus the checks over the whole universe)."""
    before = _fingerprint(DEFAULT_REPORT)
    base = tmp_path_factory.mktemp("dq_gate").resolve()
    frames = clean_frames()

    def blank_late(df):
        df.iloc[:LATE_ROWS, df.columns.get_loc(LATE)] = np.nan

    def no_volume(tickers):
        def change(df):
            df.iloc[NO_VOLUME, [df.columns.get_loc(t) for t in tickers]] = 0.0
        return change

    truncated = edited(frames, PANELS, blank_late)
    clean = write_cache(base / "clean", frames, coverage=coverage(frames))
    info = write_cache(base / "info", frames, **truncated)
    lost = write_cache(base / "lost", frames, **truncated,
                       coverage=coverage(frames))
    warn = write_cache(base / "warn", frames,
                       **edited(frames, ["volume"], no_volume([QUIET])))
    fail = write_cache(base / "fail", frames, volume=edited(
        frames, ["volume"], no_volume(TICKERS[4:7]))["volume"].drop(columns=DROPPED))
    duplicate = write_cache(base / "duplicate", frames, **{
        name: pd.concat([frames[name], frames[name].iloc[[DUP_ROW]]]).sort_index()
        for name in PANELS})
    empty = write_cache(base / "empty", frames,
                        **{name: frames[name].iloc[0:0] for name in PANELS})
    beside = write_cache(base / "beside", frames, coverage=coverage(frames))

    when = f"{frames['volume'].index[NO_VOLUME][0].date()}.." \
           f"{frames['volume'].index[NO_VOLUME][-1].date()}"
    allow = base / "allow.csv"
    pd.DataFrame([
        {"check": "zero_volume", "ticker": QUIET, "date": when,
         "note": "verified halt", "expect": ""},
        {"check": "extreme_returns", "ticker": "GONE", "date": "1999-01-04",
         "note": "matches nothing", "expect": "0.5"},
    ]).to_csv(allow, index=False)
    malformed = base / "malformed.csv"
    malformed.write_text("ticker,date,note\nX,2020-01-02,no check column\n")

    cases = {
        "clean": (clean, ["--strict"]),
        "info": (info, ["--strict"]),
        "lost": (lost, ["--strict"]),
        "warn": (warn, ["--strict"]),
        "acked": (warn, ["--known", str(allow)]),
        "acked_strict": (warn, ["--known", str(allow), "--strict"]),
        "fail": (fail, ["--strict", "--max-detail", "1"]),
        "duplicate": (duplicate, ["--strict"]),
        "empty": (empty, ["--strict"]),
        "missing": (base / "no_such_cache", ["--strict"]),
        "no_known": (clean, ["--known", str(base / "no_such_allowlist.csv")]),
        "no_known_strict": (clean, ["--known", str(base / "no_such_allowlist.csv"),
                                    "--strict"]),
        "bad_known": (warn, ["--known", str(malformed)]),
        "no_fundamentals": (clean, ["--strict", "--fundamentals",
                                    str(base / "no_such_fundamentals.csv")]),
    }
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        jobs = {name: pool.submit(run_gate, cache, base / f"{name}.json", *argv)
                for name, (cache, argv) in cases.items()}
        jobs["beside"] = pool.submit(run_copy_without_out, beside,
                                     base / "project")
        results = {name: job.result() for name, job in jobs.items()}
    results["frames"], results["allow"], results["when"] = frames, allow, when
    yield results
    shutil.rmtree(base, ignore_errors=True)
    assert _fingerprint(DEFAULT_REPORT) == before, \
        "a test run replaced the project's own results/data_quality.json"


def hits(report, check, severity=None):
    return [f for f in report["findings"] if f["check"] == check
            and severity in (None, f["severity"])]


def day(frames, pos) -> str:
    return str(frames["adj_close"].index[pos].date())


# ---------------------------------------------------------------- exit codes
def test_clean_cache_exits_0_with_no_findings(runs):
    r, frames = runs["clean"], runs["frames"]
    assert r.code == 0, r.out + r.err
    assert r.report["worst"] == "ok" and r.report["findings"] == []
    assert r.report["n_findings"] == 0
    assert r.report["as_of"] == day(frames, -1)
    assert (r.report["universe"], r.report["rows"]) == \
        (len(TICKERS), len(frames["adj_close"]))
    assert r.report["data_dir"] == str(r.cache)
    assert (r.report["strict"], r.report["known_events_file"]) == (True, None)
    assert r.report["n_acknowledged"] == 0
    assert r.report["dead_acknowledgments"] == []
    assert r.report["acknowledgments_without_expect"] == 0
    assert f"data quality: {r.cache}/, {day(frames, 0)} .. {day(frames, -1)}, " \
           f"{len(TICKERS)} tickers x {len(frames['adj_close'])} rows" in r.out
    assert "(no findings)" in r.out and "overall: ok" in r.out
    assert "FAIL findings" not in r.out and "WARN findings" not in r.out


def test_info_only_exits_0(runs):
    r = runs["info"]
    assert r.code == 0, r.out + r.err
    assert r.report["worst"] == "INFO" and "overall: INFO" in r.out
    assert [(f["check"], f["severity"], f["ticker"])
            for f in r.report["findings"]] == [("delisting", "INFO", LATE)]
    assert set(r.report["findings"][0]) == \
        {"check", "severity", "ticker", "date", "detail"}


def test_warn_exits_1(runs):
    r = runs["warn"]
    assert r.code == 1, r.out + r.err
    assert r.report["worst"] == "WARN"
    warn, = [f for f in r.report["findings"] if f["severity"] == "WARN"]
    assert (warn["check"], warn["ticker"], warn["date"]) == \
        ("zero_volume", QUIET, runs["when"])
    assert warn["value"] > 0, "a measured finding reports its value"
    assert "WARN findings:" in r.out and " ? zero_volume" in r.out
    assert "overall: WARN" in r.out and "FAIL findings" not in r.out


def test_fail_exits_2_even_when_warnings_are_present_too(runs):
    r = runs["fail"]
    assert r.code == 2, r.out + r.err
    assert r.report["worst"] == "FAIL" and "overall: FAIL" in r.out
    assert sorted(f["ticker"] for f in hits(r.report, "symbol_mapping", "FAIL")) \
        == sorted(DROPPED)
    assert len(hits(r.report, "zero_volume", "WARN")) == 3
    # --max-detail 1 trims the WARN list; FAIL lines always print
    assert "WARN findings (first 1 of 3):" in r.out
    assert r.out.count(" ? zero_volume") == 1
    assert r.out.count(" ! symbol_mapping") == len(DROPPED)


# ------------------------------------------------- failures to run at all
def test_missing_cache_exits_2_and_names_the_file_and_the_remedy(runs):
    r = runs["missing"]
    assert r.code == 2, r.out + r.err
    first = r.cache / "open.csv"
    assert f"data quality: FAIL - cannot load inputs: FileNotFoundError: " \
           f"{first} not found" in r.out, r.out
    assert "python src/download_data.py" in r.out
    assert "Traceback" not in r.err
    assert (r.report["worst"], r.report["n_findings"]) == ("FAIL", 1)
    only, = r.report["findings"]
    assert (only["check"], only["severity"]) == ("load", "FAIL")
    assert f"{first} not found" in only["detail"]


def test_missing_fundamentals_file_is_named(runs):
    r = runs["no_fundamentals"]
    assert r.code == 2, r.out + r.err
    assert "cannot load inputs: FileNotFoundError" in r.out
    assert "no_such_fundamentals.csv" in r.out, \
        "the message must say WHICH file is missing"
    assert "no_such_fundamentals.csv" in r.report["findings"][0]["detail"]


def test_explicit_known_path_must_exist(runs):
    r = runs["no_known"]
    assert r.code == 2, "a mistyped allowlist must not run as 'no allowlist'"
    assert "--known allowlist not found" in r.out
    assert "no_such_allowlist.csv" in r.out
    assert r.report["findings"][0]["check"] == "load"
    r = runs["no_known_strict"]
    assert r.code == 0, "--strict reads no allowlist, so none is needed"
    assert (r.report["strict"], r.report["known_events_file"]) == (True, None)


def test_crash_inside_the_run_exits_2_not_1(runs):
    r = runs["bad_known"]  # the allowlist has no `check` column
    assert r.code == 2, "python's own exit 1 would read as WARN to a pipeline"
    assert "data quality: FAIL - check run crashed: KeyError" in r.out, r.out
    assert "Traceback" not in r.err
    assert r.report["worst"] == "FAIL"
    assert [f["check"] for f in r.report["findings"]] == ["run"]


def test_duplicate_date_reaches_the_report_as_a_located_finding(runs):
    r, frames = runs["duplicate"], runs["frames"]
    assert r.code == 2, r.out + r.err
    assert not hits(r.report, "load"), "the checks report it, with the date"
    dup = hits(r.report, "calendar_alignment", "FAIL")
    assert len(dup) == len(PANELS)
    assert all(f["date"] == day(frames, DUP_ROW)
               and "duplicate date row" in f["detail"] for f in dup), dup
    assert r.report["rows"] == len(frames["adj_close"]) + 1


def test_cache_with_no_rows_exits_2_without_a_traceback(runs):
    r = runs["empty"]
    assert r.code == 2, r.out + r.err
    assert "Traceback" not in r.err
    assert "no rows" in r.out and "overall: FAIL" in r.out
    assert not hits(r.report, "run") and not hits(r.report, "load")
    assert [f["detail"] for f in hits(r.report, "calendar_gaps", "FAIL")] == \
        ["empty price index"]
    assert r.report["as_of"] is None and r.report["rows"] == 0


# ------------------------------------------------------- acknowledged events
def test_report_records_the_acknowledgment_context(runs):
    r = runs["acked"]
    assert r.code == 0, r.out + r.err
    assert r.report["worst"] == "INFO" and r.report["strict"] is False
    assert r.report["known_events_file"] == str(runs["allow"])
    assert r.report["n_acknowledged"] == 1
    assert r.report["dead_acknowledgments"] == \
        [["extreme_returns", "GONE", "1999-01-04"]]
    assert r.report["acknowledgments_without_expect"] == 1
    acked, = [f for f in hits(r.report, "zero_volume", "INFO")
              if "price moved" in f["detail"]]
    assert acked["date"] == runs["when"]
    assert "[acknowledged: verified halt]" in acked["detail"]
    assert "[1 findings acknowledged via allow.csv]" in r.out
    assert "dead acknowledgment" in r.out and "GONE" in r.out
    assert "[1 of 2 acknowledgment rows carry no expect value" in r.out
    r = runs["acked_strict"]
    assert r.code == 1, "--strict ignores the allowlist it is given"
    assert (r.report["strict"], r.report["known_events_file"]) == (True, None)
    assert r.report["n_acknowledged"] == 0
    assert not r.report["dead_acknowledgments"]
    assert "acknowledg" not in r.out


# --------------------------------------------------- history lost by the cache
def test_history_lost_versus_the_coverage_record_fails(runs):
    r, frames = runs["lost"], runs["frames"]
    assert r.code == 2, r.out + r.err
    lost, = hits(r.report, "history_loss")
    assert (lost["severity"], lost["ticker"]) == ("FAIL", LATE)
    assert lost["date"] == day(frames, LATE_ROWS)
    assert f"{LATE_ROWS} of the {len(frames['adj_close'])} observations" \
        in lost["detail"]
    assert f"history_loss: compared with {r.cache}/coverage.csv" in r.out
    # the same panels without the record pass as a later inception, and a
    # cache that matches its record raises nothing
    assert not hits(runs["info"].report, "history_loss")
    assert runs["info"].code == 0 and runs["clean"].report["findings"] == []


# ----------------------------------------------------------------- defaults
def test_default_report_and_allowlist_locations(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("dq_gate_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_OUT == DEFAULT_REPORT
    assert mod.DEFAULT_KNOWN == ROOT / "data" / "dq_known_events.csv"
    monkeypatch.setattr(mod.qdata, "DATA_DIR", mod.qdata.DEFAULT_DATA_DIR)
    assert mod._default_out() == DEFAULT_REPORT
    monkeypatch.setattr(mod.qdata, "DATA_DIR", tmp_path)
    assert mod._default_out() == tmp_path / "data_quality.json", \
        "another cache's report must not replace the default cache's"


def test_report_of_a_cache_named_by_the_override_is_written_beside_it(runs):
    # no --out, QCORE_DATA_DIR set: a live-session run. The report describes
    # that cache and lands beside it; the results/ directory of the project
    # the script belongs to (a throwaway root here) is not created or written
    r = runs["beside"]
    assert r.code == 0, r.out + r.err
    report = r.cache / "data_quality.json"
    assert report.exists(), r.out + r.err
    assert not (r.project / "results").exists(), \
        "the default cache's report must not be replaced by another cache's"
    saved = json.loads(report.read_text())
    assert (saved["worst"], saved["data_dir"]) == ("ok", str(r.cache))
    assert f"overall: ok   (0 findings -> {report})" in r.out, r.out
