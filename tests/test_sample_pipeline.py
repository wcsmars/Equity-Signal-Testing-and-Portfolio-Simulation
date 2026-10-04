"""The documented market-data pipeline, end to end, on the synthetic sample cache.

scripts/make_sample_data.py writes a seeded, simulated stand-in for the
downloaded price cache. This module builds it once and then runs every
documented command on it as a real subprocess, with QCORE_DATA_DIR naming
the sample:

  the data-quality gate; the default entry point of each of the eight
  strategy modules; the four sleeve records and the ensemble; the kill-rule
  monitor on the ensemble's returns; the offline order sheet for the sample's
  last session with the example ledger; the fills check on the example fills;
  the five selection sweeps and the trial registry that reads them.

Where the commands run. pairs_statarb.py, tsmom_voltarget.py, the sweeps,
ensemble.py and trial_registry.py save under results/ beside the code and
take no other location, and monitor.py --backtest and live_targets.py read
from there. So the commands run in a temporary copy of the source tree with
its own empty results/ folder and its own sample_data/. No price or result
of this checkout is used, and nothing under its results/, data/ or
sample_data/ is written; the last test checks that.

What is asserted: exit codes, that each output exists, is finite and has the
documented structure, and that every rule traded. No performance figure is
pinned: the prices are simulated, so the figures mean nothing. The one
pinned value is the digest of the sample's indices.csv, which holds the
generator to the same bytes on every supported numpy and pandas.

Commands that only read the sample are launched together; the whole module
takes well under a minute on a laptop.
"""

import hashlib
import importlib.util
import json
import math
import os
import re
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
from qcore.backtest import OOS_SPLIT  # noqa: E402
from qcore.calendar import confirmed_month_ends  # noqa: E402
from qcore.quality import nyse_bdays  # noqa: E402

# The examples sit in examples/ of the release. A development checkout that
# keeps the release in a nested folder has them one level down.
EXAMPLES = next((folder for folder in (ROOT / "examples", ROOT / "public" / "examples")
                 if (folder / "ledger.example.csv").is_file()), ROOT / "examples")
EXAMPLE_FILES = ("ledger.example.csv", "fills.example.csv", "live_returns.example.csv")
SLEEVES = ("mean_reversion", "seasonality_flows", "tsmom_trend", "xsec_etf_mom")
PRINTING = ("xsec_stock_mom", "vol_regime", "pairs_statarb")  # metrics as JSON on stdout
STRATEGIES = (*SLEEVES, *PRINTING, "tsmom_voltarget")
PANELS = ("adj_close", "close", "open", "high", "low", "volume")
SAMPLE_FILES = {f"{name}.csv" for name in (*PANELS, "indices", "coverage")} | {"README.txt"}
TICKERS = sorted(set(qdata.ETF_UNIVERSE) | set(qdata.STOCK_UNIVERSE)
                 | set(qdata.OPERATIONAL_UNIVERSE))
GUARDED = (ROOT / "results", ROOT / "results" / "recomputed", ROOT / "data", ROOT / "sample_data")
# sha256 of indices.csv for the default seed and length. The file depends on
# the factor paths, the NYSE calendar and the number format, and not on the
# ticker universes. A deliberate change to any of those needs the new digest:
#   python scripts/make_sample_data.py --out <dir> && sha256sum <dir>/indices.csv
INDICES_SHA256 = "e436f3877438fe9122c478c71cf92f4614bbb2fbfb9a77efd584ecca2da85442"


def module(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_sample_pipeline_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def generator():
    return module("scripts/make_sample_data.py")


def _sweep_commands() -> dict[str, list[str]]:
    """Saved grid -> the command the trial registry documents for it."""
    commands = module("scripts/trial_registry.py").INPUT_COMMANDS
    return {name: command.split()[1:] for name, command in commands.items()
            if name.endswith("_variants.csv")}


def _fingerprint() -> dict:
    """Name, size and modification time of every file directly inside the
    folders this module must leave alone (None for a folder that is absent)."""
    seen = {}
    for folder in GUARDED:
        seen[folder.relative_to(ROOT).as_posix()] = (
            sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir()
                   if p.is_file() and p.name != ".DS_Store") if folder.is_dir() else None)
    return seen


def _project_copy(base: Path, sweeps: dict) -> Path:
    """The source tree, the scripts, the two studies the registry names and
    the example inputs, beside an empty results folder."""
    project = base / "project"
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (project / "scripts").mkdir()
    for path in sorted((ROOT / "scripts").glob("*.py")):
        shutil.copy(path, project / "scripts")
    for command in sweeps.values():
        target = project / command[0]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / command[0], target)
    (project / "examples").mkdir()
    for name in EXAMPLE_FILES:
        shutil.copy(EXAMPLES / name, project / "examples")
    (project / "results").mkdir()
    return project


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, request):
    """Every command of the module, run once in a temporary project copy.
    .runs maps a name to its finished process; a sleeve's standard output is
    redirected into results/<key>.json, as documented, and read back."""
    before = _fingerprint()
    sweeps = _sweep_commands()
    project = _project_copy(tmp_path_factory.mktemp("sample_pipeline").resolve(), sweeps)
    cache, results = project / "sample_data", project / "results"
    env = {key: value for key, value in os.environ.items() if key != "QCORE_REBASE"}
    env.update(QCORE_DATA_DIR=str(cache), PYTHONDONTWRITEBYTECODE="1")
    # warnings turned into errors for this test run apply to the commands too
    strict = [f"-W{option}" for option in
              (*sys.warnoptions, *(request.config.getoption("pythonwarnings") or ()))]
    runs = {}

    def run(name, *argv, redirect=None):
        command = [sys.executable, *strict, *argv]
        if redirect is None:
            done = subprocess.run(command, cwd=project, env=env, capture_output=True,
                                  text=True, timeout=600)
        else:  # the documented "> results/<key>.json"
            with redirect.open("w", encoding="utf-8") as stream:
                done = subprocess.run(command, cwd=project, env=env, stdout=stream,
                                      stderr=subprocess.PIPE, text=True, timeout=600)
            done.stdout = redirect.read_text(encoding="utf-8")
        runs[name] = done
        return done

    def together(jobs):
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
            for future in [pool.submit(run, *args, **kwargs) for args, kwargs in jobs]:
                future.result()

    # no --out: the default location, which in the copy is its own sample_data/
    generated = run("generate", "scripts/make_sample_data.py")
    last = ""
    if generated.returncode == 0:
        last = (cache / "adj_close.csv").read_text().splitlines()[-1].split(",")[0]
        together(
            [(("data_quality", "scripts/data_quality.py"), {})]
            + [((key, f"src/strategies/{key}.py"), {"redirect": results / f"{key}.json"})
               for key in SLEEVES]
            + [((key, f"src/strategies/{key}.py"), {}) for key in (*PRINTING, "tsmom_voltarget")]
            + [((name, *command), {}) for name, command in sweeps.items()]
            + [(("reconcile", "scripts/reconcile.py", "examples/fills.example.csv"), {})])
        run("ensemble", "src/ensemble.py")
        together([
            (("monitor", "scripts/monitor.py", "--backtest"), {}),
            (("live_targets", "scripts/live_targets.py", "--as-of", last, "--cash", "5000",
              "--ledger", "examples/ledger.example.csv", "--ignore-ledger-date"), {}),
            (("trial_registry", "scripts/trial_registry.py"), {}),
        ])
    return SimpleNamespace(project=project, cache=cache, results=results, runs=runs,
                           sweeps=sweeps, before=before, last=last)


def _done(pipeline, name, codes=(0,)):
    """The finished run `name`, after checking that it ran and how it exited."""
    assert name in pipeline.runs, f"{name} was not run: the sample cache was not built"
    done = pipeline.runs[name]
    assert done.returncode in codes and "Traceback" not in done.stderr, (
        f"{name} exited {done.returncode}\n{done.stdout[-1500:]}\n{done.stderr[-3000:]}")
    return done


def _panel(pipeline, name: str) -> pd.DataFrame:
    return pd.read_csv(pipeline.cache / f"{name}.csv", index_col=0, parse_dates=True)


def _check_metrics(m: dict, pipeline) -> None:
    """A metrics() record of a rule that traded, on both sides of the split."""
    first = (pipeline.cache / "adj_close.csv").read_text().splitlines()[1].split(",")[0]
    assert first <= m["start"] < OOS_SPLIT < m["end"] == pipeline.last
    for block in ("full", "in_sample", "out_of_sample", "gross_full"):
        assert all(math.isfinite(m[block][key]) for key in ("cagr", "vol", "sharpe", "maxdd")), block
        assert m[block]["vol"] > 0 and -1 < m[block]["maxdd"] <= 0
    assert m["ann_turnover_oneside"] > 0 and m["ann_cost_drag"] > 0   # it traded and paid for it
    assert 0 <= m["pct_positive_months"] <= 1 and math.isfinite(m["worst_month"])


# ------------------------------------------------------------- the generator
def test_generator_writes_the_downloaders_layout_for_every_registered_ticker(pipeline, generator):
    done = _done(pipeline, "generate")
    assert "SYNTHETIC" in done.stdout and "QCORE_DATA_DIR" in done.stdout
    assert SAMPLE_FILES <= {p.name for p in pipeline.cache.iterdir()}
    frames = {name: _panel(pipeline, name) for name in PANELS}
    adj = frames["adj_close"]
    for name, frame in frames.items():
        assert list(frame.columns) == TICKERS, name
        assert frame.index.equals(adj.index) and frame.index.name == "Date", name
        assert frame.isna().equals(adj.isna()), name      # one listing date per ticker
    # real NYSE sessions, from some years before the split to a year-end
    assert adj.index.equals(nyse_bdays(adj.index[0], adj.index[-1]))
    first_year = int(OOS_SPLIT[:4]) - generator.IN_SAMPLE_YEARS
    assert adj.index[0] == nyse_bdays(f"{first_year}-01-01", OOS_SPLIT)[0]
    assert (adj.index[-1].month, adj.index[-1].day) == (12, 31) or adj.index[-1].dayofweek == 4
    assert confirmed_month_ends(adj.index)[-1] == adj.index[-1]     # monthly sleeves decide on it
    indices = _panel(pipeline, "indices")
    assert list(indices.columns) == sorted(qdata.INDEX_UNIVERSE)
    assert indices.index.equals(adj.index) and indices.notna().all().all()
    coverage = pd.read_csv(pipeline.cache / "coverage.csv", index_col=0)
    assert list(coverage.columns) == ["first", "last", "rows"] and coverage.index.name == "Ticker"
    assert list(coverage.index) == TICKERS
    assert (coverage["rows"] == adj.count()).all()
    assert (pd.to_datetime(coverage["first"]) == adj.apply(pd.Series.first_valid_index)).all()
    assert (coverage["last"] == pipeline.last).all()
    # a few tickers list after the first row, as in a real cache; all are quoted at the end
    assert set(adj.columns[adj.isna().any()]) == set(generator.LATE_LISTINGS) & set(TICKERS)
    assert adj.notna().iloc[-1].all() and adj.notna().iloc[0].mean() > 0.9


def test_sample_values_are_plausible_bars_with_dividends(pipeline, generator):
    _done(pipeline, "generate")
    market = generator.MARKET_FUND
    bonds = [t for t in generator.BOND_FUNDS if t in TICKERS]
    unpaid = [t for t in (*generator.COMMODITY_FUNDS, *generator.CURRENCY_FUNDS) if t in TICKERS]
    frames = {name: _panel(pipeline, name) for name in PANELS}
    adj, close, low, high = (frames[k] for k in ("adj_close", "close", "low", "high"))
    listed = adj.notna()
    for name in ("adj_close", "close", "open", "high", "low"):
        values = frames[name].to_numpy()[listed.to_numpy()]
        assert np.isfinite(values).all() and (values > 0).all(), name
    body_low = np.minimum(frames["open"], adj)
    body_high = np.maximum(frames["open"], adj)
    assert ((low <= body_low) | ~listed).all().all() and ((high >= body_high) | ~listed).all().all()
    volume = frames["volume"].to_numpy()[listed.to_numpy()]
    assert (volume > 0).all() and (volume == np.round(volume)).all()
    # total-return and price-return series meet on the final row and differ before ex-dates
    assert np.allclose(adj.iloc[-1], close.iloc[-1], rtol=0, atol=1e-8)
    payout = (adj.pct_change(fill_method=None) - close.pct_change(fill_method=None)).fillna(0.0)
    paying = payout > qdata.DIVIDEND_NOISE_FLOOR          # what the loader counts as a payout
    assert payout.where(~paying, 0.0).abs().max().max() < 1e-7 and payout.max().max() < 0.05
    events = paying.sum()
    years = adj.index[-1].year - adj.index[0].year + 1
    assert events[market] == 4 * years and adj[market].iloc[0] < close[market].iloc[0]
    assert (events[bonds] == 12 * years).all()            # bond funds pay monthly
    assert events[unpaid].sum() == 0                      # commodity and currency funds pay nothing
    assert (events > 0).sum() > len(TICKERS) // 2
    # returns move together through the factors, bonds apart, and the pairs are tied
    returns = adj.pct_change(fill_method=None)
    with_market = returns.corrwith(returns[market])
    assert with_market[qdata.STOCK_UNIVERSE].median() > 0.4 > with_market[bonds].median()
    for follower, (leader, _) in generator.PAIR_FOLLOWERS.items():
        assert returns[follower].corr(returns[leader]) > 0.85, follower
    indices = _panel(pipeline, "indices")
    assert indices["^IRX"].between(0, 10).all() and indices["^TNX"].between(0, 10).all()
    assert indices["^VIX"].between(9, 100).all() and indices["^VIX3M"].between(9, 100).all()
    ratio = indices["^VIX"] / indices["^VIX3M"]
    assert (ratio < 0.95).mean() > 0.3 and (ratio > 1.05).mean() > 0.03   # both regimes occur
    assert indices["^GSPC"].pct_change(fill_method=None).corr(
        close[market].pct_change(fill_method=None)) > 0.99


def test_sample_is_labelled_synthetic_and_written_in_a_fixed_number_format(pipeline, generator):
    _done(pipeline, "generate")
    readme = (pipeline.cache / "README.txt").read_text().splitlines()
    assert readme[0] == generator.MARKER and "SYNTHETIC" in readme[0]
    assert f"--seed {generator.DEFAULT_SEED} --years {generator.DEFAULT_YEARS}" in readme[2]
    source = (ROOT / "scripts" / "make_sample_data.py").read_text().splitlines()
    assert source[1].startswith("# SYNTHETIC DATA GENERATOR")
    price = re.compile(r"\d+\.\d{8}")
    for name in ("adj_close", "close", "open", "high", "low"):
        lines = (pipeline.cache / f"{name}.csv").read_text().splitlines()
        for line in (lines[1], lines[len(lines) // 2], lines[-1]):
            day, *cells = line.split(",")
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", day), name
            assert all(cell == "" or price.fullmatch(cell) for cell in cells), name
    cells = (pipeline.cache / "volume.csv").read_text().splitlines()[-1].split(",")[1:]
    assert all(cell.isdigit() for cell in cells)
    cells = (pipeline.cache / "indices.csv").read_text().splitlines()[-1].split(",")[1:]
    assert all(re.fullmatch(r"\d+\.\d{4}", cell) for cell in cells)


def test_same_seed_gives_the_same_bytes_in_another_process_and_on_every_supported_version(
        pipeline, generator):
    _done(pipeline, "generate")
    files = generator.sample_files(generator.build_sample())
    assert set(files) == SAMPLE_FILES
    for name, text in files.items():
        assert (pipeline.cache / name).read_bytes() == text.encode("utf-8"), name
    digest = hashlib.sha256(files["indices.csv"].encode("utf-8")).hexdigest()
    assert digest == INDICES_SHA256, (
        "indices.csv of the default sample changed. If the generator, the NYSE calendar or "
        "the number format was changed on purpose, pin the new digest; otherwise the sample "
        f"is no longer reproducible across library versions (got {digest})")
    # a shorter panel is the start of the longer one, except for the adjusted
    # prices, which are tied to its own final row; another seed is another path
    shorter = generator.sample_files(generator.build_sample(years=generator.MIN_YEARS))
    for name in ("close.csv", "volume.csv", "indices.csv"):
        lines = shorter[name].splitlines()
        assert 1000 < len(lines) < len(files[name].splitlines()), name
        assert files[name].splitlines()[:len(lines)] == lines, name
    assert shorter["adj_close.csv"].splitlines()[1] != files["adj_close.csv"].splitlines()[1]
    reseeded = generator.sample_files(generator.build_sample(generator.DEFAULT_SEED + 1,
                                                             generator.MIN_YEARS))
    assert reseeded["indices.csv"].splitlines()[0] == shorter["indices.csv"].splitlines()[0]
    assert reseeded["indices.csv"].splitlines()[1] != shorter["indices.csv"].splitlines()[1]
    assert reseeded["coverage.csv"] == shorter["coverage.csv"]      # same sessions and listings


def test_generator_refuses_a_directory_it_did_not_write(pipeline, generator, tmp_path, capsys):
    _done(pipeline, "generate")
    assert generator.DEFAULT_OUT == ROOT / "sample_data" != qdata.DEFAULT_DATA_DIR
    downloaded = tmp_path / "cache"
    downloaded.mkdir()
    (downloaded / "adj_close.csv").write_text("Date,SPY\n2024-01-02,470.0\n")
    assert generator.main(["--out", str(downloaded)]) == 2
    assert "sample data not written" in capsys.readouterr().err
    assert [p.name for p in downloaded.iterdir()] == ["adj_close.csv"]
    assert (downloaded / "adj_close.csv").read_text() == "Date,SPY\n2024-01-02,470.0\n"
    with pytest.raises(ValueError, match="never overwritten"):
        generator.write_sample(downloaded, {})
    # a new, an empty and an earlier sample directory are all writable
    empty = tmp_path / "empty"
    empty.mkdir()
    assert generator.refusal(tmp_path / "absent") is None and generator.refusal(empty) is None
    assert generator.refusal(pipeline.cache) is None
    (empty / "README.txt").write_text("Notes on my downloaded cache.\n")
    assert "not a sample" in generator.refusal(empty)
    if (qdata.DEFAULT_DATA_DIR / "adj_close.csv").exists():     # a real cache in this checkout
        assert generator.refusal(qdata.DEFAULT_DATA_DIR) is not None
    for argv in (["--years", str(generator.MIN_YEARS - 1)], ["--years", str(generator.MAX_YEARS + 1)],
                 ["--seed", "-1"], ["--see", "1"]):
        with pytest.raises(SystemExit) as stop:
            generator.main(["--out", str(tmp_path / "never"), *argv])
        assert stop.value.code == 2
    assert not (tmp_path / "never").exists()


# -------------------------------------------------------------- the pipeline
def test_data_quality_gate_passes_the_sample(pipeline):
    done = _done(pipeline, "data_quality", codes=(0, 1))
    report = json.loads((pipeline.cache / "data_quality.json").read_text())   # beside the cache
    not_info = [f for f in report["findings"] if f["severity"] != "INFO"]
    assert done.returncode == 0 and report["worst"] in ("ok", "INFO"), not_info[:10]
    assert "overall:" in done.stdout and "FAIL findings" not in done.stdout
    assert (report["as_of"], report["universe"]) == (pipeline.last, len(TICKERS))
    assert report["rows"] == len(_panel(pipeline, "adj_close"))
    assert report["known_events_file"] is None and report["n_acknowledged"] == 0


@pytest.mark.parametrize("key", STRATEGIES)
def test_each_strategy_entry_point_runs_and_trades(pipeline, key):
    done = _done(pipeline, key)
    if key == "tsmom_voltarget":       # prints a table; its record is the saved study
        study = json.loads((pipeline.results / "tsmom_voltarget.json").read_text())
        grid = pd.read_csv(pipeline.results / "tsmom_voltarget_variants.csv")
        assert study["study"] is True and len(grid) == 9
        assert study["winner_by_rule"] in set(grid["variant"])
        assert np.isfinite(grid.select_dtypes("number").to_numpy()).all()
        for name in ("winner_metrics", "baseline_metrics", "klev164_metrics", "conc_metrics"):
            _check_metrics(study[name], pipeline)
        return
    record = json.loads(done.stdout)
    _check_metrics(record.get("metrics", record), pipeline)
    if key == "pairs_statarb":
        assert record["params"]["pairs_passing_in_sample"]            # the combined book is not empty
        assert json.loads((pipeline.results / "pairs_statarb.json").read_text()) == record
    if key in SLEEVES:                 # the redirect left the record the ensemble reads
        assert json.loads((pipeline.results / f"{key}.json").read_text()) == record


def test_ensemble_blends_the_four_sleeve_records(pipeline):
    done = _done(pipeline, "ensemble")
    assert "sleeve weights (inverse IS-vol):" in done.stdout
    blend = json.loads((pipeline.results / "ensemble.json").read_text())
    weights = blend["sleeve_weights"]
    assert set(weights) == set(SLEEVES) and all(w > 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    _check_metrics(blend, pipeline)
    assert set(blend["rebalance_sensitivity"]) == {"daily", "monthly", "never"}
    returns = pd.read_csv(pipeline.results / "sleeve_returns.csv", index_col=0, parse_dates=True)
    assert list(returns.columns) == [*SLEEVES, "ENSEMBLE"]
    assert np.isfinite(returns.to_numpy()).all() and (returns.abs() < 0.2).all().all()
    sessions = _panel(pipeline, "adj_close").index
    assert returns.index.equals(sessions[sessions >= returns.index[0]])
    blended = sum(returns[key] * weights[key] for key in SLEEVES)
    assert np.allclose(blended, returns["ENSEMBLE"], rtol=0, atol=1e-12)
    for key in SLEEVES:                # a blend run never replaces a sleeve's own record
        assert json.loads((pipeline.results / f"{key}.json").read_text()) \
            == json.loads(pipeline.runs[key].stdout)


def test_monitor_reaches_a_verdict_on_the_blended_returns(pipeline):
    done = _done(pipeline, "monitor", codes=(0, 1, 2))     # ok, REVIEW or KILL: all are verdicts
    out = done.stdout
    assert "mode: BACKTEST calibration" in out and "cash benchmark: ^IRX" in out
    rows = len(pd.read_csv(pipeline.results / "sleeve_returns.csv"))
    assert rows >= 504 and f"judged: 1 of {rows} session(s)" in out
    assert "rolling 2y excess Sharpe" in out and "pending" not in out
    verdict = re.search(r"^overall: (ok|REVIEW|KILL)$", out, re.M)
    assert verdict and done.returncode == ["ok", "REVIEW", "KILL"].index(verdict.group(1))


def test_live_targets_prints_an_offline_order_sheet_for_the_last_session(pipeline):
    done = _done(pipeline, "live_targets")
    out = done.stdout
    assert "OFFLINE historical simulation" in out and "cash $5,000.00 (as given)" in out
    assert f"decision close: {pipeline.last}" in out
    assert out.count("MONTH-END rebalance") == 2           # the final row is a month-end
    for key in SLEEVES:
        assert f"-- {key}  (weight " in out
    assert "== NET PORTFOLIO TARGET" in out and "== MOC ORDERS (target - ledger)" in out
    orders = re.findall(r"^   (?:BUY |SELL)\s+\d+ [A-Z]+\s+MOC", out, re.M)
    assert len(orders) >= 3 and f"sh {qdata.OPERATIONAL_UNIVERSE[0]}" in out
    assert "REFUSED" not in done.stderr
    assert (pipeline.project / "examples" / "ledger.example.csv").read_bytes() \
        == (EXAMPLES / "ledger.example.csv").read_bytes()  # the input ledger is never written


def test_reconcile_accepts_the_example_fills(pipeline):
    done = _done(pipeline, "reconcile")
    assert done.stdout.rstrip().endswith("ok: slippage and commissions within the modeled assumptions")


def test_selection_sweeps_feed_the_trial_registry(pipeline):
    assert len(pipeline.sweeps) == 5
    for name in pipeline.sweeps:
        _done(pipeline, name)
        grid = pd.read_csv(pipeline.results / name)
        sharpe = next(c for c in ("full_sharpe", "Sharpe", "sharpe") if c in grid.columns)
        assert len(grid) >= 9 and np.isfinite(grid[sharpe]).all(), name
    done = _done(pipeline, "trial_registry")
    registry = json.loads((pipeline.results / "trial_registry.json").read_text())
    logged = {p.name: len(pd.read_csv(p)) for p in pipeline.results.glob("*_variants.csv")}
    assert set(pipeline.sweeps) <= set(logged)
    assert registry["study"] is True and registry["n_families"] == len(logged)
    assert registry["n_trials_total"] == sum(logged.values())
    assert f"registry: {sum(logged.values())} logged trials" in done.stdout
    assert set(registry["sleeves"]) == set(SLEEVES)
    for sleeve in registry["sleeves"].values():
        assert 0 <= sleeve["dsr_own_grid"]["dsr"] <= 1 and 0 <= sleeve["oos_psr"]["psr"] <= 1
    ensemble = registry["ensemble"]
    assert set(ensemble["dsr"]) == {"families_only", "every_logged_trial"}
    for block in ("bootstrap_full", "bootstrap_oos"):
        low, high = ensemble[block]["ci"]
        assert math.isfinite(low) and math.isfinite(high) and low <= high


def test_nothing_outside_the_temporary_copy_was_written(pipeline):
    _done(pipeline, "generate")
    assert _fingerprint() == pipeline.before
    assert not (pipeline.project / "data").exists()        # the copy has no default cache at all
    written = {p.name for p in pipeline.results.iterdir()}
    assert {"ensemble.json", "sleeve_returns.csv", "trial_registry.json", "pairs_statarb.json",
            "tsmom_voltarget.json"} <= written and "recomputed" not in written
