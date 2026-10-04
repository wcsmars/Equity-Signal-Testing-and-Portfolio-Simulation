# alpha_lab

Research toolkit for daily equity alpha signals: point-in-time
data handling, a cached feature store, a signal library, a walk-forward
backtester with transaction costs, portfolio construction, experiment tracking,
tearsheet reports, and a leakage/bias test suite.

**Read [CONVENTIONS.md](CONVENTIONS.md) first.** It defines the timing and data
assumptions (what information may be used when, default `execution_lag=2`) used
by every module. The timing, bias, and leakage tests check specific regressions
against them; passing them does not prove an arbitrary signal is bias-free.

## Layout

```
alpha_lab/
  alpha_lab/           # the library
    core/              # contracts: MarketData, interfaces, BacktestResult, registry, errors
    config/            # typed config schema + strict YAML loader
    data/              # data sources (CSV, synthetic), validation, point-in-time store
    features/          # feature library + cached FeatureStore
    signals/           # signal library + registry (example: 12-1 cross-sectional momentum)
    backtest/          # walk-forward engine, splitter, transaction cost models
    portfolio/         # construction (quantile long-short) + constraints + vol targeting
    risk/              # performance metrics incl. PSR / deflated Sharpe
    experiments/       # run tracker: config hash, artifacts, registry.jsonl
    reports/           # HTML/markdown tearsheet generator
    testing/           # importable leakage/bias check harness (use it on YOUR new signals)
  configs/             # YAML run configs
  tests/               # unit tests + leakage/bias suite
  scripts/             # run_backtest.py, make_report.py
  notebooks/           # exploratory work (results stay out of the library)
  data/                # local data files (gitignored)
  runs/                # experiment artifacts (gitignored)
```

## Quickstart

Python 3.11 or newer (`requires-python` in `pyproject.toml`; the tests run on
3.11 and 3.12).

```bash
cd alpha_lab
pip install -e '.[dev]'          # or just ensure numpy/pandas/pyyaml/matplotlib on PYTHONPATH

# run the example: 12-1 cross-sectional momentum, walk-forward, with costs
python scripts/run_backtest.py --config configs/base.yaml

# one configuration out of a sweep: say how many were tried, so the deflated
# Sharpe ratio (dsr) is deflated by that number (default 1: dsr equals psr)
python scripts/run_backtest.py --config configs/base.yaml \
    --override signal.params.window=126 --n-trials 12

# the run directory it prints contains config.yaml, metrics.json, env.json,
# result/, report/
# rebuild a report for an existing run:
python scripts/make_report.py --run runs/<run_id>

# tests, including the leakage suite
python -m pytest
```

`run_backtest.py` exits 0 on success, 1 on a configuration, data or
file-system error (a missing config file, an unwritable runs directory; each
is printed as one `error: ...` line) and 2 when market-data validation
reports errors (`--force` runs anyway and marks the run as forced in
`result/meta.json` and in its report). It warns when a run never holds a
position.

A run directory is a self-contained record: `metrics.json` is strict JSON
(a value that is not finite is stored as `null`), `env.json` lists the
interpreter and library versions and, in a git checkout, the commit of the
code, and the stored config holds data/run/cache locations relative to the
run directory instead of absolute paths. `make_report.py` recomputes the
metrics table from the stored result, so it always describes the same period
as the charts; it never rewrites `metrics.json` and prints a note when the
stored values differ (`--keep-stored-metrics` prints the stored file as is).
A report directory holds one build: rebuilding removes report files of an
earlier build that the new one does not write.

The default config uses the built-in synthetic data source (no downloads, no
API keys), so the example runs out of the box. Point `data.source: csv` at
your own files for real research. A config that sets `data.path` or
`data.format: long` must also set `data.source: csv`: the source defaults to
`synthetic`, which reads neither, so such a config is rejected when it is
loaded instead of silently running on the synthetic panel. Dates in the CSV
files must be ISO 8601 (`YYYY-MM-DD`, optionally with a time of day, or
`YYYYMMDD`); any other spelling, such as `10/01/2024`, is a data error,
because the day/month order is never guessed.

## Core flow

```
DataSource -> MarketData -> FeatureStore -> Signal.score -> PortfolioConstructor
          -> BacktestEngine (walk-forward, H_t = W_{t-lag}, costs) -> BacktestResult
          -> risk.metrics.summary -> ExperimentTracker -> reports.performance
```

Everything is wired by name through registries (`features.library.FEATURES`,
`signals.registry.SIGNALS`, cost models, constructors), so a new signal is:
implement `Signal`, register it, write a config, run. Before trusting it, run
the leakage harness on it (`alpha_lab.testing.checks`).

The truncation checks compare a sample of dates, not every date. By default
that is eight dates spread across the sample (the last date is left out: the
comparison could not fail there) plus the warm-up rows: the first two dates,
one inside the first sixth, and the first dates on which the output holds a
value and a non-zero value. Pass `dates=` to check others. `assert_cost_pit`
samples from the dates that carry a trade and raises when no checked date
does, so it cannot pass on an empty trade panel.

## Adding a signal — checklist

1. Features it needs exist in `features/library.py` (each passes truncation
   invariance in `tests/`).
2. `Signal` subclass registered in `signals/registry.py`; `fit` uses train
   data only; `score` is a pure per-date map.
3. `assert_signal_pit(...)` from `alpha_lab.testing.checks` passes.
4. Backtest via config with `execution_lag >= 1` and realistic costs.
5. Log the run with the tracker; judge with deflated Sharpe given the number
   of trials you burned, not raw Sharpe: pass that number as `--n-trials N`
   (`summary(result, n_trials=N)` in code). Without it `dsr` is not deflated.
