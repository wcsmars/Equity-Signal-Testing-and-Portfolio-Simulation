#!/usr/bin/env python3
"""Regenerate the tearsheet for an existing tracked run.

Usage:
    python scripts/make_report.py --run runs/<run_id> [--formats html,md]

Reads the stored config and result from the run directory produced by
scripts/run_backtest.py and rewrites <run>/report. The metrics table is
recomputed from the stored result with the current code, so it always
describes the same period as the charts next to it; metrics.json itself is
never modified. When the recomputed values differ from the stored file a
note names the keys. --keep-stored-metrics prints metrics.json as it is.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

# Allow `python scripts/make_report.py` without an editable install: the
# repo root (parent of scripts/) hosts the alpha_lab package.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from alpha_lab.core.errors import AlphaLabError, ConfigError, ExperimentError
from alpha_lab.core.results import BacktestResult
from alpha_lab.reports import generate_report
from alpha_lab.risk import summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory, e.g. runs/<run_id>")
    parser.add_argument(
        "--formats",
        default=None,
        help="comma-separated subset of html,md (default: the run config's report.formats)",
    )
    parser.add_argument(
        "--keep-stored-metrics",
        action="store_true",
        help="print metrics.json as stored instead of recomputing the metrics table",
    )
    return parser


def load_run(run_dir: Path) -> tuple[BacktestResult, dict | None, dict]:
    """(result, stored metrics or None, config) of a run directory.

    metrics.json and config.yaml are optional; a file that is present but
    unreadable is an ExperimentError, not a traceback.
    """
    try:
        result = BacktestResult.load(run_dir / "result")
        metrics_path = run_dir / "metrics.json"
        stored = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        config_path = run_dir / "config.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    except AlphaLabError:
        raise
    except Exception as exc:
        raise ExperimentError(f"run directory {run_dir} is unreadable: {exc}") from exc
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ExperimentError(f"{run_dir / 'config.yaml'} does not hold a mapping")
    if stored is not None and not isinstance(stored, dict):
        raise ExperimentError(f"{run_dir / 'metrics.json'} does not hold a mapping")
    return result, stored, config


def _same(stored, fresh) -> bool:
    """Equal up to float round-off; null in the file equals a non-finite value."""
    def missing(value) -> bool:
        return value is None or (isinstance(value, float) and not math.isfinite(value))

    if missing(stored) or missing(fresh):
        return missing(stored) and missing(fresh)
    numeric = (int, float)
    if isinstance(stored, numeric) and isinstance(fresh, numeric):
        return math.isclose(stored, fresh, rel_tol=1e-9, abs_tol=1e-12)
    return stored == fresh


def changed_keys(stored: dict, fresh: dict) -> list[str]:
    """Metric names whose stored and recomputed values disagree, or that
    exist on one side only."""
    return [
        key
        for key in [*stored, *(k for k in fresh if k not in stored)]
        if key not in stored or key not in fresh or not _same(stored[key], fresh[key])
    ]


def _section(config: dict, name: str) -> dict:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise ConfigError(f"no run directory at {run_dir}")

    result, stored, config = load_run(run_dir)

    if args.keep_stored_metrics:
        if stored is None:
            raise ExperimentError(f"--keep-stored-metrics: no metrics.json in {run_dir}")
        metrics = stored
    else:
        metrics = summary(result, n_trials=(stored or {}).get("n_trials", 1))
        changed = changed_keys(stored, metrics) if stored is not None else []
        if changed:
            print(
                "note: metrics recomputed from the stored result differ from metrics.json "
                f"({', '.join(changed)}); the report shows the recomputed values and "
                "metrics.json is unchanged (--keep-stored-metrics prints the stored file)",
                file=sys.stderr,
            )

    if args.formats is not None:
        formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    else:
        formats = tuple(_section(config, "report").get("formats") or ("html", "md"))

    title = _section(config, "experiment").get("name") or run_dir.resolve().name

    written = generate_report(result, metrics, run_dir / "report", formats=formats, title=title)
    for fmt, path in sorted(written.items()):
        print(f"report [{fmt}]: {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AlphaLabError, OSError) as exc:
        # OSError: a report directory that cannot be written. The message
        # names the path; a traceback would add nothing.
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
