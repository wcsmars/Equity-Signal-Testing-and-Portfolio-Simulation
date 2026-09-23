#!/usr/bin/env python3
"""Regenerate the tearsheet for an existing tracked run.

Usage:
    python scripts/make_report.py --run runs/<run_id> [--formats html,md]

Reads the stored config/metrics/result from the run directory produced by
scripts/run_backtest.py and rewrites <run>/report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Allow `python scripts/make_report.py` without an editable install: the
# repo root (parent of scripts/) hosts the alpha_lab package.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from alpha_lab.core.errors import AlphaLabError, ConfigError
from alpha_lab.core.results import BacktestResult
from alpha_lab.reports import generate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory, e.g. runs/<run_id>")
    parser.add_argument(
        "--formats",
        default=None,
        help="comma-separated subset of html,md (default: the run config's report.formats)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise ConfigError(f"no run directory at {run_dir}")

    result = BacktestResult.load(run_dir / "result")

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    config = config or {}

    if args.formats is not None:
        formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    else:
        formats = tuple((config.get("report") or {}).get("formats") or ("html", "md"))

    title = (config.get("experiment") or {}).get("name") or run_dir.name

    written = generate_report(result, metrics, run_dir / "report", formats=formats, title=title)
    for fmt, path in sorted(written.items()):
        print(f"report [{fmt}]: {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AlphaLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
