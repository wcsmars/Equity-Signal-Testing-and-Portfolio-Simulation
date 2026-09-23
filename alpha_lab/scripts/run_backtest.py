#!/usr/bin/env python3
"""Canonical backtest runner: config -> data -> features -> signal -> engine
-> metrics -> tracked run directory -> tearsheet.

Usage:
    python scripts/run_backtest.py --config configs/base.yaml
    python scripts/run_backtest.py --config configs/base.yaml \
        --override costs.model=zero
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow `python scripts/run_backtest.py` without an editable install: the
# repo root (parent of scripts/) hosts the alpha_lab package.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from alpha_lab.backtest import BacktestEngine
from alpha_lab.backtest import from_config as cost_from_config
from alpha_lab.config import load_config
from alpha_lab.core.errors import AlphaLabError
from alpha_lab.core.interfaces import FeatureSpec
from alpha_lab.data import source_from_config, validate_market
from alpha_lab.experiments import ExperimentTracker
from alpha_lab.features import FeatureStore
from alpha_lab.portfolio import from_config as portfolio_from_config
from alpha_lab.reports import generate_report
from alpha_lab.risk import summary
from alpha_lab.signals import build_signal

#: metrics shown in the console table, in display order
_TABLE_KEYS = (
    "sharpe_net",
    "sharpe_gross",
    "ann_return_net",
    "ann_vol",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "psr",
    "turnover_ann",
    "cost_drag_ann",
    "n_days",
    "n_windows",
    "mode",
)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ['a.b=1', 'c=null'] into {'a.b': 1, 'c': None}.

    Values go through yaml.safe_load so numbers, booleans and null coerce to
    the natural Python types; anything unparseable stays a string.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--override expects KEY=VALUE, got {pair!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        overrides[key.strip()] = value
    return overrides


def format_metrics_table(metrics: dict) -> str:
    """Compact two-column table of the headline metrics."""
    rows = []
    for key in _TABLE_KEYS:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            text = f"{value:.4f}"
        else:
            text = str(value)
        rows.append((key, text))
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML run config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (values parsed as YAML)",
    )
    parser.add_argument(
        "--runs-dir", default=None,
        help="override experiment.runs_dir (relative to the current directory)",
    )
    parser.add_argument("--no-report", action="store_true", help="skip tearsheet generation")
    parser.add_argument(
        "--force",
        action="store_true",
        help="proceed even if market-data validation reports errors",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = parse_overrides(args.override)
    if args.runs_dir is not None:
        overrides["experiment.runs_dir"] = str(Path(args.runs_dir).expanduser().resolve())
    cfg = load_config(args.config, overrides=overrides)

    data = source_from_config(cfg.data).load()

    report = validate_market(data)
    print(report.summary())
    if not report.ok:
        if not args.force:
            print("validation errors found; rerun with --force to proceed anyway", file=sys.stderr)
            return 2
        print("--force given: proceeding despite validation errors", file=sys.stderr)

    signal = build_signal(cfg.signal)
    specs = list(signal.required_features) + [
        FeatureSpec.make(item.name, **item.params) for item in cfg.features
    ]
    features = FeatureStore(cache_dir=cfg.experiment.feature_cache_dir).compute_all(specs, data)

    constructor = portfolio_from_config(cfg.portfolio)
    cost_model = cost_from_config(cfg.costs)

    engine = BacktestEngine(
        signal,
        constructor,
        cost_model,
        cfg.backtest,
        portfolio_value=cfg.costs.portfolio_value,
    )
    result = engine.run(data, features)
    result.config = cfg.to_dict()

    metrics = summary(result)

    tracker = ExperimentTracker(cfg.experiment.runs_dir)
    record = tracker.log_run(cfg, result, metrics)

    # Print the run pointer BEFORE report generation: if the report step
    # fails, the operator must still see where the persisted run lives.
    print(f"run_id: {record.run_id}")
    print(f"run path: {record.path}")

    if not args.no_report:
        report_dir = record.path / "report"
        written = generate_report(
            result,
            metrics,
            report_dir,
            formats=tuple(cfg.report.formats),
            title=cfg.experiment.name,
        )
        for fmt, path in sorted(written.items()):
            print(f"report [{fmt}]: {path}")

    print("metrics:")
    print(format_metrics_table(metrics))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AlphaLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
