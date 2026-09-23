"""Experiment tracking: one directory per run plus a JSONL registry index.

Layout under ``runs_dir``::

    runs_dir/
      registry.jsonl          # one compact JSON line per run (the index)
      <run_id>/
        config.yaml           # full resolved config (config.loader.save_config)
        metrics.json          # the full metrics dict, verbatim
        env.json              # python/numpy/pandas/platform/alpha_lab versions
        result/               # BacktestResult.save() output

``run_id`` = UTC timestamp + first 8 hex of the config hash + slugged name,
so ids sort chronologically and identical configs are visually groupable.
"""

from __future__ import annotations

import json
import platform
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

import alpha_lab
from alpha_lab.config.loader import config_hash, save_config
from alpha_lab.config.schema import AlphaLabConfig
from alpha_lab.core.errors import ExperimentError
from alpha_lab.core.results import BacktestResult

#: metric keys copied into each registry line; the full dict stays in metrics.json
_REGISTRY_METRICS = ("sharpe_net", "ann_return_net", "max_drawdown", "n_days")

#: stable column order for list_runs()
REGISTRY_COLUMNS = ("run_id", "ts", "name", "config_hash") + _REGISTRY_METRICS

_SLUG_RE = re.compile(r"[^a-z0-9\-_]+")


def _utcnow() -> datetime:
    """Current UTC time. Module-level indirection so tests can monkeypatch it."""
    return datetime.now(timezone.utc)


def slug(s: str) -> str:
    """Lowercase ``s``; runs of characters outside [a-z0-9-_] become one '-'."""
    out = _SLUG_RE.sub("-", s.lower())
    return out or "run"


def _json_default(obj):
    """JSON fallback: numpy scalars via .item(), everything else via str()."""
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(obj)


@dataclass
class RunRecord:
    """Handle returned by :meth:`ExperimentTracker.log_run`."""

    run_id: str
    path: Path
    config_hash: str
    metrics: dict


class ExperimentTracker:
    """Persists backtest runs under ``runs_dir`` and indexes them in JSONL.

    The registry file is append-only; each line is the compact index entry
    for one run. Corrupt lines are skipped (with a warning) on read so a
    partial write can never brick the registry.
    """

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)

    @property
    def registry_path(self) -> Path:
        return self.runs_dir / "registry.jsonl"

    # -- writing --------------------------------------------------------

    def log_run(
        self,
        cfg: AlphaLabConfig,
        result: BacktestResult,
        metrics: dict,
        name: str | None = None,
    ) -> RunRecord:
        """Persist one run (config, metrics, env, result) and index it.

        Returns a RunRecord whose ``run_id`` is unique within ``runs_dir``:
        on collision (same timestamp + hash + name) '-2', '-3', ... is
        appended.
        """
        chash = config_hash(cfg)
        run_name = name or cfg.experiment.name
        base = "_".join([_utcnow().strftime("%Y%m%d-%H%M%S"), chash[:8], slug(run_name)])
        run_id, n = base, 2
        while (self.runs_dir / run_id).exists():
            run_id = f"{base}-{n}"
            n += 1
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True)

        save_config(cfg, run_dir / "config.yaml")
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=_json_default)
        )
        env = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "alpha_lab": alpha_lab.__version__,
        }
        (run_dir / "env.json").write_text(json.dumps(env, indent=2))
        result.save(run_dir / "result")

        line = {
            "run_id": run_id,
            "ts": _utcnow().isoformat(),
            "name": run_name,
            "config_hash": chash,
        }
        line.update({k: metrics.get(k) for k in _REGISTRY_METRICS})
        with self.registry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=_json_default) + "\n")

        return RunRecord(run_id=run_id, path=run_dir, config_hash=chash, metrics=metrics)

    # -- reading --------------------------------------------------------

    def load_run(self, run_id: str) -> dict:
        """Load one run back: {config: dict, metrics: dict, result, path}."""
        run_dir = self.runs_dir / run_id
        if not run_dir.is_dir():
            raise ExperimentError(f"no run '{run_id}' under {self.runs_dir}")
        try:
            config = yaml.safe_load((run_dir / "config.yaml").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            result = BacktestResult.load(run_dir / "result")
        except ExperimentError:
            raise
        except Exception as exc:
            raise ExperimentError(f"run '{run_id}' is unreadable: {exc}") from exc
        return {"config": config, "metrics": metrics, "result": result, "path": run_dir}

    def list_runs(self) -> pd.DataFrame:
        """All registry entries as a DataFrame with stable columns.

        Missing registry file => empty frame. Corrupt lines are skipped with
        a warning — the registry read never crashes.
        """
        cols = list(REGISTRY_COLUMNS)
        if not self.registry_path.exists():
            return pd.DataFrame(columns=cols)
        rows = []
        for lineno, raw in enumerate(self.registry_path.read_text().splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
                if not isinstance(rec, dict):
                    raise ValueError("line is not a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                warnings.warn(
                    f"skipping corrupt line {lineno} of {self.registry_path}: {exc}"
                )
                continue
            rows.append(rec)
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    def compare(
        self,
        run_ids: Iterable[str],
        keys: Sequence[str] = ("sharpe_net", "ann_return_net", "max_drawdown"),
    ) -> pd.DataFrame:
        """Side-by-side metric table (index = run_id, columns = keys).

        Values come from each run's metrics.json (the full dict), not from
        the compact registry line; a key absent from a run's metrics is NaN.
        """
        rows: dict[str, dict] = {}
        for run_id in run_ids:
            path = self.runs_dir / run_id / "metrics.json"
            if not path.exists():
                raise ExperimentError(f"no run '{run_id}' under {self.runs_dir}")
            try:
                metrics = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ExperimentError(f"run '{run_id}' has corrupt metrics.json: {exc}") from exc
            rows[run_id] = {k: metrics.get(k) for k in keys}
        frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(keys))
        frame.index.name = "run_id"
        return frame
