"""Experiment tracking: one directory per run plus a JSONL registry index.

Layout under ``runs_dir``::

    runs_dir/
      registry.jsonl          # one compact JSON line per run (the index)
      <run_id>/
        config.yaml           # full resolved config, locations made portable
        metrics.json          # the full metrics dict
        env.json              # interpreter, library and code versions
        result/               # BacktestResult.save() output

``run_id`` = UTC timestamp + first 8 hex of the config hash + slugged name,
so ids sort chronologically and identical configs are visually groupable.

Stored files are strict JSON: a non-finite metric (NaN Sharpe of a flat
series, say) is written as ``null``, never as the bare token ``NaN`` that
only lenient parsers accept. Stored configs hold no absolute path: data, run
and cache locations are written relative to the run directory, which is how
``config.loader.load_config`` reads a relative path back.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import platform
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

import alpha_lab
from alpha_lab.config.loader import config_hash
from alpha_lab.config.schema import AlphaLabConfig
from alpha_lab.core.errors import ExperimentError
from alpha_lab.core.results import BacktestResult

#: metric keys copied into each registry line; the full dict stays in metrics.json
_REGISTRY_METRICS = ("sharpe_net", "ann_return_net", "max_drawdown", "n_days")

#: stable column order for list_runs()
REGISTRY_COLUMNS = ("run_id", "ts", "name", "config_hash") + _REGISTRY_METRICS

#: config entries that hold a filesystem location
_PATH_KEYS = (("data", "path"), ("experiment", "runs_dir"), ("experiment", "feature_cache_dir"))

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
            value = item()
        except Exception:
            pass
        else:
            return None if isinstance(value, float) and not math.isfinite(value) else value
    return str(obj)


def _finite_json(obj: Any) -> Any:
    """Copy of ``obj`` with every non-finite float replaced by None.

    NaN and +/-Infinity have no JSON representation (RFC 8259). ``json.dumps``
    writes them as bare ``NaN`` / ``Infinity`` tokens by default, which Python
    reads back but strict parsers reject; ``null`` is the portable spelling of
    "no value" and is what list_runs/compare already treat as missing.
    """
    if isinstance(obj, dict):
        return {key: _finite_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_json(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _finite_json(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _dumps(obj: Any, **kwargs) -> str:
    """Strict JSON text: non-finite floats as null, never NaN/Infinity."""
    return json.dumps(_finite_json(obj), default=_json_default, allow_nan=False, **kwargs)


def _portable_config(config: dict, run_dir: Path) -> dict:
    """Copy of ``config`` whose absolute locations are relative to ``run_dir``.

    ``load_config`` resolves the data, run and feature-cache paths, so a
    config saved as loaded would write the user's home directory into every
    run. Relative to the run directory the entries say the same thing without
    naming it: ``load_config`` resolves a relative path against the directory
    of the file it reads, so loading the stored ``config.yaml`` gives the
    original locations (and the same config hash) back. A path with no
    relative form (another drive) is kept as it is.
    """
    out = copy.deepcopy(config)
    base = Path(run_dir).resolve()
    for section, key in _PATH_KEYS:
        node = out.get(section)
        value = node.get(key) if isinstance(node, dict) else None
        if isinstance(value, str) and os.path.isabs(value):
            try:
                node[key] = Path(os.path.relpath(Path(value).resolve(), base)).as_posix()
            except ValueError:
                pass
    return out


def _git_revision(package_dir: Path) -> str | None:
    """Commit of the checkout that holds ``package_dir``, or None.

    The package version is a constant, so two runs made from different code
    states would otherwise carry identical provenance. Best effort: None when
    git is not installed, the directory is not in a work tree, or the
    package's own ``__init__.py`` is not tracked there (an installed copy that
    merely sits inside some unrelated repository). '+dirty' is appended when
    tracked files under the package differ from that commit.
    """
    def git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(package_dir), *args],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    try:
        if git("ls-files", "--error-unmatch", "__init__.py") is None:
            return None
        head = git("rev-parse", "HEAD")
        if not head:
            return None
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".")
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{head}+dirty" if dirty else head


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
    partial write can never brick the registry, and an append after a partial
    write starts on a fresh line so the new record is not lost with it.
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
        appended. If any step fails the run directory is removed again before
        the error propagates, so a directory under ``runs_dir`` is always a
        complete, indexed run.
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
        try:
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(_portable_config(cfg.to_dict(), run_dir), sort_keys=False)
            )
            (run_dir / "metrics.json").write_text(_dumps(metrics, indent=2))
            env = {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                "yaml": yaml.__version__,
                "platform": platform.platform(),
                "alpha_lab": alpha_lab.__version__,
            }
            revision = _git_revision(Path(alpha_lab.__file__).resolve().parent)
            if revision:
                env["git_revision"] = revision
            (run_dir / "env.json").write_text(json.dumps(env, indent=2))
            stored = result
            if isinstance(result.config, dict):
                stored = dataclasses.replace(
                    result, config=_portable_config(result.config, run_dir)
                )
            stored.save(run_dir / "result")

            line = {
                "run_id": run_id,
                "ts": _utcnow().isoformat(),
                "name": run_name,
                "config_hash": chash,
            }
            line.update({k: metrics.get(k) for k in _REGISTRY_METRICS})
            self._append_registry(_dumps(line))
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        return RunRecord(run_id=run_id, path=run_dir, config_hash=chash, metrics=metrics)

    def _append_registry(self, record: str) -> None:
        """Append one index line, starting on a line of its own.

        An interrupted earlier write can leave the file without a final
        newline; appending straight after it would glue this record onto the
        partial one and the reader would discard both.
        """
        path = self.registry_path
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    record = "\n" + record
        with path.open("a", encoding="utf-8") as fh:
            fh.write(record + "\n")

    # -- reading --------------------------------------------------------

    def load_run(self, run_id: str) -> dict:
        """Load one run back: {config: dict, metrics: dict, result, path}.

        ``metrics`` holds None where the logged value was not finite, and the
        locations in ``config`` are relative to the run directory.
        """
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
