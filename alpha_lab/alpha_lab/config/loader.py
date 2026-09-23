"""YAML -> AlphaLabConfig, with dotted-key overrides and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from alpha_lab.config.schema import AlphaLabConfig, build_dataclass
from alpha_lab.core.errors import ConfigError


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> AlphaLabConfig:
    """Load and validate a config file.

    ``overrides`` uses dotted keys, e.g. {"backtest.execution_lag": 2} —
    handy for parameter sweeps without writing N config files.

    Relative data, run, and feature-cache paths (including overrides) are
    resolved against the config file's directory, independent of the cwd.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    if not p.is_file():
        raise ConfigError(f"config path is not a file: {p}")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, value)
    cfg = build_dataclass(AlphaLabConfig, raw, path=str(p))
    base = p.resolve().parent
    for section, name in (
        (cfg.data, "path"),
        (cfg.experiment, "runs_dir"),
        (cfg.experiment, "feature_cache_dir"),
    ):
        value = getattr(section, name)
        if value is not None:
            location = Path(value).expanduser()
            if not location.is_absolute():
                location = base / location
            setattr(section, name, str(location.resolve()))
    return cfg


def config_from_dict(raw: dict, path: str = "config") -> AlphaLabConfig:
    return build_dataclass(AlphaLabConfig, raw, path=path)


def save_config(cfg: AlphaLabConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


def config_hash(cfg: AlphaLabConfig | dict) -> str:
    """12-hex digest of the canonicalized config — the experiment identity."""
    d = cfg.to_dict() if isinstance(cfg, AlphaLabConfig) else cfg
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _set_dotted(raw: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = raw
    for part in parts[:-1]:
        nxt = node.get(part)
        if nxt is None:
            nxt = node[part] = {}
        elif not isinstance(nxt, dict):
            raise ConfigError(f"override '{dotted}': '{part}' is not a mapping")
        node = nxt
    node[parts[-1]] = value
