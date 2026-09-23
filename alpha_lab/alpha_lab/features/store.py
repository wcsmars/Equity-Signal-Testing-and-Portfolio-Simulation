"""Cached feature computation.

``FeatureStore`` resolves a ``FeatureSpec`` against a feature registry,
computes the panel, and memoizes it in memory keyed by (spec.key, data
fingerprint). With ``cache_dir`` set it also persists panels to disk as
parquet (csv fallback when pyarrow is unavailable).

The fingerprint is a *heuristic* identity, not a full content hash: it
covers the panel shape, first/last date, ticker tuple, and — for EVERY
field a feature can consume (close, open, high, low, volume,
unadjusted_close, universe) — the field's presence plus its values at
three sampled rows. Two datasets differing only in cells the sample
misses would collide — which is why the disk cache is opt-in and meant for
iterating on one dataset, not as a provenance system.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alpha_lab.core.errors import DataError
from alpha_lab.core.interfaces import FeatureSpec
from alpha_lab.core.registry import Registry
from alpha_lab.core.types import MarketData
from alpha_lab.features.library import FEATURES

#: substituted for NaN in sampled rows so hashing is deterministic
_NAN_SENTINEL = -1.2345678e300

#: every MarketData field a feature may consume — ALL of them must feed the
#: fingerprint, or two datasets sharing a close panel would collide and the
#: cache would silently serve a panel computed from the wrong data (e.g.
#: adv_dollars from another dataset's volume or unadjusted_close)
_FINGERPRINT_FIELDS = ("close", "open", "high", "low", "volume", "unadjusted_close", "universe")


def fingerprint(data: MarketData) -> str:
    """Heuristic sha256 identity of a MarketData panel.

    Hashes n_rows, n_cols, first/last date (isoformat), the ticker tuple,
    and — for every field of ``_FINGERPRINT_FIELDS`` — the field's presence
    (None vs set) plus the bytes of its values at rows 0, middle, and -1
    (NaNs replaced by a sentinel). Cheap and stable, but not exhaustive —
    see the module docstring.
    """
    close = data.close
    n_rows, n_cols = close.shape
    sha = hashlib.sha256()
    sha.update(f"{n_rows}x{n_cols}".encode())
    sha.update(repr(tuple(close.columns)).encode())
    if n_rows > 0:
        sha.update(close.index[0].isoformat().encode())
        sha.update(close.index[-1].isoformat().encode())
    sample_rows = (0, n_rows // 2, n_rows - 1) if n_rows > 0 else ()
    for name in _FINGERPRINT_FIELDS:
        frame = getattr(data, name)
        sha.update(f"|{name}:{int(frame is not None)}".encode())
        if frame is None:
            continue
        for row in sample_rows:
            vals = frame.iloc[row].to_numpy(dtype=float, copy=True)
            vals[np.isnan(vals)] = _NAN_SENTINEL
            sha.update(vals.tobytes())
    return sha.hexdigest()


class FeatureStore:
    """Computes feature panels with in-memory memoization and optional disk cache.

    Returned frames are shared with the cache — treat them as read-only.
    """

    def __init__(self, registry: Registry = FEATURES, cache_dir: str | Path | None = None) -> None:
        self.registry = registry
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[tuple[str, str], pd.DataFrame] = {}

    # -- public API ------------------------------------------------------

    def get(self, spec: FeatureSpec, data: MarketData) -> pd.DataFrame:
        """Return the panel for ``spec`` on ``data``, computing at most once.

        Lookup order: in-memory memo, then disk cache (if enabled), then
        compute (persisting to disk if enabled). Raises ConfigError for an
        unknown feature name or bad params, DataError if a feature returns
        a panel not aligned to ``data.close``.
        """
        key = (spec.key, fingerprint(data))
        frame = self._memory.get(key)
        if frame is not None:
            return frame
        if self.cache_dir is not None:
            frame = self._load_disk(spec, key[1])
        if frame is None:
            frame = self._compute(spec, data)
            if self.cache_dir is not None:
                self._save_disk(spec, key[1], frame)
        self._memory[key] = frame
        return frame

    def compute_all(self, specs: Iterable[FeatureSpec], data: MarketData) -> dict[str, pd.DataFrame]:
        """Compute deduplicated ``specs`` in stable (first-seen) order.

        Returns {spec.key: panel} — the FeatureSet mapping signals consume.
        """
        return {spec.key: self.get(spec, data) for spec in dict.fromkeys(specs)}

    # -- internals ---------------------------------------------------------

    def _compute(self, spec: FeatureSpec, data: MarketData) -> pd.DataFrame:
        feature = self.registry.create(spec.name, **spec.as_dict)
        frame = feature.compute(data)
        if not frame.index.equals(data.close.index) or not frame.columns.equals(data.close.columns):
            raise DataError(f"feature '{spec.key}' output is not aligned to close")
        return frame

    def _path(self, spec: FeatureSpec, data_fp: str, ext: str) -> Path:
        # spec keys contain characters unfriendly to filesystems; hash them
        digest = hashlib.sha256(f"{spec.key}|{data_fp}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{spec.name}-{digest}.{ext}"

    def _load_disk(self, spec: FeatureSpec, data_fp: str) -> pd.DataFrame | None:
        parquet = self._path(spec, data_fp, "parquet")
        if parquet.exists():
            try:
                return pd.read_parquet(parquet)
            except ImportError:
                pass  # file written by an env that had pyarrow; try csv
        csv = self._path(spec, data_fp, "csv")
        if csv.exists():
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        return None

    def _save_disk(self, spec: FeatureSpec, data_fp: str, frame: pd.DataFrame) -> None:
        try:
            frame.to_parquet(self._path(spec, data_fp, "parquet"))
        except ImportError:
            frame.to_csv(self._path(spec, data_fp, "csv"))
