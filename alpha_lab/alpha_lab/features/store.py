"""Cached feature computation.

``FeatureStore`` resolves a ``FeatureSpec`` against a feature registry,
computes the panel, and memoizes it in memory keyed by (spec.key, data
fingerprint). With ``cache_dir`` set it also persists panels to disk as
parquet (csv fallback when pyarrow is unavailable).

The fingerprint covers every date, ticker and value in every input field.
Corrections to historical prices, membership or liquidity must invalidate
both memory and disk entries. The cache is not a code-version provenance
system: clear it when changing a feature's implementation.
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

#: every MarketData field a feature may consume — ALL of them must feed the
#: fingerprint, or two datasets sharing a close panel would collide and the
#: cache would silently serve a panel computed from the wrong data (e.g.
#: adv_dollars from another dataset's volume or unadjusted_close)
_FINGERPRINT_FIELDS = ("close", "open", "high", "low", "volume", "unadjusted_close", "universe")


def fingerprint(data: MarketData) -> str:
    """Content identity of all inputs a feature can consume.

    Missingness is hashed separately from values, avoiding collisions with
    any finite sentinel value. NaN payload bits and numeric storage dtypes
    do not change the identity of otherwise equal daily panels.
    """
    close = data.close
    n_rows, n_cols = close.shape
    sha = hashlib.sha256()
    sha.update(f"market-data-v2|{n_rows}x{n_cols}".encode())
    sha.update(repr(tuple(close.columns)).encode())
    sha.update(close.index.as_unit("ns").asi8.tobytes())
    for name in _FINGERPRINT_FIELDS:
        frame = getattr(data, name)
        sha.update(f"|{name}:{int(frame is not None)}".encode())
        if frame is None:
            continue
        vals = frame.to_numpy(dtype=float, copy=True, na_value=np.nan)
        missing = np.isnan(vals)
        sha.update(missing.tobytes())
        vals[missing] = 0.0
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
            frame = self._load_disk(spec, key[1], data)
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

    def _load_disk(self, spec: FeatureSpec, data_fp: str, data: MarketData) -> pd.DataFrame | None:
        frame = None
        parquet = self._path(spec, data_fp, "parquet")
        if parquet.exists():
            try:
                frame = pd.read_parquet(parquet)
            except ImportError:
                pass  # file written by an env that had pyarrow; try csv
        csv = self._path(spec, data_fp, "csv")
        if frame is None and csv.exists():
            frame = pd.read_csv(csv, index_col=0, parse_dates=True, float_precision="round_trip")
        return None if frame is None else self._aligned(frame, data)

    @staticmethod
    def _aligned(frame: pd.DataFrame, data: MarketData) -> pd.DataFrame | None:
        """A cached panel relabelled to ``data.close``, or None (recompute).

        The csv fallback turns non-string ticker labels (e.g. integer
        security ids) into strings; a label-only mismatch is repaired so
        signals still align to the universe. Any other mismatch is a miss.
        """
        close = data.close
        if not frame.index.equals(close.index) or frame.shape != close.shape:
            return None
        if not frame.columns.equals(close.columns):
            if [str(c) for c in frame.columns] != [str(c) for c in close.columns]:
                return None
            frame = frame.set_axis(close.columns, axis=1)
        return frame

    def _save_disk(self, spec: FeatureSpec, data_fp: str, frame: pd.DataFrame) -> None:
        try:
            frame.to_parquet(self._path(spec, data_fp, "parquet"))
        except ImportError:
            frame.to_csv(self._path(spec, data_fp, "csv"))
