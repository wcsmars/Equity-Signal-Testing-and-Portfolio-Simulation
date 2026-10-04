"""Cached feature computation.

``FeatureStore`` resolves a ``FeatureSpec`` against a feature registry,
computes the panel, and memoizes it in memory keyed by (spec.key, data
fingerprint). With ``cache_dir`` set it also persists panels to disk as
parquet (csv fallback when pyarrow is unavailable).

The fingerprint covers every date, ticker and value in every input field.
Corrections to historical prices, membership or liquidity must invalidate
both memory and disk entries.

A disk entry is also keyed by the implementation that produced it: the
qualified name of the feature's type, the source text of that type and its
bases, and the functions they define as loaded in the running process (so
a session that imported a feature before its file was edited does not
write under the edited text's key; entries are specific to the Python
version that compiled them). Editing a feature's implementation, or
registering a different one under the same name, is therefore a cache
miss. Code a feature merely calls (module-level helpers, pandas itself) is
not tracked — the cache is not a full code-version provenance system:
clear it after changing such code.

Disk entries are written to a temporary file and renamed into place, so an
interrupted or concurrent run never leaves a half-written panel under the
final name. A file that cannot be read back is reported with a warning and
treated as a miss: the panel is recomputed and the file replaced.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import types
import uuid
import warnings
from abc import ABC
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from alpha_lab.core.errors import DataError
from alpha_lab.core.interfaces import Feature, FeatureSpec
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


def _hash_loaded_code(sha, klass: type) -> None:
    """Feed ``sha`` the functions ``klass`` defines, as loaded in this process.

    Bytecode, names and constants; no file names or line numbers. Sets are
    fed in sorted order because their ``repr`` follows the hash seed.
    """

    def feed(code: types.CodeType) -> None:
        sha.update(code.co_code)
        sha.update(repr((code.co_names, code.co_varnames, code.co_freevars)).encode())
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                feed(const)
            elif isinstance(const, frozenset):
                sha.update(repr(sorted(map(repr, const))).encode())
            else:
                sha.update(repr(const).encode())

    for name, attr in sorted(vars(klass).items()):
        func = getattr(attr, "__func__", attr)  # staticmethod, classmethod
        func = getattr(func, "fget", func)  # property
        code = getattr(func, "__code__", None)
        if isinstance(code, types.CodeType):
            sha.update(name.encode())
            feed(code)


def _code_token(feature: Feature) -> str:
    """Identity of the code behind ``feature``, for the disk cache key.

    The qualified name of the feature's type plus a hash of two things, for
    that type and for every base it inherits ``compute`` or parameters from:

    - the source text, read from the file now. Source that cannot be
      retrieved (a type defined interactively) contributes nothing;
    - the functions the type defines, as loaded. The file can be edited
      after a session imported it: that session still runs the old code, and
      its panel must not be stored under the key of the edited text. This
      part also tells two versions of a type without source apart.
    """
    cls = type(feature)
    sha = hashlib.sha256()
    for klass in cls.__mro__:
        if klass in (Feature, ABC, object):
            continue
        try:
            sha.update(inspect.getsource(klass).encode())
        except (OSError, TypeError):
            pass
        _hash_loaded_code(sha, klass)
    return f"{cls.__module__}.{cls.__qualname__}:{sha.hexdigest()[:16]}"


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
        # Resolve the feature BEFORE looking at the disk: an unknown name or
        # bad params must raise even when a panel is cached under that spec,
        # and the disk key includes the identity of the feature's code.
        feature = self.registry.create(spec.name, **spec.as_dict)
        if self.cache_dir is not None:
            token = _code_token(feature)
            frame = self._load_disk(spec, token, key[1], data)
        if frame is None:
            frame = self._compute(spec, feature, data)
            if self.cache_dir is not None:
                self._save_disk(spec, token, key[1], frame)
        self._memory[key] = frame
        return frame

    def compute_all(self, specs: Iterable[FeatureSpec], data: MarketData) -> dict[str, pd.DataFrame]:
        """Compute deduplicated ``specs`` in stable (first-seen) order.

        Returns {spec.key: panel} — the FeatureSet mapping signals consume.
        """
        return {spec.key: self.get(spec, data) for spec in dict.fromkeys(specs)}

    # -- internals ---------------------------------------------------------

    def _compute(self, spec: FeatureSpec, feature: Feature, data: MarketData) -> pd.DataFrame:
        frame = feature.compute(data)
        if not frame.index.equals(data.close.index) or not frame.columns.equals(data.close.columns):
            raise DataError(f"feature '{spec.key}' output is not aligned to close")
        return frame

    def _path(self, spec: FeatureSpec, token: str, data_fp: str, ext: str) -> Path:
        # spec keys contain characters unfriendly to filesystems; hash them
        digest = hashlib.sha256(f"{spec.key}|{token}|{data_fp}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{spec.name}-{digest}.{ext}"

    def _load_disk(
        self, spec: FeatureSpec, token: str, data_fp: str, data: MarketData
    ) -> pd.DataFrame | None:
        frame = self._read(self._path(spec, token, data_fp, "parquet"), pd.read_parquet)
        if frame is None:
            frame = self._read(
                self._path(spec, token, data_fp, "csv"),
                lambda path: pd.read_csv(
                    path, index_col=0, parse_dates=True, float_precision="round_trip"
                ),
            )
        return None if frame is None else self._aligned(frame, data)

    @staticmethod
    def _read(path: Path, reader: Callable[[Path], pd.DataFrame]) -> pd.DataFrame | None:
        """The panel stored at ``path``, or None if absent or unreadable."""
        if not path.exists():
            return None
        try:
            return reader(path)
        except ImportError:
            return None  # file written by an env that had pyarrow; try csv
        except Exception as exc:  # truncated, empty or otherwise corrupt
            warnings.warn(
                f"feature cache file {path.name} is unreadable "
                f"({type(exc).__name__}: {exc}); recomputing the panel",
                UserWarning,
                stacklevel=4,
            )
            return None

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

    def _save_disk(self, spec: FeatureSpec, token: str, data_fp: str, frame: pd.DataFrame) -> None:
        try:
            self._write(self._path(spec, token, data_fp, "parquet"), frame.to_parquet)
        except ImportError:
            self._write(self._path(spec, token, data_fp, "csv"), frame.to_csv)

    @staticmethod
    def _write(path: Path, writer: Callable[[Path], None]) -> None:
        """Write to a temporary sibling, then rename it over ``path``.

        The rename is atomic, so a reader sees the old file or the complete
        new one, never a partial panel; a failed write leaves ``path`` as it
        was and removes the temporary.
        """
        tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
        try:
            writer(tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
