"""Point-in-time snapshot store: save a MarketData panel under an immutable id.

Layout: ``root/<name>/<snapshot>/`` with one CSV per non-None field
(``close.csv``, ``volume.csv``, ...) plus a ``manifest.json`` recording shape,
date range, tickers, save time, and the sha256 of every field's CSV bytes, so
a loaded snapshot is verified to be the panel that was saved.

The store is a standalone utility. No config key selects a snapshot and a run
does not record a snapshot id or data checksum, so linking a result to the
data it used is left to the caller.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_lab.core.errors import DataError
from alpha_lab.core.types import MarketData
from alpha_lab.data.sources import _parse_universe

#: save order = MarketData field order; close always first
_FIELDS = ("close", "open", "high", "low", "volume", "unadjusted_close", "universe")


class MarketDataStore:
    """Snapshot store rooted at a directory; existing snapshots cannot be overwritten."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, data: MarketData, name: str, snapshot: str | None = None) -> Path:
        """Write one snapshot and return its directory.

        ``snapshot`` defaults to the current UTC time as ``%Y%m%d-%H%M%S`` so
        lexicographic order equals chronological order.
        """
        snap = snapshot or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._check_component(name)
        self._check_component(snap)
        data.validate()
        if not len(data.dates) or not len(data.tickers):
            # a header-only CSV has no dates to parse back, so load() would fail
            raise DataError("cannot snapshot an empty panel")
        out = self.root / name / snap
        if out.exists():
            raise DataError(f"snapshot already exists: {out}; choose a new snapshot id")
        out.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete snapshots. A killed process can leave a
        # hidden pending directory, which listing/loading deliberately ignores.
        try:
            with tempfile.TemporaryDirectory(prefix=".pending-", dir=out.parent) as pending:
                stage = Path(pending)
                fields: list[str] = []
                field_hashes = {}
                for field_name in _FIELDS:
                    frame = getattr(data, field_name)
                    if frame is None:
                        continue
                    csv_bytes = frame.to_csv(index_label="date").encode("utf-8")
                    (stage / f"{field_name}.csv").write_bytes(csv_bytes)
                    field_hashes[field_name] = hashlib.sha256(csv_bytes).hexdigest()
                    fields.append(field_name)
                manifest = {
                    "fields": fields,
                    "n_rows": int(len(data.dates)),
                    "n_cols": int(len(data.tickers)),
                    "start": data.dates[0].isoformat() if len(data.dates) else None,
                    "end": data.dates[-1].isoformat() if len(data.dates) else None,
                    "tickers": data.tickers,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "close_sha256": field_hashes["close"],  # readable by earlier clients
                    "field_sha256": field_hashes,
                }
                (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
                if out.exists():
                    raise DataError(f"snapshot already exists: {out}; choose a new snapshot id")
                # The staging directory is created with mode 0700 whatever the
                # umask; publish it with the dataset directory's permissions.
                stage.chmod(out.parent.stat().st_mode & 0o777)
                stage.rename(out)
        except OSError as exc:
            raise DataError(f"could not save snapshot {out}: {exc}") from exc
        return out

    def load(self, name: str, snapshot: str | None = None) -> MarketData:
        """Load one snapshot; ``snapshot=None`` resolves to the latest by name."""
        snap_dir = self._snapshot_dir(name, snapshot)
        manifest_path = snap_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                fields = manifest["fields"]
            except (ValueError, KeyError, TypeError) as exc:
                raise DataError(f"invalid snapshot manifest at {manifest_path}") from exc
        else:
            fields = [f for f in _FIELDS if (snap_dir / f"{f}.csv").exists()]
        if not isinstance(fields, list) or any(field not in _FIELDS for field in fields) or len(set(fields)) != len(fields):
            raise DataError(f"invalid snapshot fields at {snap_dir}")
        if "close" not in fields or not (snap_dir / "close.csv").exists():
            raise DataError(f"snapshot {snap_dir} has no close.csv")
        hashes = manifest.get("field_sha256") or {}
        if not isinstance(hashes, dict):
            raise DataError(f"invalid snapshot checksums at {snap_dir}")
        # Older snapshots pinned only close.csv. Verify that checksum too.
        if manifest.get("close_sha256"):
            hashes.setdefault("close", manifest["close_sha256"])
        frames = {}
        for field_name in fields:
            file = snap_dir / f"{field_name}.csv"
            if not file.is_file():
                raise DataError(f"snapshot is missing {file.name}: {snap_dir}")
            if field_name in hashes and hashlib.sha256(file.read_bytes()).hexdigest() != hashes[field_name]:
                raise DataError(f"snapshot checksum mismatch for {file.name}: {snap_dir}")
            try:
                panel = pd.read_csv(file, index_col="date", parse_dates=["date"], float_precision="round_trip")
            except (ValueError, pd.errors.ParserError) as exc:
                raise DataError(f"cannot read snapshot field {file}") from exc
            if "tickers" in manifest:
                tickers = manifest["tickers"]
                if [str(t) for t in tickers] != list(panel.columns):
                    raise DataError(f"snapshot ticker mismatch in {file.name}")
                panel.columns = tickers
            frames[field_name] = panel
        close = frames.pop("close")
        if manifest and (manifest.get("n_rows") != len(close) or manifest.get("n_cols") != len(close.columns)):
            raise DataError(f"snapshot shape does not match manifest: {snap_dir}")
        universe = frames.pop("universe", None)
        if universe is not None:
            universe = _parse_universe(universe)
        # Persisted fields must align exactly; silently reindexing a damaged
        # snapshot could hide missing prices or change universe membership.
        return MarketData(close, universe=universe, **frames)

    def list_snapshots(self, name: str) -> list[str]:
        """Snapshot ids for ``name``, sorted ascending (latest last)."""
        self._check_component(name)
        name_dir = self.root / name
        if not name_dir.is_dir():
            raise DataError(f"no dataset '{name}' in store {self.root}")
        return sorted(p.name for p in name_dir.iterdir() if p.is_dir() and not p.name.startswith(".pending-"))

    # -- helpers --------------------------------------------------------

    def _snapshot_dir(self, name: str, snapshot: str | None) -> Path:
        snapshots = self.list_snapshots(name)  # raises DataError if name missing
        if snapshot is None:
            if not snapshots:
                raise DataError(f"dataset '{name}' has no snapshots in {self.root}")
            snapshot = snapshots[-1]
        self._check_component(snapshot)
        snap_dir = self.root / name / snapshot
        if not snap_dir.is_dir():
            raise DataError(
                f"no snapshot '{snapshot}' for dataset '{name}'; available: {snapshots}"
            )
        return snap_dir

    @staticmethod
    def _check_component(value: str) -> None:
        if not isinstance(value, str) or value in ("", ".", "..") or value.startswith(".pending-") or Path(value).name != value:
            raise DataError("dataset and snapshot ids must be single non-empty path components")
