"""Point-in-time snapshot store: pin the exact MarketData a backtest used.

Layout: ``root/<name>/<snapshot>/`` with one CSV per non-None field
(``close.csv``, ``volume.csv``, ...) plus a ``manifest.json`` recording shape,
date range, tickers, save time, and the sha256 of the close CSV bytes so a
result can later be checked against the data it claims to have used.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_lab.core.errors import DataError
from alpha_lab.core.types import MarketData

#: save order = MarketData field order; close always first
_FIELDS = ("close", "open", "high", "low", "volume", "unadjusted_close", "universe")


class MarketDataStore:
    """Snapshot store rooted at a directory; snapshots are immutable-by-convention."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, data: MarketData, name: str, snapshot: str | None = None) -> Path:
        """Write one snapshot and return its directory.

        ``snapshot`` defaults to the current UTC time as ``%Y%m%d-%H%M%S`` so
        lexicographic order equals chronological order.
        """
        snap = snapshot or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = self.root / name / snap
        out.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        close_sha256 = None
        for field_name in _FIELDS:
            frame = getattr(data, field_name)
            if frame is None:
                continue
            csv_bytes = frame.to_csv(index_label="date").encode("utf-8")
            (out / f"{field_name}.csv").write_bytes(csv_bytes)
            if field_name == "close":
                close_sha256 = hashlib.sha256(csv_bytes).hexdigest()
            fields.append(field_name)
        manifest = {
            "fields": fields,
            "n_rows": int(len(data.dates)),
            "n_cols": int(len(data.tickers)),
            "start": data.dates[0].isoformat() if len(data.dates) else None,
            "end": data.dates[-1].isoformat() if len(data.dates) else None,
            "tickers": data.tickers,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "close_sha256": close_sha256,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return out

    def load(self, name: str, snapshot: str | None = None) -> MarketData:
        """Load one snapshot; ``snapshot=None`` resolves to the latest by name."""
        snap_dir = self._snapshot_dir(name, snapshot)
        manifest_path = snap_dir / "manifest.json"
        if manifest_path.exists():
            fields = json.loads(manifest_path.read_text())["fields"]
        else:
            fields = [f for f in _FIELDS if (snap_dir / f"{f}.csv").exists()]
        if "close" not in fields or not (snap_dir / "close.csv").exists():
            raise DataError(f"snapshot {snap_dir} has no close.csv")
        frames = {
            field_name: pd.read_csv(
                snap_dir / f"{field_name}.csv", index_col="date", parse_dates=["date"]
            )
            for field_name in fields
        }
        close = frames.pop("close")
        universe = frames.pop("universe", None)
        if universe is not None:
            universe = universe.astype(bool)
        return MarketData.from_frames(close, universe=universe, **frames)

    def list_snapshots(self, name: str) -> list[str]:
        """Snapshot ids for ``name``, sorted ascending (latest last)."""
        name_dir = self.root / name
        if not name_dir.is_dir():
            raise DataError(f"no dataset '{name}' in store {self.root}")
        return sorted(p.name for p in name_dir.iterdir() if p.is_dir())

    # -- helpers --------------------------------------------------------

    def _snapshot_dir(self, name: str, snapshot: str | None) -> Path:
        snapshots = self.list_snapshots(name)  # raises DataError if name missing
        if snapshot is None:
            if not snapshots:
                raise DataError(f"dataset '{name}' has no snapshots in {self.root}")
            snapshot = snapshots[-1]
        snap_dir = self.root / name / snapshot
        if not snap_dir.is_dir():
            raise DataError(
                f"no snapshot '{snapshot}' for dataset '{name}'; available: {snapshots}"
            )
        return snap_dir
