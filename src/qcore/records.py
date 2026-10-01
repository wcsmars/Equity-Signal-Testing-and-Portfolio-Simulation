"""Guarded writes for saved result records.

Result files are records: a variants log is the denominator of a trial count
and a metrics file may be compared against pinned values. Re-running a script
after the data or the engine changed must not silently replace such a record.

Policy implemented by save_record():
  - the target does not exist            -> write it;
  - the target exists and is identical   -> leave it alone;
  - the target exists and differs        -> keep it, write this run's output to
    <target dir>/recomputed/<name> and say so.
Replacing a record is an explicit act: pass --rebase on the command line or
set QCORE_REBASE=1.
"""

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd

REBASE_FLAG = "--rebase"
REBASE_ENV = "QCORE_REBASE"
RECOMPUTED_DIR = "recomputed"


def rebase_requested(argv: list[str] | None = None) -> bool:
    """True when the caller asked to replace existing records."""
    args = sys.argv[1:] if argv is None else argv
    return REBASE_FLAG in args or os.environ.get(REBASE_ENV) == "1"


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_record(path, content: str | bytes, *, rebase: bool | None = None,
                quiet: bool = False) -> Path:
    """Write `content` to `path` without silently replacing a different
    existing record. Returns the path that holds this run's output."""
    path = Path(path)
    data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    if rebase is None:
        rebase = rebase_requested()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"refusing to write a record through a non-regular file: {path}")
    if not path.exists():
        _write_atomic(path, data)
        if not quiet:
            print(f"saved {path}")
        return path
    if path.read_bytes() == data:
        if not quiet:
            print(f"unchanged {path}")
        return path
    if rebase:
        _write_atomic(path, data)
        if not quiet:
            print(f"REBASED {path} (previous record replaced)")
        return path
    alt = path.parent / RECOMPUTED_DIR / path.name
    _write_atomic(alt, data)
    if not quiet:
        print(f"kept existing record {path}: this run differs, so its output went to "
              f"{alt}. Pass {REBASE_FLAG} (or set {REBASE_ENV}=1) to replace the record.")
    return alt


def save_json(path, obj, *, rebase: bool | None = None, quiet: bool = False,
              **dump_kwargs) -> Path:
    """json.dumps(obj, **dump_kwargs) through save_record (indent=2 by default)."""
    dump_kwargs.setdefault("indent", 2)
    return save_record(path, json.dumps(obj, **dump_kwargs), rebase=rebase, quiet=quiet)


def save_csv(path, frame: pd.DataFrame, *, rebase: bool | None = None,
             quiet: bool = False, **to_csv_kwargs) -> Path:
    """frame.to_csv(**to_csv_kwargs) through save_record."""
    buffer = io.StringIO()
    frame.to_csv(buffer, **to_csv_kwargs)
    return save_record(path, buffer.getvalue(), rebase=rebase, quiet=quiet)
