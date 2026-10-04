"""Report data-quality findings for the local price cache.

Writes results/data_quality.json (--out names another file). Exit 0 means
INFO-only, 1 means WARN findings requiring review, and 2 means FAIL
findings, unreadable inputs or a crash of the run itself. Run after
refreshing data and before interpreting strategy results.

The cache is data/ unless QCORE_DATA_DIR names another directory; the report
of such a cache is written to data_quality.json beside it (unless --out is
given), so it never replaces the default cache's report. When the
downloader's coverage.csv lies beside the price files, every series is
compared with it: observations it records that the cache no longer has are
reported as history_loss.

Optional --fundamentals path.csv and --snapshot YYYY-MM-DD validate
fundamentals. --known path.csv applies a known-events allowlist (the default
data/dq_known_events.csv is used when present; a path given explicitly must
exist) and --strict ignores it. The report records the cache directory, the
allowlist applied and what it acknowledged.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import qcore.data as qdata  # noqa: E402
from qcore.quality import PriceBundle, apply_known_events, run_all  # noqa: E402

BADGE = {"FAIL": "!", "WARN": "?", "INFO": " "}
DEFAULT_OUT = ROOT / "results" / "data_quality.json"
DEFAULT_KNOWN = ROOT / "data" / "dq_known_events.csv"


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--fundamentals", type=Path, default=None,
                    help="long-format fundamentals CSV (see check_fundamentals)")
    ap.add_argument("--snapshot", default=None,
                    help="as-of date for future-period checks")
    ap.add_argument("--max-detail", type=int, default=25,
                    help="max WARN lines printed in total (FAILs always print)")
    ap.add_argument("--known", type=Path, default=None,
                    help="adjudicated real-event allowlist (WARN -> INFO; an "
                         "optional `expect` column pins the adjudicated "
                         "value). Default: data/dq_known_events.csv, skipped "
                         "when that file is absent; a path given here must "
                         "exist")
    ap.add_argument("--strict", action="store_true",
                    help="ignore the known-events allowlist")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the JSON report (default "
                         "results/data_quality.json; for a cache named by "
                         "QCORE_DATA_DIR, data_quality.json beside that "
                         "cache)")
    args = ap.parse_args()
    if args.out is None:
        args.out = _default_out()

    try:
        if (args.known is not None and not args.strict
                and not args.known.exists()):
            raise FileNotFoundError(
                f"--known allowlist not found: {args.known}")
        bundle = PriceBundle.load()
        fund = (pd.read_csv(args.fundamentals)
                if args.fundamentals is not None else None)
    except Exception as e:  # noqa: BLE001
        _fail(args.out, "load", "cannot load inputs", e)
    try:
        _check_and_report(args, bundle, fund)
    except Exception as e:  # noqa: BLE001
        _fail(args.out, "run", "check run crashed", e)


def _default_out() -> Path:
    """Where the report goes when --out is not given: results/ for the
    default cache; beside the cache for one named by QCORE_DATA_DIR, so a
    run on another cache (a live refresh) never replaces the report of the
    default one."""
    if qdata.DATA_DIR == qdata.DEFAULT_DATA_DIR:
        return DEFAULT_OUT
    return qdata.DATA_DIR / DEFAULT_OUT.name


def _shown(path: Path) -> str:
    """`path` relative to the project root when it lies inside it, so the
    report does not carry a machine-specific prefix."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _describe(e: Exception) -> str:
    """'Type: message', with the project root stripped from paths. repr() of
    an OSError drops the file name - the one thing the operator needs."""
    msg = str(e).replace(str(ROOT) + os.sep, "")
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def _fail(out: Path, check: str, what: str, e: Exception) -> None:
    """A crash is a FAIL, not python's default exit 1, which would read as
    "WARN - eyeball" to the pipeline gating on us."""
    detail = _describe(e)
    print(f"data quality: FAIL - {what}: {detail}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"worst": "FAIL", "n_findings": 1,
                               "findings": [{"check": check,
                                             "severity": "FAIL",
                                             "ticker": "", "date": "",
                                             "detail": detail}]},
                              indent=1))
    sys.exit(2)


def _recorded_coverage():
    """The downloader's coverage.csv beside the price files, as the table
    check_history_loss compares the cache with; None when there is none. A
    file that cannot be parsed yields an empty table, which that check
    reports ("could not compare" is not "nothing lost")."""
    path = qdata.DATA_DIR / "coverage.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0, dtype=str)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _check_and_report(args, bundle, fund) -> None:
    data_dir = _shown(qdata.DATA_DIR)
    report = run_all(bundle, fundamentals=fund, snapshot_date=args.snapshot,
                     reference_coverage=_recorded_coverage())
    known_path = args.known if args.known is not None else DEFAULT_KNOWN
    n_ack, dead, applied, unpinned = 0, [], None, 0
    if not args.strict and known_path.exists():
        known = pd.read_csv(known_path, dtype=str, keep_default_na=False)
        n_ack, dead = apply_known_events(report, known)
        applied = _shown(known_path)
        unpinned = (len(known) if "expect" not in known.columns
                    else int((known["expect"].str.strip() == "").sum()))
        if n_ack:
            print(f"[{n_ack} findings acknowledged via {known_path.name}]")
        for k in dead:
            print(f"[dead acknowledgment - matches no finding: {k}]")
        if unpinned:
            print(f"[{unpinned} of {len(known)} acknowledgment rows carry no "
                  "expect value: they match on check, ticker and date alone, "
                  "whatever the finding now measures]")
        if n_ack or dead or unpinned:
            print()

    idx = bundle.adj_close.index
    span = f"{idx[0].date()} .. {idx[-1].date()}" if len(idx) else "no rows"
    print(f"data quality: {data_dir}/, {span}, "
          f"{bundle.adj_close.shape[1]} tickers x {len(idx)} rows\n")

    counts = report.counts()
    print(f" {'check':24s} {'FAIL':>5s} {'WARN':>5s} {'INFO':>5s}")
    for check, row in counts.iterrows():
        flag = "!" if row["FAIL"] else ("?" if row["WARN"] else " ")
        print(f"{flag}{check:25s} {row['FAIL']:5d} {row['WARN']:5d} "
              f"{row['INFO']:5d}")
    if counts.empty:
        print("  (no findings)")

    for sev in ("FAIL", "WARN"):
        found = report.by_severity(sev)
        if not found:
            continue
        shown = found if sev == "FAIL" else found[:args.max_detail]
        print(f"\n{sev} findings"
              + (f" (first {len(shown)} of {len(found)})"
                 if len(shown) < len(found) else "") + ":")
        for f in shown:
            loc = " ".join(x for x in (f.ticker, f.date) if x)
            print(f" {BADGE[sev]} {f.check:20s} {loc:28s} {f.detail}")
    if any(f.check == "history_loss" and f.severity != "INFO"
           for f in report.findings):
        print(f"\nhistory_loss: compared with {data_dir}/coverage.csv, the "
              "record the last download wrote beside the price files. The "
              "files no longer match it: restore them or re-run the "
              "download.")

    payload = {"as_of": str(idx[-1].date()) if len(idx) else None,
               "universe": int(bundle.adj_close.shape[1]),
               "rows": int(len(idx)),
               "data_dir": data_dir,
               "strict": bool(args.strict),
               "known_events_file": applied,
               "n_acknowledged": int(n_ack),
               "dead_acknowledgments": [list(k) for k in dead],
               "acknowledgments_without_expect": int(unpinned),
               **report.as_dict()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\noverall: {report.worst()}   ({len(report.findings)} findings "
          f"-> {_shown(args.out)})")
    if report.worst() == "FAIL":
        sys.exit(2)
    if report.worst() == "WARN":
        sys.exit(1)


if __name__ == "__main__":
    main()
