"""Report data-quality findings for the local price cache.

Writes results/data_quality.json. Exit 0 means INFO-only, 1 means WARN
findings requiring review, and 2 means FAIL findings or unreadable inputs.
Run after refreshing data and before interpreting strategy results.

Optional --fundamentals path.csv and --snapshot YYYY-MM-DD validate
fundamentals. --strict ignores the optional known-events allowlist.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.quality import PriceBundle, apply_known_events, run_all  # noqa: E402

BADGE = {"FAIL": "!", "WARN": "?", "INFO": " "}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fundamentals", type=Path, default=None,
                    help="long-format fundamentals CSV (see check_fundamentals)")
    ap.add_argument("--snapshot", default=None,
                    help="as-of date for future-period checks")
    ap.add_argument("--max-detail", type=int, default=25,
                    help="max WARN lines printed per check (FAILs always print)")
    ap.add_argument("--known", type=Path,
                    default=ROOT / "data" / "dq_known_events.csv",
                    help="adjudicated real-event allowlist (WARN -> INFO)")
    ap.add_argument("--strict", action="store_true",
                    help="ignore the known-events allowlist")
    args = ap.parse_args()

    try:
        bundle = PriceBundle.load()
        fund = (pd.read_csv(args.fundamentals)
                if args.fundamentals is not None else None)
    except Exception as e:  # noqa: BLE001
        # unreadable input is a FAIL, not a crash: python's default exit 1
        # would read as "WARN - eyeball" to the pipeline gating on us
        print(f"data quality: FAIL - cannot load inputs: {e!r}")
        out = ROOT / "results" / "data_quality.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"worst": "FAIL", "n_findings": 1,
                                   "findings": [{"check": "load",
                                                 "severity": "FAIL",
                                                 "ticker": "", "date": "",
                                                 "detail": repr(e)}]},
                                  indent=1))
        sys.exit(2)
    report = run_all(bundle, fundamentals=fund, snapshot_date=args.snapshot)
    if not args.strict and args.known.exists():
        known = pd.read_csv(args.known, dtype=str, keep_default_na=False)
        n_ack, dead = apply_known_events(report, known)
        if n_ack:
            print(f"[{n_ack} findings acknowledged via {args.known.name}]")
        for k in dead:
            print(f"[dead acknowledgment - matches no finding: {k}]")
        if n_ack or dead:
            print()

    idx = bundle.adj_close.index
    print(f"data quality: data/, {idx[0].date()} .. {idx[-1].date()}, "
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

    out = ROOT / "results" / "data_quality.json"
    payload = {"as_of": str(idx[-1].date()),
               "universe": int(bundle.adj_close.shape[1]),
               "rows": int(len(idx)),
               **report.as_dict()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\noverall: {report.worst()}   ({len(report.findings)} findings "
          f"-> {out.relative_to(ROOT)})")
    if report.worst() == "FAIL":
        sys.exit(2)
    if report.worst() == "WARN":
        sys.exit(1)


if __name__ == "__main__":
    main()
