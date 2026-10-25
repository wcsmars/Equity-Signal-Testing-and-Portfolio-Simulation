"""NAV log -> the daily returns file scripts/monitor.py judges.

The monitor reads the account's own daily returns, net of every cost, with
deposits and withdrawals taken out. This script derives that file from a
NAV log kept by hand or copied from the broker's statements: one row per
NYSE session with the account's net liquidation value at the close and any
external cash moved in or out that day. It reads and writes local files
only; it talks to no broker.

Input: a CSV with exactly the columns date, nav, flow (any order):
  date  the NYSE session, written YYYY-MM-DD
  nav   net liquidation value at that session's close, USD; finite, > 0
  flow  external cash moved into (+, deposit) or out of (-, withdrawal) the
        account on that session, USD; blank = 0. Trading gains, dividends,
        interest, commissions and fees are NOT flows: they are the return.
examples/nav.example.csv is a made-up log in this format.

Convention: a flow is available from the START of its session, so it earns
that session's return:
  r_t = nav_t / (nav_{t-1} + flow_t) - 1
The first row is the starting point and gives no return (a flow on it is
already inside its nav). Example: nav 101,000, then a 5,000 deposit and a
close of 107,060 is 107,060 / 106,000 - 1 = +1.00% for that session.

Output: the monitor's returns file - header Date,ENSEMBLE, one row per
session after the first, Date written YYYY-MM-DD, ENSEMBLE the return as a
fraction with 12 decimals (0.001200000000 = +0.12%). Default
state/live_returns.csv; --out names another file.

Fail closed. Nothing is written, and the run exits 3, when
  - the header is not exactly date,nav,flow, a row does not have three
    cells, or the log has no row;
  - a date is not YYYY-MM-DD, is not an NYSE session (qcore.quality's
    rule-based calendar, which cannot know an unscheduled closure), or the
    dates are not strictly increasing;
  - an NYSE session between the first and last row has no row (the first
    missing session is named): a missing NAV would fold two sessions'
    returns into one;
  - a nav is blank, not a number, not finite or not positive; a flow is not
    a number or not finite; or nav_{t-1} + flow_t is not positive (a
    withdrawal larger than the account);
  - the output exists and disagrees with the log (below), its folder does
    not exist, or it is the log itself.

Append-only. When the output exists, its rows must be the first rows the log
implies, in order, each within 1e-10 of the implied return (absolute; the
file's 12 written decimals round by at most 5e-13); otherwise the run is
refused and the first differing date is named. Rows already in the file are
never rewritten: the new rows are appended to its unchanged bytes, written
to a temporary file beside it and renamed over it, so a failed run leaves
the old file as it was. A correction to an old NAV or flow is therefore a
refusal, not a silent rewrite of returns the monitor has already judged;
--rebuild replaces the file from the log deliberately.

Usage:
  python scripts/nav_to_returns.py examples/nav.example.csv --out <new file>
  python scripts/monitor.py <that file> --as-of 2025-04-21
  python scripts/nav_to_returns.py state/nav.csv          # -> state/live_returns.csv
  python scripts/nav_to_returns.py state/nav.csv --rebuild
  python scripts/monitor.py state/live_returns.csv

Exit codes (never 1 or 2, which mean REVIEW and KILL to scripts/monitor.py):
  0  the returns file agrees with the log: rows appended, file written, or
     nothing new (a log of one row has no return yet: nothing is written,
     except that --rebuild leaves the header alone in an existing file)
  3  refused: invalid NAV log, an output that disagrees with the log, a
     file that cannot be read or written, or a usage error; the returns
     file is unchanged
  4  internal error, including a dependency or qcore that cannot be
     imported at start-up (one line); the returns file is unchanged or
     completely written, never partly

Limits: the returns are only as good as the NAV log. A flow left out, or
booked on the wrong session, shows up as a gain or a loss; a NAV taken at
another time of day than the close mixes two sessions. The session check
uses a rule-based NYSE calendar.
"""

import argparse
import csv
import math
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXIT_OK = 0
EXIT_REFUSED = 3   # the monitor's "invalid input": never read as REVIEW (1) or KILL (2)
EXIT_CRASH = 4     # the monitor's internal-error status

try:  # an uncaught import error would exit 1, the monitor's REVIEW status
    import pandas as pd

    from qcore.quality import nyse_bdays
except Exception as exc:  # noqa: BLE001 - a missing or broken install
    print(f"nav_to_returns: start-up failed, cannot import a dependency "
          f"({type(exc).__name__}: {exc}); install the packages in requirements.txt")
    sys.exit(EXIT_CRASH)

DEFAULT_OUT = ROOT / "state" / "live_returns.csv"
LOG_COLUMNS = ("date", "nav", "flow")
OUT_HEADER = ["Date", "ENSEMBLE"]
TOLERANCE = 1e-10  # |file - log| allowed for a row already written
DECIMALS = 12
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


class Refused(Exception):
    """An input or output this script will not turn into returns."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse's own status 2 would read as KILL
        self.print_usage(sys.stderr)
        raise Refused(message)


def _shown(path: Path) -> str:
    """`path` relative to the project root when it lies inside it."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _day(text: str, where: str) -> pd.Timestamp:
    if not _DAY.fullmatch(text):
        raise Refused(f"{where}: date {text!r} is not written YYYY-MM-DD")
    try:
        return pd.Timestamp(pd.to_datetime(text, format="%Y-%m-%d"))
    except ValueError:
        raise Refused(f"{where}: date {text!r} is not a calendar date") from None


def _number(text: str, where: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise Refused(f"{where}: {name} {text!r} is not a number") from None
    if not math.isfinite(value):
        raise Refused(f"{where}: {name} {text!r} is not finite")
    return value


def _rows(path: Path, what: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """(stripped header, [(line number, stripped cells)]) of a CSV; blank
    lines are skipped."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            table = list(csv.reader(stream))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Refused(f"{what} {_shown(path)} is not a readable CSV ({exc})") from None
    if not table:
        raise Refused(f"{what} {_shown(path)} is empty")
    header = [name.strip() for name in table[0]]
    rows = [(number, [cell.strip() for cell in row])
            for number, row in enumerate(table[1:], start=2) if any(cell.strip() for cell in row)]
    return header, rows


def read_nav_log(path: Path) -> pd.DataFrame:
    """The NAV log as a frame indexed by session date, columns nav and flow
    (blank flow = 0), after every check in the module docstring."""
    header, rows = _rows(path, "NAV log")
    if len(header) != len(set(header)) or set(header) != set(LOG_COLUMNS):
        raise Refused(f"NAV log header must be exactly {','.join(LOG_COLUMNS)} (any order), "
                      f"got {','.join(header)}")
    if not rows:
        raise Refused("NAV log has no rows")
    at = {name: header.index(name) for name in LOG_COLUMNS}
    dates, navs, flows = [], [], []
    for number, cells in rows:
        where = f"NAV log line {number}"
        if len(cells) != len(header):
            raise Refused(f"{where}: {len(cells)} cells, expected {len(header)}")
        dates.append(_day(cells[at["date"]], where))
        nav = cells[at["nav"]]
        if not nav:
            raise Refused(f"{where}: nav is blank")
        navs.append(_number(nav, where, "nav"))
        if navs[-1] <= 0:
            raise Refused(f"{where}: nav {nav} is not positive")
        flow = cells[at["flow"]]
        flows.append(_number(flow, where, "flow") if flow else 0.0)
    index = pd.DatetimeIndex(dates)
    for earlier, later in zip(index[:-1], index[1:]):
        if later <= earlier:
            raise Refused(f"NAV log dates must be strictly increasing: {later.date()} follows "
                          f"{earlier.date()}")
    sessions = nyse_bdays(index[0], index[-1])
    closed = index.difference(sessions)
    if len(closed):
        raise Refused(f"NAV log has {len(closed)} row(s) on days the NYSE was closed, first "
                      f"{closed[0].date()}")
    missing = sessions.difference(index)
    if len(missing):
        raise Refused(f"NAV log is missing {len(missing)} NYSE session(s) between its first and "
                      f"last row, first {missing[0].date()}: add the row (a missing NAV would "
                      "fold two sessions into one return)")
    return pd.DataFrame({"nav": navs, "flow": flows}, index=index)


def daily_returns(log: pd.DataFrame) -> pd.Series:
    """r_t = nav_t / (nav_{t-1} + flow_t) - 1 for every row after the first."""
    base = log["nav"].shift(1) + log["flow"]
    bad = base.iloc[1:] <= 0
    if bad.any():
        day = bad.index[bad.to_numpy()][0]
        raise Refused(f"on {day.date()} the previous nav plus the flow is "
                      f"{base[day]:,.2f}, not positive: a withdrawal cannot exceed the account")
    return (log["nav"] / base - 1).iloc[1:].rename("ENSEMBLE")


def read_written(path: Path) -> list[tuple[str, float]]:
    """(date text, return) rows of an existing returns file of this format."""
    if path.stat().st_size == 0:
        raise Refused(f"returns file {_shown(path)} is empty; --rebuild writes it from the NAV log")
    header, rows = _rows(path, "returns file")
    if header != OUT_HEADER:
        raise Refused(f"returns file {_shown(path)} has header {','.join(header)}, not "
                      f"{','.join(OUT_HEADER)}; --rebuild replaces it from the NAV log")
    written = []
    for number, cells in rows:
        where = f"returns file {_shown(path)} line {number}"
        try:
            if len(cells) != 2:
                raise Refused(f"{where}: {len(cells)} cells, expected 2")
            written.append((str(_day(cells[0], where).date()), _number(cells[1], where, "return")))
        except Refused as exc:
            raise Refused(f"{exc}; --rebuild replaces the file from the NAV log") from None
    return written


def first_difference(written: list[tuple[str, float]], implied: pd.Series) -> str | None:
    """Why the file's rows are not the start of the implied series; None
    when they are."""
    for k, (day, value) in enumerate(written):
        if k >= len(implied):
            return (f"it has a row for {day}, but the NAV log implies returns only up to "
                    f"{implied.index[-1].date() if len(implied) else 'no date (one NAV row)'}")
        expected_day = str(implied.index[k].date())
        if day != expected_day:
            return f"its row {k + 1} is dated {day}, where the NAV log implies {expected_day}"
        if not abs(value - float(implied.iloc[k])) <= TOLERANCE:
            return (f"on {day} it holds {value:.{DECIMALS}f}, the NAV log implies "
                    f"{float(implied.iloc[k]):.{DECIMALS}f} (tolerance {TOLERANCE:g})")
    return None


def _lines(returns: pd.Series, newline: str) -> str:
    return "".join(f"{day.date()},{value:.{DECIMALS}f}{newline}" for day, value in returns.items())


def write_atomically(path: Path, data: bytes) -> None:
    """Replace `path` with `data` in one rename: readers see the old file or
    the new one, never a partial write."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _run(argv) -> int:
    ap = _Parser(allow_abbrev=False, description=__doc__.split("\n")[0])
    ap.add_argument("nav_log", type=Path, help="NAV log CSV (date,nav,flow)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="returns file to append to (default state/live_returns.csv)")
    ap.add_argument("--rebuild", action="store_true",
                    help="replace the returns file from the NAV log instead of appending")
    args = ap.parse_args(argv)
    out = args.out
    if not args.nav_log.is_file():
        raise Refused(f"NAV log not found: {args.nav_log}")
    if out.exists() and out.resolve() == args.nav_log.resolve():
        raise Refused("--out names the NAV log itself")
    if not out.parent.is_dir():
        raise Refused(f"output folder does not exist: {out.parent} (create it first)")
    if out.exists() and not out.is_file():
        raise Refused(f"--out is not a file: {out}")

    log = read_nav_log(args.nav_log)
    implied = daily_returns(log)
    print(f"nav_to_returns: {_shown(args.nav_log)}, {len(log)} NAV row(s), "
          f"{log.index[0].date()} .. {log.index[-1].date()}")
    deposits, withdrawals = log["flow"].iloc[1:].clip(lower=0), log["flow"].iloc[1:].clip(upper=0)
    print(f"flows after the first row: {int((deposits > 0).sum())} deposit(s) "
          f"${deposits.sum():,.2f}, {int((withdrawals < 0).sum())} withdrawal(s) "
          f"${-withdrawals.sum():,.2f}")
    if log["flow"].iloc[0]:
        print(f"first-row flow {log['flow'].iloc[0]:,.2f} is part of the starting nav and enters "
              "no return")

    if out.exists() and not args.rebuild:
        written = read_written(out)
        if reason := first_difference(written, implied):
            raise Refused(f"{_shown(out)} disagrees with the NAV log: {reason}. Rows already "
                          "written are never changed: correct the log, or replace the file "
                          "deliberately with --rebuild")
        new = implied.iloc[len(written):]
        if new.empty:
            print(f"{_shown(out)} is up to date: {len(written)} row(s), nothing to append")
            return EXIT_OK
        old = out.read_bytes()
        newline = "\r\n" if b"\r\n" in old else "\n"
        if old and not old.endswith(b"\n"):
            old += newline.encode()
        write_atomically(out, old + _lines(new, newline).encode())
        print(f"appended {len(new)} row(s) to {_shown(out)}, {new.index[0].date()} .. "
              f"{new.index[-1].date()}; it now holds {len(written) + len(new)} row(s)")
        return EXIT_OK

    replaced = out.exists()
    if implied.empty and not replaced:
        print(f"no return yet: the first NAV row is the starting point; {_shown(out)} not written")
        return EXIT_OK
    write_atomically(out, (",".join(OUT_HEADER) + "\n" + _lines(implied, "\n")).encode())
    span = (f", {implied.index[0].date()} .. {implied.index[-1].date()}" if len(implied)
            else " (one NAV row: no return yet)")
    print(f"{'rebuilt' if replaced else 'wrote'} {_shown(out)}: {len(implied)} row(s){span}"
          + (" (--rebuild replaced the previous file)" if replaced else ""))
    return EXIT_OK


def main(argv=None) -> int:
    """Exit status of one run; 3 for a refusal, 4 for a crash."""
    try:
        return _run(argv)
    except Refused as exc:
        print(f"nav_to_returns REFUSED, nothing written: {exc}")
        return EXIT_REFUSED
    except OSError as exc:  # unreadable log, unwritable folder: the file is unchanged
        print(f"nav_to_returns REFUSED, nothing written: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 - a crash must not look like success
        traceback.print_exc()
        print(f"nav_to_returns CRASHED: {type(exc).__name__}: {exc}")
        return EXIT_CRASH


if __name__ == "__main__":
    sys.exit(main())
