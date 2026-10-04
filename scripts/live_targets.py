"""Execution: the order sheet for the four-sleeve ensemble (market-on-close).

This script sends nothing and talks to no broker. It prints target shares
per sleeve and, with a ledger, one net market-on-close (MOC) order per
ticker for a person to enter by hand.

It mirrors the engine's execution convention: weights DECIDED at close t are
traded AT close t through an MOC order. The decision close here is the data
cache's last row, so the flow for a market-on-close submission is (New York
time; the NYSE MOC cutoff is 10 minutes before the close):

              export QCORE_DATA_DIR="$PWD/state/live_cache"
  ~3:40pm ET  python src/download_data.py       # in-progress day = near-close
  ~3:42pm ET  python scripts/live_targets.py --cash <broker cash balance> \
                  --ledger state/ledger.csv
  ~3:45pm ET  submit the printed MOC orders (NYSE cutoff 3:50, Nasdaq 3:55)

QCORE_DATA_DIR names the price cache the downloader fills and this script
reads. Exporting it for the live session keeps the daily refresh in a
separate live cache, so the research cache in data/ is never replaced; every
command of the session must see the same value. Without it both read data/.
Research scripts must run without it, on the research cache. state/ is
excluded by .gitignore: ledgers, fills and live caches are never committed.

The script prints the cutoff in ET and in this machine's local time. On a
scheduled 1pm early close (the Friday after Thanksgiving; July 3 and
December 24 when they fall Monday to Thursday) every step moves three hours
earlier.

Live mode (no --as-of) fails closed unless ALL of these hold:
  - the cache's last row is today's New York date, a scheduled NYSE session;
  - New York wall time is inside the submission window, from 30 minutes
    before the close to the MOC cutoff 10 minutes before it. Earlier, the
    snapshot is not a near-close price; later, an MOC order fills at the
    NEXT close, which is a day-late catch-up order;
  - the price cache itself was written inside that window (and is not
    stamped later than the clock this run reads).
Old decisions therefore cannot generate catch-up orders. Use
--as-of YYYY-MM-DD for an explicit offline inspection at any time of day;
the supplied date must equal the cache's last row.

Signal paths are the strategy modules' own builders (src/strategies/*),
including the live-edge calendar guards: on a mid-month decision close the
monthly sleeves (tsmom_trend, xsec_etf_mom) emit HOLD, never a phantom
rebalance. Seasonality's window test for the NEXT session uses the same
NYSE-calendar rule as its backtest. A sleeve that decides today needs a
valid decision-close quote for every listed ticker it reads: a missing quote
would re-rank or re-scale the other names. This script checks the decision
row before it calls a builder and refuses the run, naming the sleeve and the
tickers. The builders that read prices (tsmom_trend, xsec_etf_mom,
mean_reversion) also raise on a blank close after listing, so this is the
first of two checks; it alone rejects a zero or negative quote and covers
SPY for seasonality, whose builder reads only the calendar.

Account size: with --ledger there is no default. Pass --cash (the broker
cash balance; equity = cash + ledger market value at the decision close) or
--equity (today's net liquidation value; the implied cash is printed and
must match the broker). A negative cash balance needs --allow-margin.
Sleeve capital = equity x the in-sample ensemble weight saved in
results/ensemble.json. The idle remainder is parked in SGOV after reserving
modeled commissions, sell fees and the configurable adverse-price allowance.
Sleeve targets are never cut to force an unfunded order list through;
--min-order only trims the SGOV parking that its suppressed sells would have
funded.

Ledger (optional, --ledger): per-sleeve share positions CSV with columns
sleeve,ticker,shares and an optional as_of column (the session whose fills
it reflects). examples/ledger.example.csv is a made-up ledger in that
format. With a ledger, the output is per-sleeve share DELTAS netted into one
order per ticker; without it, target shares only. A dated ledger must be as
of the previous NYSE session (--ignore-ledger-date overrides), so a ledger
that already contains today's fills cannot print the same orders twice.
This script never changes the input ledger and no other script maintains
it: --write-ledger PATH writes the per-sleeve ledger that results IF every
printed order fills in full; correct it from the broker's confirmed fills
before the next run.

Inputs: the price cache (adj_close.csv and close.csv, built by
python src/download_data.py), results/ensemble.json (built by
python src/ensemble.py; without it the run is refused with that command)
and, optionally, a ledger. Offline example on any cache:
  python scripts/live_targets.py --as-of <last date in the cache> \
      --cash 5000 --ledger examples/ledger.example.csv --ignore-ledger-date

Exit codes: 0 = order sheet printed; 2 = refused (failed gate, invalid or
missing input, unfunded orders) - nothing printed above the message may be
traded; 3 = unexpected crash.

Limits: orders are whole shares, floored per sleeve, and the rounding gap is
printed; an unfunded book is refused as a whole rather than trimmed; early
closes follow a rule table and an unscheduled one cannot be detected (use
--as-of); share counts are sized on a near-close snapshot, not on the
official close at which the orders fill (scripts/reconcile.py compares the
fills with that close afterwards); nothing here reads the broker's actual
positions, so the ledger is only as good as its upkeep.
"""

import argparse
import csv
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "strategies"))

from qcore.calendar import confirmed_month_ends  # noqa: E402
from qcore.costs import IBKRHKCostModel  # noqa: E402
from qcore.data import DATA_DIR, load, load_prices  # noqa: E402
from qcore.quality import nyse_bdays  # noqa: E402

CASH_ETF = "SGOV"  # 0-3m T-bills; the engine's ^IRX-10bp credit made real
SLEEVES = ("mean_reversion", "seasonality_flows", "tsmom_trend", "xsec_etf_mom")
NY = "America/New_York"
SUBMIT_WINDOW_MINUTES = 30  # snapshot and order sheet no earlier than this before the close
MOC_CUTOFF_MINUTES = 10     # NYSE market-on-close cutoff before the close
SNAPSHOT_CLOCK_SKEW_SECONDS = 60  # a cache stamped later than "now" by more than this is refused
DEFAULT_EQUITY = 100_000.0  # ledger-less target display only; never an order list
EXIT_REFUSED = 2
EXIT_CRASH = 3


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def current_ny_time() -> pd.Timestamp:
    return _utc_now().tz_convert(NY)


def current_ny_session_date() -> pd.Timestamp:
    return current_ny_time().tz_localize(None).normalize()


def nyse_early_close(day) -> bool:
    """Scheduled 1pm ET closes under the regular NYSE rules.

    The Friday after Thanksgiving, and July 3 / December 24 when they fall
    Monday to Thursday. Approximate like the holiday rules: unscheduled
    early closes cannot be anticipated, so verify the published calendar.
    """
    day = pd.Timestamp(day).normalize()
    if (day.month, day.day) in {(7, 3), (12, 24)}:
        return day.dayofweek <= 3
    return day.month == 11 and day.dayofweek == 4 and 23 <= day.day <= 29


def submission_window(day) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(window opens, MOC cutoff, close) in New York time for a session date."""
    hour = 13 if nyse_early_close(day) else 16
    close = (pd.Timestamp(day).normalize() + pd.Timedelta(hour, unit="h")).tz_localize(NY)
    return (close - pd.Timedelta(SUBMIT_WINDOW_MINUTES, unit="min"),
            close - pd.Timedelta(MOC_CUTOFF_MINUTES, unit="min"), close)


def cache_snapshot_time() -> pd.Timestamp:
    """When the price cache was written (the older close file), New York time."""
    written = min((DATA_DIR / f"{name}.csv").stat().st_mtime for name in ("adj_close", "close"))
    return pd.Timestamp(written, unit="s", tz="UTC").tz_convert(NY)


def submission_window_error(today: pd.Timestamp, now: pd.Timestamp,
                            snapshot: pd.Timestamp) -> str | None:
    """Why a live order sheet must not be printed now; None inside the window."""
    opens, cutoff, close = submission_window(today)
    if now.tz_localize(None).normalize() != today:
        return f"the New York date changed to {now.date()} while checking session {today.date()}"
    if now < opens:
        return (f"New York time is {now:%H:%M}, before the submission window opens at "
                f"{opens:%H:%M} ET (close {close:%H:%M} ET): prices this far from the close "
                "are not the decision close")
    if now >= cutoff:
        return (f"New York time is {now:%H:%M}, past the MOC cutoff {cutoff:%H:%M} ET "
                f"(close {close:%H:%M} ET): an MOC order entered now fills at the NEXT close")
    if snapshot < opens:
        return (f"the price cache was written {snapshot:%Y-%m-%d %H:%M} ET, before the "
                f"submission window opened at {opens:%H:%M} ET: refresh the data now")
    if snapshot > now + pd.Timedelta(SNAPSHOT_CLOCK_SKEW_SECONDS, unit="s"):
        return (f"the price cache is stamped {snapshot:%Y-%m-%d %H:%M} ET, after the current "
                f"New York time {now:%H:%M}: its age cannot be checked. Check the system "
                "clock and refresh the data")
    return None


def _local_label(moment: pd.Timestamp) -> str:
    return moment.to_pydatetime().astimezone().strftime("%a %H:%M %Z").strip()


def ensemble_weights() -> dict[str, float]:
    path = ROOT / "results" / "ensemble.json"
    if not path.is_file():
        raise FileNotFoundError("results/ensemble.json not found: it holds the sleeve weights "
                                "that size every order. Create it first: python src/ensemble.py")
    w = json.loads(path.read_text())["sleeve_weights"]
    expected = {"mean_reversion", "seasonality_flows", "tsmom_trend", "xsec_etf_mom"}
    values = np.asarray(list(w.values()), dtype=float)
    if (set(w) != expected or not np.isfinite(values).all()
            or (values <= 0).any() or abs(values.sum() - 1) > 0.00025):
        raise ValueError("ensemble must have all four positive sleeve weights summing to one")
    # Historical JSONs rounded each share to 4dp; normalize rounding only.
    return {k: float(v) / values.sum() for k, v in w.items()}


def require_decision_row(px: pd.DataFrame, tickers, sleeve: str) -> None:
    """Every listed ticker a deciding sleeve reads needs a decision-close quote.

    A missing last-row price would take the name out of the ranking or the
    inverse-vol denominator and change the remaining weights. The builders
    that read prices raise on a blank close after listing themselves
    (qcore.data.require_listed_closes); this check runs first, names the
    sleeve, and also rejects a zero, negative or non-finite quote. A ticker
    with no earlier observation at all is not yet listed and is left to the
    builder.
    """
    frame = px[list(tickers)]
    listed = frame.iloc[:-1].notna().any().to_numpy()
    last = frame.iloc[-1].to_numpy(dtype=float)
    bad = frame.columns[listed & (~np.isfinite(last) | (last <= 0))]
    if len(bad):
        raise ValueError(f"{sleeve}: no valid decision-close quote on {px.index[-1].date()} for: "
                         f"{', '.join(bad)}; refresh the data cache (a missing quote would "
                         "silently change this sleeve's targets)")


def sleeve_universes() -> dict[str, set[str]]:
    """Tickers each production sleeve can hold (ledger sanity check)."""
    import mean_reversion as mr
    import tsmom_trend as tt
    import xsec_etf_mom as xe
    trend = set(tt.RISK) | ({tt.CASH} if tt.BEST["sleeve"] == "shy" else set())
    return {"mean_reversion": set(mr.UNIVERSE), "seasonality_flows": {"SPY"},
            "tsmom_trend": trend, "xsec_etf_mom": set(xe.EQ_UNIVERSE) | {xe.DEFENSIVE}}


def sleeve_mean_reversion(px: pd.DataFrame) -> tuple[pd.Series, str]:
    import mean_reversion as mr
    require_decision_row(px, mr.UNIVERSE, "mean_reversion")
    w = mr.build_weights(px[mr.UNIVERSE], **mr.BEST_PARAMS)
    return w.iloc[-1], "daily decision (RSI-2 dip state machine)"


def sleeve_seasonality(px: pd.DataFrame) -> tuple[pd.Series, str]:
    """Use the same calendar-based decision as the production backtest."""
    import seasonality_flows as sf
    require_decision_row(px, ["SPY"], "seasonality_flows")
    return sf.tom_weights(px["SPY"]).iloc[-1], "daily decision (NYSE turn-of-month window)"


def read_ledger(path: Path) -> pd.Series:
    """An explicit ledger is a complete position snapshot; omitted rows are zero.

    The optional as_of column (one date on every row) is returned in
    ``.attrs["as_of"]``; None when the ledger is undated or has no rows.
    """
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = [name.strip() for name in next(csv.reader(stream), [])]
    if not header or len(header) != len(set(header)):
        raise ValueError("ledger CSV needs a nonempty unique header")
    df = pd.read_csv(path)
    df.columns = [str(name).strip() for name in df.columns]
    if not {"sleeve", "ticker", "shares"}.issubset(df.columns):
        raise ValueError("ledger needs sleeve,ticker,shares columns")
    if df[["sleeve", "ticker"]].isna().any().any():
        raise ValueError("ledger sleeve/ticker cannot be blank")
    df["sleeve"] = df["sleeve"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    allowed = {"mean_reversion", "seasonality_flows", "tsmom_trend", "xsec_etf_mom", "cash"}
    if not set(df["sleeve"]).issubset(allowed) or (df["ticker"] == "").any():
        raise ValueError("ledger has unknown sleeve or blank ticker (use sleeve 'cash' for SGOV)")
    shares = pd.to_numeric(df["shares"], errors="raise")
    if not np.isfinite(shares).all() or (shares < 0).any():
        raise ValueError("ledger shares must be finite and nonnegative for this long-only portfolio")
    if not np.isclose(shares, np.round(shares), atol=1e-9, rtol=0).all():
        raise ValueError("ledger must contain whole shares; fractional-share MOC orders are unsupported")
    df["shares"] = shares.round()
    if df.duplicated(["sleeve", "ticker"]).any():
        raise ValueError("ledger has duplicate sleeve/ticker positions")
    as_of = None
    if "as_of" in df.columns and len(df):
        try:
            dates = pd.to_datetime(df["as_of"].astype(str).str.strip(), format="%Y-%m-%d")
        except (ValueError, TypeError) as exc:
            raise ValueError("ledger as_of must be one YYYY-MM-DD session date on every row") from exc
        if dates.isna().any() or dates.nunique() != 1:
            raise ValueError("ledger as_of must be one YYYY-MM-DD session date on every row")
        as_of = dates.iloc[0]
    out = df.set_index(["sleeve", "ticker"])["shares"]
    out.attrs["as_of"] = as_of
    return out


def ledger_date_error(as_of: pd.Timestamp, t: pd.Timestamp) -> str | None:
    """A dated ledger must reflect exactly the previous session's fills."""
    if as_of >= t:
        return (f"ledger is dated {as_of.date()}, not before the decision close {t.date()}: it "
                "already contains this session's fills, so these orders would be sent twice")
    previous = nyse_bdays(t - pd.Timedelta(10, unit="D"), t - pd.Timedelta(1, unit="D"))[-1]
    if as_of != previous:
        return (f"ledger is dated {as_of.date()} but the previous NYSE session was "
                f"{previous.date()}: any fills since then are missing from it (if no order "
                "was sent since then, the ledger is still current)")
    return None


def require_prices(prices: pd.Series, tickers) -> None:
    quote = prices.reindex(list(tickers))
    bad = quote.index[~np.isfinite(quote.to_numpy(dtype=float)) | (quote <= 0)]
    if len(bad):
        raise ValueError(f"missing or invalid executable close for: {', '.join(bad)}")


def execution_reserve(deltas: dict[str, float], prices: pd.Series,
                      price_buffer_bps: float) -> float:
    """Reserve fixed commissions, actual sell-side fees and adverse fills.

    The price buffer applies to gross buy and sell notionals: higher buys
    and lower sale proceeds both consume cash. This is a sizing allowance,
    not a guarantee that a market-on-close fill stays inside that buffer.
    """
    cm = IBKRHKCostModel(slippage_bps=0.0)
    reserve = 0.0
    for ticker, shares in deltas.items():
        if not shares:
            continue
        notional = abs(shares) * float(prices[ticker])
        commission = min(max(cm.min_commission, cm.commission_per_share * abs(shares)),
                         cm.max_commission_pct * notional)
        fees = (cm.sec_fee_rate * notional + min(cm.finra_taf_per_share * abs(shares), 8.30)
                if shares < 0 else 0.0)
        reserve += commission + fees + notional * price_buffer_bps / 1e4
    return reserve


def funded_targets(targets: dict[str, float], held: pd.Series, prices: pd.Series,
                   equity: float, cash_parking_shares: float,
                   min_order: float, price_buffer_bps: float,
                   unknown_hold_value: float = 0.0) -> tuple[dict[str, float], float]:
    """Keep research targets intact; reduce only newly parked SGOV for costs.

    Validate funding after the minimum-order filter on the actual net
    broker orders. Unknown HOLD capital is reserved in target-only mode.
    A sell suppressed by the minimum-order filter leaves its position in
    place, so the cash it would have raised is taken out of SGOV parking;
    the run is refused only when parking cannot cover the gap.
    """
    targets = dict(targets)
    tickers = set(targets) | set(held.index)

    def plan():
        deltas = {t: targets.get(t, 0.0) - float(held.get(t, 0.0)) for t in tickers}
        executable = {t: d for t, d in deltas.items()
                      if d and abs(d) * float(prices[t]) >= min_order}
        value = unknown_hold_value + sum(float(sh) * float(prices[t]) for t, sh in held.items() if sh)
        value += sum(delta * float(prices[t]) for t, delta in executable.items())
        return value, execution_reserve(executable, prices, price_buffer_bps)

    value, reserve = plan()
    while value + reserve > equity + 1e-8 and cash_parking_shares > 0:
        deficit = value + reserve - equity
        reduction = min(cash_parking_shares, max(1.0, float(np.ceil(deficit / prices[CASH_ETF]))))
        cash_parking_shares -= reduction
        targets[CASH_ETF] -= reduction
        value, reserve = plan()
    if value + reserve > equity + 1e-8:
        hint = (" (--min-order may be suppressing the sells that fund the buys)"
                if min_order > 0 else "")
        raise ValueError(f"orders are unfunded after commissions, sell fees and price buffer "
                         f"(need ${value + reserve:,.2f}, equity ${equity:,.2f}){hint}; "
                         "keep more cash or revise sleeve allocations")
    return targets, reserve


def sleeve_tsmom(px_all: pd.DataFrame) -> tuple[pd.Series | None, str]:
    import tsmom_trend as tt
    px = px_all[tt.RISK + [tt.CASH]]
    if px.index[-1] not in confirmed_month_ends(px.index):
        return None, "monthly sleeve - decision close is not a month-end: HOLD"
    require_decision_row(px, px.columns, "tsmom_trend")
    sigs, elig, vol_me, shy_ok = tt.build_signals(px)
    w = tt.month_end_weights(sigs[tt.BEST["signal"]], elig, vol_me, shy_ok,
                             tt.BEST["weighting"], tt.BEST["sleeve"])
    return w.iloc[-1], (f"MONTH-END rebalance (trend blend, iv weights, "
                        f"{tt.BEST['sleeve']} off-sleeve)")


def sleeve_xsec_etf(px_all: pd.DataFrame) -> tuple[pd.Series | None, str]:
    import xsec_etf_mom as xe
    cols = xe.EQ_UNIVERSE + [xe.DEFENSIVE]
    if px_all[cols].index[-1] not in confirmed_month_ends(px_all[cols].index):
        return None, "monthly sleeve - decision close is not a month-end: HOLD"
    require_decision_row(px_all, cols, "xsec_etf_mom")
    p = {k: v for k, v in xe.BEST_PARAMS.items() if k != "drift"}
    w = xe.build_targets(px_all, **p)
    return w.iloc[-1], "MONTH-END rebalance (top-K momentum, B=9 hysteresis)"


def operator_alerts(recorded) -> list[str]:
    """The project's own alerts, each once, in the order raised.

    Only plain UserWarning is an operator alert (the calendar's "verify
    this month-end" notice). Deprecation and runtime chatter from numerical
    libraries is not an instruction to the person submitting orders.
    """
    return list(dict.fromkeys(str(w.message) for w in recorded if w.category is UserWarning))


def post_trade_ledger(sleeve_shares: dict[str, dict[str, float]], ledger: pd.Series,
                      unsent: set[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-sleeve positions IF every printed order fills in full.

    A ticker whose net order was not sent (below --min-order) trades
    nothing, so every sleeve keeps its current ledger shares of it.
    """
    rows = []
    for sleeve in (*SLEEVES, "cash"):
        book = dict(sleeve_shares.get(sleeve, {}))
        prior = ledger.loc[sleeve] if sleeve in ledger.index.get_level_values(0) else {}
        for ticker in unsent:
            book[ticker] = float(prior.get(ticker, 0.0))
        rows += [{"sleeve": sleeve, "ticker": ticker, "shares": int(round(book[ticker])),
                  "as_of": str(as_of.date())} for ticker in sorted(book) if book[ticker]]
    return pd.DataFrame(rows, columns=["sleeve", "ticker", "shares", "as_of"])


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0])
    money = ap.add_mutually_exclusive_group()
    money.add_argument("--equity", type=float, default=None,
                       help="account net liquidation value in USD right now; with --ledger "
                            "either this or --cash is required (ledger-less runs default to "
                            "a hypothetical 100000)")
    money.add_argument("--cash", type=float, default=None,
                       help="broker USD cash balance; equity = cash + ledger market value "
                            "at the decision close (needs --ledger)")
    ap.add_argument("--allow-margin", action="store_true",
                    help="accept a negative cash balance (ledger worth more than equity)")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="per-sleeve positions CSV (sleeve,ticker,shares[,as_of])")
    ap.add_argument("--write-ledger", type=Path, default=None,
                    help="write the per-sleeve ledger that results if every printed order "
                         "fills in full to this NEW file (never overwrites)")
    ap.add_argument("--ignore-ledger-date", action="store_true",
                    help="accept a ledger whose as_of is not the previous NYSE session")
    ap.add_argument("--min-order", type=float, default=0.0,
                    help="suppress orders below this notional (USD)")
    ap.add_argument("--price-buffer-bps", type=float, default=25.0,
                    help="adverse fill cash allowance per gross order notional (default 25 bps)")
    ap.add_argument("--as-of", default=None,
                    help="offline inspection at any time of day; must equal the cache's last date")
    args = ap.parse_args()
    if args.equity is not None and (not np.isfinite(args.equity) or args.equity <= 0):
        ap.error("--equity must be finite and positive")
    if args.cash is not None:
        if not np.isfinite(args.cash):
            ap.error("--cash must be finite")
        if args.ledger is None:
            ap.error("--cash needs --ledger: equity is cash plus the ledger's market value")
    if args.ledger is not None and args.equity is None and args.cash is None:
        ap.error("--ledger needs --cash (broker cash balance) or --equity (today's net "
                 "liquidation value): an order list has no default account size")
    if not np.isfinite(args.min_order) or args.min_order < 0:
        ap.error("--min-order must be finite and nonnegative")

    if not np.isfinite(args.price_buffer_bps) or not 0 <= args.price_buffer_bps < 10_000:
        ap.error("--price-buffer-bps must be finite and between 0 and 10000 (exclusive)")
    as_of = None
    if args.as_of is not None:
        try:
            as_of = pd.Timestamp(args.as_of)
        except (ValueError, TypeError):
            as_of = pd.NaT
        if as_of is pd.NaT:
            ap.error(f"--as-of must be a date YYYY-MM-DD, got {args.as_of!r}")
    if args.ledger is not None and not args.ledger.exists():
        ap.error(f"ledger not found: {args.ledger} (omit --ledger for target shares only)")
    if args.write_ledger is not None:
        if args.ledger is None:
            ap.error("--write-ledger needs --ledger: without positions there are no orders to apply")
        if args.write_ledger.exists():
            ap.error(f"--write-ledger refuses to overwrite {args.write_ledger}; name a new file")
        if not args.write_ledger.parent.is_dir():
            ap.error(f"--write-ledger folder does not exist: {args.write_ledger.parent}")

    px = load_prices()
    t = px.index[-1]
    # Signals use adjusted prices; actual share counts and ledger valuation
    # require dividend-unadjusted closes in the same units as broker fills.
    close = load("close")
    if t not in close.index or close.index[-1] != t:
        raise ValueError("adjusted and executable close caches have different latest dates")
    last_px = close.loc[t]
    today = current_ny_session_date()
    if as_of is not None:
        if as_of != t:
            ap.error("--as-of must equal the cache's last date; use a truncated cache for earlier dates")
        print("OFFLINE historical simulation: targets and orders below are hypothetical")
        print("share/price units follow the split-adjusted cache; historical broker ledgers "
              "must be converted to those same split units")
    elif t != today or today not in nyse_bdays(today, today):
        ap.error(f"live targets require today's New York scheduled session ({today.date()}); "
                 f"the cache in {DATA_DIR} ends {t.date()}. Refresh data during the current "
                 "session; QCORE_DATA_DIR selects the cache the downloader fills and this "
                 "script reads, so set it the same for both to keep a live cache apart from "
                 "the frozen research cache (use --as-of for an explicit offline simulation)")
    else:
        now = current_ny_time()
        snapshot = cache_snapshot_time()
        reason = submission_window_error(today, now, snapshot)
        if reason:
            ap.error(f"no live order sheet: {reason} (use --as-of {t.date()} for an explicit "
                     "offline inspection)")
        _, cutoff, session_close = submission_window(today)
        print(f"LIVE session {today.date()}: New York time {now:%H:%M}, MOC cutoff "
              f"{cutoff:%H:%M} ET ({_local_label(cutoff)} local), close {session_close:%H:%M} ET; "
              f"price cache written {snapshot:%H:%M} ET")

    ledger = None
    equity = args.equity
    if args.ledger is not None:
        ledger = read_ledger(args.ledger)
        require_prices(last_px, ledger[ledger != 0].index.get_level_values("ticker").unique())
        dated = ledger.attrs.get("as_of")
        if dated is None:
            if (ledger != 0).any():
                print("ledger has no as_of date: cannot check that it reflects the previous "
                      "session's fills")
        elif not args.ignore_ledger_date:
            if reason := ledger_date_error(dated, t):
                raise ValueError(f"{reason} (--ignore-ledger-date overrides)")
        held_value = float(sum(sh * float(last_px[tkr]) for (_, tkr), sh in ledger.items() if sh))
        if args.cash is not None:
            equity = args.cash + held_value
            if equity <= 0:
                raise ValueError(f"--cash {args.cash:,.2f} plus ledger market value "
                                 f"${held_value:,.2f} is not a positive equity")
        cash = equity - held_value
        if cash < -0.005 and not args.allow_margin:
            raise ValueError(f"ledger market value ${held_value:,.2f} exceeds equity ${equity:,.2f}: "
                             f"cash would be ${cash:,.2f}. Check the equity figure and the ledger "
                             "(pass --cash to derive equity, or --allow-margin for a margin debit)")
        print("ledger treated as complete: omitted sleeve/ticker rows mean zero shares")
        source = "as given" if args.cash is not None else "equity - ledger; must equal the broker cash balance"
        print(f"ledger market value ${held_value:,.2f} at the decision close; cash ${cash:,.2f} ({source})")
    elif equity is None:
        equity = DEFAULT_EQUITY
        print(f"--equity not given: sizing a hypothetical ${equity:,.0f} account (targets only)")
    print(f"decision close: {t.date()}  (equity ${equity:,.0f})")

    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        sleeves = {
            "mean_reversion": sleeve_mean_reversion(px),
            "seasonality_flows": sleeve_seasonality(px),
            "tsmom_trend": sleeve_tsmom(px),
            "xsec_etf_mom": sleeve_xsec_etf(px),
        }
    for message in operator_alerts(wl):
        print(f"  !! {message}")

    ew = ensemble_weights()
    if ledger is not None:
        universes = {**sleeve_universes(), "cash": {CASH_ETF}}
        for (key, tkr), sh in ledger.items():
            if sh and key in universes and tkr not in universes[key]:
                holding = key in sleeves and sleeves[key][0] is None
                action = ("is carried untouched while the sleeve HOLDs" if holding
                          else "gets a zero target (sold)")
                print(f"  !! ledger holds {sh:.0f} {tkr} under {key}, which never trades it: "
                      f"the position {action}; check the ledger's sleeve column")

    for key, (weights, _) in sleeves.items():
        if weights is not None:
            if (not np.isfinite(weights.to_numpy()).all() or (weights < 0).any()
                    or weights.sum() > 1 + 1e-8):
                raise ValueError(f"{key}: invalid long-only target weights")
            require_prices(last_px, weights.index[weights > 1e-9])

    port_target_shares: dict[str, float] = {}
    sleeve_shares: dict[str, dict[str, float]] = {}
    deployed = 0.0
    unknown_hold_value = 0.0
    target_dollars = 0.0
    sized_dollars = 0.0
    for key, (w, note) in sleeves.items():
        cap = equity * ew[key]
        book = sleeve_shares.setdefault(key, {})
        print(f"\n-- {key}  (weight {ew[key]:.1%}, capital ${cap:,.0f})  {note}")
        if w is None:  # monthly sleeve between rebalances: existing shares stand
            if ledger is not None:
                held = ledger.loc[key] if key in ledger.index.get_level_values(0) else pd.Series(dtype=float)
                for tkr, sh in held[held != 0].items():
                    port_target_shares[tkr] = port_target_shares.get(tkr, 0.0) + sh
                    book[tkr] = sh
                    deployed += sh * last_px[tkr]
                    print(f"     {tkr:5s} holds {sh:>6.0f} sh @ {last_px[tkr]:,.2f}"
                          f"  (${sh * last_px[tkr]:>9,.0f})")
                if not book:
                    print("     no ledger position")
            else:
                # holdings unknown: keep this capital out of the cash remainder
                deployed += cap
                unknown_hold_value += cap
                print("     holdings unknown without --ledger: existing positions "
                      "stand (capital excluded from the cash remainder)")
            continue
        w = w[w.abs() > 1e-9]
        for tkr, wt in w.sort_values(ascending=False).items():
            dollars = cap * wt
            shares = np.floor(dollars / last_px[tkr])
            deployed += shares * last_px[tkr]
            target_dollars += dollars
            sized_dollars += shares * last_px[tkr]
            port_target_shares[tkr] = port_target_shares.get(tkr, 0.0) + shares
            book[tkr] = book.get(tkr, 0.0) + shares
            print(f"     {tkr:5s} {wt:7.2%}  ${dollars:>9,.0f}  -> {shares:>6.0f} sh"
                  f" @ {last_px[tkr]:,.2f}")
        if len(w) == 0:
            print("     flat (all cash)")
    if target_dollars > 0:
        print(f"\nwhole-share rounding: today's sleeve targets ${target_dollars:,.0f} are sized "
              f"${sized_dollars:,.0f} ({sized_dollars / target_dollars - 1:+.1%}); the difference "
              "stays in cash parking")

    # park the un-deployed remainder in T-bills
    resid = equity - deployed
    if resid < -1e-8:
        raise ValueError(f"preserved HOLD positions plus new targets exceed account equity by ${-resid:,.2f}; "
                         "reconcile sleeve allocations before generating orders")
    sgov = 0.0
    if CASH_ETF in last_px.index and np.isfinite(last_px.get(CASH_ETF, np.nan)) and last_px[CASH_ETF] > 0:
        sgov = np.floor(max(resid, 0.0) / last_px[CASH_ETF])
        port_target_shares[CASH_ETF] = port_target_shares.get(CASH_ETF, 0.0) + sgov
    else:
        print(f"\n-- cash remainder ${resid:,.0f} stays unallocated; refresh the cache "
              f"to include {CASH_ETF} for expense-reserved cash parking")

    held_net = ledger.groupby("ticker").sum() if ledger is not None else pd.Series(dtype=float)
    before_sgov = port_target_shares.get(CASH_ETF, 0.0)
    port_target_shares, reserve = funded_targets(
        port_target_shares, held_net, last_px, equity, sgov,
        args.min_order if ledger is not None else 0.0, args.price_buffer_bps, unknown_hold_value)
    parked = sgov - (before_sgov - port_target_shares.get(CASH_ETF, 0.0))
    sleeve_shares["cash"] = {CASH_ETF: parked} if parked else {}

    # Size every order before printing any: an order that cannot be priced
    # stops the run instead of silently dropping out of the sheet.
    orders: list[tuple[str, float, float]] = []
    unsent: set[str] = set()
    skipped: list[str] = []
    if ledger is not None:
        for tkr in sorted(set(port_target_shares) | set(held_net.index)):
            delta = port_target_shares.get(tkr, 0.0) - float(held_net.get(tkr, 0.0))
            if delta == 0:
                continue
            notional = abs(delta) * float(last_px.get(tkr, np.nan))
            if not np.isfinite(notional):
                raise ValueError(f"no executable close for {tkr}: its order cannot be sized")
            if notional < args.min_order:
                unsent.add(tkr)
                skipped.append(f"   {tkr:5s} skip {delta:+.0f} sh (${notional:,.0f} < min-order)")
                continue
            orders.append((tkr, delta, notional))
        proposed = post_trade_ledger(sleeve_shares, ledger, unsent, t)
        after = proposed.groupby("ticker")["shares"].sum()
        for tkr in set(after.index) | set(held_net.index) | {o[0] for o in orders}:
            sent = sum(delta for name, delta, _ in orders if name == tkr)
            if float(after.get(tkr, 0.0)) != float(held_net.get(tkr, 0.0)) + sent:
                raise ValueError(f"per-sleeve positions do not add up to the net order for {tkr}")

    if CASH_ETF in port_target_shares:
        print(f"\n-- cash parking after expense reserve: {parked:.0f} sh {CASH_ETF}")
    print(f"execution cash reserve ${reserve:,.2f} (commissions, sell fees, "
          f"{args.price_buffer_bps:g} bps adverse-price allowance)")
    print("\n== NET PORTFOLIO TARGET (shares at decision close)")
    for tkr in sorted(port_target_shares):
        sh = port_target_shares[tkr]
        if sh != 0:
            print(f"   {tkr:5s} {sh:>8.0f} sh  (${sh * last_px.get(tkr, np.nan):>10,.0f})")

    if ledger is not None:
        # The ledger file is written before any order line: a file that
        # cannot be written stops the run while there is still no order
        # sheet, never after one has been printed.
        if args.write_ledger is not None:
            with args.write_ledger.open("x", newline="", encoding="utf-8") as stream:
                proposed.to_csv(stream, index=False)
        print("\n== MOC ORDERS (target - ledger)")
        for line in skipped:
            print(line)
        for tkr, delta, notional in orders:
            side = "BUY " if delta > 0 else "SELL"
            print(f"   {side} {abs(delta):>6.0f} {tkr:5s} MOC   (~${notional:,.0f})")
        if not orders:
            print("   none")
        if args.write_ledger is not None:
            print(f"\nproposed post-trade ledger written to {args.write_ledger} (as_of {t.date()}): "
                  "it assumes every order above fills in full; correct it from the broker's "
                  "confirmed fills before the next run")
    else:
        print("\n(no --ledger given: showing targets only, no order deltas)")


def cli() -> None:
    """Exit 2 when the run is refused, 3 on a crash; never the default 1."""
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"\nLIVE TARGETS REFUSED: {exc}\nno order sheet: nothing printed above may be traded",
              file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    except Exception:  # noqa: BLE001 - a crash must not look like a verdict
        traceback.print_exc()
        print("\nLIVE TARGETS CRASHED: nothing printed above may be traded", file=sys.stderr)
        sys.exit(EXIT_CRASH)


if __name__ == "__main__":
    cli()
