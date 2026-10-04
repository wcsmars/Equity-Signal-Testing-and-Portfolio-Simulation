"""Execution check: measured fills against the backtest's assumptions.

The backtest assumes three things about every trade: it fills AT the
official close (a market-on-close order), it pays the modeled per-share
commission with a $1 minimum plus sell-side regulatory fees, and slippage is
2, 3 or 5 bps per side depending on the liquidity tier. This script measures
whether a set of actual fills agrees. It reads a file and prints a verdict;
it places no orders and talks to no broker.

Input: a fills CSV (exported from the broker or kept by hand), columns:
    date,ticker,side,shares,price,commission,reference_close
side in {BUY,SELL}. One row per fill; date is YYYY-MM-DD (a broker's
yyyyMMdd integer is accepted). commission is everything the broker charged
for the fill, including regulatory fees. examples/fills.example.csv is a
made-up file in this format.
Every fill requires reference_close: the final official close in the same
historical traded units as the fill. The price cache's close.csv is
split-adjusted, and a same-date cache may still contain a pre-close
snapshot. Neither the latest row nor the current date proves an official
traded-unit reference, so no cached-price fallback is used and no data cache
is needed.

For each fill it reports:
  - slippage vs the supplied official close of that date, SIGNED so that
    positive = worse than MOC (bought above close / sold below close);
  - commission vs the cost model's prediction. Rows sharing date, ticker
    and side are one order: the $1 minimum applies once and the modeled
    cost is split across its partial fills by shares. Sells include the
    modeled SEC and FINRA fees, as the engine and the order sizing do;
and summarizes per ticker and overall vs the modeled assumptions. Slippage
is judged NOTIONAL-WEIGHTED (dollars lost over dollars traded), the unit in
which the engine charges it: a large bad fill is not diluted by small good
ones, one odd lot cannot breach on its own, and the verdict does not depend
on how the broker split an order into partial fills.

Exit codes:
  0  slippage and commissions within the modeled assumptions (a REVIEW
     line is printed, and no "ok", when commissions exceed the model by
     less than the commission tolerance or fills beat the close implausibly)
  1  BREACH - notional-weighted slippage exceeds the modeled assumption by
     more than 2 bps (assumed + 2 bps): the pre-registered signal to stop
     and re-examine execution.
     COMMISSION REVIEW TRIGGER - commissions paid exceed the modeled
     commissions and fees by more than 2 bps of traded notional. This
     second gate also exits 1 but is a review trigger, not a pre-registered
     rule: it reports that the cost model understates the charges and asks
     for the pricing plan and fee schedule to be checked. It has its own
     constant and leaves the slippage threshold untouched.
  2  RECONCILIATION FAILED - fills file missing, unreadable or invalid
  3  unexpected crash (never a verdict)

Usage: python scripts/reconcile.py examples/fills.example.csv
       python scripts/reconcile.py fills.csv [--slippage-assumed 3.0]

Limits: the verdict is only as good as the reference_close column, which
the user supplies; the default assumption is 3 bps for every fill, so a file
that mixes liquidity tiers should be split or judged at the tier that
applies (--slippage-assumed); a handful of fills is weak evidence either way.
"""

import argparse
import csv
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.costs import IBKRHKCostModel  # noqa: E402

TOLERANCE_BPS = 2.0  # breach = notional-weighted slippage > assumed + this
# Review trigger, not a pre-registered rule: commissions paid above the
# modeled commissions and fees by more than this many bps of traded notional.
COMMISSION_REVIEW_BPS = 2.0
ROUNDING_PER_ORDER = 0.005  # USD; a commission gap inside cent rounding is not an excess
EXIT_BREACH = 1
EXIT_INVALID = 2
EXIT_CRASH = 3
_DATE_ERROR = "fill dates must be valid timezone-naive trading dates"


def _parse_dates(column: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(column):
        raise ValueError(_DATE_ERROR)
    if pd.api.types.is_numeric_dtype(column):
        # A broker's yyyyMMdd integer would otherwise be read as nanoseconds
        # since 1970 and rejected with a misleading time-of-day message.
        values = column.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values != np.round(values)).any():
            raise ValueError(_DATE_ERROR)
        text = pd.Series([str(int(v)) for v in values], index=column.index)
        if (text.str.len() != 8).any() or (values <= 0).any():
            raise ValueError("numeric fill dates must be yyyyMMdd")
        try:
            return pd.to_datetime(text, format="%Y%m%d", errors="raise")
        except ValueError as exc:
            raise ValueError("numeric fill dates must be yyyyMMdd") from exc
    with warnings.catch_warnings():
        # Mixed UTC offsets: an error in new pandas, a FutureWarning before.
        warnings.simplefilter("error", FutureWarning)
        try:
            parsed = pd.to_datetime(column, errors="raise")
        except FutureWarning as exc:
            raise ValueError(_DATE_ERROR) from exc
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError(_DATE_ERROR)
    return parsed


def modeled_cost(cm: IBKRHKCostModel, side: str, shares: float, notional: float) -> float:
    """Modeled commission for one order, plus sell-side SEC and FINRA fees."""
    commission = min(max(cm.min_commission, cm.commission_per_share * shares),
                     cm.max_commission_pct * notional)
    fees = (cm.sec_fee_rate * notional + min(cm.finra_taf_per_share * shares, 8.30)
            if side == "SELL" else 0.0)
    return commission + fees


def reconcile_fills(fills: pd.DataFrame, close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reconcile every input fill, or fail; missing observations never mean OK.

    ``close`` is accepted for older callers and never read: a cached price
    cannot prove a traded-unit official close, so each fill carries its own
    reference_close.
    """
    need = {"date", "ticker", "side", "shares", "price", "commission"}
    if not fills.columns.is_unique:
        raise ValueError("fills columns must be unique")
    if missing := need - set(fills.columns):
        raise ValueError(f"fills CSV missing columns: {sorted(missing)}")
    if fills.empty:
        raise ValueError("fills CSV is empty")
    fills = fills.copy()
    fills["date"] = _parse_dates(fills["date"])
    if fills["date"].isna().any() or fills["date"].dt.tz is not None:
        raise ValueError(_DATE_ERROR)
    if not fills["date"].equals(fills["date"].dt.normalize()):
        raise ValueError("fill dates must be session dates without time-of-day")
    for col in ("ticker", "side"):
        if fills[col].isna().any():
            raise ValueError(f"fill {col} cannot be missing")
        fills[col] = fills[col].astype(str).str.strip().str.upper()
    if (fills["ticker"] == "").any():
        raise ValueError("fill ticker cannot be blank")
    if not fills["side"].isin(["BUY", "SELL"]).all():
        raise ValueError("fill side must be BUY or SELL")
    for col in ("shares", "price", "commission"):
        fills[col] = pd.to_numeric(fills[col], errors="raise")
        if not np.isfinite(fills[col]).all():
            raise ValueError(f"fill {col} must be finite")
    if (fills[["shares", "price"]] <= 0).any().any() or (fills["commission"] < 0).any():
        raise ValueError("fill shares/price must be positive and commission nonnegative")
    cm = IBKRHKCostModel()
    # One order = every fill row sharing date, ticker and side: the order
    # minimum is charged once, not once per partial fill.
    order_keys = [fills["date"], fills["ticker"], fills["side"]]
    order_shares = fills.groupby(order_keys)["shares"].transform("sum")
    order_notional = (fills["shares"] * fills["price"]).groupby(order_keys).transform("sum")
    rows = []
    for position, f in enumerate(fills.itertuples()):
        supplied = getattr(f, "reference_close", np.nan)
        if pd.notna(supplied):
            try:
                ref = float(supplied)
            except (TypeError, ValueError) as exc:
                raise ValueError("reference_close must be a finite positive traded-unit price") from exc
        else:
            raise ValueError("every fill requires reference_close: the final official close "
                             "in the fill's historical traded units; cached prices cannot prove this")
        if not np.isfinite(ref) or ref <= 0:
            raise ValueError(f"invalid reference_close for {f.ticker} on {f.date.date()}")
        sign = 1.0 if f.side == "BUY" else -1.0
        slip_bps = sign * (f.price - ref) / ref * 1e4
        whole_order = modeled_cost(cm, f.side, float(order_shares.iloc[position]),
                                   float(order_notional.iloc[position]))
        commission_model = whole_order * f.shares / float(order_shares.iloc[position])
        rows.append({
            "date": f.date.date(), "ticker": f.ticker, "side": f.side,
            "shares": f.shares, "fill": f.price, "close": ref,
            "slip_bps": slip_bps, "comm_paid": f.commission,
            "comm_model": commission_model,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, slippage_assumed: float) -> dict:
    """Overall measures and the verdicts they imply, from reconcile_fills output."""
    sign = np.where(df["side"] == "BUY", 1.0, -1.0)
    slip_usd = sign * (df["fill"] - df["close"]) * df["shares"]
    notional = float((df["close"] * df["shares"]).sum())
    weighted = float(slip_usd.sum()) / notional * 1e4
    comm_gap = float((df["comm_paid"] - df["comm_model"]).sum())
    comm_gap_bps = comm_gap / notional * 1e4
    orders = int(len(df.groupby(["date", "ticker", "side"])))
    return {
        "fills": int(len(df)), "orders": orders, "notional": notional,
        "slip_usd": float(slip_usd.sum()), "slip_bps_weighted": weighted,
        "slip_bps_mean": float(df["slip_bps"].mean()),
        "comm_gap": comm_gap, "comm_gap_bps": comm_gap_bps,
        "slippage_breach": weighted > slippage_assumed + TOLERANCE_BPS,
        "commission_review_trigger": comm_gap_bps > COMMISSION_REVIEW_BPS,
        "commission_excess": comm_gap > ROUNDING_PER_ORDER * orders,
        "implausibly_favourable": weighted < -(slippage_assumed + TOLERANCE_BPS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0])
    ap.add_argument("fills", type=Path)
    ap.add_argument("--slippage-assumed", type=float, default=3.0,
                    help="modeled bps/side for this book (2/3/5 by tier)")
    args = ap.parse_args()
    if not np.isfinite(args.slippage_assumed) or args.slippage_assumed < 0:
        ap.error("--slippage-assumed must be finite and nonnegative")

    try:
        with args.fills.open(newline="", encoding="utf-8-sig") as stream:
            header = [name.strip() for name in next(csv.reader(stream), [])]
        if not header or len(header) != len(set(header)):
            raise ValueError("fills CSV needs a nonempty unique header")
        fills = pd.read_csv(args.fills)
        fills.columns = [str(name).strip() for name in fills.columns]
    except (OSError, ValueError, csv.Error) as exc:  # ParserError, EmptyDataError, bad encoding
        print(f"RECONCILIATION FAILED: cannot read fills file {args.fills}: {exc}")
        sys.exit(EXIT_INVALID)
    try:
        df = reconcile_fills(fills)
    except ValueError as exc:
        print(f"RECONCILIATION FAILED: {exc}")
        sys.exit(EXIT_INVALID)
    print(df.round({"close": 4, "slip_bps": 2, "comm_model": 2}).to_string(index=False))
    print("\ncommission model: fills sharing date, ticker and side are one order (one $1 "
          "minimum); sells include modeled SEC and FINRA fees")

    print("\nper-ticker slippage (bps, +ve = worse than MOC):")
    sign = np.where(df["side"] == "BUY", 1.0, -1.0)
    work = df.assign(notional=df["close"] * df["shares"],
                     slip_usd=sign * (df["fill"] - df["close"]) * df["shares"])
    per = work.groupby("ticker").agg(fills=("slip_bps", "count"), notional=("notional", "sum"),
                                     slip_usd=("slip_usd", "sum"), mean=("slip_bps", "mean"),
                                     max=("slip_bps", "max"))
    per.insert(2, "weighted", per["slip_usd"] / per["notional"] * 1e4)
    print(per.round(2).to_string())

    s = summarize(df, args.slippage_assumed)
    print(f"\noverall: {s['fills']} fills in {s['orders']} orders | traded notional "
          f"${s['notional']:,.0f} | slippage {s['slip_bps_weighted']:+.2f} bps notional-weighted "
          f"(${s['slip_usd']:+,.2f}; unweighted mean {s['slip_bps_mean']:+.2f}; modeled "
          f"{args.slippage_assumed:.1f}) | commission paid-vs-model ${s['comm_gap']:+.2f} total "
          f"({s['comm_gap_bps']:+.2f} bps of notional)")

    if s["slippage_breach"]:
        print(f"\nBREACH: notional-weighted slippage exceeds the model by more than "
              f"{TOLERANCE_BPS} bps - execution does not match the backtest's "
              "assumptions. Stop and investigate (order type? timing? venue?).")
    if s["commission_review_trigger"]:
        print(f"\nCOMMISSION REVIEW TRIGGER: commissions paid exceed the modeled commissions and "
              f"fees by more than {COMMISSION_REVIEW_BPS} bps of traded notional - the cost model "
              "understates this account's charges. Stop and investigate (pricing plan? order "
              "splitting? fees?). This is a review trigger with exit code 1, not the "
              "pre-registered slippage rule.")
    if s["slippage_breach"] or s["commission_review_trigger"]:
        sys.exit(EXIT_BREACH)
    review = False
    if s["commission_excess"]:
        review = True
        print(f"\nREVIEW: commissions paid exceed the model by ${s['comm_gap']:.2f} "
              f"({s['comm_gap_bps']:.2f} bps of traded notional, inside the "
              f"{COMMISSION_REVIEW_BPS} bps review trigger). Check the fee schedule against the "
              "cost model.")
    if s["implausibly_favourable"]:
        review = True
        print(f"\nREVIEW: fills beat the official close by {-s['slip_bps_weighted']:.2f} bps, more "
              "than the breach gate's own width. An MOC order fills at the close: check each "
              "fill's side and reference_close (units, split adjustment).")
    if review:
        print("\nno breach, but not confirmed: resolve the REVIEW lines above")
    else:
        print("\nok: slippage and commissions within the modeled assumptions")


def cli() -> None:
    """Exit 3 on a crash: python's default 1 would read as a BREACH."""
    try:
        main()
    except Exception:  # noqa: BLE001 - a crash must not look like a verdict
        traceback.print_exc()
        print("RECONCILIATION CRASHED: no verdict")
        sys.exit(EXIT_CRASH)


if __name__ == "__main__":
    cli()
