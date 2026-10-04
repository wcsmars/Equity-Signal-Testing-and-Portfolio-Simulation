"""The made-up example inputs in examples/ work with the scripts they illustrate.

examples/live_returns.example.csv  ->  scripts/monitor.py
examples/fills.example.csv         ->  scripts/reconcile.py
examples/ledger.example.csv        ->  scripts/live_targets.py

The numbers in those files are invented. These tests only pin that each file
is in the documented format and that the documented command reaches a
verdict or an order sheet with it. Prices are synthetic; no data cache and no
saved result is read.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

from test_live_targets_rules import _live, _orders, _panel, _run, module

ROOT = Path(__file__).resolve().parents[1]
# The examples sit in examples/ of the release. A development checkout that
# keeps the release in a nested folder has them one level down.
EXAMPLES = next((folder for folder in (ROOT / "examples", ROOT / "public" / "examples")
                 if (folder / "ledger.example.csv").is_file()), ROOT / "examples")
LAST_RETURN_DATE = "2025-03-31"


def _main(mod, monkeypatch, capsys, name, *argv):
    monkeypatch.setattr(sys, "argv", [name, *map(str, argv)])
    try:
        mod.main()
        code = 0
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


def test_example_files_are_present():
    for name in ("live_returns.example.csv", "fills.example.csv", "ledger.example.csv"):
        assert (EXAMPLES / name).is_file(), name


def test_example_live_returns_reach_a_verdict_without_a_data_cache(monkeypatch, capsys):
    monitor = module("scripts/monitor.py")

    def no_cache(*args, **kwargs):
        raise AssertionError("the cash rate was read for a file shorter than two years")

    monkeypatch.setattr(monitor, "_irx_series", no_cache)
    path = EXAMPLES / "live_returns.example.csv"
    frame = monitor.read_returns(path)
    assert list(frame.columns) == ["ENSEMBLE"] and str(frame.index[-1].date()) == LAST_RETURN_DATE
    off_session, missing = monitor.session_mismatch(frame.index)
    assert len(off_session) == 0 and len(missing) == 0
    assert frame["ENSEMBLE"].abs().max() < 0.02          # fractions, not percent
    code, out = _main(monitor, monkeypatch, capsys, "monitor.py", path, "--as-of", LAST_RETURN_DATE)
    assert code == 0
    assert "mode: LIVE account" in out and "an offline replay" in out
    assert f"pending: {len(frame)}/504 observations" in out
    assert out.rstrip().endswith("overall: ok")
    # judged against today's clock the same file is stale: a refusal, not a verdict
    monkeypatch.setattr(monitor, "current_ny_date", lambda: pd.Timestamp("2026-01-05"))
    code, out = _main(monitor, monkeypatch, capsys, "monitor.py", path)
    assert code == 3 and "stale live file" in out and "overall" not in out


def test_example_fills_reconcile_to_ok(monkeypatch, capsys):
    reconcile = module("scripts/reconcile.py")
    path = EXAMPLES / "fills.example.csv"
    fills = pd.read_csv(path)
    assert list(fills.columns) == ["date", "ticker", "side", "shares", "price", "commission",
                                   "reference_close"]
    out = reconcile.reconcile_fills(fills)
    summary = reconcile.summarize(out, 3.0)
    assert (summary["fills"], summary["orders"]) == (7, 6)   # one order arrived as two partial fills
    assert not summary["slippage_breach"] and not summary["commission_review_trigger"]
    assert not summary["commission_excess"] and not summary["implausibly_favourable"]
    assert 0 < summary["slip_bps_weighted"] < 1
    code, shown = _main(reconcile, monkeypatch, capsys, "reconcile.py", path)
    assert code == 0
    assert shown.rstrip().endswith("ok: slippage and commissions within the modeled assumptions")


def _priced(probe, end):
    px = _panel(probe, end, start="2023-01-03")
    px["SGOV"] = 100.5
    return px


def test_example_ledger_is_dated_and_holds_each_ticker_under_a_sleeve_that_trades_it():
    live = module("scripts/live_targets.py")
    ledger = live.read_ledger(EXAMPLES / "ledger.example.csv")
    assert ledger.attrs["as_of"] == pd.Timestamp(LAST_RETURN_DATE)
    universes = {**live.sleeve_universes(), "cash": {live.CASH_ETF}}
    assert set(ledger.index.get_level_values("sleeve")) == set(universes)
    for (sleeve, ticker), shares in ledger.items():
        assert ticker in universes[sleeve] and shares > 0, (sleeve, ticker)


def test_example_ledger_prints_an_order_sheet_on_the_session_after_its_date(monkeypatch, capsys):
    probe = module("scripts/live_targets.py")
    ledger = EXAMPLES / "ledger.example.csv"
    before = ledger.read_bytes()
    live = _live(monkeypatch, _priced(probe, "2025-04-01"))      # the session after the ledger's as_of
    assert _run(live, monkeypatch, "--as-of", "2025-04-01", "--cash", "5000", "--ledger", str(ledger)) == 0
    out = capsys.readouterr().out
    assert "OFFLINE historical simulation" in out and "cash $5,000.00 (as given)" in out
    assert "!!" not in out and "no as_of date" not in out
    orders = _orders(out)
    assert "MOC" in orders                                   # the daily sleeves decide every session
    for line in out.splitlines():                            # the monthly sleeves carry their shares
        if " holds " in line:
            assert line.split()[0] in {"GLD", "IEF", "EWJ", "XLK"}
    assert out.count(" holds ") == 4
    assert ledger.read_bytes() == before                     # the input ledger is never written

    # one session later the same ledger is a day old: refused until the flag is given
    later = _live(monkeypatch, _priced(probe, "2025-04-02"))
    argv = ["--as-of", "2025-04-02", "--cash", "5000", "--ledger", str(ledger)]
    monkeypatch.setattr(sys, "argv", ["live_targets.py", *argv])
    with pytest.raises(ValueError, match="previous NYSE session was 2025-04-01"):
        later.main()
    assert "MOC ORDERS" not in capsys.readouterr().out
    assert _run(later, monkeypatch, *argv, "--ignore-ledger-date") == 0
    assert "MOC ORDERS" in capsys.readouterr().out
