"""Independent operational funding and fill-reference regressions (no orders)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cash_parking_keeps_cash_for_commissions_and_adverse_prices():
    live = module("live_targets")
    prices = pd.Series({"SPY": 100.0, "SGOV": 100.0})
    target, reserve = live.funded_targets({"SPY": 8.0, "SGOV": 2.0}, pd.Series(dtype=float),
                                         prices, 1000.0, 2.0, 0.0, 25.0)
    assert target == {"SPY": 8.0, "SGOV": 1.0}
    assert reserve == pytest.approx(2.0 + 900.0 * 0.0025)
    assert sum(shares * prices[ticker] for ticker, shares in target.items()) + reserve <= 1000.0


def test_fully_invested_risky_target_does_not_silently_cut_strategy():
    live = module("live_targets")
    with pytest.raises(ValueError, match="unfunded after commissions"):
        live.funded_targets({"SPY": 10.0}, pd.Series(dtype=float), pd.Series({"SPY": 100.0}),
                            1000.0, 0.0, 0.0, 0.0)


def test_sell_side_fees_are_not_averaged_away_for_funding():
    live = module("live_targets")
    prices = pd.Series({"SPY": 100.0})
    buy = live.execution_reserve({"SPY": 10.0}, prices, 0.0)
    sell = live.execution_reserve({"SPY": -10.0}, prices, 0.0)
    assert buy == 1.0
    assert sell == pytest.approx(1.0 + 1000 * 0.0000278 + 10 * 0.000166)


def test_minimum_order_filter_cannot_hide_a_funding_sale():
    live = module("live_targets")
    prices = pd.Series({"SGOV": 100.0, "OLD": 20.0, "SPY": 100.0})
    # The suppressed $20 sale stays in the book: its cash comes out of the
    # newly parked SGOV (one share), not out of the whole order sheet.
    target, reserve = live.funded_targets({"SGOV": 10.0}, pd.Series({"OLD": 1.0}), prices,
                                          1000.0, 10.0, 50.0, 25.0)
    assert target == {"SGOV": 9.0}
    assert reserve == pytest.approx(1.0 + 900.0 * 0.0025)
    assert 20.0 + 900.0 + reserve <= 1000.0
    # Without the filter the sale is sent and nothing has to be trimmed.
    target, _ = live.funded_targets({"SGOV": 9.0}, pd.Series({"OLD": 1.0}), prices,
                                    1000.0, 9.0, 0.0, 25.0)
    assert target == {"SGOV": 9.0}
    # Parking that cannot cover the suppressed sale still refuses the sheet.
    with pytest.raises(ValueError, match="orders are unfunded .*--min-order may be suppressing"):
        live.funded_targets({"SPY": 10.0, "SGOV": 0.0}, pd.Series({"OLD": 2.0}), prices,
                            1000.0, 0.0, 50.0, 0.0)
    with pytest.raises(ValueError, match="orders are unfunded"):
        live.funded_targets({"SPY": 10.0, "SGOV": 1.0}, pd.Series({"OLD": 2.0}), prices,
                            1000.0, 1.0, 50.0, 25.0)


def test_research_targets_are_never_trimmed_to_fund_a_suppressed_sale():
    live = module("live_targets")
    prices = pd.Series({"SGOV": 100.0, "OLD": 20.0, "SPY": 100.0})
    target, _ = live.funded_targets({"SPY": 5.0, "SGOV": 5.0}, pd.Series({"OLD": 1.0}), prices,
                                    1000.0, 5.0, 50.0, 25.0)
    assert target == {"SPY": 5.0, "SGOV": 4.0}


def test_rounding_near_integer_ledger_values_cannot_emit_fractional_orders(tmp_path):
    live = module("live_targets")
    path = tmp_path / "ledger.csv"
    path.write_text("sleeve,ticker,shares\ncash,SGOV,0.0000000005\n")
    assert live.read_ledger(path).iloc[0] == 0.0


@pytest.mark.parametrize("reference", [None, np.nan, np.inf, -1.0, 0.0])
def test_every_fill_needs_an_explicit_valid_traded_unit_reference(reference):
    reconcile = module("reconcile")
    fills = pd.DataFrame({"date": ["2024-01-02"], "ticker": ["SPY"], "side": ["BUY"],
                          "shares": [10.0], "price": [400.0], "commission": [1.0]})
    if reference is not None:
        fills["reference_close"] = reference
    # Even the latest date in a truncated cache can be split-adjusted by
    # events occurring after the historical fill.
    close = pd.DataFrame({"SPY": [100.0]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError, match="reference_close"):
        reconcile.reconcile_fills(fills, close)


def test_fill_reference_works_without_a_market_data_cache():
    reconcile = module("reconcile")
    fills = pd.DataFrame({"date": ["2024-01-02"], "ticker": ["SPY"], "side": ["BUY"],
                          "shares": [10.0], "price": [400.0], "commission": [1.0],
                          "reference_close": [400.0]})
    assert reconcile.reconcile_fills(fills).slip_bps.iloc[0] == 0.0


def test_explicit_reference_does_not_allow_an_unidentified_fill():
    reconcile = module("reconcile")
    fills = pd.DataFrame({"date": ["2024-01-02"], "ticker": ["  "], "side": ["BUY"],
                          "shares": [10.0], "price": [400.0], "commission": [1.0],
                          "reference_close": [400.0]})
    with pytest.raises(ValueError, match="ticker cannot be blank"):
        reconcile.reconcile_fills(fills)


def test_sgov_operational_quote_does_not_enter_research_universes(monkeypatch):
    from qcore import data
    assert "SGOV" in data.OPERATIONAL_UNIVERSE
    assert "SGOV" not in data.ETF_UNIVERSE + data.STOCK_UNIVERSE
    frame = pd.DataFrame({"SPY": [100.0], "SGOV": [100.0]}, index=pd.to_datetime(["2024-01-02"]))
    monkeypatch.setattr(data, "load", lambda name: frame)
    assert list(data.load_prices()) == ["SPY"]
