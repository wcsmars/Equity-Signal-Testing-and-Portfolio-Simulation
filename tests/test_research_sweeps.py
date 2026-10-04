"""Selection sweeps under research/: the mean-reversion grid and the rank
hysteresis / execution-mode study.

Covers: both scripts write result files only through qcore.records; the
mean-reversion grid is built by the sleeve's own state machine, keeps a saved
grid that differs and reports a tie for first place; the hysteresis rule, the
month-end-only expansion, the reproduction of the strategy modules' own
weights and the guarded records of the hysteresis study.

Everything runs on synthetic prices with ROOT redirected to tmp_path: no test
reads the data cache or touches results/.
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore.costs import IBKRHKCostModel
from qcore.quality import nyse_bdays

from test_strategy_rules_nonprod import _direct_writes

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("mean_reversion_sweep", "rank_hysteresis_sweep")


def module(name):
    path = ROOT / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name + "_sweep_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No data cache, no --rebase, and the scripts' sys.path edits undone."""
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())
    monkeypatch.setattr(bt, "_dividend_yields", lambda: pd.DataFrame())
    monkeypatch.setattr(bt, "_irx_series", lambda: pd.Series(dtype=float))
    monkeypatch.delenv("QCORE_REBASE", raising=False)
    monkeypatch.setattr(sys, "argv", ["study.py"])
    monkeypatch.setattr(sys, "path", list(sys.path))


# ---- writes and run instructions ---------------------------------------------

@pytest.mark.parametrize("name", SCRIPTS)
def test_sweeps_write_results_only_through_the_record_helpers(name):
    assert _direct_writes(ROOT / "research" / f"{name}.py") == []


@pytest.mark.parametrize("name", SCRIPTS)
def test_sweep_docs_state_the_run_line_the_record_policy_and_the_exit_status(name):
    doc = ast.get_docstring(ast.parse((ROOT / "research" / f"{name}.py").read_text()))
    assert f" research/{name}.py" in doc
    assert "--rebase" in doc and "results/recomputed/" in doc
    assert "exit status 1" in doc.lower() or "exit status: 0" in doc.lower()


# ---- mean-reversion grid -----------------------------------------------------

def test_mean_reversion_grid_never_replaces_a_saved_grid_that_differs(monkeypatch, tmp_path, capsys):
    mod = module("mean_reversion_sweep")
    idx = pd.bdate_range("2016-06-01", periods=420)

    def panel(seed):
        steps = np.random.default_rng(seed).normal(0.0003, 0.012, (len(idx), len(mod.UNIVERSE)))
        return pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=idx,
                            columns=mod.UNIVERSE)

    held = {"px": panel(1)}
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_prices", lambda: held["px"])
    grid = tmp_path / "results" / "mean_reversion_variants.csv"

    mod.main()
    first = grid.read_text()
    frame = pd.read_csv(grid)
    assert len(frame) == 12 and list(frame.columns)[:4] == ["name", "entry_th", "exit_th", "max_hold"]
    assert frame["ann_turnover"].gt(0).any()

    held["px"] = panel(2)
    mod.main()
    assert "kept existing record" in capsys.readouterr().out
    assert grid.read_text() == first
    assert (tmp_path / "results" / "recomputed" / "mean_reversion_variants.csv").read_text() != first


def _reversion_grid(monkeypatch, tmp_path):
    mod = module("mean_reversion_sweep")
    idx = pd.bdate_range("2016-06-01", periods=420)
    steps = np.random.default_rng(5).normal(0.0003, 0.012, (len(idx), len(mod.UNIVERSE)))
    px = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=mod.UNIVERSE)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_prices", lambda: px)
    return mod, px


def test_mean_reversion_grid_is_built_by_the_production_state_machine(monkeypatch, tmp_path):
    from strategies import mean_reversion as production
    mod, px = _reversion_grid(monkeypatch, tmp_path)
    assert mod.build_weights is production.build_weights and not hasattr(mod, "rsi")
    assert mod.UNIVERSE == production.UNIVERSE and mod.SLIPPAGE_BPS == production.SLIPPAGE_BPS
    assert mod.W_MAX == production.BEST_PARAMS["w_max"]
    tree = ast.parse((ROOT / "research" / "mean_reversion_sweep.py").read_text())
    assert not [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name in {"build_weights", "rsi"}]  # no private copy of the rule

    calls = []

    def spy(prices, entry_th, exit_th, max_hold, w_max):
        calls.append((entry_th, exit_th, max_hold, w_max))
        pd.testing.assert_frame_equal(prices, px)
        return production.build_weights(prices, entry_th, exit_th, max_hold, w_max)

    monkeypatch.setattr(mod, "build_weights", spy)
    mod.main()
    cells = [(e, x, h, 0.20) for e in (5, 10, 15) for x in (60, 70) for h in (5, 10)]
    assert calls[:12] == cells and calls[12] in cells and len(calls) == 13
    # the retained sleeve parameters are one cell of this grid
    best = production.BEST_PARAMS
    assert (best["entry_th"], best["exit_th"], best["max_hold"], best["w_max"]) in cells
    frame = pd.read_csv(tmp_path / "results" / "mean_reversion_variants.csv").set_index("name")
    w = production.build_weights(px, 10, 70, 5, 0.20)
    assert frame.loc["e10_x70_h5", "avg_exposure"] == round(float(w.sum(axis=1).mean()), 3)


def test_mean_reversion_grid_reports_a_tie_for_first_place(monkeypatch, tmp_path, capsys):
    mod, _ = _reversion_grid(monkeypatch, tmp_path)
    engine_metrics = mod.metrics
    level = {}

    def fixed(res):
        m = engine_metrics(res)
        m["in_sample"] = dict(m["in_sample"], sharpe=level.get(m["name"], 0.10))
        return m

    monkeypatch.setattr(mod, "metrics", fixed)
    level.update(e10_x60_h5=0.50, e5_x70_h10=0.50)      # two cells tie at 2 decimals
    mod.main()
    shown = capsys.readouterr().out
    assert "BEST BY IS SHARPE: e5_x70_h10" in shown       # grid order, not a selection
    assert "NOTE: 2 variants tie for the best IS Sharpe" in shown
    assert "e5_x70_h10, e10_x60_h5" in shown
    order = list(pd.read_csv(tmp_path / "results" / "mean_reversion_variants.csv")["name"])
    assert order[:2] == ["e5_x70_h10", "e10_x60_h5"]
    assert order[2:] == [f"e{e}_x{x}_h{h}" for e in (5, 10, 15) for x in (60, 70) for h in (5, 10)
                         if (e, x, h) not in {(5, 70, 10), (10, 60, 5)}]

    level["e10_x60_h5"] = 0.49                            # a clear winner: nothing to report
    mod.main()
    shown = capsys.readouterr().out
    assert "BEST BY IS SHARPE: e5_x70_h10" in shown and "NOTE:" not in shown


# ---- rank hysteresis: the rule -----------------------------------------------

def _scores(**values):
    return pd.Series(values, dtype=float)


def test_hysteresis_keeps_an_incumbent_inside_the_buffer_and_sells_it_outside():
    mod = module("rank_hysteresis_sweep")
    s = _scores(a=9, b=8, c=7, d=6, e=5, f=4)             # ranks 1..6
    # nothing held: the plain top K
    assert mod.pick_holdings(s, [], k=2, buffer=5) == ["a", "b"]
    # e is rank 5: kept at buffer 5, with the best outsider beside it ...
    assert mod.pick_holdings(s, ["e"], k=2, buffer=5) == ["e", "a"]
    # ... and sold at buffer 4
    assert mod.pick_holdings(s, ["e"], k=2, buffer=4) == ["a", "b"]
    # two incumbents inside the buffer leave no free slot, whatever ranks above them
    assert mod.pick_holdings(s, ["d", "e"], k=2, buffer=5) == ["d", "e"]
    # an incumbent that is no longer eligible (no score) is dropped
    assert mod.pick_holdings(s, ["zz", "c"], k=2, buffer=3) == ["c", "a"]


@pytest.mark.parametrize("held", [[], ["f"], ["c", "f"], ["b", "a"], ["e", "d", "f"]])
def test_buffer_equal_to_k_is_the_plain_top_k_rule(held):
    mod = module("rank_hysteresis_sweep")
    s = _scores(a=9, b=8, c=7, d=6, e=5, f=4)
    for k in (1, 2, 3):
        assert set(mod.pick_holdings(s, held, k=k, buffer=k)) == set(s.nlargest(k).index)


# ---- rank hysteresis: synthetic market ---------------------------------------

def _world(mod, seed, start="2014-01-02", end="2019-12-16"):
    """Lognormal closes for every ETF and stock the study reads, on NYSE days."""
    cols = list(dict.fromkeys(mod.etf.EQ_UNIVERSE + [mod.etf.DEFENSIVE] + mod.STOCK_UNIVERSE))
    days = nyse_bdays(start, end)
    rng = np.random.default_rng(seed)
    drift = rng.uniform(-0.0002, 0.0008, len(cols))      # persistent winners and losers
    steps = drift + rng.normal(0.0, 0.011, (len(days), len(cols)))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=days, columns=cols)


def test_study_rules_reproduce_the_strategy_modules_own_weights():
    mod = module("rank_hysteresis_sweep")
    px = _world(mod, seed=11)
    mod.reproduce_baselines(px)                            # raises on any difference
    # the three ties it checks, spelled out for the ETF sleeve
    legacy = mod.ETF_LEGACY
    plain = mod.expand_retarget(mod.etf_monthly(px, buffer=legacy["k"], **legacy), px.index)
    pd.testing.assert_frame_equal(plain, mod.etf.build_weights(px, **legacy))
    adopted = mod.expand_drift(mod.etf_monthly(px, buffer=mod.etf.BEST_PARAMS["buffer"], **legacy), px)
    pd.testing.assert_frame_equal(adopted, mod.etf.build_weights(px, **mod.etf.BEST_PARAMS))
    assert mod.etf.BEST_PARAMS["buffer"] in mod.ETF_BUFFERS and legacy["k"] in mod.ETF_BUFFERS
    assert mod.stk.K in mod.STK_BUFFERS


def test_a_wider_buffer_holds_names_longer_and_never_changes_what_is_eligible():
    mod = module("rank_hysteresis_sweep")
    px = _world(mod, seed=11)
    legacy = mod.ETF_LEGACY
    stats = {b: mod.swap_stats(mod.etf_monthly(px, buffer=b, **legacy), mod.etf.EQ_UNIVERSE)
             for b in (3, 12)}
    assert stats[12]["entries_per_year"] < stats[3]["entries_per_year"]
    assert stats[12]["avg_hold_months"] > stats[3]["avg_hold_months"]
    for b in (3, 12):
        monthly = mod.etf_monthly(px, buffer=b, **legacy)
        rows = monthly.sum(axis=1)
        assert set(np.round(rows.unique(), 12)) <= {0.0, 1.0}   # flat before history, then fully invested
        risky = monthly[mod.etf.EQ_UNIVERSE]
        assert set(risky.gt(0).sum(axis=1).unique()) <= {0, legacy["k"]}


def test_swap_stats_count_entries_and_holding_months():
    mod = module("rank_hysteresis_sweep")
    idx = pd.DatetimeIndex([pd.Timestamp(2020, m, 1) + pd.offsets.MonthEnd(0) for m in range(1, 13)])
    monthly = pd.DataFrame(0.0, index=idx, columns=["A", "B", "DEF"])
    monthly.iloc[0:6, 0] = 1.0                           # A for six months
    monthly.iloc[6:9, 1] = 1.0                           # B for three
    monthly.iloc[9:12, 2] = 1.0                          # then the defensive asset (not counted)
    stats = mod.swap_stats(monthly, ["A", "B"])
    assert stats == {"entries_per_year": 2.0, "avg_hold_months": 4.5}
    empty = mod.swap_stats(monthly * 0.0, ["A", "B"])
    assert np.isnan(empty["entries_per_year"]) and np.isnan(empty["avg_hold_months"])


def _two_asset_book():
    """Two assets with different drifts and a 50/50 target at each month-end."""
    days = nyse_bdays("2021-01-04", "2021-06-30")
    px = pd.DataFrame({"UP": 100.0 * 1.002 ** np.arange(len(days)),
                       "DOWN": 100.0 * 0.999 ** np.arange(len(days))}, index=days)
    month = days.to_period("M")
    month_ends = days[np.append(month[1:] != month[:-1], True)]
    monthly = pd.DataFrame(0.5, index=month_ends, columns=px.columns)
    return px, monthly


def test_month_end_only_expansion_trades_at_month_ends_and_nowhere_else():
    mod = module("rank_hysteresis_sweep")
    px, monthly = _two_asset_book()
    cm = IBKRHKCostModel(slippage_bps=3.0)

    def run(w):
        return bt.run_backtest(w, px, cm, name="book", cash_rate=0.0, withholding=0.0)

    drift = run(mod.expand_drift(monthly, px))
    mod.check_no_intramonth_trades(drift, monthly.index)   # asserts zero turnover off the month-ends
    on = drift["turnover"].reindex(monthly.index)
    assert (on.iloc[1:] > 1e-6).all()                      # each month-end puts the split back to 50/50
    # daily re-targeting of the same targets trades on the days in between
    retarget = run(mod.expand_retarget(monthly, px.index))
    with pytest.raises(AssertionError, match="intramonth trades"):
        mod.check_no_intramonth_trades(retarget, monthly.index)
    assert drift["costs"].sum() < retarget["costs"].sum()
    # lazy: the names never change, so after the first purchase nothing is ordered at all
    lazy = run(mod.expand_drift(monthly, px, lazy=True))
    assert lazy["turnover"].reindex(monthly.index).iloc[1:].abs().max() < 1e-12
    weights = mod.expand_drift(monthly, px, lazy=True)
    assert weights["UP"].iloc[-1] > 0.55 > 0.45 > weights["DOWN"].iloc[-1]   # left to drift


def test_month_end_only_expansion_counts_the_cash_credit_of_a_partly_invested_book(monkeypatch):
    mod = module("rank_hysteresis_sweep")
    px, monthly = _two_asset_book()
    monthly = monthly * 0.6                                # 30% in each asset, 40% in T-bills
    rate = pd.Series(5.0, index=nyse_bdays("2020-12-01", "2021-06-30"))   # annual %, as ^IRX is quoted
    monkeypatch.setattr(bt, "_irx_series", lambda: rate)
    drifted = mod.expand_drift(monthly, px)
    # the engine's own expansion, cash credit included, row for row
    pd.testing.assert_frame_equal(drifted, bt.drift_weights(monthly, px))
    res = bt.run_backtest(drifted, px, IBKRHKCostModel(slippage_bps=3.0), name="book", withholding=0.0)
    assert res["cash_returns"].sum() > 0
    mod.check_no_intramonth_trades(res, monthly.index)


# ---- rank hysteresis: the study run ------------------------------------------

def _study(monkeypatch, tmp_path, seed=11):
    mod = module("rank_hysteresis_sweep")
    px = _world(mod, seed=seed)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_prices", lambda: px)
    return mod


def test_hysteresis_study_logs_every_variant_and_keeps_saved_records(monkeypatch, tmp_path, capsys):
    mod = _study(monkeypatch, tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    log, record = results / "rank_hysteresis_variants.csv", results / "rank_hysteresis.json"
    log.write_text("saved earlier\n")
    record.write_text('{"saved": "earlier"}')

    mod.main()
    shown = capsys.readouterr().out
    assert "reproduction check:" in shown and shown.count("kept existing record") == 2
    assert log.read_text() == "saved earlier\n" and record.read_text() == '{"saved": "earlier"}'

    frame = pd.read_csv(results / "recomputed" / log.name)
    expected = ([f"etf_B{b}_{m}" for b in mod.ETF_BUFFERS for m in mod.MODES]
                + [f"stock_B{b}_{m}" for b in mod.STK_BUFFERS for m in mod.MODES])
    assert list(frame["name"]) == expected and len(frame) == 33
    assert list(frame.columns)[:5] == ["sleeve", "k", "buffer", "mode", "name"]   # no index column
    assert np.isfinite(frame[["is_sharpe", "oos_sharpe", "is_sharpe_2x"]].to_numpy()).all()
    for (_, _), rows in frame.groupby(["sleeve", "buffer"]):
        by_mode = rows.set_index("mode")
        # the holdings are the same in every execution mode; only the orders differ
        assert rows["entries_per_year"].nunique() == 1 and rows["avg_hold_months"].nunique() == 1
        assert by_mode.loc["drift", "ann_turnover"] < by_mode.loc["retarget", "ann_turnover"]
        assert by_mode.loc["drift", "ann_cost_drag"] < by_mode.loc["retarget", "ann_cost_drag"]
        # doubled slippage can only cost more
        assert (rows["ann_cost_drag_2x"] > rows["ann_cost_drag"]).all()

    summary = json.loads((results / "recomputed" / record.name).read_text())
    assert set(summary) == {"etf", "stock"}
    for sleeve, k in (("etf", mod.etf.BEST_PARAMS["k"]), ("stock", mod.stk.K)):
        block = summary[sleeve]
        assert block["baseline_E0"]["name"] == f"{sleeve}_B{k}_retarget"
        rows = frame[frame["sleeve"] == sleeve]
        for mode in mod.MODES:
            in_mode = rows[rows["mode"] == mode]
            # the buffer is chosen on in-sample Sharpe alone; ties go to the larger buffer
            best = in_mode[in_mode["is_sharpe"] == in_mode["is_sharpe"].max()]["buffer"].max()
            assert block[f"chosen_{mode}"]["buffer"] == best
            assert set(block[f"delta_{mode}_vs_E0base"]) == {
                "ann_cost_drag", "ann_turnover", "gross_full_cagr", "full_cagr",
                "full_sharpe", "oos_sharpe"}


def test_hysteresis_study_writes_nothing_when_its_reproduction_check_fails(monkeypatch, tmp_path):
    mod = _study(monkeypatch, tmp_path)
    plain = mod.pick_holdings
    # a selection rule that no longer reduces to top-K at B = K
    monkeypatch.setattr(mod, "pick_holdings", lambda s, held, k, buffer: plain(s, held, k, buffer)[::-1][:k - 1])
    with pytest.raises(AssertionError):
        mod.main()
    assert not (tmp_path / "results").exists()
