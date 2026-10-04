"""Selection sweeps of the pairs and stock-momentum examples.

scripts/research_pairs_statarb.py and scripts/xsec_stock_mom_sweep.py are the
grids behind src/strategies/pairs_statarb.py and xsec_stock_mom.py. Everything
runs on synthetic data with ROOT redirected to tmp_path: no test reads the
data cache or touches results/.
"""
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore.calendar import confirmed_month_ends
from qcore.costs import IBKRHKCostModel

from test_strategy_rules_nonprod import _direct_writes, _stock_panel

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("scripts/research_pairs_statarb.py", "scripts/xsec_stock_mom_sweep.py")


def module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_nonprod_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())
    monkeypatch.delenv("QCORE_REBASE", raising=False)


def _engine(w, px, cm=None, name="strategy"):
    return bt.run_backtest(w, px, cm, name=name, cash_rate=0.0, withholding=0.0)


# ---- command line and writes ----------------------------------------------

@pytest.mark.parametrize("relative", SCRIPTS)
@pytest.mark.parametrize("argv,code", [(["--help"], 0), (["--rebse"], 2), (["extra"], 2)])
def test_sweep_scripts_reject_unknown_arguments_before_any_backtest(monkeypatch, capsys, relative, argv, code):
    import qcore.data

    def refuse(*args, **kwargs):
        raise AssertionError("a backtest was started")

    monkeypatch.setattr(qcore.data, "load_prices", refuse)
    monkeypatch.setattr(sys, "path", list(sys.path))
    script = ROOT / relative
    monkeypatch.setattr(sys, "argv", [str(script)] + argv)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(script), run_name="__main__")
    assert stop.value.code == code
    shown = capsys.readouterr()
    assert "usage:" in (shown.out if code == 0 else shown.err)


@pytest.mark.parametrize("relative", SCRIPTS)
def test_sweep_scripts_save_results_only_through_the_record_helpers(relative):
    assert _direct_writes(ROOT / relative) == []


# ---- pairs sweep: one builder, guarded variants log ------------------------

def test_pairs_sweep_uses_the_strategy_builder_and_borrow():
    from strategies import pairs_statarb as ps
    sweep = module("scripts/research_pairs_statarb.py")
    assert sweep.pair_weights is ps.pair_weights
    assert sweep.apply_borrow is ps.apply_borrow
    assert sweep.PAIRS is ps.PAIRS and sweep.SLIPPAGE_BPS == ps.SLIPPAGE_BPS
    # The script no longer carries rule constants of its own to drift.
    for stale in ("ENTRY", "EXIT_Z", "TIMEOUT", "BORROW_RATE", "OOS_SPLIT"):
        assert not hasattr(sweep, stale)


def test_pairs_sweep_never_replaces_a_variants_log_that_differs(monkeypatch, tmp_path, capsys):
    import qcore.data
    sweep = module("scripts/research_pairs_statarb.py")
    idx = pd.bdate_range("2024-01-02", periods=5)
    px = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    level = {"sharpe": 0.5}
    seen = []

    def weights(p, a, b, H, Z, gross=1.0):
        seen.append((H, Z))
        return pd.DataFrame(0.0, index=p.index, columns=[a, b])

    block = lambda: {"sharpe": level["sharpe"], "cagr": 0.01, "vol": 0.05, "maxdd": -0.02}  # noqa: E731
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    monkeypatch.setattr(sweep, "PAIRS", [("A", "B")])
    monkeypatch.setattr(qcore.data, "load_prices", lambda: px)
    monkeypatch.setattr(sweep, "pair_weights", weights)
    monkeypatch.setattr(sweep, "run_backtest",
                        lambda w, p, cm, name: {"name": name, "returns": pd.Series(0.0, index=p.index)})
    monkeypatch.setattr(sweep, "metrics", lambda res: {
        "name": res["name"], "full": block(), "in_sample": block(), "out_of_sample": block(),
        "ann_turnover_oneside": 1.0, "ann_cost_drag": 0.001})
    monkeypatch.setattr(sys, "argv", ["research_pairs_statarb.py"])
    log = tmp_path / "results" / "pairs_statarb_variants.csv"

    sweep.main()
    capsys.readouterr()
    assert sorted(set(seen)) == [(h, z) for h in (60, 90, 120) for z in (20, 40, 60)]
    first = log.read_text()
    frame = pd.read_csv(log)
    assert list(frame["variant"]) == [f"H{h}_Z{z}" for h in (60, 90, 120) for z in (20, 40, 60)]
    assert list(frame.columns)[:4] == ["variant", "H", "Z", "n_pass"]  # no index column

    level["sharpe"] = 0.9
    sweep.main()
    assert "kept existing record" in capsys.readouterr().out
    assert log.read_text() == first
    assert pd.read_csv(log.parent / "recomputed" / log.name)["is_sharpe"].eq(0.9).all()

    monkeypatch.setattr(sys, "argv", ["research_pairs_statarb.py", "--rebase"])
    sweep.main()
    assert pd.read_csv(log)["is_sharpe"].eq(0.9).all()


# ---- stock-momentum sweep: benchmark, run names, guarded records -----------

def _engine_stubs(monkeypatch, mod):
    monkeypatch.setattr(mod, "run_backtest", _engine)
    monkeypatch.setattr(mod, "drift_weights", lambda targets, px: bt.drift_weights(targets, px, cash_rate=0.0))


def test_stock_sweep_benchmark_orders_only_at_month_ends_over_the_strategy_window(monkeypatch):
    mod = module("scripts/xsec_stock_mom_sweep.py")
    _engine_stubs(monkeypatch, mod)
    px = _stock_panel(late=3).iloc[:, :6]  # three names list six weeks late
    month_ends = confirmed_month_ends(px.index)
    cm = IBKRHKCostModel(slippage_bps=5.0)

    start = month_ends[3]
    res = mod.ew_benchmark(px, cm, start=start)
    assert res["name"] == "EW_universe"
    assert res["returns"].index[0] == start
    rebalance = res["turnover"].index.isin(month_ends)
    # Drift between month-ends is left alone: no orders, no costs.
    assert res["turnover"][~rebalance].abs().max() < 1e-12
    assert res["costs"][~rebalance].eq(0.0).all()
    assert res["turnover"].loc[start] == pytest.approx(1.0)
    assert (res["turnover"][rebalance].iloc[1:] > 0).all()
    # The previous construction re-targeted the same weights every day and
    # paid for it on every session.
    valid = px.notna()
    daily = valid.div(valid.sum(axis=1), axis=0).loc[month_ends].loc[start:].reindex(px.index).ffill().fillna(0.0)
    old = _engine(daily, px, cm, name="daily")
    assert old["costs"].index.equals(res["costs"].index)
    assert (old["costs"][~rebalance] > 0).mean() > 0.9
    assert res["costs"].sum() < old["costs"].sum()

    # Without a start the basket begins at the first month-end and buys the
    # late names when they list.
    whole = mod.ew_benchmark(px, cm)
    assert whole["returns"].index[0] == month_ends[0]
    with pytest.raises(ValueError, match="no month-end decision"):
        mod.ew_benchmark(px, cm, start=px.index[-1] + pd.Timedelta(1, unit="D"))


def test_stock_sweep_active_stats_use_the_common_window_only():
    mod = module("scripts/xsec_stock_mom_sweep.py")
    idx = pd.bdate_range("2021-01-04", periods=504)
    swing = np.where(np.arange(len(idx)) % 2 == 0, 1.0, -1.0)
    benchmark = pd.Series(0.0002 + 0.004 * swing, index=idx)
    benchmark.iloc[:252] += 0.01  # before the strategy starts: must not count
    strategy = (pd.Series(0.0002 + 0.004 * swing, index=idx) + 0.0004 + 0.001 * swing).iloc[252:]
    stats = mod.active_stats(strategy, benchmark)
    assert stats["start"] == str(idx[252].date()) and stats["end"] == str(idx[-1].date())
    s, b = strategy, benchmark.iloc[252:]
    gap = (1 + s).prod() - (1 + b).prod()  # one year on the common window
    assert stats["cagr_gap"] == round(float(gap), 4) > 0
    diff = s - b
    assert stats["active_sharpe"] == round(float(diff.mean() / diff.std() * 252 ** 0.5), 2) > 0
    with pytest.raises(ValueError, match="fewer than two dates"):
        mod.active_stats(strategy, benchmark.iloc[:252])


def test_stock_sweep_names_runs_like_the_module_and_keeps_saved_records(monkeypatch, tmp_path, capsys):
    from strategies import xsec_stock_mom as xs
    mod = module("scripts/xsec_stock_mom_sweep.py")
    assert mod.run_name is xs.run_name and mod.build_weights is xs.build_weights
    _engine_stubs(monkeypatch, mod)
    px = _stock_panel()
    level = {"bonus": 0.0}

    def metrics(res):
        best = res["name"].endswith("K5_lam0.25")
        block = {"sharpe": (1.0 if best else 0.5) + level["bonus"], "cagr": 0.1, "vol": 0.2, "maxdd": -0.3}
        return {"name": res["name"], "start": str(res["returns"].index[0].date()),
                "full": dict(block), "in_sample": dict(block), "out_of_sample": dict(block),
                "ann_turnover_oneside": 3.0, "ann_cost_drag": 0.01,
                "pct_positive_months": 0.6, "worst_month": -0.1}

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "STOCK_UNIVERSE", list(px.columns))
    monkeypatch.setattr(mod, "load_prices", lambda: px)
    monkeypatch.setattr(mod, "metrics", metrics)
    monkeypatch.setattr(sys, "argv", ["xsec_stock_mom_sweep.py"])
    log = tmp_path / "results" / "xsec_stock_mom_variants.csv"
    record = tmp_path / "results" / "xsec_stock_mom.json"

    mod.main()
    shown = capsys.readouterr().out
    frame = pd.read_csv(log, dtype={"name": str})
    # The log keeps its short labels; the run itself carries the module's name,
    # so the declared name is one src/ensemble.py can match to the module's run.
    assert list(frame["name"]) == [f"K{k}_lam{lam:g}" for k in (5, 8, 10, 15) for lam in (0.0, 0.25, 0.5)]
    assert list(frame.columns)[:3] == ["name", "K", "lam"]
    saved = json.loads(record.read_text())
    assert saved["params"] == {"K": 5, "lam": 0.25}
    assert saved["metrics"]["name"] == xs.run_name() == "xsec_stock_mom_K5_lam0.25"
    assert record.read_text() == json.dumps(saved, indent=2)
    assert "BEST BY IS SHARPE: K5_lam0.25" in shown
    assert "month-end orders only, from 2022-01-31" in shown
    assert "K5_lam0.25 vs benchmark, 2022-01-31..2022-02-28: CAGR gap" in shown

    first_log, first_record = log.read_text(), record.read_text()
    level["bonus"] = 0.2
    mod.main()
    assert capsys.readouterr().out.count("kept existing record") == 2
    assert log.read_text() == first_log and record.read_text() == first_record
    assert pd.read_csv(log.parent / "recomputed" / log.name)["is_sharpe"].max() == 1.2
    assert json.loads((log.parent / "recomputed" / record.name).read_text())["metrics"]["full"]["sharpe"] == 1.2

    monkeypatch.setattr(sys, "argv", ["xsec_stock_mom_sweep.py", "--rebase"])
    mod.main()
    assert json.loads(record.read_text())["metrics"]["full"]["sharpe"] == 1.2


def test_stock_sweep_states_the_fill_and_the_universe_has_fifty_names():
    from qcore.data import STOCK_UNIVERSE
    assert len(STOCK_UNIVERSE) == len(set(STOCK_UNIVERSE)) == 50
    doc = module("scripts/xsec_stock_mom_sweep.py").__doc__
    assert "the engine fills at that close" in doc and "trade next close" not in doc
