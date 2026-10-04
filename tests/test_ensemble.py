"""Ensemble construction on synthetic sleeves: weights, capital passes, saved records."""
import ast
import importlib.util
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ("ensemble.json", "sleeve_returns.csv")


def load():
    path = ROOT / "src" / "ensemble.py"
    spec = importlib.util.spec_from_file_location("ensemble_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod(monkeypatch, tmp_path):
    """A fresh module instance rooted in an empty temporary project."""
    mod = load()
    (tmp_path / "results").mkdir()
    (tmp_path / "strategies").mkdir()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "SRC", tmp_path)
    monkeypatch.delenv("QCORE_REBASE", raising=False)
    return mod


def scripted_passes(mod, monkeypatch, passes):
    """Drive build() with one scripted returns frame per capital pass.
    Returns (scales each pass was called with, the dict handed to metrics)."""
    calls, seen = [], {}

    def collect(keys, scale=None):
        calls.append(scale)
        rets = passes[min(len(calls), len(passes)) - 1]
        mod._collected_results.clear()
        for k in rets:
            mod._collected_results[k] = {
                "returns": rets[k], "gross_returns": rets[k],
                "turnover": pd.Series(0.0, index=rets.index),
                "costs": pd.Series(0.0, index=rets.index)}
        return rets.copy(), pd.Series(0.0, index=rets.index)

    def capture_metrics(res):
        seen.update(res)
        return {k: {} for k in ["full", "in_sample", "out_of_sample",
                                "pct_positive_months", "worst_month"]}

    monkeypatch.setattr(mod, "_collect", collect)
    monkeypatch.setattr(mod, "metrics", capture_metrics)
    return calls, seen


def alternating(index, size):
    sign = np.where(np.arange(len(index)) % 2 == 0, 1.0, -1.0)
    return pd.Series(sign * size, index=index)


def flip_vol_sleeves(names=("a", "b")):
    """First sleeve quiet in sample and wild out of sample; second the reverse."""
    idx = pd.bdate_range("2017-01-02", "2018-12-31")
    in_sample = idx < pd.Timestamp("2018-01-01")
    frame = pd.DataFrame({
        names[0]: alternating(idx, 1.0) * np.where(in_sample, 0.005, 0.03),
        names[1]: alternating(idx, 1.0) * np.where(in_sample, 0.02, 0.002)})
    return frame, in_sample


def saved(tmp_path, name="ensemble.json"):
    return json.loads((tmp_path / "results" / name).read_text())


# ------------------------------------------------------------------ weights
def test_weights_use_in_sample_volatility_only_and_blend_the_second_pass(mod, monkeypatch, tmp_path):
    rets, in_sample = flip_vol_sleeves()
    second = rets - 0.0001                       # pass 2 pays more commission
    calls, seen = scripted_passes(mod, monkeypatch, [rets, second])
    mod.build(["a", "b"])
    weights = saved(tmp_path)["sleeve_weights"]
    inverse = 1.0 / rets[in_sample].std()
    assert weights == pytest.approx((inverse / inverse.sum()).to_dict())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] > 0.75                   # full-sample vol would give a < 0.5
    full = 1.0 / rets.std()
    assert abs(weights["a"] - (full / full.sum())["a"]) > 0.2
    # pass 1 at full capital, pass 2 at exactly the published weight share
    assert calls[0] is None
    assert calls[1] == pytest.approx(weights)
    # the blend and the saved sleeve file come from pass 2, not pass 1
    blend = second["a"] * weights["a"] + second["b"] * weights["b"]
    pd.testing.assert_series_equal(seen["returns"], blend, check_names=False)
    frame = pd.read_csv(tmp_path / "results" / "sleeve_returns.csv", index_col=0, parse_dates=True)
    assert frame["a"].to_numpy() == pytest.approx(second["a"].to_numpy())
    assert frame["ENSEMBLE"].to_numpy() == pytest.approx(blend.to_numpy())
    assert saved(tmp_path)["capital_accounting"]["ann_drag_vs_full_capital"] > 0


def test_split_day_is_out_of_sample(mod, monkeypatch, tmp_path):
    idx = pd.date_range("2017-12-18", "2018-01-14")          # 2018-01-01 is a row
    rets = pd.DataFrame({"a": alternating(idx, 0.01), "b": alternating(idx, 0.02)})
    rets.loc["2018-01-01", "a"] = 0.5                         # must not reach the weights
    scripted_passes(mod, monkeypatch, [rets])
    mod.build(["a", "b"])
    inverse = 1.0 / rets.loc[:"2017-12-31"].std()
    assert saved(tmp_path)["sleeve_weights"] == pytest.approx((inverse / inverse.sum()).to_dict())


def test_volatility_is_floored_at_two_percent(mod, monkeypatch, tmp_path):
    idx = pd.bdate_range("2016-01-04", periods=300)
    rets = pd.DataFrame({"quiet": alternating(idx, 1e-5), "loud": alternating(idx, 0.01)})
    scripted_passes(mod, monkeypatch, [rets])
    mod.build(["quiet", "loud"])
    weights = saved(tmp_path)["sleeve_weights"]
    loud_vol = float(rets["loud"].std() * np.sqrt(252))
    assert loud_vol > 0.02
    assert weights["quiet"] == pytest.approx((1 / 0.02) / (1 / 0.02 + 1 / loud_vol))
    assert weights["quiet"] < 0.9                # unfloored would be ~0.999


def test_weights_need_in_sample_history(mod, monkeypatch, tmp_path):
    idx = pd.bdate_range("2019-01-01", periods=80)
    rets = pd.DataFrame({"a": alternating(idx, 0.01), "b": alternating(idx, 0.02)})
    scripted_passes(mod, monkeypatch, [rets])
    with pytest.raises(ValueError, match="in-sample volatility"):
        mod.build(["a", "b"])
    assert not (tmp_path / "results" / "ensemble.json").exists()


@pytest.mark.parametrize("change", ["dates", "names"])
def test_sleeves_may_not_change_between_capital_passes(mod, monkeypatch, tmp_path, change):
    rets, _ = flip_vol_sleeves()
    second = rets.iloc[:-1] if change == "dates" else rets.rename(columns={"b": "c"})
    scripted_passes(mod, monkeypatch, [rets, second])
    with pytest.raises(ValueError, match="changed between capital passes"):
        mod.build(["a", "b"])
    assert not (tmp_path / "results" / "ensemble.json").exists()


def test_weights_are_estimated_on_the_cash_padded_frame(mod, monkeypatch, tmp_path):
    """The estimator in force: a late starter's pre-live T-bill rows sit
    inside its in-sample volatility window. Changing that moves the saved
    sleeve weights, so it must not happen by accident."""
    idx = pd.bdate_range("2014-01-01", "2018-12-31")
    rng = np.random.default_rng(7)
    rf = pd.Series(1e-4, index=idx)
    streams = {"a": pd.Series(rng.normal(0, 0.01, len(idx)), index=idx),
               "b": pd.Series(rng.normal(0, 0.01, len(idx)), index=idx).loc["2016-01-01":]}

    def sleeve(key):
        r = streams[key]
        zero = pd.Series(0.0, index=r.index)
        mod._last_sleeve_result = {"returns": r, "gross_returns": r, "turnover": zero, "costs": zero}
        return r

    monkeypatch.setattr(mod, "sleeve_returns", sleeve)
    monkeypatch.setattr(mod, "cash_daily_return", lambda i: rf.reindex(i))
    mod.build(["a", "b"])
    weights = saved(tmp_path)["sleeve_weights"]
    padded = pd.DataFrame(streams)
    padded["b"] = padded["b"].fillna(rf)
    inverse = 1.0 / padded.loc[:"2017-12-31"].std()
    assert weights == pytest.approx((inverse / inverse.sum()).to_dict())
    live = 1.0 / pd.Series({k: s.loc[:"2017-12-31"].std() for k, s in streams.items()})
    assert weights["b"] - (live / live.sum())["b"] > 0.02     # the padding is not a rounding detail


# ------------------------------------------------------------ capital passes
@pytest.mark.parametrize("how", ["positional", "keyword", "default", "none", "positional_none"])
def test_second_pass_scales_cost_model_capital(mod, monkeypatch, how):
    from qcore.costs import IBKRHKCostModel
    received = []

    def engine(weights, prices, cost_model=None, name="strategy", **kwargs):
        received.append(cost_model)
        return {"name": name}

    monkeypatch.setattr(mod, "_orig_run_backtest", engine)
    monkeypatch.setattr(mod, "_captured", [])
    supplied = IBKRHKCostModel(capital=40_000.0, slippage_bps=7.0)

    def call():
        if how == "positional":
            return mod._capturing_run_backtest("w", "px", supplied, name="x")
        if how == "keyword":
            return mod._capturing_run_backtest("w", "px", cost_model=supplied, name="x")
        if how == "none":
            return mod._capturing_run_backtest("w", "px", cost_model=None, name="x")
        if how == "positional_none":
            return mod._capturing_run_backtest("w", "px", None, name="x")
        return mod._capturing_run_backtest("w", "px", name="x")

    base = 40_000.0 if how in ("positional", "keyword") else IBKRHKCostModel().capital
    monkeypatch.setattr(mod, "_capital_scale", 0.25)
    assert call() == {"name": "x"}
    assert received[-1].capital == pytest.approx(0.25 * base)
    if how in ("positional", "keyword"):
        assert received[-1].slippage_bps == 7.0  # every other cost field survives
        assert supplied.capital == 40_000.0      # the caller's model is not mutated
    monkeypatch.setattr(mod, "_capital_scale", 1.0)
    call()
    assert received[-1] is (supplied if how in ("positional", "keyword") else None)
    assert len(mod._captured) == 2


def test_pre_live_gap_earns_treasury_bills_and_capital_share_is_scoped(mod, monkeypatch):
    idx = pd.bdate_range("2010-01-04", periods=5)
    rf = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=idx)
    monkeypatch.setattr(mod, "cash_daily_return", lambda i: rf.reindex(i))
    returns = {"early": pd.Series(0.01, index=idx), "late": pd.Series(-0.02, index=idx[2:])}
    scales = []

    def sleeve(key):
        scales.append(mod._capital_scale)
        if key == "boom":
            raise RuntimeError("sleeve failed")
        return returns[key]

    monkeypatch.setattr(mod, "sleeve_returns", sleeve)
    rets, cash = mod._collect(["early", "late"], scale={"early": 0.7, "late": 0.3})
    assert rets["late"].tolist() == pytest.approx([0.001, 0.002, -0.02, -0.02, -0.02])
    assert rets["early"].tolist() == pytest.approx([0.01] * 5)
    pd.testing.assert_series_equal(cash, rf)
    assert scales == [0.7, 0.3]
    assert mod._capital_scale == 1.0             # back to full capital afterwards
    with pytest.raises(RuntimeError, match="sleeve failed"):
        mod._collect(["boom"], scale={"boom": 0.4})
    assert mod._capital_scale == 1.0             # ... also when a sleeve raises
    for bad in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="invalid sleeve capital scale"):
            mod._collect(["early", "late"], scale={"early": bad, "late": 0.3})
        assert mod._capital_scale == 1.0         # a rejected share is never left in force
    for keys in ([], ["early", "early"]):
        with pytest.raises(ValueError, match="nonempty and unique"):
            mod._collect(keys)


def test_results_marked_as_study_are_refused(mod, monkeypatch, tmp_path):
    (tmp_path / "strategies" / "sample.py").touch()
    (tmp_path / "results" / "sample.json").write_text(json.dumps({"study": True, "name": "frozen"}))
    ran = []
    monkeypatch.setattr(mod.runpy, "run_path", lambda *a, **k: ran.append(a))
    with pytest.raises(ValueError, match="marked 'study'"):
        mod.sleeve_returns("sample")
    assert ran == []                             # rejected before the module is executed
    with pytest.raises(ValueError, match="no strategy module"):
        mod.sleeve_returns("absent")


# -------------------------------------------------------------- saved records
def default_passes(mod):
    idx = pd.bdate_range("2017-01-02", "2018-12-31")
    return pd.DataFrame({key: alternating(idx, 0.004 * (n + 1))
                         for n, key in enumerate(mod.DEFAULT_SLEEVES)})


def seed_production(tmp_path, text="RETAINED"):
    for name in PRODUCTION:
        (tmp_path / "results" / name).write_text(text)


def listing(tmp_path):
    return sorted(str(p.relative_to(tmp_path / "results"))
                  for p in (tmp_path / "results").rglob("*") if p.is_file())


def test_default_run_keeps_a_differing_record_and_parks_its_output(mod, monkeypatch, tmp_path):
    rets = default_passes(mod)
    scripted_passes(mod, monkeypatch, [rets])
    seed_production(tmp_path)
    mod.main([])
    for name in PRODUCTION:
        assert (tmp_path / "results" / name).read_text() == "RETAINED"
    assert listing(tmp_path) == sorted(list(PRODUCTION) + [f"recomputed/{n}" for n in PRODUCTION])
    parked = saved(tmp_path, "recomputed/ensemble.json")
    assert list(parked["sleeve_weights"]) == mod.DEFAULT_SLEEVES


def test_rebase_replaces_the_record_in_the_original_format(mod, monkeypatch, tmp_path):
    rets = default_passes(mod)
    calls, seen = scripted_passes(mod, monkeypatch, [rets])
    seed_production(tmp_path)
    mod.main(["--rebase"])
    assert listing(tmp_path) == sorted(PRODUCTION)
    text = (tmp_path / "results" / "ensemble.json").read_text()
    assert text == json.dumps(json.loads(text), indent=2)
    frame = rets.assign(ENSEMBLE=seen["returns"])
    assert (tmp_path / "results" / "sleeve_returns.csv").read_text() == frame.to_csv()
    # an identical rerun is not a difference: nothing is parked
    mod.main([])
    assert listing(tmp_path) == sorted(PRODUCTION)


def test_environment_rebase_is_honoured_for_the_default_blend(mod, monkeypatch, tmp_path):
    scripted_passes(mod, monkeypatch, [default_passes(mod)])
    seed_production(tmp_path)
    monkeypatch.setenv("QCORE_REBASE", "1")
    mod.main([])
    assert listing(tmp_path) == sorted(PRODUCTION)
    assert list(saved(tmp_path)["sleeve_weights"]) == mod.DEFAULT_SLEEVES


def test_record_names_are_production_only_for_the_model(mod, tmp_path):
    results = tmp_path / "results"
    assert mod.record_paths(mod.DEFAULT_SLEEVES) == (results / "ensemble.json",
                                                     results / "sleeve_returns.csv")
    for keys in (["a", "b"], mod.DEFAULT_SLEEVES[:3], mod.DEFAULT_SLEEVES[::-1],
                 mod.DEFAULT_SLEEVES + ["extra"]):
        paths = mod.record_paths(keys)
        assert all(p.parent == results / "adhoc" for p in paths), keys
        assert not {p.name for p in paths} & set(PRODUCTION)
    assert mod.record_paths(["a", "b"])[0].name == "ensemble_a+b.json"
    assert mod.record_paths(["a", "b"])[1].name == "sleeve_returns_a+b.csv"


def test_build_refuses_to_shadow_the_production_record_with_another_blend(mod, monkeypatch, tmp_path):
    rets, _ = flip_vol_sleeves()
    calls, _ = scripted_passes(mod, monkeypatch, [rets])
    seed_production(tmp_path)
    for rebase in (None, True):
        with pytest.raises(ValueError, match="production blend"):
            mod.build(["a", "b"], rebase=rebase)
    assert calls == []                           # refused before any sleeve is run
    assert listing(tmp_path) == sorted(PRODUCTION)
    mod.build(["a", "b"], paths=mod.record_paths(["a", "b"]))
    assert (tmp_path / "results" / "adhoc" / "ensemble_a+b.json").exists()
    assert (tmp_path / "results" / "ensemble.json").read_text() == "RETAINED"


def test_command_line_checks_keys_and_flags(mod, monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(mod, "build", lambda keys, paths=None, rebase=None:
                        built.append((list(keys), paths, rebase)))
    results = tmp_path / "results"
    mod.main([])
    mod.main(["--rebase"])
    assert built == [
        (mod.DEFAULT_SLEEVES, (results / "ensemble.json", results / "sleeve_returns.csv"), None),
        (mod.DEFAULT_SLEEVES, (results / "ensemble.json", results / "sleeve_returns.csv"), True)]
    with pytest.raises(SystemExit, match="unknown sleeve key"):
        mod.main(["typo"])
    with pytest.raises(SystemExit):
        mod.main(["--no-such-flag"])
    assert len(built) == 2


FAKE_SLEEVE = '''
    import json
    from pathlib import Path

    import qcore.backtest as bt
    from qcore.records import save_json

    ROOT = Path(__file__).resolve().parents[1]

    if __name__ == "__main__":
        bt.run_backtest("weights", "prices", name="{key}_best")
        # a sleeve that publishes its own record: directly, and as a guarded record
        (ROOT / "results" / "{key}.json").write_text(json.dumps({{"name": "{key}_best", "direct": True}}))
        save_json(ROOT / "results" / "{key}.json", {{"name": "{key}_best", "guarded": True}})
        save_json(ROOT / "results" / "{key}_variants.json", {{"rows": 1}})
'''


def fake_sleeves(mod, monkeypatch, tmp_path, keys):
    """Strategy modules that run a stub engine and then try to save records."""
    rets, _ = flip_vol_sleeves(keys)
    seen = []

    def engine(weights, prices, cost_model=None, name="strategy", **kwargs):
        seen.append((name, None if cost_model is None else cost_model.capital))
        r = rets[name.removesuffix("_best")]
        zero = pd.Series(0.0, index=r.index)
        return {"name": name, "returns": r, "gross_returns": r, "turnover": zero, "costs": zero}

    monkeypatch.setattr(mod, "_orig_run_backtest", engine)
    monkeypatch.setattr(mod, "cash_daily_return", lambda i: pd.Series(0.0, index=i))
    for key in keys:
        (tmp_path / "strategies" / f"{key}.py").write_text(textwrap.dedent(FAKE_SLEEVE.format(key=key)))
        (tmp_path / "results" / f"{key}.json").write_text(json.dumps({"name": f"{key}_best"}))
    return seen


@pytest.mark.parametrize("rebase_env", [False, True])
def test_blending_never_rewrites_a_sleeve_record_or_the_production_files(
        mod, monkeypatch, tmp_path, rebase_env):
    from qcore import records
    from qcore.costs import IBKRHKCostModel
    import qcore.backtest as bt
    seen = fake_sleeves(mod, monkeypatch, tmp_path, ("s", "t"))
    seed_production(tmp_path)
    before = {p: (tmp_path / "results" / p).read_bytes() for p in listing(tmp_path)}
    engine_before, save_before = bt.run_backtest, records.save_record
    if rebase_env:
        monkeypatch.setenv("QCORE_REBASE", "1")
    mod.main(["s", "t"] + (["--rebase"] if rebase_env else []))
    blend = ["adhoc/ensemble_s+t.json", "adhoc/sleeve_returns_s+t.csv"]
    assert listing(tmp_path) == sorted(list(before) + blend)
    for name, content in before.items():
        assert (tmp_path / "results" / name).read_bytes() == content, name
    assert bt.run_backtest is engine_before and records.save_record is save_before
    weights = saved(tmp_path, blend[0])["sleeve_weights"]
    assert list(weights) == ["s", "t"]
    header = (tmp_path / "results" / blend[1]).read_text().splitlines()[0]
    assert header == ",s,t,ENSEMBLE"
    # pass 1 at full capital (engine default), pass 2 at each sleeve's share
    full = IBKRHKCostModel().capital
    assert seen[:2] == [("s_best", None), ("t_best", None)]
    assert seen[2:] == [("s_best", pytest.approx(weights["s"] * full)),
                        ("t_best", pytest.approx(weights["t"] * full))]


def test_a_sleeve_that_deletes_its_record_gets_it_back(mod, monkeypatch, tmp_path):
    (tmp_path / "strategies" / "s.py").touch()
    record = tmp_path / "results" / "s.json"
    record.write_text(json.dumps({"name": "frozen"}))
    kept = record.read_bytes()
    idx = pd.date_range("2020", periods=3)

    def run(*args, **kwargs):
        mod._captured.append({"name": "frozen", "returns": pd.Series(0.01, index=idx)})
        record.unlink()

    monkeypatch.setattr(mod.runpy, "run_path", run)
    assert mod.sleeve_returns("s").tolist() == [0.01] * 3
    assert record.read_bytes() == kept


def test_a_declared_name_the_module_does_not_produce_is_explained(mod, monkeypatch, tmp_path):
    """A stored record can carry a run name its module has since changed. The
    record is kept as it is, so the message must not suggest rewriting it."""
    (tmp_path / "strategies" / "s.py").touch()
    record = tmp_path / "results" / "s.json"
    record.write_text(json.dumps({"metrics": {"name": "K5_lam0.25"}}))
    kept = record.read_bytes()
    idx = pd.date_range("2020", periods=3)
    runs = [{"name": "s_K5_lam0.25", "returns": pd.Series(0.01, index=idx)}]
    monkeypatch.setattr(mod.runpy, "run_path", lambda *a, **k: mod._captured.extend(runs))
    with pytest.raises(ValueError) as stop:
        mod.sleeve_returns("s")
    message = str(stop.value)
    assert message.startswith("s: expected one captured run named 'K5_lam0.25'; "
                              "found 0 (the module ran: s_K5_lam0.25). ")
    assert "results/s.json declares a name the module does not produce" in message
    assert "preserved record" in message and "not edited or re-based" in message
    assert "separate copy of the project" in message
    assert "Refresh" not in message              # the stored record is not to be refreshed
    assert record.read_bytes() == kept
    # two runs under the declared name: the module is at fault, not the record
    runs[:] = [{"name": "K5_lam0.25", "returns": pd.Series(0.01, index=idx)}] * 2
    with pytest.raises(ValueError, match="found 2") as stop:
        mod.sleeve_returns("s")
    assert "share a name" in str(stop.value) and "preserved record" not in str(stop.value)
    # a long list of runs is shortened, never the declared name
    runs[:] = [{"name": f"run{n:02d}", "returns": pd.Series(0.01, index=idx)} for n in range(12)]
    with pytest.raises(ValueError, match=r"the module ran: run00, .*run07, \.\.\.\)") as stop:
        mod.sleeve_returns("s")
    assert "run08" not in str(stop.value)


def test_missing_data_cache_ends_the_command_with_one_line(mod, tmp_path):
    from qcore import records
    import qcore.backtest as bt
    message = "data/adj_close.csv not found - build the local cache first: python src/download_data.py"
    (tmp_path / "strategies" / "s.py").write_text(f"raise FileNotFoundError({message!r})\n")
    (tmp_path / "results" / "s.json").write_text(json.dumps({"name": "s_best"}))
    engine, save = bt.run_backtest, records.save_record
    with pytest.raises(SystemExit) as stop:
        mod.main(["s"])
    assert stop.value.code == message            # the message alone: exit status 1, no traceback
    assert bt.run_backtest is engine and records.save_record is save
    assert listing(tmp_path) == ["s.json"]
    # any other failure of a sleeve is not turned into a plain exit
    (tmp_path / "strategies" / "s.py").write_text("raise RuntimeError('sleeve failed')\n")
    with pytest.raises(RuntimeError, match="sleeve failed"):
        mod.main(["s"])


def test_missing_sleeve_record_ends_the_command_with_one_line_naming_the_command(mod, tmp_path):
    """A checkout that has not saved the sleeve records yet: the default run
    says which file is missing and how to write it, and runs nothing."""
    ran = tmp_path / "ran"
    for key in mod.DEFAULT_SLEEVES + ["s"]:
        (tmp_path / "strategies" / f"{key}.py").write_text(f"open({str(ran)!r}, 'w').close()\n")
    with pytest.raises(SystemExit) as stop:
        mod.main([])
    message = stop.value.code
    assert isinstance(message, str) and len(message.splitlines()) == 1
    first = mod.DEFAULT_SLEEVES[0]
    assert message.startswith(f"results/{first}.json not found: it declares which run")
    assert message.endswith(f"python src/strategies/{first}.py > results/{first}.json")
    assert "likewise for every other sleeve" in message
    assert not ran.exists() and listing(tmp_path) == []
    # a named key without a record is not called "unknown", and a sleeve that
    # is not one of the defaults gets no command it may not support
    with pytest.raises(SystemExit) as stop:
        mod.main(["s"])
    assert stop.value.code == ("results/s.json not found: it declares which run of "
                               "src/strategies/s.py the blend uses")
    with pytest.raises(SystemExit, match="unknown sleeve key\\(s\\): typo"):
        mod.main(["s", "typo"])
    assert not ran.exists() and listing(tmp_path) == []
    # the same line reaches a caller that uses the function directly
    with pytest.raises(FileNotFoundError, match="results/s.json not found"):
        mod.sleeve_returns("s")


def test_documented_record_command_matches_what_the_default_sleeves_print():
    """The missing-record line tells the reader to redirect a sleeve module's
    output into results/<key>.json. That only works while each default sleeve
    prints one JSON document carrying the run name the blend looks up."""
    mod = load()
    for key in mod.DEFAULT_SLEEVES:
        tree = ast.parse((ROOT / "src" / "strategies" / f"{key}.py").read_text())
        dumps = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "print" and n.args
                 and isinstance(n.args[0], ast.Call)
                 and ast.unparse(n.args[0].func) == "json.dumps"]
        assert dumps, f"{key}: the module no longer prints its record as JSON"


def test_an_empty_or_partial_sleeve_record_is_named_not_a_json_traceback(mod, monkeypatch, tmp_path):
    (tmp_path / "strategies" / "s.py").touch()
    record = tmp_path / "results" / "s.json"
    ran = []
    monkeypatch.setattr(mod.runpy, "run_path", lambda *a, **k: ran.append(a))
    for content in ("", '{"name": "s_best"'):       # an interrupted redirect
        record.write_text(content)
        with pytest.raises(ValueError, match=r"results/s\.json is not valid JSON") as stop:
            mod.sleeve_returns("s")
        assert stop.value.__cause__ is None and ran == []
        assert record.read_text() == content         # left for the reader to inspect


# ------------------------------------------------------------ blend convention
def test_held_blend_matches_constant_mix_and_buy_and_hold(mod):
    idx = pd.bdate_range("2020-01-01", periods=3)
    rets = pd.DataFrame({"a": [0.10, -0.05, 0.02], "b": [-0.02, 0.04, 0.01]}, index=idx)
    w = pd.Series({"a": 0.25, "b": 0.75})
    daily, moved = mod.held_blend(rets, w, [True] * 3)
    assert daily.to_numpy() == pytest.approx((rets * w).sum(axis=1).to_numpy())
    assert moved > 0
    held, none = mod.held_blend(rets, w, [False] * 3)
    wealth = ((1 + rets).cumprod() * w).sum(axis=1)
    assert (1 + held).cumprod().to_numpy() == pytest.approx(wealth.to_numpy())
    assert none == 0.0
    # by hand: day 1 grows the shares to .275 / .735 of 1.01, then holds them
    assert held.iloc[1] == pytest.approx((0.275 * -0.05 + 0.735 * 0.04) / 1.01)
    first, turnover = mod.held_blend(rets.iloc[:1], w, [True])
    assert turnover == pytest.approx(abs(0.25 - 0.275 / 1.01))
    for bad_w, reset in [(pd.Series({"a": 0.5, "b": 0.6}), [True] * 3), (w, [True] * 2),
                         (pd.Series({"a": 1.5, "b": -0.5}), [True] * 3)]:
        with pytest.raises(ValueError, match="held_blend needs"):
            mod.held_blend(rets, bad_w, reset)
    with pytest.raises(ValueError, match="held_blend needs"):
        mod.held_blend(rets.assign(a=[0.1, np.nan, 0.0]), w, [True] * 3)


def test_rebalance_sensitivity_daily_row_is_the_headline_blend(mod):
    from qcore.backtest import _stats
    idx = pd.bdate_range("2017-01-02", "2018-12-31")
    rng = np.random.default_rng(11)
    rets = pd.DataFrame({"a": rng.normal(4e-4, 0.01, len(idx)),
                         "b": rng.normal(2e-4, 0.004, len(idx))}, index=idx)
    w = pd.Series({"a": 0.3, "b": 0.7})
    rf = pd.Series(1e-5, index=idx)
    table = mod.rebalance_sensitivity(rets, w, rf)
    assert list(table) == ["daily", "monthly", "never"]
    combo = (rets * w).sum(axis=1)
    split = idx >= pd.Timestamp("2018-01-01")
    assert table["daily"]["full_sharpe"] == _stats(combo, rf)["sharpe"]
    assert table["daily"]["is_sharpe"] == _stats(combo[~split], rf)["sharpe"]
    assert table["daily"]["oos_sharpe"] == _stats(combo[split], rf)["sharpe"]
    assert table["daily"]["full_cagr"] == _stats(combo, rf)["cagr"]
    turnover = {k: v["ann_intersleeve_turnover_oneside"] for k, v in table.items()}
    assert turnover["never"] == 0.0
    assert turnover["daily"] > turnover["monthly"] > 0
    # monthly resets fall on the last session of each calendar month only
    month = idx.to_period("M")
    resets = np.append(month[1:] != month[:-1], True)
    held, _ = mod.held_blend(rets, w, resets)
    assert table["monthly"]["full_sharpe"] == _stats(held, rf)["sharpe"]
    assert resets.sum() == 24 and all(idx[resets] == pd.Series(idx).groupby(month).max().to_numpy())


def test_saved_metrics_state_the_blend_convention(mod, monkeypatch, tmp_path):
    rets, _ = flip_vol_sleeves()
    scripted_passes(mod, monkeypatch, [rets.iloc[:40]])        # too short for any Sharpe
    mod.build(["a", "b"])
    text = (tmp_path / "results" / "ensemble.json").read_text()
    out = json.loads(text)
    assert out["blend_convention"].startswith("constant-mix")
    assert set(out["rebalance_sensitivity"]) == {"daily", "monthly", "never"}
    assert out["rebalance_sensitivity"]["daily"]["full_sharpe"] is None
    assert "NaN" not in text
