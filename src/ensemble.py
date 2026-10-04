"""Portfolio construction: combine the strategy sleeves into one portfolio.

Each module in src/strategies/ runs its retained rule when executed. This
script executes every sleeve module with qcore.backtest.run_backtest patched
to capture the result of each backtest the module runs, keeps the one run
whose name equals the name declared in results/<key>.json, weights the
sleeves by inverse volatility estimated on in-sample data only (before
2018), and reports the blended portfolio. A run is identified by its
declared name and is never chosen by its realized performance.

Capital accounting: the sleeves are run twice. Pass 1, at the cost model's
full capital, fixes the in-sample inverse-volatility weights. Pass 2 re-runs
each sleeve with capital scaled to its weight share: a 35% sleeve trades
about $35k, where the broker's $1 minimum commission weighs more than it
does at $100k. The blend and every reported number use pass 2.

Volatility window: the in-sample volatility behind the weights is taken on
the blend's own frame, where a sleeve's days before it starts trading are
T-bill days. Those rows carry almost no variance, so a sleeve that starts
later is measured a little calmer than on its own span. Volatility is
floored at 2% a year before it is inverted.

Blend convention: the headline is a constant-mix blend, the same weights
applied to every day's sleeve returns. That equals resetting each sleeve's
capital share at every close, and the re-split between sleeves is not
costed. The output's rebalance_sensitivity block re-blends the same streams
with the split reset at month-ends only, and never, to show how much the
headline leans on that assumption.

Inputs
  - the price cache built by python src/download_data.py;
  - results/<key>.json for every sleeve: the record that declares which run
    of the sleeve module is blended. Each of the four default sleeve modules
    prints that record as JSON, so it is created with
        mkdir -p results
        python src/strategies/mean_reversion.py > results/mean_reversion.json
    and likewise for seasonality_flows, tsmom_trend and xsec_etf_mom.

Usage: python src/ensemble.py [key ...] [--rebase]
  With no keys the four default sleeves are blended. Any other blend needs
  its keys named, and a record marked "study" is refused.

Outputs: for the default blend, results/ensemble.json (metrics, sleeve
weights and correlations, the drag from sleeve-share capital, the
rebalance-sensitivity table) and results/sleeve_returns.csv (daily net
returns of each sleeve and of the blend, column ENSEMBLE). Any other blend
is filed under results/adhoc/. An existing record that differs from this
run is kept and the new output goes to a recomputed/ folder beside it;
replacing a record takes --rebase (qcore.records). A sleeve executed for
the blend never saves its own record: its results/<key>.json is left as it
was.

Exit status: 0 when the blend was built; 1, with one line and no traceback,
when the price cache or a sleeve record is missing or a key is unknown (the
line names the file and the command that creates it); 2 on a usage error.
Any other failure, such as a declared name the module does not produce or
invalid sleeve returns, raises with a traceback.

Limits: the weights are fixed for the whole history from in-sample data and
are never re-estimated; the blend inherits every limit of the sleeve
backtests (same-close execution, modeled costs, and a 2018 split that later
research choices have consulted); moving capital between sleeves is free in
the headline figures.
"""

import argparse
import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import qcore.backtest as bt
from qcore import records
from qcore.backtest import OOS_SPLIT, TRADING_DAYS, _stats, cash_daily_return, metrics
from qcore.costs import IBKRHKCostModel

DEFAULT_SLEEVES = ["mean_reversion", "seasonality_flows",
                   "tsmom_trend", "xsec_etf_mom"]

_captured: list[dict] = []
_orig_run_backtest = bt.run_backtest
_capital_scale = 1.0
_last_sleeve_result: dict = {}
_collected_results: dict[str, dict] = {}


def _capturing_run_backtest(*args, **kwargs):
    if _capital_scale != 1.0:  # deployment-true: sleeve trades its share
        if len(args) >= 3:
            cm = args[2] or IBKRHKCostModel()
            args = (*args[:2], replace(cm, capital=cm.capital * _capital_scale),
                    *args[3:])
        else:
            cm = kwargs.get("cost_model") or IBKRHKCostModel()
            kwargs["cost_model"] = replace(cm, capital=cm.capital * _capital_scale)
    res = _orig_run_backtest(*args, **kwargs)
    _captured.append(res)
    return res


def _skip_sleeve_record(path, content, **_) -> Path:
    """Stands in for qcore.records.save_record while a sleeve runs for the
    blend. The run exists to capture a return stream; in pass 2 its metrics
    are at weight-share capital, which is not the sleeve's own record."""
    print(f"not saved while blending: {path}")
    return Path(path)


def missing_record(key: str) -> str:
    """One line for a sleeve whose results/<key>.json does not exist: what
    the file is for and, for the default sleeves, the command that writes it
    (each of those modules prints its record as JSON on standard output)."""
    line = (f"results/{key}.json not found: it declares which run of "
            f"src/strategies/{key}.py the blend uses")
    if key in DEFAULT_SLEEVES:
        line += (". Create it first, and likewise for every other sleeve: "
                 f"python src/strategies/{key}.py > results/{key}.json")
    return line


def sleeve_returns(key: str) -> pd.Series | None:
    """Execute a strategy module, return the daily net returns of the run
    identified by its declared name. Never select using realized Sharpe.
    The sleeve's own results/<key>.json is left byte-for-byte as it was."""
    global _last_sleeve_result
    module = SRC / "strategies" / f"{key}.py"
    if not module.exists():
        raise ValueError(f"{key}: no strategy module")
    record = ROOT / "results" / f"{key}.json"
    if not record.is_file():
        raise FileNotFoundError(missing_record(key))
    kept = record.read_bytes()
    try:
        declared = json.loads(kept)
    except ValueError as err:
        # an interrupted "> results/<key>.json" leaves an empty or partial file
        raise ValueError(f"results/{key}.json is not valid JSON ({err}); it must hold "
                         "the record the strategy module prints") from None
    if isinstance(declared, dict) and declared.get("study"):
        raise ValueError(f"{key}: marked 'study', cannot include in an ensemble")
    declared_name = declared.get("name") or declared.get("metrics", {}).get("name")

    _captured.clear()
    previous_run = bt.run_backtest
    previous_argv = sys.argv
    previous_save = records.save_record
    bt.run_backtest = _capturing_run_backtest
    sys.argv = [str(module)]
    records.save_record = _skip_sleeve_record
    try:
        runpy.run_path(str(module), run_name="__main__")
    finally:
        bt.run_backtest = previous_run
        sys.argv = previous_argv
        records.save_record = previous_save
        # a module that writes its results file directly is undone here
        if not record.is_file() or record.read_bytes() != kept:
            record.write_bytes(kept)
            print(f"restored {record}: a sleeve run for the blend does not "
                  "replace the sleeve's own record")
    if not _captured:
        raise ValueError(f"{key}: module produced no backtest runs")
    matches = [r for r in _captured if r.get("name") == declared_name]
    if len(matches) != 1:
        names = sorted({str(r.get("name")) for r in _captured})
        ran = ", ".join(names[:8]) + (", ..." if len(names) > 8 else "")
        found = (f"{key}: expected one captured run named {declared_name!r}; "
                 f"found {len(matches)} (the module ran: {ran}). ")
        if matches:
            raise ValueError(found + "Runs that share a name cannot be told apart; "
                             "the module must name each run once.")
        raise ValueError(
            found + f"results/{key}.json declares a name the module does not "
            "produce. That file is a preserved record: it is not edited or "
            "re-based to make a blend run, and the blend never picks a run by "
            "its performance instead. To blend this sleeve, work in a separate "
            f"copy of the project: regenerate results/{key}.json there so that "
            "it declares the run the module now produces, and run the blend in "
            "that copy.")
    best = matches[0]
    _last_sleeve_result = best
    return best["returns"].rename(key)


def _collect(keys: list[str], scale: dict | None = None):
    """Run each sleeve (optionally with weight-share capital) and return
    the returns frame with each sleeve's pre-live leading gap parked in
    T-bills; interior/trailing NaNs stay loud."""
    global _capital_scale
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("sleeve keys must be nonempty and unique")
    _collected_results.clear()
    sleeves = {}
    for key in keys:
        share = float(scale[key]) if scale is not None else 1.0
        if not np.isfinite(share) or share <= 0:
            raise ValueError(f"{key}: invalid sleeve capital scale")
        _capital_scale = share
        tag = f" (capital x{_capital_scale:.2f})" if scale is not None else ""
        print(f"running sleeve: {key}{tag}")
        try:
            r = sleeve_returns(key)
        finally:
            _capital_scale = 1.0
        if (r is None or r.empty or not isinstance(r.index, pd.DatetimeIndex)
                or not r.index.is_unique or not r.index.is_monotonic_increasing
                or not np.isfinite(r.to_numpy(dtype=float)).all()):
            raise ValueError(f"{key}: missing, nonfinite or invalid dated returns")
        sleeves[key] = r
        _collected_results[key] = _last_sleeve_result
    if not sleeves:
        sys.exit("no sleeves captured")
    rets = pd.DataFrame(sleeves)
    rf = cash_daily_return(rets.index)
    for k in rets:
        pre = rets.index < rets[k].first_valid_index()
        rets.loc[pre, k] = rf[pre]
    if not np.isfinite(rets.to_numpy()).all():
        raise ValueError("sleeves have interior or trailing missing returns")
    return rets, rf


def _cagr(r: pd.Series) -> float:
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def held_blend(rets: pd.DataFrame, w: pd.Series, reset) -> tuple[pd.Series, float]:
    """Blend sleeve returns when the capital split is put back to `w` only
    on the rows flagged in `reset` and drifts with relative performance in
    between. Returns the blended daily returns and the one-sided turnover
    between sleeves that those resets add up to: at each reset, half the sum
    of absolute share changes, so 1% of capital moved from one sleeve to
    another counts as 1%."""
    r = rets.to_numpy(dtype=float)
    target = w.reindex(rets.columns).to_numpy(dtype=float)
    reset = np.asarray(reset, dtype=bool)
    if (len(reset) != len(r) or not np.isfinite(r).all() or (r <= -1.0).any()
            or not np.isfinite(target).all() or (target < 0).any()
            or abs(target.sum() - 1.0) > 1e-9):
        raise ValueError("held_blend needs finite returns above -100%, one reset "
                         "flag per row and nonnegative weights summing to one")
    share, out, moved = target.copy(), np.empty(len(r)), 0.0
    for i in range(len(r)):
        out[i] = share @ r[i]
        grown = share * (1.0 + r[i])
        share = grown / grown.sum()
        if reset[i]:
            moved += 0.5 * np.abs(target - share).sum()
            share = target.copy()
    return pd.Series(out, index=rets.index), float(moved)


def rebalance_sensitivity(rets: pd.DataFrame, w: pd.Series, rf: pd.Series) -> dict:
    """The same sleeve streams and weights with the split between sleeves
    reset every close (the headline constant-mix blend), at month-ends only,
    and never. Turnover is the uncosted re-split between sleeves.

    Two turnover keys end up in the saved metrics. Both end in _oneside, but
    their definitions differ by a factor of two:
      ann_intersleeve_turnover_oneside (produced here) is conventional
        one-way turnover: 0.5 x sum |target share - drifted share| at each
        reset, per year. Moving 1% of capital between two sleeves counts 1%.
      ann_turnover_oneside (produced by qcore.backtest.metrics; for the
        blend it is the weighted trading inside the sleeves, without this
        re-split) is the full sum of absolute weight changes, buys plus
        sells, per year. The same 1% move would count 2%.
    Double the first, or halve the second, before putting them side by
    side. The key names are kept because saved records use them."""
    month = rets.index.to_period("M")
    oos = rets.index >= pd.Timestamp(OOS_SPLIT)
    years = len(rets) / TRADING_DAYS
    cadences = {"daily": np.ones(len(rets), dtype=bool),
                "monthly": np.append(month[1:] != month[:-1], True),
                "never": np.zeros(len(rets), dtype=bool)}
    out = {}
    for name, reset in cadences.items():
        r, moved = held_blend(rets, w, reset)
        full, ins, outs = _stats(r, rf), _stats(r[~oos], rf), _stats(r[oos], rf)
        row = {"full_sharpe": full["sharpe"], "is_sharpe": ins["sharpe"],
               "oos_sharpe": outs["sharpe"], "full_cagr": full["cagr"],
               "full_maxdd": full["maxdd"],
               "ann_intersleeve_turnover_oneside": round(moved / years, 3)}
        out[name] = {k: (v if np.isfinite(v) else None) for k, v in row.items()}
    return out


def record_paths(keys: list[str]) -> tuple[Path, Path]:
    """Where a blend of `keys` is saved. results/ensemble.json and
    results/sleeve_returns.csv are what live sizing, the kill-rule monitor
    and the pinned registry read, so only THE MODEL gets those names."""
    results = ROOT / "results"
    if list(keys) == DEFAULT_SLEEVES:
        return results / "ensemble.json", results / "sleeve_returns.csv"
    tag = "+".join(keys)
    return (results / "adhoc" / f"ensemble_{tag}.json",
            results / "adhoc" / f"sleeve_returns_{tag}.csv")


def build(keys: list[str], paths: tuple[Path, Path] | None = None,
          rebase: bool | None = None) -> None:
    """Blend `keys` and save the metrics JSON and the sleeve-returns CSV to
    `paths` (default: the production names). rebase=None defers to --rebase
    / QCORE_REBASE."""
    if paths is None:
        paths = record_paths(DEFAULT_SLEEVES)
        if list(keys) != DEFAULT_SLEEVES and any(p.exists() for p in paths):
            raise ValueError("results/ensemble.json and results/sleeve_returns.csv hold "
                             "the production blend; save any other blend elsewhere "
                             "(paths=record_paths(keys))")
    json_path, csv_path = paths

    # PASS 1: full-capital runs fix the weights (vol is cost-insensitive;
    # fixing them here avoids the weight->capital->vol circularity)
    rets1, _ = _collect(keys)

    # inverse-vol weights from IN-SAMPLE data only; renormalized to sum 1.
    # The frame is cash-padded: a sleeve's pre-live rows count as T-bill days.
    is_rets = rets1.loc[: pd.Timestamp(OOS_SPLIT) - pd.Timedelta(1, unit="D")]
    vol = is_rets.std() * np.sqrt(TRADING_DAYS)
    if len(is_rets) < 2 or not np.isfinite(vol.to_numpy()).all():
        raise ValueError("every sleeve needs finite in-sample volatility")
    iv = (1.0 / vol.clip(lower=0.02))
    w = iv / iv.sum()
    print("\nsleeve weights (inverse IS-vol):")
    print((w * 100).round(1).astype(str) + "%")

    # PASS 2: deployment-true accounting at weight-share capital
    print("\nsecond pass (sleeve-share capital):")
    rets, rf = _collect(list(rets1.columns), scale=w.to_dict())
    if not rets.index.equals(rets1.index) or not rets.columns.equals(rets1.columns):
        raise ValueError("sleeve dates or identities changed between capital passes")

    combo = (rets * w).sum(axis=1, skipna=False)

    def blended_component(field, cash_padding=False):
        frame = pd.DataFrame({k: _collected_results[k][field] for k in rets})
        frame = frame.reindex(rets.index)
        for k in frame:
            pre = frame.index < _collected_results[k]["returns"].index[0]
            frame.loc[pre, k] = rf[pre] if cash_padding else 0.0
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"nonfinite sleeve {field}")
        return frame.mul(w).sum(axis=1, skipna=False)

    # "turnover" is the weighted trading inside the sleeves on the engine's
    # definition (sum of absolute weight changes, buys plus sells), so the
    # blend's ann_turnover_oneside is on that basis: twice the one-way
    # convention of ann_intersleeve_turnover_oneside (rebalance_sensitivity).
    res = {"name": "ensemble", "returns": combo,
           "gross_returns": blended_component("gross_returns", cash_padding=True),
           "equity": (1 + combo).cumprod(),
           "turnover": blended_component("turnover"),
           "costs": blended_component("costs"),
           "rf_daily": rf}
    if all("withholding" in _collected_results[k] for k in rets):
        res["withholding"] = blended_component("withholding")
    if all("cash_returns" in _collected_results[k] for k in rets):
        res["cash_returns"] = blended_component("cash_returns", cash_padding=True)
    m = metrics(res)
    m["sleeve_weights"] = {k: float(v) for k, v in w.items()}
    m["sleeve_correlations"] = rets.corr().round(2).to_dict()
    m["capital_accounting"] = {
        "mode": "sleeve-share capital (pass 2); weights from full-capital pass 1",
        "ann_drag_vs_full_capital": round(
            _cagr((rets1 * w).sum(axis=1)) - _cagr(combo), 4),
    }
    m["blend_convention"] = ("constant-mix: sleeve shares reset to the in-sample "
                             "weights at every close; the re-split between sleeves "
                             "is not costed")
    m["rebalance_sensitivity"] = rebalance_sensitivity(rets, w, rf)

    print("\nensemble metrics:")
    print(json.dumps({k: m[k] for k in ["full", "in_sample", "out_of_sample",
                                        "pct_positive_months", "worst_month"]}, indent=2))
    print()
    records.save_json(json_path, m, rebase=rebase)
    records.save_csv(csv_path, rets.assign(ENSEMBLE=combo), rebase=rebase)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0])
    ap.add_argument("keys", nargs="*",
                    help="sleeve keys to blend (default: the four production sleeves)")
    ap.add_argument(records.REBASE_FLAG, action="store_true",
                    help="replace a saved record that differs from this run")
    args = ap.parse_args(argv)
    keys = args.keys
    if keys:
        # explicit keys must resolve fully -- a typo'd key silently skipped
        # would write ensemble.json minus the intended sleeve
        bad = [k for k in keys if not (SRC / "strategies" / f"{k}.py").exists()]
        if bad:
            sys.exit(f"unknown sleeve key(s): {', '.join(bad)} -- need both "
                     "src/strategies/<key>.py and results/<key>.json")
        unrecorded = [k for k in keys if not (ROOT / "results" / f"{k}.json").exists()]
        if unrecorded:
            sys.exit(missing_record(unrecorded[0]))
    else:
        # THE MODEL only - never silently blend killed sleeves or studies
        # (an earlier glob default did exactly that)
        keys = DEFAULT_SLEEVES
    try:
        build(keys, paths=record_paths(keys), rebase=True if args.rebase else None)
    except FileNotFoundError as err:
        # a saved input is missing: the data cache (qcore.data names the file
        # and the downloader) or a sleeve record (missing_record names the
        # command that writes it). One line, no traceback.
        sys.exit(str(err))


if __name__ == "__main__":
    main()
