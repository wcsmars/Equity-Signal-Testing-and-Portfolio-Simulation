"""Multiple-testing evidence: a registry of logged trials and deflated Sharpes.

Scans every saved results/*_variants.csv, leaves out explicit diagnostic
placeholders, counts the logged trials, and asks:

    Given everything that was tried, what is the probability that the
    headline Sharpe is an artifact of the search itself?

It reports the deflated Sharpe ratio (DSR, qcore.stats.deflated_sharpe) of
each production sleeve and of the blended portfolio, the probabilistic
Sharpe ratio (PSR) after 2018, and block-bootstrap confidence intervals for
the blend.

Method and judgment calls (all surfaced in the output, none hidden):
  - Trial counts come from the variants CSVs. Variants within a family are
    highly correlated, so the raw count OVERSTATES the effective number of
    independent trials; the family count UNDERSTATES it. DSR is therefore
    reported under three explicit assumptions (per-sleeve grid; families
    only; every logged trial) rather than one pretended-precise number.
  - Per-sleeve DSR uses that sleeve's own selection grid as the trial pool
    (its variants CSV: N and the trial-Sharpe dispersion), evaluated on the
    sleeve's daily excess returns from results/sleeve_returns.csv. Where a
    later logged search also chose parameters of the deployed variant (the
    xsec_etf_mom buffer/drift rule came from the ETF rows of the
    rank_hysteresis study, so the deployed variant is not a row of its own
    grid), dsr_selection_path pools those rows with the own grid and is
    reported beside the unchanged dsr_own_grid.
  - Post-2018 PSR is descriptive: which families were kept and later
    specification changes consulted this segment, so it is not an untouched
    holdout.
  - Ensemble returns in sleeve_returns.csv are the sleeve-share-capital
    (pass 2) stream; they are checked against results/ensemble.json at load.

Inputs, all under results/ (none is shipped with the repository; each is
written by the command in INPUT_COMMANDS):
  mean_reversion_variants.csv     python research/mean_reversion_sweep.py
  seasonality_flows_variants.csv  python src/strategies/seasonality_flows.py --sweep
  tsmom_trend_variants.csv        python src/strategies/tsmom_trend.py --sweep
  xsec_etf_mom_variants.csv       python src/strategies/xsec_etf_mom.py --sweep
  rank_hysteresis_variants.csv    python research/rank_hysteresis_sweep.py
  sleeve_returns.csv, ensemble.json   python src/ensemble.py
plus the price cache, whose ^IRX series is the cash rate. Any other
*_variants.csv found in results/ is counted too.

Run: python scripts/trial_registry.py [--rebase]
Output: a console table and results/trial_registry.json ("study": true).
  The JSON is a record (qcore.records): a first run writes it, an identical
  re-run leaves it alone, and a run that differs keeps the saved record and
  writes to results/recomputed/trial_registry.json. Pass --rebase (or set
  QCORE_REBASE=1) to replace the record.

Exit status: 0 on completion; 1, with one line, when a saved input is
missing (the line names the file and the command that writes it); 2 for an
unknown or shortened flag. Inputs that contradict each other, such as a
returns file that does not reproduce the Sharpe declared in ensemble.json,
raise ValueError with a traceback and write nothing.

Limits: only the logs present in results/ are counted. A checkout that has
run just the sweeps published here counts far fewer trials than a research
history that also holds every rejected idea, so its DSR is flattering; the
count is a lower bound on what was tried, never a complete ledger. The
trial count that DSR needs is a judgment call, and the two ensemble
scenarios are sensitivity cases, not statistical bounds.
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import OOS_SPLIT, TRADING_DAYS, cash_daily_return  # noqa: E402
from qcore.records import save_json  # noqa: E402
from qcore.stats import (block_bootstrap_sharpe_ci, deflated_sharpe,  # noqa: E402
                         probabilistic_sharpe)

LIVE_SLEEVES = ["mean_reversion", "seasonality_flows", "tsmom_trend", "xsec_etf_mom"]
# family -> the variants CSV that constitutes its selection grid
SLEEVE_GRID = {
    "mean_reversion": "mean_reversion_variants.csv",
    "seasonality_flows": "seasonality_flows_variants.csv",
    "tsmom_trend": "tsmom_trend_variants.csv",
    "xsec_etf_mom": "xsec_etf_mom_variants.csv",
}
# sleeve -> later logged searches that also chose parameters of the deployed
# variant, as (variants CSV, {column: required value}) row selections
SLEEVE_EXTRA_GRIDS = {
    "xsec_etf_mom": [("rank_hysteresis_variants.csv", {"sleeve": "etf"})],
}
# saved input in results/ -> the command that writes it, in the order a fresh
# checkout produces them (selection grids, then the blend)
INPUT_COMMANDS = {
    "mean_reversion_variants.csv": "python research/mean_reversion_sweep.py",
    "seasonality_flows_variants.csv": "python src/strategies/seasonality_flows.py --sweep",
    "tsmom_trend_variants.csv": "python src/strategies/tsmom_trend.py --sweep",
    "xsec_etf_mom_variants.csv": "python src/strategies/xsec_etf_mom.py --sweep",
    "rank_hysteresis_variants.csv": "python research/rank_hysteresis_sweep.py",
    "sleeve_returns.csv": "python src/ensemble.py",
    "ensemble.json": "python src/ensemble.py",
}
SHARPE_COLS = ("full_sharpe", "Sharpe", "sharpe")  # column-name variants seen in CSVs
DIAGNOSTIC_FILES = {
    "family_dsr": "reconstructions and scenarios of already logged trials",
    "joint_tail_memo": "diagnostic placeholder; no variant selection",
    "production_registry": "infrastructure placeholder; no backtests",
    "sleeve_monitor": "diagnostic placeholder; no variant selection",
}


def _full_sharpes(df: pd.DataFrame) -> list[float]:
    for c in SHARPE_COLS:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="raise").dropna()
            if not np.isfinite(vals).all():
                raise ValueError(f"nonfinite logged Sharpe in {c}")
            return [float(x) for x in vals]
    return []


def scan_registry() -> dict:
    files = sorted(glob.glob(str(ROOT / "results" / "*_variants.csv")))
    reg, excluded, total = {}, {}, 0
    for f in files:
        df = pd.read_csv(f)
        stem = Path(f).stem.replace("_variants", "")
        if stem in DIAGNOSTIC_FILES:
            excluded[stem] = {"n_rows": len(df), "reason": DIAGNOSTIC_FILES[stem]}
            continue
        reg[stem] = {"n_trials": len(df), "full_sharpes": _full_sharpes(df)}
        total += len(df)
    return {"families": reg, "n_files": len(reg), "n_trials_total": total,
            "n_files_scanned": len(files), "excluded_diagnostics": excluded}


def selection_path_pool(sleeve: str) -> dict | None:
    """Own grid plus the rows of every later search that selected part of
    the deployed variant; None when the own grid is the whole search."""
    extras = SLEEVE_EXTRA_GRIDS.get(sleeve)
    if not extras:
        return None
    own = pd.read_csv(ROOT / "results" / SLEEVE_GRID[sleeve])
    n, sharpes = len(own), _full_sharpes(own)
    sources = {SLEEVE_GRID[sleeve]: len(own)}
    for fname, row_filter in extras:
        df = pd.read_csv(ROOT / "results" / fname)
        for col, val in row_filter.items():
            df = df[df[col] == val]
        if df.empty:
            raise ValueError(f"{fname}: no rows match {row_filter} for {sleeve}")
        n += len(df)
        sharpes += _full_sharpes(df)
        sources[fname] = len(df)
    return {"n_trials": n, "full_sharpes": sharpes, "sources": sources}


def excess_returns() -> dict[str, pd.Series]:
    # round_trip: the saved stream is read back digit for digit (the default
    # parser can be off in the last bit)
    r = pd.read_csv(ROOT / "results" / "sleeve_returns.csv",
                    index_col=0, parse_dates=True, float_precision="round_trip")
    if (r.empty or not isinstance(r.index, pd.DatetimeIndex)
            or not r.index.is_unique or not r.index.is_monotonic_increasing
            or not np.isfinite(r.to_numpy(dtype=float)).all()):
        raise ValueError("sleeve returns must be finite on unique ordered dates")
    rf = cash_daily_return(r.index)
    ex = r.sub(rf, axis=0)
    # sleeves are rf-filled before their live date (excess ~0; the CSV
    # float round-trip leaves ~1e-17 residuals, so test with a tolerance):
    # slice each column to its genuinely-live span so skew/kurt/T are honest
    out = {}
    for c in r.columns:
        live = ex[c][ex[c].abs() > 1e-12]
        # Preserve the full portfolio history, including its initial cash,
        # so the ENSEMBLE reproduction checks the same span as metrics().
        out[c] = ex[c] if c == "ENSEMBLE" or live.empty else ex[c].loc[live.index[0]:]
    return out


def missing_inputs() -> list[str]:
    """Saved inputs this report cannot run without, in the order they are made."""
    return [name for name in INPUT_COMMANDS if not (ROOT / "results" / name).is_file()]


def main() -> None:
    missing = missing_inputs()
    if missing:
        more = (f" ({len(missing) - 1} more saved input(s) are also missing: "
                f"{', '.join(missing[1:])})" if len(missing) > 1 else "")
        sys.exit(f"results/{missing[0]} not found: the trial registry reads the saved "
                 "selection grids and the blended returns. Create it first: "
                 f"{INPUT_COMMANDS[missing[0]]}{more}")
    registry = scan_registry()
    ex = excess_returns()
    split = pd.Timestamp(OOS_SPLIT)

    # sanity: the ENSEMBLE column must match the declared deployment-true JSON
    decl = json.loads((ROOT / "results" / "ensemble.json").read_text())
    ens = ex["ENSEMBLE"]
    got = round(float(ens.mean() / ens.std(ddof=1)) * math.sqrt(TRADING_DAYS), 2)
    if not np.isfinite(got) or abs(got - decl["full"]["sharpe"]) > 0.011:
        raise ValueError(f"sleeve_returns ENSEMBLE Sharpe {got} != declared {decl['full']['sharpe']}")
    if (str(ens.index[0].date()) != decl["start"]
            or str(ens.index[-1].date()) != decl["end"]):
        raise ValueError("sleeve_returns ENSEMBLE date span differs from ensemble.json")

    all_sharpes = [s for fam in registry["families"].values()
                   for s in fam["full_sharpes"]]
    n_total = registry["n_trials_total"]
    n_families = registry["n_files"]

    out = {"study": True, "key": "trial_registry",
           "registry": {k: v["n_trials"] for k, v in registry["families"].items()},
           "n_trials_total": n_total, "n_families": n_families,
           "excluded_diagnostics": registry["excluded_diagnostics"],
           "conventions": "daily excess returns vs engine rf; Sharpes annualized "
                          "for display; deployment-true (post 2026-07-02 hardening)",
           "sleeves": {}, "ensemble": {}}

    print(f"registry: {n_total} logged trials across {n_families} variant files\n")
    print(f"{'series':22s} {'SR':>5s}  {'N':>4s}  {'hurdle':>6s}  {'DSR':>6s}   "
          f"{'OOS PSR>0':>9s}")

    for k in LIVE_SLEEVES:
        grid = registry["families"][Path(SLEEVE_GRID[k]).stem.replace("_variants", "")]
        d = deflated_sharpe(ex[k], n_trials=grid["n_trials"],
                            trial_sharpes_ann=grid["full_sharpes"])
        oos = probabilistic_sharpe(ex[k].loc[split:])
        out["sleeves"][k] = {"dsr_own_grid": d, "oos_psr": oos}
        print(f"{k:22s} {d['sharpe_ann']:5.2f}  {d['n_trials']:4d}  "
              f"{d['hurdle_expected_max_sharpe_ann']:6.2f}  {d['dsr']:6.1%}   "
              f"{oos['psr']:9.1%}")
        path = selection_path_pool(k)
        if path is not None:
            dp = deflated_sharpe(ex[k], n_trials=path["n_trials"],
                                 trial_sharpes_ann=path["full_sharpes"])
            dp["sources"] = path["sources"]
            out["sleeves"][k]["dsr_selection_path"] = dp
            print(f"{'  + selection path':22s} {dp['sharpe_ann']:5.2f}  "
                  f"{dp['n_trials']:4d}  "
                  f"{dp['hurdle_expected_max_sharpe_ann']:6.2f}  {dp['dsr']:6.1%}")

    scenarios = {
        # lenient: selection happened at family level (variant files as proxy)
        "families_only": dict(n_trials=n_families, trial_sharpes_ann=None),
        # brutal: every logged variant counts as an independent trial
        "every_logged_trial": dict(n_trials=n_total, trial_sharpes_ann=None),
    }
    ens_block = {"bootstrap_full": block_bootstrap_sharpe_ci(ens),
                 "bootstrap_oos": block_bootstrap_sharpe_ci(ens.loc[split:]),
                 "oos_psr": probabilistic_sharpe(ens.loc[split:]),
                 "dsr": {}}
    for name, kw in scenarios.items():
        d = deflated_sharpe(ens, n_trials=kw["n_trials"],
                            trial_sharpes_ann=all_sharpes)
        ens_block["dsr"][name] = d
        print(f"{'ENSEMBLE ('+name+')':40s} N={d['n_trials']:<4d} "
              f"hurdle {d['hurdle_expected_max_sharpe_ann']:5.2f}  "
              f"DSR {d['dsr']:6.1%}")
    out["ensemble"] = ens_block
    out["caveats"] = [
        "n_trials is a judgment call: correlated variants are not independent "
        "trials; families_only and every_logged_trial are sensitivity scenarios, "
        "not statistical bounds on the unknown effective search size",
        "trial-Sharpe dispersion pooled across all logged variants for the "
        "ensemble scenarios; per-sleeve DSR uses the sleeve's own grid",
        "dsr_selection_path (xsec_etf_mom) pools the own grid with the ETF rows "
        "of rank_hysteresis, the later search that chose the deployed "
        "buffer/drift rule; dsr_own_grid is unchanged",
        "Post-2018 PSR is descriptive: family survival and later strategy "
        "revisions consulted this segment, including the trend cash-residual "
        "override. This is not an untouched holdout",
        "Only the saved results/*_variants.csv records are counted. A sweep "
        "re-run whose rows differ keeps the saved log and writes to "
        "results/recomputed/, which this scan does not read, and a log "
        "replaced with --rebase loses its earlier rows, so this registry is "
        "not a complete historical or prospective experiment ledger",
        "ensemble returns are deployment-true pass-2 (withholding + "
        "sleeve-share capital) from sleeve_returns.csv",
    ]

    print(f"\nENSEMBLE full  bootstrap CI {ens_block['bootstrap_full']['ci']}  "
          f"P(SR<=0) {ens_block['bootstrap_full']['prob_sharpe_le_0']:.1%}")
    print(f"ENSEMBLE OOS   bootstrap CI {ens_block['bootstrap_oos']['ci']}  "
          f"P(SR<=0) {ens_block['bootstrap_oos']['prob_sharpe_le_0']:.1%}   "
          f"PSR>0 {ens_block['oos_psr']['psr']:.1%}")

    print()
    save_json(ROOT / "results" / "trial_registry.json", out)


def _parse_args(argv=None):
    # allow_abbrev=False: a shortened flag such as --reb is refused, not accepted
    # by the parser and then missed by the exact-name check that acts on it
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--rebase", action="store_true",
                        help="replace results/trial_registry.json when this run differs from it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    main()
