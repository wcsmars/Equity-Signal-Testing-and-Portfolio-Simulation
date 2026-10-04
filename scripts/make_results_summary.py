#!/usr/bin/env python3
"""Results summary: one table and three charts from the strategy backtests.

Runs the eight strategy modules of src/strategies/ at their default
parameters and the four-strategy blend of src/ensemble.py, in this process,
on the local price cache, and writes what the README shows:

  summary.json            every figure of the table, and the blend's weights,
                          correlations and rebalance sensitivity
  summary.md              the table as Markdown, with notes on how to read it
  equity_production.png   growth of $1: the four kept strategies and the blend
  drawdown_ensemble.png   the blend's drawdown from its running peak
  equity_rejected.png     growth of $1: the four rejected strategies

Usage:
  python scripts/make_results_summary.py [--out DIR]

Without --out the files go to examples/qcore/ under the project root. The
price cache is data/, or the directory named by QCORE_DATA_DIR (qcore.data).

How the figures are produced. Each strategy is run through its module's own
default entry point, the function that `python src/strategies/<key>.py`
calls, so parameters, universe, cost settings and run name come from the
module and are not repeated here. The engine result behind the metrics that
function reports is kept, and qcore.backtest.metrics turns it into the table
row. tsmom_voltarget is a study of nine variants and has no single default
portfolio; its row is the constant-leverage variant (1.64x), the one its
selection rule picks on market data.
The blend comes from ensemble.build. That function reads each sleeve's
declared run from results/<key>.json and saves its output under results/;
here it is pointed at a temporary directory holding the four records just
computed. Modules that save a record of their own (pairs_statarb) run with
saving switched off. Nothing under results/ is read or written, and an --out
inside results/ is refused.

Only derived statistics and curves are written: no price series, and no
path of this machine. Two runs on the same cache and library versions give
the same summary.json and summary.md.

Charts are drawn with matplotlib's Agg canvas through its object interface.
pyplot is never imported, so the caller's backend and rcParams are left as
they were. Colours: four hues that stay apart in every pairing under
simulated red-green colour blindness, with the blend in black and heavier;
every line is also named in a legend and labelled with its end value.

Exit status: 0 the five files were written; 1, with one line, when the price
cache is missing; 2 refused (an --out inside results/, or a synthetic sample
cache with the default --out, which holds the market-data results) or a
usage error.

Limits: the figures are those of this code on the cache it was given, so a
later download gives slightly different numbers. Every limit of the
backtests applies: execution assumed at the decision close, modelled costs,
a 2018 split that later research choices consulted, and a stock universe
made of today's large companies. The chart of the rejected strategies is a
record of what was tried, not a set of results.
"""

import argparse
import contextlib
import importlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path[:0] = [str(SRC), str(ROOT / "scripts")]

import ensemble  # noqa: E402
import qcore.data as qdata  # noqa: E402
from make_sample_data import MARKER as SAMPLE_MARKER, README_NAME as SAMPLE_README  # noqa: E402
from monitor import KILL_MAX_DRAWDOWN  # noqa: E402
from qcore import records  # noqa: E402
from qcore.backtest import OOS_SPLIT, TBILL_HAIRCUT_BPS, TRADING_DAYS, metrics  # noqa: E402
from qcore.costs import IBKRHKCostModel, US_DIV_WITHHOLDING  # noqa: E402

DEFAULT_OUT = ROOT / "examples" / "qcore"
RESULTS = ROOT / "results"
FILES = ("summary.json", "summary.md", "equity_production.png",
         "drawdown_ensemble.png", "equity_rejected.png")
EXIT_OK, EXIT_NO_DATA, EXIT_REFUSED = 0, 1, 2

# Chart colours. Text and axes use the three inks; data uses HUES in this
# fixed order (blue, orange, green, violet) and black for the blend.
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS, STOP = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#d03b3b"
HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
GROWTH_TICKS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FINE_GROWTH_TICKS = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 7.5, 10)  # for a narrow range


# ------------------------------------------------------------ strategy runs
def _reported(module, default_run) -> dict:
    """Call a module's default run and return the engine result it turned
    into metrics last (a module that screens candidates first, as
    pairs_statarb does with its pairs, reports its portfolio last)."""
    seen, original = [], module.metrics

    def recording(result, *args, **kwargs):
        seen.append(result)
        return original(result, *args, **kwargs)

    module.metrics = recording
    try:
        default_run()
    finally:
        module.metrics = original
    return seen[-1]


def _trend_default(module) -> dict:
    prices = module.load_prices()[module.RISK + [module.CASH]]
    return _reported(module, lambda: module.run_variant(
        prices, *module.build_signals(prices), **module.BEST))


def _constant_leverage(module) -> dict:
    prices = module.load_prices()[module.RISK + [module.CASH]]
    return module.run_variant(prices, *module.build_signals(prices), "dilute", "klev164")[1]


# key -> (chart label, how the engine result of the module's default run is obtained)
STRATEGIES = {
    "mean_reversion": ("Dip buying", lambda m: _reported(m, m.main)),
    "seasonality_flows": ("Turn of month", lambda m: _reported(m, m.run_best)),
    "tsmom_trend": ("Trend following", _trend_default),
    "xsec_etf_mom": ("ETF momentum rotation",
                     lambda m: m.run_variant(m.load_prices(), **m.BEST_PARAMS)),
    "xsec_stock_mom": ("Stock momentum (biased universe)", lambda m: _reported(m, m.main)),
    "vol_regime": ("VIX regime switch", lambda m: _reported(m, m.main)),
    "pairs_statarb": ("ETF pairs", lambda m: _reported(m, m.main)),
    "tsmom_voltarget": ("Trend at 1.64x leverage", _constant_leverage),
}
KEPT = tuple(ensemble.DEFAULT_SLEEVES)
REJECTED = tuple(key for key in STRATEGIES if key not in KEPT)


@contextlib.contextmanager
def _quiet_and_unsaved():
    """Run strategy code without its console output and without letting it
    save a record: qcore.records.save_record is the one function every
    module writes through."""
    original = records.save_record
    records.save_record = lambda path, content, **_: Path(path)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        records.save_record = original


def run_strategies() -> dict:
    """key -> engine result of the module's default run, for all eight."""
    results = {}
    for key, (_, default_result) in STRATEGIES.items():
        module = importlib.import_module(f"strategies.{key}")
        with _quiet_and_unsaved():
            results[key] = default_result(module)
    return results


def run_blend(sleeve_metrics: dict) -> tuple[dict, pd.DataFrame]:
    """ensemble.build on the four kept sleeves, with a temporary directory
    standing in for the project root that it reads sleeve records from and
    saves to. Returns its metrics and its daily returns frame (one column
    per sleeve at its share of capital, and ENSEMBLE)."""
    project_root = ensemble.ROOT
    with tempfile.TemporaryDirectory() as scratch:
        saved = Path(scratch) / "results"
        saved.mkdir()
        for key in KEPT:
            (saved / f"{key}.json").write_text(json.dumps(sleeve_metrics[key]), encoding="utf-8")
        ensemble.ROOT = Path(scratch)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ensemble.build(list(KEPT))
        finally:
            ensemble.ROOT = project_root
        blend = json.loads((saved / "ensemble.json").read_text(encoding="utf-8"))
        streams = pd.read_csv(saved / "sleeve_returns.csv", index_col=0, parse_dates=True)
    return blend, streams


# ------------------------------------------------------------------ summary
def _finite(value):
    """JSON has no NaN: a statistic that is undefined on a short sample is null."""
    if isinstance(value, dict):
        return {name: _finite(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _row(m: dict) -> dict:
    return {"run": m["name"], "start": m["start"], "end": m["end"],
            "full": m["full"], "in_sample": m["in_sample"], "out_of_sample": m["out_of_sample"],
            "gross_sharpe_full": m["gross_full"]["sharpe"],
            "ann_turnover_buys_plus_sells": m["ann_turnover_oneside"],
            "ann_cost_drag": m["ann_cost_drag"],
            "ann_withholding_drag": m.get("ann_withholding_drag"),
            "ann_cash_income": m.get("ann_cash_income")}


def build_summary(results: dict, all_metrics: dict, blend: dict, prices: pd.DataFrame,
                  synthetic: bool) -> dict:
    strategies = []
    for key in (*KEPT, *REJECTED):
        row = {"key": key, "module": f"src/strategies/{key}.py",
               "label": STRATEGIES[key][0],
               "description": importlib.import_module(f"strategies.{key}").__doc__.splitlines()[0],
               "status": "kept" if key in KEPT else "rejected",
               **_row(all_metrics[key])}
        if "borrow_drag_ann" in results[key]:  # pairs_statarb: charged outside the engine
            row["ann_borrow_drag"] = round(float(results[key]["borrow_drag_ann"]), 4)
        if "margin_drag" in results[key]:      # tsmom_voltarget: charged outside the engine
            row["ann_financing_drag"] = round(
                float(results[key]["margin_drag"].mean() * TRADING_DAYS), 4)
        strategies.append(row)
    return _finite({
        "written_by": "scripts/make_results_summary.py",
        "synthetic_sample_data": synthetic,
        "price_cache": {"first_session": str(prices.index[0].date()),
                        "last_session": str(prices.index[-1].date()),
                        "sessions": int(len(prices))},
        "conventions": {
            "in_sample_before": OOS_SPLIT,
            "sharpe": "annualised, on daily returns in excess of the cash rate",
            "cash_rate": f"13-week bill yield less {TBILL_HAIRCUT_BPS:g} bps, previous close",
            "dividend_withholding": US_DIV_WITHHOLDING,
            "turnover": "yearly sum of absolute weight changes, buys plus sells",
            "cost_drag": "yearly return lost to modelled commissions, fees and slippage",
            "capital": IBKRHKCostModel().capital,
        },
        "strategies": strategies,
        "ensemble": {
            "module": "src/ensemble.py", "label": "Blend of the four kept strategies",
            "status": "kept", **_row(blend),
            "sleeve_weights": {k: round(v, 6) for k, v in blend["sleeve_weights"].items()},
            "sleeve_correlations": blend["sleeve_correlations"],
            "ann_drag_from_sleeve_share_capital":
                blend["capital_accounting"]["ann_drag_vs_full_capital"],
            "blend_convention": blend["blend_convention"],
            "rebalance_sensitivity": blend["rebalance_sensitivity"],
        },
    })


def _pct(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _num(value, pattern: str = "{:.2f}") -> str:
    return "n/a" if value is None else pattern.format(value)


def _table_line(name: str, row: dict) -> str:
    sharpes = " / ".join(_num(row[part]["sharpe"]) for part in ("full", "in_sample", "out_of_sample"))
    return (f"| {name} | {row['status']} | {row['start']} to {row['end']} | {_pct(row['full']['cagr'])} "
            f"| {_pct(row['full']['vol'])} | {sharpes} | {_pct(row['full']['maxdd'])} "
            f"| {_num(row['ann_turnover_buys_plus_sells'], '{:.1f}x')} | {_pct(row['ann_cost_drag'])} |")


def summary_markdown(summary: dict) -> str:
    conv, blend, cache = summary["conventions"], summary["ensemble"], summary["price_cache"]
    kept = [row for row in summary["strategies"] if row["status"] == "kept"]
    rejected = [row for row in summary["strategies"] if row["status"] == "rejected"]
    head = ("| Strategy | Status | Sample | CAGR | Volatility | Sharpe full / in sample / out of sample "
            "| Max drawdown | Turnover | Cost drag |\n"
            "| --- | --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |")

    def named(row):
        return f"{row['label']} (`{row['key']}`)"

    weights = ", ".join(f"`{key}` {_pct(share, 1)}" for key, share in blend["sleeve_weights"].items())
    cadence = blend["rebalance_sensitivity"]
    lines = [
        "# Backtest results: eight strategies and the four-strategy blend",
        "",
        *(["**SYNTHETIC SAMPLE DATA: these figures come from simulated prices and mean nothing.**", ""]
          if summary["synthetic_sample_data"] else []),
        f"Written by `python scripts/make_results_summary.py` from a price cache of "
        f"{cache['sessions']:,} sessions, {cache['first_session']} to {cache['last_session']}. "
        "Every figure is computed by the code in this repository at each module's default "
        "parameters; none is typed in by hand. `summary.json` holds the same figures.",
        "",
        head,
        *(_table_line(named(row), row) for row in kept),
        _table_line(f"**{blend['label']}** (`ensemble`)", blend),
        *(_table_line(named(row), row) for row in rejected),
        "",
        "How to read it:",
        "",
        "- Returns are net of modelled commissions, fees and slippage and of "
        f"{conv['dividend_withholding']:.0%} withholding on inferred dividends (Treasury funds "
        f"exempt). Idle cash earns the {conv['cash_rate'].replace(', previous close', '')}. "
        "These are modelling assumptions.",
        "- Sharpe ratios are in excess of that cash rate. In sample is before "
        f"{conv['in_sample_before']}; out of sample is from that date. The later period was "
        "inspected during development, so it is not an untouched holdout.",
        "- Turnover is the yearly sum of absolute weight changes: buys plus sells. Cost drag is "
        "the yearly return lost to commissions, fees and slippage.",
        "- A sample starts at the strategy's first position. The blend starts with the earliest "
        "strategy and holds Treasury bills for the others until they start.",
        "- Rejected strategies are listed for the record. Their figures are not investable "
        "results; each module's docstring gives the reason:",
        *(f"  - `{row['key']}`: {row['description']}" for row in rejected),
        "",
        "The blend:",
        "",
        f"- Weights, from the inverse of in-sample volatility and then fixed: {weights}.",
        f"- Each strategy trades its share of ${conv['capital']:,.0f}, where order minimums weigh "
        f"more. That costs {_pct(blend['ann_drag_from_sleeve_share_capital'])} a year against "
        "running each at the full amount, and is included above.",
        "- The blend's turnover is the trading inside the strategies. Moving capital between "
        "them is not costed: the split is reset at every close. Full-sample Sharpe with the "
        f"split reset every close / at month-ends / never: {_num(cadence['daily']['full_sharpe'])} / "
        f"{_num(cadence['monthly']['full_sharpe'])} / {_num(cadence['never']['full_sharpe'])}.",
        "",
        "Charts: `equity_production.png` (the kept strategies and the blend), "
        "`drawdown_ensemble.png` (the blend's drawdown) and `equity_rejected.png` (the rejected "
        "strategies).",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- charts
def _axes(title: str, subtitle: str):
    """A 9 x 4.9 inch figure on an Agg canvas and its single axes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.0, 4.9), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.065, 0.09, 0.835, 0.72), facecolor=SURFACE)
    fig.text(0.065, 0.94, title, fontsize=12.5, fontweight="bold", color=INK, va="center")
    fig.text(0.065, 0.875, subtitle, fontsize=9.5, color=SECONDARY, va="center")
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=MUTED)
    ax.tick_params(axis="x", length=3, color=AXIS, labelsize=9, labelcolor=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _time_axis(ax, first, last, split, labels_on_top: bool = True) -> None:
    """Year ticks, and the in-sample / out-of-sample boundary when it is in range."""
    import matplotlib.dates as mdates

    years = (last - first).days / 365.25
    ax.set_xlim(first, last)
    ax.xaxis.set_major_locator(mdates.YearLocator(5 if years > 15 else 2 if years > 6 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if first < split < last:
        ax.axvline(split, color=MUTED, linewidth=0.9, zorder=1)
        for text, offset, side in ((f"in sample: before {split.year}", -6, "right"),
                                   (f"out of sample: {split.year} onwards", 6, "left")):
            ax.annotate(text, xy=(split, 1.0 if labels_on_top else 0.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(offset, -3 if labels_on_top else 4), textcoords="offset points",
                        ha=side, va="top" if labels_on_top else "bottom",
                        fontsize=8.5, color=SECONDARY)


def _spread(targets: list, gap: float) -> list:
    """Positions as close to the ascending `targets` as possible with at
    least `gap` between neighbours: touching labels are grouped and each
    group is centred on the mean of its targets."""
    groups = [[target] for target in targets]

    def span(group):
        centre, half = sum(group) / len(group), gap * (len(group) - 1) / 2
        return centre - half, centre + half

    merged = True
    while merged:
        merged = False
        for i in range(len(groups) - 1):
            if span(groups[i + 1])[0] - span(groups[i])[1] < gap:
                groups[i:i + 2] = [groups[i] + groups[i + 1]]
                merged = True
                break
    return [span(group)[0] + gap * k for group in groups for k in range(len(group))]


def _end_labels(ax, curves: list) -> None:
    """A dot and the final value at the end of each line. Labels of lines
    that finish close together are moved apart and tied back with a leader."""
    import matplotlib.dates as mdates

    ends = sorted((float(equity.iloc[-1]), equity.index[-1], colour) for _, equity, colour, _ in curves)
    pixels = [ax.transData.transform((mdates.date2num(when), value))[1] for value, when, _ in ends]
    to_points = 72.0 / ax.figure.dpi
    for (value, when, colour), pixel, placed in zip(ends, pixels, _spread(pixels, 12.5 / to_points)):
        ax.plot([when], [value], marker="o", markersize=6, markerfacecolor=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.3, linestyle="none",
                clip_on=False, zorder=6)
        shift = (placed - pixel) * to_points
        leader = (dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=1, shrinkB=4)
                  if abs(shift) > 1.5 else None)
        ax.annotate(f"${value:,.2f}", xy=(when, value), xytext=(9, shift),
                    textcoords="offset points", va="center", fontsize=9, color=INK,
                    arrowprops=leader, annotation_clip=False)


def growth_chart(path: Path, curves: list, title: str, subtitle: str) -> None:
    """curves: (legend label, equity Series, colour, line width), in legend order."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, ax = _axes(title, subtitle)
    for rank, (label, equity, colour, width) in enumerate(curves):
        ax.plot(equity.index, equity.to_numpy(), color=colour, linewidth=width, label=label,
                solid_joinstyle="round", solid_capstyle="round", zorder=5 - rank)
    low = min(float(equity.min()) for _, equity, _, _ in curves) / 1.1
    high = max(float(equity.max()) for _, equity, _, _ in curves) * 1.2
    low = min(low, 0.9)
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ticks = [t for t in GROWTH_TICKS if low <= t <= high]
    if len(ticks) < 4:
        ticks = [t for t in FINE_GROWTH_TICKS if low <= t <= high]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    _time_axis(ax, min(equity.index[0] for _, equity, _, _ in curves),
               max(equity.index[-1] for _, equity, _, _ in curves), pd.Timestamp(OOS_SPLIT))
    _end_labels(ax, curves)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=SECONDARY,
              handlelength=1.6, borderaxespad=0.3, labelspacing=0.35)
    fig.savefig(path, metadata={"Software": None})


def drawdown_chart(path: Path, returns: pd.Series, title: str, subtitle: str) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1.0  # as qcore.backtest measures it
    worst_day, worst = drawdown.idxmin(), float(drawdown.min())
    fig, ax = _axes(title, subtitle)
    ax.fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color=INK, alpha=0.10, linewidth=0)
    ax.plot(drawdown.index, drawdown.to_numpy(), color=INK, linewidth=0.8)
    ax.set_ylim(min(worst, KILL_MAX_DRAWDOWN) * 1.18, 0.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    _time_axis(ax, drawdown.index[0], drawdown.index[-1], pd.Timestamp(OOS_SPLIT),
               labels_on_top=False)  # the top of this chart is where the data is
    ax.axhline(KILL_MAX_DRAWDOWN, color=STOP, linewidth=1.3)
    ax.annotate(f"registered stop level: {KILL_MAX_DRAWDOWN:.0%}", xy=(0.0, KILL_MAX_DRAWDOWN),
                xycoords=("axes fraction", "data"), xytext=(4, -5), textcoords="offset points",
                va="top", fontsize=9, color=INK)
    ax.plot([worst_day], [worst], marker="o", markersize=6, markerfacecolor=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.3, linestyle="none", zorder=6)
    late = (worst_day - drawdown.index[0]) > 0.5 * (drawdown.index[-1] - drawdown.index[0])
    ax.annotate(f"deepest: {worst:.1%} on {worst_day:%Y-%m-%d}", xy=(worst_day, worst),
                xytext=(-9 if late else 9, -2), textcoords="offset points",
                ha="right" if late else "left", va="center", fontsize=9, color=INK)
    fig.savefig(path, metadata={"Software": None})


def write_charts(out: Path, results: dict, blend_returns: pd.Series, summary: dict) -> None:
    import matplotlib.style

    note = "SYNTHETIC SAMPLE DATA. " if summary["synthetic_sample_data"] else ""
    basis = (f"Daily backtest to {summary['price_cache']['last_session']}, after modelled costs "
             "and dividend withholding. Log scale.")
    record = (f"Kept as a record of what was tried, not as results. Daily backtest to "
              f"{summary['price_cache']['last_session']}, after modelled costs. Log scale.")

    def curves(keys):
        return [(STRATEGIES[key][0], results[key]["equity"], colour, 1.5)
                for key, colour in zip(keys, HUES)]

    blend_curve = ("Blend of the four", (1.0 + blend_returns).cumprod(), INK, 2.3)
    # default style inside the block only: the caller's rcParams come back on exit
    with matplotlib.style.context("default"):
        growth_chart(out / "equity_production.png", [blend_curve, *curves(KEPT)],
                     f"{note}Growth of $1: the four kept strategies and their blend", basis)
        drawdown_chart(out / "drawdown_ensemble.png", blend_returns,
                       f"{note}Blend of the four kept strategies: drawdown from the running peak",
                       f"Daily backtest to {summary['price_cache']['last_session']}, after "
                       "modelled costs. The red line is the drawdown at which the stop rule fires.")
        growth_chart(out / "equity_rejected.png", curves(REJECTED),
                     f"{note}Growth of $1: the four rejected strategies", record)


# --------------------------------------------------------------------- main
def _is_sample(cache: Path) -> bool:
    try:
        return (cache / SAMPLE_README).read_text(encoding="utf-8").splitlines()[:1] == [SAMPLE_MARKER]
    except (OSError, UnicodeDecodeError):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_results_summary.py", allow_abbrev=False,
        description="Write the results table and charts for the eight strategies and the blend.")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: examples/qcore/ under the project root); "
                             "a directory inside results/ is refused")
    args = parser.parse_args(argv)
    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    if out == RESULTS.resolve() or RESULTS.resolve() in out.parents:
        print("summary not written: results/ holds saved records; name another directory with --out",
              file=sys.stderr)
        return EXIT_REFUSED
    try:
        prices = qdata.load_prices()
    except FileNotFoundError as exc:  # no price cache: the loader's one line
        print(exc, file=sys.stderr)
        return EXIT_NO_DATA
    synthetic = _is_sample(qdata.DATA_DIR)
    if synthetic and args.out is None:
        print("summary not written: the price cache is the synthetic sample, and the default "
              "output directory holds the market-data results; name another directory with --out",
              file=sys.stderr)
        return EXIT_REFUSED

    results = run_strategies()
    all_metrics = {key: metrics(result) for key, result in results.items()}
    blend, streams = run_blend(all_metrics)
    summary = build_summary(results, all_metrics, blend, prices, synthetic)

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    write_charts(out, results, streams["ENSEMBLE"], summary)

    print(f"{'':26s}{'Sharpe full / in / out':>24s}{'CAGR':>9s}{'max DD':>9s}  status")
    for row in (*summary["strategies"], summary["ensemble"]):
        sharpes = " / ".join(_num(row[part]["sharpe"]) for part in ("full", "in_sample", "out_of_sample"))
        print(f"{row.get('key', 'ensemble'):26s}{sharpes:>24s}{_pct(row['full']['cagr']):>9s}"
              f"{_pct(row['full']['maxdd']):>9s}  {row['status']}")
    print(f"wrote {', '.join(FILES)} to {out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
