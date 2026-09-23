"""Tearsheet generation from a BacktestResult.

Produces a single self-contained ``report.html`` (inline CSS, every figure
embedded as a base64 data URI — zero external assets) and/or a ``report.md``
with the figures written alongside as PNG files and referenced relatively.
"""

from __future__ import annotations

# Headless backend: reports are generated in tests/CI/cron with no display
# server, and matplotlib's backend must be selected BEFORE pyplot is imported.
import matplotlib

matplotlib.use("Agg")

import base64
import calendar
import html as _htmlmod
import io
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.results import BacktestResult, WalkForwardWindow
from alpha_lab.core.types import TRADING_DAYS_PER_YEAR
from alpha_lab.risk.metrics import drawdown_series

#: rolling windows (trading days) for the diagnostic figures
ROLLING_SHARPE_WINDOW = 126
TURNOVER_WINDOW = 63

_SUPPORTED_FORMATS = ("html", "md")
#: metric keys containing any of these substrings are rendered as percentages
_PERCENT_TOKENS = ("return", "drawdown", "vol", "drag")


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def format_metric(key: str, value: Any) -> str:
    """Format one metric value: 4 significant digits, percent for rate-like keys.

    A key is rate-like when it contains 'return', 'drawdown', 'vol' or 'drag'
    (case-insensitive). Non-numeric values pass through as ``str(value)``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    v = float(value)
    if not np.isfinite(v):
        return str(v)
    if any(tok in key.lower() for tok in _PERCENT_TOKENS):
        return f"{v * 100.0:.4g}%"
    return f"{v:.4g}"


def _metric_rows(metrics: dict) -> list[tuple[str, str]]:
    return [(str(k), format_metric(str(k), v)) for k, v in metrics.items()]


def _window_rows(windows: Sequence[WalkForwardWindow]) -> list[tuple]:
    return [
        (
            i + 1,
            w.train_start.date().isoformat(),
            w.train_end.date().isoformat(),
            w.test_start.date().isoformat(),
            w.test_end.date().isoformat(),
        )
        for i, w in enumerate(windows)
    ]


# --------------------------------------------------------------------------
# figures (each returns PNG bytes, or None when the series is too short)
# --------------------------------------------------------------------------

def _render_png(fig) -> bytes:
    """Serialize a figure to PNG bytes, always closing it (no leaks)."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


def _fig_equity(net: pd.Series, gross: pd.Series) -> bytes | None:
    """Net vs gross equity (log y) with the net drawdown shaded below."""
    net_eq = (1.0 + net.fillna(0.0)).cumprod()
    if len(net_eq) < 2:
        return None
    gross_eq = (1.0 + gross.fillna(0.0)).cumprod()
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, sharex=True, figsize=(9.0, 6.0), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_eq.plot(net_eq.index, net_eq.values, color="#1f77b4", lw=1.4, label="net (after costs)")
    ax_eq.plot(gross_eq.index, gross_eq.values, color="#666666", lw=1.0, ls="--", label="gross (before costs)")
    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("equity (log)")
    ax_eq.set_title("Equity curve — net vs gross")
    ax_eq.legend(loc="best")
    ax_eq.grid(True, alpha=0.3)
    drawdown = drawdown_series(net)
    ax_dd.fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.4)
    ax_dd.set_ylabel("drawdown")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_dd.grid(True, alpha=0.3)
    fig.tight_layout()
    return _render_png(fig)


def _fig_rolling_sharpe(net: pd.Series) -> bytes | None:
    """Rolling 126-day annualized Sharpe of net returns."""
    if len(net) < ROLLING_SHARPE_WINDOW:
        return None
    mean = net.rolling(ROLLING_SHARPE_WINDOW).mean()
    std = net.rolling(ROLLING_SHARPE_WINDOW).std()
    sharpe = (mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)).replace([np.inf, -np.inf], np.nan)
    if sharpe.dropna().empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.plot(sharpe.index, sharpe.values, color="#2ca02c", lw=1.3)
    ax.axhline(0.0, color="#666666", lw=0.8, ls="--")
    ax.set_ylabel("Sharpe (ann.)")
    ax.set_title(f"Rolling {ROLLING_SHARPE_WINDOW}d Sharpe — net")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _render_png(fig)


def _fig_turnover_costs(turnover: pd.Series, costs: pd.Series) -> bytes | None:
    """63-day rolling mean turnover with cumulative costs on a twin axis."""
    if len(turnover) < TURNOVER_WINDOW:
        return None
    rolled = turnover.rolling(TURNOVER_WINDOW).mean()
    if rolled.dropna().empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    line_to, = ax.plot(
        rolled.index, rolled.values, color="#1f77b4", lw=1.3,
        label=f"{TURNOVER_WINDOW}d mean turnover",
    )
    ax.set_ylabel("turnover (NAV frac/day)")
    ax.grid(True, alpha=0.3)
    ax_cost = ax.twinx()
    cum_costs = costs.fillna(0.0).cumsum()
    line_c, = ax_cost.plot(
        cum_costs.index, cum_costs.values, color="#d62728", lw=1.3, label="cumulative costs"
    )
    ax_cost.set_ylabel("cumulative cost (NAV frac)")
    ax.legend(handles=[line_to, line_c], loc="best")
    ax.set_title("Turnover and cumulative costs")
    fig.tight_layout()
    return _render_png(fig)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def monthly_returns(net: pd.Series) -> pd.DataFrame | None:
    """Net returns compounded per calendar month, pivoted year x month."""
    clean = net.dropna()
    if clean.empty:
        return None
    compounded = (1.0 + clean).groupby([clean.index.year, clean.index.month]).prod() - 1.0
    table = compounded.unstack(level=-1).reindex(columns=range(1, 13))
    table.columns = [calendar.month_abbr[m] for m in table.columns]
    table.index.name = "year"
    return table


def _monthly_cell(value: float) -> str:
    return "" if pd.isna(value) else f"{value * 100.0:.2f}%"


def _monthly_cell_style(value: float, vmax: float) -> str:
    """Green/red background scaled by |monthly return| relative to the max."""
    if pd.isna(value):
        return ""
    alpha = 0.15 + 0.55 * min(abs(value) / vmax, 1.0) if vmax > 0 else 0.25
    rgb = "46,160,67" if value >= 0 else "218,54,51"
    return f"background-color:rgba({rgb},{alpha:.2f});"


def _html_table(headers: Iterable, rows: list[tuple], styles: list[list[str]] | None = None) -> str:
    head = "".join(f"<th>{_htmlmod.escape(str(h))}</th>" for h in headers)
    body_rows = []
    for i, row in enumerate(rows):
        cells = []
        for j, cell in enumerate(row):
            style = styles[i][j] if styles else ""
            attr = f' style="{style}"' if style else ""
            cells.append(f"<td{attr}>{_htmlmod.escape(str(cell))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _md_table(headers: Iterable, rows: list[tuple]) -> str:
    headers = [str(h) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _monthly_headers_rows(table: pd.DataFrame) -> tuple[list, list[tuple], list[list[str]]]:
    values = table.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.0
    headers = ["year", *table.columns]
    rows, styles = [], []
    for year, row in table.iterrows():
        rows.append((year, *[_monthly_cell(v) for v in row]))
        styles.append(["", *[_monthly_cell_style(v, vmax) for v in row]])
    return headers, rows, styles


_WINDOW_HEADERS = ("#", "train_start", "train_end", "test_start", "test_end")


# --------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       max-width: 960px; margin: 2em auto; padding: 0 1em; color: #1c2733; }
h1 { border-bottom: 2px solid #d0d7de; padding-bottom: 0.3em; }
h2 { margin-top: 1.6em; color: #24425c; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.9em; }
th, td { border: 1px solid #d0d7de; padding: 4px 10px; text-align: right; }
th { background: #f0f3f6; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; height: auto; display: block; margin: 0.8em 0; }
.note { color: #6a737d; font-style: italic; }
"""

_FIGURE_TITLES = {
    "equity": "Equity and drawdown",
    "rolling_sharpe": f"Rolling {ROLLING_SHARPE_WINDOW}d Sharpe",
    "turnover_costs": "Turnover and costs",
}


def _build_html(
    title: str,
    metrics: dict,
    figures: dict[str, bytes],
    monthly: pd.DataFrame | None,
    windows: Sequence[WalkForwardWindow] | None,
    period_note: str,
) -> str:
    esc_title = _htmlmod.escape(title)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{esc_title}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        f"<h1>{esc_title}</h1>",
        f'<p class="note">{_htmlmod.escape(period_note)}</p>',
        "<h2>Key metrics</h2>",
    ]
    if metrics:
        parts.append(_html_table(("metric", "value"), _metric_rows(metrics)))
    else:
        parts.append('<p class="note">No metrics provided.</p>')

    for name, png in figures.items():
        b64 = base64.b64encode(png).decode("ascii")
        parts.append(f"<h2>{_htmlmod.escape(_FIGURE_TITLES.get(name, name))}</h2>")
        parts.append(f'<img alt="{_htmlmod.escape(name)}" src="data:image/png;base64,{b64}"/>')

    if monthly is not None:
        headers, rows, styles = _monthly_headers_rows(monthly)
        parts.append("<h2>Monthly net returns</h2>")
        parts.append(_html_table(headers, rows, styles))

    if windows:
        parts.append("<h2>Walk-forward windows</h2>")
        parts.append(_html_table(_WINDOW_HEADERS, _window_rows(windows)))

    parts.append("</body></html>")
    return "\n".join(parts)


def _build_md(
    title: str,
    metrics: dict,
    figure_files: dict[str, str],
    monthly: pd.DataFrame | None,
    windows: Sequence[WalkForwardWindow] | None,
    period_note: str,
) -> str:
    parts = [f"# {title}", "", period_note, "", "## Key metrics", ""]
    if metrics:
        parts.append(_md_table(("metric", "value"), _metric_rows(metrics)))
    else:
        parts.append("_No metrics provided._")

    for name, filename in figure_files.items():
        parts += ["", f"## {_FIGURE_TITLES.get(name, name)}", "", f"![{name}]({filename})"]

    if monthly is not None:
        headers, rows, _ = _monthly_headers_rows(monthly)
        parts += ["", "## Monthly net returns", "", _md_table(headers, rows)]

    if windows:
        parts += ["", "## Walk-forward windows", "", _md_table(_WINDOW_HEADERS, _window_rows(windows))]

    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def generate_report(
    result: BacktestResult,
    metrics: dict,
    out_dir: str | Path,
    formats: Sequence[str] = ("html", "md"),
    title: str = "alpha_lab tearsheet",
) -> dict[str, Path]:
    """Write a tearsheet for ``result`` into ``out_dir``.

    ``formats`` selects the outputs: 'html' writes one self-contained
    ``report.html`` (figures inlined as base64 data URIs); 'md' writes
    ``report.md`` plus the figure PNGs alongside it. Figures whose rolling
    window exceeds the series length are skipped, not errors. Returns a
    mapping of format -> written report path. Walk-forward charts and tables
    start at the first test date, matching ``risk.metrics.summary``; the
    training prefix remains available in the original result and saved files.
    """
    fmts = (formats,) if isinstance(formats, str) else tuple(formats)
    if not fmts:
        raise ConfigError("no report formats requested")
    unknown = set(fmts) - set(_SUPPORTED_FORMATS)
    if unknown:
        raise ConfigError(
            f"unknown report formats {sorted(unknown)}; supported: {list(_SUPPORTED_FORMATS)}"
        )
    if result.net_returns is None or len(result.net_returns) == 0:
        raise DataError("cannot build a tearsheet: result.net_returns is empty")

    net, gross = result.net_returns, result.gross_returns
    turnover, costs = result.turnover, result.costs
    is_walkforward = result.meta.get("mode") == "walkforward" and bool(result.windows)
    if is_walkforward:
        # Keep this boundary consistent with risk.metrics.summary. Slicing
        # local references preserves the full result for persistence/inspection.
        active_start = result.windows[0].test_start
        net, gross = net.loc[active_start:], gross.loc[active_start:]
        turnover = turnover.loc[active_start:] if turnover is not None else None
        costs = costs.loc[active_start:] if costs is not None else None
    if net.empty:
        raise DataError("cannot build a tearsheet: evaluation period has no returns")

    period_note = (
        f"Charts and monthly returns cover {net.index[0].date().isoformat()} "
        f"through {net.index[-1].date().isoformat()}. "
    )
    if is_walkforward:
        period_note += "The training prefix before the first walk-forward test date is excluded. "
    elif result.meta.get("mode") == "insample":
        period_note += "This is a full-sample, in-sample diagnostic. "
    period_note += "Partial months include only dates within this period."
    if (result.config or {}).get("data", {}).get("source") == "synthetic":
        period_note = (
            "Synthetic data demonstrate the research pipeline; these results are not "
            "evidence of investment performance. " + period_note
        )

    metrics = metrics or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures: dict[str, bytes] = {}
    equity_png = _fig_equity(net, gross)
    if equity_png is not None:
        figures["equity"] = equity_png
    sharpe_png = _fig_rolling_sharpe(net)
    if sharpe_png is not None:
        figures["rolling_sharpe"] = sharpe_png
    if turnover is not None and costs is not None:
        to_png = _fig_turnover_costs(turnover, costs)
        if to_png is not None:
            figures["turnover_costs"] = to_png

    monthly = monthly_returns(net)
    windows = result.windows

    outputs: dict[str, Path] = {}
    if "html" in fmts:
        path = out / "report.html"
        path.write_text(_build_html(title, metrics, figures, monthly, windows, period_note), encoding="utf-8")
        outputs["html"] = path
    if "md" in fmts:
        figure_files: dict[str, str] = {}
        for name, png in figures.items():
            filename = f"{name}.png"
            (out / filename).write_bytes(png)
            figure_files[name] = filename
        path = out / "report.md"
        path.write_text(_build_md(title, metrics, figure_files, monthly, windows, period_note), encoding="utf-8")
        outputs["md"] = path
    return outputs
