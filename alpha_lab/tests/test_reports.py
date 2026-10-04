"""Tests for the tearsheet generator (alpha_lab.reports.performance)."""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.results import BacktestResult, WalkForwardWindow
from alpha_lab.reports import performance
from alpha_lab.reports.performance import generate_report
from alpha_lab.risk.metrics import summary

TICKERS = ["AAA", "BBB", "CCC"]


def make_result(dates: pd.DatetimeIndex, seed: int = 42, with_windows: bool = True) -> BacktestResult:
    """Deterministic BacktestResult with plausible turnover and costs."""
    n = len(dates)
    rng = np.random.default_rng(seed)
    gross = pd.Series(rng.normal(5e-4, 0.01, n), index=dates)
    costs = pd.Series(rng.uniform(2e-5, 2e-4, n), index=dates)
    net = gross - costs
    turnover = pd.Series(rng.uniform(0.05, 0.35, n), index=dates)
    holdings = pd.DataFrame(rng.normal(0.0, 0.1, (n, len(TICKERS))), index=dates, columns=TICKERS)
    windows = None
    if with_windows and n >= 200:
        windows = [
            WalkForwardWindow(dates[0], dates[99], dates[105], dates[199]),
            WalkForwardWindow(dates[100], dates[199], dates[205], dates[-1]),
        ]
    return BacktestResult(
        gross_returns=gross,
        costs=costs,
        net_returns=net,
        turnover=turnover,
        holdings=holdings,
        target_weights=holdings.copy(),
        windows=windows,
    )


METRICS = {
    "ann_return": 0.1234,
    "sharpe": 1.5678,
    "max_drawdown": -0.0876,
    "ann_vol": 0.1512,
    "cost_drag": 0.0123,
    "n_trades": 812,
}


@pytest.fixture()
def result_400(market_simple):
    # only the fixture's date index is used; the fixture itself is untouched
    return make_result(market_simple.dates)


def test_html_report_contents(tmp_path, result_400):
    outputs = generate_report(result_400, METRICS, tmp_path, title="demo tearsheet")
    assert set(outputs) == {"html", "md"}
    html_path = outputs["html"]
    assert html_path.exists()
    text = html_path.read_text(encoding="utf-8")
    # 400 days > both rolling windows: equity + sharpe + turnover figures inline
    assert text.count("data:image/png;base64,") == 3
    assert "demo tearsheet" in text
    assert "1.568" in text        # sharpe, 4 significant digits
    assert "12.34%" in text       # ann_return rendered as percent
    assert "-8.76%" in text       # max_drawdown rendered as percent
    assert "Walk-forward windows" in text


def test_md_report_and_pngs(tmp_path, result_400):
    outputs = generate_report(result_400, METRICS, tmp_path)
    md_path = outputs["md"]
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "| metric | value |" in md
    assert "![equity](equity.png)" in md
    for name in ("equity.png", "rolling_sharpe.png", "turnover_costs.png"):
        assert (tmp_path / name).exists(), name


def test_short_series_skips_rolling_figures(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=30)
    result = make_result(dates, seed=1, with_windows=False)
    outputs = generate_report(result, METRICS, tmp_path)
    assert outputs["html"].exists()
    html = outputs["html"].read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == 1  # equity only
    # 30 days < both rolling windows: those PNGs must not be written
    assert not (tmp_path / "rolling_sharpe.png").exists()
    assert not (tmp_path / "turnover_costs.png").exists()
    assert (tmp_path / "equity.png").exists()


def test_md_only_writes_no_html(tmp_path, result_400):
    outputs = generate_report(result_400, METRICS, tmp_path, formats=("md",))
    assert set(outputs) == {"md"}
    assert (tmp_path / "report.md").exists()
    assert not (tmp_path / "report.html").exists()
    assert (tmp_path / "equity.png").exists()


def test_empty_metrics_and_no_windows(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=150)
    result = make_result(dates, seed=5, with_windows=False)
    outputs = generate_report(result, {}, tmp_path)
    html = outputs["html"].read_text(encoding="utf-8")
    assert "No metrics provided" in html
    assert "Walk-forward windows" not in html


def test_unknown_format_raises(tmp_path, result_400):
    with pytest.raises(ConfigError):
        generate_report(result_400, METRICS, tmp_path, formats=("pdf",))


def test_empty_result_raises(tmp_path, result_400):
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    result = BacktestResult(
        gross_returns=empty,
        costs=empty,
        net_returns=empty,
        turnover=empty,
        holdings=pd.DataFrame(),
        target_weights=pd.DataFrame(),
    )
    with pytest.raises(DataError):
        generate_report(result, METRICS, tmp_path)


def test_walkforward_charts_and_months_use_metric_evaluation_period(tmp_path, monkeypatch):
    dates = pd.bdate_range("2023-01-02", "2024-09-30")
    start = pd.Timestamp("2024-01-15")
    result = make_result(dates, with_windows=False)
    result.windows = [WalkForwardWindow(dates[0], pd.Timestamp("2023-12-29"), start, dates[-1])]
    result.meta["mode"] = "walkforward"
    result.config = {"data": {"source": "synthetic"}}
    # Deliberately large pre-test values reveal accidental inclusion in the
    # curves, rolling statistics, cumulative costs, and monthly compounding.
    for series in (result.net_returns, result.gross_returns, result.turnover, result.costs):
        series.loc[series.index < start] = 0.5
    original = result.summary_frame().copy(deep=True)
    plotted = {}

    def capture_figure(fig):
        plotted[fig.axes[0].get_title()] = [
            [(line.get_xdata().copy(), line.get_ydata().copy()) for line in ax.lines]
            for ax in fig.axes
        ]
        return b"test figure"

    monkeypatch.setattr(performance, "_render_png", capture_figure)
    metrics = summary(result)
    outputs = generate_report(result, metrics, tmp_path)
    active = original.loc[start:]
    assert pd.Timestamp(metrics["start"]) == active.index[0]
    assert pd.Timestamp(metrics["end"]) == active.index[-1]

    equity = plotted["Equity curve — net vs gross"][0]
    for (x, y), field in zip(equity, ("net", "gross")):
        pd.testing.assert_index_equal(pd.DatetimeIndex(x), active.index)
        np.testing.assert_allclose(y, (1.0 + active[field]).cumprod())
    sharpe_x, sharpe_y = plotted["Rolling 126d Sharpe — net"][0][0]
    pd.testing.assert_index_equal(pd.DatetimeIndex(sharpe_x), active.index)
    expected_sharpe = active.net.rolling(126).mean() / active.net.rolling(126).std() * np.sqrt(252)
    np.testing.assert_allclose(sharpe_y, expected_sharpe, equal_nan=True)
    turnover_axes = plotted["Turnover and cumulative costs"]
    for (x, y), expected in (
        (turnover_axes[0][0], active.turnover.rolling(63).mean()),
        (turnover_axes[1][0], active.costs.cumsum()),
    ):
        pd.testing.assert_index_equal(pd.DatetimeIndex(x), active.index)
        np.testing.assert_allclose(y, expected, equal_nan=True)

    january = (1 + active.loc["2024-01", "net"]).prod() - 1
    for fmt, path in outputs.items():
        text = path.read_text(encoding="utf-8")
        assert "2024-01-15 through 2024-09-30" in text
        assert "training prefix before the first walk-forward test date is excluded" in text
        assert "not evidence of investment performance" in text
        monthly_section = text.split("Monthly net returns", 1)[1].split("Walk-forward windows", 1)[0]
        assert "2023" not in monthly_section
        assert f"{january * 100:.2f}%" in monthly_section
    pd.testing.assert_frame_equal(result.summary_frame(), original)


@pytest.mark.parametrize("mode", ["insample", ""])
def test_non_walkforward_report_keeps_full_result(tmp_path, monkeypatch, mode):
    dates = pd.bdate_range("2023-12-01", periods=40)
    result = make_result(dates, with_windows=False)
    # Window metadata alone must not silently alter the sample used by the
    # metrics. In-sample and unspecified modes retain the complete result.
    result.windows = [WalkForwardWindow(dates[0], dates[19], dates[20], dates[-1])]
    result.meta["mode"] = mode
    plotted_dates = []

    def capture_figure(fig):
        plotted_dates.extend(fig.axes[0].lines[0].get_xdata())
        return b"test figure"

    monkeypatch.setattr(performance, "_render_png", capture_figure)
    outputs = generate_report(result, summary(result), tmp_path)
    pd.testing.assert_index_equal(pd.DatetimeIndex(plotted_dates), dates)
    for path in outputs.values():
        text = path.read_text(encoding="utf-8")
        assert "2023-12-01 through 2024-01-25" in text
        assert "training prefix" not in text
        assert "not evidence of investment performance" not in text
        if mode == "insample":
            assert "full-sample, in-sample diagnostic" in text
        monthly_section = text.split("Monthly net returns", 1)[1].split("Walk-forward windows", 1)[0]
        assert "2023" in monthly_section


def test_equity_drawdown_counts_loss_from_initial_capital(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=2)
    net = pd.Series([-0.05, -0.05], index=dates)
    plotted_drawdown = []
    fill_between = matplotlib.axes.Axes.fill_between

    def capture_fill(ax, x, y1, *args, **kwargs):
        plotted_drawdown.extend(y1)
        return fill_between(ax, x, y1, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", capture_fill)
    png = performance._fig_equity(net, net)
    assert png.startswith(b"\x89PNG")
    np.testing.assert_allclose(plotted_drawdown, [-0.05, -0.0975])


# --------------------------------------------------------------------------
# value formatting
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key,value,expected",
    [
        # counts are exact at every size; .4g gave '1e+04', '1.26e+04', '8.123e+04'
        ("n_days", 1003, "1003"),
        ("n_days", 10000, "10000"),
        ("n_days", 12600, "12600"),
        ("n_trades", np.int64(81234), "81234"),
        ("n_obs", 12345678901234567, "12345678901234567"),  # beyond float precision
        ("n_trials", 2.5, "2.5"),
        # large floats keep their integer part instead of an exponent
        ("turnover_ann", 12345.678, "12346"),
        ("turnover_ann", 9999.4, "9999"),
        ("turnover_ann", 86.2767, "86.28"),
        ("sharpe", 1.5678, "1.568"),
        ("sharpe", -0.03394, "-0.03394"),
        # rate-like keys stay percentages, integers included
        ("ann_return", 0.1234, "12.34%"),
        ("max_drawdown", 0, "0%"),
        ("ann_return", 123.456, "12346%"),
        # non-finite, missing and non-numeric values
        ("sharpe", float("nan"), "nan"),
        ("sharpe", None, "nan"),
        ("calmar", float("inf"), "inf"),
        ("mode", "walkforward", "walkforward"),
        ("flag", True, "True"),
        # tiny values keep the exponent form (fixed point would print zeros)
        ("turnover_daily_mean", 1.2345e-05, "1.234e-05"),
    ],
)
def test_format_metric(key, value, expected):
    assert performance.format_metric(key, value) == expected


def test_long_run_counts_are_not_printed_in_scientific_notation(tmp_path):
    dates = pd.bdate_range("1975-01-02", periods=12600)
    result = make_result(dates, with_windows=False)
    outputs = generate_report(result, summary(result), tmp_path)
    assert "| n_days | 12600 |" in outputs["md"].read_text(encoding="utf-8")
    assert "<td>n_days</td><td>12600</td>" in outputs["html"].read_text(encoding="utf-8")


def test_markdown_escapes_pipes_newlines_and_html(tmp_path, result_400):
    metrics = {"note": "a|b", "multi": "x\ny", "<b>k</b>": 1.0, "path": "C:\\dir|x"}
    title = "a|b <script>alert(1)</script> & co"
    outputs = generate_report(result_400, metrics, tmp_path, title=title)
    lines = outputs["md"].read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# a\\|b &lt;script&gt;alert(1)&lt;/script&gt; &amp; co"
    table = lines[lines.index("| metric | value |"):][:6]
    assert table[2:] == [
        "| note | a\\|b |",
        "| multi | x y |",
        "| &lt;b&gt;k&lt;/b&gt; | 1 |",
        "| path | C:\\\\dir\\|x |",
    ]
    # every metric row still has exactly two cells
    for row in table[2:]:
        assert len(row.replace("\\\\", "").replace("\\|", "").split("|")) == 4
    assert "&lt;script&gt;" in outputs["html"].read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# one build per report directory
# --------------------------------------------------------------------------

def test_rebuild_removes_outputs_it_no_longer_writes(tmp_path, result_400):
    generate_report(result_400, METRICS, tmp_path)  # html + md + three figures
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")
    short = make_result(pd.bdate_range("2024-01-02", periods=30), seed=1, with_windows=False)

    outputs = generate_report(short, METRICS, tmp_path, formats=("md",))

    assert set(outputs) == {"md"}
    # only what this build wrote is left: no report or chart of the first result
    assert sorted(p.name for p in tmp_path.iterdir()) == ["equity.png", "notes.txt", "report.md"]
    md = outputs["md"].read_text(encoding="utf-8")
    assert "rolling_sharpe" not in md and "turnover_costs" not in md

    html_only = generate_report(short, METRICS, tmp_path, formats=("html",))
    assert set(html_only) == {"html"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt", "report.html"]


def test_rebuild_removes_nothing_but_its_own_report_files(tmp_path, result_400):
    """The clean-up is limited to the five file names a build can write,
    directly inside the report directory it was given."""
    assert {*performance._REPORT_FILES.values(), *performance._FIGURE_FILES} == {
        "report.html", "report.md", "equity.png", "rolling_sharpe.png", "turnover_costs.png",
    }
    run = tmp_path / "run"
    out = run / "report"
    generate_report(result_400, METRICS, out)  # html + md + three figures
    bystanders = [
        # the run directory around the report directory
        run / "report.md", run / "report.html", run / "equity.png", run / "metrics.json",
        # other files inside the report directory, similar names included
        out / "notes.txt", out / "report.md.bak", out / "report.htm", out / "equity.png.txt",
        out / "my_equity.png", out / "rolling_sharpe_252.png",
        # the same five names one level down
        out / "figures" / "equity.png", out / "figures" / "report.html",
    ]
    for path in bystanders:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep me")
    short = make_result(pd.bdate_range("2024-01-02", periods=30), seed=1, with_windows=False)

    generate_report(short, METRICS, out, formats=("html",))

    left = sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file())
    assert left == sorted([*(p.relative_to(run).as_posix() for p in bystanders), "report/report.html"])
    assert all(path.read_text() == "keep me" for path in bystanders)


# --------------------------------------------------------------------------
# period note
# --------------------------------------------------------------------------

def test_forced_run_is_flagged_in_both_reports(tmp_path, result_400):
    result_400.meta["validation"] = {"errors": 3, "warnings": 1, "forced": True}
    for path in generate_report(result_400, METRICS, tmp_path).values():
        text = path.read_text(encoding="utf-8")
        assert "validation reported 3 error(s) and the run was forced" in text
    result_400.meta["validation"] = {"errors": 0, "warnings": 1, "forced": False}
    for path in generate_report(result_400, METRICS, tmp_path).values():
        assert "forced" not in path.read_text(encoding="utf-8")


def test_insample_report_discloses_the_flat_warm_up(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=60)
    result = make_result(dates, with_windows=False)
    result.meta["mode"] = "insample"
    result.holdings.iloc[:25] = 0.0
    for path in generate_report(result, {}, tmp_path).values():
        text = path.read_text(encoding="utf-8")
        assert "Its first 25 of 60 days precede the first position" in text
        assert "count as flat days in every metric" in text
    result.holdings.iloc[:] = 0.0
    for path in generate_report(result, {}, tmp_path).values():
        text = path.read_text(encoding="utf-8")
        assert "No position is held on any date in this period." in text
        assert "precede the first position" not in text


def test_fully_invested_report_has_no_flat_day_sentence(tmp_path, result_400):
    result_400.meta["mode"] = "insample"
    for path in generate_report(result_400, METRICS, tmp_path).values():
        text = path.read_text(encoding="utf-8")
        assert "precede the first position" not in text
        assert "No position is held" not in text


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _captured_figures(monkeypatch):
    figures = []

    def capture(fig):
        figures.append(fig)
        return b"test figure"

    monkeypatch.setattr(performance, "_render_png", capture)
    return figures


def _tick_labels(axis):
    formatter = axis.get_major_formatter()
    low, high = axis.get_view_interval()
    return [formatter(v, i) for i, v in enumerate(axis.get_majorticklocs()) if low <= v <= high]


def test_equity_axis_has_plain_labels_on_both_sides_of_one(monkeypatch):
    figures = _captured_figures(monkeypatch)
    dates = pd.bdate_range("2024-01-02", periods=5)
    net = pd.Series([-0.10, -0.15, 0.20, 0.25, 0.20], index=dates)  # equity 0.765 .. 1.377
    performance._fig_equity(net, net)
    axis = figures[0].axes[0].yaxis
    assert axis.get_scale() == "log"
    # the default log formatter gave '8 x 10^-1', '9 x 10^-1', '10^0' and
    # no labelled tick at all above 1
    assert _tick_labels(axis) == ["0.8", "0.9", "1", "1.1", "1.2", "1.3", "1.4"]
    assert len(axis.get_minorticklocs()) == 0


def test_constant_equity_curve_gets_a_single_tick(monkeypatch):
    figures = _captured_figures(monkeypatch)
    flat = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=5))
    performance._fig_equity(flat, flat)
    assert _tick_labels(figures[0].axes[0].yaxis) == ["1"]


def test_equity_axis_over_several_decades_uses_one_two_five_ticks(monkeypatch):
    figures = _captured_figures(monkeypatch)
    dates = pd.bdate_range("2024-01-02", periods=6)
    net = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=dates)    # equity 2 .. 64
    performance._fig_equity(net, net)
    assert _tick_labels(figures[0].axes[0].yaxis) == ["2", "5", "10", "20", "50"]


def test_turnover_legend_sits_below_the_axes(monkeypatch, result_400):
    figures = _captured_figures(monkeypatch)
    performance._fig_turnover_costs(result_400.turnover, result_400.costs)
    fig = figures[0]
    FigureCanvasAgg(fig).draw()
    axes_box = fig.axes[0].get_window_extent()
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["63d mean turnover", "cumulative costs"]
    # entirely under the plot area: it cannot cover either line
    assert legend.get_window_extent().y1 <= axes_box.y0


def test_report_module_leaves_the_matplotlib_backend_alone():
    """Importing the report module and drawing a figure must not switch the
    backend an interactive session has already chosen."""
    code = (
        "import sys\n"
        "import alpha_lab.reports\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'the report module imported pyplot'\n"
        "import matplotlib.pyplot as plt\n"
        "plt.figure()\n"
        "before = plt.get_backend()\n"
        "import pandas as pd\n"
        "from alpha_lab.reports import performance\n"
        "r = pd.Series([0.01, -0.02, 0.03], index=pd.bdate_range('2024-01-02', periods=3))\n"
        "assert performance._fig_equity(r, r).startswith(b'\\x89PNG')\n"
        "assert plt.get_backend() == before == 'svg', (before, plt.get_backend())\n"
        "assert len(plt.get_fignums()) == 1, 'a report figure was registered with pyplot'\n"
    )
    package_root = str(Path(performance.__file__).resolve().parents[2])
    env = dict(os.environ, MPLBACKEND="svg")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
