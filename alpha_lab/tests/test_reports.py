"""Tests for the tearsheet generator (alpha_lab.reports.performance)."""

import numpy as np
import pandas as pd
import pytest

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
    assert text.count("data:image/png;base64,") >= 2
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
        performance.plt.close(fig)
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
        performance.plt.close(fig)
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
    fill_between = performance.plt.Axes.fill_between

    def capture_fill(ax, x, y1, *args, **kwargs):
        plotted_drawdown.extend(y1)
        return fill_between(ax, x, y1, *args, **kwargs)

    monkeypatch.setattr(performance.plt.Axes, "fill_between", capture_fill)
    png = performance._fig_equity(net, net)
    assert png.startswith(b"\x89PNG")
    np.testing.assert_allclose(plotted_drawdown, [-0.05, -0.0975])
