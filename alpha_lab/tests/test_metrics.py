"""Tests for alpha_lab.risk.metrics: hand-computed values on toy series."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.core.results import BacktestResult
from alpha_lab.risk import metrics as m


def _series(values, start="2020-01-06"):
    values = list(values)
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --------------------------------------------------------------------------
# point metrics on toy series
# --------------------------------------------------------------------------

class TestPointMetrics:
    def test_ann_return_constant(self):
        r = _series([0.001] * 252)
        assert m.ann_return(r) == pytest.approx(1.001**252 - 1.0, rel=1e-12)
        # annualization exponent: half a year of the same drift, same answer
        half = _series([0.001] * 126)
        assert m.ann_return(half) == pytest.approx(1.001**252 - 1.0, rel=1e-9)

    def test_sharpe_formula(self):
        rng = np.random.default_rng(42)
        r = _series(rng.normal(0.0005, 0.01, 400))
        expected = float(r.mean()) / float(r.std(ddof=1)) * math.sqrt(252)
        assert m.sharpe(r) == pytest.approx(expected, rel=1e-12)
        assert m.ann_vol(r) == pytest.approx(float(r.std(ddof=1)) * math.sqrt(252), rel=1e-12)

    def test_alternating_hit_rate_and_tiny_ann_return(self):
        r = _series([0.01, -0.01] * 126)  # 252 days, equal up/down counts
        assert m.hit_rate(r) == pytest.approx(0.5)
        ar = m.ann_return(r)
        # each +1%/-1% pair loses 1bp: slightly negative, tiny magnitude
        assert -0.02 < ar < 0.0
        assert ar == pytest.approx(0.9999**126 - 1.0, rel=1e-9)

    def test_sortino_is_target_zero_downside_deviation(self):
        # Regression: the old denominator (std of negative days around their
        # own mean) exploded on consistently sized losses.
        r = _series([-0.01] * 50 + [0.02] * 50 + [-0.0100001] * 10)
        downside = np.minimum(r.to_numpy(), 0.0)
        dd = math.sqrt(float(np.mean(downside**2)))
        expected = float(r.mean()) / dd * math.sqrt(252)
        assert m.sortino(r) == pytest.approx(expected, rel=1e-12)
        assert m.sortino(r) < 100.0  # the old formula gave ~1.5e6 here

    def test_zero_vol_sharpe_is_nan_not_raise(self):
        r = _series([0.001] * 100)
        assert math.isnan(m.sharpe(r))
        assert math.isnan(m.sharpe_se(r))
        assert math.isnan(m.psr(r))
        assert math.isnan(m.skewness(r))
        assert math.isnan(m.kurtosis(r))

    def test_degenerate_inputs_nan(self):
        empty = pd.Series(dtype=float)
        assert math.isnan(m.ann_return(empty))
        assert math.isnan(m.ann_vol(empty))
        assert math.isnan(m.sharpe(empty))
        assert math.isnan(m.hit_rate(empty))
        assert math.isnan(m.sortino(_series([0.01, 0.02, 0.03])))  # no down days
        assert math.isnan(m.cost_drag(empty))
        one = _series([0.01])
        assert math.isnan(m.sharpe(one))
        assert math.isnan(m.ann_vol(one))

    def test_moments_on_normal_ish_sample(self):
        rng = np.random.default_rng(7)
        r = _series(rng.normal(0.0, 0.01, 5000), start="2005-01-03")
        assert abs(m.skewness(r)) < 0.15
        assert m.kurtosis(r) == pytest.approx(3.0, abs=0.35)  # raw, not excess

    def test_sharpe_se_matches_formula(self):
        rng = np.random.default_rng(3)
        r = _series(rng.normal(0.001, 0.012, 300))
        sr = float(r.mean()) / float(r.std(ddof=1))
        skew, kurt = m.skewness(r), m.kurtosis(r)
        expected = math.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (len(r) - 1))
        assert m.sharpe_se(r) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# drawdown
# --------------------------------------------------------------------------

class TestDrawdown:
    def test_max_drawdown_exact_path(self):
        # equity: 1.10 (peak) -> 0.88 (trough, -20%) -> 0.968 -> 1.1132 (recovered)
        r = _series([0.10, -0.20, 0.10, 0.15])
        d = m.max_drawdown(r)
        assert d["depth"] == pytest.approx(-0.20, rel=1e-12)
        assert d["peak_date"] == r.index[0]
        assert d["trough_date"] == r.index[1]
        assert d["recovery_date"] == r.index[3]
        assert d["duration_days"] == 3

    def test_max_drawdown_unrecovered(self):
        r = _series([0.10, -0.20, 0.05])
        d = m.max_drawdown(r)
        assert d["depth"] == pytest.approx(-0.20, rel=1e-12)
        assert d["recovery_date"] is None
        assert d["duration_days"] == 2  # peak (day 0) to last date (day 2)

    def test_drawdown_from_inception(self):
        # equity never exceeds the starting 1.0 — drawdown vs inception peak
        r = _series([-0.05, -0.05])
        d = m.max_drawdown(r)
        assert d["depth"] == pytest.approx(0.95 * 0.95 - 1.0, rel=1e-12)
        assert d["trough_date"] == r.index[1]

    def test_no_drawdown(self):
        d = m.max_drawdown(_series([0.01, 0.02, 0.01]))
        assert d["depth"] == 0.0
        assert d["recovery_date"] is None

    def test_drawdown_series_values(self):
        dd = m.drawdown_series(_series([0.10, -0.20, 0.10, 0.15]))
        assert dd.iloc[0] == pytest.approx(0.0, abs=1e-15)
        assert dd.iloc[1] == pytest.approx(-0.20, rel=1e-12)
        assert dd.iloc[3] == pytest.approx(0.0, abs=1e-15)

    def test_calmar(self):
        r = _series([0.10, -0.20, 0.10, 0.15])
        assert m.calmar(r) == pytest.approx(m.ann_return(r) / 0.20, rel=1e-9)
        assert math.isnan(m.calmar(_series([0.01, 0.01])))  # zero drawdown

    def test_degenerate_drawdown(self):
        d = m.max_drawdown(pd.Series(dtype=float))
        assert math.isnan(d["depth"])
        assert d["peak_date"] is None and d["recovery_date"] is None


# --------------------------------------------------------------------------
# normal quantile, PSR, DSR
# --------------------------------------------------------------------------

class TestPsrDsr:
    @pytest.mark.parametrize(
        "p,z", [(0.95, 1.6449), (0.975, 1.9600), (0.99, 2.3263)]
    )
    def test_norm_ppf_accuracy(self, p, z):
        assert m.norm_ppf(p) == pytest.approx(z, abs=1e-3)
        assert m.norm_ppf(1.0 - p) == pytest.approx(-z, abs=1e-3)  # symmetry

    def test_norm_ppf_edges(self):
        assert m.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
        assert math.isnan(m.norm_ppf(0.0))
        assert math.isnan(m.norm_ppf(1.0))
        assert math.isnan(m.norm_ppf(-0.1))

    def test_norm_cdf_roundtrip(self):
        for p in (0.05, 0.3, 0.5, 0.9, 0.999):
            assert m.norm_cdf(m.norm_ppf(p)) == pytest.approx(p, abs=1e-8)

    def test_psr_bounds_and_increasing_in_T(self):
        # modest daily Sharpe (~0.02) so PSR does not saturate to 1.0 at T=480
        base = [0.010, -0.009, 0.008, -0.007, 0.002, -0.003]
        short = _series(base * 20, start="2018-01-02")   # T = 120
        long = _series(base * 80, start="2015-01-02")    # T = 480, same per-day stats
        p_short, p_long = m.psr(short), m.psr(long)
        assert 0.0 < p_short < 1.0
        assert 0.0 < p_long < 1.0
        assert p_long > p_short

    def test_dsr_below_psr_for_many_trials(self):
        rng = np.random.default_rng(11)
        r = _series(rng.normal(0.0012, 0.010, 500), start="2016-01-04")
        assert m.sharpe(r) > 0
        p = m.psr(r)
        d10 = m.dsr(r, n_trials=10)
        assert 0.0 < d10 < p
        # n_trials=1 deflates by nothing: identical to PSR against 0
        assert m.dsr(r, n_trials=1) == pytest.approx(p, rel=1e-12)

    def test_expected_max_sharpe(self):
        assert m.expected_max_sharpe(1, 0.01) == 0.0
        e10 = m.expected_max_sharpe(10, 0.01)
        e100 = m.expected_max_sharpe(100, 0.01)
        assert 0.0 < e10 < e100  # grows with the number of trials
        assert math.isnan(m.expected_max_sharpe(0, 0.01))
        assert math.isnan(m.expected_max_sharpe(10, -1.0))


# --------------------------------------------------------------------------
# series / table outputs
# --------------------------------------------------------------------------

class TestTables:
    def test_monthly_returns_compounds_january(self):
        jan = pd.bdate_range("2020-01-01", "2020-01-31")
        feb = pd.bdate_range("2020-02-03", "2020-02-07")
        r = pd.Series(
            [0.01] * len(jan) + [-0.02] * len(feb), index=jan.append(feb), dtype=float
        )
        table = m.monthly_returns(r)
        assert table.loc[2020, 1] == pytest.approx(1.01 ** len(jan) - 1.0, rel=1e-12)
        assert table.loc[2020, 2] == pytest.approx(0.98 ** len(feb) - 1.0, rel=1e-12)
        assert list(table.columns) == list(range(1, 13))
        assert math.isnan(table.loc[2020, 3])  # no March data

    def test_rolling_sharpe(self):
        rng = np.random.default_rng(5)
        r = _series(rng.normal(0.0005, 0.01, 300))
        rs = m.rolling_sharpe(r, window=126)
        assert rs.index.equals(r.index)
        assert rs.iloc[:125].isna().all()
        window = r.iloc[0:126]
        expected = float(window.mean()) / float(window.std(ddof=1)) * math.sqrt(252)
        assert rs.iloc[125] == pytest.approx(expected, rel=1e-12)

    def test_turnover_and_cost_drag(self):
        t = _series([0.10, 0.20, 0.30])
        stats = m.turnover_stats(t)
        assert stats["daily_mean"] == pytest.approx(0.20, rel=1e-12)
        assert stats["annualized"] == pytest.approx(0.20 * 252, rel=1e-12)
        assert m.cost_drag(_series([0.0002] * 50)) == pytest.approx(0.0002 * 252, rel=1e-9)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

SUMMARY_KEYS = {
    "ann_return_net", "ann_return_gross", "ann_vol", "sharpe_net",
    "sharpe_gross", "sharpe_se", "sortino", "max_drawdown",
    "max_drawdown_peak", "max_drawdown_trough", "calmar", "hit_rate",
    "psr", "dsr", "n_trials", "turnover_daily_mean", "turnover_ann",
    "cost_drag_ann", "n_days", "start", "end", "n_windows", "mode",
}


class TestSummary:
    def _result(self):
        rng = np.random.default_rng(9)
        idx = pd.bdate_range("2019-01-02", periods=300)
        gross = pd.Series(rng.normal(0.0006, 0.008, 300), index=idx)
        costs = pd.Series(0.0001, index=idx)
        net = gross - costs
        turnover = pd.Series(np.abs(rng.normal(0.15, 0.03, 300)), index=idx)
        weights = pd.DataFrame(
            {"SYMA": 0.5, "SYMB": -0.5}, index=idx, dtype=float
        )
        return BacktestResult(
            gross_returns=gross,
            costs=costs,
            net_returns=net,
            turnover=turnover,
            holdings=weights.copy(),
            target_weights=weights,
            meta={"mode": "insample"},
        )

    def test_summary_keys_and_json(self):
        s = m.summary(self._result(), n_trials=5)
        assert set(s) == SUMMARY_KEYS
        json.dumps(s)  # every value json-serializable
        for key, val in s.items():
            assert isinstance(val, (float, int, str)), f"{key}: {type(val)}"

    def test_summary_values(self):
        result = self._result()
        s = m.summary(result, n_trials=5)
        assert s["n_days"] == 300
        assert s["n_windows"] == 0  # windows is None
        assert s["n_trials"] == 5
        assert s["mode"] == "insample"
        assert s["start"] == result.net_returns.index[0].isoformat()
        assert s["end"] == result.net_returns.index[-1].isoformat()
        assert s["ann_return_gross"] > s["ann_return_net"]  # costs drag net down
        assert s["cost_drag_ann"] == pytest.approx(0.0001 * 252, rel=1e-9)
        assert s["max_drawdown"] <= 0.0
        assert "T" in s["max_drawdown_peak"] or s["max_drawdown_peak"] == ""  # iso string
        assert 0.0 <= s["psr"] <= 1.0
        assert s["dsr"] <= s["psr"]  # 5 trials deflate

    def test_walkforward_summary_trims_flat_prefix(self):
        """Regression: metrics of a walk-forward result must cover the ACTIVE
        period only — the train+purge prefix of forced-zero returns dilutes
        Sharpe/vol/hit_rate by a train_days-dependent factor otherwise."""
        from alpha_lab.core.results import WalkForwardWindow

        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2019-01-02", periods=300)
        prefix = 120  # structurally flat pre-first-test-window period
        active = pd.Series(rng.normal(0.0008, 0.008, 300 - prefix), index=idx[prefix:])
        net = pd.Series(0.0, index=idx)
        net.loc[idx[prefix]:] = active
        turnover = pd.Series(0.0, index=idx)
        turnover.iloc[prefix:] = 0.2
        weights = pd.DataFrame({"SYMA": 0.5, "SYMB": -0.5}, index=idx, dtype=float)
        windows = [WalkForwardWindow(idx[0], idx[prefix - 6], idx[prefix], idx[-1])]
        result = BacktestResult(
            gross_returns=net.copy(),
            costs=pd.Series(0.0, index=idx),
            net_returns=net,
            turnover=turnover,
            holdings=weights.copy(),
            target_weights=weights,
            windows=windows,
            meta={"mode": "walkforward"},
        )
        s = m.summary(result)
        assert s["n_days"] == 300 - prefix
        assert s["start"] == idx[prefix].isoformat()
        assert s["sharpe_net"] == pytest.approx(m.sharpe(active), rel=1e-12)
        assert s["hit_rate"] == pytest.approx(m.hit_rate(active), rel=1e-12)
        assert s["turnover_daily_mean"] == pytest.approx(0.2, rel=1e-12)
