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
    def test_annual_return_avoids_intermediate_compounding_overflow(self):
        # 11^1000 overflows, while 11^252 (the annual result) is finite.
        assert m.ann_return(_series([10.0] * 1000)) == pytest.approx(11.0**252 - 1.0, rel=1e-12)
        assert math.isnan(m.ann_return(_series([100.0])))  # annual value itself is unrepresentable

    def test_annual_return_distinguishes_total_loss_from_impossible_recovery(self):
        assert m.ann_return(_series([0.1, -1.0, 0.2])) == -1.0
        assert math.isnan(m.ann_return(_series([-2.0, -2.0])))

    def test_sharpe_moments_scale_without_overflow(self):
        ordinary = _series([1.0, -2.0, 3.0, -1.0, 0.5])
        huge = ordinary * 1e150
        for fn in (m.sharpe, m.skewness, m.kurtosis, m.psr):
            assert fn(huge) == pytest.approx(fn(ordinary), rel=1e-12)

    @pytest.mark.parametrize("fn", [m.ann_return, m.sharpe, m.hit_rate, m.sortino])
    def test_infinite_returns_are_invalid_not_positive_days(self, fn):
        assert math.isnan(fn(_series([0.01, np.inf, -0.01])))

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
    def test_long_positive_path_does_not_overflow(self):
        dd = m.drawdown_series(_series([10.0] * 1000 + [-0.1]))
        assert np.isfinite(dd).all()
        assert dd.iloc[-1] == pytest.approx(-0.1, abs=1e-12)

    def test_invalid_loss_path_returns_degenerate_drawdown(self):
        assert math.isnan(m.max_drawdown(_series([-2.0, -2.0]))["depth"])

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

    def test_norm_ppf_accepts_numpy_scalars(self):
        # np.float32 is not a Python float; it used to fall through to NaN
        assert m.norm_ppf(np.float64(0.975)) == m.norm_ppf(0.975)
        assert m.norm_ppf(np.float32(0.975)) == m.norm_ppf(float(np.float32(0.975)))
        assert m.norm_ppf(np.float32(0.975)) == pytest.approx(1.96, abs=1e-4)
        assert math.isnan(m.norm_ppf(True))
        assert math.isnan(m.norm_ppf("0.5"))
        assert math.isnan(m.norm_ppf(None))

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
    "sharpe_gross", "sharpe_se_ann", "sortino", "max_drawdown",
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
        # strict JSON: on a well-behaved series no value is NaN or infinite
        json.dumps(s, allow_nan=False)
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
        assert s["dsr"] < s["psr"]  # 5 trials deflate
        # Sharpe-family statistics come from NET returns; n_trials reaches DSR
        net, gross = result.net_returns, result.gross_returns
        assert s["sharpe_net"] == m.sharpe(net)
        assert s["sharpe_gross"] == m.sharpe(gross)
        assert s["sharpe_net"] < s["sharpe_gross"]
        assert s["ann_vol"] == m.ann_vol(net)
        assert s["psr"] == m.psr(net)
        assert s["dsr"] == m.dsr(net, 5)
        assert s["sharpe_se_ann"] == pytest.approx(m.sharpe_se(net) * math.sqrt(252), rel=1e-12)
        assert m.summary(result, n_trials=np.int64(5))["dsr"] == s["dsr"]

    def test_summary_net_statistics_are_not_computed_from_gross(self):
        rng = np.random.default_rng(9)
        idx = pd.bdate_range("2019-01-02", periods=300)
        gross = pd.Series(rng.normal(0.0002, 0.008, 300), index=idx)
        costs = pd.Series(0.0005, index=idx)  # large enough to flip many days
        net = gross - costs
        weights = pd.DataFrame({"SYMA": 0.5, "SYMB": -0.5}, index=idx, dtype=float)
        s = m.summary(BacktestResult(
            gross_returns=gross, costs=costs, net_returns=net,
            turnover=pd.Series(0.1, index=idx), holdings=weights.copy(),
            target_weights=weights, meta={"mode": "insample"},
        ))
        assert s["sortino"] == m.sortino(net) != m.sortino(gross)
        assert s["hit_rate"] == m.hit_rate(net) != m.hit_rate(gross)
        assert s["calmar"] == m.calmar(net) != m.calmar(gross)
        assert s["max_drawdown"] == m.max_drawdown(net)["depth"] != m.max_drawdown(gross)["depth"]
        assert s["max_drawdown_trough"] == m.max_drawdown(net)["trough_date"].isoformat()
        assert s["ann_return_net"] == m.ann_return(net)
        assert s["ann_return_gross"] == m.ann_return(gross)

    def test_hit_rate_counts_strictly_positive_days(self):
        assert m.hit_rate(_series([0.01, 0.0, 0.0, -0.01])) == 0.25

    @pytest.mark.parametrize("n_trials,stored", [(5, 5), (5.0, 5), (np.int64(5), 5), (2.5, 2.5),
                                                 (np.float32(2.5), 2.5), (0, 0)])
    def test_summary_reports_the_trial_count_it_deflated_by(self, n_trials, stored):
        result = self._result()
        s = m.summary(result, n_trials=n_trials)
        # int() used to truncate 2.5 to 2 while the DSR was deflated by 2.5
        assert s["n_trials"] == stored and type(s["n_trials"]) is type(stored)
        expected = m.dsr(result.net_returns, n_trials)
        assert s["dsr"] == expected or (math.isnan(s["dsr"]) and math.isnan(expected))
        if stored == 2.5:
            assert m.dsr(result.net_returns, 3) < s["dsr"] < m.dsr(result.net_returns, 2)

    @pytest.mark.parametrize("n_trials", [None, "3", float("nan"), float("inf"), True])
    def test_summary_never_raises_on_an_unusable_trial_count(self, n_trials):
        s = m.summary(self._result(), n_trials=n_trials)  # None used to raise TypeError
        assert math.isnan(s["dsr"]) and math.isnan(s["n_trials"])
        assert isinstance(s["n_trials"], float)
        assert s["psr"] == m.summary(self._result())["psr"]  # the rest is unaffected

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

    def test_walkforward_summary_trims_gross_and_costs_too(self):
        from alpha_lab.core.results import WalkForwardWindow

        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2019-01-02", periods=300)
        prefix = 120
        gross = pd.Series(0.0, index=idx)
        gross.iloc[prefix:] = rng.normal(0.0008, 0.008, 300 - prefix)
        costs = pd.Series(0.0, index=idx)
        costs.iloc[prefix:] = 0.0002
        turnover = pd.Series(0.0, index=idx)
        turnover.iloc[prefix:] = 0.2
        weights = pd.DataFrame({"SYMA": 0.5, "SYMB": -0.5}, index=idx, dtype=float)
        result = BacktestResult(
            gross_returns=gross, costs=costs, net_returns=gross - costs, turnover=turnover,
            holdings=weights.copy(), target_weights=weights,
            windows=[WalkForwardWindow(idx[0], idx[prefix - 6], idx[prefix], idx[-1])],
            meta={"mode": "walkforward"},
        )
        s = m.summary(result)
        assert s["cost_drag_ann"] == pytest.approx(0.0002 * 252, rel=1e-12)
        assert s["sharpe_gross"] == pytest.approx(m.sharpe(gross.iloc[prefix:]), rel=1e-12)
        assert s["ann_return_gross"] == pytest.approx(m.ann_return(gross.iloc[prefix:]), rel=1e-12)
        assert s["turnover_ann"] == pytest.approx(0.2 * 252, rel=1e-12)
        assert s["n_windows"] == 1 and s["mode"] == "walkforward"


# --------------------------------------------------------------------------
# PSR / DSR pinned to independent reference values
# --------------------------------------------------------------------------

def _skewed_fat_tailed():
    rng = np.random.default_rng(2024)
    return pd.Series(
        0.0004 + 0.01 * rng.standard_t(5, 750) + 0.004 * (rng.exponential(1.0, 750) - 1.0)
    )


def test_psr_dsr_match_reference_values():
    """Reference numbers computed once with scipy.stats (bias=True skew,
    fisher=False kurtosis, norm.cdf/ppf) and hard-coded, so CI needs no scipy."""
    r = _skewed_fat_tailed()
    assert m.skewness(r) == pytest.approx(0.27831086499026986, rel=1e-10)
    assert m.kurtosis(r) == pytest.approx(6.386032552594862, rel=1e-10)
    assert m.psr(r) == pytest.approx(0.9776326200054409, rel=1e-9)
    assert m.psr(r, sr_star=0.02) == pytest.approx(0.9273249310322265, rel=1e-9)
    assert m.sharpe_se(r) == pytest.approx(0.03629854372616901, rel=1e-10)
    assert m.expected_max_sharpe(10, 1.0) == pytest.approx(1.57459830134575, rel=1e-8)
    assert m.dsr(r, 7) == pytest.approx(0.7324914166404943, rel=1e-8)
    assert m.dsr(r, np.int64(7)) == m.dsr(r, 7)
