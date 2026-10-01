"""qcore.stats pinned to independently computed reference values.

The samples are built from integer arithmetic (no random stream), so the
hard-coded numbers do not depend on a NumPy version. Reference values were
computed once outside this module with scipy.stats (bias-corrected skewness
and kurtosis, norm.cdf / norm.ppf, numerical integration for the expected
maximum of N normals) and are written out here so the suite needs no scipy.
Closed-form cases are derived in the comments.
"""

import io
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from qcore import backtest as bt
from qcore.stats import (block_bootstrap_sharpe_ci, deflated_sharpe,
                         expected_max_sharpe_daily, probabilistic_sharpe, sharpe_daily)


def _uniform(n):
    """Deterministic values spread evenly over [-1, 1] (37 is coprime to 101)."""
    return (np.arange(n) * 37 % 101) / 50.0 - 1.0


def _skewed_fat_tailed() -> pd.Series:
    """750 daily excess returns with rare large losses (skew -2.06, kurtosis 13.1)."""
    i = np.arange(750)
    x = 0.0008 + 0.01 * _uniform(750)
    x = x + np.where(i % 61 == 0, -0.04, 0.0) + np.where(i % 43 == 7, 0.015, 0.0)
    return pd.Series(x)


def _two_point(n=252) -> pd.Series:
    """Alternating +2% / -1%: mean 0.5%, every deviation exactly +/-1.5%."""
    return pd.Series(np.where(np.arange(n) % 2 == 0, 0.02, -0.01))


# ------------------------------------------------------- Sharpe and PSR
def test_sharpe_and_psr_match_reference_values():
    r = _skewed_fat_tailed()
    assert sharpe_daily(r) == pytest.approx(0.05433985229984699, rel=1e-9)
    assert probabilistic_sharpe(r) == {
        "sharpe_ann": 0.863, "benchmark_ann": 0.0, "T": 750,
        "skew": -2.06, "kurtosis": 13.1, "psr": 0.9199}
    at_002 = probabilistic_sharpe(r, sr_benchmark_daily=0.02)
    assert at_002["psr"] == 0.8126 and at_002["benchmark_ann"] == 0.317  # 0.02 x sqrt(252)
    assert probabilistic_sharpe(r, sr_benchmark_daily=0.05)["psr"] == 0.5447
    # a benchmark equal to the estimate is a coin flip; a far higher one is hopeless
    assert probabilistic_sharpe(r, sr_benchmark_daily=sharpe_daily(r))["psr"] == 0.5
    assert probabilistic_sharpe(r, sr_benchmark_daily=0.2)["psr"] < 0.001


def test_psr_closed_form_on_a_two_point_sample():
    n = 252
    r = _two_point(n)
    # sample sd = 0.015 x sqrt(n / (n - 1)), so SR = (1/3) x sqrt(251/252)
    sr = math.sqrt(251 / 252) / 3
    assert sharpe_daily(r) == pytest.approx(sr, rel=1e-12)
    # a symmetric two-point sample has zero skewness and bias-corrected
    # Pearson kurtosis 3 - 2(n - 1)/(n - 3)
    g4 = 3 - 2 * (n - 1) / (n - 3)
    out = probabilistic_sharpe(r, sr_benchmark_daily=0.25)
    assert out["skew"] == 0.0 and out["kurtosis"] == round(g4, 2) == 0.98
    assert out["sharpe_ann"] == round(sr * math.sqrt(252), 3) == 5.281
    # PSR = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)), z = 1.31005
    z = (sr - 0.25) * math.sqrt(n - 1) / math.sqrt(1 + (g4 - 1) / 4 * sr * sr)
    assert z == pytest.approx(1.31005, abs=1e-5)
    assert out["psr"] == round(NormalDist().cdf(z), 4) == 0.9049


def test_psr_penalises_negative_skew_and_fat_tails_at_equal_sharpe():
    base = 0.001 + 0.01 * _uniform(2020)
    crash = base.copy()
    crash[::200] -= 0.08                                    # rare large losses
    crash = (crash - crash.mean()) / crash.std(ddof=1) * base.std(ddof=1) + base.mean()
    a, b = pd.Series(base), pd.Series(crash)
    assert sharpe_daily(a) == pytest.approx(sharpe_daily(b), rel=1e-9)   # same mean and sd
    hurdle = sharpe_daily(a) - 0.03
    pa, pb = probabilistic_sharpe(a, hurdle), probabilistic_sharpe(b, hurdle)
    assert abs(pa["skew"]) < 0.01 and pb["skew"] < -1 and pb["kurtosis"] > 6 > pa["kurtosis"]
    assert pb["psr"] < pa["psr"] - 0.02


# ------------------------------------------------ expected maximum Sharpe
def test_expected_max_sharpe_reference_values_scaling_and_order_statistics():
    assert expected_max_sharpe_daily(10, 1.0) == pytest.approx(1.57459830134575, rel=1e-9)
    assert expected_max_sharpe_daily(100, 1.0) == pytest.approx(2.5306028932016846, rel=1e-9)
    assert expected_max_sharpe_daily(250, 0.09 / 252) == pytest.approx(0.05362705603223539, rel=1e-9)
    # linear in the trial standard deviation, increasing in the trial count
    assert expected_max_sharpe_daily(20, 4.0) == pytest.approx(2.0 * expected_max_sharpe_daily(20, 1.0))
    assert expected_max_sharpe_daily(2, 1.0) < expected_max_sharpe_daily(20, 1.0) < expected_max_sharpe_daily(200, 1.0)
    # the approximation tracks the exact expected maximum of N standard normals
    # (2.50759 for N = 100, 3.24144 for N = 1000, by numerical integration)
    assert expected_max_sharpe_daily(100, 1.0) == pytest.approx(2.50759, rel=0.01)
    assert expected_max_sharpe_daily(1000, 1.0) == pytest.approx(3.24144, rel=0.005)


# ------------------------------------------------------- deflated Sharpe
def test_deflated_sharpe_converts_annualised_trials_and_applies_the_hurdle():
    r = _skewed_fat_tailed()
    trials = [0.2, 0.5, 0.9, 0.4, 0.7, 0.1, 0.6]   # annualised Sharpes, as a variants table logs them
    out = deflated_sharpe(r, n_trials=7, trial_sharpes_ann=trials)
    # sample sd of the trials is 0.2795 annualised; hurdle = 0.3875 annualised
    assert out == {"sharpe_ann": 0.863, "hurdle_expected_max_sharpe_ann": 0.388,
                   "n_trials": 7, "trial_sd_ann": 0.279, "T": 750,
                   "skew": -2.06, "kurtosis": 13.1, "dsr": 0.7804}
    # the same dispersion supplied directly in daily^2 units gives the same answer
    var_daily = float(np.var(np.array(trials) / math.sqrt(252), ddof=1))
    assert var_daily == pytest.approx(0.2794552524 ** 2 / 252, rel=1e-9)
    assert deflated_sharpe(r, 7, var_trials_daily=var_daily) == out
    # more trials raise the hurdle and lower the DSR; one trial deflates nothing
    more = deflated_sharpe(r, 250, trial_sharpes_ann=trials)
    assert more["hurdle_expected_max_sharpe_ann"] == 0.793 and more["dsr"] == 0.5451
    assert more["dsr"] < out["dsr"] < probabilistic_sharpe(r)["psr"]
    single = deflated_sharpe(r, 1, trial_sharpes_ann=trials)
    assert single["dsr"] == probabilistic_sharpe(r)["psr"] and single["hurdle_expected_max_sharpe_ann"] == 0.0
    # DSR is PSR evaluated at the hurdle
    hurdle = expected_max_sharpe_daily(7, var_daily)
    assert out["dsr"] == probabilistic_sharpe(r, sr_benchmark_daily=hurdle)["psr"]


# ------------------------------------------------------- block bootstrap
def _reference_bootstrap(r, n_boot, block, alpha, seed):
    """Plain-loop moving-block bootstrap using the same block starts."""
    T = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(T / block)
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    out = []
    for row in starts:
        sample = np.concatenate([r[s:s + block] for s in row])[:T]
        out.append(sample.mean() / sample.std(ddof=1) * math.sqrt(252))
    out = np.array(out)
    return np.quantile(out, [alpha / 2, 1 - alpha / 2]), float((out <= 0).mean())


def test_block_bootstrap_matches_a_plain_loop_reference():
    r = _skewed_fat_tailed()
    got = block_bootstrap_sharpe_ci(r, n_boot=300, block=21, alpha=0.10, seed=3)
    (lo, hi), p_le_0 = _reference_bootstrap(r.to_numpy(), 300, 21, 0.10, 3)
    assert got["ci"] == [round(float(lo), 3), round(float(hi), 3)]
    assert got["prob_sharpe_le_0"] == round(p_le_0, 4)
    assert got["sharpe_ann"] == 0.863 and got["T"] == 750 and got["block"] == 21
    assert got["alpha"] == 0.10 and got["n_boot"] == 300
    # annualised units: the interval brackets the annualised estimate and is
    # of the order of 2 x 1.645 / sqrt(years), not of a daily Sharpe
    assert got["ci"][0] < got["sharpe_ann"] < got["ci"][1]
    assert 0.8 < got["ci"][1] - got["ci"][0] < 3.5
    wide = block_bootstrap_sharpe_ci(r, n_boot=300, block=21, alpha=0.01, seed=3)["ci"]
    assert wide[0] < got["ci"][0] and wide[1] > got["ci"][1]
    # the mirrored series has the mirrored interval and the complementary tail
    neg = block_bootstrap_sharpe_ci(-r, n_boot=300, block=21, alpha=0.10, seed=3)
    assert neg["ci"] == [-got["ci"][1], -got["ci"][0]]
    assert got["prob_sharpe_le_0"] < 0.5 < neg["prob_sharpe_le_0"]
    assert neg["prob_sharpe_le_0"] == pytest.approx(1.0 - got["prob_sharpe_le_0"], abs=1e-4)


def test_blocks_preserve_autocorrelation_that_iid_resampling_destroys():
    i = np.arange(3000)
    r = pd.Series(0.0005 + 0.01 * np.sin(2 * np.pi * i / 250) + 0.002 * _uniform(3000))  # long swings
    blocked = block_bootstrap_sharpe_ci(r, n_boot=400, block=63, seed=1)["ci"]
    iid = block_bootstrap_sharpe_ci(r, n_boot=400, block=1, seed=1)["ci"]
    assert (blocked[1] - blocked[0]) > 2.0 * (iid[1] - iid[0])


# ------------------------------------------------ degenerate and bad input
def test_zero_excess_with_float_noise_scores_zero():
    # an all-cash book's excess return is zero up to float residue of ~1e-20
    noise = pd.Series(np.where(np.arange(300) % 7 == 0, 1.4e-20, 0.0))
    assert sharpe_daily(noise) == 0.0
    assert probabilistic_sharpe(noise)["psr"] == 0.5
    assert probabilistic_sharpe(noise)["sharpe_ann"] == 0.0
    boot = block_bootstrap_sharpe_ci(noise, n_boot=50)
    assert boot["sharpe_ann"] == 0.0 and boot["ci"] == [0.0, 0.0] and boot["prob_sharpe_le_0"] == 1.0
    assert deflated_sharpe(noise, 20, var_trials_daily=1e-4)["dsr"] < 0.5


def test_all_cash_book_scores_zero_after_a_csv_round_trip(monkeypatch):
    monkeypatch.setattr(bt, "_raw_close", lambda: pd.DataFrame())
    idx = pd.bdate_range("2021-01-04", periods=300)
    annual = pd.Series(0.5 + 2.0 * (1.0 + _uniform(300)), index=idx)   # T-bill yield, % per year
    px = pd.DataFrame({"ZZA": 50.0}, index=idx)
    res = bt.run_backtest(px * 0.0, px, cash_rate=annual, withholding=0.0)
    assert res["returns"].iloc[1:].gt(0).all()
    text = res["returns"].to_frame("cash").to_csv()
    back = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True)["cash"]
    excess = back - res["rf_daily"]
    assert excess.abs().max() < 1e-15
    assert sharpe_daily(excess) == 0.0
    assert probabilistic_sharpe(excess)["psr"] == 0.5
    assert block_bootstrap_sharpe_ci(excess, n_boot=50)["ci"] == [0.0, 0.0]
    assert bt.metrics(dict(res, returns=back))["full"]["sharpe"] == 0.0


def test_constant_nonzero_excess_has_no_sharpe():
    with pytest.raises(ValueError, match="constant nonzero"):
        sharpe_daily(pd.Series(np.full(90, 0.001)))
    # the same book with float residue is still constant, not an astronomic Sharpe
    wobble = pd.Series(0.001 + np.where(np.arange(90) % 5 == 0, 2e-18, 0.0))
    assert wobble.nunique() == 2
    with pytest.raises(ValueError, match="constant nonzero"):
        sharpe_daily(wobble)
    with pytest.raises(ValueError, match="constant nonzero"):
        probabilistic_sharpe(wobble)


def test_statistics_reject_short_non_finite_and_malformed_input():
    r = _skewed_fat_tailed()
    with pytest.raises(ValueError, match=">= 60 daily observations, got 59"):
        sharpe_daily(r.iloc[:59])
    assert math.isfinite(sharpe_daily(r.iloc[:60]))
    assert probabilistic_sharpe(pd.concat([r, pd.Series([np.nan] * 5)]))["T"] == 750   # NaN rows are dropped
    with pytest.raises(ValueError, match="finite"):
        sharpe_daily(pd.concat([r, pd.Series([np.inf])]))
    with pytest.raises(ValueError, match="benchmark must be finite"):
        probabilistic_sharpe(r, sr_benchmark_daily=float("nan"))
    with pytest.raises(ValueError, match="trial_sharpes_ann"):
        deflated_sharpe(r, 5)
    with pytest.raises(ValueError, match="trial_sharpes_ann"):
        deflated_sharpe(r, 5, trial_sharpes_ann=[0.4])
    with pytest.raises(ValueError, match="trial Sharpes must be finite"):
        deflated_sharpe(r, 5, trial_sharpes_ann=[0.4, float("nan")])
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            expected_max_sharpe_daily(bad, 1.0)
    with pytest.raises(ValueError, match="alpha"):
        block_bootstrap_sharpe_ci(r, n_boot=10, alpha=1.0)
    with pytest.raises(ValueError, match="n_boot"):
        block_bootstrap_sharpe_ci(r, n_boot=0)
    with pytest.raises(ValueError, match="block"):
        block_bootstrap_sharpe_ci(r, n_boot=10, block=0)
