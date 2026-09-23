"""Statistical diagnostics for daily excess returns and repeated trials.

Probabilistic and deflated Sharpe calculations follow Bailey and Lopez de
Prado's Sharpe Ratio Efficient Frontier (2012) and Deflated Sharpe Ratio
(2014). Trial counts should reflect the full search, including unsuccessful
experiments; these adjustments cannot repair an incomplete selection record.

Inputs and internal Sharpe calculations use daily units. Annualized display
values have the suffix _ann. Normal CDF calculations use statistics.NormalDist.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_PHI = NormalDist()
EULER_GAMMA = 0.5772156649015329


def _clean(excess: pd.Series) -> np.ndarray:
    r = pd.Series(excess).dropna().to_numpy(dtype=float)
    if len(r) < 60:
        raise ValueError(f"need >= 60 daily observations, got {len(r)}")
    return r


def sharpe_daily(excess: pd.Series) -> float:
    r = _clean(excess)
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd)


def probabilistic_sharpe(excess: pd.Series, sr_benchmark_daily: float = 0.0) -> dict:
    """PSR: probability the TRUE Sharpe exceeds the benchmark, given the
    estimated Sharpe, sample length, skewness and kurtosis (fat tails and
    negative skew widen the estimator's variance and lower PSR)."""
    r = _clean(excess)
    s = pd.Series(r)
    T = len(r)
    sr = sharpe_daily(s)
    g3 = float(s.skew())            # sample skewness
    g4 = float(s.kurt()) + 3.0      # Pearson kurtosis (normal = 3)
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr))
    z = (sr - sr_benchmark_daily) * math.sqrt(T - 1) / denom
    return {
        "sharpe_ann": round(sr * math.sqrt(TRADING_DAYS), 3),
        "benchmark_ann": round(sr_benchmark_daily * math.sqrt(TRADING_DAYS), 3),
        "T": T, "skew": round(g3, 2), "kurtosis": round(g4, 2),
        "psr": round(_PHI.cdf(z), 4),
    }


def expected_max_sharpe_daily(n_trials: int, var_trials_daily: float) -> float:
    """E[max SR] of n_trials independent zero-true-Sharpe trials whose
    estimated Sharpes have variance var_trials_daily (daily^2 units).
    This is the hurdle a search's winner must clear to mean anything."""
    n = max(2, int(n_trials))
    sd = math.sqrt(max(var_trials_daily, 1e-18))
    return sd * ((1.0 - EULER_GAMMA) * _PHI.inv_cdf(1.0 - 1.0 / n)
                 + EULER_GAMMA * _PHI.inv_cdf(1.0 - 1.0 / (n * math.e)))


def deflated_sharpe(excess: pd.Series, n_trials: int,
                    trial_sharpes_ann: list[float] | None = None,
                    var_trials_daily: float | None = None) -> dict:
    """DSR = PSR evaluated at the expected max Sharpe of the search.
    Supply either the logged trial Sharpes (annualized, straight from a
    variants CSV — converted here) or a variance directly. n_trials is a
    JUDGMENT CALL (correlated variants are not independent trials): report
    results under more than one assumption rather than hiding the choice."""
    if var_trials_daily is None:
        if not trial_sharpes_ann or len(trial_sharpes_ann) < 2:
            raise ValueError("need trial_sharpes_ann (>=2) or var_trials_daily")
        arr = np.array(trial_sharpes_ann, dtype=float) / math.sqrt(TRADING_DAYS)
        var_trials_daily = float(arr.var(ddof=1))
    sr0 = expected_max_sharpe_daily(n_trials, var_trials_daily)
    out = probabilistic_sharpe(excess, sr_benchmark_daily=sr0)
    return {
        "sharpe_ann": out["sharpe_ann"],
        "hurdle_expected_max_sharpe_ann": round(sr0 * math.sqrt(TRADING_DAYS), 3),
        "n_trials": int(n_trials),
        "trial_sd_ann": round(math.sqrt(var_trials_daily) * math.sqrt(TRADING_DAYS), 3),
        "T": out["T"], "skew": out["skew"], "kurtosis": out["kurtosis"],
        "dsr": out["psr"],
    }


def block_bootstrap_sharpe_ci(excess: pd.Series, n_boot: int = 2000,
                              block: int = 21, alpha: float = 0.05,
                              seed: int = 7) -> dict:
    """Moving-block bootstrap CI for the ANNUALIZED Sharpe. Blocks preserve
    short-range autocorrelation/vol-clustering that iid resampling destroys."""
    r = _clean(excess)
    T = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(T / block)
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T]
    samples = r[idx]
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    sr = np.where(sd > 0, mu / sd, 0.0) * math.sqrt(TRADING_DAYS)
    lo, hi = np.quantile(sr, [alpha / 2, 1 - alpha / 2])
    return {
        "sharpe_ann": round(float(r.mean() / r.std(ddof=1)) * math.sqrt(TRADING_DAYS), 3),
        "ci": [round(float(lo), 3), round(float(hi), 3)],
        "alpha": alpha, "n_boot": n_boot, "block": block, "T": T,
        "prob_sharpe_le_0": round(float((sr <= 0).mean()), 4),
    }
