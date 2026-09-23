"""Performance and risk metrics over daily return series.

All functions take a daily return ``pd.Series`` (close-to-close, per
CONVENTIONS.md) and annualize with a factor of 252 trading days. Degenerate
inputs — empty series, all-NaN, too few observations, zero variance — return
``float('nan')`` (or NaN/None-valued fields for dict/frame outputs) and never
raise: metric evaluation must not kill a batch run.

Probabilistic and Deflated Sharpe Ratio follow Bailey & Lopez de Prado
("The Sharpe Ratio Efficient Frontier", 2012; "The Deflated Sharpe Ratio",
2014). No scipy: the normal CDF uses ``math.erf`` and the inverse normal CDF
uses Peter Acklam's rational approximation (see :func:`norm_ppf`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.core.results import BacktestResult
from alpha_lab.core.types import TRADING_DAYS_PER_YEAR

#: Euler-Mascheroni constant, used by expected_max_sharpe.
EULER_GAMMA = 0.5772156649015329

_ANN = float(TRADING_DAYS_PER_YEAR)
_NAN = float("nan")


# --------------------------------------------------------------------------
# normal distribution helpers (no scipy)
# --------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard normal CDF, Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
    if not math.isfinite(x):
        if math.isnan(x):
            return _NAN
        return 0.0 if x < 0 else 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam coefficients (lower/central/upper rational approximations).
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function) at probability ``p``.

    Implements Peter J. Acklam's rational approximation ("An algorithm for
    computing the inverse normal cumulative distribution function", 2003,
    formerly at home.online.no/~pjacklam/notes/invnorm/), accurate to a
    relative error below 1.15e-9 over the full open interval. Inputs outside
    (0, 1) return NaN rather than +/-inf, per this module's degenerate-input
    policy.
    """
    if not (isinstance(p, (int, float)) and math.isfinite(p)) or p <= 0.0 or p >= 1.0:
        return _NAN
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p > 1.0 - _P_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
        (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------

def _as_series(returns: Any) -> pd.Series:
    """Coerce input to a float Series; anything unusable becomes empty."""
    if isinstance(returns, pd.Series):
        try:
            return returns.astype(float)
        except (TypeError, ValueError):
            return pd.Series(dtype=float)
    try:
        return pd.Series(returns, dtype=float)
    except (TypeError, ValueError):
        return pd.Series(dtype=float)


def _moments(returns: pd.Series) -> tuple[int, float, float, float] | None:
    """(T, daily SR, population skew, raw population kurtosis), or None.

    None signals a degenerate input: fewer than two observations or zero
    variance. Skew/kurtosis are plain moment (population) estimators — the
    same quantities the PSR/DSR formulas expect.
    """
    x = _as_series(returns).dropna().to_numpy(dtype=float)
    t = int(x.size)
    if t < 2:
        return None
    # exact-constant check: summation error can leave a ~1e-19 residual std
    # on a constant series, which would fake an astronomical Sharpe
    if (x == x[0]).all():
        return None
    std = float(x.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return None
    dev = x - x.mean()
    m2 = float(np.mean(dev**2))
    if m2 <= 0.0:
        return None
    skew = float(np.mean(dev**3) / m2**1.5)
    kurt = float(np.mean(dev**4) / m2**2)
    sr = float(x.mean()) / std
    return t, sr, skew, kurt


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------

def ann_return(returns: pd.Series) -> float:
    """Geometric annualized return: (prod(1 + r)) ** (252 / T) - 1."""
    r = _as_series(returns).dropna()
    if r.empty:
        return _NAN
    growth = float((1.0 + r).prod())
    if growth <= 0.0:
        return _NAN
    return growth ** (_ANN / len(r)) - 1.0


def ann_vol(returns: pd.Series) -> float:
    """Annualized volatility: daily std (ddof=1) * sqrt(252)."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return _NAN
    std = float(r.std(ddof=1))
    if not math.isfinite(std):
        return _NAN
    return std * math.sqrt(_ANN)


def sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio at rf=0: (daily mean / daily std) * sqrt(252)."""
    mom = _moments(returns)
    if mom is None:
        return _NAN
    _, sr, _, _ = mom
    return sr * math.sqrt(_ANN)


def sortino(returns: pd.Series) -> float:
    """Annualized Sortino ratio at rf=0 (target-0 downside deviation).

    Denominator is the standard downside deviation about a 0 target over
    ALL observations, sqrt(mean(min(r, 0)^2)) — the magnitude of losses,
    not their spread (the std of negative days around their own mean would
    reward consistently sized losses with an arbitrarily large ratio).
    NaN when there are no negative days (downside deviation 0) or fewer
    than two observations.
    """
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return _NAN
    downside = np.minimum(r.to_numpy(dtype=float), 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    if not math.isfinite(dd) or dd <= 0.0:
        return _NAN
    return float(r.mean()) / dd * math.sqrt(_ANN)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of (non-NaN) days with a strictly positive return."""
    r = _as_series(returns).dropna()
    if r.empty:
        return _NAN
    return float((r > 0.0).sum()) / len(r)


def skewness(returns: pd.Series) -> float:
    """Population (moment) skewness, m3 / m2^1.5 — no small-sample correction."""
    mom = _moments(returns)
    if mom is None:
        return _NAN
    return mom[2]


def kurtosis(returns: pd.Series) -> float:
    """RAW (non-excess) population kurtosis, m4 / m2^2 — a normal gives 3.0.

    This is excess kurtosis + 3, the form the PSR/DSR formulas consume.
    """
    mom = _moments(returns)
    if mom is None:
        return _NAN
    return mom[3]


def sharpe_se(returns: pd.Series) -> float:
    """Standard error of the DAILY Sharpe ratio estimator.

    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier":

        SE(SR) = sqrt((1 - skew * SR + (kurt - 1) / 4 * SR^2) / (T - 1))

    with SR the daily (non-annualized) Sharpe, ``skew`` the population
    skewness and ``kurt`` the raw (non-excess) kurtosis. For normal returns
    (skew 0, kurt 3) this reduces to Lo (2002)'s sqrt((1 + SR^2/2)/(T-1)).
    """
    mom = _moments(returns)
    if mom is None:
        return _NAN
    t, sr, skew, kurt = mom
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (t - 1)
    if not math.isfinite(var) or var <= 0.0:
        return _NAN
    return math.sqrt(var)


# --------------------------------------------------------------------------
# drawdown
# --------------------------------------------------------------------------

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Per-date drawdown of the compounded equity curve, as a fraction <= 0.

    Equity starts at 1.0 (so an initial losing streak is itself a drawdown);
    NaN returns are treated as flat days.
    """
    r = _as_series(returns)
    if r.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + r.fillna(0.0)).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> dict:
    """Deepest peak-to-trough loss of the compounded equity curve.

    Returns a dict with:
      depth          most negative drawdown (e.g. -0.20 for a 20% loss);
                     0.0 when the curve never falls below a prior peak
      peak_date      date the violated peak was set (first index date when
                     the drawdown starts at inception)
      trough_date    date of the maximum loss
      recovery_date  first date at/above the prior peak after the trough,
                     or None if never recovered
      duration_days  trading days from peak to recovery (to the last date
                     when unrecovered)

    Degenerate input returns NaN depth/duration and None dates.
    """
    out: dict[str, Any] = {
        "depth": _NAN,
        "peak_date": None,
        "trough_date": None,
        "recovery_date": None,
        "duration_days": _NAN,
    }
    r = _as_series(returns).dropna()
    if r.empty:
        return out
    dd = drawdown_series(r)
    trough = dd.idxmin()
    depth = float(dd.loc[trough])
    if depth > -1e-15:  # curve never dipped below a prior peak
        first = dd.index[0]
        return {
            "depth": 0.0,
            "peak_date": first,
            "trough_date": first,
            "recovery_date": None,
            "duration_days": 0,
        }
    # peak = last at-high date at/before the trough; inception if none.
    before = dd.loc[:trough]
    at_high = before[before > -1e-12]
    peak_date = at_high.index[-1] if len(at_high) else dd.index[0]
    after = dd.loc[trough:].iloc[1:]
    recovered = after[after > -1e-12]
    recovery_date = recovered.index[0] if len(recovered) else None
    end_date = recovery_date if recovery_date is not None else dd.index[-1]
    duration = int(dd.index.get_loc(end_date)) - int(dd.index.get_loc(peak_date))
    out.update(
        depth=depth,
        peak_date=peak_date,
        trough_date=trough,
        recovery_date=recovery_date,
        duration_days=duration,
    )
    return out


def calmar(returns: pd.Series) -> float:
    """Calmar ratio: geometric annualized return / |max drawdown depth|."""
    depth = max_drawdown(returns)["depth"]
    if not isinstance(depth, float) or math.isnan(depth) or depth == 0.0:
        return _NAN
    ar = ann_return(returns)
    if math.isnan(ar):
        return _NAN
    return ar / abs(depth)


# --------------------------------------------------------------------------
# PSR / DSR
# --------------------------------------------------------------------------

def psr(returns: pd.Series, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio against benchmark daily Sharpe ``sr_star``.

    Bailey & Lopez de Prado (2012):

        PSR = Phi(((SR - SR*) * sqrt(T - 1))
                  / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2))

    All Sharpe quantities in DAILY units; ``kurt`` is raw (non-excess).
    Probability that the true Sharpe exceeds ``sr_star`` given estimation
    noise from a T-observation track record.
    """
    mom = _moments(returns)
    if mom is None or not (isinstance(sr_star, (int, float)) and math.isfinite(sr_star)):
        return _NAN
    t, sr, skew, kurt = mom
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if not math.isfinite(denom_sq) or denom_sq <= 0.0:
        return _NAN
    z = (sr - float(sr_star)) * math.sqrt(t - 1) / math.sqrt(denom_sq)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum daily Sharpe among ``n_trials`` zero-skill trials.

    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio":

        E[max SR] = sqrt(var_sr) * ((1 - gamma) * z(1 - 1/N)
                                    + gamma * z(1 - 1/(N * e)))

    with gamma the Euler-Mascheroni constant and z the inverse normal CDF.
    ``var_sr`` is the cross-trial variance of estimated daily Sharpes.
    Returns 0.0 for n_trials == 1 (a single trial carries no selection
    bias), NaN for invalid inputs.
    """
    if not isinstance(n_trials, (int, float)) or isinstance(n_trials, bool):
        return _NAN
    if not (isinstance(var_sr, (int, float)) and math.isfinite(var_sr)) or var_sr < 0.0:
        return _NAN
    n = float(n_trials)
    if not math.isfinite(n) or n < 1.0:
        return _NAN
    if n == 1.0:
        return 0.0
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * math.e))
    if math.isnan(z1) or math.isnan(z2):
        return _NAN
    return math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def dsr(returns: pd.Series, n_trials: int, var_sr: float | None = None) -> float:
    """Deflated Sharpe Ratio: PSR against the expected max Sharpe of N trials.

    ``var_sr`` defaults to sharpe_se(returns)**2 — the estimator-variance
    proxy for the cross-trial Sharpe variance when the true dispersion of
    the trials is unknown. n_trials == 1 deflates by nothing (PSR against 0).
    """
    if var_sr is None:
        se = sharpe_se(returns)
        if math.isnan(se):
            return _NAN
        var_sr = se * se
    sr_star = expected_max_sharpe(n_trials, var_sr)
    if math.isnan(sr_star):
        return _NAN
    return psr(returns, sr_star=sr_star)


# --------------------------------------------------------------------------
# series / table outputs
# --------------------------------------------------------------------------

def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    """Rolling annualized Sharpe over ``window`` days; NaN where undefined."""
    r = _as_series(returns)
    if r.empty:
        return pd.Series(dtype=float)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    out = mu / sd * math.sqrt(_ANN)
    return out.replace([np.inf, -np.inf], np.nan)


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Calendar table of monthly returns: year rows x month (1..12) columns.

    Each cell is the return COMPOUNDED within the month, prod(1 + r) - 1.
    Months with no observations are NaN. Empty input gives an empty frame.
    """
    r = _as_series(returns).dropna()
    if r.empty or not isinstance(r.index, pd.DatetimeIndex):
        return pd.DataFrame()
    compounded = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    table = compounded.unstack(level=1).reindex(columns=range(1, 13))
    table.index.name = "year"
    table.columns.name = "month"
    return table


def turnover_stats(turnover: pd.Series) -> dict:
    """Mean daily turnover and its annualized (x252) equivalent."""
    t = _as_series(turnover).dropna()
    if t.empty:
        return {"daily_mean": _NAN, "annualized": _NAN}
    daily = float(t.mean())
    return {"daily_mean": daily, "annualized": daily * _ANN}


def cost_drag(costs: pd.Series) -> float:
    """Annualized cost drag: mean daily cost (NAV fraction) * 252."""
    c = _as_series(costs).dropna()
    if c.empty:
        return _NAN
    return float(c.mean()) * _ANN


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def _iso(ts: Any) -> str:
    """ISO string for a timestamp-like, '' for None/unparseable."""
    if ts is None:
        return ""
    try:
        return pd.Timestamp(ts).isoformat()
    except (TypeError, ValueError):
        return ""


def summary(result: BacktestResult, n_trials: int = 1) -> dict:
    """Flat, json-serializable metric dict for one BacktestResult.

    All values are plain floats/ints/str (dates as ISO strings). Sharpe-family
    statistics (sharpe_se, psr, dsr, sortino, hit_rate, drawdown, calmar) are
    computed on NET returns; ``n_trials`` feeds the DSR deflation.

    Walk-forward results are trimmed to the ACTIVE period (first test-window
    date onward): the structurally flat train+purge prefix carries forced-zero
    returns whose length varies with train_days, so including it would dilute
    Sharpe/vol/hit-rate by a config-dependent factor and bias comparisons
    across sweeps. ``n_days``/``start`` reflect the trimmed series.
    """
    net = _as_series(result.net_returns)
    gross = _as_series(result.gross_returns)
    turnover = _as_series(result.turnover)
    costs = _as_series(result.costs)
    if result.meta.get("mode") == "walkforward" and result.windows:
        active_start = result.windows[0].test_start
        net = net.loc[active_start:]
        gross = gross.loc[active_start:]
        turnover = turnover.loc[active_start:]
        costs = costs.loc[active_start:]
    mdd = max_drawdown(net)
    tstats = turnover_stats(turnover)
    idx = net.index
    return {
        "ann_return_net": float(ann_return(net)),
        "ann_return_gross": float(ann_return(gross)),
        "ann_vol": float(ann_vol(net)),
        "sharpe_net": float(sharpe(net)),
        "sharpe_gross": float(sharpe(gross)),
        "sharpe_se": float(sharpe_se(net)),
        "sortino": float(sortino(net)),
        "max_drawdown": float(mdd["depth"]),
        "max_drawdown_peak": _iso(mdd["peak_date"]),
        "max_drawdown_trough": _iso(mdd["trough_date"]),
        "calmar": float(calmar(net)),
        "hit_rate": float(hit_rate(net)),
        "psr": float(psr(net)),
        "dsr": float(dsr(net, n_trials)),
        "n_trials": int(n_trials),
        "turnover_daily_mean": float(tstats["daily_mean"]),
        "turnover_ann": float(tstats["annualized"]),
        "cost_drag_ann": float(cost_drag(costs)),
        "n_days": int(len(net)),
        "start": _iso(idx[0]) if len(idx) else "",
        "end": _iso(idx[-1]) if len(idx) else "",
        "n_windows": int(len(result.windows)) if result.windows is not None else 0,
        "mode": str(result.meta.get("mode", "")),
    }
