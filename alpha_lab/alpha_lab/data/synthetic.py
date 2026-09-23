"""Deterministic synthetic equity panel.

Purpose: (1) the platform runs out of the box with no data vendor; (2) tests
get a panel with *known* structure — persistent cross-sectional drifts (so
momentum is detectable), a 4:1 split (so unadjusted-price share math is
testable), and universe entry/exit (so point-in-time handling is testable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.core.types import TRADING_DAYS_PER_YEAR, MarketData


def make_market(
    n_assets: int = 20,
    n_days: int = 756,
    seed: int = 7,
    start: str = "2015-01-02",
    drift_dispersion: float = 0.10,
    base_vol: float = 0.20,
    split_asset: bool = True,
    universe_churn: bool = True,
) -> MarketData:
    """Build a synthetic MarketData panel. Same args => identical output.

    Returns follow a one-factor model plus a slowly mean-reverting (AR(1),
    phi=0.995) per-asset drift whose cross-sectional dispersion is
    ``drift_dispersion`` annualized — that persistence is what a momentum
    signal should be able to pick up.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"SYM{i:02d}" for i in range(n_assets)]
    ann = np.sqrt(TRADING_DAYS_PER_YEAR)

    vols = base_vol * np.exp(rng.uniform(-0.5, 0.5, n_assets)) / ann  # daily idio vol
    betas = rng.uniform(0.5, 1.5, n_assets)
    market = rng.normal(0.0002, 0.16 / ann, n_days)

    phi = 0.995
    mu_scale = drift_dispersion / TRADING_DAYS_PER_YEAR
    mus = np.zeros((n_days, n_assets))
    mus[0] = rng.normal(0.0, mu_scale, n_assets)
    shocks = rng.normal(0.0, mu_scale * np.sqrt(1 - phi**2), (n_days, n_assets))
    for t in range(1, n_days):
        mus[t] = phi * mus[t - 1] + shocks[t]

    rets = mus + betas * market[:, None] + rng.standard_normal((n_days, n_assets)) * vols
    rets = np.clip(rets, -0.4, 0.4)
    p0 = rng.uniform(20.0, 200.0, n_assets)
    close = pd.DataFrame(p0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=tickers)

    open_noise = rng.normal(0.0, 0.002, (n_days, n_assets))
    open_ = (close.shift(1) * (1.0 + open_noise)).fillna(close)
    span = np.abs(rng.normal(0.0, 0.004, (n_days, n_assets)))
    high = pd.DataFrame(np.maximum(open_.values, close.values) * (1.0 + span), index=dates, columns=tickers)
    low = pd.DataFrame(np.minimum(open_.values, close.values) * (1.0 - span), index=dates, columns=tickers)

    adv_shares = 10.0 ** rng.uniform(5.5, 7.5, n_assets)
    volume = pd.DataFrame(
        np.round(adv_shares * rng.lognormal(0.0, 0.4, (n_days, n_assets))),
        index=dates,
        columns=tickers,
    )

    # 4:1 split on the first ticker at ~60% of the sample: the raw (unadjusted)
    # price is 4x the adjusted level before the split date. Share-based math
    # (commissions, dollar volume) done on the adjusted close would be 4x off
    # there — the cost-model regression test relies on this.
    unadjusted = close.copy()
    if split_asset and n_days >= 120:
        k = int(n_days * 0.6)
        unadjusted.iloc[:k, 0] = unadjusted.iloc[:k, 0] * 4.0

    universe = pd.DataFrame(True, index=dates, columns=tickers)
    if universe_churn and n_assets >= 4:
        entrant, delisted = tickers[-1], tickers[-2]
        entry = int(n_days * 0.25)
        exit_ = int(n_days * 0.80)
        universe.loc[dates[:entry], entrant] = False
        universe.loc[dates[exit_:], delisted] = False
        for frame in (close, open_, high, low, volume, unadjusted):
            frame.loc[dates[:entry], entrant] = np.nan
            frame.loc[dates[exit_:], delisted] = np.nan

    return MarketData.from_frames(
        close,
        open=open_,
        high=high,
        low=low,
        volume=volume,
        unadjusted_close=unadjusted,
        universe=universe,
    )
