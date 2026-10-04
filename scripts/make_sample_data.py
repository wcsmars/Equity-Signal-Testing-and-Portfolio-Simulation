#!/usr/bin/env python3
# SYNTHETIC DATA GENERATOR. Every value this script writes is simulated from a
# seeded random model. Nothing here is, or is derived from, a market price,
# and results computed from its output say nothing about any market or rule.
"""Write a small SYNTHETIC sample of the market-data cache.

The strategy scripts read a price cache that src/download_data.py fills from
a data vendor. That takes a network connection, and vendor data may not be
redistributed. This script writes a stand-in cache in exactly the same
layout, so every market-data script can be run offline:

  adj_close.csv  close.csv  open.csv  high.csv  low.csv  volume.csv
  indices.csv    coverage.csv  README.txt

Usage:
  python scripts/make_sample_data.py [--out DIR] [--seed N] [--years N]
  export QCORE_DATA_DIR="$PWD/sample_data"
  python scripts/data_quality.py
  python src/strategies/mean_reversion.py

Without --out the files go to sample_data/ under the project root, never to
the default cache directory data/. QCORE_DATA_DIR then points every loader
at them; unset it to return to the default cache.

What is simulated (all of it; the ticker symbols only label the columns the
code expects):
  - one row per NYSE session (qcore.quality.nyse_bdays), from the first
    session of the year IN_SAMPLE_YEARS before the engine's in-sample /
    out-of-sample split (qcore.backtest.OOS_SPLIT) to the last session of
    the final year, so both segments exist and the final row is a month-end;
  - every ticker of the ETF, stock and cash-parking universes in qcore.data.
    Daily returns come from six correlated factors (equity, rates,
    commodities, dollar, non-US equity, growth) with slowly changing drifts,
    plus a ticker-specific part. A stress state raises volatility, pulls the
    equity factor down on the day it jumps and lifts bonds;
  - dividends: equity funds and most stocks pay quarterly, bond and bill
    funds monthly, commodity and currency funds nothing. close.csv is the
    price-return series and adj_close.csv the total-return series, equal to
    it on the final row, so the two differ before every ex-date;
  - open, high and low around each close, in adjusted units; share volume;
  - three pairs whose second fund follows the first with a mean-reverting
    spread (PAIR_FOLLOWERS), so the spread rule has something to trade, and
    a few tickers that list after the first row (LATE_LISTINGS);
  - indices.csv: a 13-week bill yield path (^IRX), a 10-year yield tied to
    the rates factor (^TNX), an equity index that follows the SPY price
    return (^GSPC) and two volatility indices driven by the stress state
    (^VIX above ^VIX3M in stress, below it otherwise).

Determinism: the same seed and years give the same files, byte for byte.
Random draws come from numpy's legacy RandomState, whose stream is frozen,
keyed by the seed and the ticker name, so adding or removing a ticker in a
universe leaves every other ticker's path as it was (a pair follower needs
its leader). The arithmetic is limited to elementwise + - * /, running
products and square roots in a fixed order (no library sums, powers or
logarithms), and the values are written with a fixed number of decimals by
this script, not by a CSV library. A longer panel starts with the same
closes, volumes and index series as a shorter one with the same seed; its
adjusted prices differ, because they are tied to the new final row.

Refusals (exit status 2, nothing written): --years outside MIN_YEARS to
MAX_YEARS, a seed outside 0 .. 2**32 - 1, and an output directory that
already holds anything this script did not write. A directory is recognised
as a sample by the first line of its README.txt; a downloaded cache has no
such file and is never overwritten.

Exit status: 0 the sample was written; 2 refused or a usage error.

Limits: the default seed is one for which every documented command completes
and every rule trades. Another seed gives other paths, and a script whose
outcome depends on the path can then end differently (tsmom_voltarget.py,
for example, declares no result when no variant lands in its volatility
band). The model has no splits, no delistings, no missing bars and no bad
prints, so it exercises the pipeline, not the failure cases of the
data-quality checks. Performance figures computed from the sample are
artefacts of the model's parameters.
"""

import argparse
import math
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import OOS_SPLIT  # noqa: E402
from qcore.data import (ETF_UNIVERSE, INDEX_UNIVERSE, OPERATIONAL_UNIVERSE,  # noqa: E402
                        STOCK_UNIVERSE)
from qcore.quality import nyse_bdays  # noqa: E402

DEFAULT_OUT = ROOT / "sample_data"
DEFAULT_SEED = 23
DEFAULT_YEARS = 10
IN_SAMPLE_YEARS = 6      # calendar years before OOS_SPLIT
MIN_YEARS = 7            # at least one out-of-sample year
MAX_YEARS = 14
PRICE_FILES = ("adj_close", "close", "open", "high", "low")
PRICE_FORMAT = "%.8f"    # fixed decimals: the written text is the sample
INDEX_FORMAT = "%.4f"
MARKER = "SYNTHETIC SAMPLE DATA - simulated values, not market prices."
README_NAME = "README.txt"
EXIT_REFUSED = 2

YEAR = 252.0
ROOT_YEAR = math.sqrt(YEAR)

# ------------------------------------------------------------ model: factors
# name: (daily mean, daily volatility, share of the stress state that scales
# the volatility). The order fixes the order of the random draws.
FACTORS = {
    "equity": (0.00050, 0.0075, 1.0),
    "rates": (0.00012, 0.0035, 0.4),
    "commodity": (0.00004, 0.0090, 0.6),
    "dollar": (0.00000, 0.0040, 0.3),
    "non_us": (-0.00004, 0.0050, 0.8),
    "growth": (0.00003, 0.0040, 0.6),
}
DRIFT_PERSISTENCE = 0.99   # slow drift state of each factor
DRIFT_SHOCK = 0.003        # ...its daily innovation, in units of the factor's volatility
STRESS_DECAY = 0.94        # stress state: half-life about 11 sessions
STRESS_JUMP_ODDS = 1.0 / 90.0
STRESS_JUMP_SIZE = (0.4, 1.6)
STRESS_EQUITY_HIT = 0.020  # the equity factor falls by this times the jump
STRESS_BOND_LIFT = 0.006   # ...and the rates factor rises by this times the jump
RATES_EQUITY_CORR = -0.30
COMMODITY_DOLLAR_BETA = -0.5
NON_US_DOLLAR_BETA = -0.6
SPECIFIC_STRESS_BETA = 0.5  # ticker-specific volatility grows with stress too

# Bill yield (^IRX, percent a year) at these years since the first row;
# straight lines in between, held flat after the last knot. A stylised rate
# cycle (near zero, a slow rise, a cut back to zero, a steep rise): the knots
# are round design values, not observations.
BILL_YIELD_KNOTS = ((0.0, 0.05), (4.0, 0.10), (7.0, 2.40), (8.0, 1.50), (8.25, 0.10),
                    (10.0, 0.10), (11.0, 2.00), (12.0, 5.00), (14.0, 4.50))
BILL_FUND_FEE = 0.09       # percent a year a bill fund keeps of the bill yield
TEN_YEAR_START, TEN_YEAR_ANCHOR, TEN_YEAR_PULL, TEN_YEAR_FLOOR = 2.00, 2.50, 0.002, 0.30
DURATION = 8.0             # years; links the rates factor to the 10-year yield
INDEX_START = 3000.0       # ^GSPC on the first row
VOL_INDEX_PREMIUM = 1.18   # implied over realised volatility
VOL_TERM_BASE, VOL_TERM_STRESS = 1.13, 0.58  # ^VIX3M = calm ^VIX x (base + stress x state)

# ------------------------------------------------------------ model: tickers
# Funds that are not plain equity funds. Any ETF not named below is modelled
# as an equity fund and any stock as a stock, so a universe change in
# qcore.data needs no change here.
BOND_FUNDS = {  # ticker: (rates beta, equity beta, annual specific vol, annual yield)
    "TLT": (2.30, 0.00, 0.020, 0.025), "IEF": (1.00, 0.00, 0.006, 0.020),
    "SHY": (0.20, 0.00, 0.003, 0.010), "TIP": (0.85, 0.00, 0.020, 0.020),
    "AGG": (0.65, 0.00, 0.010, 0.025), "LQD": (1.00, 0.12, 0.020, 0.035),
    "HYG": (0.25, 0.40, 0.025, 0.055), "EMB": (0.70, 0.30, 0.035, 0.045),
}
COMMODITY_FUNDS = {  # ticker: (commodity beta, equity beta, dollar beta, annual specific vol, annual drift)
    "GLD": (0.55, 0.00, -0.80, 0.10, 0.02), "SLV": (0.90, 0.10, -0.90, 0.18, 0.00),
    "GDX": (1.00, 0.50, -0.80, 0.22, 0.00), "DBC": (1.00, 0.10, 0.00, 0.04, 0.00),
    "USO": (1.60, 0.20, 0.00, 0.20, -0.05), "UNG": (0.80, 0.00, 0.00, 0.30, -0.10),
}
CURRENCY_FUNDS = {  # ticker: (dollar beta, rates beta)
    "UUP": (1.00, 0.00), "FXE": (-1.10, 0.00), "FXY": (-0.90, 0.30),
}
NON_US_FUNDS = {"EFA", "EEM", "VGK", "EWJ", "FXI", "EWY", "EWT", "EWZ", "EWA", "EWC",
                "EWG", "EWU", "EWH"}
COMMODITY_LINKED = {"XLE", "XOP", "XME", "XLB", "EWA", "EWC", "EWZ"}
MARKET_FUND = "SPY"        # the equity factor itself; ^GSPC follows its price return
MARKET_FUND_YIELD = 0.019
# follower: (leader, multiple of the leader's return). The follower's own
# part is a stationary deviation, so the pair's spread reverts.
PAIR_FOLLOWERS = {"XOP": ("XLE", 1.25), "EWC": ("EWA", 0.90), "XLK": ("QQQ", 1.05)}
PAIR_PERSISTENCE = 0.95
PAIR_SHOCK = 0.0045
# ticker: sessions after the first row before its first bar
LATE_LISTINGS = {"META": 100, "ABBV": 250, "SGOV": 500}

_HEAD = 16     # uniforms at the start of a ticker's stream that set its parameters
_PER_DAY = 39  # uniforms a ticker uses per session: 3 normals of 12, then high, low, volume


def _stream(seed: int, label: str) -> np.random.RandomState:
    """Frozen-stream generator keyed by the seed and a name."""
    return np.random.RandomState([seed, zlib.crc32(label.encode("utf-8"))])


def _normals(uniforms: np.ndarray) -> np.ndarray:
    """Approximately standard normal draws: the sum of twelve uniforms minus
    six, added one column at a time so the order of the additions is fixed."""
    total = uniforms[..., 0].copy()
    for k in range(1, 12):
        total = total + uniforms[..., k]
    return total - 6.0


def _sessions(years: int) -> list:
    first_year = int(OOS_SPLIT[:4]) - IN_SAMPLE_YEARS
    days = nyse_bdays(f"{first_year}-01-01", f"{first_year + years - 1}-12-31")
    return [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in days]


def _bill_yield(t: int) -> float:
    age = t / YEAR
    for (a0, y0), (a1, y1) in zip(BILL_YIELD_KNOTS, BILL_YIELD_KNOTS[1:]):
        if age <= a1:
            return y0 + (y1 - y0) * (age - a0) / (a1 - a0)
    return BILL_YIELD_KNOTS[-1][1]


def _factor_paths(seed: int, n: int) -> dict:
    """Daily factor returns, the stress state and four of the index series."""
    names = list(FACTORS)
    k = len(names)
    draws = _stream(seed, "factors").random_sample((n, 12 * (2 * k + 3) + 2))
    z = _normals(draws[:, :12 * (2 * k + 3)].reshape(n, 2 * k + 3, 12))
    jump_u, size_u = draws[:, -2], draws[:, -1]

    ret = {name: np.empty(n) for name in names}
    stress = np.empty(n)
    indices = {name: np.empty(n) for name in ("^IRX", "^TNX", "^VIX", "^VIX3M")}
    drift = [0.0] * k
    state, ten_year, bill_noise = 0.0, TEN_YEAR_START, 0.0
    tilt = math.sqrt(1.0 - RATES_EQUITY_CORR * RATES_EQUITY_CORR)
    calm_vol = 100.0 * ROOT_YEAR * FACTORS["equity"][1] * VOL_INDEX_PREMIUM
    for t in range(n):
        jump = 0.0
        if jump_u[t] < STRESS_JUMP_ODDS:
            lo, hi = STRESS_JUMP_SIZE
            jump = lo + (hi - lo) * float(size_u[t])
        state = STRESS_DECAY * state + jump
        row = [float(v) for v in z[t]]
        shock = dict(zip(names, row[:k]))
        shock["rates"] = RATES_EQUITY_CORR * shock["equity"] + tilt * shock["rates"]
        today = {}
        for j, name in enumerate(names):
            mean, vol, spill = FACTORS[name]
            drift[j] = DRIFT_PERSISTENCE * drift[j] + DRIFT_SHOCK * vol * row[k + j]
            today[name] = mean + drift[j] + vol * (1.0 + spill * state) * shock[name]
        today["equity"] = today["equity"] - STRESS_EQUITY_HIT * jump
        today["rates"] = today["rates"] + STRESS_BOND_LIFT * jump
        today["commodity"] = today["commodity"] + COMMODITY_DOLLAR_BETA * today["dollar"]
        today["non_us"] = today["non_us"] + NON_US_DOLLAR_BETA * today["dollar"]
        for name in names:
            ret[name][t] = today[name]
        stress[t] = state

        bill_noise = 0.9 * bill_noise + 0.006 * row[2 * k]
        indices["^IRX"][t] = max(_bill_yield(t) + bill_noise, 0.01)
        ten_year = (ten_year + TEN_YEAR_PULL * (TEN_YEAR_ANCHOR - ten_year)
                    - 100.0 * (today["rates"] - FACTORS["rates"][0]) / DURATION)
        if ten_year < TEN_YEAR_FLOOR:
            ten_year = 2.0 * TEN_YEAR_FLOOR - ten_year
        indices["^TNX"][t] = ten_year
        indices["^VIX"][t] = max(calm_vol * (1.0 + state) + 0.4 * row[2 * k + 1], 9.0)
        indices["^VIX3M"][t] = max(
            calm_vol * (VOL_TERM_BASE + VOL_TERM_STRESS * state) + 0.3 * row[2 * k + 2], 9.0)
    return {"returns": ret, "stress": stress, "indices": indices}


def _profile(ticker: str, head: np.ndarray) -> dict:
    """Factor betas, specific risk, payout and size of one ticker. `head`
    holds the first uniforms of the ticker's own stream."""
    u = [float(v) for v in head]
    p = {"betas": dict.fromkeys(FACTORS, 0.0), "drift": 0.0, "bill_fund": False,
         "specific": 0.0, "yield": 0.0, "payouts": 0,
         "price": 25.0 + 100.0 * u[4], "volume": 300_000.0 + 12_000_000.0 * u[5] * u[5],
         "ex_month": int(3 * u[6]) % 3, "ex_session": 3 + int(10 * u[7]) % 10}
    b = p["betas"]
    if ticker in OPERATIONAL_UNIVERSE:
        p.update(bill_fund=True, specific=0.0003, payouts=12, price=100.0)
    elif ticker in BOND_FUNDS:
        b["rates"], b["equity"], p["specific"], p["yield"] = BOND_FUNDS[ticker]
        p.update(payouts=12, price=80.0 + 40.0 * u[4])
    elif ticker in COMMODITY_FUNDS:
        b["commodity"], b["equity"], b["dollar"], p["specific"], p["drift"] = COMMODITY_FUNDS[ticker]
        p.update(price=40.0 + 100.0 * u[4])
    elif ticker in CURRENCY_FUNDS:
        b["dollar"], b["rates"] = CURRENCY_FUNDS[ticker]
        p.update(specific=0.02, volume=200_000.0 + 2_000_000.0 * u[5])
    elif ticker == MARKET_FUND:
        b["equity"] = 1.0
        p.update(specific=0.004, payouts=4, price=130.0, volume=90_000_000.0)
        p["yield"] = MARKET_FUND_YIELD
    elif ticker in STOCK_UNIVERSE:
        b["equity"], b["growth"] = 0.7 + 0.8 * u[0], 2.0 * (u[1] - 0.5)
        p.update(specific=0.14 + 0.18 * u[2], drift=0.06 * (u[8] - 0.5),
                 price=30.0 + 150.0 * u[4], volume=2_000_000.0 + 38_000_000.0 * u[5] * u[5])
        if u[3] >= 0.3:  # about three stocks in ten pay nothing
            p.update(payouts=4)
            p["yield"] = 0.008 + 0.030 * u[9]
    else:  # an equity fund
        b["equity"], b["growth"] = 0.6 + 0.7 * u[0], u[1] - 0.5
        b["non_us"] = 1.0 if ticker in NON_US_FUNDS else 0.0
        b["commodity"] = 0.4 if ticker in COMMODITY_LINKED else 0.0
        p.update(specific=0.04 + 0.06 * u[2], payouts=4)
        p["yield"] = 0.010 + 0.020 * u[3]
    return p


def _ex_sessions(dates: list, payouts: int, phase: int, session: int) -> np.ndarray:
    """True on one session of each paying month: the one `session` sessions
    after the month's first. Quarterly payers pay in the months of `phase`."""
    flags = np.zeros(len(dates), dtype=bool)
    rank, month = 0, ""
    for t, day in enumerate(dates):
        rank = rank + 1 if day[:7] == month else 0
        month = day[:7]
        pays = payouts == 12 or (payouts == 4 and (int(day[5:7]) - 1) % 3 == phase)
        flags[t] = pays and rank == session
    return flags


def _ticker_panel(ticker: str, seed: int, dates: list, factors: dict, leader=None) -> dict:
    """Daily total return and the six bars of one ticker over every session.
    `leader` is (the leader's daily total returns, multiple) for a pair follower."""
    n = len(dates)
    rs = _stream(seed, ticker)
    p = _profile(ticker, rs.random_sample(_HEAD))
    draws = rs.random_sample((n, _PER_DAY))
    z = _normals(draws[:, :36].reshape(n, 3, 12))
    stress = factors["stress"]

    daily_vol = p["specific"] / ROOT_YEAR
    if p["bill_fund"]:
        bills = factors["indices"]["^IRX"]
        accrual = np.empty(n)
        accrual[0] = 0.0   # each session accrues the yield printed the session before
        accrual[1:] = np.maximum(bills[:-1] - BILL_FUND_FEE, 0.0) / 100.0 / YEAR
        total = accrual + 0.05 * daily_vol * z[:, 0]
        typical = daily_vol
    elif leader is not None:
        base, multiple = leader
        deviation, change = 0.0, np.empty(n)
        for t in range(n):
            new = PAIR_PERSISTENCE * deviation + PAIR_SHOCK * float(z[t, 2])
            change[t] = new - deviation
            deviation = new
        total = multiple * base + change
        typical = 0.012 * multiple
    else:
        total = np.full(n, p["drift"] / YEAR)
        systematic = 0.0
        for name, beta in p["betas"].items():
            if beta:
                total = total + beta * factors["returns"][name]
                systematic = systematic + beta * beta * FACTORS[name][1] * FACTORS[name][1]
        total = total + daily_vol * (1.0 + SPECIFIC_STRESS_BETA * stress) * z[:, 0]
        typical = math.sqrt(systematic + daily_vol * daily_vol)

    start = min(LATE_LISTINGS.get(ticker, 0), n - 1)
    growth = 1.0 + total
    growth[:start + 1] = 1.0                      # the listing bar is the base
    ex = _ex_sessions(dates, p["payouts"], p["ex_month"], p["ex_session"])
    ex[:start + 1] = False
    paid = np.zeros(n)                            # share of the price paid out on an ex-date
    if p["bill_fund"]:                            # pays out what it accrued above its base
        level = 1.0
        for t in range(start + 1, n):
            if ex[t] and level > 1.00002:
                paid[t] = 1.0 - 1.0 / level
                level = 1.0
            level = level * float(growth[t])
    elif p["payouts"]:
        paid[ex] = p["yield"] / p["payouts"]
    total_index = np.cumprod(growth)
    close = p["price"] * np.cumprod(growth * (1.0 - paid))
    adj = close[-1] * total_index / total_index[-1]

    previous = np.empty(n)
    previous[0] = adj[0]
    previous[1:] = adj[:-1]
    gap = 0.35 * (growth - 1.0) + 0.30 * typical * z[:, 1]
    open_ = previous * (1.0 + gap)
    high = np.maximum(open_, adj) * (1.0 + 0.6 * typical * draws[:, 36])
    low = np.minimum(open_, adj) * (1.0 - 0.6 * typical * draws[:, 37])
    volume = np.floor(p["volume"] * (0.6 + 0.8 * draws[:, 38]) * (1.0 + 0.8 * stress))
    panel = {"adj_close": adj, "close": close, "open": open_, "high": high, "low": low,
             "volume": volume}
    for values in panel.values():
        values[:start] = np.nan
    panel["total"] = growth - 1.0
    return panel


def build_sample(seed: int = DEFAULT_SEED, years: int = DEFAULT_YEARS) -> dict:
    """The whole sample in memory: 'dates' (YYYY-MM-DD strings), 'tickers',
    one (sessions x tickers) array per price file and for 'volume', and
    'indices' (name -> array). Nothing is read from or written to disk."""
    if isinstance(years, bool) or not isinstance(years, int) or not MIN_YEARS <= years <= MAX_YEARS:
        raise ValueError(f"years must be an integer from {MIN_YEARS} to {MAX_YEARS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer from 0 to 2**32 - 1")
    unknown = sorted(set(INDEX_UNIVERSE) - {"^IRX", "^TNX", "^VIX", "^VIX3M", "^GSPC"})
    if unknown:
        raise ValueError(f"no synthetic model for index series: {', '.join(unknown)}")
    dates = _sessions(years)
    factors = _factor_paths(seed, len(dates))
    tickers = sorted(set(ETF_UNIVERSE) | set(STOCK_UNIVERSE) | set(OPERATIONAL_UNIVERSE))
    panels = {}
    for ticker in tickers:  # leaders and unpaired tickers first
        if ticker not in PAIR_FOLLOWERS or PAIR_FOLLOWERS[ticker][0] not in tickers:
            panels[ticker] = _ticker_panel(ticker, seed, dates, factors)
    for ticker in tickers:
        if ticker not in panels:
            base, multiple = PAIR_FOLLOWERS[ticker]
            panels[ticker] = _ticker_panel(ticker, seed, dates, factors,
                                           leader=(panels[base]["total"], multiple))

    indices = dict(factors["indices"])
    market = panels[MARKET_FUND]["total"] if MARKET_FUND in panels else factors["returns"]["equity"]
    growth = 1.0 + (market - MARKET_FUND_YIELD / YEAR)
    growth[0] = 1.0
    indices["^GSPC"] = INDEX_START * np.cumprod(growth)

    sample = {"dates": dates, "tickers": tickers, "seed": seed, "years": years,
              "indices": {name: indices[name] for name in sorted(INDEX_UNIVERSE)}}
    for field in (*PRICE_FILES, "volume"):
        sample[field] = np.column_stack([panels[t][field] for t in tickers])
    return sample


def _csv_text(header: list, dates: list, columns: np.ndarray, fmt: str) -> str:
    lines = [",".join(header)]
    for day, row in zip(dates, columns.tolist()):
        lines.append(",".join([day] + ["" if v != v else fmt % v for v in row]))  # v != v: not listed yet
    return "\n".join(lines) + "\n"


def sample_files(sample: dict) -> dict:
    """File name -> text, exactly as written to the output directory."""
    dates, tickers = sample["dates"], sample["tickers"]
    files = {}
    for field in PRICE_FILES:
        files[f"{field}.csv"] = _csv_text(["Date", *tickers], dates, sample[field], PRICE_FORMAT)
    files["volume.csv"] = _csv_text(["Date", *tickers], dates, sample["volume"], "%d")
    names = list(sample["indices"])
    files["indices.csv"] = _csv_text(
        ["Date", *names], dates, np.column_stack([sample["indices"][k] for k in names]),
        INDEX_FORMAT)
    rows = ["Ticker,first,last,rows"]
    listed = ~np.isnan(sample["adj_close"])
    for j, ticker in enumerate(tickers):
        held = np.flatnonzero(listed[:, j])
        rows.append(f"{ticker},{dates[held[0]]},{dates[held[-1]]},{len(held)}")
    files["coverage.csv"] = "\n".join(rows) + "\n"
    files[README_NAME] = "\n".join([
        MARKER,
        "",
        f"Written by scripts/make_sample_data.py --seed {sample['seed']} --years {sample['years']}:",
        f"{len(tickers)} tickers, {len(dates)} NYSE sessions, {dates[0]} .. {dates[-1]}.",
        "",
        "Every number in this directory comes from a seeded random model. The ticker",
        "symbols only label the columns the code expects; no value is, or is derived",
        "from, a market price. Results computed from these files say nothing about any",
        "market or trading rule.",
        "",
        "Use: set QCORE_DATA_DIR to this directory, then run the market-data scripts.",
        "The first line of this file marks the directory as a sample: the generator",
        "overwrites a directory that carries it and refuses any other non-empty one.",
        "",
    ])
    return files


def refusal(out: Path) -> str | None:
    """Why `out` must not be written to; None for a new, an empty or a sample directory."""
    if not out.exists():
        return None
    if not out.is_dir():
        return f"{out} exists and is not a directory"
    if all(entry.name == ".DS_Store" for entry in out.iterdir()):
        return None
    readme = out / README_NAME
    try:
        marked = readme.is_file() and readme.read_text(encoding="utf-8").splitlines()[:1] == [MARKER]
    except (OSError, UnicodeDecodeError):
        marked = False
    if marked:
        return None
    return (f"{out} already holds files that are not a sample written by this script "
            f"(no {README_NAME} starting with the sample marker). A downloaded cache is "
            "never overwritten: name a new or empty directory with --out")


def write_sample(out: Path, sample: dict) -> list:
    """Write the sample to `out`; returns the file names written. The marker
    file goes first, so an interrupted run can simply be repeated."""
    reason = refusal(out)
    if reason:
        raise ValueError(reason)
    out.mkdir(parents=True, exist_ok=True)
    files = sample_files(sample)
    order = [README_NAME] + [name for name in files if name != README_NAME]
    for name in order:
        staged = out / f".{name}.tmp-{os.getpid()}"
        try:
            with staged.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(files[name])
            os.replace(staged, out / name)
        finally:
            staged.unlink(missing_ok=True)
    return order


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_sample_data.py", allow_abbrev=False,
        description="Write a small synthetic sample of the market-data cache (simulated values).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: sample_data/ under the project root); "
                             "must be new, empty or an earlier sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed, 0 to 2**32 - 1 (default {DEFAULT_SEED})")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"calendar years of sessions, {MIN_YEARS} to {MAX_YEARS} "
                             f"(default {DEFAULT_YEARS})")
    args = parser.parse_args(argv)
    if not MIN_YEARS <= args.years <= MAX_YEARS:
        parser.error(f"--years must be from {MIN_YEARS} to {MAX_YEARS}")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be from 0 to 2**32 - 1")
    out = args.out.expanduser()
    reason = refusal(out)
    if reason:
        print(f"sample data not written: {reason}", file=sys.stderr)
        return EXIT_REFUSED
    sample = build_sample(args.seed, args.years)
    try:
        written = write_sample(out, sample)
    except OSError as exc:
        print(f"sample data not written: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    dates = sample["dates"]
    print(f"wrote {len(written)} files to {out}: SYNTHETIC sample, {len(sample['tickers'])} "
          f"tickers x {len(dates)} sessions, {dates[0]} .. {dates[-1]} (seed {args.seed})")
    print("point the loaders at it, for example:")
    print(f'  export QCORE_DATA_DIR="{out.resolve()}"')
    print("  python scripts/data_quality.py")
    print("  python src/strategies/mean_reversion.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
