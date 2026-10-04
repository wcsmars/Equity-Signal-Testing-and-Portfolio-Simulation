"""Monthly stock momentum with a reversal tilt. Rejected: biased universe.

Hypothesis
    Stocks that outperformed over the past year, leaving out the latest
    month, tend to keep outperforming for some months: investors react
    slowly to company news, and holders sell winners early to realise gains.
    The latest month tends to reverse in part as short-term price pressure
    fades. The other side is the holder who sells a winner too soon.

Rule
    Universe: the fixed list of 50 large US stocks in
    qcore.data.STOCK_UNIVERSE. It is today's list, not point-in-time index
    membership.
    At each month-end close, for every stock with 13 month-end closes and a
    volatility estimate, from dividend-adjusted closes:
      mom   = close one month ago / close twelve months ago - 1
      ret1m = close now / close one month ago - 1
      vol   = standard deviation of daily returns over 63 trading days
      score = z(mom) - 0.25 * z(ret1m), z-scores taken across the eligible
              stocks of that month.
    With fewer than 20 eligible stocks the rule holds cash. Otherwise it
    holds the top K=5 by score, weighted in proportion to 1/vol, fully
    invested, long-only. The monthly target is held constant between
    decisions, so the engine re-trades to it daily. The engine assumes
    execution at the decision close.
    Costs are modelling assumptions: the commissions and fees of
    qcore.costs plus 5 bps of slippage per side.

Variants tried
    12: K {5, 8, 10, 15} x reversal weight {0, 0.25, 0.5}.
    scripts/xsec_stock_mom_sweep.py runs the grid through build_weights and
    prints the equal-weight comparison quoted in the verdict.

Selection
    In-sample Sharpe (data before 2018-01-01): K=5, weight 0.25, at 0.75;
    K=5, weight 0.5 is next at 0.74. Out-of-sample results (2018-01-01
    onwards) are reported, not selected on.

Result
    This code on data ending 2026-07-01, net of the modelled costs, Sharpe
    ratios in excess of the cash rate, from 2001-01-31:
    Sharpe 0.77 full sample / 0.75 in sample / 0.82 out of sample;
    CAGR 20.25%, volatility 26.18%, maximum drawdown -41.79%. Turnover is
    13.7x a year (buys plus sells) and costs take 2.51% a year (overstated
    in the early years for stocks with large later splits; see
    qcore.backtest).

Verdict
    Rejected; not part of the combined portfolio. The figures above are not
    an investable result. The universe is a list of companies that are large
    today, applied back to 2001: firms that failed or shrank are absent, so
    the backtest could only choose among eventual winners, and the 2018+
    segment is selected by the same hindsight.
    An equal-weight basket of the same universe, ordered at month-ends and
    left to drift, has Sharpe 0.70 full sample / 0.65 in sample / 0.82 out
    of sample over the same window. The strategy adds 5.6 percentage points
    of CAGR to it, with an active Sharpe of 0.37, and out of sample its
    Sharpe is the same as the basket's. Even that excess is measured inside
    the biased universe. Neither the absolute figures nor the comparison
    estimates an unbiased result; the rule cannot be evaluated without
    point-in-time index membership and delisted names.

Run the module to print the metrics for the locally obtained cache.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcore.backtest import metrics, run_backtest
from qcore.calendar import confirmed_month_ends
from qcore.costs import IBKRHKCostModel
from qcore.data import STOCK_UNIVERSE, load_prices

# --- recorded variant: K and LAM with the best in-sample Sharpe of the 12 ---
K = 5            # number of names held
LAM = 0.25       # short-term-reversal tilt weight
VOL_WIN = 63     # trailing window (days) for inverse-vol weights
MIN_NAMES = 20   # minimum eligible names before trading
SLIPPAGE_BPS = 5.0


def run_name(K: int = K, lam: float = LAM) -> str:
    """Name of the backtest run for one (K, lam) variant."""
    return f"xsec_stock_mom_K{K}_lam{lam:g}"


def build_weights(px: pd.DataFrame, K: int = K, lam: float = LAM) -> pd.DataFrame:
    """Monthly target weights on month-end trading dates, ffilled to daily.

    Uses only information available at each month-end close: momentum and
    reversal from past month-end closes, vol from trailing daily returns,
    z-scores computed cross-sectionally (per date, never across time).
    """
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or not 1 <= K <= len(px.columns):
        raise ValueError("K must be an integer within the stock universe")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    me_dates = confirmed_month_ends(px.index)  # live-edge safe (qcore.calendar)
    pm = px.loc[me_dates]

    mom = pm.shift(1) / pm.shift(12) - 1.0
    ret1m = pm / pm.shift(1) - 1.0
    vol = px.pct_change(fill_method=None).rolling(VOL_WIN).std().loc[me_dates]

    elig = mom.notna() & ret1m.notna() & vol.notna() & (vol > 0)

    def zscore(df):
        mu = df.mean(axis=1)
        sd = df.std(axis=1)
        return df.sub(mu, axis=0).div(sd.where(sd != 0, 1.0), axis=0)

    score = zscore(mom.where(elig)) - lam * zscore(ret1m.where(elig))

    w = pd.DataFrame(0.0, index=pm.index, columns=pm.columns)
    for dt in pm.index:
        s = score.loc[dt].dropna()
        if len(s) < MIN_NAMES:
            continue
        top = s.nlargest(K).index
        iv = 1.0 / vol.loc[dt, top]
        w.loc[dt, top] = (iv / iv.sum()).values

    return w.reindex(px.index).ffill().fillna(0.0)


def main():
    px = load_prices()[STOCK_UNIVERSE]
    w = build_weights(px)
    res = run_backtest(w, px, IBKRHKCostModel(slippage_bps=SLIPPAGE_BPS),
                       name=run_name())
    m = metrics(res)
    print(json.dumps(m, indent=2))
    return m


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    return parser.parse_args(argv)


if __name__ == "__main__":
    _parse_args()
    try:
        main()
    except FileNotFoundError as exc:  # no data cache: the loader's one line, no traceback
        sys.exit(str(exc))
