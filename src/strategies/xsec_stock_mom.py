"""Monthly stock momentum with a short-term reversal tilt.

For eligible stocks, rank z(12-1 momentum) - 0.25*z(one-month return).
With at least 20 eligible names, hold the top five using inverse 63-day
volatility weights; otherwise hold modeled cash. The engine assumes
execution at the decision close and 5 bps per-side slippage.

The fixed mega-cap universe is survivorship-biased and omits historical
membership and delisted names. Neither absolute returns nor performance
relative to that same universe estimates an unbiased investable result.
Run the module to print descriptive metrics for the locally obtained cache.
"""
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

# --- retained example parameters ---
K = 5            # number of names held
LAM = 0.25       # short-term-reversal tilt weight
VOL_WIN = 63     # trailing window (days) for inverse-vol weights
MIN_NAMES = 20   # minimum eligible names before trading
SLIPPAGE_BPS = 5.0


def build_weights(px: pd.DataFrame, K: int = K, lam: float = LAM) -> pd.DataFrame:
    """Monthly target weights on month-end trading dates, ffilled to daily.

    Uses only information available at each month-end close: momentum and
    reversal from past month-end closes, vol from trailing daily returns,
    z-scores computed cross-sectionally (per date, never across time).
    """
    me_dates = confirmed_month_ends(px.index)  # live-edge safe (qcore.calendar)
    pm = px.loc[me_dates]

    mom = pm.shift(1) / pm.shift(12) - 1.0
    ret1m = pm / pm.shift(1) - 1.0
    vol = px.pct_change(fill_method=None).rolling(VOL_WIN).std().loc[me_dates]

    elig = mom.notna() & ret1m.notna() & vol.notna() & (vol > 0)

    def zscore(df):
        mu = df.mean(axis=1)
        sd = df.std(axis=1)
        return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

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
                       name=f"xsec_stock_mom_K{K}_lam{LAM:g}")
    m = metrics(res)
    print(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    main()
