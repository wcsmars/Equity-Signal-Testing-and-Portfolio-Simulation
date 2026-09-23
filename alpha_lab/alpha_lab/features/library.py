"""Built-in feature library.

Every feature here is a pure map from MarketData to a wide panel aligned
exactly to ``data.close`` (same index, same columns), and obeys the timing
law of CONVENTIONS.md: the value at date t uses information through close t
only (truncation invariance). Only trailing operations are used — shifts by
non-negative amounts and rolling windows; never negative shifts, never
full-sample statistics.

Registered names and parameter names are a public contract: configs across
the project instantiate these via ``FEATURES.create(name, **params)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import Feature
from alpha_lab.core.registry import Registry
from alpha_lab.core.types import TRADING_DAYS_PER_YEAR, MarketData

#: the feature registry — configs resolve feature names against this
FEATURES = Registry("feature")


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Validate an integer parameter at construction time (ConfigError)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


@FEATURES.register("returns")
class WindowReturn(Feature):
    """Trailing ``window``-day simple return: close_t / close_{t-window} - 1.

    ``window=1`` is the daily close-to-close return r_t of CONVENTIONS.md.
    close.shift(window) needs a bar at t-window, so the first non-NaN value
    needs window+1 closes: lookback=window+1.
    """

    name = "returns"

    def __init__(self, window: int = 1) -> None:
        self.window = _check_int(window, "window", minimum=1)
        self.lookback = self.window + 1

    def compute(self, data: MarketData) -> pd.DataFrame:
        close = data.close
        return close / close.shift(self.window) - 1.0


@FEATURES.register("momentum")
class Momentum(Feature):
    """Return from t-window to t-skip: close_{t-skip} / close_{t-window} - 1.

    The defaults (window=252, skip=21) give classic 12-1 momentum — the
    trailing year's return excluding the most recent month (which tends to
    reverse). Requires window > skip >= 0. close.shift(window) needs a bar
    at t-window: lookback=window+1.
    """

    name = "momentum"

    def __init__(self, window: int = 252, skip: int = 21) -> None:
        self.window = _check_int(window, "window", minimum=1)
        self.skip = _check_int(skip, "skip", minimum=0)
        if self.window <= self.skip:
            raise ConfigError(
                f"momentum requires window > skip >= 0, got window={self.window}, skip={self.skip}"
            )
        self.lookback = self.window + 1

    def compute(self, data: MarketData) -> pd.DataFrame:
        close = data.close
        return close.shift(self.skip) / close.shift(self.window) - 1.0


@FEATURES.register("realized_vol")
class RealizedVol(Feature):
    """Annualized realized volatility of daily returns over ``window`` days.

    Rolling std of close-to-close returns (min_periods=window) scaled by
    sqrt(252). Needs window returns, i.e. window+1 closes: lookback=window+1.
    """

    name = "realized_vol"

    def __init__(self, window: int = 63) -> None:
        self.window = _check_int(window, "window", minimum=2)
        self.lookback = self.window + 1

    def compute(self, data: MarketData) -> pd.DataFrame:
        rets = data.close.pct_change(fill_method=None)
        vol = rets.rolling(self.window, min_periods=self.window).std()
        return vol * np.sqrt(TRADING_DAYS_PER_YEAR)


@FEATURES.register("zscore_return")
class ZScoreReturn(Feature):
    """Rolling z-score of the daily return: (r_t - mean) / std over ``window``.

    Mean and std are trailing rolling statistics (min_periods=window), never
    full-sample — full-sample standardization is a forbidden pattern.
    lookback=window+1 (window returns need window+1 closes).
    """

    name = "zscore_return"

    def __init__(self, window: int = 126) -> None:
        self.window = _check_int(window, "window", minimum=2)
        self.lookback = self.window + 1

    def compute(self, data: MarketData) -> pd.DataFrame:
        rets = data.close.pct_change(fill_method=None)
        mean = rets.rolling(self.window, min_periods=self.window).mean()
        std = rets.rolling(self.window, min_periods=self.window).std()
        return (rets - mean) / std


@FEATURES.register("adv_dollars")
class ADVDollars(Feature):
    """Average daily dollar volume over ``window`` days.

    Rolling mean (min_periods=min(window, max(5, window//2)), so short
    windows are valid) of volume * unadjusted close. Requires
    ``data.volume``; falls back to the adjusted close only if
    ``data.unadjusted_close`` is missing. ``lookback`` = window, the bars
    needed for a full window (values can appear from min_periods bars on).
    """

    name = "adv_dollars"

    def __init__(self, window: int = 21) -> None:
        self.window = _check_int(window, "window", minimum=1)
        self.lookback = self.window

    def compute(self, data: MarketData) -> pd.DataFrame:
        if data.volume is None:
            raise DataError("adv_dollars requires data.volume")
        # volume is raw shares, so dollar volume needs the raw (unadjusted)
        # close: the adjusted close is wrong before any split (a 4:1 split
        # makes adjusted-close dollar volume 4x too small pre-split).
        price = data.unadjusted_close
        if price is None:
            price = data.close
        dollars = data.volume * price
        # min(window, ...): pandas rejects min_periods > window, and windows
        # of 1-4 days are legal at construction time
        min_periods = min(self.window, max(5, self.window // 2))
        return dollars.rolling(self.window, min_periods=min_periods).mean()


@FEATURES.register("rsi")
class RSI(Feature):
    """Relative Strength Index over ``window`` days, in [0, 100].

    Standard construction from daily close changes: rolling mean of gains
    vs rolling mean of losses (min_periods=window), RSI = 100 * avg_gain /
    (avg_gain + avg_loss). Needs window changes: lookback=window+1.
    """

    name = "rsi"

    def __init__(self, window: int = 14) -> None:
        self.window = _check_int(window, "window", minimum=1)
        self.lookback = self.window + 1

    def compute(self, data: MarketData) -> pd.DataFrame:
        delta = data.close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(self.window, min_periods=self.window).mean()
        avg_loss = loss.rolling(self.window, min_periods=self.window).mean()
        denom = avg_gain + avg_loss
        rsi = 100.0 * avg_gain / denom
        # A fully flat window gives 0/0: define it as neutral 50. mask (not
        # where) so cells with NaN denominators stay NaN.
        return rsi.mask(denom == 0.0, 50.0)
