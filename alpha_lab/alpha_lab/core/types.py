"""MarketData: the single container all modules consume.

Wide-panel convention: every field is a DataFrame with rows = trading dates
(tz-naive DatetimeIndex, ascending, unique) and columns = tickers, aligned
exactly to ``close``. See CONVENTIONS.md for the information-timing rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from alpha_lab.core.errors import DataError

TRADING_DAYS_PER_YEAR = 252

#: optional fields, all aligned to close when present
_OPTIONAL_FIELDS = ("open", "high", "low", "volume", "unadjusted_close", "universe")


@dataclass
class MarketData:
    """Aligned daily panels for one universe of equities.

    close/open/high/low are split- and dividend-adjusted. ``unadjusted_close``
    is the raw close — all share math (per-share commissions, share counts,
    dollar volume) must use it. ``volume`` is raw shares. ``universe`` is a
    boolean point-in-time membership panel; assets carry NaN prices outside
    their listed life.
    """

    close: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    unadjusted_close: pd.DataFrame | None = None
    universe: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.validate()

    # -- construction --------------------------------------------------

    @classmethod
    def from_frames(cls, close: pd.DataFrame, **fields: pd.DataFrame | None) -> "MarketData":
        """Build a MarketData, reindexing optional fields to close's axes.

        Unlike the constructor (which demands exact alignment), this aligns
        for you: missing cells become NaN, universe cells become False.
        """
        unknown = set(fields) - set(_OPTIONAL_FIELDS)
        if unknown:
            raise DataError(f"unknown MarketData fields: {sorted(unknown)}")
        aligned: dict[str, pd.DataFrame | None] = {}
        for name in _OPTIONAL_FIELDS:
            frame = fields.get(name)
            if frame is None:
                aligned[name] = None
            elif name == "universe":
                if (not all(is_bool_dtype(dt) for dt in frame.dtypes)
                        and not frame.map(lambda value: pd.isna(value) or isinstance(value, (bool, np.bool_))).all().all()):
                    raise DataError("universe must contain boolean values (missing cells are False)")
                aligned[name] = (
                    frame.reindex(index=close.index, columns=close.columns)
                    .fillna(False)
                    .astype(bool)
                )
            else:
                aligned[name] = frame.reindex(index=close.index, columns=close.columns)
        return cls(close=close, **aligned)

    # -- validation ----------------------------------------------------

    def validate(self) -> None:
        if not isinstance(self.close, pd.DataFrame):
            raise DataError("close must be a DataFrame")
        idx = self.close.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise DataError("close index must be a DatetimeIndex")
        if idx.tz is not None:
            raise DataError("close index must be tz-naive")
        if idx.hasnans:
            raise DataError("close index contains missing dates")
        if not idx.is_monotonic_increasing:
            raise DataError("close index must be ascending")
        if idx.has_duplicates:
            raise DataError("close index has duplicate dates")
        if self.close.columns.has_duplicates:
            raise DataError("close has duplicate tickers")
        for name in ("close", *_OPTIONAL_FIELDS):
            frame = getattr(self, name)
            if frame is None:
                continue
            if not isinstance(frame, pd.DataFrame):
                raise DataError(f"{name} must be a DataFrame")
            if not frame.index.equals(idx) or not frame.columns.equals(self.close.columns):
                raise DataError(
                    f"{name} is not aligned with close; use MarketData.from_frames to align"
                )
            if name == "universe":
                continue
            if not all(is_numeric_dtype(dt) and not is_bool_dtype(dt) for dt in frame.dtypes):
                raise DataError(f"{name} must contain numeric values")
            values = frame.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                raise DataError(f"{name} contains infinite values")
            if name != "volume" and (values <= 0).any():
                raise DataError(f"non-positive {name} prices")
        if self.universe is not None and not all(dt == bool for dt in self.universe.dtypes):
            raise DataError("universe must be boolean")

    # -- accessors -----------------------------------------------------

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def returns(self) -> pd.DataFrame:
        """Close-to-close simple returns, r_t = close_t / close_{t-1} - 1."""
        return self.close.pct_change(fill_method=None)

    def effective_universe(self) -> pd.DataFrame:
        """Boolean panel: universe member AND price available at t."""
        eff = self.close.notna()
        if self.universe is not None:
            eff = eff & self.universe
        return eff

    # -- slicing (the tool for truncation-invariance tests) -------------

    def slice_until(self, t) -> "MarketData":
        """All data with date <= t. This is 'the world as known at close t'."""
        pos = self.dates.searchsorted(pd.Timestamp(t), side="right")
        return self._iloc(slice(0, int(pos)))

    def slice_range(self, start=None, end=None) -> "MarketData":
        """Data with start <= date <= end (either bound optional)."""
        lo = 0 if start is None else int(self.dates.searchsorted(pd.Timestamp(start), side="left"))
        hi = len(self.dates) if end is None else int(
            self.dates.searchsorted(pd.Timestamp(end), side="right")
        )
        return self._iloc(slice(lo, hi))

    def _iloc(self, rows: slice) -> "MarketData":
        def cut(frame: pd.DataFrame | None) -> pd.DataFrame | None:
            return None if frame is None else frame.iloc[rows]

        return MarketData(
            close=self.close.iloc[rows],
            open=cut(self.open),
            high=cut(self.high),
            low=cut(self.low),
            volume=cut(self.volume),
            unadjusted_close=cut(self.unadjusted_close),
            universe=cut(self.universe),
        )
