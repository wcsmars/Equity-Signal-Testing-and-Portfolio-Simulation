"""Data sources: the synthetic generator and CSV readers, registered by name.

A source produces a :class:`~alpha_lab.core.types.MarketData`. Both concrete
sources are registered in ``SOURCES`` so configs can instantiate them by name;
``source_from_config`` maps a validated ``DataConfig`` onto a source instance.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config.schema import DataConfig
from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import DataSource
from alpha_lab.core.registry import Registry
from alpha_lab.core.types import MarketData
from alpha_lab.data.synthetic import make_market

SOURCES = Registry("data_source")

#: optional price/volume panels a CSV source may provide (besides close)
_PRICE_FIELDS = ("open", "high", "low", "volume", "unadjusted_close")

#: value columns accepted in a long-format file
_LONG_FIELDS = ("close",) + _PRICE_FIELDS

_TRUE_STRINGS = {"true", "t", "1", "1.0", "yes", "y"}
_FALSE_STRINGS = {"false", "f", "0", "0.0", "no", "n", ""}


@SOURCES.register("synthetic")
class SyntheticSource(DataSource):
    """Deterministic synthetic panel; constructor mirrors ``SyntheticConfig``.

    ``load`` delegates to :func:`alpha_lab.data.synthetic.make_market`, so the
    same arguments always yield an identical panel.
    """

    name = "synthetic"

    def __init__(
        self,
        n_assets: int = 20,
        n_days: int = 1512,
        seed: int = 7,
        start: str = "2015-01-02",
        drift_dispersion: float = 0.10,
        base_vol: float = 0.20,
        split_asset: bool = True,
        universe_churn: bool = True,
    ) -> None:
        self.n_assets = n_assets
        self.n_days = n_days
        self.seed = seed
        self.start = start
        self.drift_dispersion = drift_dispersion
        self.base_vol = base_vol
        self.split_asset = split_asset
        self.universe_churn = universe_churn

    def load(self) -> MarketData:
        return make_market(
            n_assets=self.n_assets,
            n_days=self.n_days,
            seed=self.seed,
            start=self.start,
            drift_dispersion=self.drift_dispersion,
            base_vol=self.base_vol,
            split_asset=self.split_asset,
            universe_churn=self.universe_churn,
        )


@SOURCES.register("csv")
class CSVSource(DataSource):
    """Reads MarketData from CSV files in wide or long layout.

    wide: ``path`` is a directory containing ``close.csv`` (required) and
    optionally ``open.csv``, ``high.csv``, ``low.csv``, ``volume.csv``,
    ``unadjusted_close.csv``, ``universe.csv``. Each file has a first column
    named ``date`` (becomes the index); remaining columns are tickers.

    long: ``path`` is a single file with columns ``date,ticker,close`` plus
    optionally ``open,high,low,volume,unadjusted_close``; each value column is
    pivoted to a wide panel.

    ``tickers`` / ``start`` / ``end`` subset the loaded panel; a requested
    ticker absent from the files raises ``DataError`` naming it.
    """

    name = "csv"

    def __init__(
        self,
        path: str | Path,
        format: str = "wide",
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        if format not in ("wide", "long"):
            raise ConfigError(f"unknown csv format '{format}'; expected 'wide' or 'long'")
        self.path = Path(path)
        self.format = format
        self.tickers = list(tickers) if tickers is not None else None
        self.start = start
        self.end = end

    def load(self) -> MarketData:
        frames = self._read_wide() if self.format == "wide" else self._read_long()
        close = self._select(frames.pop("close"))
        universe = frames.pop("universe", None)
        if universe is not None:
            universe = _parse_universe(universe)
        # from_frames re-aligns every optional panel to the (subset) close axes
        return MarketData.from_frames(close, universe=universe, **frames)

    # -- readers --------------------------------------------------------

    def _read_wide(self) -> dict[str, pd.DataFrame]:
        if not self.path.is_dir():
            raise DataError(f"wide csv source expects a directory of panels: {self.path}")
        close_path = self.path / "close.csv"
        if not close_path.exists():
            raise DataError(f"required close.csv not found in {self.path}")
        frames = {"close": _read_panel_csv(close_path)}
        for field in (*_PRICE_FIELDS, "universe"):
            file = self.path / f"{field}.csv"
            if file.exists():
                frames[field] = _read_panel_csv(file)
        return frames

    def _read_long(self) -> dict[str, pd.DataFrame]:
        if not self.path.is_file():
            raise DataError(f"long csv source expects a single file: {self.path}")
        try:
            raw = pd.read_csv(self.path, parse_dates=["date"])
        except (ValueError, KeyError) as exc:
            raise DataError(f"{self.path}: could not parse long csv ({exc})") from exc
        required = {"date", "ticker", "close"}
        missing = required - set(raw.columns)
        if missing:
            raise DataError(f"{self.path}: long csv missing columns {sorted(missing)}")
        value_cols = [c for c in raw.columns if c not in ("date", "ticker")]
        unknown = [c for c in value_cols if c not in _LONG_FIELDS]
        if unknown:
            raise DataError(
                f"{self.path}: unknown long csv columns {unknown}; allowed: {list(_LONG_FIELDS)}"
            )
        frames: dict[str, pd.DataFrame] = {}
        for col in value_cols:
            try:
                frames[col] = raw.pivot(index="date", columns="ticker", values=col).sort_index()
            except ValueError as exc:  # duplicate (date, ticker) rows
                raise DataError(f"{self.path}: cannot pivot '{col}' to a panel ({exc})") from exc
        return frames

    # -- subsetting -----------------------------------------------------

    def _select(self, close: pd.DataFrame) -> pd.DataFrame:
        if self.tickers is not None:
            unknown = [t for t in self.tickers if t not in close.columns]
            if unknown:
                raise DataError(
                    f"unknown tickers {unknown}; available: {list(close.columns)}"
                )
            close = close.loc[:, self.tickers]
        if self.start is not None:
            close = close.loc[close.index >= pd.Timestamp(self.start)]
        if self.end is not None:
            close = close.loc[close.index <= pd.Timestamp(self.end)]
        return close


def source_from_config(cfg: DataConfig) -> DataSource:
    """Instantiate the DataSource described by a validated ``DataConfig``."""
    if cfg.source == "synthetic":
        return SyntheticSource(**dataclasses.asdict(cfg.synthetic))
    if cfg.source == "csv":
        return CSVSource(
            path=cfg.path,
            format=cfg.format,
            tickers=cfg.tickers,
            start=cfg.start,
            end=cfg.end,
        )
    raise ConfigError(f"unknown data source '{cfg.source}'; available: {SOURCES.names()}")


# -- helpers -------------------------------------------------------------


def _read_panel_csv(path: Path) -> pd.DataFrame:
    """Read one wide panel: first column 'date' (parsed index), rest tickers."""
    try:
        frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    except (ValueError, KeyError) as exc:
        raise DataError(f"{path}: first column must be 'date' ({exc})") from exc
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.sort_index()


def _parse_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a universe panel of 0/1 or true/false cells to bool (NaN -> False)."""

    def one(value) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return False
            if value in (0, 1):
                return bool(value)
            raise DataError(f"universe value {value!r} is not 0/1")
        text = str(value).strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise DataError(f"universe value {value!r} is not boolean-like")

    return frame.map(one).astype(bool)
