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

from alpha_lab.config.schema import DataConfig, iso_date
from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import DataSource
from alpha_lab.core.registry import Registry
from alpha_lab.core.types import ISO_DATE_PATTERN, MarketData
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

    Dates must be ISO 8601 in both layouts: ``YYYY-MM-DD`` (optionally with a
    time of day, no UTC offset) or ``YYYYMMDD``. Other spellings such as
    ``10/01/2024`` raise ``DataError``; the day/month order is never guessed.
    A wide panel with a repeated or blank ticker header, or an optional panel
    that shares no ticker or no date with ``close.csv``, also raises.

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
        for bound, value in (("start", start), ("end", end)):
            if value is not None:
                iso_date(value, f"csv source {bound}")
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
        close = _read_panel_csv(close_path)
        frames = {"close": close}
        for field in (*_PRICE_FIELDS, "universe"):
            file = self.path / f"{field}.csv"
            if not file.exists():
                continue
            frame = _read_panel_csv(file)
            # Optional panels are aligned to close afterwards. One that shares
            # no ticker or no date with it (headers in another case, another
            # date range) would turn into an all-missing panel without notice.
            for axis, theirs, ours in (
                ("tickers", frame.columns, close.columns),
                ("dates", frame.index, close.index),
            ):
                if len(ours) and not len(theirs.intersection(ours)):
                    raise DataError(f"{file}: shares no {axis} with close.csv")
            frames[field] = frame
        return frames

    def _read_long(self) -> dict[str, pd.DataFrame]:
        if not self.path.is_file():
            raise DataError(f"long csv source expects a single file: {self.path}")
        try:
            # tickers stay literal strings, as in the wide format and config:
            # no numeric inference ('10001') and no NA conversion ('NA')
            raw = pd.read_csv(
                self.path,
                dtype={"date": str},
                converters={"ticker": str},
                float_precision="round_trip",
            )
        except (ValueError, KeyError) as exc:
            raise DataError(f"{self.path}: could not parse long csv ({exc})") from exc
        required = {"date", "ticker", "close"}
        missing = required - set(raw.columns)
        if missing:
            raise DataError(f"{self.path}: long csv missing columns {sorted(missing)}")
        raw["date"] = _parse_dates(raw["date"], self.path)
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


def _parse_dates(values, path: Path) -> pd.DatetimeIndex:
    """Parse a column of date strings strictly as ISO dates.

    Format inference is not used: it reads ``10/01/2024`` month-first and
    ``15/01/2024`` day-first within one file, after which sorting reorders the
    rows and the price history is scrambled without any error.
    """
    text = pd.Series(np.asarray(values, dtype=object))
    bad = text.isna() | ~text.astype(str).str.fullmatch(ISO_DATE_PATTERN)
    if bad.any():
        row = int(bad.to_numpy().argmax())
        found = "is blank" if pd.isna(text.iloc[row]) else f"{text.iloc[row]!r} is not an ISO date"
        raise DataError(
            f"{path}: date in data row {row + 1} {found}; "
            "use YYYY-MM-DD (optionally with a time, no UTC offset) or YYYYMMDD"
        )
    try:
        return pd.DatetimeIndex(pd.to_datetime(text.to_numpy(), format="ISO8601"))
    except (ValueError, TypeError) as exc:
        reason = str(exc).split(". You might want to try")[0].splitlines()[0]
        raise DataError(f"{path}: could not parse dates ({reason})") from exc


def _read_panel_csv(path: Path) -> pd.DataFrame:
    """Read one wide panel: first column 'date' (ISO dates, the index), rest tickers."""
    try:
        header = pd.read_csv(path, header=None, nrows=1, dtype=str, keep_default_na=False)
        # round_trip: the default float parser can be one unit in the last
        # place away from the value written in the file
        frame = pd.read_csv(
            path, index_col="date", dtype={"date": str}, float_precision="round_trip"
        )
    except (ValueError, KeyError) as exc:
        raise DataError(f"{path}: first column must be 'date' ({exc})") from exc
    # read_csv renames a repeated header ('A' -> 'A.1') and names a blank one
    # 'Unnamed: n', so both would load as extra tickers; check the raw header.
    names = [str(cell).strip() for cell in header.iloc[0]]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise DataError(f"{path}: repeated column headers {repeated}")
    if "" in names:
        raise DataError(f"{path}: blank column header (column {names.index('') + 1})")
    frame.index = _parse_dates(frame.index, path).rename("date")
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
