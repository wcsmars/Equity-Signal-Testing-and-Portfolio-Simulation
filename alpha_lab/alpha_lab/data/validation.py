"""Data quality checks that report issues instead of raising.

``validate_market`` scans a MarketData panel for suspicious values (extreme
moves, negative volume, price gaps, stale prices, unadjusted-split fingerprints,
universe members without prices, contradictory price bars, optional fields
that are missing or zero for a priced ticker) and returns a
``ValidationReport``. It never raises: QA is advisory, structural problems are
the job of ``MarketData.validate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from alpha_lab.core.types import MarketData

#: one-day price ratios left behind by a split that was not adjusted for:
#: forward N-for-1 and 3-for-2 splits, and reverse 1-for-N splits
_SPLIT_RATIOS = (
    ("2-for-1", 1 / 2),
    ("3-for-1", 1 / 3),
    ("4-for-1", 1 / 4),
    ("5-for-1", 1 / 5),
    ("10-for-1", 1 / 10),
    ("3-for-2", 2 / 3),
    ("1-for-2", 2.0),
    ("1-for-3", 3.0),
    ("1-for-4", 4.0),
    ("1-for-5", 5.0),
    ("1-for-10", 10.0),
)

#: relative slack before open/close count as outside the day's high-low range
_BAR_TOLERANCE = 1e-9


@dataclass
class Issue:
    """One finding: severity ('error' | 'warning'), machine code, location, text."""

    severity: str
    code: str
    ticker: str | None
    date: pd.Timestamp | None
    message: str

    def __str__(self) -> str:
        where = " ".join(
            part
            for part in (
                str(self.ticker) if self.ticker is not None else "",
                pd.Timestamp(self.date).strftime("%Y-%m-%d") if self.date is not None else "",
            )
            if part
        )
        prefix = f"[{self.severity}] {self.code}"
        return f"{prefix} {where}: {self.message}" if where else f"{prefix}: {self.message}"


@dataclass
class ValidationReport:
    """Collected issues from one ``validate_market`` pass."""

    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff no error-severity issues (warnings do not fail QA)."""
        return not any(issue.severity == "error" for issue in self.issues)

    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def summary(self) -> str:
        """Readable multi-line report: one header line, then one line per issue."""
        if not self.issues:
            return "market data OK: no issues found"
        header = f"{len(self.errors())} error(s), {len(self.warnings())} warning(s)"
        return "\n".join([header, *(str(issue) for issue in self.issues)])


def validate_market(
    data: MarketData,
    max_daily_move: float = 0.5,
    stale_days: int = 10,
    split_tolerance: float = 0.02,
) -> ValidationReport:
    """Run all QA checks on a MarketData panel and return a report.

    Checks (each reports, never raises):
      - ``extreme_move``: |daily return| > ``max_daily_move`` (warning),
        escalated to error above 100%.
      - ``negative_volume``: volume < 0 (error).
      - ``price_gap``: NaN close runs longer than 3 days inside an asset's
        listed life (warning).
      - ``stale_price``: >= ``stale_days`` identical consecutive closes
        (warning).
      - ``possible_unadjusted_split``: return < -45% with same-day volume above
        3x the trailing 21-day median volume, or a one-day price ratio within
        ``split_tolerance`` (relative) of a common split ratio: 2-, 3-, 4-, 5-
        or 10-for-1, 3-for-2, and the reverse 1-for-N splits (warning). The
        ratio rule needs no volume, so it also catches a 2-for-1 split whose
        return stays above -50% and whose volume only doubles. A genuine move
        of that size is flagged as well; the warning asks for a look.
      - ``member_without_price``: universe True but close NaN (warning).
      - ``ohlc_inconsistent``: high below low, or open/close outside the
        day's high-low range (warning).
      - ``missing_field``: an optional price/volume field has no value at all
        for a ticker that has close prices, as happens when an optional file
        is keyed by other tickers or dates (warning, one per ticker and field).
      - ``zero_volume``: volume is zero on priced days (warning, one per
        ticker with the number of days).
    """
    issues: list[Issue] = []
    returns = data.returns()

    # (a) extreme moves
    for date, ticker in _true_cells(returns.abs() > max_daily_move):
        move = float(returns.at[date, ticker])
        severity = "error" if abs(move) > 1.0 else "warning"
        issues.append(
            Issue(severity, "extreme_move", ticker, date, f"daily return {move:+.1%}")
        )

    # (b) negative volume
    if data.volume is not None:
        for date, ticker in _true_cells(data.volume < 0):
            shares = float(data.volume.at[date, ticker])
            issues.append(
                Issue("error", "negative_volume", ticker, date, f"volume {shares:,.0f}")
            )

    # (c) NaN gaps inside the listed life; (d) stale prices
    for ticker in data.tickers:
        series = data.close[ticker]
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            continue
        life = series.loc[first:last]
        for run_start, length in _runs(life.isna()):
            if length > 3:
                issues.append(
                    Issue(
                        "warning",
                        "price_gap",
                        ticker,
                        run_start,
                        f"{length}-day NaN gap inside listed life",
                    )
                )
        # a run of L consecutive equal-to-previous closes = L + 1 identical closes
        for run_start, length in _runs(life.eq(life.shift(1))):
            count = length + 1
            if count >= stale_days:
                start_pos = max(int(life.index.get_loc(run_start)) - 1, 0)
                issues.append(
                    Issue(
                        "warning",
                        "stale_price",
                        ticker,
                        life.index[start_pos],
                        f"{count} identical consecutive closes",
                    )
                )

    # (e) split fingerprint: crash-sized return with a volume spike, or a
    # one-day price ratio close to a common split ratio
    volume_spike: set[tuple[int, int]] = set()
    if data.volume is not None:
        trailing_median = data.volume.shift(1).rolling(21, min_periods=5).median()
        suspect = (returns < -0.45) & (data.volume > 3.0 * trailing_median)
        volume_spike = set(zip(*(axis.tolist() for axis in np.nonzero(suspect.to_numpy(dtype=bool)))))
    split_like: dict[tuple[int, int], str] = {}
    price_ratio = 1.0 + returns.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        for label, target in _SPLIT_RATIOS:
            near = np.abs(price_ratio / target - 1.0) <= split_tolerance
            for cell in zip(*(axis.tolist() for axis in np.nonzero(near))):
                split_like[cell] = label
    for row, col in sorted(volume_spike | set(split_like)):
        date, ticker = returns.index[row], returns.columns[col]
        move = float(returns.iat[row, col])
        if (row, col) in volume_spike:
            shares = float(data.volume.iat[row, col])
            median = float(trailing_median.iat[row, col])
            message = (
                f"return {move:+.1%} with volume {shares:,.0f}"
                f" > 3x trailing median {median:,.0f}"
            )
        else:
            message = f"return {move:+.1%} is close to a {split_like[(row, col)]} split ratio"
        issues.append(Issue("warning", "possible_unadjusted_split", ticker, date, message))

    # (f) universe member with no price
    if data.universe is not None:
        for date, ticker in _true_cells(data.universe & data.close.isna()):
            issues.append(
                Issue(
                    "warning",
                    "member_without_price",
                    ticker,
                    date,
                    "universe member has NaN close",
                )
            )

    # (g) price bars that contradict themselves
    if data.high is not None and data.low is not None:
        for date, ticker in _true_cells(data.high < data.low):
            issues.append(
                Issue(
                    "warning",
                    "ohlc_inconsistent",
                    ticker,
                    date,
                    f"high {float(data.high.at[date, ticker]):g}"
                    f" below low {float(data.low.at[date, ticker]):g}",
                )
            )
    for name in ("open", "close"):
        price = getattr(data, name)
        if price is None:
            continue
        for bound, limit, outside in (
            ("high", data.high, lambda p, lim: p > lim * (1.0 + _BAR_TOLERANCE)),
            ("low", data.low, lambda p, lim: p < lim * (1.0 - _BAR_TOLERANCE)),
        ):
            if limit is None:
                continue
            side = "above" if bound == "high" else "below"
            for date, ticker in _true_cells(outside(price, limit)):
                issues.append(
                    Issue(
                        "warning",
                        "ohlc_inconsistent",
                        ticker,
                        date,
                        f"{name} {float(price.at[date, ticker]):g}"
                        f" {side} {bound} {float(limit.at[date, ticker]):g}",
                    )
                )

    # (h) optional fields that are absent or zero for a priced ticker
    priced = data.close.notna()
    priced_days = priced.sum()
    for name in ("open", "high", "low", "volume", "unadjusted_close"):
        frame = getattr(data, name)
        if frame is None:
            continue
        observed = frame.notna().sum()
        for ticker in data.tickers:
            if priced_days[ticker] > 0 and observed[ticker] == 0:
                issues.append(
                    Issue(
                        "warning",
                        "missing_field",
                        ticker,
                        None,
                        f"{name} has no values while close has"
                        f" {int(priced_days[ticker])} prices",
                    )
                )
    if data.volume is not None:
        zero = (data.volume == 0) & priced
        zero_days = zero.sum()
        for ticker in data.tickers:
            days = int(zero_days[ticker])
            if days:
                issues.append(
                    Issue(
                        "warning",
                        "zero_volume",
                        ticker,
                        zero.index[int(zero[ticker].to_numpy().argmax())],
                        f"volume is zero on {days} of {int(priced_days[ticker])} priced days",
                    )
                )

    return ValidationReport(issues=issues)


# -- helpers -------------------------------------------------------------


def _true_cells(mask: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    """(date, ticker) for every True cell of a boolean panel."""
    rows, cols = np.nonzero(mask.to_numpy(dtype=bool))
    return [(mask.index[r], mask.columns[c]) for r, c in zip(rows, cols)]


def _runs(mask: pd.Series) -> Iterator[tuple[pd.Timestamp, int]]:
    """Yield (start_date, run_length) for each maximal run of True values."""
    values = mask.to_numpy(dtype=bool)
    start = None
    for i, hit in enumerate(values):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            yield mask.index[start], i - start
            start = None
    if start is not None:
        yield mask.index[start], len(values) - start
