"""Data quality checks that report issues instead of raising.

``validate_market`` scans a MarketData panel for suspicious values (extreme
moves, negative volume, price gaps, stale prices, unadjusted-split fingerprints,
universe members without prices) and returns a ``ValidationReport``. It never
raises: QA is advisory, structural problems are the job of
``MarketData.validate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from alpha_lab.core.types import MarketData


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
        3x the trailing 21-day median volume (warning).
      - ``member_without_price``: universe True but close NaN (warning).
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

    # (e) split fingerprint: crash-sized return with a volume spike
    if data.volume is not None:
        trailing_median = data.volume.shift(1).rolling(21, min_periods=5).median()
        suspect = (returns < -0.45) & (data.volume > 3.0 * trailing_median)
        for date, ticker in _true_cells(suspect):
            move = float(returns.at[date, ticker])
            shares = float(data.volume.at[date, ticker])
            median = float(trailing_median.at[date, ticker])
            issues.append(
                Issue(
                    "warning",
                    "possible_unadjusted_split",
                    ticker,
                    date,
                    f"return {move:+.1%} with volume {shares:,.0f}"
                    f" > 3x trailing median {median:,.0f}",
                )
            )

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
