"""Chronological walk-forward splitter (CONVENTIONS.md, "Walk-forward").

Positional layout on a trading-day index of length ``n`` with
``gap = purge_days + embargo_days``:

- the first test block starts at position ``p0 = train_days + gap``;
- test blocks are ``[p, p + test_days)`` stepping by ``test_days`` to the end
  of the index — contiguous and non-overlapping; the final block may be
  shorter and is kept if it spans at least one day;
- for a block starting at ``p``: train end position is ``p - gap - 1``
  (inclusive) and train start is ``p - gap - train_days`` for the
  ``rolling`` scheme, ``0`` for ``expanding``.

Train therefore never touches ``[test_start - gap, ...)``: the ``gap``
positions between train end and test start are used by neither side.
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.config.schema import WalkForwardConfig
from alpha_lab.core.errors import ConfigError
from alpha_lab.core.results import WalkForwardWindow


class WalkForwardSplitter:
    """Splits a trading-day index into purged train/test windows.

    Window boundaries are actual timestamps taken from the supplied index
    (``dates[pos]``), all inclusive, so they can be used directly with
    ``MarketData.slice_range`` and label-based ``.loc`` slicing.
    """

    def __init__(self, cfg: WalkForwardConfig) -> None:
        self.cfg = cfg

    def split(self, dates: pd.DatetimeIndex) -> list[WalkForwardWindow]:
        """Return the list of walk-forward windows for ``dates``.

        Raises ConfigError when the index is too short to fit one train
        window, the purge+embargo gap, and at least one test day.
        """
        cfg = self.cfg
        gap = cfg.purge_days + cfg.embargo_days
        n = len(dates)
        min_len = cfg.train_days + gap + 1
        if n < min_len:
            raise ConfigError(
                f"walk-forward needs at least {min_len} dates "
                f"(train_days={cfg.train_days} + purge+embargo gap={gap} + 1 test day); "
                f"got {n}"
            )

        p0 = cfg.train_days + gap
        windows: list[WalkForwardWindow] = []
        for p in range(p0, n, cfg.test_days):
            test_end_pos = min(p + cfg.test_days, n) - 1  # truncated final block kept
            train_end_pos = p - gap - 1
            train_start_pos = 0 if cfg.scheme == "expanding" else p - gap - cfg.train_days
            windows.append(
                WalkForwardWindow(
                    train_start=dates[train_start_pos],
                    train_end=dates[train_end_pos],
                    test_start=dates[p],
                    test_end=dates[test_end_pos],
                )
            )
        return windows
