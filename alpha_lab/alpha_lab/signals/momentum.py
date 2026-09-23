"""Cross-sectional momentum signal — the worked example of the signal pattern.

The pattern (CONVENTIONS.md clause 2): declare the point-in-time feature
panels you need via ``required_features``, then make ``score`` a pure
per-date map — mask each row to the effective universe and standardize it
cross-sectionally. Because every output row depends only on that row's
inputs, scoring the full panel and scoring day by day are equivalent, and
the composed feature+signal pipeline inherits truncation invariance from
the features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import FeatureSet, FeatureSpec, Signal
from alpha_lab.core.types import MarketData
from alpha_lab.signals.registry import SIGNALS


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Validate an integer parameter at construction time (ConfigError)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def cross_sectional_zscore(panel: pd.DataFrame, min_names: int) -> pd.DataFrame:
    """Per-date z-score: (x - row mean) / row std over each date's valid cells.

    Rows with fewer than ``min_names`` valid values are all-NaN (too thin a
    cross-section to standardize), as are rows with zero std (no dispersion
    to rank). Uses each date's row only — a pure per-date map, never a
    full-sample or time-series statistic (a CONVENTIONS.md forbidden
    pattern).
    """
    valid = panel.notna().sum(axis=1)
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)  # ddof=1, so each scored row has std exactly 1
    usable = (valid >= min_names) & (std > 0)
    # std.where(usable) is NaN on unusable rows, so the division blanks them.
    return panel.sub(mean, axis=0).div(std.where(usable), axis=0)


def lookup_feature(features: FeatureSet, spec: FeatureSpec, signal_name: str) -> pd.DataFrame:
    """Fetch ``spec.key`` from a FeatureSet, raising DataError if absent."""
    try:
        return features[spec.key]
    except KeyError:
        raise DataError(
            f"signal '{signal_name}' requires feature '{spec.key}'; "
            f"got {sorted(features)}"
        ) from None


@SIGNALS.register("xs_momentum")
class CrossSectionalMomentum(Signal):
    """Cross-sectional z-score of (window, skip) price momentum.

    Higher trailing return from t-window to t-skip => higher score. The
    defaults give the classic 12-1 construction (a year of return excluding
    the most recent month). Stateless: ``fit`` is the inherited no-op.
    """

    name = "xs_momentum"

    def __init__(self, window: int = 252, skip: int = 21, min_names: int = 5) -> None:
        self.window = _check_int(window, "window", minimum=1)
        self.skip = _check_int(skip, "skip", minimum=0)
        if self.window <= self.skip:
            raise ConfigError(
                f"xs_momentum requires window > skip >= 0, "
                f"got window={self.window}, skip={self.skip}"
            )
        # min_names >= 2: a one-name cross-section has no sample std (ddof=1)
        self.min_names = _check_int(min_names, "min_names", minimum=2)

    @property
    def required_features(self) -> tuple[FeatureSpec, ...]:
        return (FeatureSpec.make("momentum", window=self.window, skip=self.skip),)

    def score(self, features: FeatureSet, data: MarketData) -> pd.DataFrame:
        panel = lookup_feature(features, self.required_features[0], self.name)
        masked = panel.where(data.effective_universe())
        return cross_sectional_zscore(masked, self.min_names)
