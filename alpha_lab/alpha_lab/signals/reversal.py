"""Short-term reversal signal — the second worked example of the pattern.

Same shape as ``alpha_lab.signals.momentum``: declare the point-in-time
feature, mask to the effective universe, standardize per date. The only
twist is the sign flip — recent losers score high — showing how a new
signal is a feature spec plus a per-date transform, nothing more.
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.core.interfaces import FeatureSet, FeatureSpec, Signal
from alpha_lab.core.types import MarketData

# Shared helpers live with the first worked example; sibling import, no cycle.
from alpha_lab.signals.momentum import _check_int, cross_sectional_zscore, lookup_feature
from alpha_lab.signals.registry import SIGNALS


@SIGNALS.register("xs_reversal")
class ShortTermReversal(Signal):
    """Cross-sectional z-score of MINUS the trailing ``window``-day return.

    Short-horizon returns tend to revert, so recent losers (most negative
    window return) get the highest score. Stateless: ``fit`` is the
    inherited no-op.
    """

    name = "xs_reversal"

    def __init__(self, window: int = 5, min_names: int = 5) -> None:
        self.window = _check_int(window, "window", minimum=1)
        # min_names >= 2: a one-name cross-section has no sample std (ddof=1)
        self.min_names = _check_int(min_names, "min_names", minimum=2)

    @property
    def required_features(self) -> tuple[FeatureSpec, ...]:
        return (FeatureSpec.make("returns", window=self.window),)

    def score(self, features: FeatureSet, data: MarketData) -> pd.DataFrame:
        panel = lookup_feature(features, self.required_features[0], self.name)
        # Sign flip BEFORE masking/standardizing: score the negated return.
        masked = (-panel).where(data.effective_universe())
        return cross_sectional_zscore(masked, self.min_names)
