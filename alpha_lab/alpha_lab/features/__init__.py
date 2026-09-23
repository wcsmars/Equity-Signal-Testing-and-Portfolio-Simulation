"""Feature layer: the feature library (registry FEATURES) and cached FeatureStore.

Every Feature must satisfy truncation invariance — see CONVENTIONS.md.
"""

from alpha_lab.features.library import (
    FEATURES,
    ADVDollars,
    Momentum,
    RSI,
    RealizedVol,
    WindowReturn,
    ZScoreReturn,
)
from alpha_lab.features.store import FeatureStore, fingerprint

__all__ = [
    "FEATURES",
    "ADVDollars",
    "FeatureStore",
    "Momentum",
    "RSI",
    "RealizedVol",
    "WindowReturn",
    "ZScoreReturn",
    "fingerprint",
]
