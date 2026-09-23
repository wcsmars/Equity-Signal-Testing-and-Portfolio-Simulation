"""Abstract interfaces implemented across the package.

Every cross-module contract lives here so concrete modules depend only on
core. The docstrings restate the relevant clause of CONVENTIONS.md — read
that file for the full timing law.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from alpha_lab.core.types import MarketData


def _freeze(value: Any) -> Any:
    """Recursively convert lists/dicts to tuples so params are hashable."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class FeatureSpec:
    """A feature request: registry name + params, hashable, with a stable key.

    The key (e.g. ``momentum(skip=21,window=252)``) names the feature panel in
    the features mapping passed to signals, and keys the FeatureStore cache.
    """

    name: str
    params: tuple = field(default_factory=tuple)  # sorted (key, value) pairs

    @staticmethod
    def make(name: str, **params: Any) -> "FeatureSpec":
        return FeatureSpec(name=name, params=tuple(sorted((k, _freeze(v)) for k, v in params.items())))

    @property
    def as_dict(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def key(self) -> str:
        inner = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.name}({inner})"


#: what signals receive: FeatureSpec.key -> wide DataFrame (dates x tickers)
FeatureSet = Mapping[str, pd.DataFrame]


class DataSource(ABC):
    """Produces MarketData. Implementations: CSV files, synthetic generator."""

    name: str = "source"

    @abstractmethod
    def load(self) -> MarketData:
        """Return a validated MarketData panel."""


class Feature(ABC):
    """One derived panel. Value at date t uses information through close t ONLY.

    Must satisfy truncation invariance:
        compute(data.slice_until(t)).loc[t] == compute(data).loc[t]
    ``lookback`` declares how many bars of history suffice for a defined
    value (a panel of ``lookback`` rows yields a non-NaN final row) —
    advisory metadata for warm-up/eligibility logic; nothing in the engine
    consumes it automatically.
    """

    name: str = "feature"
    lookback: int = 0

    @abstractmethod
    def compute(self, data: MarketData) -> pd.DataFrame:
        """Wide panel aligned to data.close (same index and columns)."""


class Signal(ABC):
    """Maps feature panels to a score panel. Higher score = more attractive.

    ``score`` must be a pure per-date map: the score at t depends only on
    feature rows <= t (features are already point-in-time, so a row-t-only
    function is the common case). ``fit`` may only look at ``train_dates``.
    Scores should be NaN wherever no opinion / not in universe.
    """

    name: str = "signal"

    @property
    def required_features(self) -> tuple[FeatureSpec, ...]:
        """Feature panels this signal needs; the runner computes them."""
        return ()

    def fit(self, features: FeatureSet, data: MarketData, train_dates: pd.DatetimeIndex) -> None:
        """Optional parameter fitting on the train window. Default: no-op."""

    @abstractmethod
    def score(self, features: FeatureSet, data: MarketData) -> pd.DataFrame:
        """Score panel aligned to data.close."""


class PortfolioConstructor(ABC):
    """Turns scores at t into target weights W_t (decided at close t).

    May use market info through close t (e.g. trailing vol). Output must be 0
    (not NaN) outside the effective universe.
    """

    @abstractmethod
    def weights(self, scores: pd.DataFrame, data: MarketData) -> pd.DataFrame:
        """Target weight panel aligned to scores."""


class CostModel(ABC):
    """Prices a panel of trades, in NAV-fraction units per date.

    ``trades``: signed weight changes at each date (dates x tickers).
    Rolling inputs (ADV, trailing vol) evaluated at trade date t must use
    data through t-1 (CONVENTIONS.md rule 8). Share math (commissions,
    participation) must use unadjusted_close.
    """

    @abstractmethod
    def cost(self, trades: pd.DataFrame, data: MarketData, portfolio_value: float) -> pd.Series:
        """Total cost per date as a fraction of NAV (index = trades.index)."""


class ZeroCost(CostModel):
    """Free trading. For engine tests and cost-attribution baselines."""

    def cost(self, trades: pd.DataFrame, data: MarketData, portfolio_value: float) -> pd.Series:
        return pd.Series(0.0, index=trades.index)
