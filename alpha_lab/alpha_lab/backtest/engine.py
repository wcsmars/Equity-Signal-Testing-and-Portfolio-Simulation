"""BacktestEngine: applies the CONVENTIONS.md timing law to one strategy.

The law, restated (clauses 4-8):

- Holdings ``H_t = W_{t-lag}`` with ``lag >= 1`` — the engine raises
  LookaheadError for anything less, independently of config validation.
- ``gross_t = sum_i H_{t,i} * r_{t,i}``.
- ``trades_t = H_t - drift(H_{t-1})`` where
  ``drift(H_{t-1})_i = H_{t-1,i} * (1 + r_{t-1,i}) / (1 + gross_{t-1})``:
  the trade dated t executes at close t-1 (clause 7), when the prior book
  has drifted with returns realized through THAT close — never with day-t
  returns, which print after the trade is fixed (rule 8). (Drift adjustment
  configurable via ``drift_adjust_turnover``; on by default.)
  ``turnover_t = sum_i |trades_{t,i}|``.
- Costs are NAV-fraction per date, charged the same date as the trade:
  ``net_t = gross_t - cost_t``.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from alpha_lab.config.schema import BacktestConfig
from alpha_lab.core.errors import DataError, LookaheadError
from alpha_lab.core.interfaces import CostModel, FeatureSet, PortfolioConstructor, Signal
from alpha_lab.core.results import BacktestResult, WalkForwardWindow
from alpha_lab.core.types import MarketData
from alpha_lab.backtest.walkforward import WalkForwardSplitter


class BacktestEngine:
    """Runs one signal + constructor + cost model through the timing law.

    Two modes, chosen by ``config.walkforward``:

    - ``walkforward`` set: signals are fitted per train window (on a fresh
      deep copy each time, so no state bleeds across windows), scored on the
      full panel (legal — signals are pure per-date maps, CONVENTIONS.md
      clause 2), and only each window's test rows are kept in the stitched
      score panel. Dates before the first test window keep NaN scores.
    - ``walkforward`` None: IN-SAMPLE DIAGNOSTIC. ``fit`` sees every date and
      the score panel is produced in one pass. THIS IS NOT OUT-OF-SAMPLE —
      use it only for debugging and capacity checks, never as evidence a
      strategy works. ``meta['mode']`` is set to ``'insample'`` so downstream
      reports can flag it.
    """

    def __init__(
        self,
        signal: Signal,
        constructor: PortfolioConstructor,
        cost_model: CostModel,
        config: BacktestConfig,
        portfolio_value: float = 1_000_000.0,
    ) -> None:
        self.signal = signal
        self.constructor = constructor
        self.cost_model = cost_model
        self.config = config
        self.portfolio_value = float(portfolio_value)

    # -- public ---------------------------------------------------------

    def run(self, data: MarketData, features: FeatureSet | None = None) -> BacktestResult:
        """Backtest ``data`` and return a BacktestResult.

        ``features`` may be precomputed (keyed by FeatureSpec.key); when None
        the engine computes ``signal.required_features`` on the FULL panel via
        the FeatureStore — legal even under walk-forward because features are
        point-in-time by contract (CONVENTIONS.md clause 1).
        """
        lag = self.config.execution_lag
        if lag < 1:
            # Defense in depth: the schema rejects lag < 1 too, but a config
            # object can be built or mutated programmatically.
            raise LookaheadError(
                f"execution_lag must be >= 1, got {lag}: lag 0 trades on "
                "information unavailable at execution time (CONVENTIONS.md clause 4)"
            )

        if features is None:
            features = self._compute_features(data)

        if self.config.walkforward is None:
            scores, windows, mode = self._score_insample(data, features), None, "insample"
        else:
            scores, windows = self._score_walkforward(data, features)
            mode = "walkforward"

        weights = self.constructor.weights(scores, data)
        if not weights.index.equals(data.dates) or not weights.columns.equals(data.close.columns):
            raise DataError("constructor weights are not aligned to the data panel")
        weights = weights.fillna(0.0)

        # Clause 4: the timing law. H_t = W_{t-lag}; pre-history holds cash.
        holdings = weights.shift(lag).fillna(0.0)

        returns = data.returns()
        # A delisted asset held into its NaN-return days contributes 0 that
        # day (position closed at last realized return; the constructor
        # zeroes its weight from the delisting date). Never a forward-fill.
        rr = returns.fillna(0.0)
        gross = (holdings * rr).sum(axis=1)

        if self.config.drift_adjust_turnover:
            # Clause 6: drift(H_{t-1})_i = H_{t-1,i} (1 + r_{t-1,i}) / (1 + gross_{t-1}).
            # The trade dated t executes at close t-1 (clause 7), when H_{t-1}
            # has drifted with the returns realized through THAT close — so the
            # per-date drift factor is computed on each date's own returns and
            # then shifted forward one day. Using day-t returns here would make
            # trades_t (and hence cost_t) depend on a close that prints after
            # the trade is fixed — a rule-8 lookahead. The denom guard handles
            # pathological rows where 1 + gross == 0 (a -100% day wipes the
            # book; the drifted position is treated as flat).
            denom = 1.0 + gross
            denom = denom.mask(denom == 0.0, np.nan)
            drifted = holdings.mul(1.0 + rr).div(denom, axis=0).shift(1)
        else:
            drifted = holdings.shift(1)
        drifted = drifted.fillna(0.0)  # first row: trades equal H row 0

        trades = holdings - drifted
        turnover = trades.abs().sum(axis=1)

        costs = self.cost_model.cost(trades, data, self.portfolio_value)
        costs = costs.reindex(data.dates).fillna(0.0)
        net = gross - costs

        return BacktestResult(
            gross_returns=gross,
            costs=costs,
            net_returns=net,
            turnover=turnover,
            holdings=holdings,
            target_weights=weights,
            scores=scores,
            windows=windows,
            meta={
                "mode": mode,
                "execution_lag": lag,
                "portfolio_value": self.portfolio_value,
            },
        )

    # -- scoring ----------------------------------------------------------

    def _score_insample(self, data: MarketData, features: FeatureSet) -> pd.DataFrame:
        """One fit + one score over every date. Diagnostic only, never OOS."""
        sig = copy.deepcopy(self.signal)
        sig.fit(features, data, data.dates)
        return sig.score(features, data)

    def _score_walkforward(
        self, data: MarketData, features: FeatureSet
    ) -> tuple[pd.DataFrame, list[WalkForwardWindow]]:
        """Fit per train window, keep only test rows, stitch across windows."""
        windows = WalkForwardSplitter(self.config.walkforward).split(data.dates)
        stitched = pd.DataFrame(np.nan, index=data.dates, columns=data.close.columns)
        for window in windows:
            sig = copy.deepcopy(self.signal)  # no state bleed across windows
            train_data = data.slice_range(window.train_start, window.train_end)
            train_features = {
                key: panel.loc[window.train_start : window.train_end]
                for key, panel in features.items()
            }
            sig.fit(train_features, train_data, train_data.dates)
            # Scoring the full panel then masking to the test rows is
            # equivalent to day-by-day scoring: signals are pure per-date
            # maps of point-in-time features (CONVENTIONS.md clause 2).
            full = sig.score(features, data)
            stitched.loc[window.test_start : window.test_end] = full.loc[
                window.test_start : window.test_end
            ]
        return stitched, windows

    # -- features ---------------------------------------------------------

    def _compute_features(self, data: MarketData) -> FeatureSet:
        """Compute the signal's required features on the full panel.

        Imported lazily to avoid an import cycle with the features package.
        """
        from alpha_lab.features import FeatureStore

        store = FeatureStore()
        specs = tuple(self.signal.required_features)
        if hasattr(store, "compute"):
            return store.compute(specs, data)
        return {spec.key: store.get(spec, data) for spec in specs}
