"""Portfolio constructors: score panels -> target weight panels W_t.

Timing (CONVENTIONS.md clause 3): W_t is decided at close t and may use any
information known at close t, including the day-t return itself (e.g. a
trailing volatility computed THROUGH t). The execution lag that turns W into
holdings lives in the backtest engine, not here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from alpha_lab.config.schema import PortfolioConfig
from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import PortfolioConstructor
from alpha_lab.core.registry import Registry
from alpha_lab.core.types import TRADING_DAYS_PER_YEAR, MarketData
from alpha_lab.portfolio.constraints import cap_weights

CONSTRUCTORS = Registry("constructor")

#: keeps score-proportional bucket weights defined when a bucket is flat
#: (all scores equal -> every name gets eps -> equal weights)
_EPS = 1e-9


@CONSTRUCTORS.register("quantile_long_short")
class QuantileLongShort(PortfolioConstructor):
    """Long the top score quantile, short the bottom (or long-only top).

    Per date, among names in the effective universe with a non-NaN score:
    ``k = max(1, floor(n * quantile))`` names go in each bucket. ``equal``
    weighting spreads gross_leverage/2 evenly per side; ``score`` weighting
    is proportional to the score's distance from the bucket's worst score.
    Dates with fewer than ``max(min_names, 2)`` valid names get an all-zero
    row. Weights are 0.0 (never NaN) outside the effective universe.

    With ``vol_target`` set (annualized), each row is scaled by
    ``clip(target_daily / est_t, 0, 3)`` where ``est_t`` is a diagonal
    portfolio-vol estimate from trailing per-name vol through t, then the
    per-name cap is re-applied.
    """

    def __init__(
        self,
        quantile: float = 0.2,
        weighting: str = "equal",
        dollar_neutral: bool = True,
        gross_leverage: float = 2.0,
        max_weight: float = 0.10,
        vol_target: float | None = None,
        vol_lookback: int = 63,
        min_names: int = 4,
    ) -> None:
        if not 0 < quantile <= 0.5:
            raise ConfigError("quantile must be in (0, 0.5]")
        if weighting not in ("equal", "score"):
            raise ConfigError(f"unknown weighting '{weighting}'")
        if gross_leverage <= 0:
            raise ConfigError("gross_leverage must be positive")
        if max_weight <= 0:
            raise ConfigError("max_weight must be positive")
        if vol_target is not None and vol_target <= 0:
            raise ConfigError("vol_target must be positive when set")
        if vol_lookback < 5:
            raise ConfigError("vol_lookback too short")
        if min_names < 0:
            raise ConfigError("min_names must be >= 0")
        self.quantile = float(quantile)
        self.weighting = weighting
        self.dollar_neutral = bool(dollar_neutral)
        self.gross_leverage = float(gross_leverage)
        self.max_weight = float(max_weight)
        self.vol_target = None if vol_target is None else float(vol_target)
        self.vol_lookback = int(vol_lookback)
        self.min_names = int(min_names)
        self._warned_infeasible_cap = False

    # -- PortfolioConstructor ------------------------------------------

    def weights(self, scores: pd.DataFrame, data: MarketData) -> pd.DataFrame:
        """Target weight panel aligned to ``scores`` (same index/columns)."""
        if not isinstance(scores, pd.DataFrame):
            raise DataError("scores must be a wide DataFrame (dates x tickers)")
        eff = (
            data.effective_universe()
            .reindex(index=scores.index, columns=scores.columns)
            .fillna(False)
            .astype(bool)
        )
        masked = scores.where(eff)  # outside the effective universe -> NaN

        out = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        min_active = max(self.min_names, 2)
        for t in scores.index:
            row = masked.loc[t].dropna()
            n = len(row)
            if n < min_active:
                continue  # no-opinion row stays all-zero
            # +1e-9 guards float artifacts like 10 * 0.3 == 2.9999999999999996
            k = max(1, int(math.floor(n * self.quantile + 1e-9)))
            # stable sort => deterministic bucket membership under score ties;
            # quantile <= 0.5 guarantees 2k <= n, so buckets never overlap
            order = row.sort_values(kind="mergesort")
            if self.dollar_neutral:
                half = self.gross_leverage / 2.0
                longs = order.iloc[-k:]
                shorts = order.iloc[:k]
                out.loc[t, longs.index] = self._bucket_weights(longs, half, long=True)
                out.loc[t, shorts.index] = -self._bucket_weights(shorts, half, long=False)
            else:
                longs = order.iloc[-k:]
                out.loc[t, longs.index] = self._bucket_weights(
                    longs, self.gross_leverage, long=True
                )

        requested_gross = out.abs().sum(axis=1)
        out = cap_weights(out, self.max_weight)
        # The infeasible-cap branch of cap_weights silently flattens every
        # active name to exactly max_weight and SHRINKS the row's gross. In
        # that regime gross_leverage / weighting / vol_target are all inert
        # knobs — warn once so a sweep over them isn't read as "no effect".
        if not self._warned_infeasible_cap:
            capped_gross = out.abs().sum(axis=1)
            shrunk = capped_gross < requested_gross * (1.0 - 1e-9) - 1e-12
            active = requested_gross > 0.0
            if bool((shrunk & active).any()):
                n_bad = int((shrunk & active).sum())
                warnings.warn(
                    f"max_weight={self.max_weight} is infeasible for the "
                    f"requested gross on {n_bad} of {int(active.sum())} active "
                    f"dates: every selected name is flattened to the cap and "
                    f"achieved gross falls short of gross_leverage="
                    f"{self.gross_leverage} — gross_leverage, weighting and "
                    f"vol_target have no effect in this regime. Raise "
                    f"max_weight or lower gross_leverage/quantile.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_infeasible_cap = True
        if self.vol_target is not None:
            out = self._apply_vol_target(out, data)
        return out

    # -- internals ------------------------------------------------------

    def _bucket_weights(self, bucket: pd.Series, target: float, long: bool) -> pd.Series:
        """Non-negative weights for one bucket, summing to ``target``."""
        if self.weighting == "equal":
            return pd.Series(target / len(bucket), index=bucket.index)
        # 'score': proportional to distance from the bucket's worst score
        # (bucket min for longs, bucket max for shorts, mirrored so the most
        # extreme score gets the most weight on both sides).
        raw = (bucket - bucket.min() + _EPS) if long else (bucket.max() - bucket + _EPS)
        return raw / raw.sum() * target

    def _apply_vol_target(self, weights: pd.DataFrame, data: MarketData) -> pd.DataFrame:
        """Scale each row toward the annualized vol target, then re-cap."""
        # Trailing per-name vol THROUGH decision date t. Using r_t here is
        # deliberate and legal: CONVENTIONS.md timing-law clause 3 lets the
        # constructor use information known at close t (rule 8's t-1 shift
        # applies to cost models pricing the trade, not to this decision).
        sigma = (
            data.returns()
            .rolling(self.vol_lookback, min_periods=self.vol_lookback // 2)
            .std()
            .reindex(index=weights.index, columns=weights.columns)
        )
        # Diagonal covariance approximation, documented: cross-correlations
        # are ignored, est_t = sqrt(sum_i w_i^2 sigma_i,t^2).
        est = np.sqrt((weights.pow(2) * sigma.pow(2)).sum(axis=1))
        daily_target = self.vol_target / math.sqrt(TRADING_DAYS_PER_YEAR)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = daily_target / est
        scale = raw.clip(0.0, 3.0)
        # Rows with no usable estimate keep their weights: zero-holding rows
        # (est == 0) and rows where a held name has no vol history yet — a
        # blind 3x lever-up on est ~ 0 would be spurious.
        incomplete = ((weights != 0.0) & sigma.isna()).any(axis=1)
        usable = np.isfinite(raw) & (est > 0.0) & ~incomplete
        scale = scale.where(usable, 1.0)
        return cap_weights(weights.mul(scale, axis=0), self.max_weight)


def from_config(cfg: PortfolioConfig) -> PortfolioConstructor:
    """Instantiate the configured constructor via the CONSTRUCTORS registry."""
    return CONSTRUCTORS.create(
        cfg.method,
        quantile=cfg.quantile,
        weighting=cfg.weighting,
        dollar_neutral=cfg.dollar_neutral,
        gross_leverage=cfg.gross_leverage,
        max_weight=cfg.max_weight,
        vol_target=cfg.vol_target,
        vol_lookback=cfg.vol_lookback,
        min_names=cfg.min_names,
    )
