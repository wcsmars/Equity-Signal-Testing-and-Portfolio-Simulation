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

from alpha_lab.config.schema import PortfolioConfig, finite_number, integer_at_least
from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import PortfolioConstructor
from alpha_lab.core.registry import Registry
from alpha_lab.core.types import TRADING_DAYS_PER_YEAR, MarketData
from alpha_lab.portfolio.constraints import cap_weights

CONSTRUCTORS = Registry("constructor")

#: keeps score-proportional bucket weights defined when a bucket is flat
#: (all scores equal -> every name gets eps -> equal weights)
_EPS = 1e-9

#: largest multiple by which vol targeting may scale a row. The constructor
#: builds the book at ``gross_leverage`` first and scales it afterwards, so
#: with ``vol_target`` set the gross can reach
#: ``VOL_TARGET_MAX_SCALE * gross_leverage`` in a calm market (and
#: ``max_weight`` is then the only other bound). Not a config field.
VOL_TARGET_MAX_SCALE = 3.0


@CONSTRUCTORS.register("quantile_long_short")
class QuantileLongShort(PortfolioConstructor):
    """Long the top score quantile, short the bottom (or long-only top).

    Per date, among names in the effective universe with a finite score
    (NaN and +-inf count as no opinion):
    ``k = max(1, floor(n * quantile))`` names go in each bucket. ``equal``
    weighting spreads gross_leverage/2 evenly per side; ``score`` weighting
    is proportional to the score's distance from the bucket's worst score,
    so the bucket's least extreme name gets (almost) no weight: a k-name
    bucket holds k - 1 names in effect, and with k = 2 the whole side sits
    on one name, limited only by ``max_weight``.
    Dates with fewer than ``max(min_names, 2)`` valid names get an all-zero
    row, and so do dates whose valid scores are all equal (no dispersion is
    no opinion). Weights are 0.0 (never NaN) outside the effective universe.

    Ties never depend on column order. When the score at a bucket's edge is
    shared by names on both sides of the edge, the whole tied group joins
    the bucket and, under ``equal`` weighting, splits the slots it straddles
    equally (m names tied over j slots each get j/m of a slot); under
    ``score`` weighting it carries the bucket's least extreme score and its
    near-zero weight. A group tied across both edges is long and short at
    once and nets out, so such a row can hold less than ``gross_leverage``.

    ``gross_leverage`` is the gross of the book BEFORE vol targeting. With
    ``vol_target`` set (annualized), each row is then scaled by
    ``clip(target_daily / est_t, 0, VOL_TARGET_MAX_SCALE)`` (3x), where
    ``est_t`` is a diagonal portfolio-vol estimate from trailing per-name
    vol through t, and the per-name cap is re-applied: gross can fall below
    ``gross_leverage`` or rise to 3x that figure. A held name with fewer
    than ``vol_lookback // 2`` days of return history (e.g. a new entrant)
    is given the largest trailing vol known on that date, so one untested
    name neither counts as riskless nor switches the scaling off. Rows with
    no estimate at all keep their unscaled weights: rows holding nothing,
    and the warm-up rows before ANY name has ``vol_lookback // 2`` returns.
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
        for name, value in (("quantile", quantile), ("gross_leverage", gross_leverage), ("max_weight", max_weight)):
            finite_number(value, name)
        if vol_target is not None:
            finite_number(vol_target, "vol_target")
        integer_at_least(vol_lookback, "vol_lookback", 5)
        integer_at_least(min_names, "min_names", 0)
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
        # fill_value keeps the frame boolean: dates or tickers the data does
        # not carry are simply outside the universe
        eff = (
            data.effective_universe()
            .reindex(index=scores.index, columns=scores.columns, fill_value=False)
            .astype(bool)
        )
        masked = scores.where(eff)  # outside the effective universe -> NaN

        out = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        min_active = max(self.min_names, 2)
        for t in scores.index:
            # float first: object/bool score rows would break isfinite
            row = pd.to_numeric(masked.loc[t], errors="coerce").astype(float)
            row = row[np.isfinite(row)]  # NaN and +-inf scores: no opinion
            n = len(row)
            if n < min_active:
                continue  # no-opinion row stays all-zero
            if row.max() == row.min():
                continue  # no dispersion to rank: no opinion either
            # +1e-9 guards float artifacts like 100 * 0.29 == 28.999999999999996
            k = max(1, int(math.floor(n * self.quantile + 1e-9)))
            # quantile <= 0.5 guarantees 2k <= n, so the two k-name buckets
            # never overlap; only a group tied across both edges can sit on
            # both sides, and it is netted below
            order = row.sort_values(kind="mergesort")
            book = pd.Series(0.0, index=order.index)
            if self.dollar_neutral:
                half = self.gross_leverage / 2.0
                longs = self._bucket_weights(order, k, half, long=True)
                shorts = self._bucket_weights(order, k, half, long=False)
                book.loc[longs.index] += longs
                book.loc[shorts.index] -= shorts
            else:
                longs = self._bucket_weights(order, k, self.gross_leverage, long=True)
                book.loc[longs.index] += longs
            out.loc[t, book.index] = book

        requested_gross = out.abs().sum(axis=1)
        out = cap_weights(out, self.max_weight)
        # The infeasible-cap branch of cap_weights silently flattens every
        # active name to exactly max_weight and SHRINKS the row's gross. In
        # that regime gross_leverage / weighting are inert knobs (vol_target
        # can only scale down) — warn once so a sweep over them isn't read as
        # "no effect".
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
                    f"{self.gross_leverage} — gross_leverage and weighting "
                    f"have no effect in this regime and vol_target can only "
                    f"scale the book down. Raise max_weight or quantile, or "
                    f"lower gross_leverage.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_infeasible_cap = True
        if self.vol_target is not None:
            out = self._apply_vol_target(out, data)
        return out

    # -- internals ------------------------------------------------------

    def _bucket_weights(self, order: pd.Series, k: int, target: float, long: bool) -> pd.Series:
        """Non-negative weights for one bucket, summing to ``target``.

        ``order`` is the date's valid scores sorted ascending; the bucket is
        its top (``long``) or bottom ``k`` names. If the score at the
        bucket's edge also occurs outside the bucket, every name with that
        score becomes a member with share j/m (m tied names over the j slots
        they straddle), so the result does not depend on column order.
        """
        values = order.to_numpy()
        inside = np.zeros(len(values), dtype=bool)
        if long:
            inside[-k:] = True
            tied = values == values[-k]
        else:
            inside[:k] = True
            tied = values == values[k - 1]
        member = inside | tied
        bucket = order[member]
        if self.weighting == "equal":
            # exactly 1.0 for every member when the tied group lies inside
            # the bucket; the shares always add up to k
            share = np.where(tied, (tied & inside).sum() / tied.sum(), 1.0)[member]
            return pd.Series(share * (target / k), index=bucket.index)
        # 'score': proportional to distance from the bucket's worst score
        # (bucket min for longs, bucket max for shorts, mirrored so the most
        # extreme score gets the most weight on both sides). A group tied at
        # the edge IS the worst score: its names get the same near-zero
        # weight, whatever their share of the bucket.
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
        # A held name with no estimate of its own yet (a new entrant) gets
        # the largest trailing vol known on that date. Leaving it out of the
        # sum would count it as riskless and over-lever the row; dropping the
        # whole row back to scale 1.0 would switch the risk control off (and
        # jump the gross to gross_leverage) whenever one such name is held.
        held = weights != 0.0
        sigma = sigma.where(sigma.notna() | ~held, sigma.max(axis=1), axis=0)
        # Diagonal covariance approximation, documented: cross-correlations
        # are ignored, est_t = sqrt(sum_i w_i^2 sigma_i,t^2).
        est = np.sqrt((weights.pow(2) * sigma.pow(2)).sum(axis=1))
        daily_target = self.vol_target / math.sqrt(TRADING_DAYS_PER_YEAR)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = daily_target / est
        scale = raw.clip(0.0, VOL_TARGET_MAX_SCALE)
        # Rows with no estimate at all keep their weights: zero-holding rows
        # and warm-up rows where no name has vol history yet (est == 0 in
        # both, since the sum skips the missing terms) — a blind lever-up on
        # est == 0 would be spurious.
        usable = np.isfinite(raw) & (est > 0.0)
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
