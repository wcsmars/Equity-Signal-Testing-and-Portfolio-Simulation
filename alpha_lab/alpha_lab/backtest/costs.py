"""Transaction cost models.

Every model prices a panel of trades (signed weight changes, dates x tickers)
and returns total cost per date in NAV-fraction units (CONVENTIONS.md rule 7).
Rolling inputs evaluated at trade date t use data through t-1 only (rule 8):
the trade is being priced before t's close prints.

Share math (per-share commissions, dollar ADV) uses ``unadjusted_close``.
This is the platform's most important cost rule: adjusted prices misstate
share counts across splits — before a 4:1 split the raw price is 4x the
adjusted level, so ``|trade| * NAV / close`` claims 4x the shares actually
bought, and per-share commissions come out 4x too high.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from alpha_lab.config.schema import CostConfig
from alpha_lab.core.errors import ConfigError
from alpha_lab.core.interfaces import CostModel, ZeroCost
from alpha_lab.core.registry import Registry

COST_MODELS = Registry("cost_model")


@COST_MODELS.register("zero")
class Zero(ZeroCost):
    """Free trading — core ZeroCost re-exported under the cost-model registry."""


@COST_MODELS.register("fixed_bps")
class FixedBps(CostModel):
    """Flat proportional cost: cost_t = sum_i |trade_{t,i}| * bps * 1e-4."""

    def __init__(self, bps: float = 5.0) -> None:
        if bps < 0:
            raise ConfigError("fixed_bps: bps must be >= 0")
        self.bps = float(bps)

    def cost(self, trades: pd.DataFrame, data, portfolio_value: float) -> pd.Series:
        """Total cost per date as a NAV fraction; NaN trades count as no trade."""
        return trades.abs().sum(axis=1) * self.bps * 1e-4


@COST_MODELS.register("realistic")
class RealisticCost(CostModel):
    """Spread + per-share commission + square-root market impact.

    Components at trade date t (all in NAV-fraction units):

    - spread_t     = sum_i |trade| * half_spread_bps * 1e-4
    - commission_t = sum_i shares_i * commission_per_share / NAV, with
      shares_i = |trade| * NAV / unadjusted_close at t-1 (the trade dated t
      executes at close t-1, CONVENTIONS.md rule 7, so that is the price the
      shares actually printed at; the first date has no t-1 price and is
      commission-free)
    - impact_t     = sum_i |trade| * impact_coeff * vol_daily * sqrt(participation),
      participation = |trade| * NAV / ADV_dollars, capped at PARTICIPATION_CAP

    ADV and trailing vol are rolling statistics shifted by one day so the
    value used at t only sees data through t-1 (CONVENTIONS.md rule 8).
    Where a traded cell has no ADV/vol history (early window, or a ticker with
    no volume data), the conservative defaults VOL_FALLBACK / PARTICIPATION_FALLBACK
    apply instead of free trading. Cells with |trade| == 0 cost exactly 0.

    If the data has no ``unadjusted_close``, the model warns once (per
    instance) and falls back to the adjusted close — degraded mode, because
    adjusted prices misstate share counts across splits.
    """

    #: conservative defaults for traded cells with missing rolling history
    VOL_FALLBACK = 0.02
    PARTICIPATION_FALLBACK = 0.05
    #: participation cap — beyond ~2x ADV the sqrt impact form has no
    #: empirical support and a single runaway trade would dominate the run
    PARTICIPATION_CAP = 2.0

    def __init__(
        self,
        commission_per_share: float = 0.005,
        half_spread_bps: float = 2.5,
        impact_coeff: float = 0.1,
        adv_window: int = 21,
        vol_window: int = 63,
    ) -> None:
        for name, val in (
            ("commission_per_share", commission_per_share),
            ("half_spread_bps", half_spread_bps),
            ("impact_coeff", impact_coeff),
        ):
            if val < 0:
                raise ConfigError(f"realistic cost model: {name} must be >= 0")
        # min_periods below are max(5, w//2) and max(10, w//2); shorter windows
        # would make min_periods exceed the window and pandas would reject it
        if adv_window < 5:
            raise ConfigError("realistic cost model: adv_window must be >= 5")
        if vol_window < 10:
            raise ConfigError("realistic cost model: vol_window must be >= 10")
        self.commission_per_share = float(commission_per_share)
        self.half_spread_bps = float(half_spread_bps)
        self.impact_coeff = float(impact_coeff)
        self.adv_window = int(adv_window)
        self.vol_window = int(vol_window)
        self._warned_no_unadjusted = False

    def cost(self, trades: pd.DataFrame, data, portfolio_value: float) -> pd.Series:
        """Total cost per date as a NAV fraction (index = trades.index)."""
        if portfolio_value <= 0:
            raise ConfigError("realistic cost model: portfolio_value must be positive")

        abs_tr = trades.abs().fillna(0.0)  # NaN trade == no trade
        traded = abs_tr > 0

        price_raw = data.unadjusted_close
        if price_raw is None:
            # THE key cost rule: share math needs the raw close. Adjusted
            # prices misstate share counts across splits (pre-split the raw
            # price is a multiple of the adjusted one), so commissions and
            # dollar ADV computed from `close` are off by the split factor.
            if not self._warned_no_unadjusted:
                warnings.warn(
                    "MarketData has no unadjusted_close; falling back to the "
                    "adjusted close for share math — commissions and ADV will "
                    "be misstated across splits",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_no_unadjusted = True
            price_raw = data.close

        # -- spread ------------------------------------------------------
        spread = abs_tr.sum(axis=1) * self.half_spread_bps * 1e-4

        # -- commission ----------------------------------------------------
        # Rule 7/8: the trade dated t executes at close t-1, and t's close is
        # not yet known when the trade prints — so the share conversion uses
        # the raw close at t-1, the same shift discipline as ADV/vol below.
        # (On a split ex-date the unshifted price would misstate the share
        # count by the full split factor.)
        px = price_raw.shift(1).reindex(index=abs_tr.index, columns=abs_tr.columns)
        shares = abs_tr * portfolio_value / px
        # |trade| > 0 with a NaN price is either an engine artifact at
        # delisting (the close-out row after prices stop printing) or the
        # first date (no t-1 price exists): contributes 0.
        shares = shares.where(traded & (px > 0), 0.0)
        commission = shares.sum(axis=1) * self.commission_per_share / portfolio_value

        # -- impact --------------------------------------------------------
        # Rule 8: rolling inputs priced at trade date t must only see data
        # through t-1 — compute rolling panels on data's index, then shift(1).
        if data.volume is not None:
            adv_dollars = (
                (data.volume * price_raw)
                .rolling(self.adv_window, min_periods=max(5, self.adv_window // 2))
                .mean()
                .shift(1)
                .reindex(index=abs_tr.index, columns=abs_tr.columns)
            )
            participation = abs_tr * portfolio_value / adv_dollars
        else:
            participation = pd.DataFrame(
                np.nan, index=abs_tr.index, columns=abs_tr.columns
            )
        # cap (also swallows inf from a zero-ADV day); NaN survives the clip
        # and is handled by the fallback below
        participation = participation.clip(upper=self.PARTICIPATION_CAP)

        vol_daily = (
            data.returns()
            .rolling(self.vol_window, min_periods=max(10, self.vol_window // 2))
            .std()
            .shift(1)
            .reindex(index=abs_tr.index, columns=abs_tr.columns)
        )

        # conservative defaults where a traded cell lacks history (early window)
        vol_daily = vol_daily.where(vol_daily.notna(), self.VOL_FALLBACK)
        participation = participation.where(
            participation.notna(), self.PARTICIPATION_FALLBACK
        )

        impact_cells = abs_tr * self.impact_coeff * vol_daily * np.sqrt(participation)
        # zero-trade cells contribute EXACTLY 0 regardless of fill values
        impact_cells = impact_cells.where(traded, 0.0)
        impact = impact_cells.sum(axis=1)

        return spread + commission + impact


def from_config(cfg: CostConfig) -> CostModel:
    """Instantiate the cost model named by ``cfg.model`` with its fields."""
    if not isinstance(cfg, CostConfig):
        raise ConfigError(f"from_config expects a CostConfig, got {type(cfg).__name__}")
    if cfg.model == "zero":
        return COST_MODELS.create("zero")
    if cfg.model == "fixed_bps":
        return COST_MODELS.create("fixed_bps", bps=cfg.fixed_bps)
    if cfg.model == "realistic":
        return COST_MODELS.create(
            "realistic",
            commission_per_share=cfg.commission_per_share,
            half_spread_bps=cfg.half_spread_bps,
            impact_coeff=cfg.impact_coeff,
            adv_window=cfg.adv_window,
            vol_window=cfg.vol_window,
        )
    raise ConfigError(
        f"unknown cost model '{cfg.model}'; available: {COST_MODELS.names()}"
    )
