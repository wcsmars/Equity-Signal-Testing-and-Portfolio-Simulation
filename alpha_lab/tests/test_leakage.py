"""Adversarial leakage suite: every shipped component must be point-in-time,
and the detector itself must catch a planted leak.

Uses the ``market`` fixture (12x756 with a 4:1 split and universe churn) so
truncation invariance is tested on a panel with NaN lifecycles, not just a
clean rectangle. Dates are sampled (~8 across the sample plus the warm-up
rows per check) to keep runtime sane.
"""

import copy
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.costs import RealisticCost
from alpha_lab.core.errors import DataError, LookaheadError
from alpha_lab.core.interfaces import CostModel, Feature, PortfolioConstructor, Signal
from alpha_lab.core.types import MarketData
from alpha_lab.features import FEATURES
from alpha_lab.portfolio import QuantileLongShort
from alpha_lab.signals import CrossSectionalMomentum, ShortTermReversal
from alpha_lab.testing.checks import (
    PlantedLeakFeature,
    assert_constructor_pit,
    assert_cost_pit,
    assert_feature_pit,
    assert_signal_pit,
    assert_truncation_invariant,
    _resolve_dates,
    _resolve_trade_dates,
)

#: one non-default parameterization per registered feature. The coverage
#: test below fails when a new feature is registered without adding it here.
NON_DEFAULT_PARAMS = {
    "returns": {"window": 5},
    "momentum": {"window": 126, "skip": 5},
    "realized_vol": {"window": 21},
    "zscore_return": {"window": 63},
    "adv_dollars": {"window": 10},
    "rsi": {"window": 28},
}


def test_every_registered_feature_is_covered():
    assert set(FEATURES.names()) == set(NON_DEFAULT_PARAMS), (
        "a feature was (de)registered; update NON_DEFAULT_PARAMS so the "
        "leakage suite keeps covering the whole registry"
    )


@pytest.mark.parametrize("name", sorted(NON_DEFAULT_PARAMS))
def test_feature_pit_default_params(name, market):
    assert_feature_pit(FEATURES.create(name), market)


@pytest.mark.parametrize("name", sorted(NON_DEFAULT_PARAMS))
def test_feature_pit_non_default_params(name, market):
    assert_feature_pit(FEATURES.create(name, **NON_DEFAULT_PARAMS[name]), market)


# -- signals (feature + score composition) ---------------------------------

def test_xs_momentum_signal_pit(market):
    assert_signal_pit(CrossSectionalMomentum(), market)


def test_xs_momentum_signal_pit_short_window(market):
    assert_signal_pit(CrossSectionalMomentum(window=63, skip=5), market)


def test_xs_reversal_signal_pit(market):
    assert_signal_pit(ShortTermReversal(), market)


def test_xs_reversal_signal_pit_non_default(market):
    assert_signal_pit(ShortTermReversal(window=10, min_names=4), market)


# -- constructor ------------------------------------------------------------

@pytest.fixture(scope="module")
def momentum_scores(market):
    sig = CrossSectionalMomentum(window=63, skip=5)
    feats = {
        spec.key: FEATURES.create(spec.name, **spec.as_dict).compute(market)
        for spec in sig.required_features
    }
    return sig.score(feats, market)


def test_constructor_pit_no_vol_target(momentum_scores, market):
    ctor = QuantileLongShort(quantile=0.25, max_weight=0.5)
    assert_constructor_pit(ctor, momentum_scores, market)


def test_constructor_pit_vol_target(momentum_scores, market):
    ctor = QuantileLongShort(quantile=0.25, max_weight=0.5, vol_target=0.10)
    assert_constructor_pit(ctor, momentum_scores, market)


# -- cost model (Timing item 4: market inputs through t-1) ------------------

# The harness fills same-day missing cells (e.g. a delisted close) to probe
# missingness dependence, which triggers the per-cell raw-price fallback.
@pytest.mark.filterwarnings("ignore:unadjusted_close is missing on:UserWarning")
def test_realistic_cost_pit(market):
    trades = pd.DataFrame(0.0, index=market.dates, columns=market.tickers)
    trades.iloc[100, 0] = 0.05    # pre-split trade in the split name
    trades.iloc[100, 3] = -0.03
    trades.iloc[300, 1] = 0.02
    trades.iloc[500, 0] = -0.04   # post-split
    trades.iloc[520, 5] = 0.01
    dates = market.dates[[100, 101, 300, 500, 520, len(market.dates) - 1]]
    assert_cost_pit(RealisticCost(), trades, market, 1_000_000.0, dates=dates)


class _MarketInputCost(CostModel):
    """Probe one market input; lag 0 deliberately violates cost timing."""

    def __init__(self, source, lag):
        self.source = source
        self.lag = lag

    def cost(self, trades, data, portfolio_value):
        if self.source == "adv_dollars":
            panel = (data.unadjusted_close * data.volume).rolling(5).mean()
        elif self.source == "volatility":
            panel = data.returns().rolling(5).std()
        else:
            panel = getattr(data, self.source).astype(float)
        return (trades.abs() * panel.shift(self.lag)).sum(axis=1)


@pytest.mark.parametrize(
    "source",
    ["close", "open", "high", "low", "unadjusted_close", "volume",
     "adv_dollars", "volatility", "universe"],
)
@pytest.mark.parametrize("lag", [0, 1])
def test_cost_market_inputs_must_precede_trade_date(market_simple, source, lag):
    data = market_simple.slice_until(market_simple.dates[25])
    t = data.dates[20]
    trades = pd.DataFrame(0.0, index=data.dates, columns=data.tickers)
    # Only t has a trade: a correct checker must keep that supplied trade
    # while changing market data. Otherwise even the lagged model would fail.
    trades.loc[t, data.tickers[0]] = 0.05
    model = _MarketInputCost(source, lag)
    if lag == 0:
        with pytest.raises(LookaheadError, match="same-day"):
            assert_cost_pit(model, trades, data, 1_000_000.0, dates=[t])
    else:
        assert_cost_pit(model, trades, data, 1_000_000.0, dates=[t])


def test_cost_future_input_is_still_caught(market_simple):
    trades = pd.DataFrame(0.05, index=market_simple.dates, columns=market_simple.tickers)
    with pytest.raises(LookaheadError, match="truncation"):
        assert_cost_pit(
            _MarketInputCost("close", lag=-1), trades, market_simple,
            1_000_000.0, dates=[market_simple.dates[20]],
        )


def test_cost_pit_preserves_inputs_and_handles_missing_optional_fields(market_simple):
    data = copy.deepcopy(market_simple.slice_until(market_simple.dates[25]))
    data.open = data.high = data.low = data.universe = None
    original = copy.deepcopy(data)
    trades = pd.DataFrame(0.05, index=data.dates, columns=data.tickers)
    original_trades = trades.copy(deep=True)
    assert_cost_pit(
        RealisticCost(), trades, data, 1_000_000.0,
        dates=data.dates[[0, 1, 20, -1]],
    )
    pd.testing.assert_frame_equal(trades, original_trades)
    for field in ("close", "open", "high", "low", "volume", "unadjusted_close", "universe"):
        panel, before = getattr(data, field), getattr(original, field)
        if before is None:
            assert panel is None
        else:
            pd.testing.assert_frame_equal(panel, before)


# -- the detector detects -----------------------------------------------------

def test_planted_leak_is_caught(market_simple):
    with pytest.raises(LookaheadError, match="planted_leak"):
        assert_feature_pit(PlantedLeakFeature(), market_simple)


def test_full_sample_statistic_is_caught(market_simple):
    # Full-sample z-scoring (a CONVENTIONS.md forbidden pattern) changes past
    # values when future rows are removed, with no NaN-pattern tell — this
    # exercises the value-deviation branch of the detector.
    def full_sample_zscore(data):
        close = data.close
        return (close - close.mean()) / close.std()

    # A mid-sample date: on the first date the one-row slice has no standard
    # deviation at all, which the NaN-pattern branch reports instead.
    with pytest.raises(LookaheadError, match="deviation"):
        assert_truncation_invariant(
            full_sample_zscore, market_simple, dates=[market_simple.dates[200]]
        )
    # the default sample catches it too
    with pytest.raises(LookaheadError):
        assert_truncation_invariant(full_sample_zscore, market_simple)


def test_planted_leak_message_names_the_date(market_simple):
    t = market_simple.dates[200]
    with pytest.raises(LookaheadError) as excinfo:
        assert_feature_pit(PlantedLeakFeature(), market_simple, dates=[t])
    # the message must carry the offending date as ISO yyyy-mm-dd (ticker
    # names and "clause 1" contain digits and hyphens too, so match the shape)
    found = re.findall(r"\d{4}-\d{2}-\d{2}", str(excinfo.value))
    assert found and set(found) == {t.date().isoformat()}


def test_unknown_truncation_date_is_a_data_error(market_simple):
    with pytest.raises(DataError):
        assert_feature_pit(
            FEATURES.create("returns"), market_simple, dates=["1999-01-04"]
        )


def test_clean_callable_passes(market_simple):
    # sanity: the harness does not cry wolf on a trivially trailing compute
    assert_truncation_invariant(lambda d: d.close.rolling(5).mean(), market_simple)


# -- every exported checker must catch a planted violation -------------------


class _LeakySignal(Signal):
    """Scores with tomorrow's close — a lookahead the signal check must catch."""

    name = "leaky_signal"

    def score(self, features, data):
        return data.close.shift(-1) / data.close - 1.0


class _FullSampleZSignal(Signal):
    """Full-sample time-series z-score — no NaN tell, only value deviation."""

    name = "full_sample_z"

    def score(self, features, data):
        c = data.close
        return (c - c.mean()) / c.std()


@pytest.mark.parametrize("signal", [_LeakySignal(), _FullSampleZSignal()], ids=lambda s: s.name)
def test_signal_pit_catches_leaks(signal, market_simple):
    with pytest.raises(LookaheadError, match=signal.name):
        assert_signal_pit(signal, market_simple)


class _LeakyConstructor(PortfolioConstructor):
    """Ranks tomorrow's scores — the constructor check must catch it."""

    def weights(self, scores, data):
        nxt = scores.shift(-1)
        return nxt.sub(nxt.mean(axis=1), axis=0).fillna(0.0)


def test_constructor_pit_catches_leak(momentum_scores, market):
    with pytest.raises(LookaheadError, match="_LeakyConstructor"):
        assert_constructor_pit(_LeakyConstructor(), momentum_scores, market)


class _InfIfFutureCost(CostModel):
    """Cost at dates[-2] is +inf only when the next row exists — a future-row
    dependence that a relative tolerance on inf would wave through."""

    def __init__(self, last):
        self.last = last

    def cost(self, trades, data, portfolio_value):
        out = trades.abs().sum(axis=1) * 1e-4
        if data.dates[-1] == self.last:
            out.iloc[-2] = np.inf
        return out


def test_cost_pit_catches_infinite_value_change(market_simple):
    trades = pd.DataFrame(0.01, index=market_simple.dates, columns=market_simple.tickers)
    with pytest.raises(LookaheadError):
        assert_cost_pit(_InfIfFutureCost(market_simple.dates[-1]), trades, market_simple,
                        1_000_000.0, dates=[market_simple.dates[-2]])


def test_infinite_value_change_is_caught(market_simple):
    # the full-panel value is +inf only because a future row exists; a
    # relative tolerance on an infinite value must not wave it through
    last = market_simple.dates[-1]

    def inf_if_future(data):
        out = data.close.copy()
        if data.dates[-1] == last:
            out.iloc[-2] = np.inf
        return out

    with pytest.raises(LookaheadError, match="infinite"):
        assert_truncation_invariant(inf_if_future, market_simple, dates=[market_simple.dates[-2]])


# -- the default sample must reach the rows where a leak can hide -------------


def test_default_sample_covers_the_warm_up_and_skips_the_final_date(market_simple):
    dates = market_simple.dates
    n = len(dates)
    pos = [dates.get_loc(t) for t in _resolve_dates(market_simple, None)]
    assert pos == sorted(set(pos))
    # warm-up rows: the first two dates and the middle of the first sixth
    assert {0, 1, (n // 6) // 2} <= set(pos)
    # slicing at the final date returns the whole panel: that comparison
    # could never fail, so the sample ends one row earlier
    assert n - 1 not in pos and n - 2 in pos
    assert len([p for p in pos if p >= n // 6]) == 8

    # with the full-panel output, the first row holding a value is added
    full = market_simple.close.rolling(30).mean()
    with_full = [dates.get_loc(t) for t in _resolve_dates(market_simple, None, full)]
    assert set(with_full) == set(pos) | {29}
    # an all-missing output adds nothing and does not fail
    empty = pd.DataFrame(np.nan, index=dates, columns=market_simple.tickers)
    assert list(_resolve_dates(market_simple, None, empty)) == list(_resolve_dates(market_simple, None))


class _BackfilledMomentum(Feature):
    """21-day momentum whose warm-up is filled from LATER rows."""

    name = "backfilled_momentum"
    lookback = 21

    def __init__(self, fill):
        self.fill = fill

    def compute(self, data):
        mom = data.close / data.close.shift(21) - 1.0
        if self.fill == "bfill":
            return mom.bfill()
        if self.fill == "bfill_limit":
            return mom.bfill(limit=3)  # only rows 18-20 are filled
        if self.fill == "interpolate":
            return mom.interpolate(limit_direction="both")
        return mom.fillna(mom.iloc[21:42].mean())  # a later statistic


@pytest.mark.parametrize("fill", ["bfill", "bfill_limit", "interpolate", "later_mean"])
def test_backfilled_warm_up_is_caught_by_the_default_sample(fill, market_simple):
    # Every row from ~1/6 of the index onwards is clean: the future values
    # sit only in the warm-up, which the default sample used to skip.
    feature = _BackfilledMomentum(fill)
    clean_from = market_simple.dates[len(market_simple.dates) // 6:]
    assert_feature_pit(feature, market_simple, dates=clean_from[::40])
    with pytest.raises(LookaheadError, match="backfilled_momentum"):
        assert_feature_pit(feature, market_simple)


class _BackfilledSignal(Signal):
    """Scores the warm-up with the first available later score."""

    name = "backfilled_signal"

    def score(self, features, data):
        return (data.close / data.close.shift(21) - 1.0).bfill()


def test_signal_pit_catches_backfilled_warm_up(market_simple):
    clean_from = market_simple.dates[len(market_simple.dates) // 6:]
    assert_signal_pit(_BackfilledSignal(), market_simple, dates=clean_from[::40])
    with pytest.raises(LookaheadError, match="backfilled_signal"):
        assert_signal_pit(_BackfilledSignal(), market_simple)


class _BackfilledConstructor(PortfolioConstructor):
    """Takes its first positions from scores that do not exist yet."""

    def weights(self, scores, data):
        centred = scores.sub(scores.mean(axis=1), axis=0)
        return centred.bfill().fillna(0.0)


def test_constructor_pit_catches_backfilled_warm_up(market_simple):
    # clean panel: scores are missing only in the 21-row warm-up
    scores = market_simple.close / market_simple.close.shift(21) - 1.0
    ctor = _BackfilledConstructor()
    clean_from = market_simple.dates[len(market_simple.dates) // 6:]
    assert_constructor_pit(ctor, scores, market_simple, dates=clean_from[::40])
    with pytest.raises(LookaheadError, match="_BackfilledConstructor"):
        assert_constructor_pit(ctor, scores, market_simple)


# -- the cost check must look at dates that carry a trade --------------------


class _SameDayVolumeCost(CostModel):
    """Prices with the trade date's own volume (forbidden by Timing item 4)."""

    def cost(self, trades, data, portfolio_value):
        relative = data.volume / data.volume.rolling(5).mean()
        return (trades.abs() * relative).sum(axis=1) * 1e-4


def _sparse_trades(data, rows):
    trades = pd.DataFrame(0.0, index=data.dates, columns=data.tickers)
    trades.iloc[rows, 0] = 0.05
    return trades


def test_cost_pit_default_sample_is_drawn_from_traded_dates(market_simple):
    dates = market_simple.dates
    rows = list(range(5, len(dates), 20))  # a periodic rebalance
    trades = _sparse_trades(market_simple, rows)
    # none of the evenly spread truncation dates carries a trade ...
    assert trades.loc[_resolve_dates(market_simple, None)].abs().to_numpy().sum() == 0.0
    # ... so the cost check samples the traded dates instead
    sampled = _resolve_trade_dates(market_simple, trades, None, "cost model")
    assert len(sampled) == 8
    assert sampled[0] == dates[rows[0]] and sampled[-1] == dates[rows[-1]]
    assert (trades.loc[sampled].abs().sum(axis=1) > 0).all()
    # fewer traded dates than samples: every one of them is checked
    few = _sparse_trades(market_simple, [40, 90, 300])
    assert list(_resolve_trade_dates(market_simple, few, None, "cost model")) == list(dates[[40, 90, 300]])


def test_cost_pit_catches_same_day_input_with_sparse_trades(market_simple):
    trades = _sparse_trades(market_simple, list(range(5, len(market_simple.dates), 20)))
    with pytest.raises(LookaheadError, match="same-day volume"):
        assert_cost_pit(_SameDayVolumeCost(), trades, market_simple, 1_000_000.0)
    # the shipped model passes on the same sparse trades
    assert_cost_pit(RealisticCost(), trades, market_simple, 1_000_000.0)


def test_cost_pit_refuses_to_pass_without_a_trade(market_simple):
    no_trades = pd.DataFrame(0.0, index=market_simple.dates, columns=market_simple.tickers)
    with pytest.raises(DataError, match="vacuously"):
        assert_cost_pit(_SameDayVolumeCost(), no_trades, market_simple, 1_000_000.0)
    nan_trades = no_trades * np.nan  # NaN trade == no trade
    with pytest.raises(DataError, match="vacuously"):
        assert_cost_pit(_SameDayVolumeCost(), nan_trades, market_simple, 1_000_000.0)
    # explicit dates: at least one of them must carry a trade
    trades = _sparse_trades(market_simple, [25])
    idle = market_simple.dates[[24, 26, 200]]
    with pytest.raises(DataError, match="vacuously"):
        assert_cost_pit(_SameDayVolumeCost(), trades, market_simple, 1_000_000.0, dates=idle)
    with pytest.raises(LookaheadError, match="same-day volume"):
        assert_cost_pit(
            _SameDayVolumeCost(), trades, market_simple, 1_000_000.0,
            dates=market_simple.dates[[24, 25]],
        )


def test_cost_pit_handles_whole_number_panels(market_simple):
    # A volume file of whole shares loads as int64; the probe values are not
    # integers, so the perturbation must not be written into an integer panel.
    base = market_simple.slice_until(market_simple.dates[119])
    data = MarketData.from_frames(
        base.close, volume=base.volume.astype("int64"), unadjusted_close=base.unadjusted_close
    )
    assert (data.volume.dtypes == "int64").all()
    trades = _sparse_trades(data, [30, 60, 90])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert_cost_pit(RealisticCost(), trades, data, 1_000_000.0)
        # and the probe still bites on the integer panel
        with pytest.raises(LookaheadError, match="same-day volume"):
            assert_cost_pit(_SameDayVolumeCost(), trades, data, 1_000_000.0)
    assert (data.volume.dtypes == "int64").all()  # the caller's panel is untouched


class _SameDayAvailabilityCost(CostModel):
    """Charges only names whose close has printed on the trade date."""

    def cost(self, trades, data, portfolio_value):
        return (trades.abs() * data.close.notna()).sum(axis=1) * 1e-4


def test_cost_pit_probes_same_day_missingness(market):
    # The delisted name has no close on t. A model keyed on whether the close
    # exists is caught only because the probe also fills missing cells.
    delisted = market.tickers[-2]
    t = market.dates[-10]
    assert np.isnan(market.close.loc[t, delisted])
    trades = pd.DataFrame(0.0, index=market.dates, columns=market.tickers)
    trades.loc[t, delisted] = 0.05
    with pytest.raises(LookaheadError, match="same-day close"):
        assert_cost_pit(_SameDayAvailabilityCost(), trades, market, 1_000_000.0, dates=[t])
