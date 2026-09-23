"""Adversarial leakage suite: every shipped component must be point-in-time,
and the detector itself must catch a planted leak.

Uses the ``market`` fixture (12x756 with a 4:1 split and universe churn) so
truncation invariance is tested on a panel with NaN lifecycles, not just a
clean rectangle. Dates are sampled (~8 per check) to keep runtime sane.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.costs import RealisticCost
from alpha_lab.core.errors import DataError, LookaheadError
from alpha_lab.core.interfaces import CostModel
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

    with pytest.raises(LookaheadError, match="deviation"):
        assert_truncation_invariant(full_sample_zscore, market_simple)


def test_planted_leak_message_names_the_date(market_simple):
    with pytest.raises(LookaheadError) as excinfo:
        assert_feature_pit(PlantedLeakFeature(), market_simple)
    # the message must carry an actionable date (ISO yyyy-mm-dd)
    assert any(ch.isdigit() for ch in str(excinfo.value))
    assert "-" in str(excinfo.value)


def test_unknown_truncation_date_is_a_data_error(market_simple):
    with pytest.raises(DataError):
        assert_feature_pit(
            FEATURES.create("returns"), market_simple, dates=["1999-01-04"]
        )


def test_clean_callable_passes(market_simple):
    # sanity: the harness does not cry wolf on a trivially trailing compute
    assert_truncation_invariant(lambda d: d.close.rolling(5).mean(), market_simple)
