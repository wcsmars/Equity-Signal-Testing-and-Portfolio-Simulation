"""Bias suite: a real walk-forward backtest must show no survivorship,
in-sample bleed, purge, or split-adjustment bias.

Runs the actual BacktestEngine (not a mock) on the ``market`` fixture — 12
assets x 756 days with a 4:1 split on SYM00, a late entrant (last ticker),
and a delisting (second-to-last ticker).
"""

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest import BacktestEngine, RealisticCost
from alpha_lab.config.schema import BacktestConfig, WalkForwardConfig
from alpha_lab.core.types import MarketData
from alpha_lab.data.synthetic import make_market
from alpha_lab.portfolio import QuantileLongShort
from alpha_lab.signals import CrossSectionalMomentum
from alpha_lab.testing.checks import (
    assert_no_position_outside_universe,
    assert_purge_gap,
)

PURGE_DAYS = 5
EMBARGO_DAYS = 2


@pytest.fixture(scope="module")
def wf_result(market):
    """One full walk-forward backtest: real engine + quantile + realistic costs."""
    cfg = BacktestConfig(
        execution_lag=1,
        walkforward=WalkForwardConfig(
            scheme="rolling",
            train_days=252,
            test_days=63,
            purge_days=PURGE_DAYS,
            embargo_days=EMBARGO_DAYS,
        ),
    )
    engine = BacktestEngine(
        signal=CrossSectionalMomentum(window=63, skip=5),
        constructor=QuantileLongShort(quantile=0.25, max_weight=0.5),
        cost_model=RealisticCost(),
        config=cfg,
    )
    return engine.run(market)


def _membership_bounds(data: MarketData, ticker: str) -> tuple[int, int]:
    """(first_true_pos, last_true_pos) of the ticker's universe membership."""
    mask = data.universe[ticker].to_numpy()
    true_pos = np.flatnonzero(mask)
    return int(true_pos[0]), int(true_pos[-1])


# -- (a) universe discipline on a real run -----------------------------------

def test_no_position_outside_universe(wf_result, market):
    assert_no_position_outside_universe(wf_result, market)


def test_entrant_gets_no_weight_before_entry(wf_result, market):
    entrant = market.tickers[-1]
    entry_pos, _ = _membership_bounds(market, entrant)
    assert entry_pos > 0  # fixture sanity: it really is a late entrant
    lag = wf_result.meta["execution_lag"]
    assert (wf_result.target_weights[entrant].iloc[:entry_pos] == 0.0).all()
    assert (wf_result.holdings[entrant].iloc[: entry_pos + lag] == 0.0).all()
    # meaningful test: the entrant does participate after entering
    assert wf_result.holdings[entrant].abs().sum() > 0


def test_delisted_goes_to_zero_after_exit(wf_result, market):
    delisted = market.tickers[-2]
    _, last_pos = _membership_bounds(market, delisted)
    exit_pos = last_pos + 1
    assert exit_pos < len(market.dates)  # fixture sanity: it really delists
    lag = wf_result.meta["execution_lag"]
    assert (wf_result.target_weights[delisted].iloc[exit_pos:] == 0.0).all()
    # holdings may carry for at most `lag` days (H_t = W_{t-lag}), then zero
    assert (wf_result.holdings[delisted].iloc[exit_pos + lag :] == 0.0).all()
    # meaningful test: it was actually held while alive
    assert wf_result.holdings[delisted].abs().sum() > 0


# -- (b) no in-sample bleed ----------------------------------------------------

def test_zero_weights_before_first_test_window(wf_result, market):
    first_test_start = wf_result.windows[0].test_start
    pos = market.dates.get_loc(first_test_start)
    pre = wf_result.target_weights.iloc[:pos]
    assert float(pre.abs().to_numpy().sum()) == 0.0
    # and weights are nonzero somewhere after it, so the assertion has teeth
    assert wf_result.target_weights.iloc[pos:].abs().to_numpy().sum() > 0


# -- (c) purge gap ------------------------------------------------------------

def test_purge_gap(wf_result, market):
    assert wf_result.windows, "walk-forward run produced no windows"
    assert_purge_gap(wf_result.windows, market.dates, PURGE_DAYS, EMBARGO_DAYS)


# -- (d) split bias regression: costs must consume unadjusted_close ------------

def test_cost_model_consumes_unadjusted_close(market):
    # The fixture plants a 4:1 split on the first ticker at ~60% of the
    # sample: unadjusted_close is exactly 4x the adjusted close before it.
    split_name = market.tickers[0]
    ratio = (market.unadjusted_close[split_name] / market.close[split_name]).dropna()
    split_pos = int((ratio > 1.0).sum())
    assert ratio.iloc[0] == pytest.approx(4.0) and ratio.iloc[-1] == pytest.approx(1.0)

    pre_pos, post_pos = split_pos // 2, min(split_pos + 100, len(market.dates) - 1)
    trades = pd.DataFrame(0.0, index=market.dates, columns=market.tickers)
    trades.iloc[pre_pos, 0] = 0.05
    trades.iloc[post_pos, 0] = 0.05

    # commission-only model isolates the share math
    kwargs = dict(commission_per_share=0.01, half_spread_bps=0.0, impact_coeff=0.0)
    cost_real = RealisticCost(**kwargs).cost(trades, market, 1_000_000.0)

    # counterfactual: unadjusted_close replaced by the adjusted close
    counterfactual = MarketData.from_frames(
        market.close,
        open=market.open,
        high=market.high,
        low=market.low,
        volume=market.volume,
        unadjusted_close=market.close,
        universe=market.universe,
    )
    cost_fake = RealisticCost(**kwargs).cost(trades, counterfactual, 1_000_000.0)

    # pre-split: the adjusted price is 1/4 the raw price, so share counts and
    # commissions come out exactly 4x too high on the counterfactual
    assert cost_fake.iloc[pre_pos] == pytest.approx(4.0 * cost_real.iloc[pre_pos], rel=1e-9)
    assert cost_fake.iloc[pre_pos] > cost_real.iloc[pre_pos] > 0
    # post-split the two prices agree, so the costs must too
    assert cost_fake.iloc[post_pos] == pytest.approx(cost_real.iloc[post_pos], rel=1e-9)


# -- (e) survivorship discipline is universe-driven, not price-driven ----------

def test_universe_false_ticker_with_valid_prices_gets_no_weight():
    base = make_market(
        n_assets=8, n_days=360, seed=5, split_asset=False, universe_churn=False
    )
    excluded = base.tickers[-1]
    universe = pd.DataFrame(True, index=base.dates, columns=base.close.columns)
    universe[excluded] = False
    data = MarketData.from_frames(
        base.close,
        open=base.open,
        high=base.high,
        low=base.low,
        volume=base.volume,
        unadjusted_close=base.unadjusted_close,
        universe=universe,
    )
    # the point: prices are perfectly valid the whole time — only the
    # universe flag says "not tradable"
    assert data.close[excluded].notna().all()

    cfg = BacktestConfig(
        execution_lag=1,
        walkforward=WalkForwardConfig(
            train_days=126, test_days=42, purge_days=3, embargo_days=0
        ),
    )
    engine = BacktestEngine(
        signal=CrossSectionalMomentum(window=21, skip=1, min_names=4),
        # max_weight=1.0: with 7 in-universe names, quantile 0.25 gives 1
        # name/side — a tighter cap would be infeasible and (correctly) warn
        constructor=QuantileLongShort(quantile=0.25, max_weight=1.0),
        cost_model=RealisticCost(),
        config=cfg,
    )
    result = engine.run(data)

    assert (result.target_weights[excluded] == 0.0).all()
    assert (result.holdings[excluded] == 0.0).all()
    assert_no_position_outside_universe(result, data)
    # and the strategy did trade the includable names
    assert result.target_weights.abs().to_numpy().sum() > 0
