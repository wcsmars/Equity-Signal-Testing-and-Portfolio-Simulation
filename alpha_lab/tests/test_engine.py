"""BacktestEngine: timing law, drift/turnover math, isolation, determinism."""

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.engine import BacktestEngine
from alpha_lab.config.schema import BacktestConfig, WalkForwardConfig
from alpha_lab.core.errors import LookaheadError
from alpha_lab.core.interfaces import PortfolioConstructor, Signal, ZeroCost
from alpha_lab.core.types import MarketData


# -- in-test dummies ------------------------------------------------------


class ConstSignal(Signal):
    """Scores 1.0 everywhere. Stateless."""

    name = "const"

    def score(self, features, data):
        return pd.DataFrame(1.0, index=data.dates, columns=data.close.columns)


class StatefulSignal(Signal):
    """Records every fit call on the instance — for deepcopy isolation."""

    name = "stateful"

    def __init__(self):
        self.fit_calls = []

    def fit(self, features, data, train_dates):
        self.fit_calls.append((train_dates[0], train_dates[-1]))

    def score(self, features, data):
        return pd.DataFrame(1.0, index=data.dates, columns=data.close.columns)


class ImpulseConstructor(PortfolioConstructor):
    """Weight 1.0 on one asset at exactly one decision date, 0 elsewhere."""

    def __init__(self, date, asset):
        self.date = date
        self.asset = asset

    def weights(self, scores, data):
        w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        w.loc[self.date, self.asset] = 1.0
        return w


class FixedWeights(PortfolioConstructor):
    """Returns a pre-built weight panel, ignoring scores."""

    def __init__(self, panel):
        self.panel = panel

    def weights(self, scores, data):
        return self.panel.copy()


class ScoreProportional(PortfolioConstructor):
    """w = score / sum|score| per date; NaN scores -> 0 weight."""

    def weights(self, scores, data):
        filled = scores.fillna(0.0)
        gross = filled.abs().sum(axis=1)
        return filled.div(gross.where(gross > 0.0, 1.0), axis=0)


def _engine(constructor, config, signal=None):
    return BacktestEngine(
        signal=signal or ConstSignal(),
        constructor=constructor,
        cost_model=ZeroCost(),
        config=config,
    )


def _insample_cfg(lag=1, **kw):
    return BacktestConfig(execution_lag=lag, walkforward=None, **kw)


# -- impulse: the timing law ----------------------------------------------


@pytest.mark.parametrize("lag", [1, 2])
def test_impulse_gross_lands_exactly_at_lag(market_simple, lag):
    dates = market_simple.dates
    asset = market_simple.tickers[2]
    k = 50
    engine = _engine(ImpulseConstructor(dates[k], asset), _insample_cfg(lag=lag))
    result = engine.run(market_simple, features={})

    r = market_simple.returns()
    expected = r.loc[dates[k + lag], asset]
    assert result.gross_returns.loc[dates[k + lag]] == pytest.approx(expected)
    # zero everywhere else — in particular at d_k itself and at d_{k+1} when lag=2
    others = result.gross_returns.drop(dates[k + lag])
    assert (others == 0.0).all()
    assert result.gross_returns.loc[dates[k]] == 0.0
    if lag == 2:
        assert result.gross_returns.loc[dates[k + 1]] == 0.0
    # holdings are the shifted target weights
    assert result.holdings.loc[dates[k + lag], asset] == 1.0
    assert result.holdings.loc[dates[k], asset] == 0.0


# -- hand-computed drift / turnover (clause 6) -----------------------------


def _tiny_market():
    dates = pd.bdate_range("2021-01-04", periods=3)
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 55.0]}, index=dates
    )
    return MarketData.from_frames(close), dates


def test_drift_adjusted_turnover_matches_hand_computation():
    data, dates = _tiny_market()
    # r: A -> [NaN, 0.10, -0.10]; B -> [NaN, 0.0, 0.10]
    w = pd.DataFrame({"A": [0.6, 0.5, 0.2], "B": [0.4, 0.5, 0.8]}, index=dates)
    result = _engine(FixedWeights(w), _insample_cfg(lag=1)).run(data, features={})

    # H: t0 [0,0]; t1 [0.6,0.4]; t2 [0.5,0.5]
    # gross: t0 0; t1 0.6*0.10 = 0.06; t2 0.5*(-0.10)+0.5*0.10 = 0.0
    assert result.gross_returns.tolist() == pytest.approx([0.0, 0.06, 0.0])
    # The trade dated t2 executes at close t1, so H_{t1} has drifted with
    # r_{t1} (clause 6): [0.6*1.10, 0.4*1.0] / (1 + 0.06) = [33/53, 20/53].
    # trades t2 = [0.5 - 33/53, 0.5 - 20/53] = [-6.5/53, 6.5/53]; turnover 13/53
    assert result.turnover.tolist() == pytest.approx([0.0, 1.0, 13.0 / 53.0])
    trades_t2 = result.holdings.iloc[2] - pd.Series({"A": 33.0 / 53.0, "B": 20.0 / 53.0})
    assert trades_t2.abs().sum() == pytest.approx(13.0 / 53.0)


def test_drift_turnover_does_not_depend_on_same_day_close():
    """Regression: trades_t are fixed at close t-1, so perturbing only the
    FINAL day's close must not change the final day's turnover (the old
    clause-6 formula drifted with day-t returns — a rule-8 lookahead)."""
    rng = np.random.default_rng(0)
    n = 12
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, n)),
            "B": 50.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, n)),
        },
        index=dates,
    )
    w = pd.DataFrame({"A": 0.5, "B": 0.3}, index=dates)

    base = _engine(FixedWeights(w), _insample_cfg(lag=1)).run(
        MarketData.from_frames(close), features={}
    )
    bumped_close = close.copy()
    bumped_close.iloc[-1] *= 1.10  # information that prints AFTER the last trade
    bumped = _engine(FixedWeights(w), _insample_cfg(lag=1)).run(
        MarketData.from_frames(bumped_close), features={}
    )
    assert bumped.turnover.iloc[-1] == pytest.approx(base.turnover.iloc[-1], rel=1e-12)


def test_turnover_without_drift_adjustment():
    data, dates = _tiny_market()
    w = pd.DataFrame({"A": [0.6, 0.5, 0.2], "B": [0.4, 0.5, 0.8]}, index=dates)
    cfg = BacktestConfig(execution_lag=1, drift_adjust_turnover=False, walkforward=None)
    result = _engine(FixedWeights(w), cfg).run(data, features={})
    # trades t2 = H2 - H1 = [-0.1, 0.1] -> turnover 0.2
    assert result.turnover.tolist() == pytest.approx([0.0, 1.0, 0.2])


def test_first_row_trades_equal_first_holdings():
    data, dates = _tiny_market()
    w = pd.DataFrame({"A": [0.6, 0.5, 0.2], "B": [0.4, 0.5, 0.8]}, index=dates)
    # lag=1: H row 0 is all zero, so turnover row 0 must be 0 (not NaN)
    result = _engine(FixedWeights(w), _insample_cfg(lag=1)).run(data, features={})
    assert result.turnover.iloc[0] == 0.0
    assert not result.turnover.isna().any()


# -- lookahead defense ------------------------------------------------------


def test_engine_raises_lookahead_on_mutated_config(market_simple):
    cfg = BacktestConfig(execution_lag=1, walkforward=None)
    object.__setattr__(cfg, "execution_lag", 0)  # bypass schema validation
    engine = _engine(ImpulseConstructor(market_simple.dates[10], market_simple.tickers[0]), cfg)
    with pytest.raises(LookaheadError):
        engine.run(market_simple, features={})


# -- walk-forward: isolation, stitching, determinism -------------------------


def _wf_cfg(train=100, test=50, purge=2, embargo=0, lag=1):
    return BacktestConfig(
        execution_lag=lag,
        walkforward=WalkForwardConfig(
            scheme="rolling", train_days=train, test_days=test,
            purge_days=purge, embargo_days=embargo,
        ),
    )


def test_deepcopy_isolation_original_signal_untouched(market_simple):
    original = StatefulSignal()
    result = _engine(ScoreProportional(), _wf_cfg(), signal=original).run(
        market_simple, features={}
    )
    assert len(result.windows) >= 3  # a real multi-window run happened
    assert original.fit_calls == []  # every fit hit a deep copy, never this object


def test_stitched_scores_nan_and_weights_zero_before_first_test_window(market_simple):
    cfg = _wf_cfg()
    result = _engine(ScoreProportional(), cfg).run(market_simple, features={})
    first_test = result.windows[0].test_start
    dates = market_simple.dates
    p0 = dates.get_loc(first_test)
    assert p0 == 100 + 2  # train_days + gap

    pre = dates[:p0]
    assert result.scores.loc[pre].isna().all().all()
    assert (result.target_weights.loc[pre] == 0.0).all().all()
    # holdings lag one further day behind
    assert (result.holdings.loc[dates[: p0 + cfg.execution_lag]] == 0.0).all().all()
    # and from the first test date the constructor does take positions
    assert result.target_weights.loc[first_test].abs().sum() > 0.0
    assert result.scores.loc[first_test:].notna().any().any()
    assert result.meta["mode"] == "walkforward"


def test_determinism_identical_net_returns(market_simple):
    def run_once():
        return _engine(ScoreProportional(), _wf_cfg(), signal=StatefulSignal()).run(
            market_simple, features={}
        )

    a, b = run_once(), run_once()
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)
    pd.testing.assert_frame_equal(a.holdings, b.holdings)


def test_fit_sees_only_train_dates(market_simple):
    """Each deep copy's fit window must match the splitter's train windows."""

    seen = []

    class SpySignal(StatefulSignal):
        def fit(self, features, data, train_dates):
            seen.append((train_dates[0], train_dates[-1]))

    result = _engine(ScoreProportional(), _wf_cfg(), signal=SpySignal()).run(
        market_simple, features={}
    )
    assert seen == [(w.train_start, w.train_end) for w in result.windows]
