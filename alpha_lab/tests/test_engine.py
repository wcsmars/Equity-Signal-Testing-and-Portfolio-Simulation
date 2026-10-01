"""BacktestEngine: timing law, drift/turnover math, isolation, determinism."""

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.costs import RealisticCost
from alpha_lab.backtest.engine import BacktestEngine
from alpha_lab.config.schema import BacktestConfig, WalkForwardConfig
from alpha_lab.core.errors import DataError, LookaheadError
from alpha_lab.core.interfaces import CostModel, PortfolioConstructor, Signal, ZeroCost
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
    # the lag recorded for downstream checks is the lag that was applied
    assert result.meta["execution_lag"] == lag
    assert result.meta["mode"] == "insample"


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


@pytest.mark.parametrize("scheme", ["rolling", "expanding"])
def test_fit_receives_only_train_rows_of_every_feature_panel(market_simple, scheme):
    """Feature panels are computed on the full sample, so the slice the engine
    hands to ``fit`` is the only thing keeping later rows out of the fit."""

    full = {"x": market_simple.close.copy(), "y": market_simple.returns()}
    seen = []

    class FeatureSpy(Signal):
        name = "feature_spy"

        def fit(self, features, data, train_dates):
            assert set(features) == set(full)
            for key, panel in features.items():
                assert panel.index.equals(train_dates), f"{key} leaks rows outside the train window"
                pd.testing.assert_frame_equal(panel, full[key].loc[train_dates])
            assert data.dates.equals(train_dates)
            seen.append((features["x"].index[0], features["x"].index[-1], len(features["y"])))

        def score(self, features, data):
            # scoring is a per-date map of the full panels; the engine keeps
            # only each window's test rows of the result
            assert all(panel.index.equals(data.dates) for panel in features.values())
            return features["x"]

    cfg = BacktestConfig(
        execution_lag=1,
        walkforward=WalkForwardConfig(
            scheme=scheme, train_days=100, test_days=50, purge_days=2, embargo_days=0,
        ),
    )
    result = _engine(ScoreProportional(), cfg, signal=FeatureSpy()).run(
        market_simple, features=full
    )
    assert len(result.windows) >= 3
    pos = market_simple.dates.get_loc
    assert seen == [
        (w.train_start, w.train_end, pos(w.train_end) - pos(w.train_start) + 1)
        for w in result.windows
    ]
    if scheme == "rolling":
        assert {n for _, _, n in seen} == {100}
    else:
        assert [n for _, _, n in seen] == [100 + 50 * i for i in range(len(seen))]


# -- cost wiring (clauses 6-7: trades priced by the cost model, same date) ---


class SpyCost(CostModel):
    """Records the trades it is handed and returns a known per-date series."""

    def __init__(self, series=None):
        self.series = series
        self.trades = None
        self.portfolio_values = []

    def cost(self, trades, data, portfolio_value):
        self.trades = trades.copy()
        self.portfolio_values.append(portfolio_value)
        if self.series is not None:
            return self.series
        return pd.Series(0.001 * (1 + np.arange(len(trades))), index=trades.index)


def test_cost_model_receives_drifted_trades_and_is_charged_same_date():
    data, dates = _tiny_market()
    w = pd.DataFrame({"A": [0.6, 0.5, 0.2], "B": [0.4, 0.5, 0.8]}, index=dates)
    spy = SpyCost()
    engine = BacktestEngine(ConstSignal(), FixedWeights(w), spy, _insample_cfg(lag=1))
    result = engine.run(data, features={})
    # hand-computed in test_drift_adjusted_turnover_matches_hand_computation
    assert spy.trades.loc[dates[1]].tolist() == pytest.approx([0.6, 0.4])
    assert spy.trades.loc[dates[2]].tolist() == pytest.approx([-6.5 / 53.0, 6.5 / 53.0])
    expected_costs = pd.Series([0.001, 0.002, 0.003], index=dates)
    pd.testing.assert_series_equal(result.costs, expected_costs, check_names=False)
    pd.testing.assert_series_equal(
        result.net_returns, result.gross_returns - expected_costs, check_names=False
    )


def test_cost_model_is_priced_at_the_engine_portfolio_value():
    """Share counts and participation scale with the book: the cost model must
    be handed the configured reference NAV, not a default."""
    data, dates = _tiny_market()
    w = pd.DataFrame({"A": [0.6, 0.5, 0.2], "B": [0.4, 0.5, 0.8]}, index=dates)
    for nav in (50_000_000.0, 250_000.0):
        spy = SpyCost()
        result = BacktestEngine(
            ConstSignal(), FixedWeights(w), spy, _insample_cfg(lag=1), portfolio_value=nav
        ).run(data, features={})
        assert spy.portfolio_values == [nav]
        assert result.meta["portfolio_value"] == nav
    # the default reference book
    spy = SpyCost()
    BacktestEngine(ConstSignal(), FixedWeights(w), spy, _insample_cfg(lag=1)).run(data, features={})
    assert spy.portfolio_values == [1_000_000.0]


def test_realistic_cost_scales_with_the_engine_portfolio_value():
    """End to end: with a real impact model a 100x larger book pays 10x the
    impact per unit of NAV (square-root participation), through the engine."""
    n = 80
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = pd.DataFrame({"A": [100.0 if i % 2 == 0 else 104.0 for i in range(n)]}, index=dates)
    volume = pd.DataFrame({"A": 1_000_000.0}, index=dates)
    data = MarketData.from_frames(close, volume=volume, unadjusted_close=close.copy())
    w = pd.DataFrame({"A": 0.0}, index=dates)
    w.loc[dates[60]:, "A"] = 0.5
    costs = {}
    for nav in (1_000_000.0, 100_000_000.0):
        model = RealisticCost(commission_per_share=0.0, half_spread_bps=0.0,
                              adv_window=20, vol_window=10)
        result = BacktestEngine(
            ConstSignal(), FixedWeights(w), model, _insample_cfg(lag=1), portfolio_value=nav
        ).run(data, features={})
        costs[nav] = result.costs.loc[dates[61]]  # the entry trade
    assert costs[1_000_000.0] > 0.0
    assert costs[100_000_000.0] == pytest.approx(10.0 * costs[1_000_000.0], rel=1e-9)


@pytest.mark.parametrize("bad", ["misaligned", "nan", "inf", "negative", "frame", "text"])
def test_bad_cost_model_output_raises(bad):
    data, dates = _tiny_market()
    w = pd.DataFrame({"A": [0.6, 0.5, 0.2], "B": [0.4, 0.5, 0.8]}, index=dates)
    series = {
        "misaligned": pd.Series([0.001], index=dates[:1]),
        "nan": pd.Series([0.001, np.nan, 0.0], index=dates),
        "inf": pd.Series([0.001, np.inf, 0.0], index=dates),
        "negative": pd.Series([0.001, -0.1, 0.0], index=dates),
        "frame": pd.DataFrame({"cost": [0.0, 0.0, 0.0]}, index=dates),
        "text": pd.Series(["invalid"] * 3, index=dates),
    }[bad]
    engine = BacktestEngine(ConstSignal(), FixedWeights(w), SpyCost(series), _insample_cfg(lag=1))
    with pytest.raises(DataError):
        engine.run(data, features={})


@pytest.mark.parametrize("bad", [np.inf, -np.inf, "invalid"])
def test_invalid_constructor_weights_raise(bad):
    data, dates = _tiny_market()
    weights = pd.DataFrame(bad, index=dates, columns=data.close.columns)
    with pytest.raises(DataError, match="weights"):
        _engine(FixedWeights(weights), _insample_cfg()).run(data, features={})


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_reference_portfolio_value_raises(value):
    data, _ = _tiny_market()
    engine = BacktestEngine(ConstSignal(), ScoreProportional(), ZeroCost(), _insample_cfg(), value)
    with pytest.raises(DataError, match="portfolio_value"):
        engine.run(data, features={})


def test_bankruptcy_does_not_restart_flat_book():
    data, dates = _tiny_market()
    # A short at 10x leverage loses all capital on the +10% second bar.
    weights = pd.DataFrame({"A": -10.0, "B": 0.0}, index=dates)
    with pytest.raises(DataError, match="insolvency"):
        _engine(FixedWeights(weights), _insample_cfg()).run(data, features={})


def test_costs_cannot_make_portfolio_insolvent():
    data, dates = _tiny_market()
    costs = SpyCost(pd.Series([0.0, 2.0, 0.0], index=dates))
    engine = BacktestEngine(ConstSignal(), ScoreProportional(), costs, _insample_cfg())
    with pytest.raises(DataError, match="after costs"):
        engine.run(data, features={})


def test_missing_held_returns_are_counted_and_warned():
    dates = pd.bdate_range("2024-01-01", periods=3)
    data = MarketData(pd.DataFrame({"A": [100.0, np.nan, 101.0]}, index=dates))
    weights = pd.DataFrame({"A": 0.5}, index=dates)
    with pytest.warns(UserWarning, match="2 held asset-return cells"):
        result = _engine(FixedWeights(weights), _insample_cfg()).run(data, {})
    assert result.meta["missing_held_return_cells"] == 2
    assert result.net_returns.eq(0.0).all()


def test_precomputed_features_require_exact_alignment():
    data, _ = _tiny_market()
    with pytest.raises(DataError, match="feature.*aligned"):
        _engine(ScoreProportional(), _insample_cfg()).run(data, {"broken": data.close.iloc[:-1]})


def test_insample_score_columns_are_aligned_by_label():
    class ReorderedSignal(ConstSignal):
        def score(self, features, data):
            return pd.DataFrame({"B": 2.0, "A": 1.0}, index=data.dates)

    data, _ = _tiny_market()
    result = _engine(ScoreProportional(), _insample_cfg(), ReorderedSignal()).run(data, {})
    assert result.scores.columns.equals(data.close.columns)
    assert result.target_weights.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3])


@pytest.mark.parametrize("mode", [None, WalkForwardConfig(train_days=1, test_days=1, purge_days=0)])
def test_duplicate_signal_labels_rejected_in_both_modes(mode):
    class DuplicateSignal(ConstSignal):
        def score(self, features, data):
            return pd.concat([data.close, data.close], axis=1)

    data, _ = _tiny_market()
    cfg = BacktestConfig(execution_lag=1, walkforward=mode)
    with pytest.raises(DataError, match="duplicate"):
        _engine(ScoreProportional(), cfg, DuplicateSignal()).run(data, {})


# -- walk-forward stitching -------------------------------------------------


def test_walkforward_stitches_scores_by_label_not_position(market_simple):
    class ReversedColumns(Signal):
        """Correct scores, columns in reverse order (as a pivot might return)."""

        name = "reversed"

        def score(self, features, data):
            ranks = pd.DataFrame(
                np.tile(np.arange(data.close.shape[1], dtype=float), (len(data.dates), 1)),
                index=data.dates, columns=data.close.columns,
            )
            return ranks[list(reversed(data.close.columns))]

    result = _engine(ScoreProportional(), _wf_cfg(), signal=ReversedColumns()).run(
        market_simple, features={}
    )
    t = result.windows[0].test_start
    expected = pd.Series(np.arange(market_simple.close.shape[1], dtype=float), index=market_simple.close.columns)
    pd.testing.assert_series_equal(result.scores.loc[t], expected, check_names=False)


def test_walkforward_rejects_scores_for_unknown_tickers(market_simple):
    class ExtraTicker(Signal):
        name = "extra"

        def score(self, features, data):
            out = data.close.copy()
            out["ZZZ"] = 1.0  # would be dropped silently by a reindex
            return out

    with pytest.raises(DataError, match="not in the data panel"):
        _engine(ScoreProportional(), _wf_cfg(), signal=ExtraTicker()).run(market_simple, features={})


def test_each_window_scored_by_its_own_fit(market_simple):
    """Test rows carry the score of the fit on THEIR train window, and fit
    only ever sees data through that window's train_end."""

    class TrainEndSignal(Signal):
        name = "train_end"

        def fit(self, features, data, train_dates):
            assert data.dates[-1] == train_dates[-1]
            self.stamp = float(train_dates[-1].toordinal())

        def score(self, features, data):
            return pd.DataFrame(self.stamp, index=data.dates, columns=data.close.columns)

    result = _engine(ScoreProportional(), _wf_cfg(), signal=TrainEndSignal()).run(
        market_simple, features={}
    )
    dates = market_simple.dates
    for w in result.windows:
        block = result.scores.loc[w.test_start : w.test_end]
        assert (block == float(w.train_end.toordinal())).all().all()
        assert dates.get_loc(w.test_start) - dates.get_loc(w.train_end) > 1  # purge gap
