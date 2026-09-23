"""Tests for portfolio construction (QuantileLongShort) and cap_weights."""

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config.schema import PortfolioConfig
from alpha_lab.core.errors import ConfigError
from alpha_lab.data.synthetic import make_market
from alpha_lab.portfolio.constraints import cap_weights
from alpha_lab.portfolio.construction import CONSTRUCTORS, QuantileLongShort, from_config


def _scores(data, seed=0):
    """Seeded random score panel aligned to the market's close panel."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((len(data.dates), len(data.tickers))),
        index=data.dates,
        columns=data.tickers,
    )


# ---------------------------------------------------------------------------
# cap_weights
# ---------------------------------------------------------------------------


def test_cap_redistributes_pro_rata_preserving_side_sum():
    w = pd.DataFrame([[0.5, 0.3, 0.2, -0.5, -0.3, -0.2]], columns=list("ABCDEF"))
    capped = cap_weights(w, 0.4)
    row = capped.iloc[0]
    # excess 0.1 from A goes to B and C pro-rata (0.3:0.2)
    np.testing.assert_allclose(row[["A", "B", "C"]], [0.4, 0.36, 0.24], atol=1e-9)
    np.testing.assert_allclose(row[["D", "E", "F"]], [-0.4, -0.36, -0.24], atol=1e-9)
    assert abs(row[["A", "B", "C"]].sum() - 1.0) < 1e-9
    assert abs(row[["D", "E", "F"]].sum() + 1.0) < 1e-9
    assert (row.abs() <= 0.4 + 1e-9).all()


def test_cap_iterates_to_fixed_point():
    # first redistribution pushes B over the cap; needs a second pass
    w = pd.DataFrame([[0.6, 0.35, 0.05]], columns=list("ABC"))
    capped = cap_weights(w, 0.4)
    np.testing.assert_allclose(capped.iloc[0], [0.4, 0.4, 0.2], atol=1e-9)
    assert abs(capped.iloc[0].sum() - 1.0) < 1e-9


def test_cap_infeasible_side_shrinks_gross_without_error():
    # 3 names, cap 0.05, side target 1.0: 3 * 0.05 < 1.0 is infeasible;
    # documented behaviour is everyone at the cap, gross shrinks to 0.15
    w = pd.DataFrame([[0.5, 0.3, 0.2]], columns=list("ABC"))
    capped = cap_weights(w, 0.05)
    np.testing.assert_allclose(capped.iloc[0], [0.05, 0.05, 0.05], atol=1e-12)
    assert abs(capped.iloc[0].sum() - 0.15) < 1e-12


def test_cap_treats_nan_as_zero_and_validates():
    w = pd.DataFrame([[np.nan, 0.3, 0.2]], columns=list("ABC"))
    capped = cap_weights(w, 0.4)
    assert capped.notna().all().all()
    np.testing.assert_allclose(capped.iloc[0], [0.0, 0.3, 0.2], atol=1e-12)
    with pytest.raises(ConfigError):
        cap_weights(w, 0.0)


# ---------------------------------------------------------------------------
# QuantileLongShort basics
# ---------------------------------------------------------------------------


def test_infeasible_cap_warns_once(market_simple):
    """Regression: a cap too tight for the requested gross silently flattened
    every name to the cap, making gross_leverage/weighting/vol_target inert
    with no signal to the user."""
    import warnings as _warnings

    # 6 assets, quantile 0.5 -> 3/side at side target 1.0; 3 * 0.1 = 0.3 < 1.0
    ctor = QuantileLongShort(quantile=0.5, gross_leverage=2.0, max_weight=0.1, min_names=2)
    with pytest.warns(UserWarning, match="infeasible"):
        w = ctor.weights(_scores(market_simple), market_simple)
    # documented cap behaviour still applies: every active name at the cap
    active = w[w != 0.0].iloc[10].dropna()
    np.testing.assert_allclose(active.abs(), 0.1, atol=1e-12)
    # warns once per instance
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        ctor.weights(_scores(market_simple), market_simple)


def test_feasible_cap_does_not_warn(market_simple):
    import warnings as _warnings

    ctor = QuantileLongShort(quantile=0.5, gross_leverage=2.0, max_weight=0.5, min_names=2)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        ctor.weights(_scores(market_simple), market_simple)


def test_net_zero_and_gross_at_target(market_simple):
    ctor = QuantileLongShort(
        quantile=0.5, weighting="equal", gross_leverage=2.0, max_weight=0.5, min_names=2
    )
    w = ctor.weights(_scores(market_simple), market_simple)
    assert w.notna().all().all()
    assert (w.sum(axis=1).abs() < 1e-9).all()
    np.testing.assert_allclose(w.abs().sum(axis=1), 2.0, atol=1e-9)


def test_net_zero_and_gross_with_score_weighting(market_simple):
    ctor = QuantileLongShort(
        quantile=0.5, weighting="score", gross_leverage=2.0, max_weight=1.0, min_names=2
    )
    w = ctor.weights(_scores(market_simple), market_simple)
    assert (w.sum(axis=1).abs() < 1e-9).all()
    np.testing.assert_allclose(w.abs().sum(axis=1), 2.0, atol=1e-9)


def test_long_only_sums_to_gross(market_simple):
    ctor = QuantileLongShort(
        quantile=0.5, dollar_neutral=False, gross_leverage=1.0, max_weight=0.5, min_names=2
    )
    w = ctor.weights(_scores(market_simple), market_simple)
    assert (w.to_numpy() >= 0.0).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_bucket_size_matches_quantile(market_simple):
    scores = _scores(market_simple)
    for q, k in ((0.2, 1), (0.34, 2), (0.5, 3)):  # n = 6 valid names
        ctor = QuantileLongShort(quantile=q, max_weight=1.0, min_names=2)
        w = ctor.weights(scores, market_simple)
        assert ((w > 0).sum(axis=1) == k).all()
        assert ((w < 0).sum(axis=1) == k).all()


def test_min_names_produces_zero_rows(market_simple):
    # panel has 6 names; min_names=10 can never be met
    w = QuantileLongShort(min_names=10).weights(_scores(market_simple), market_simple)
    assert (w.to_numpy() == 0.0).all()

    # a single date with only 3 valid scores drops below min_names=4
    scores = _scores(market_simple).copy()
    t = market_simple.dates[50]
    scores.loc[t, market_simple.tickers[3:]] = np.nan
    w = QuantileLongShort(quantile=0.5, max_weight=1.0, min_names=4).weights(
        scores, market_simple
    )
    assert (w.loc[t] == 0.0).all()
    assert w.abs().sum(axis=1).gt(0).drop(t).all()


def test_universe_entrant_and_delisted_get_zero_weight(market):
    ctor = QuantileLongShort(quantile=0.2, max_weight=1.0, min_names=4)
    w = ctor.weights(_scores(market), market)
    assert w.notna().all().all()
    entrant, delisted = market.tickers[-1], market.tickers[-2]
    # scores exist on every date, but weight must be exactly 0 outside the
    # point-in-time universe (before entry / from the delisting date on)
    assert (w.loc[~market.universe[entrant], entrant] == 0.0).all()
    assert (w.loc[~market.universe[delisted], delisted] == 0.0).all()
    # sanity: both names do get weight at some point while members
    assert w[entrant].abs().sum() > 0
    assert w[delisted].abs().sum() > 0


def test_score_weighting_monotone_within_buckets(market_simple):
    ctor = QuantileLongShort(
        quantile=0.5, weighting="score", gross_leverage=2.0, max_weight=1.0, min_names=2
    )
    scores = _scores(market_simple)
    w = ctor.weights(scores, market_simple)
    for idx in (10, 100, 250, 399):
        row_w, row_s = w.iloc[idx], scores.iloc[idx]
        longs = row_w[row_w > 0]
        by_score = longs[row_s[longs.index].sort_values().index]
        assert (by_score.diff().dropna() >= -1e-12).all()  # higher score -> more weight
        shorts = row_w[row_w < 0]
        by_score = shorts[row_s[shorts.index].sort_values().index]
        assert (by_score.diff().dropna() >= -1e-12).all()  # lower score -> more negative


# ---------------------------------------------------------------------------
# vol targeting
# ---------------------------------------------------------------------------


def test_vol_target_is_point_in_time(market_simple):
    ctor = QuantileLongShort(
        quantile=0.5, gross_leverage=2.0, max_weight=1.0, vol_target=0.10, min_names=2
    )
    scores = _scores(market_simple)
    full = ctor.weights(scores, market_simple)
    for idx in (100, 150, 220, 300, 399):
        t = market_simple.dates[idx]
        sub = ctor.weights(scores.loc[:t], market_simple.slice_until(t))
        np.testing.assert_allclose(
            sub.loc[t].to_numpy(), full.loc[t].to_numpy(), atol=1e-12,
            err_msg=f"weights at {t} differ between truncated and full panels",
        )


def test_vol_target_shrinks_gross_when_trailing_vol_high():
    calm = make_market(
        n_assets=6, n_days=300, seed=5, base_vol=0.10, split_asset=False, universe_churn=False
    )
    wild = make_market(
        n_assets=6, n_days=300, seed=5, base_vol=0.50, split_asset=False, universe_churn=False
    )
    kwargs = dict(quantile=0.5, gross_leverage=2.0, max_weight=1.0, min_names=2)
    targeted = QuantileLongShort(vol_target=0.10, **kwargs)
    untargeted = QuantileLongShort(vol_target=None, **kwargs)

    g_wild = targeted.weights(_scores(wild), wild).abs().sum(axis=1).iloc[100:]
    g_calm = targeted.weights(_scores(calm), calm).abs().sum(axis=1).iloc[100:]
    g_off = untargeted.weights(_scores(wild), wild).abs().sum(axis=1).iloc[100:]

    assert g_wild.mean() < g_calm.mean()  # higher trailing vol -> smaller book
    assert (g_wild < g_off - 1e-9).all()  # and strictly below the unscaled gross
    np.testing.assert_allclose(g_off, 2.0, atol=1e-9)


# ---------------------------------------------------------------------------
# registry / config
# ---------------------------------------------------------------------------


def test_from_config_round_trip():
    cfg = PortfolioConfig(
        quantile=0.25,
        weighting="score",
        dollar_neutral=False,
        gross_leverage=1.5,
        max_weight=0.2,
        vol_target=0.15,
        vol_lookback=42,
    )
    ctor = from_config(cfg)
    assert isinstance(ctor, QuantileLongShort)
    assert ctor.quantile == cfg.quantile
    assert ctor.weighting == cfg.weighting
    assert ctor.dollar_neutral == cfg.dollar_neutral
    assert ctor.gross_leverage == cfg.gross_leverage
    assert ctor.max_weight == cfg.max_weight
    assert ctor.vol_target == cfg.vol_target
    assert ctor.vol_lookback == cfg.vol_lookback


def test_unknown_constructor_raises_config_error():
    with pytest.raises(ConfigError):
        CONSTRUCTORS.create("no_such_constructor")
    with pytest.raises(ConfigError):
        from_config(PortfolioConfig(method="no_such_constructor"))


def test_bad_params_raise_config_error():
    with pytest.raises(ConfigError):
        QuantileLongShort(quantile=0.7)
    with pytest.raises(ConfigError):
        QuantileLongShort(weighting="rank")
    with pytest.raises(ConfigError):
        QuantileLongShort(vol_target=-0.1)
