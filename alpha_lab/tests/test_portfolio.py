"""Tests for portfolio construction (QuantileLongShort) and cap_weights."""

import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config.schema import PortfolioConfig
from alpha_lab.core.errors import ConfigError
from alpha_lab.core.types import MarketData
from alpha_lab.data.synthetic import make_market
from alpha_lab.portfolio.constraints import cap_weights
from alpha_lab.portfolio.construction import (
    CONSTRUCTORS,
    VOL_TARGET_MAX_SCALE,
    QuantileLongShort,
    from_config,
)


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


def test_cap_preserves_feasible_gross_despite_small_iteration_hint():
    w = pd.DataFrame([[0.6, 0.35, 0.05]], columns=list("ABC"))
    capped = cap_weights(w, 0.4, n_iter=1)
    np.testing.assert_allclose(capped.iloc[0], [0.4, 0.4, 0.2], atol=1e-12)


def test_cap_redistributes_to_tiny_positive_positions():
    w = pd.DataFrame([[1.0, 1e-15, 1e-15]], columns=list("ABC"))
    capped = cap_weights(w, 0.4)
    np.testing.assert_allclose(capped.iloc[0], [0.4, 0.3, 0.3], atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"gross_leverage": np.nan}, {"max_weight": np.inf}, {"vol_target": np.nan}, {"vol_lookback": 5.5}, {"min_names": True}])
def test_constructor_rejects_invalid_direct_parameters(kwargs):
    with pytest.raises(ConfigError):
        QuantileLongShort(**kwargs)


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
        _warnings.simplefilter("error", UserWarning)
        ctor.weights(_scores(market_simple), market_simple)


def test_feasible_cap_does_not_warn(market_simple):
    import warnings as _warnings

    ctor = QuantileLongShort(quantile=0.5, gross_leverage=2.0, max_weight=0.5, min_names=2)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
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


@pytest.mark.parametrize("weighting", ["equal", "score"])
def test_longs_are_top_scores_and_shorts_bottom(market_simple, weighting):
    scores = _scores(market_simple)
    ctor = QuantileLongShort(quantile=0.34, weighting=weighting, max_weight=1.0, min_names=2)
    w = ctor.weights(scores, market_simple)
    for t in market_simple.dates[::37]:
        row, sc = w.loc[t], scores.loc[t]
        longs, shorts = sc[row > 0], sc[row < 0]
        assert len(longs) and len(shorts)
        assert longs.min() > sc[row == 0].max() > shorts.max()
        assert row[sc.idxmax()] > 0 and row[sc.idxmin()] < 0


def test_long_only_holds_top_scores(market_simple):
    scores = _scores(market_simple)
    ctor = QuantileLongShort(quantile=0.34, dollar_neutral=False, max_weight=1.0, min_names=2)
    w = ctor.weights(scores, market_simple)
    t = market_simple.dates[100]
    held = scores.loc[t][w.loc[t] > 0]
    assert set(held.index) == set(scores.loc[t].nlargest(len(held)).index)


@pytest.mark.parametrize("weighting", ["equal", "score"])
def test_infinite_scores_count_as_no_opinion(market_simple, weighting):
    scores = _scores(market_simple)
    t = market_simple.dates[50]
    scores.loc[t, scores.columns[0]] = np.inf
    scores.loc[t, scores.columns[1]] = -np.inf
    ctor = QuantileLongShort(quantile=0.25, weighting=weighting, max_weight=1.0, min_names=2)
    w = ctor.weights(scores, market_simple)
    row = w.loc[t]
    assert row.iloc[:2].eq(0.0).all()
    assert abs(row.sum()) < 1e-9
    assert row.abs().sum() == pytest.approx(2.0)


def test_object_dtype_scores_are_accepted(market_simple):
    # e.g. a boolean signal stitched into a float panel becomes object dtype
    scores = _scores(market_simple)
    ctor = QuantileLongShort(quantile=0.34, max_weight=1.0, min_names=2)
    expected = ctor.weights(scores, market_simple)
    got = ctor.weights(scores.astype(object), market_simple)
    pd.testing.assert_frame_equal(got, expected)
    flags = (scores > 0).astype(object)
    assert ctor.weights(flags, market_simple).notna().all().all()


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


def test_single_valid_name_is_a_zero_row_even_with_min_names_zero(market_simple):
    # one name cannot be ranked against anything: min_names below 2 is
    # floored at 2, in both modes
    scores = _scores(market_simple).copy()
    t = market_simple.dates[50]
    scores.loc[t, market_simple.tickers[1:]] = np.nan
    for dollar_neutral in (True, False):
        ctor = QuantileLongShort(
            quantile=0.5, dollar_neutral=dollar_neutral, max_weight=1.0, min_names=0
        )
        w = ctor.weights(scores, market_simple)
        assert (w.loc[t] == 0.0).all()
        assert w.abs().sum(axis=1).gt(0).drop(t).all()


def test_bucket_size_survives_float_product_just_below_an_integer():
    # 100 * 0.29 == 28.999999999999996 in floating point; the bucket is 29
    assert 100 * 0.29 < 29
    dates = pd.bdate_range("2020-01-02", periods=2)
    tickers = [f"T{i:03d}" for i in range(100)]
    data = MarketData.from_frames(pd.DataFrame(50.0, index=dates, columns=tickers))
    rng = np.random.default_rng(1)
    scores = pd.DataFrame(rng.standard_normal((2, 100)), index=dates, columns=tickers)
    w = QuantileLongShort(quantile=0.29, max_weight=1.0).weights(scores, data)
    assert ((w > 0).sum(axis=1) == 29).all()
    assert ((w < 0).sum(axis=1) == 29).all()


def test_scores_wider_than_the_data_get_zero_weight_without_warnings(market_simple):
    # a date or ticker the data does not carry is outside the universe; the
    # alignment used to go through an object frame and a pandas FutureWarning
    scores = _scores(market_simple)
    wide = scores.copy()
    wide["EXTRA"] = 5.0
    wide.loc[market_simple.dates[-1] + pd.Timedelta(1, unit="D")] = 1.0
    ctor = QuantileLongShort(quantile=0.34, max_weight=1.0, min_names=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = ctor.weights(wide, market_simple)
    assert got.index.equals(wide.index) and got.columns.equals(wide.columns)
    assert (got["EXTRA"] == 0.0).all()
    assert (got.iloc[-1] == 0.0).all()
    pd.testing.assert_frame_equal(
        got.loc[scores.index, scores.columns], ctor.weights(scores, market_simple), check_freq=False
    )


# ---------------------------------------------------------------------------
# ties
# ---------------------------------------------------------------------------


def _flat_market(tickers, n_days=3):
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    return MarketData.from_frames(pd.DataFrame(100.0, index=dates, columns=list(tickers)))


def _one_row(ctor, row, data):
    scores = pd.DataFrame([row] * len(data.dates), index=data.dates, columns=data.tickers)
    return ctor.weights(scores.astype(float), data).iloc[0]


@pytest.mark.parametrize("weighting", ["equal", "score"])
@pytest.mark.parametrize("dollar_neutral", [True, False])
def test_fully_tied_row_is_a_zero_row(weighting, dollar_neutral):
    # no cross-sectional dispersion is no opinion; the stable sort used to
    # short the first tickers and buy the last at full gross
    data = _flat_market("ABCDEF")
    ctor = QuantileLongShort(
        quantile=0.34, weighting=weighting, dollar_neutral=dollar_neutral,
        max_weight=1.0, min_names=2,
    )
    assert (_one_row(ctor, [1.0] * 6, data) == 0.0).all()
    assert (_one_row(ctor, [0.0] * 6, data) == 0.0).all()


def test_binary_scores_never_short_a_top_scored_name():
    data = _flat_market("ABCDEF")
    ctor = QuantileLongShort(quantile=0.34, max_weight=1.0, min_names=2)  # k = 2 per side
    row = _one_row(ctor, [1, 1, 1, 1, 1, 0], data)
    # five names tie for the 2 long slots (0.4 slot each -> +0.2) and for
    # the 1 short slot left beside F (0.2 slot each -> -0.1): net +0.1 each
    np.testing.assert_allclose(row[list("ABCDE")], 0.1, atol=1e-12)
    assert row["F"] == pytest.approx(-0.5)
    assert abs(row.sum()) < 1e-12  # still dollar-neutral
    flags = pd.DataFrame(
        [[True, True, True, True, True, False]] * 3, index=data.dates, columns=data.tickers
    ).astype(object)
    pd.testing.assert_series_equal(ctor.weights(flags, data).iloc[0], row)


@pytest.mark.parametrize("weighting", ["equal", "score"])
@pytest.mark.parametrize("dollar_neutral", [True, False])
def test_tied_scores_do_not_depend_on_column_order(weighting, dollar_neutral):
    tickers = list("ABCDEFGH")
    by_ticker = dict(zip(tickers, [3.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, -1.0]))
    ctor = QuantileLongShort(
        quantile=0.25, weighting=weighting, dollar_neutral=dollar_neutral,
        max_weight=1.0, min_names=2,
    )
    base = _one_row(ctor, [by_ticker[c] for c in tickers], _flat_market(tickers))
    for seed in range(5):
        order = list(np.random.default_rng(seed).permutation(tickers))
        got = _one_row(ctor, [by_ticker[c] for c in order], _flat_market(order))
        pd.testing.assert_series_equal(got[tickers], base, check_exact=False, atol=1e-12)
    # equal scores -> equal weights
    assert base["B"] == base["C"] == base["D"]
    assert base["F"] == base["G"]
    assert base.abs().sum() == pytest.approx(2.0)  # no group straddles both edges


def test_tie_at_the_bucket_edge_splits_the_straddled_slot():
    data = _flat_market("ABCDEFGH")
    ctor = QuantileLongShort(quantile=0.25, gross_leverage=2.0, max_weight=1.0, min_names=2)
    # k = 2: A is inside the long bucket; B, C, D tie for the one slot left
    row = _one_row(ctor, [3, 2, 2, 2, 1, 0, 0, -1], data)
    assert row["A"] == pytest.approx(0.5)
    np.testing.assert_allclose(row[list("BCD")], 0.5 / 3.0, atol=1e-12)
    assert row["E"] == 0.0
    # H is inside the short bucket; F and G tie for the other slot
    assert row["H"] == pytest.approx(-0.5)
    np.testing.assert_allclose(row[list("FG")], -0.25, atol=1e-12)
    # a tie that lies wholly inside a bucket changes nothing
    inside = _one_row(ctor, [3, 3, 2, 1, 0.5, 0, -1, -1], data)
    np.testing.assert_allclose(inside, [0.5, 0.5, 0, 0, 0, 0, -0.5, -0.5], atol=1e-12)


def test_score_weighting_matches_hand_formula():
    # weight ~ distance from the bucket's own worst score, so the least
    # extreme of the k names carries (almost) nothing: documented behaviour
    data = _flat_market("ABCDEFGHIJ")
    ctor = QuantileLongShort(quantile=0.3, weighting="score", max_weight=1.0, min_names=2)
    row = _one_row(ctor, [3, 2, 1, 0, -1, -2, -3, -4, -5, -6], data)  # k = 3
    np.testing.assert_allclose(row[list("ABC")], [2 / 3, 1 / 3, 0.0], atol=1e-8)
    np.testing.assert_allclose(row[list("HIJ")], [0.0, -1 / 3, -2 / 3], atol=1e-8)
    assert (row[list("DEFG")] == 0.0).all()
    assert row["C"] > 0.0 > row["H"]  # in the bucket, at weight ~1e-9
    # with k = 2 each side is in effect a single name
    pair = _one_row(
        QuantileLongShort(quantile=0.2, weighting="score", max_weight=1.0, min_names=2),
        [3, 2, 1, 0, -1, -2, -3, -4, -5, -6], data,
    )
    np.testing.assert_allclose(pair[["A", "B", "I", "J"]], [1.0, 0.0, 0.0, -1.0], atol=1e-8)


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


def test_vol_target_scale_matches_hand_formula():
    data = make_market(n_assets=6, n_days=300, seed=5, base_vol=0.30, split_asset=False, universe_churn=False)
    kwargs = dict(quantile=0.5, gross_leverage=2.0, max_weight=1.0, min_names=2, vol_lookback=63)
    scores = _scores(data)
    raw = QuantileLongShort(vol_target=None, **kwargs).weights(scores, data)
    scaled = QuantileLongShort(vol_target=0.10, **kwargs).weights(scores, data)
    sigma = data.returns().rolling(63, min_periods=31).std()
    for t in data.dates[[120, 200, 299]]:
        w = raw.loc[t]
        est = np.sqrt((w.pow(2) * sigma.loc[t].pow(2)).sum())
        scale = np.clip(0.10 / np.sqrt(252) / est, 0.0, 3.0)
        expected = cap_weights((w * scale).to_frame().T, 1.0).iloc[0]
        np.testing.assert_allclose(scaled.loc[t], expected, rtol=1e-12, atol=1e-15)
        assert scaled.loc[t].abs().max() <= 1.0 + 1e-12


def _calm_market(**kwargs):
    return make_market(
        n_assets=6, n_days=300, seed=5, base_vol=0.02, split_asset=False, **kwargs
    )


def test_vol_target_scale_is_clipped_at_the_max_scale_and_recapped():
    assert VOL_TARGET_MAX_SCALE == 3.0  # documented bound: gross <= 3x gross_leverage
    calm = _calm_market(universe_churn=False)
    scores = _scores(calm)
    kwargs = dict(quantile=0.5, gross_leverage=2.0, min_names=2, vol_lookback=63)
    raw = QuantileLongShort(vol_target=None, max_weight=10.0, **kwargs).weights(scores, calm)
    # an unreachable target on a calm panel pins the scale at the clamp
    levered = QuantileLongShort(vol_target=5.0, max_weight=10.0, **kwargs).weights(scores, calm)
    np.testing.assert_allclose(levered.iloc[100:], 3.0 * raw.iloc[100:], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(levered.iloc[100:].abs().sum(axis=1), 6.0, rtol=1e-12)
    # the per-name cap is re-applied AFTER scaling: 3 names a side x 0.5
    capped = QuantileLongShort(vol_target=5.0, max_weight=0.5, **kwargs).weights(scores, calm)
    assert capped.abs().to_numpy().max() <= 0.5 + 1e-12
    np.testing.assert_allclose(capped.iloc[100:].abs().sum(axis=1), 3.0, atol=1e-9)
    pd.testing.assert_frame_equal(capped.iloc[100:], cap_weights(3.0 * raw.iloc[100:], 0.5))
    # with a cap that binds only for some names, the excess is redistributed
    partly = QuantileLongShort(
        vol_target=5.0, weighting="score", max_weight=1.5, **kwargs
    ).weights(scores, calm)
    raw_score = QuantileLongShort(
        vol_target=None, weighting="score", max_weight=10.0, **kwargs
    ).weights(scores, calm)
    assert (3.0 * raw_score.iloc[100:]).abs().to_numpy().max() > 1.5  # the cap does bind
    pd.testing.assert_frame_equal(partly.iloc[100:], cap_weights(3.0 * raw_score.iloc[100:], 1.5))


def test_vol_target_warm_up_rows_keep_unscaled_weights():
    # no name has vol_lookback // 2 = 20 returns before row 20: there is no
    # estimate at all, and the documented choice is the unscaled book
    calm = _calm_market(universe_churn=False)
    scores = _scores(calm)
    kwargs = dict(quantile=0.5, gross_leverage=2.0, max_weight=10.0, min_names=2, vol_lookback=40)
    raw = QuantileLongShort(vol_target=None, **kwargs).weights(scores, calm)
    scaled = QuantileLongShort(vol_target=5.0, **kwargs).weights(scores, calm)
    np.testing.assert_allclose(raw.iloc[:20].abs().sum(axis=1), 2.0, rtol=1e-12)
    pd.testing.assert_frame_equal(scaled.iloc[:20], raw.iloc[:20])
    # the estimate exists from exactly row 20 (returns 1..20) on
    np.testing.assert_allclose(scaled.iloc[20:30], 3.0 * raw.iloc[20:30], rtol=1e-12, atol=1e-15)


def test_vol_target_stays_on_when_a_held_name_lacks_vol_history():
    """One held name without a vol estimate (a new entrant under a fast
    signal) used to drop the whole row back to scale 1.0, so the gross
    jumped from the vol-scaled level to the full gross_leverage."""
    data = make_market(
        n_assets=12, n_days=400, seed=7, base_vol=0.40, split_asset=False, universe_churn=True
    )
    entrant = data.tickers[-1]
    entry = int(np.flatnonzero(data.universe[entrant].to_numpy())[0])
    scores = _scores(data)
    scores[entrant] = 10.0  # top score from its first tradable day
    kwargs = dict(quantile=0.25, gross_leverage=2.0, max_weight=0.5, min_names=4, vol_lookback=63)
    raw = QuantileLongShort(vol_target=None, **kwargs).weights(scores, data)
    scaled = QuantileLongShort(vol_target=0.05, **kwargs).weights(scores, data)

    young = slice(entry, entry + 31)  # held, but fewer than 31 returns of its own
    sigma = data.returns().rolling(63, min_periods=31).std()
    assert (raw[entrant].iloc[young] > 0).all()
    assert sigma[entrant].iloc[young].isna().all() and sigma[entrant].iloc[entry + 31] > 0

    gross = scaled.abs().sum(axis=1)
    # a 40%-vol market against a 5% target: far below the unscaled 2.0 ...
    assert gross.iloc[entry - 40 : entry].max() < 0.5
    # ... and it stays vol-scaled through the entrant's first month
    assert gross.iloc[young].max() < 0.5

    # hand formula: the entrant carries the largest trailing vol known that day
    for pos in (entry, entry + 15, entry + 30):
        t = data.dates[pos]
        w = raw.loc[t]
        s = sigma.loc[t].copy()
        s[entrant] = sigma.loc[t].max()
        est = np.sqrt((w.pow(2) * s.pow(2)).sum())
        scale = np.clip(0.05 / np.sqrt(252) / est, 0.0, 3.0)
        expected = cap_weights((w * scale).to_frame().T, 0.5).iloc[0]
        np.testing.assert_allclose(scaled.loc[t], expected, rtol=1e-12, atol=1e-15)
    # once it has its own estimate the proxy is gone
    t = data.dates[entry + 31]
    est = np.sqrt((raw.loc[t].pow(2) * sigma.loc[t].pow(2)).sum())
    np.testing.assert_allclose(
        scaled.loc[t], raw.loc[t] * (0.05 / np.sqrt(252) / est), rtol=1e-12, atol=1e-15
    )


def test_vol_target_with_entrant_is_point_in_time():
    # the proxy is a same-date cross-sectional maximum: truncating the panel
    # at t must not change row t
    data = make_market(
        n_assets=12, n_days=400, seed=7, base_vol=0.40, split_asset=False, universe_churn=True
    )
    entrant = data.tickers[-1]
    entry = int(np.flatnonzero(data.universe[entrant].to_numpy())[0])
    scores = _scores(data)
    scores[entrant] = 10.0
    ctor = QuantileLongShort(quantile=0.25, max_weight=0.5, vol_target=0.05, min_names=4)
    full = ctor.weights(scores, data)
    for pos in (entry, entry + 10, entry + 30):
        t = data.dates[pos]
        sub = ctor.weights(scores.loc[:t], data.slice_until(t))
        np.testing.assert_allclose(sub.loc[t].to_numpy(), full.loc[t].to_numpy(), atol=1e-12)


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
