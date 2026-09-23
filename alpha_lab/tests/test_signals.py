"""Signal library tests.

Covers: predictive rank-IC of xs_momentum against next-day returns on the
long panel, per-row z-score discipline, exact-math spot checks against the
raw feature, universe masking, min_names gating, zero-dispersion rows,
end-to-end point-in-time invariance (feature + signal composed), and
config-driven construction via build_signal.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config.schema import SignalConfig
from alpha_lab.core.errors import ConfigError, DataError
from alpha_lab.core.interfaces import Signal
from alpha_lab.core.types import MarketData
from alpha_lab.data.synthetic import make_market
from alpha_lab.features.library import Momentum, WindowReturn
from alpha_lab.features.store import FeatureStore
from alpha_lab.signals.momentum import CrossSectionalMomentum
from alpha_lab.signals.registry import SIGNALS, build_signal
from alpha_lab.signals.reversal import ShortTermReversal


def _score(signal, data, store=None):
    """Compute the signal's required features and score them, like the runner."""
    store = FeatureStore() if store is None else store
    features = store.compute_all(signal.required_features, data)
    return signal.score(features, data)


# --------------------------------------------------------------------------
# predictive power: rank-IC on the long panel
# --------------------------------------------------------------------------

def test_xs_momentum_rank_ic_positive(market_long):
    scores = _score(CrossSectionalMomentum(), market_long)
    # EVALUATION ONLY: r_{t+1} needs a negative shift, which is forbidden in
    # library code. Test code may compute forward returns to *measure* the
    # signal's predictive power after the fact.
    fwd_returns = market_long.returns().shift(-1)

    scoreable = scores.index[scores.notna().sum(axis=1) >= 5]
    # drop the final date (no next-day return exists). All ~1259 scoreable
    # dates are used: the mean IC is ~0.017 on any trailing window (including
    # the last 700), but a 700-date sample alone leaves the t-stat marginal
    # (~1.9) purely from sample size — same test, more power, here.
    eval_dates = [t for t in scoreable if t != market_long.dates[-1]]
    assert len(eval_dates) >= 700  # sanity: the panel really supports this

    ics = []
    for t in eval_dates:
        s, f = scores.loc[t], fwd_returns.loc[t]
        valid = s.notna() & f.notna()
        # Spearman rank-IC as the Pearson correlation of average ranks: the
        # same value as method="spearman", which would import scipy (not a
        # dependency).
        ics.append(s[valid].rank().corr(f[valid].rank()))
    ics = pd.Series(ics, dtype=float).dropna()
    assert len(ics) >= 700

    mean_ic = ics.mean()
    t_stat = mean_ic / (ics.std() / np.sqrt(len(ics)))
    # The synthetic panel plants persistent (AR(1), phi=0.995) drifts, so
    # 12-1 momentum must show a solidly positive next-day rank-IC.
    assert mean_ic > 0.01
    assert t_stat > 2.0


# --------------------------------------------------------------------------
# per-row z-score discipline
# --------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [CrossSectionalMomentum(), ShortTermReversal()])
def test_scored_rows_are_standardized(market, signal):
    scores = _score(signal, market)
    rows = scores.dropna(how="all")
    assert len(rows) > 300
    # every surviving row has >= min_names valid values (the gate held) ...
    assert (rows.notna().sum(axis=1) >= signal.min_names).all()
    # ... and is exactly standardized: mean 0, sample std 1
    np.testing.assert_allclose(rows.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(rows.std(axis=1), 1.0, atol=1e-10)


def test_momentum_score_matches_manual_zscore(market_simple):
    signal = CrossSectionalMomentum(window=63, skip=5, min_names=5)
    scores = _score(signal, market_simple)
    feat = Momentum(window=63, skip=5).compute(market_simple)
    for pos in (100, 250, 399):
        row = feat.iloc[pos]
        expected = (row - row.mean()) / row.std()
        pd.testing.assert_series_equal(scores.iloc[pos], expected, check_names=False)


def test_reversal_score_is_z_of_minus_return(market_simple):
    signal = ShortTermReversal(window=5, min_names=5)
    scores = _score(signal, market_simple)
    feat = WindowReturn(window=5).compute(market_simple)
    for pos in (50, 300):
        row = -feat.iloc[pos]
        expected = (row - row.mean()) / row.std()
        pd.testing.assert_series_equal(scores.iloc[pos], expected, check_names=False)
    # the biggest loser over the window gets the highest score
    t = market_simple.dates[300]
    assert scores.loc[t].idxmax() == feat.loc[t].idxmin()


# --------------------------------------------------------------------------
# universe and NaN discipline
# --------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [CrossSectionalMomentum(), ShortTermReversal()])
def test_entrant_has_nan_score_before_entry(market, signal):
    entrant = market.tickers[-1]
    first_valid = market.close[entrant].first_valid_index()
    pre_entry = market.dates[market.dates < first_valid]
    assert len(pre_entry) > 30  # sanity: fixture really has churn
    scores = _score(signal, market)
    assert scores.loc[pre_entry, entrant].isna().all()


def test_universe_mask_overrides_valid_prices(market_simple):
    # SYM00 keeps valid prices but is declared out of the universe: the
    # score must be NaN from the mask alone, not from missing data.
    universe = pd.DataFrame(
        True, index=market_simple.dates, columns=market_simple.tickers
    )
    universe["SYM00"] = False
    data = MarketData.from_frames(market_simple.close, universe=universe)
    scores = _score(ShortTermReversal(window=5, min_names=5), data)
    assert scores["SYM00"].isna().all()
    # the remaining 5 names still meet min_names=5 and score normally
    others = scores.drop(columns=["SYM00"]).iloc[10:]
    assert others.notna().all().all()


def test_min_names_blanks_thin_cross_sections():
    tiny = make_market(
        n_assets=3, n_days=120, seed=5, split_asset=False, universe_churn=False
    )
    # 3 names < min_names=5 (default): every row must be all-NaN
    assert _score(ShortTermReversal(window=5), tiny).isna().all().all()
    assert _score(CrossSectionalMomentum(window=20, skip=1), tiny).isna().all().all()
    # loosening the gate to the panel width makes the same rows scoreable
    loose = _score(ShortTermReversal(window=5, min_names=3), tiny)
    assert loose.iloc[10:].notna().all().all()


def test_zero_dispersion_row_is_nan():
    dates = pd.bdate_range("2020-01-01", periods=30)
    close = pd.DataFrame(100.0, index=dates, columns=[f"S{i}" for i in range(6)])
    data = MarketData.from_frames(close)
    # constant prices -> every 5-day return is exactly 0 -> row std 0
    scores = _score(ShortTermReversal(window=5), data)
    assert scores.isna().all().all()


# --------------------------------------------------------------------------
# end-to-end point-in-time: feature + signal truncation invariance composed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [CrossSectionalMomentum(), ShortTermReversal()])
def test_end_to_end_pit_at_sampled_dates(market, signal):
    """Recompute features on data.slice_until(t), score, compare row t.

    Sampled dates straddle the split (~row 453) and the delisting (~row 604).
    Passing proves the composed pipeline sees the same world at t whether or
    not the future exists in the panel (CONVENTIONS.md rules 1-2).
    """
    store = FeatureStore()  # fingerprint distinguishes truncations, safe to share
    full_scores = _score(signal, market, store)
    for pos in (300, 420, 470, 620, 755):
        t = market.dates[pos]
        truncated = _score(signal, market.slice_until(t), store)
        assert truncated.index[-1] == t
        pd.testing.assert_series_equal(
            truncated.loc[t], full_scores.loc[t], check_names=False
        )


# --------------------------------------------------------------------------
# registry and config round trip
# --------------------------------------------------------------------------

def test_registry_has_exact_contract_names():
    assert SIGNALS.names() == ["xs_momentum", "xs_reversal"]


def test_build_signal_round_trip():
    cfg = SignalConfig(name="xs_momentum", params={"window": 126, "skip": 10, "min_names": 4})
    signal = build_signal(cfg)
    assert isinstance(signal, CrossSectionalMomentum)
    assert (signal.window, signal.skip, signal.min_names) == (126, 10, 4)
    assert signal.required_features[0].key == "momentum(skip=10,window=126)"

    reversal = build_signal(SignalConfig(name="xs_reversal", params={"window": 10}))
    assert isinstance(reversal, ShortTermReversal)
    assert (reversal.window, reversal.min_names) == (10, 5)
    assert reversal.required_features[0].key == "returns(window=10)"


def test_build_signal_defaults():
    signal = build_signal(SignalConfig())  # schema default: xs_momentum, {}
    assert isinstance(signal, CrossSectionalMomentum)
    assert (signal.window, signal.skip, signal.min_names) == (252, 21, 5)


def test_build_signal_unknown_name_is_config_error():
    with pytest.raises(ConfigError):
        build_signal(SignalConfig(name="does_not_exist"))


def test_build_signal_bad_params_are_config_errors():
    with pytest.raises(ConfigError):
        build_signal(SignalConfig(name="xs_reversal", params={"nope": 1}))
    with pytest.raises(ConfigError):
        build_signal(SignalConfig(name="xs_momentum", params={"window": 10, "skip": 20}))
    with pytest.raises(ConfigError):
        build_signal(SignalConfig(name="xs_momentum", params={"min_names": 1}))
    with pytest.raises(ConfigError):
        build_signal(SignalConfig(name="xs_reversal", params={"window": 2.5}))


def test_missing_required_feature_is_data_error(market_simple):
    with pytest.raises(DataError):
        CrossSectionalMomentum().score({}, market_simple)


def test_signals_are_stateless():
    # neither signal overrides fit: the base-class no-op is the contract
    assert CrossSectionalMomentum.fit is Signal.fit
    assert ShortTermReversal.fit is Signal.fit
