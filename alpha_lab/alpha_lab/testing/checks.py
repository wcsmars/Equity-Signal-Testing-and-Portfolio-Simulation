"""Adversarial leakage and bias checks — the mechanical CONVENTIONS.md enforcer.

The core primitive is truncation invariance (CONVENTIONS.md clause 1): a
point-in-time computation must produce the same row t on
``data.slice_until(t)`` — "the world as known at close t" — as on the full
panel. Any dependence on future rows changes row t between the two runs and
is reported as a :class:`~alpha_lab.core.errors.LookaheadError` naming the
offending date, tickers, and maximum deviation, so a leak is a loud test
failure instead of a quiet Sharpe boost.

Layered wrappers compose the same invariant up the pipeline: features
(:func:`assert_feature_pit`), feature+signal (:func:`assert_signal_pit`),
constructor (:func:`assert_constructor_pit`), and cost model
(:func:`assert_cost_pit`, which checks the cost inputs' t-1 cutoff in
CONVENTIONS.md, Timing item 4). Panel-level bias checks cover universe discipline
(:func:`assert_no_position_outside_universe`) and walk-forward purging
(:func:`assert_purge_gap`). :class:`PlantedLeakFeature` is a deliberately
broken feature proving the detector actually detects.
"""

from __future__ import annotations

import copy
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from alpha_lab.core.errors import DataError, LookaheadError
from alpha_lab.core.interfaces import CostModel, Feature, PortfolioConstructor, Signal
from alpha_lab.core.registry import Registry
from alpha_lab.core.results import BacktestResult, WalkForwardWindow
from alpha_lab.core.types import MarketData
from alpha_lab.features.library import FEATURES

#: number of truncation dates sampled when the caller does not pass ``dates``
DEFAULT_SAMPLES = 8


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

def _resolve_dates(data: MarketData, dates: Iterable | None) -> pd.DatetimeIndex:
    """Normalize ``dates`` to timestamps present in ``data.dates``.

    When ``dates`` is None, sample ``DEFAULT_SAMPLES`` positions spread from
    ~1/6 into the index (skipping the mostly-degenerate warm-up range where
    long-lookback features are all-NaN anyway) through the final date.
    """
    idx = data.dates
    if dates is None:
        if len(idx) < 2:
            raise DataError("need at least 2 dates to test truncation invariance")
        lo = len(idx) // 6
        pos = np.unique(np.linspace(lo, len(idx) - 1, DEFAULT_SAMPLES).round().astype(int))
        return idx[pos]
    out = []
    for t in dates:
        ts = pd.Timestamp(t)
        if ts not in idx:
            raise DataError(f"truncation date {ts.date()} is not in the data index")
        out.append(ts)
    if not out:
        raise DataError("empty dates iterable")
    return pd.DatetimeIndex(out)


def _compare_rows(
    trunc_row: pd.Series,
    full_row: pd.Series,
    t: pd.Timestamp,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    """Raise LookaheadError unless the two rows match (NaN pattern + values)."""
    trunc_row = trunc_row.reindex(full_row.index)
    trunc_na = trunc_row.isna()
    full_na = full_row.isna()
    mismatch = trunc_na != full_na
    if mismatch.any():
        bad = list(full_row.index[mismatch])
        raise LookaheadError(
            f"{label}: NaN pattern at {t.date()} changes under truncation for "
            f"tickers {bad} — row {t.date()} computed on data sliced at "
            f"{t.date()} must equal the full-panel row (CONVENTIONS.md clause 1)"
        )
    both = ~full_na
    if not both.any():
        return
    diff = (trunc_row[both] - full_row[both]).abs()
    tol = atol + rtol * full_row[both].abs()
    viol = diff > tol
    if viol.any():
        worst = diff[viol].sort_values(ascending=False)
        raise LookaheadError(
            f"{label}: values at {t.date()} change under truncation for tickers "
            f"{list(worst.index)}; max deviation {worst.iloc[0]:.6e} "
            f"(rtol={rtol}, atol={atol}) — the computation is using future rows"
        )


def _compare_scalars(
    trunc_val: float,
    full_val: float,
    t: pd.Timestamp,
    rtol: float,
    atol: float,
    label: str,
    check: str = "truncation",
) -> None:
    """Scalar version of :func:`_compare_rows` for per-date cost series."""
    trunc_na, full_na = pd.isna(trunc_val), pd.isna(full_val)
    if trunc_na != full_na:
        raise LookaheadError(
            f"{label}: value at {t.date()} is {'NaN' if trunc_na else trunc_val!r} "
            f"after {check} but {'NaN' if full_na else full_val!r} originally"
        )
    if full_na:
        return
    dev = abs(float(trunc_val) - float(full_val))
    if dev > atol + rtol * abs(float(full_val)):
        raise LookaheadError(
            f"{label}: value at {t.date()} changes under {check} "
            f"(changed {trunc_val!r} vs original {full_val!r}, deviation {dev:.6e}) "
            f"— cost inputs may only use market data through t-1 "
            f"(CONVENTIONS.md, Timing item 4)"
        )


def _compute_feature_panels(
    specs: Sequence, data: MarketData, registry: Registry
) -> dict[str, pd.DataFrame]:
    """Fresh feature panels for ``specs`` on ``data`` (no cache, no reuse)."""
    return {
        spec.key: registry.create(spec.name, **spec.as_dict).compute(data)
        for spec in dict.fromkeys(specs)
    }


# --------------------------------------------------------------------------
# truncation-invariance checks
# --------------------------------------------------------------------------

def assert_truncation_invariant(
    compute: Callable[[MarketData], pd.DataFrame],
    data: MarketData,
    dates: Iterable | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    label: str = "compute",
) -> None:
    """Assert ``compute(data.slice_until(t)).loc[t] == compute(data).loc[t]``.

    ``compute`` is any callable MarketData -> wide DataFrame. For each
    sampled t (default: ~``DEFAULT_SAMPLES`` dates across the valid range)
    the truncated row t must match the full-panel row t exactly — identical
    NaN pattern, values equal within ``rtol``/``atol``. Raises LookaheadError
    with the offending date, tickers, and max deviation.
    """
    full = compute(data)
    if not isinstance(full, pd.DataFrame):
        raise DataError(f"{label} must return a DataFrame, got {type(full).__name__}")
    for t in _resolve_dates(data, dates):
        sliced = compute(data.slice_until(t))
        if t not in sliced.index:
            raise LookaheadError(
                f"{label}: output on data sliced at {t.date()} has no row for "
                f"{t.date()} — the computation dropped the as-of date"
            )
        _compare_rows(sliced.loc[t], full.loc[t], t, rtol, atol, label)


def assert_feature_pit(
    feature: Feature,
    data: MarketData,
    dates: Iterable | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Truncation invariance for one Feature (CONVENTIONS.md clause 1)."""
    assert_truncation_invariant(
        feature.compute, data, dates=dates, rtol=rtol, atol=atol,
        label=f"feature '{feature.name}'",
    )


def assert_signal_pit(
    signal: Signal,
    data: MarketData,
    dates: Iterable | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    registry: Registry = FEATURES,
) -> None:
    """Composed feature+signal point-in-time check (clauses 1 + 2).

    For each sampled t the signal's ``required_features`` are RECOMPUTED from
    scratch on ``data.slice_until(t)`` and scored on that slice; row t must
    match the full-panel pipeline (features and score both computed on all of
    ``data``). This catches leaks in the features, in ``score``, and in their
    composition. The signal is deep-copied per evaluation so no state can
    bleed between runs; ``fit`` is not called — this checks the pure scoring
    path (train-window fitting discipline is the walk-forward engine's job).
    """
    label = f"signal '{signal.name}'"
    specs = tuple(signal.required_features)
    full_scores = copy.deepcopy(signal).score(
        _compute_feature_panels(specs, data, registry), data
    )
    for t in _resolve_dates(data, dates):
        sliced = data.slice_until(t)
        trunc_scores = copy.deepcopy(signal).score(
            _compute_feature_panels(specs, sliced, registry), sliced
        )
        if t not in trunc_scores.index:
            raise LookaheadError(f"{label}: no score row for as-of date {t.date()}")
        _compare_rows(trunc_scores.loc[t], full_scores.loc[t], t, rtol, atol, label)


def assert_constructor_pit(
    constructor: PortfolioConstructor,
    scores: pd.DataFrame,
    data: MarketData,
    dates: Iterable | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Point-in-time check for a PortfolioConstructor (clause 3).

    W_t may use information through close t, so the weights row t built from
    ``scores.loc[:t]`` and ``data.slice_until(t)`` must equal the row t of
    the full-panel weights. Raises LookaheadError on any deviation.
    """
    label = f"constructor '{type(constructor).__name__}'"
    full = constructor.weights(scores, data)
    for t in _resolve_dates(data, dates):
        if t not in scores.index:
            raise DataError(f"{label}: date {t.date()} not in the scores index")
        trunc = constructor.weights(scores.loc[:t], data.slice_until(t))
        if t not in trunc.index:
            raise LookaheadError(f"{label}: no weights row for as-of date {t.date()}")
        _compare_rows(trunc.loc[t], full.loc[t], t, rtol, atol, label)


def assert_cost_pit(
    cost_model: CostModel,
    trades: pd.DataFrame,
    data: MarketData,
    portfolio_value: float,
    dates: Iterable | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Check a CostModel's t-1 market-data cutoff (Timing item 4).

    The cost charged at trade date t is priced before t's close prints, so
    prices, liquidity, volatility, and membership inputs may only use market
    data through t-1. For each sampled date, first remove later rows, then
    independently perturb each available market field at t. Cost at t must
    stay unchanged in both checks. The supplied trades at t remain intact:
    they are known when the cost is priced.

    Perturbations use copied panels and preserve missing optional fields.
    These sampled probes detect specific dependencies; they do not prove
    that an arbitrary custom cost model is free of every possible leak.
    """
    label = f"cost model '{type(cost_model).__name__}'"
    full = cost_model.cost(trades, data, portfolio_value)
    for t in _resolve_dates(data, dates):
        if t not in trades.index:
            raise DataError(f"{label}: date {t.date()} not in the trades index")
        sliced = data.slice_until(t)
        known_trades = trades.loc[:t]
        trunc = cost_model.cost(known_trades, sliced, portfolio_value)
        if t not in trunc.index:
            raise LookaheadError(f"{label}: no cost value for as-of date {t.date()}")
        _compare_scalars(trunc.loc[t], full.loc[t], t, rtol, atol, label)

        # Truncation alone leaves t's data visible. Change fields separately
        # so an invalid ratio of two current inputs cannot hide a shared
        # perturbation. Fill missing cells too, to test missingness dependence.
        for field in ("close", "open", "high", "low", "volume", "unadjusted_close", "universe"):
            panel = getattr(sliced, field)
            if panel is None:
                continue
            changed = copy.deepcopy(sliced)
            changed_panel = getattr(changed, field)
            if field == "universe":
                changed_panel.loc[t] = ~panel.loc[t]
            else:
                changed_panel.loc[t] = (panel.loc[t] * 1.5 + 1.0).fillna(1.0)
            perturbed = cost_model.cost(known_trades, changed, portfolio_value)
            if t not in perturbed.index:
                raise LookaheadError(
                    f"{label}: no cost value for {t.date()} after same-day {field} perturbation"
                )
            _compare_scalars(
                perturbed.loc[t], trunc.loc[t], t, rtol, atol, label,
                check=f"same-day {field} perturbation",
            )


# --------------------------------------------------------------------------
# panel-level bias checks
# --------------------------------------------------------------------------

def assert_no_position_outside_universe(
    result: BacktestResult, data: MarketData, tol: float = 1e-12
) -> None:
    """Assert holdings and target_weights are 0 outside the effective universe.

    Target weights get no allowance: the constructor decides W_t at close t
    with full knowledge of membership at t, so any |W_{t,i}| > tol where
    ``effective_universe`` is False is a violation.

    Holdings get exactly the execution-lag carryover the engine's timing law
    implies: H_t = W_{t-lag} (engine.py, clause 4), so a name whose
    membership ended at date e legally carries holdings on the first ``lag``
    days at/after e — those come from weights decided at t-lag <= e-1, while
    it was still a member. Mechanically, H_{t,i} != 0 is allowed iff the
    effective universe held at (t-lag, i). ``lag`` is read from
    ``result.meta['execution_lag']`` (set by BacktestEngine.run).
    """
    lag = result.meta.get("execution_lag")
    if lag is None:
        raise DataError(
            "result.meta has no 'execution_lag'; cannot derive the holdings "
            "carryover allowance (run through BacktestEngine, or set it)"
        )
    lag = int(lag)
    eff = data.effective_universe()

    def _check(panel: pd.DataFrame, allowed: pd.DataFrame, what: str) -> None:
        aligned = panel.reindex(index=eff.index, columns=eff.columns)
        # ~(|x| <= tol) instead of |x| > tol so NaN cells count as violations
        bad = ~(aligned.abs() <= tol) & ~allowed
        if bad.any().any():
            first = bad.index[bad.any(axis=1)][0]
            tickers = list(bad.columns[bad.loc[first]])
            worst = aligned.where(bad).abs().max().max()
            raise LookaheadError(
                f"{what} outside the effective universe: first violation at "
                f"{first.date()} for {tickers} "
                f"({int(bad.to_numpy().sum())} cells, max |value|={worst:.3e})"
            )

    _check(result.target_weights, eff, "target_weights")
    # eff shifted down by lag, pre-history False — via numpy so the bool
    # dtype survives (bool.shift + fillna round-trips through object dtype)
    if lag < 1:
        raise DataError(f"execution_lag must be >= 1, got {lag}")
    shifted = np.zeros(eff.shape, dtype=bool)
    if lag < len(eff):
        shifted[lag:] = eff.to_numpy()[:-lag]
    holdings_allowed = pd.DataFrame(shifted, index=eff.index, columns=eff.columns)
    _check(result.holdings, holdings_allowed, f"holdings (lag={lag} carryover allowed)")


def assert_purge_gap(
    windows: Sequence[WalkForwardWindow],
    dates: pd.DatetimeIndex,
    purge_days: int,
    embargo_days: int,
) -> None:
    """Assert every walk-forward window honors the purge+embargo gap.

    WalkForwardSplitter semantics (backtest/walkforward.py): for a test block
    starting at position p, ``train_end_pos = p - gap - 1`` with
    ``gap = purge_days + embargo_days`` — the positional distance from
    train_end to test_start is exactly ``gap + 1``, and the ``gap`` positions
    in between belong to neither side. This asserts the CONVENTIONS.md
    guarantee ("train never touches the gap"): train is the contiguous
    [train_start, train_end] block, so the positional equality plus
    train_start <= train_end < test_start <= test_end pins train strictly
    below the gap.
    """
    if not windows:
        raise DataError("no walk-forward windows to check")
    idx = pd.DatetimeIndex(dates)
    gap = int(purge_days) + int(embargo_days)
    for w in windows:
        try:
            tr_s = idx.get_loc(pd.Timestamp(w.train_start))
            tr_e = idx.get_loc(pd.Timestamp(w.train_end))
            te_s = idx.get_loc(pd.Timestamp(w.test_start))
            te_e = idx.get_loc(pd.Timestamp(w.test_end))
        except KeyError as exc:
            raise DataError(f"window boundary {exc} is not a date of the index") from exc
        if tr_s > tr_e or te_s > te_e:
            raise DataError(f"malformed walk-forward window: {w}")
        if te_s - tr_e != gap + 1:
            raise LookaheadError(
                f"purge gap violated: train_end {w.train_end.date()} to "
                f"test_start {w.test_start.date()} leaves {te_s - tr_e - 1} "
                f"excluded trading days, expected purge+embargo={gap}"
            )
        if tr_e >= te_s:
            raise LookaheadError(
                f"train window ends {w.train_end.date()} at/after its test "
                f"window starts {w.test_start.date()}"
            )


# --------------------------------------------------------------------------
# the planted leak
# --------------------------------------------------------------------------

class PlantedLeakFeature(Feature):
    """DELIBERATE LOOKAHEAD LEAK — TEST FIXTURE ONLY. NEVER REGISTER OR TRADE.

    ``compute`` returns tomorrow's return at today's date
    (``data.returns().shift(-1)``), the canonical forbidden pattern of
    CONVENTIONS.md. It exists so tests/test_leakage.py can prove the detector
    detects: :func:`assert_feature_pit` must raise LookaheadError on this
    class. It is intentionally NOT in the FEATURES registry, so it can never
    be reached from a config.
    """

    name = "planted_leak"
    lookback = 1

    def compute(self, data: MarketData) -> pd.DataFrame:
        # shift(-1): the one negative shift in the codebase, planted on purpose.
        return data.returns().shift(-1)
