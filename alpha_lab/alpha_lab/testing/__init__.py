"""Importable leakage/bias check harness. Run these on every new feature and
signal BEFORE trusting a backtest of it. Used by tests/test_leakage.py."""

from alpha_lab.testing.checks import (
    PlantedLeakFeature,
    assert_constructor_pit,
    assert_cost_pit,
    assert_feature_pit,
    assert_no_position_outside_universe,
    assert_purge_gap,
    assert_signal_pit,
    assert_truncation_invariant,
)

__all__ = [
    "PlantedLeakFeature",
    "assert_constructor_pit",
    "assert_cost_pit",
    "assert_feature_pit",
    "assert_no_position_outside_universe",
    "assert_purge_gap",
    "assert_signal_pit",
    "assert_truncation_invariant",
]
