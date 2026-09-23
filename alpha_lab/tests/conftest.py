"""Shared fixtures. Treat fixture MarketData as read-only — they are
session-scoped for speed; never mutate them in a test."""

import pytest

from alpha_lab.data.synthetic import make_market


@pytest.fixture(scope="session")
def market():
    """12 assets x 756 days, with a 4:1 split and universe entry/exit."""
    return make_market(n_assets=12, n_days=756, seed=7)


@pytest.fixture(scope="session")
def market_simple():
    """Clean panel — no split, no churn. For exact-math tests."""
    return make_market(n_assets=6, n_days=400, seed=3, split_asset=False, universe_churn=False)


@pytest.fixture(scope="session")
def market_long():
    """Enough history for walk-forward with 504-day train windows."""
    return make_market(n_assets=20, n_days=1512, seed=11)
