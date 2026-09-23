"""alpha_lab: research platform for daily equity alpha signals.

Read CONVENTIONS.md before writing code against this package — it defines the
timing law (what information may be used when) that everything here obeys.
"""

__version__ = "0.1.0"

from alpha_lab.core.types import MarketData, TRADING_DAYS_PER_YEAR
from alpha_lab.core.results import BacktestResult, WalkForwardWindow
from alpha_lab.core.errors import (
    AlphaLabError,
    ConfigError,
    DataError,
    LookaheadError,
)

__all__ = [
    "MarketData",
    "BacktestResult",
    "WalkForwardWindow",
    "TRADING_DAYS_PER_YEAR",
    "AlphaLabError",
    "ConfigError",
    "DataError",
    "LookaheadError",
    "__version__",
]
