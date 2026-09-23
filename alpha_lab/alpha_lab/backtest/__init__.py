"""Walk-forward backtest engine, splitter, and transaction cost models.

The timing law lives in CONVENTIONS.md: H_t = W_{t-lag}, lag >= 1 enforced.
"""

from alpha_lab.backtest.costs import COST_MODELS, FixedBps, RealisticCost, Zero, from_config
from alpha_lab.backtest.engine import BacktestEngine
from alpha_lab.backtest.walkforward import WalkForwardSplitter

__all__ = [
    "BacktestEngine",
    "WalkForwardSplitter",
    "COST_MODELS",
    "FixedBps",
    "RealisticCost",
    "Zero",
    "from_config",
]
