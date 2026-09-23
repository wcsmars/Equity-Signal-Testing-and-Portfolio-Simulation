"""Signal library and registry (SIGNALS). Signals are pure per-date maps of
point-in-time features; fit() sees train dates only. See CONVENTIONS.md."""

# registry must be imported first: it pulls in momentum then reversal in
# dependency order (reversal borrows helpers from momentum at import time).
# This __init__ runs before any submodule import, so the order here governs
# every entry point into the package.
from alpha_lab.signals.registry import SIGNALS, build_signal
from alpha_lab.signals.momentum import CrossSectionalMomentum, cross_sectional_zscore
from alpha_lab.signals.reversal import ShortTermReversal

__all__ = [
    "SIGNALS",
    "CrossSectionalMomentum",
    "ShortTermReversal",
    "build_signal",
    "cross_sectional_zscore",
]
