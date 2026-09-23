"""Signal registry and config-driven construction.

``SIGNALS`` maps registered names ('xs_momentum', 'xs_reversal', ...) to
Signal classes; ``build_signal`` instantiates one from a ``SignalConfig``.
Registered names and parameter names are a public contract: configs across
the project resolve signals via ``SIGNALS.create(name, **params)``.
"""

from __future__ import annotations

from alpha_lab.config.schema import SignalConfig
from alpha_lab.core.interfaces import Signal
from alpha_lab.core.registry import Registry

#: the signal registry — configs resolve signal names against this
SIGNALS = Registry("signal")


def build_signal(cfg: SignalConfig) -> Signal:
    """Instantiate the signal named by ``cfg.name`` with ``cfg.params``.

    Raises ConfigError for an unknown signal name or bad parameters.
    """
    return SIGNALS.create(cfg.name, **cfg.params)


# Imported at the bottom (not the top) for their registration side effects:
# each library module's @SIGNALS.register decorators run at import time, so
# importing this registry module alone populates the signal library. The
# imports must come after SIGNALS is defined because both modules import
# SIGNALS from here (a deliberate, benign import cycle).
import alpha_lab.signals.momentum  # noqa: E402,F401
import alpha_lab.signals.reversal  # noqa: E402,F401
