"""Exception hierarchy. Raise these, not bare ValueError, at module boundaries."""


class AlphaLabError(Exception):
    """Base class for all alpha_lab errors."""


class DataError(AlphaLabError):
    """Malformed or misaligned market data."""


class ConfigError(AlphaLabError):
    """Invalid, unknown, or missing configuration."""


class LookaheadError(AlphaLabError):
    """A component attempted to use information from the future."""


class ExperimentError(AlphaLabError):
    """Experiment tracking failure (missing run, corrupt registry, ...)."""
