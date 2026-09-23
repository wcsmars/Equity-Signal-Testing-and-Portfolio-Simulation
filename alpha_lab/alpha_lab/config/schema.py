"""Typed config schema. Strict: unknown keys are errors, not silent defaults.

Every dataclass field has a default, so partial YAML files work; validation
of ranges happens in __post_init__ so a bad config fails at load time, not
mid-backtest.
"""

import dataclasses
import datetime
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from alpha_lab.core.errors import ConfigError


# --------------------------------------------------------------------------
# strict dict -> dataclass builder
# --------------------------------------------------------------------------

def build_dataclass(cls: type, raw: Any, path: str = "config"):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    fmap = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fmap)
    if unknown:
        raise ConfigError(
            f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(fmap)}"
        )
    kwargs = {name: _coerce(hints[name], raw[name], f"{path}.{name}") for name in fmap if name in raw}
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except Exception as exc:  # __post_init__ range checks etc.
        raise ConfigError(f"{path}: {exc}") from exc


def _coerce(hint: Any, val: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        if val is None:
            if len(non_none) == len(typing.get_args(hint)):
                raise ConfigError(f"{path}: null not allowed")
            return None
        return _coerce(non_none[0], val, path)
    if dataclasses.is_dataclass(hint):
        return build_dataclass(hint, val, path)
    if origin is list:
        if not isinstance(val, list):
            raise ConfigError(f"{path}: expected a list, got {type(val).__name__}")
        args = typing.get_args(hint) or (Any,)
        return [_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(val)]
    if hint is dict or origin is dict:
        if not isinstance(val, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(val).__name__}")
        return val
    if hint is bool:
        if not isinstance(val, bool):
            raise ConfigError(f"{path}: expected bool, got {val!r}")
        return val
    if hint is int:
        if isinstance(val, bool) or not isinstance(val, int):
            raise ConfigError(f"{path}: expected int, got {val!r}")
        return val
    if hint is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ConfigError(f"{path}: expected number, got {val!r}")
        return float(val)
    if hint is str:
        # YAML parses unquoted dates (`start: 2018-01-01`) to datetime.date;
        # accept the natural spelling instead of forcing operators to quote.
        if isinstance(val, (datetime.date, datetime.datetime)):
            return val.isoformat()
        if not isinstance(val, str):
            raise ConfigError(f"{path}: expected string, got {val!r}")
        return val
    return val


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    n_assets: int = 20
    n_days: int = 1512
    seed: int = 7
    start: str = "2015-01-02"
    drift_dispersion: float = 0.10   # annualized dispersion of persistent drifts
    base_vol: float = 0.20           # annualized idio vol level
    split_asset: bool = True         # simulate a 4:1 split on the first ticker
    universe_churn: bool = True      # one late entrant, one delisting

    def __post_init__(self) -> None:
        if self.n_assets < 2 or self.n_days < 10:
            raise ValueError("synthetic panel too small")


@dataclass
class DataConfig:
    source: str = "synthetic"        # "synthetic" | "csv"
    path: Optional[str] = None       # csv: directory with close.csv etc., or long file
    format: str = "wide"             # csv layout: "wide" | "long"
    tickers: Optional[List[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    def __post_init__(self) -> None:
        if self.source not in ("synthetic", "csv"):
            raise ValueError(f"unknown data source '{self.source}'")
        if self.source == "csv" and not self.path:
            raise ValueError("data.path required for csv source")
        if self.format not in ("wide", "long"):
            raise ValueError(f"unknown csv format '{self.format}'")
        if self.source == "synthetic":
            # These subset keys are consumed only by the csv source; silently
            # ignoring them would hand back the full synthetic panel while
            # the user believes they subset it.
            set_keys = [
                name for name in ("tickers", "start", "end")
                if getattr(self, name) is not None
            ]
            if set_keys:
                raise ValueError(
                    f"data.{'/'.join(set_keys)} not supported for the synthetic "
                    "source; configure data.synthetic (n_assets, n_days, start) instead"
                )


@dataclass
class FeatureItem:
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature item needs a name")


@dataclass
class SignalConfig:
    name: str = "xs_momentum"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    method: str = "quantile_long_short"
    quantile: float = 0.2            # fraction in each of long / short buckets
    weighting: str = "equal"         # "equal" | "score"
    dollar_neutral: bool = True
    gross_leverage: float = 2.0      # sum of |weights|
    max_weight: float = 0.10         # per-name cap on |weight|
    vol_target: Optional[float] = None  # annualized; None = off
    vol_lookback: int = 63
    min_names: int = 4               # dates with fewer valid names get zero weights

    def __post_init__(self) -> None:
        if not 0 < self.quantile <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        if self.weighting not in ("equal", "score"):
            raise ValueError(f"unknown weighting '{self.weighting}'")
        if self.gross_leverage <= 0 or self.max_weight <= 0:
            raise ValueError("gross_leverage and max_weight must be positive")
        if self.vol_lookback < 5:
            raise ValueError("vol_lookback too short")
        if self.min_names < 0:
            raise ValueError("min_names must be >= 0")


@dataclass
class CostConfig:
    model: str = "realistic"         # "realistic" | "fixed_bps" | "zero"
    commission_per_share: float = 0.005
    half_spread_bps: float = 2.5
    impact_coeff: float = 0.1        # cost += coeff * daily_vol * sqrt(participation)
    fixed_bps: float = 5.0           # for model == "fixed_bps"
    adv_window: int = 21
    vol_window: int = 63
    portfolio_value: float = 1_000_000.0

    def __post_init__(self) -> None:
        for name in ("commission_per_share", "half_spread_bps", "impact_coeff", "fixed_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")


@dataclass
class WalkForwardConfig:
    scheme: str = "rolling"          # "rolling" | "expanding"
    train_days: int = 504
    test_days: int = 126
    purge_days: int = 5
    embargo_days: int = 0

    def __post_init__(self) -> None:
        if self.scheme not in ("rolling", "expanding"):
            raise ValueError(f"unknown walk-forward scheme '{self.scheme}'")
        if self.train_days <= 0 or self.test_days <= 0:
            raise ValueError("train_days and test_days must be positive")
        if self.purge_days < 0 or self.embargo_days < 0:
            raise ValueError("purge_days and embargo_days must be >= 0")


@dataclass
class BacktestConfig:
    execution_lag: int = 2           # H_t = W_{t-lag}; next-close execution
    drift_adjust_turnover: bool = True
    walkforward: Optional[WalkForwardConfig] = field(default_factory=WalkForwardConfig)

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError(
                "execution_lag must be >= 1 — lag 0 trades on information "
                "unavailable at execution time (see CONVENTIONS.md)"
            )


@dataclass
class ExperimentConfig:
    name: str = "run"
    runs_dir: str = "runs"
    # persist computed feature panels here (parquet/csv) and reuse them across
    # runs; None = compute in-process only. feature_cache/ is gitignored.
    feature_cache_dir: Optional[str] = None


@dataclass
class ReportConfig:
    formats: List[str] = field(default_factory=lambda: ["html", "md"])

    def __post_init__(self) -> None:
        bad = set(self.formats) - {"html", "md"}
        if bad:
            raise ValueError(f"unknown report formats {sorted(bad)}")
        if not self.formats:
            # An empty list would run the full backtest and then die inside
            # report generation AFTER the run is persisted — reject at load.
            raise ValueError(
                "report.formats must not be empty; use --no-report to skip "
                "report generation"
            )


@dataclass
class AlphaLabConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: List[FeatureItem] = field(default_factory=list)
    signal: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
