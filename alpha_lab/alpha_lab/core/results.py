"""BacktestResult: everything a run produces, saveable/loadable as flat files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha_lab.core.errors import DataError


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WalkForwardWindow":
        return cls(**{k: pd.Timestamp(v) for k, v in d.items()})


@dataclass
class BacktestResult:
    """Per-date series/panels of one backtest, plus provenance.

    Timing (CONVENTIONS.md): holdings H_t = W_{t-lag}; gross_t = sum(H_t * r_t);
    net_t = gross_t - costs_t; turnover_t = sum(|trades_t|).
    """

    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    holdings: pd.DataFrame
    target_weights: pd.DataFrame
    scores: pd.DataFrame | None = None
    windows: list[WalkForwardWindow] | None = None
    config: dict | None = None
    meta: dict = field(default_factory=dict)

    def equity_curve(self, initial: float = 1.0) -> pd.Series:
        return (1.0 + self.net_returns.fillna(0.0)).cumprod() * initial

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gross": self.gross_returns,
                "costs": self.costs,
                "net": self.net_returns,
                "turnover": self.turnover,
            }
        )

    # -- persistence (CSV + JSON: diffable, no extra deps) ---------------

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.summary_frame().to_csv(out / "series.csv", index_label="date")
        self.holdings.to_csv(out / "holdings.csv", index_label="date")
        self.target_weights.to_csv(out / "target_weights.csv", index_label="date")
        if self.scores is not None:
            self.scores.to_csv(out / "scores.csv", index_label="date")
        meta = {
            "windows": [w.to_dict() for w in self.windows] if self.windows else None,
            "config": self.config,
            "meta": self.meta,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "BacktestResult":
        src = Path(path)
        if not (src / "series.csv").exists():
            raise DataError(f"no backtest result at {src}")

        def read_panel(name: str) -> pd.DataFrame | None:
            f = src / name
            if not f.exists():
                return None
            return pd.read_csv(f, index_col="date", parse_dates=["date"])

        series = read_panel("series.csv")
        meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
        windows = meta.get("windows")
        return cls(
            gross_returns=series["gross"],
            costs=series["costs"],
            net_returns=series["net"],
            turnover=series["turnover"],
            holdings=read_panel("holdings.csv"),
            target_weights=read_panel("target_weights.csv"),
            scores=read_panel("scores.csv"),
            windows=[WalkForwardWindow.from_dict(w) for w in windows] if windows else None,
            config=meta.get("config"),
            meta=meta.get("meta") or {},
        )
