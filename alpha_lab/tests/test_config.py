"""Config loader/schema regressions: operator errors must die as ConfigError
(one clean line from the runner), never as raw tracebacks or silent no-ops."""

import datetime

import pytest

from alpha_lab.config.loader import config_from_dict, load_config
from alpha_lab.config.schema import DataConfig, ReportConfig
from alpha_lab.core.errors import ConfigError


# -- loader ------------------------------------------------------------------


def test_invalid_yaml_is_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


def test_directory_path_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not a file"):
        load_config(tmp_path)


def test_missing_file_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_config_paths_follow_config_location(tmp_path, monkeypatch):
    config_dir = tmp_path / "checkout with spaces" / "configs"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "run.yaml"
    config_file.write_text(
        "data:\n  source: csv\n  path: ../data\n"
        "experiment:\n  runs_dir: ../runs\n  feature_cache_dir: ../cache\n"
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config(config_file)
    assert cfg.data.path == str(config_dir.parent / "data")
    assert cfg.experiment.runs_dir == str(config_dir.parent / "runs")
    assert cfg.experiment.feature_cache_dir == str(config_dir.parent / "cache")
    override = load_config(config_file, {"experiment.runs_dir": "../other runs"})
    assert override.experiment.runs_dir == str(config_dir.parent / "other runs")


def test_absolute_config_path_override_is_preserved(tmp_path):
    config_file = tmp_path / "run.yaml"
    config_file.write_text("{}\n")
    cfg = load_config(config_file, {"experiment.runs_dir": str(tmp_path / "output")})
    assert cfg.experiment.runs_dir == str(tmp_path / "output")


# -- date coercion -------------------------------------------------------------


def test_unquoted_yaml_dates_accepted(tmp_path):
    # `start: 2018-01-01` parses to datetime.date — the natural YAML spelling
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "data:\n  source: csv\n  path: somewhere\n  start: 2018-01-01\n  end: 2019-06-30\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.data.start == "2018-01-01"
    assert cfg.data.end == "2019-06-30"


def test_date_object_override_coerced():
    cfg = config_from_dict(
        {"data": {"source": "csv", "path": "x", "start": datetime.date(2018, 1, 1)}}
    )
    assert cfg.data.start == "2018-01-01"


# -- synthetic source vs csv-only subset keys -----------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"start": "2018-01-01"},
        {"end": "2019-01-01"},
        {"tickers": ["A000"]},
    ],
)
def test_synthetic_rejects_csv_only_subset_keys(extra):
    """Regression: these keys were silently ignored for synthetic data — the
    user believed they subset the panel and got the full panel instead."""
    with pytest.raises((ConfigError, ValueError), match="not supported"):
        DataConfig(source="synthetic", **extra)


def test_csv_source_keeps_subset_keys():
    cfg = DataConfig(source="csv", path="x", start="2018-01-01", tickers=["A"])
    assert cfg.start == "2018-01-01"
    assert cfg.tickers == ["A"]


# -- report formats --------------------------------------------------------------


def test_empty_report_formats_rejected_at_load():
    """Regression: formats=[] used to pass validation, run the full backtest,
    then die inside report generation after the run was persisted."""
    with pytest.raises((ConfigError, ValueError), match="must not be empty"):
        ReportConfig(formats=[])
    with pytest.raises(ConfigError, match="must not be empty"):
        config_from_dict({"report": {"formats": []}})


def test_base_yaml_loads_clean():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "base.yaml")
    assert cfg.backtest.execution_lag == 2
    assert config_from_dict({}).backtest.execution_lag == 2
    # demo config must be self-consistent: per-name cap x names-per-side
    # must reach the requested per-side gross (see configs/base.yaml comment)
    n_side = int(cfg.data.synthetic.n_assets * cfg.portfolio.quantile)
    assert n_side * cfg.portfolio.max_weight >= cfg.portfolio.gross_leverage / 2.0 - 1e-12


# -- config-driven knobs wired through to their consumers ----------------------


def test_min_names_flows_from_config_to_constructor():
    from alpha_lab.config.schema import PortfolioConfig
    from alpha_lab.portfolio import from_config as portfolio_from_config

    ctor = portfolio_from_config(config_from_dict({"portfolio": {"min_names": 6}}).portfolio)
    assert ctor.min_names == 6
    # default still applies when unset
    assert portfolio_from_config(PortfolioConfig()).min_names == 4
    with pytest.raises(ConfigError, match="min_names"):
        config_from_dict({"portfolio": {"min_names": -1}})


def test_feature_cache_dir_persists_and_reuses_panels(tmp_path):
    """experiment.feature_cache_dir wires FeatureStore's disk cache: files
    appear on first compute and a fresh store serves from them."""
    from alpha_lab.core.interfaces import FeatureSpec
    from alpha_lab.data.synthetic import make_market
    from alpha_lab.features import FeatureStore

    cfg = config_from_dict(
        {"experiment": {"feature_cache_dir": str(tmp_path / "fc")}}
    )
    assert cfg.experiment.feature_cache_dir == str(tmp_path / "fc")
    assert config_from_dict({}).experiment.feature_cache_dir is None

    data = make_market(n_assets=6, n_days=120, seed=5, split_asset=False, universe_churn=False)
    spec = FeatureSpec.make("momentum", window=63, skip=5)
    first = FeatureStore(cache_dir=cfg.experiment.feature_cache_dir).get(spec, data)
    cached_files = list((tmp_path / "fc").iterdir())
    assert cached_files, "disk cache directory should contain the persisted panel"
    # a brand-new store (empty memo) must reproduce the panel from disk
    second = FeatureStore(cache_dir=cfg.experiment.feature_cache_dir).get(spec, data)
    import pandas.testing as pdt

    # the disk round trip drops the index freq attribute; values must match
    pdt.assert_frame_equal(first, second, check_freq=False)
