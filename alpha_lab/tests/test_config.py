"""Config loader/schema regressions: operator errors must die as ConfigError
(one clean line from the runner), never as raw tracebacks or silent no-ops."""

import datetime

import pytest

from alpha_lab.config.loader import config_from_dict, config_hash, load_config
from alpha_lab.config.schema import DataConfig, ReportConfig
from alpha_lab.core.errors import ConfigError


# -- loader ------------------------------------------------------------------


def test_invalid_yaml_is_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


def test_duplicate_yaml_key_is_config_error(tmp_path):
    # a repeated section must not silently reset the first one to defaults
    cfg = tmp_path / "dup.yaml"
    cfg.write_text("costs:\n  model: fixed_bps\nbacktest:\n  execution_lag: 2\ncosts:\n  model: zero\n")
    with pytest.raises(ConfigError, match="duplicate key 'costs'"):
        load_config(cfg)


def test_yaml_merge_keys_still_load(tmp_path):
    # anchors + '<<' merges are standard YAML; explicit keys override merged
    # ones, and duplicate explicit keys next to a merge are still rejected
    cfg = tmp_path / "merge.yaml"
    cfg.write_text(
        "_shared: &cost\n  model: fixed_bps\n  fixed_bps: 5.0\n"
        "costs:\n  <<: *cost\n  fixed_bps: 9.0\n"
    )
    import yaml
    from alpha_lab.config.loader import _StrictLoader
    raw = yaml.load(cfg.read_text(), Loader=_StrictLoader)
    assert raw["costs"] == {"model": "fixed_bps", "fixed_bps": 9.0}
    # the schema is strict, so the anchor has to live inside a real section:
    # a top-level holder such as `_shared` is an unknown key
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(cfg)
    inside = tmp_path / "merge_inside.yaml"
    inside.write_text(
        "signal:\n  name: xs_momentum\n  params: &mom {window: 252, skip: 21}\n"
        "features:\n  - name: momentum\n    params:\n      <<: *mom\n      skip: 5\n"
    )
    loaded = load_config(inside)
    assert loaded.signal.params == {"window": 252, "skip": 21}
    assert loaded.features[0].params == {"window": 252, "skip": 5}
    dup = tmp_path / "dup_merge.yaml"
    dup.write_text("_s: &s\n  a: 1\nc:\n  <<: *s\n  b: 2\n  b: 3\n")
    with pytest.raises(ConfigError, match="duplicate key 'b'"):
        load_config(dup)


def test_override_through_one_alias_leaves_the_aliased_sibling_alone(tmp_path):
    # the YAML parser returns one dict for an anchor and its aliases; an
    # in-place override through signal.params must not rewrite the feature
    cfg_file = tmp_path / "alias.yaml"
    cfg_file.write_text(
        "signal:\n  name: xs_momentum\n  params: &p {window: 252, skip: 21}\n"
        "features:\n  - name: momentum\n    params: *p\n"
    )
    cfg = load_config(cfg_file, {"signal.params.window": 126})
    assert cfg.signal.params == {"window": 126, "skip": 21}
    assert cfg.features[0].params == {"window": 252, "skip": 21}
    plain = load_config(cfg_file)
    assert plain.signal.params == plain.features[0].params
    assert plain.signal.params is not plain.features[0].params


def test_recursive_yaml_alias_is_config_error(tmp_path):
    cfg_file = tmp_path / "loop.yaml"
    cfg_file.write_text("features: &loop\n  - *loop\n")
    with pytest.raises(ConfigError, match="recursive YAML alias"):
        load_config(cfg_file)


@pytest.mark.parametrize("document", ["[]", "false", "0", "abc", "- a\n- b", "''"])
def test_non_mapping_document_is_config_error(tmp_path, document):
    # a falsy document ([], false, 0) used to be read as an empty config and
    # ran the all-defaults experiment
    cfg_file = tmp_path / "scalar.yaml"
    cfg_file.write_text(document + "\n")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(cfg_file)


@pytest.mark.parametrize("document", ["", "# only a comment\n", "null\n", "~\n", "{}\n"])
def test_empty_document_loads_defaults(tmp_path, document):
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text(document)
    assert load_config(cfg_file).to_dict() == {
        **config_from_dict({}).to_dict(),
        "experiment": {
            "name": "run", "runs_dir": str(tmp_path.resolve() / "runs"), "feature_cache_dir": None,
        },
    }


def test_undecodable_config_file_is_config_error(tmp_path):
    cfg_file = tmp_path / "binary.yaml"
    cfg_file.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(cfg_file)


def test_override_through_a_non_mapping_is_config_error(tmp_path):
    cfg_file = tmp_path / "run.yaml"
    cfg_file.write_text("backtest:\n  execution_lag: 2\n")
    with pytest.raises(ConfigError, match="'execution_lag' is not a mapping"):
        load_config(cfg_file, {"backtest.execution_lag.value": 3})
    # a missing intermediate section is created
    assert load_config(cfg_file, {"costs.model": "zero"}).costs.model == "zero"


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_non_finite_number_is_config_error(tmp_path, value):
    cfg = tmp_path / "nan.yaml"
    cfg.write_text(f"costs:\n  half_spread_bps: {value}\n")
    with pytest.raises(ConfigError, match="expected a finite number"):
        load_config(cfg)


def test_config_hash_ignores_output_locations(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "c.yaml").write_text("backtest:\n  execution_lag: 2\n")
    a, b = load_config(tmp_path / "a" / "c.yaml"), load_config(tmp_path / "b" / "c.yaml")
    assert a.experiment.runs_dir != b.experiment.runs_dir
    assert config_hash(a) == config_hash(b)
    changed = load_config(tmp_path / "a" / "c.yaml", {"backtest.execution_lag": 3})
    assert config_hash(changed) != config_hash(a)
    cached = load_config(tmp_path / "a" / "c.yaml", {"experiment.feature_cache_dir": "cache"})
    assert cached.experiment.feature_cache_dir != a.experiment.feature_cache_dir
    assert config_hash(cached) == config_hash(a)


def test_config_hash_follows_the_input_data_path(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("data:\n  source: csv\n  path: panel_a\n")
    first = load_config(cfg_file)
    assert config_hash(first) == config_hash(load_config(cfg_file))
    assert config_hash(load_config(cfg_file, {"data.path": "panel_b"})) != config_hash(first)


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


@pytest.mark.parametrize(
    "section, field",
    [("data", "start"), ("data", "end")],
)
@pytest.mark.parametrize(
    "bad",
    ["notadate", "10/01/2024", "2018-13-45", "2018", "today", "", "2018-01-01T00:00:00+00:00"],
)
def test_bad_csv_date_bounds_fail_at_load(section, field, bad):
    # these strings reach pandas mid-run; an invalid one used to leave the
    # runner as a raw DateParseError and a day-first one was read month-first
    with pytest.raises(ConfigError, match=f"data.{field} must be"):
        config_from_dict({section: {"source": "csv", "path": "x", field: bad}})


@pytest.mark.parametrize("bad", ["notadate", "02/01/2015", "2015-02-30", ""])
def test_bad_synthetic_start_fails_at_load(bad):
    with pytest.raises(ConfigError, match="data.synthetic.start must be"):
        config_from_dict({"data": {"synthetic": {"start": bad}}})
    from alpha_lab.config.schema import SyntheticConfig

    with pytest.raises(ConfigError, match="data.synthetic.start must be"):
        SyntheticConfig(start=bad)


@pytest.mark.parametrize(
    "good",
    ["2018-01-02", "2018-01-02 00:00:00", "2018-01-02T16:00", "20180102",
     datetime.date(2018, 1, 2), datetime.datetime(2018, 1, 2, 16, 0)],
)
def test_iso_date_spellings_accepted(good):
    cfg = config_from_dict({"data": {"source": "csv", "path": "x", "start": good, "end": good}})
    assert cfg.data.start == cfg.data.end
    assert config_from_dict({"data": {"synthetic": {"start": good}}}).data.synthetic.start


def test_date_objects_must_be_timezone_naive_and_real():
    import pandas as pd

    aware = datetime.datetime(2018, 1, 2, tzinfo=datetime.timezone.utc)
    for bad in (aware, pd.NaT, 20180102, 1.5):
        with pytest.raises(ConfigError, match="data.start must be"):
            DataConfig(source="csv", path="x", start=bad)
    # YAML parses a `Z` timestamp to an aware datetime
    with pytest.raises(ConfigError, match="data.end must be"):
        config_from_dict({"data": {"source": "csv", "path": "x", "end": aware}})


def test_start_after_end_fails_at_load():
    with pytest.raises(ConfigError, match="data.start 2020-01-01 is after data.end 2019-01-01"):
        config_from_dict(
            {"data": {"source": "csv", "path": "x", "start": "2020-01-01", "end": "2019-01-01"}}
        )
    same_day = config_from_dict(
        {"data": {"source": "csv", "path": "x", "start": "2020-01-01", "end": "2020-01-01"}}
    )
    assert same_day.data.start == same_day.data.end


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


@pytest.mark.parametrize(
    "extra, named",
    [
        ({"path": "somewhere"}, "data.path"),
        ({"path": ""}, "data.path"),
        ({"format": "long"}, "data.format"),
        ({"path": "somewhere", "format": "long"}, "data.path/format"),
    ],
)
def test_synthetic_rejects_csv_only_location_keys(extra, named):
    """Regression: source defaults to synthetic, so a config that set
    data.path without `source: csv` ran on the synthetic panel."""
    with pytest.raises(ValueError, match=f"{named} not supported.*source: csv"):
        DataConfig(**extra)
    with pytest.raises(ConfigError, match="source: csv"):
        config_from_dict({"data": {"source": "synthetic", **extra}})


def test_path_without_csv_source_fails_at_load(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("data:\n  path: ../wide\n")
    with pytest.raises(ConfigError, match="data.path not supported"):
        load_config(cfg_file)
    cfg_file.write_text("data:\n  source: synthetic\n")
    with pytest.raises(ConfigError, match="data.path not supported"):
        load_config(cfg_file, overrides={"data.path": "../wide"})
    # switching the same file to csv from the command line still works
    switched = load_config(cfg_file, overrides={"data.source": "csv", "data.path": "../wide"})
    assert switched.data.source == "csv"
    assert switched.data.path == str((tmp_path / ".." / "wide").resolve())


def test_synthetic_accepts_explicit_csv_defaults():
    # a saved config.yaml spells out `path: null` and `format: wide`
    cfg = config_from_dict({"data": {"source": "synthetic", "path": None, "format": "wide"}})
    assert cfg.data.path is None and cfg.data.format == "wide"
    assert config_from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_csv_source_keeps_subset_keys():
    cfg = DataConfig(source="csv", path="x", start="2018-01-01", tickers=["A"])
    assert cfg.start == "2018-01-01"
    assert cfg.tickers == ["A"]


def test_csv_source_requires_a_path():
    with pytest.raises(ConfigError, match="data.path required for csv source"):
        config_from_dict({"data": {"source": "csv"}})
    with pytest.raises(ConfigError, match="data.path required for csv source"):
        config_from_dict({"data": {"source": "csv", "path": ""}})
    with pytest.raises(ConfigError, match="unknown data source"):
        config_from_dict({"data": {"source": "parquet"}})
    with pytest.raises(ConfigError, match="unknown csv format"):
        config_from_dict({"data": {"source": "csv", "path": "x", "format": "tall"}})


# -- strict schema ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, where",
    [
        ({"backtset": {}}, r"config: unknown keys \['backtset'\]"),
        ({"backtest": {"lag": 2}}, r"config.backtest: unknown keys \['lag'\]"),
        ({"backtest": {"walkforward": {"train": 1}}}, r"config.backtest.walkforward: unknown keys"),
        ({"data": {"synthetic": {"n_asset": 5}}}, r"config.data.synthetic: unknown keys"),
        ({"features": [{"name": "momentum", "param": {}}]}, r"config.features\[0\]: unknown keys"),
        # keys of mixed types do not sort; the message must still be produced
        ({1: "a", "foo": "b"}, r"config: unknown keys \[1, 'foo'\]"),
    ],
)
def test_unknown_keys_are_config_errors(raw, where):
    with pytest.raises(ConfigError, match=where):
        config_from_dict(raw)


def test_mixed_type_unknown_keys_fail_cleanly_from_yaml(tmp_path):
    cfg_file = tmp_path / "mixed.yaml"
    cfg_file.write_text("1: a\nfoo: b\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(cfg_file)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"backtest": {"execution_lag": "2"}}, "expected int"),
        ({"backtest": {"execution_lag": 2.0}}, "expected int"),
        ({"backtest": {"execution_lag": True}}, "expected int"),
        ({"backtest": {"execution_lag": None}}, "expected int"),
        ({"backtest": {"drift_adjust_turnover": 1}}, "expected bool"),
        ({"backtest": {"drift_adjust_turnover": "true"}}, "expected bool"),
        ({"portfolio": {"quantile": "0.2"}}, "expected number"),
        ({"portfolio": {"quantile": True}}, "expected number"),
        ({"report": {"formats": "html"}}, "expected a list"),
        ({"features": {"name": "momentum"}}, "expected a list"),
        ({"signal": {"params": [1, 2]}}, "expected a mapping"),
        ({"signal": {"name": 5}}, "expected string"),
        ({"signal": "xs_momentum"}, "expected a mapping"),
        ({"data": {"tickers": ["A", 5]}}, r"tickers\[1\]: expected string"),
    ],
)
def test_wrongly_typed_values_are_config_errors(raw, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


@pytest.mark.parametrize("lag", [0, -1])
def test_schema_rejects_execution_lag_below_one(lag):
    # CONVENTIONS.md: lag 0 trades on information unavailable at execution
    with pytest.raises(ConfigError, match="execution_lag must be an integer >= 1"):
        config_from_dict({"backtest": {"execution_lag": lag}})
    from alpha_lab.config.schema import BacktestConfig

    with pytest.raises(ConfigError, match="execution_lag"):
        BacktestConfig(execution_lag=lag)
    assert config_from_dict({"backtest": {"execution_lag": 1}}).backtest.execution_lag == 1


@pytest.mark.parametrize("quantile", [0, 0.0, -0.1, 0.5000001, 0.6, 1.0])
def test_quantile_outside_unit_half_interval_rejected(quantile):
    with pytest.raises(ConfigError, match="quantile must be in"):
        config_from_dict({"portfolio": {"quantile": quantile}})


@pytest.mark.parametrize("quantile", [0.01, 0.5])
def test_quantile_bounds_accepted(quantile):
    assert config_from_dict({"portfolio": {"quantile": quantile}}).portfolio.quantile == quantile


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
    # must reach the requested per-side gross (see configs/base.yaml comment).
    # Universe churn (one late entrant, one delisting) leaves n_assets - 1
    # names on part of the sample, and that smaller count sets the bound.
    synthetic = cfg.data.synthetic
    assert synthetic.universe_churn and synthetic.n_assets >= 4
    n_side = int((synthetic.n_assets - 1) * cfg.portfolio.quantile)
    assert n_side == 3
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
