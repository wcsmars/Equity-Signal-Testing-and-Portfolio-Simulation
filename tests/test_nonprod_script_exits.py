"""How the selection sweep scripts stop when they cannot or must not run.

Covers the two sweeps in scripts/ and the two in research/ (the strategy
modules have the same checks in test_strategy_rules_nonprod.py). Each script
is executed as __main__ with the data directory pointed at an empty folder or
the loader replaced, and with record writes refused: nothing here can run a
backtest or touch results/.
"""
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("scripts/research_pairs_statarb.py", "scripts/xsec_stock_mom_sweep.py",
           "research/mean_reversion_sweep.py", "research/rank_hysteresis_sweep.py")


def _refuse(*args, **kwargs):
    raise AssertionError("a backtest was started or a result file was written")


def _run_as_main(monkeypatch, relative, argv):
    import qcore.records
    monkeypatch.setattr(qcore.records, "save_record", _refuse)
    monkeypatch.setattr(sys, "path", list(sys.path))
    script = ROOT / relative
    monkeypatch.setattr(sys, "argv", [str(script)] + argv)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(script), run_name="__main__")
    return stop.value.code


@pytest.mark.parametrize("relative", SCRIPTS)
def test_sweep_scripts_without_a_data_cache_stop_with_one_line(monkeypatch, tmp_path, relative):
    """The loader names the missing file and the download command; the script
    passes that line on as its exit message instead of a traceback."""
    import qcore.data
    monkeypatch.setattr(qcore.data, "DATA_DIR", tmp_path)  # an empty directory
    message = _run_as_main(monkeypatch, relative, [])
    assert isinstance(message, str) and len(message.splitlines()) == 1
    assert message.startswith(str(tmp_path / "adj_close.csv")) and "download_data.py" in message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("relative", SCRIPTS)
def test_sweep_scripts_refuse_a_shortened_rebase_flag(monkeypatch, capsys, relative):
    """--rebase is acted on by exact name (qcore.records reads sys.argv), so a
    prefix the parser would otherwise expand must be an error, not a run that
    looks like a re-base and is not one."""
    import qcore.data
    monkeypatch.setattr(qcore.data, "load_prices", _refuse)
    assert _run_as_main(monkeypatch, relative, ["--reb"]) == 2
    assert "unrecognized arguments: --reb" in capsys.readouterr().err
