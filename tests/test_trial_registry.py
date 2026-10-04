"""Trial registry: excess returns, each scenario's trial count, the
selection-path pool and the guarded write of the saved record."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASH = 0.0001
GRIDS = {'mean_reversion': [0.2, 0.5, 0.8], 'seasonality_flows': [0.1, 0.4],
         'tsmom_trend': [0.3, 0.6, 0.9, 1.2], 'xsec_etf_mom': [0.0, 0.5]}
KILLED_IDEA = [-0.2, 0.1, 0.3, 0.0]        # plus one failed row with no Sharpe
HYSTERESIS_ETF = [0.55, 0.6, 0.65]         # the later search behind the deployed xsec rule
HYSTERESIS_STOCK = [0.9, 1.0]              # another sleeve's rows in the same file


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """scripts/trial_registry.py pointed at an empty temporary results folder,
    so the saved results/trial_registry.json is never touched."""
    spec = importlib.util.spec_from_file_location('trial_registry_test', ROOT / 'scripts/trial_registry.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / 'results').mkdir()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'cash_daily_return', lambda i: pd.Series(CASH, index=i))
    monkeypatch.setattr(sys, 'argv', ['trial_registry.py'])
    monkeypatch.delenv('QCORE_REBASE', raising=False)
    return mod


def write_results(res):
    """Four sleeve grids, one killed family, one diagnostic placeholder, the
    hysteresis study, a 260-session return panel and its declared metrics."""
    for name, sharpes in GRIDS.items():
        pd.DataFrame({'name': range(len(sharpes)), 'full_sharpe': sharpes}).to_csv(
            res / f'{name}_variants.csv', index=False)
    (res / 'killed_idea_variants.csv').write_text('name,full_sharpe\na,-0.2\nb,0.1\nc,\nd,0.3\ne,0.0\n')
    (res / 'sleeve_monitor_variants.csv').write_text('variant,is_sharpe\nn/a,\n')
    pd.DataFrame({'sleeve': ['etf'] * 3 + ['stock'] * 2, 'name': list('abcde'),
                  'full_sharpe': HYSTERESIS_ETF + HYSTERESIS_STOCK}).to_csv(
        res / 'rank_hysteresis_variants.csv', index=False)
    idx = pd.bdate_range('2017-07-03', periods=260)  # spans the 2018-01-01 split
    rng = np.random.RandomState(5)                   # frozen legacy stream
    frame = pd.DataFrame({k: CASH + rng.normal(0.0005, 0.01, len(idx)) for k in GRIDS}, index=idx)
    frame.iloc[:40, frame.columns.get_loc('xsec_etf_mom')] = CASH  # cash-filled before going live
    frame['ENSEMBLE'] = frame.mean(axis=1)
    frame.to_csv(res / 'sleeve_returns.csv')
    excess = frame['ENSEMBLE'] - CASH
    declared = {'full': {'sharpe': round(float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)), 2)},
                'start': str(idx[0].date()), 'end': str(idx[-1].date())}
    (res / 'ensemble.json').write_text(json.dumps(declared))
    return idx, declared


def test_excess_returns_subtract_cash_and_start_each_sleeve_at_its_live_date(registry, tmp_path, monkeypatch):
    idx = pd.bdate_range('2020-01-01', periods=5)
    cash = 0.001
    monkeypatch.setattr(registry, 'cash_daily_return', lambda i: pd.Series(cash, index=i))
    pd.DataFrame({'late_sleeve': [cash, cash, 0.011, cash, -0.004],
                  'cash_sleeve': cash,
                  'ENSEMBLE': [cash, 0.006, 0.003, cash, 0.0]}, index=idx).to_csv(
        tmp_path / 'results' / 'sleeve_returns.csv')
    out = registry.excess_returns()
    # Cash-filled days before the sleeve went live are dropped; a later flat day is kept.
    assert out['late_sleeve'].index.equals(idx[2:])
    np.testing.assert_allclose(out['late_sleeve'], [0.010, 0.0, -0.005], atol=1e-12)
    # The portfolio keeps its whole history, initial cash included.
    assert out['ENSEMBLE'].index.equals(idx)
    np.testing.assert_allclose(out['ENSEMBLE'], [0.0, 0.005, 0.002, 0.0, -0.001], atol=1e-12)
    assert out['cash_sleeve'].index.equals(idx)
    np.testing.assert_allclose(out['cash_sleeve'], 0.0, atol=1e-12)


def test_saved_returns_are_read_back_digit_for_digit(registry, tmp_path, monkeypatch):
    # pandas' default float parser can be off in the last bit; the saved
    # stream is a record, so it is parsed exactly.
    monkeypatch.setattr(registry, 'cash_daily_return', lambda i: pd.Series(0.0, index=i))
    idx = pd.bdate_range('2012-01-02', periods=3000)
    values = np.random.RandomState(3).normal(0.0004, 0.01, len(idx))  # frozen legacy stream
    pd.DataFrame({'ENSEMBLE': values}, index=idx).to_csv(tmp_path / 'results' / 'sleeve_returns.csv')
    assert (registry.excess_returns()['ENSEMBLE'].to_numpy() == values).all()


def test_report_uses_each_scenarios_own_trial_count(registry, tmp_path, capsys):
    res = tmp_path / 'results'
    idx, declared = write_results(res)
    registry.main()
    printed = capsys.readouterr().out
    out = json.loads((res / 'trial_registry.json').read_text())
    assert out['study'] is True
    assert out['registry'] == {'killed_idea': 5, 'mean_reversion': 3, 'rank_hysteresis': 5,
                               'seasonality_flows': 2, 'tsmom_trend': 4, 'xsec_etf_mom': 2}
    assert (out['n_trials_total'], out['n_families']) == (21, 6)
    assert list(out['excluded_diagnostics']) == ['sleeve_monitor']
    n_oos = int((idx >= '2018-01-01').sum())
    assert 0 < n_oos < len(idx)
    for name, sharpes in GRIDS.items():
        own = out['sleeves'][name]['dsr_own_grid']
        live_days = 220 if name == 'xsec_etf_mom' else 260
        assert (own['n_trials'], own['T']) == (len(sharpes), live_days)
        assert own['trial_sd_ann'] == pytest.approx(np.std(sharpes, ddof=1), abs=1e-3)
        assert out['sleeves'][name]['oos_psr']['T'] == n_oos
        assert out['sleeves'][name]['oos_psr']['benchmark_ann'] == 0.0
    ens = out['ensemble']
    lenient, brutal = ens['dsr']['families_only'], ens['dsr']['every_logged_trial']
    assert (lenient['n_trials'], brutal['n_trials']) == (6, 21)
    assert lenient['sharpe_ann'] == brutal['sharpe_ann'] == pytest.approx(declared['full']['sharpe'], abs=0.011)
    # Both scenarios pool every finite logged Sharpe; only the trial count differs.
    pooled = [s for sharpes in GRIDS.values() for s in sharpes] + KILLED_IDEA + HYSTERESIS_ETF + HYSTERESIS_STOCK
    assert lenient['trial_sd_ann'] == brutal['trial_sd_ann'] == pytest.approx(np.std(pooled, ddof=1), abs=1e-3)
    assert 0 < lenient['hurdle_expected_max_sharpe_ann'] < brutal['hurdle_expected_max_sharpe_ann']
    assert brutal['dsr'] < lenient['dsr']
    assert (ens['bootstrap_full']['T'], ens['bootstrap_oos']['T'], ens['oos_psr']['T']) == (260, n_oos, n_oos)
    assert 'registry: 21 logged trials across 6 variant files' in printed


def test_selection_path_pools_the_later_search_beside_the_own_grid(registry, tmp_path):
    res = tmp_path / 'results'
    write_results(res)
    registry.main()
    sleeves = json.loads((res / 'trial_registry.json').read_text())['sleeves']
    # Only xsec_etf_mom's deployed rule came from a second logged search.
    assert [k for k, v in sleeves.items() if 'dsr_selection_path' in v] == ['xsec_etf_mom']
    own, path = sleeves['xsec_etf_mom']['dsr_own_grid'], sleeves['xsec_etf_mom']['dsr_selection_path']
    assert (own['n_trials'], path['n_trials']) == (2, 5)  # own grid, then own grid + ETF rows only
    assert path['sources'] == {'xsec_etf_mom_variants.csv': 2, 'rank_hysteresis_variants.csv': 3}
    assert path['trial_sd_ann'] == pytest.approx(
        np.std(GRIDS['xsec_etf_mom'] + HYSTERESIS_ETF, ddof=1), abs=1e-3)
    assert path['sharpe_ann'] == own['sharpe_ann'] and path['T'] == own['T']

    pool = registry.selection_path_pool('xsec_etf_mom')
    assert pool['full_sharpes'] == GRIDS['xsec_etf_mom'] + HYSTERESIS_ETF
    assert registry.selection_path_pool('tsmom_trend') is None
    (res / 'rank_hysteresis_variants.csv').write_text('sleeve,name,full_sharpe\nstock,s1,0.9\n')
    with pytest.raises(ValueError, match='no rows match'):
        registry.selection_path_pool('xsec_etf_mom')


def test_returns_that_do_not_reproduce_the_declared_ensemble_are_refused(registry, tmp_path):
    res = tmp_path / 'results'
    _, declared = write_results(res)
    (res / 'ensemble.json').write_text(json.dumps(
        {**declared, 'full': {'sharpe': declared['full']['sharpe'] + 0.5}}))
    with pytest.raises(ValueError, match='!= declared'):
        registry.main()
    (res / 'ensemble.json').write_text(json.dumps({**declared, 'end': '2099-01-01'}))
    with pytest.raises(ValueError, match='date span'):
        registry.main()
    assert not (res / 'trial_registry.json').exists()


def test_rerun_keeps_a_saved_registry_that_differs(registry, tmp_path, monkeypatch, capsys):
    res = tmp_path / 'results'
    write_results(res)
    registry.main()
    saved = (res / 'trial_registry.json').read_bytes()
    assert json.loads(saved)['n_trials_total'] == 21
    registry.main()  # nothing changed: the record stays, no second copy appears
    assert (res / 'trial_registry.json').read_bytes() == saved
    assert not (res / 'recomputed').exists()

    # A new trial is logged: the re-scan differs from the saved record.
    (res / 'killed_idea_variants.csv').write_text('name,full_sharpe\na,-0.2\nb,0.1\nc,\nd,0.3\ne,0.0\nf,0.7\n')
    capsys.readouterr()
    registry.main()
    assert 'kept existing record' in capsys.readouterr().out
    assert (res / 'trial_registry.json').read_bytes() == saved
    assert json.loads((res / 'recomputed' / 'trial_registry.json').read_text())['n_trials_total'] == 22

    monkeypatch.setattr(sys, 'argv', ['trial_registry.py', '--rebase'])
    registry.main()
    assert json.loads((res / 'trial_registry.json').read_text())['n_trials_total'] == 22


def test_missing_saved_inputs_end_the_run_with_one_line_naming_the_command(registry, tmp_path):
    """A checkout that has not run the sweeps or the ensemble yet: the report
    says what to create first and how, and writes nothing."""
    res = tmp_path / 'results'
    with pytest.raises(SystemExit) as stop:
        registry.main()
    message = stop.value.code
    assert isinstance(message, str) and len(message.splitlines()) == 1
    assert message.startswith('results/mean_reversion_variants.csv not found: ')
    assert 'Create it first: python research/mean_reversion_sweep.py (6 more saved input(s)' in message
    assert message.endswith('sleeve_returns.csv, ensemble.json)')
    assert list(res.iterdir()) == []

    write_results(res)
    assert registry.missing_inputs() == []
    (res / 'sleeve_returns.csv').unlink()
    with pytest.raises(SystemExit) as stop:
        registry.main()
    assert stop.value.code == ('results/sleeve_returns.csv not found: the trial registry reads the '
                               'saved selection grids and the blended returns. Create it first: '
                               'python src/ensemble.py')
    assert not (res / 'trial_registry.json').exists()


def test_every_grid_the_report_reads_has_a_command_that_writes_it(registry):
    named = set(registry.SLEEVE_GRID.values())
    named |= {name for extras in registry.SLEEVE_EXTRA_GRIDS.values() for name, _ in extras}
    named |= {'sleeve_returns.csv', 'ensemble.json'}
    assert named == set(registry.INPUT_COMMANDS)
    # each command names a script that is part of this repository
    for command in registry.INPUT_COMMANDS.values():
        script = command.split()[1]
        assert command.startswith('python ') and (ROOT / script).is_file(), command


@pytest.mark.parametrize('argv, code', [(['--help'], 0), (['--reb'], 2), (['extra'], 2)])
def test_command_line_refuses_unknown_and_shortened_flags_before_reading_anything(
        monkeypatch, capsys, argv, code):
    """--rebase is acted on by exact name (qcore.records reads sys.argv), so a
    prefix the parser would otherwise expand must be an error, not a run that
    looks like a re-base and is not one."""
    import qcore.records

    def refuse(*args, **kwargs):
        raise AssertionError('the report ran')

    script = ROOT / 'scripts' / 'trial_registry.py'
    monkeypatch.setattr(qcore.records, 'save_record', refuse)
    monkeypatch.setattr(pd, 'read_csv', refuse)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', [str(script)] + argv)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(script), run_name='__main__')
    assert stop.value.code == code
    shown = capsys.readouterr()
    assert 'usage:' in (shown.out if code == 0 else shown.err)


def test_recomputed_sweep_logs_are_not_counted_and_the_caveat_says_so(registry, tmp_path):
    res = tmp_path / 'results'
    write_results(res)
    # Where a guarded sweep re-run that differs from its saved log is written.
    (res / 'recomputed').mkdir()
    (res / 'recomputed' / 'killed_idea_variants.csv').write_text('name,full_sharpe\ny,0.8\nz,0.9\n')
    (res / 'recomputed' / 'unsaved_idea_variants.csv').write_text('name,full_sharpe\nz,0.9\n')
    registry.main()
    out = json.loads((res / 'trial_registry.json').read_text())
    assert (out['n_trials_total'], out['n_families']) == (21, 6)
    assert out['registry']['killed_idea'] == 5 and 'unsaved_idea' not in out['registry']
    [caveat] = [c for c in out['caveats'] if 'experiment ledger' in c]
    assert 'results/recomputed/, which this scan does not read' in caveat
    assert '--rebase' in caveat and 'overwrite' not in caveat
