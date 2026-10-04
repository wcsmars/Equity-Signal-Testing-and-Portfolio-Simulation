"""Order-sheet decision rules pinned at their boundaries (no orders, no network).

Synthetic prices and ledgers only. These pin what the live order generator
decides: when a live sheet may be printed at all (New York date, time of
day, early closes, cache age), that a missing decision-close quote stops the
run, how the account size is tied to the ledger, sizing and cash parking,
the exact order lines, the minimum-order filter, the ledger date guard, the
proposed post-trade ledger, the operator alert channel and the exit codes.
"""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SLEEVES = ['mean_reversion', 'seasonality_flows', 'tsmom_trend', 'xsec_etf_mom']
DAY = '2024-01-02'
NY = 'America/New_York'


def module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem + '_live_rules_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _live(monkeypatch, px, weights=None, raw=None):
    mod = module('scripts/live_targets.py')
    monkeypatch.setattr(mod, 'load_prices', lambda: px)
    monkeypatch.setattr(mod, 'load', lambda _: px if raw is None else raw)
    monkeypatch.setattr(mod, 'ensemble_weights', lambda: weights or dict.fromkeys(SLEEVES, .25))
    return mod


def _sleeves(monkeypatch, mod, daily, monthly=None):
    for attr in ['sleeve_mean_reversion', 'sleeve_seasonality']:
        monkeypatch.setattr(mod, attr, lambda _: (pd.Series(daily), 'daily'))
    for attr in ['sleeve_tsmom', 'sleeve_xsec_etf']:
        monkeypatch.setattr(mod, attr, lambda _: ((None, 'HOLD') if monthly is None
                                                  else (pd.Series(monthly), 'MONTH-END')))


def _run(mod, monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', *argv])
    try:
        mod.main()
    except SystemExit as exc:
        return exc.code
    return 0


def _prices(day=DAY, **quotes):
    return pd.DataFrame({k: [float(v)] for k, v in quotes.items()}, index=pd.to_datetime([day]))


def _ledger(tmp_path, rows, header='sleeve,ticker,shares', name='ledger.csv'):
    path = tmp_path / name
    path.write_text(header + '\n' + ''.join(row + '\n' for row in rows))
    return path


def _orders(out):
    return out.split('== MOC ORDERS')[1]


# ============================================================ session gate
def test_session_date_and_time_are_new_york_not_utc_or_local(monkeypatch):
    mod = module('scripts/live_targets.py')
    # 15:42 in New York on 1 July is 19:42 UTC, and already 2 July from UTC+5 eastward
    monkeypatch.setattr(mod, '_utc_now', lambda: pd.Timestamp('2026-07-01 19:42', tz='UTC'))
    assert mod.current_ny_session_date() == pd.Timestamp('2026-07-01')
    assert mod.current_ny_time().strftime('%H:%M %z') == '15:42 -0400'
    # 23:30 in New York: the UTC date has rolled over, the session date has not
    monkeypatch.setattr(mod, '_utc_now', lambda: pd.Timestamp('2026-07-02 03:30', tz='UTC'))
    assert mod.current_ny_session_date() == pd.Timestamp('2026-07-01')
    # standard time: 15:42 EST is 20:42 UTC
    monkeypatch.setattr(mod, '_utc_now', lambda: pd.Timestamp('2026-12-01 20:42', tz='UTC'))
    assert mod.current_ny_time().strftime('%Y-%m-%d %H:%M') == '2026-12-01 15:42'


@pytest.mark.parametrize('day,early', [
    ('2025-11-28', True), ('2026-11-27', True), ('2027-11-26', True),   # Friday after Thanksgiving
    ('2026-11-26', False), ('2026-11-20', False), ('2024-11-22', False),
    ('2024-11-29', True),                                              # latest possible date
    ('2025-07-03', True), ('2024-07-03', True), ('2023-07-03', True),   # July 3, Monday-Thursday
    ('2026-07-02', False), ('2026-07-01', False),
    ('2026-12-24', True), ('2025-12-24', True), ('2027-12-23', False),  # December 24, Monday-Thursday
    ('2021-12-24', False),                                             # a Friday: full holiday, not a half day
    ('2024-01-02', False),
])
def test_scheduled_early_closes(day, early):
    mod = module('scripts/live_targets.py')
    assert mod.nyse_early_close(pd.Timestamp(day)) is early


def test_submission_window_is_thirty_to_ten_minutes_before_the_close():
    mod = module('scripts/live_targets.py')
    assert (mod.SUBMIT_WINDOW_MINUTES, mod.MOC_CUTOFF_MINUTES) == (30, 10)
    stamp = lambda text: pd.Timestamp(text, tz=NY)
    assert mod.submission_window(pd.Timestamp('2026-07-01')) == (
        stamp('2026-07-01 15:30'), stamp('2026-07-01 15:50'), stamp('2026-07-01 16:00'))
    assert mod.submission_window(pd.Timestamp('2025-11-28')) == (
        stamp('2025-11-28 12:30'), stamp('2025-11-28 12:50'), stamp('2025-11-28 13:00'))


@pytest.mark.parametrize('day,now,snapshot,expected', [
    ('2026-07-01', '15:42', '15:40', None),
    ('2026-07-01', '15:30', '15:30', None),                       # window opens
    ('2026-07-01', '15:49', '15:31', None),
    ('2026-07-01', '15:29', '15:28', 'before the submission window opens at 15:30'),
    ('2026-07-01', '09:00', '08:59', 'before the submission window'),
    ('2026-07-01', '15:50', '15:45', 'past the MOC cutoff 15:50'),  # the cutoff itself is too late
    ('2026-07-01', '20:30', '20:29', 'fills at the NEXT close'),
    ('2026-07-01', '23:55', '23:54', 'past the MOC cutoff'),
    ('2026-07-01', '15:42', '10:05', 'price cache was written 2026-07-01 10:05'),
    ('2026-07-01', '15:42', '15:29', 'refresh the data now'),
    ('2026-07-01', '15:42', '15:42:45', None),                    # inside the clock-skew allowance
    ('2026-07-01', '15:42', '15:43:01', 'after the current New York time 15:42'),
    ('2026-07-01', '15:42', '15:47', 'its age cannot be checked'),
    ('2025-11-28', '15:42', '15:40', 'past the MOC cutoff 12:50'),  # half day, usual run time
    ('2025-11-28', '12:42', '12:40', None),
    ('2025-11-28', '12:42', '12:29', 'before the submission window opened at 12:30'),
])
def test_live_sheet_only_inside_the_submission_window(day, now, snapshot, expected):
    mod = module('scripts/live_targets.py')
    reason = mod.submission_window_error(pd.Timestamp(day), pd.Timestamp(f'{day} {now}', tz=NY),
                                         pd.Timestamp(f'{day} {snapshot}', tz=NY))
    if expected is None:
        assert reason is None
    else:
        assert expected in reason


def test_a_clock_that_rolled_past_midnight_is_refused():
    mod = module('scripts/live_targets.py')
    reason = mod.submission_window_error(pd.Timestamp('2026-07-01'),
                                         pd.Timestamp('2026-07-02 00:05', tz=NY),
                                         pd.Timestamp('2026-07-01 15:40', tz=NY))
    assert 'date changed' in reason


def test_yesterdays_snapshot_is_stale_even_at_the_right_time_of_day():
    mod = module('scripts/live_targets.py')
    reason = mod.submission_window_error(pd.Timestamp('2026-07-01'),
                                         pd.Timestamp('2026-07-01 15:42', tz=NY),
                                         pd.Timestamp('2026-06-30 15:41', tz=NY))
    assert 'price cache was written 2026-06-30 15:41' in reason


def test_a_snapshot_stamped_after_the_clock_is_refused_not_trusted():
    mod = module('scripts/live_targets.py')
    assert mod.SNAPSHOT_CLOCK_SKEW_SECONDS == 60
    reason = mod.submission_window_error(pd.Timestamp('2026-07-01'),
                                         pd.Timestamp('2026-07-01 15:42', tz=NY),
                                         pd.Timestamp('2026-07-02 09:00', tz=NY))
    assert 'price cache is stamped 2026-07-02 09:00 ET, after the current New York time 15:42' in reason
    assert 'system clock' in reason


def test_cache_snapshot_time_is_the_older_close_file_in_new_york_time(monkeypatch, tmp_path):
    mod = module('scripts/live_targets.py')
    monkeypatch.setattr(mod, 'DATA_DIR', tmp_path)
    for name, text in [('adj_close', '2026-07-01 19:41'), ('close', '2026-07-01 14:05')]:
        path = tmp_path / f'{name}.csv'
        path.write_text('x')
        stamp = pd.Timestamp(text, tz='UTC').timestamp()
        os.utime(path, (stamp, stamp))
    assert mod.cache_snapshot_time() == pd.Timestamp('2026-07-01 10:05', tz=NY)


def _live_session(monkeypatch, now, snapshot, day=DAY):
    mod = _live(monkeypatch, _prices(day, SPY=100, SGOV=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    monkeypatch.setattr(mod, 'current_ny_time', lambda: pd.Timestamp(f'{day} {now}', tz=NY))
    monkeypatch.setattr(mod, 'cache_snapshot_time', lambda: pd.Timestamp(f'{day} {snapshot}', tz=NY))
    return mod


def test_live_run_inside_the_window_prints_the_cutoff_and_the_orders(monkeypatch, tmp_path, capsys):
    mod = _live_session(monkeypatch, '15:42', '15:40')
    ledger = _ledger(tmp_path, [])
    assert _run(mod, monkeypatch, '--equity', '1000', '--ledger', str(ledger)) == 0
    out = capsys.readouterr().out
    assert 'LIVE session 2024-01-02: New York time 15:42, MOC cutoff 15:50 ET' in out
    assert 'close 16:00 ET; price cache written 15:40 ET' in out
    assert 'OFFLINE' not in out and 'no as_of date' not in out   # an empty ledger has nothing to date
    assert 'BUY       2 SPY' in _orders(out)


@pytest.mark.parametrize('now,snapshot,day,message', [
    ('20:30', '20:29', DAY, 'past the MOC cutoff 15:50 ET'),       # an evening run, hours after the close
    ('09:00', '08:59', DAY, 'before the submission window opens'),
    ('15:42', '09:35', DAY, 'price cache was written'),            # morning download, afternoon run
    ('15:42', '15:40', '2025-11-28', 'past the MOC cutoff 12:50 ET'),
])
def test_live_run_outside_the_window_prints_no_orders(monkeypatch, tmp_path, capsys, now, snapshot,
                                                      day, message):
    mod = _live_session(monkeypatch, now, snapshot, day)
    ledger = _ledger(tmp_path, [])
    assert _run(mod, monkeypatch, '--equity', '1000', '--ledger', str(ledger)) == 2
    captured = capsys.readouterr()
    assert message in captured.err and f'--as-of {day}' in captured.err
    assert 'MOC ORDERS' not in captured.out and 'decision close' not in captured.out


@pytest.mark.parametrize('now,cache_end', [
    ('2024-01-03 15:42', '2024-01-02'),              # yesterday's decision cannot be sent today
    ('2024-01-06 15:42', '2024-01-06'),              # a Saturday row is not a session
    ('2024-07-04 15:42', '2024-07-04'),              # nor is a holiday row
])
def test_live_run_needs_todays_scheduled_session_before_any_time_check(monkeypatch, tmp_path, capsys,
                                                                      now, cache_end):
    mod = _live(monkeypatch, _prices(cache_end, SPY=100, SGOV=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    monkeypatch.setattr(mod, 'current_ny_time', lambda: pd.Timestamp(now, tz=NY))
    monkeypatch.setattr(mod, 'cache_snapshot_time', lambda: pd.Timestamp(now, tz=NY))
    assert _run(mod, monkeypatch, '--equity', '1000', '--ledger', str(_ledger(tmp_path, []))) == 2
    captured = capsys.readouterr()
    assert "today's New York scheduled session" in captured.err and 'MOC ORDERS' not in captured.out
    assert f'the cache in {mod.DATA_DIR} ends {cache_end}' in captured.err
    assert 'QCORE_DATA_DIR' in captured.err


def test_half_day_run_inside_its_own_window_is_accepted(monkeypatch, tmp_path, capsys):
    mod = _live_session(monkeypatch, '12:42', '12:40', '2025-11-28')
    assert _run(mod, monkeypatch, '--equity', '1000', '--ledger', str(_ledger(tmp_path, []))) == 0
    assert 'MOC cutoff 12:50 ET' in capsys.readouterr().out


def test_as_of_is_the_explicit_offline_override_at_any_time_of_day(monkeypatch, capsys):
    mod = _live_session(monkeypatch, '20:30', '20:29')
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY) == 0
    out = capsys.readouterr().out
    assert 'OFFLINE historical simulation' in out and 'LIVE session' not in out


def test_documented_live_flow_selects_the_live_cache_before_the_download():
    doc = module('scripts/live_targets.py').__doc__
    assert 'export QCORE_DATA_DIR=' in doc
    assert doc.index('export QCORE_DATA_DIR=') < doc.index(' src/download_data.py')
    assert doc.index(' src/download_data.py') < doc.index(' scripts/live_targets.py')


def _project_copy(tmp_path, weights=True):
    """The script and the source tree it imports, copied beside an empty
    results folder: a subprocess run then needs no saved file of this
    checkout, and can be shown what happens when one is missing."""
    project = tmp_path / 'project'
    shutil.copytree(ROOT / 'src', project / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    (project / 'scripts').mkdir()
    shutil.copy(ROOT / 'scripts' / 'live_targets.py', project / 'scripts')
    (project / 'results').mkdir()
    if weights:
        (project / 'results' / 'ensemble.json').write_text(
            json.dumps({'sleeve_weights': dict.fromkeys(SLEEVES, .25)}))
    return project


def _cache(tmp_path, end):
    probe = module('scripts/live_targets.py')
    px = _panel(probe, end).round(4)
    cache = tmp_path / 'live_cache'
    cache.mkdir()
    for name in ('adj_close', 'close'):
        px.to_csv(cache / f'{name}.csv')
    return cache


def test_script_reads_the_cache_named_by_qcore_data_dir(tmp_path):
    # the documented live flow refreshes a separate cache through this variable;
    # the order generator must read that cache, not the default one
    cache = _cache(tmp_path, '2024-04-15')
    env = {**os.environ, 'QCORE_DATA_DIR': str(cache), 'PYTHONDONTWRITEBYTECODE': '1'}
    script = str(_project_copy(tmp_path) / 'scripts' / 'live_targets.py')
    offline = subprocess.run([sys.executable, script, '--as-of', '2024-04-15', '--equity', '100000'],
                             env=env, capture_output=True, text=True)
    assert offline.returncode == 0, offline.stderr
    assert 'decision close: 2024-04-15  (equity $100,000)' in offline.stdout
    assert '(weight 25.0%, capital $25,000)' in offline.stdout
    live = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
    assert live.returncode == 2 and 'MOC ORDERS' not in live.stdout
    assert f'the cache in {cache.resolve()} ends 2024-04-15' in live.stderr
    assert 'QCORE_DATA_DIR' in live.stderr


def test_missing_ensemble_weights_refuse_the_run_and_name_the_command(tmp_path):
    # a checkout that has not built the ensemble yet: no weights, no order sheet
    cache = _cache(tmp_path, '2024-04-15')
    env = {**os.environ, 'QCORE_DATA_DIR': str(cache), 'PYTHONDONTWRITEBYTECODE': '1'}
    script = str(_project_copy(tmp_path, weights=False) / 'scripts' / 'live_targets.py')
    done = subprocess.run([sys.executable, script, '--as-of', '2024-04-15', '--equity', '100000'],
                          env=env, capture_output=True, text=True)
    assert done.returncode == 2 and 'Traceback' not in done.stderr
    assert ('LIVE TARGETS REFUSED: results/ensemble.json not found: it holds the sleeve weights '
            'that size every order. Create it first: python src/ensemble.py') in done.stderr
    assert 'NET PORTFOLIO TARGET' not in done.stdout and 'MOC ORDERS' not in done.stdout


def test_missing_data_cache_refuses_the_run_and_names_the_downloader(tmp_path):
    empty = tmp_path / 'no_cache'
    empty.mkdir()
    env = {**os.environ, 'QCORE_DATA_DIR': str(empty), 'PYTHONDONTWRITEBYTECODE': '1'}
    script = str(_project_copy(tmp_path) / 'scripts' / 'live_targets.py')
    done = subprocess.run([sys.executable, script, '--as-of', '2024-04-15'],
                          env=env, capture_output=True, text=True)
    assert done.returncode == 2 and 'Traceback' not in done.stderr
    assert 'LIVE TARGETS REFUSED:' in done.stderr and 'adj_close.csv not found' in done.stderr
    assert 'python src/download_data.py' in done.stderr and done.stdout == ''


def test_as_of_must_equal_the_cache_end_and_both_caches_must_agree(monkeypatch, capsys):
    px = pd.DataFrame({'SPY': [100., 101.]}, index=pd.to_datetime(['2024-01-02', '2024-01-03']))
    mod = _live(monkeypatch, px)
    assert _run(mod, monkeypatch, '--as-of', '2024-01-02') == 2
    assert '--as-of must equal' in capsys.readouterr().err
    mod = _live(monkeypatch, px, raw=px.iloc[:1])    # raw close cache one day behind
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--as-of', '2024-01-03'])
    with pytest.raises(ValueError, match='different latest dates'):
        mod.main()


def test_unparseable_as_of_is_a_usage_error_not_a_traceback(monkeypatch, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    assert _run(mod, monkeypatch, '--as-of', 'banana') == 2
    assert "--as-of must be a date YYYY-MM-DD, got 'banana'" in capsys.readouterr().err


# ===================================================== decision-row quotes
def _universe(mod):
    import mean_reversion as mr
    import tsmom_trend as tt
    import xsec_etf_mom as xe
    return sorted(set(mr.UNIVERSE + tt.RISK + [tt.CASH] + xe.EQ_UNIVERSE + [xe.DEFENSIVE]))


def _panel(mod, end, start='2022-01-03'):
    from qcore.quality import nyse_bdays
    tickers = _universe(mod)
    idx = nyse_bdays(start, end)
    rng = np.random.default_rng(4)
    steps = rng.normal(0.0003, 0.01, (len(idx), len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)


def test_monthly_sleeves_hold_between_month_ends_and_decide_at_month_end():
    live = module('scripts/live_targets.py')
    mid = _panel(live, '2024-04-15')
    for sleeve in (live.sleeve_tsmom, live.sleeve_xsec_etf):
        weights, note = sleeve(mid)
        assert weights is None and 'HOLD' in note
    end = _panel(live, '2024-04-30')
    w_trend, note = live.sleeve_tsmom(end)
    assert 'MONTH-END' in note and w_trend.name == end.index[-1]
    assert 0.0 <= w_trend.sum() <= 1.0 + 1e-9 and (w_trend >= 0).all()
    w_xsec, note = live.sleeve_xsec_etf(end)
    assert 'MONTH-END' in note and w_xsec.name == end.index[-1]
    assert w_xsec.sum() == pytest.approx(1.0)


def test_missing_decision_quote_stops_a_deciding_sleeve_instead_of_reweighting():
    live = module('scripts/live_targets.py')
    end = _panel(live, '2024-04-30')
    for sleeve, ticker in [(live.sleeve_xsec_etf, 'EWY'), (live.sleeve_tsmom, 'LQD'),
                           (live.sleeve_mean_reversion, 'XLE'), (live.sleeve_seasonality, 'SPY')]:
        assert sleeve(end)[0] is not None            # the clean panel decides
        for bad in (np.nan, 0.0):
            broken = end.copy()
            broken.loc[broken.index[-1], ticker] = bad
            with pytest.raises(ValueError, match=f'no valid decision-close quote on 2024-04-30 for: {ticker}'):
                sleeve(broken)


def test_missing_quote_matters_only_to_a_sleeve_that_decides_today():
    live = module('scripts/live_targets.py')
    mid = _panel(live, '2024-04-15')
    mid.loc[mid.index[-1], ['EWY', 'LQD']] = np.nan
    assert live.sleeve_xsec_etf(mid)[0] is None and live.sleeve_tsmom(mid)[0] is None
    mid.loc[mid.index[-1], 'XLE'] = np.nan         # the daily sleeve decides every session
    with pytest.raises(ValueError, match='mean_reversion: no valid decision-close quote'):
        live.sleeve_mean_reversion(mid)


def test_decision_row_check_names_every_bad_listed_ticker_and_skips_unlisted_ones():
    live = module('scripts/live_targets.py')
    idx = pd.bdate_range('2024-01-02', periods=3)
    px = pd.DataFrame({'A': [1., 2., np.nan], 'B': [1., 2., 0.], 'C': [1., 2., np.inf],
                       'D': [1., 2., -3.], 'NEW': [np.nan] * 3, 'OK': [1., 2., 3.]}, index=idx)
    with pytest.raises(ValueError, match='for: A, B, C, D;'):
        live.require_decision_row(px, list(px.columns), 'sleeve')
    live.require_decision_row(px, ['NEW', 'OK'], 'sleeve')   # not yet listed: the builder's business


def test_missing_quote_at_month_end_prints_no_order_sheet(monkeypatch, tmp_path, capsys):
    probe = module('scripts/live_targets.py')
    px = _panel(probe, '2024-04-30')
    px['SGOV'] = 100.0
    px.loc[px.index[-1], 'EWY'] = np.nan
    mod = _live(monkeypatch, px)
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--equity', '100000', '--as-of', '2024-04-30',
                                      '--ledger', str(_ledger(tmp_path, []))])
    with pytest.raises(ValueError, match='xsec_etf_mom: no valid decision-close quote .* EWY'):
        mod.main()
    out = capsys.readouterr().out
    assert 'MOC ORDERS' not in out and 'NET PORTFOLIO TARGET' not in out


def test_daily_sleeves_use_the_decision_row_not_the_previous_one():
    live = module('scripts/live_targets.py')
    import mean_reversion as mr
    from qcore.quality import nyse_bdays
    idx = pd.bdate_range('2023-01-02', periods=260)
    px = pd.DataFrame({c: 100 + 0.1 * np.arange(260) for c in mr.UNIVERSE}, index=idx)
    px.loc[idx[-2], 'XLE'] = px.loc[idx[-3], 'XLE'] - 1.0     # two down closes in an uptrend:
    px.loc[idx[-1], 'XLE'] = px.loc[idx[-3], 'XLE'] - 2.0     # RSI-2 crosses below 5 only today
    weights, _ = live.sleeve_mean_reversion(px)
    assert weights.name == idx[-1] and weights['XLE'] == pytest.approx(0.20)
    assert weights.drop('XLE').eq(0).all()
    # turn-of-month: the session after 24 April 2024 is the fourth-last of the month
    spy = lambda end: pd.DataFrame({'SPY': 100.0}, index=nyse_bdays('2024-03-01', end))
    assert live.sleeve_seasonality(spy('2024-04-24'))[0]['SPY'] == 1.0
    assert live.sleeve_seasonality(spy('2024-04-23'))[0]['SPY'] == 0.0


# ================================================== account size and ledger
def test_an_order_list_has_no_default_account_size(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,5'])
    assert _run(mod, monkeypatch, '--as-of', DAY, '--ledger', str(ledger)) == 2
    captured = capsys.readouterr()
    assert '--ledger needs --cash' in captured.err and 'decision close' not in captured.out
    # without a ledger there are no orders: the documented offline command keeps a labelled default
    assert _run(mod, monkeypatch, '--as-of', DAY) == 0
    out = capsys.readouterr().out
    assert 'sizing a hypothetical $100,000 account (targets only)' in out
    assert 'decision close: 2024-01-02  (equity $100,000)' in out
    assert _run(mod, monkeypatch, '--as-of', DAY, '--equity', '100000') == 0
    assert 'hypothetical $100,000' not in capsys.readouterr().out


def test_cash_derives_equity_from_the_ledger_at_the_decision_close(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, QQQ=50))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,5', 'tsmom_trend,QQQ,1'])
    assert _run(mod, monkeypatch, '--as-of', DAY, '--cash', '450', '--ledger', str(ledger)) == 0
    out = capsys.readouterr().out
    assert 'ledger market value $550.00 at the decision close; cash $450.00 (as given)' in out
    assert 'decision close: 2024-01-02  (equity $1,000)' in out
    assert '(weight 25.0%, capital $250)' in out


def test_equity_prints_the_cash_it_implies(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, QQQ=50))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,5', 'tsmom_trend,QQQ,1'])
    assert _run(mod, monkeypatch, '--as-of', DAY, '--equity', '1000', '--ledger', str(ledger)) == 0
    assert ('ledger market value $550.00 at the decision close; cash $450.00 '
            '(equity - ledger; must equal the broker cash balance)') in capsys.readouterr().out


def test_ledger_worth_more_than_equity_is_refused_unless_margin_is_declared(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,12'])
    for money in (['--equity', '1000'], ['--cash', '-200']):
        monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--as-of', DAY, *money, '--ledger', str(ledger)])
        with pytest.raises(ValueError, match=r'ledger market value \$1,200.00 exceeds equity \$1,000.00'):
            mod.main()
        assert 'MOC ORDERS' not in capsys.readouterr().out
        assert _run(mod, monkeypatch, '--as-of', DAY, *money, '--allow-margin', '--ledger', str(ledger)) == 0
        assert 'SELL     12 SPY' in _orders(capsys.readouterr().out)


def test_account_size_arguments_are_validated(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    ledger = str(_ledger(tmp_path, ['mean_reversion,SPY,5']))
    for argv, message in [
        (['--cash', '500'], '--cash needs --ledger'),
        (['--cash', '500', '--equity', '1000', '--ledger', ledger], 'not allowed with'),
        (['--equity', '0', '--ledger', ledger], '--equity must be finite and positive'),
        (['--cash', 'nan', '--ledger', ledger], '--cash must be finite'),
        (['--equity', '1000', '--ledger', str(tmp_path / 'absent.csv')], 'ledger not found'),
    ]:
        assert _run(mod, monkeypatch, '--as-of', DAY, *argv) == 2
        assert message in capsys.readouterr().err
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--as-of', DAY, '--cash', '-500', '--allow-margin',
                                      '--ledger', ledger])
    with pytest.raises(ValueError, match='not a positive equity'):
        mod.main()


def test_ledger_header_spacing_is_tolerated(tmp_path):
    mod = module('scripts/live_targets.py')
    ledger = mod.read_ledger(_ledger(tmp_path, ['cash, sgov, 3'], header='sleeve, ticker, shares'))
    assert ledger.loc[('cash', 'SGOV')] == 3 and ledger.attrs['as_of'] is None


def test_ledger_as_of_is_one_session_date(tmp_path):
    mod = module('scripts/live_targets.py')
    header = 'sleeve,ticker,shares,as_of'
    good = _ledger(tmp_path, ['cash,SGOV,3,2024-01-02', 'mean_reversion,SPY,1,2024-01-02'], header)
    assert mod.read_ledger(good).attrs['as_of'] == pd.Timestamp('2024-01-02')
    assert mod.read_ledger(_ledger(tmp_path, [], header)).attrs['as_of'] is None
    for rows in (['cash,SGOV,3,2024-01-02', 'mean_reversion,SPY,1,2024-01-03'],
                 ['cash,SGOV,3,2024-01-02', 'mean_reversion,SPY,1,'],
                 ['cash,SGOV,3,yesterday'], ['cash,SGOV,3,20240102']):
        with pytest.raises(ValueError, match='as_of must be one YYYY-MM-DD session date'):
            mod.read_ledger(_ledger(tmp_path, rows, header))


@pytest.mark.parametrize('as_of,expected', [
    ('2024-01-02', None),                           # the previous session
    ('2024-01-03', 'sent twice'),                   # already holds this session's fills
    ('2024-01-04', 'sent twice'),
    ('2023-12-29', 'previous NYSE session was 2024-01-02'),   # a session of fills is missing
])
def test_dated_ledger_must_be_as_of_the_previous_session(as_of, expected):
    mod = module('scripts/live_targets.py')
    reason = mod.ledger_date_error(pd.Timestamp(as_of), pd.Timestamp('2024-01-03'))
    assert reason is None if expected is None else expected in reason


def test_previous_session_skips_weekends_and_holidays():
    mod = module('scripts/live_targets.py')
    # Tuesday 2 January 2024 follows Friday 29 December 2023 (weekend, then New Year's Day)
    assert mod.ledger_date_error(pd.Timestamp('2023-12-29'), pd.Timestamp('2024-01-02')) is None
    assert 'previous NYSE session was 2023-12-29' in mod.ledger_date_error(
        pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'))


def test_ledger_that_already_holds_todays_fills_cannot_repeat_the_orders(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    header = 'sleeve,ticker,shares,as_of'
    same_day = _ledger(tmp_path, [f'mean_reversion,SPY,5,{DAY}'], header)
    argv = ['live_targets.py', '--as-of', DAY, '--equity', '1000', '--ledger', str(same_day)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='sent twice .*--ignore-ledger-date overrides'):
        mod.main()
    assert 'MOC ORDERS' not in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', argv + ['--ignore-ledger-date'])
    mod.main()
    assert 'SELL      3 SPY' in _orders(capsys.readouterr().out)
    previous = _ledger(tmp_path, ['mean_reversion,SPY,5,2023-12-29'], header, name='previous.csv')
    assert _run(mod, monkeypatch, '--as-of', DAY, '--equity', '1000', '--ledger', str(previous)) == 0
    out = capsys.readouterr().out
    assert 'no as_of date' not in out and 'SELL      3 SPY' in _orders(out)
    undated = _ledger(tmp_path, ['mean_reversion,SPY,5'], name='undated.csv')
    assert _run(mod, monkeypatch, '--as-of', DAY, '--equity', '1000', '--ledger', str(undated)) == 0
    assert 'ledger has no as_of date' in capsys.readouterr().out


def test_a_ledger_older_than_the_previous_session_needs_the_explicit_override(monkeypatch, tmp_path, capsys):
    # a skipped session: the ledger may still be current, but only the operator knows
    mod = _live(monkeypatch, _prices('2024-01-04', SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    old = _ledger(tmp_path, ['mean_reversion,SPY,5,2024-01-02'], 'sleeve,ticker,shares,as_of')
    argv = ['--as-of', '2024-01-04', '--equity', '1000', '--ledger', str(old)]
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', *argv])
    with pytest.raises(ValueError, match='previous NYSE session was 2024-01-03: .*if no order was sent '
                                         'since then.*--ignore-ledger-date overrides'):
        mod.main()
    assert 'MOC ORDERS' not in capsys.readouterr().out
    assert _run(mod, monkeypatch, *argv, '--ignore-ledger-date') == 0
    assert 'SELL      3 SPY' in _orders(capsys.readouterr().out)


# ===================================================== sizing and parking
def test_sleeve_capital_follows_the_ensemble_weight_and_rounds_down(monkeypatch, capsys):
    weights = dict(zip(SLEEVES, [.4, .3, .2, .1]))
    mod = _live(monkeypatch, _prices(SPY=30), weights)
    monkeypatch.setattr(mod, 'sleeve_mean_reversion', lambda _: (pd.Series({'SPY': 1.}), 'daily'))
    monkeypatch.setattr(mod, 'sleeve_seasonality', lambda _: (pd.Series({'SPY': 0.5}), 'daily'))
    for attr in ['sleeve_tsmom', 'sleeve_xsec_etf']:
        monkeypatch.setattr(mod, attr, lambda _: (None, 'HOLD'))
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY) == 0
    out = capsys.readouterr().out
    assert '(weight 40.0%, capital $400)' in out and '->     13 sh @ 30.00' in out   # 400/30 = 13.3
    assert '(weight 30.0%, capital $300)' in out and '->      5 sh @ 30.00' in out   # 150/30 = 5.0
    assert 'SPY         18 sh' in out
    # the cost of flooring to whole shares is stated, not hidden: $550 of targets, $540 sized
    assert "whole-share rounding: today's sleeve targets $550 are sized $540 (-1.8%)" in out


def test_hold_without_ledger_keeps_that_capital_out_of_cash_parking(monkeypatch, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, SGOV=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.})          # daily sleeves flat, monthly sleeves HOLD
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY) == 0
    out = capsys.readouterr().out
    # $500 is spoken for by the two HOLD sleeves; of the other $500 one SGOV
    # share is given up for the $1 commission and the 25 bps price allowance
    assert 'cash parking after expense reserve: 4 sh SGOV' in out
    assert 'execution cash reserve $2.00' in out and '25 bps adverse-price allowance' in out


def test_cash_parking_rounds_down_to_whole_shares(monkeypatch, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, SGOV=100.4))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY) == 0   # $500 / $100.40 = 4.98
    out = capsys.readouterr().out
    assert 'cash parking after expense reserve: 4 sh SGOV' in out
    assert 'SGOV         4 sh' in out


def test_cash_parking_never_rounds_up_into_an_unsent_order(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, SGOV=100.4))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    ledger = _ledger(tmp_path, ['cash,SGOV,9'])      # $1,000 / $100.40 = 9.96 -> 9 shares, already held
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY, '--ledger', str(ledger),
                '--min-order', '150') == 0
    out = capsys.readouterr().out
    assert 'cash parking after expense reserve: 9 sh SGOV' in out
    assert 'skip' not in _orders(out) and _orders(out).strip().endswith('none')


def test_invalid_sleeve_weights_stop_order_generation(monkeypatch, capsys):
    for bad in ({'SPY': -0.1}, {'SPY': 0.8, 'QQQ': 0.4}, {'SPY': np.nan}):
        mod = _live(monkeypatch, _prices(SPY=100, QQQ=50))
        _sleeves(monkeypatch, mod, bad)
        monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--equity', '1000', '--as-of', DAY])
        with pytest.raises(ValueError, match='invalid long-only target weights'):
            mod.main()
        assert 'NET PORTFOLIO TARGET' not in capsys.readouterr().out


def test_ensemble_weights_must_be_the_four_sleeves_summing_to_one(monkeypatch, tmp_path):
    mod = module('scripts/live_targets.py')
    (tmp_path / 'results').mkdir()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    target = tmp_path / 'results' / 'ensemble.json'
    with pytest.raises(FileNotFoundError, match='Create it first: python src/ensemble.py'):
        mod.ensemble_weights()                       # not built yet: said in one line
    target.write_text(json.dumps({'sleeve_weights': dict(zip(SLEEVES, [.3961, .1823, .2749, .1466]))}))
    weights = mod.ensemble_weights()                 # 4dp rounding (sum .9999) is normalised
    assert sum(weights.values()) == pytest.approx(1.0) and weights['mean_reversion'] == pytest.approx(.3961 / .9999)
    for bad in ([.4, .2, .2, .1], [.5, .5, .1, -.1], [.25, .25, .25, float('nan')]):
        target.write_text(json.dumps({'sleeve_weights': dict(zip(SLEEVES, bad))}))
        with pytest.raises(ValueError, match='four positive sleeve weights'):
            mod.ensemble_weights()
    target.write_text(json.dumps({'sleeve_weights': dict(zip(SLEEVES[:3], [.4, .3, .3]))}))
    with pytest.raises(ValueError):
        mod.ensemble_weights()


# ============================================================ order lines
def test_orders_are_target_minus_ledger_with_the_right_side(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, QQQ=50))
    _sleeves(monkeypatch, mod, {'SPY': 0.4}, monthly={'QQQ': 0.4})   # targets: 2 SPY, 4 QQQ
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,5', 'tsmom_trend,QQQ,1'])
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY, '--ledger', str(ledger)) == 0
    lines = [line.strip() for line in _orders(capsys.readouterr().out).splitlines()[1:] if line.strip()]
    assert lines == ['BUY       3 QQQ   MOC   (~$150)', 'SELL      3 SPY   MOC   (~$300)']


def test_one_net_order_per_ticker_across_sleeves(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})                         # 1 share in each daily sleeve
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,2', 'seasonality_flows,SPY,3'])
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY, '--ledger', str(ledger)) == 0
    orders = _orders(capsys.readouterr().out)
    assert 'SELL      3 SPY' in orders and orders.count('SPY') == 1   # 5 held in total, 2 wanted


def test_hold_sleeve_keeps_its_ledger_shares_and_prints_them(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, QQQ=50))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    ledger = _ledger(tmp_path, ['tsmom_trend,QQQ,7', 'mean_reversion,SPY,1'])
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY, '--ledger', str(ledger)) == 0
    out = capsys.readouterr().out
    assert 'QQQ   holds      7 sh @ 50.00  ($      350)' in out
    assert 'no ledger position' in out               # the other HOLD sleeve owns nothing
    assert 'QQQ          7 sh' in out
    orders = _orders(out)
    assert 'QQQ' not in orders and 'SELL      1 SPY' in orders


def test_matching_ledger_prints_no_orders(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,1', 'seasonality_flows,SPY,1'])
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY, '--ledger', str(ledger)) == 0
    orders = _orders(capsys.readouterr().out)
    assert orders.strip().endswith('none') and 'BUY' not in orders and 'SELL' not in orders


def test_ticker_outside_a_sleeves_universe_is_flagged(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, NVDA=200, AAPL=300))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    ledger = _ledger(tmp_path, ['xsec_etf_mom,NVDA,5', 'mean_reversion,AAPL,1', 'seasonality_flows,SPY,1'])
    assert _run(mod, monkeypatch, '--equity', '5000', '--as-of', DAY, '--ledger', str(ledger)) == 0
    out = capsys.readouterr().out
    assert ('!! ledger holds 5 NVDA under xsec_etf_mom, which never trades it: the position '
            'is carried untouched while the sleeve HOLDs') in out
    assert '!! ledger holds 1 AAPL under mean_reversion, which never trades it: the position gets a zero target' in out
    assert out.count('!!') == 2                      # SPY belongs to the seasonality sleeve
    assert 'NVDA' not in _orders(out) and 'SELL      1 AAPL' in _orders(out)


def test_non_parking_ticker_under_the_cash_sleeve_is_flagged_as_sold(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, QQQ=50, SGOV=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    ledger = _ledger(tmp_path, ['cash,QQQ,2', 'cash,SGOV,1'])
    assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY, '--ledger', str(ledger)) == 0
    out = capsys.readouterr().out
    assert ('!! ledger holds 2 QQQ under cash, which never trades it: the position gets a zero '
            'target (sold)') in out
    assert out.count('!!') == 1                      # the parking ETF itself belongs there
    assert 'SELL      2 QQQ' in _orders(out)


# ============================================================== min-order
def _min_order_case(monkeypatch, tmp_path, equity='1000'):
    mod = _live(monkeypatch, _prices(SPY=100, OLD=20, SGOV=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.})
    ledger = _ledger(tmp_path, ['mean_reversion,OLD,1'])
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--equity', equity, '--as-of', DAY,
                                      '--ledger', str(ledger), '--min-order', '50'])
    return mod


def test_min_order_trims_cash_parking_instead_of_aborting_the_sheet(monkeypatch, tmp_path, capsys):
    mod = _min_order_case(monkeypatch, tmp_path)
    mod.main()
    out = capsys.readouterr().out
    # the $20 sale is suppressed, so one fewer SGOV share is parked: $20 + $900 + $3.25 <= $1,000
    assert 'cash parking after expense reserve: 9 sh SGOV' in out
    orders = _orders(out)
    assert 'OLD   skip -1 sh ($20 < min-order)' in orders
    assert 'BUY       9 SGOV' in orders and 'SELL' not in orders


def test_min_order_filter_judges_the_net_order_at_the_threshold(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, SGOV=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})         # two $100 legs net to one $200 order
    ledger = _ledger(tmp_path, [])
    for threshold, sent in (('100.01', True), ('200', True), ('200.01', False)):
        assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY, '--ledger', str(ledger),
                    '--min-order', threshold) == 0
        orders = _orders(capsys.readouterr().out)
        assert ('BUY       2 SPY   MOC' in orders) is sent
        assert ('SPY   skip +2 sh ($200 < min-order)' in orders) is (not sent)


def test_parking_that_cannot_cover_a_suppressed_sale_still_refuses(monkeypatch, tmp_path, capsys):
    mod = _live(monkeypatch, _prices(SPY=100, OLD=20, AGED=20))  # no cash ETF quote: nothing to trim
    _sleeves(monkeypatch, mod, {'SPY': 1.0}, monthly={'SPY': 1.0})   # $800 of SPY to buy
    ledger = _ledger(tmp_path, ['mean_reversion,OLD,20', 'cash,AGED,20'])   # two $400 sales fund it
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--equity', '1000', '--as-of', DAY,
                                      '--ledger', str(ledger), '--min-order', '450'])
    with pytest.raises(ValueError, match='orders are unfunded .*--min-order may be suppressing'):
        mod.main()
    assert 'MOC ORDERS' not in capsys.readouterr().out


# ======================================================= post-trade ledger
def _book(monkeypatch, tmp_path, day=DAY):
    mod = _live(monkeypatch, _prices(day, SPY=100, QQQ=50, OLD=20, SGOV=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.4})         # $500 a sleeve: two SPY in each daily sleeve
    return mod


def test_write_ledger_records_per_sleeve_positions_if_every_order_fills(monkeypatch, tmp_path, capsys):
    mod = _book(monkeypatch, tmp_path)
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,5', 'mean_reversion,OLD,1', 'tsmom_trend,QQQ,7',
                                'cash,SGOV,1'])
    before = ledger.read_text()
    out_path = tmp_path / 'after.csv'
    argv = ['--equity', '2000', '--as-of', DAY, '--ledger', str(ledger), '--min-order', '50',
            '--write-ledger', str(out_path)]
    assert _run(mod, monkeypatch, *argv) == 0
    out = capsys.readouterr().out
    orders = _orders(out)
    assert 'SELL      1 SPY' in orders and 'BUY      11 SGOV' in orders and 'OLD   skip -1 sh' in orders
    assert 'proposed post-trade ledger written to' in out and 'confirmed fills' in out
    assert ledger.read_text() == before                              # the input is never touched
    assert out_path.read_text().splitlines() == [
        'sleeve,ticker,shares,as_of',
        'mean_reversion,OLD,1,2024-01-02',       # its sale was below --min-order: still held
        'mean_reversion,SPY,2,2024-01-02',
        'seasonality_flows,SPY,2,2024-01-02',
        'tsmom_trend,QQQ,7,2024-01-02',          # HOLD sleeve carried
        'cash,SGOV,12,2024-01-02',
    ]
    # an existing file is never overwritten, and the refusal comes before anything is printed
    assert _run(mod, monkeypatch, *argv) == 2
    captured = capsys.readouterr()
    assert 'refuses to overwrite' in captured.err and captured.out == ''
    # the same session cannot be run again from the ledger it produced ...
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--equity', '2000', '--as-of', DAY,
                                      '--ledger', str(out_path)])
    with pytest.raises(ValueError, match='sent twice'):
        mod.main()
    capsys.readouterr()
    # ... and at the next session, with unchanged targets, it asks for nothing new
    nxt = _book(monkeypatch, tmp_path, '2024-01-03')
    assert _run(nxt, monkeypatch, '--equity', '2000', '--as-of', '2024-01-03', '--ledger', str(out_path),
                '--min-order', '50') == 0
    orders = _orders(capsys.readouterr().out)
    assert 'MOC   (' not in orders and 'OLD   skip -1 sh' in orders


def test_write_ledger_needs_a_ledger(monkeypatch, tmp_path, capsys):
    mod = _book(monkeypatch, tmp_path)
    assert _run(mod, monkeypatch, '--equity', '2000', '--as-of', DAY,
                '--write-ledger', str(tmp_path / 'after.csv')) == 2
    assert '--write-ledger needs --ledger' in capsys.readouterr().err
    assert not (tmp_path / 'after.csv').exists()


def test_write_ledger_into_a_missing_folder_is_refused_before_anything_is_printed(
        monkeypatch, tmp_path, capsys):
    mod = _book(monkeypatch, tmp_path)
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,5'])
    target = tmp_path / 'absent' / 'after.csv'
    assert _run(mod, monkeypatch, '--equity', '2000', '--as-of', DAY, '--ledger', str(ledger),
                '--write-ledger', str(target)) == 2
    captured = capsys.readouterr()
    assert '--write-ledger folder does not exist' in captured.err and captured.out == ''
    assert not target.parent.exists()


def test_a_ledger_file_that_cannot_be_written_stops_the_run_before_any_order_line(
        monkeypatch, tmp_path, capsys):
    # an order sheet must never be printed and then disowned by a late refusal
    mod = _book(monkeypatch, tmp_path)
    ledger = _ledger(tmp_path, ['mean_reversion,SPY,5'])
    target = tmp_path / 'after.csv'
    real = mod.post_trade_ledger

    def raced(*args):
        target.write_text('written by someone else after the argument check\n')
        return real(*args)

    monkeypatch.setattr(mod, 'post_trade_ledger', raced)
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--equity', '2000', '--as-of', DAY,
                                      '--ledger', str(ledger), '--write-ledger', str(target)])
    with pytest.raises(FileExistsError):
        mod.main()
    out = capsys.readouterr().out
    assert 'MOC ORDERS' not in out and 'MOC   (' not in out
    assert target.read_text().startswith('written by someone else')   # and it is not overwritten


def test_post_trade_ledger_keeps_unsent_tickers_at_their_current_per_sleeve_shares():
    mod = module('scripts/live_targets.py')
    ledger = pd.Series([4., 6., 2.], index=pd.MultiIndex.from_tuples(
        [('mean_reversion', 'SPY'), ('seasonality_flows', 'SPY'), ('cash', 'SGOV')],
        names=['sleeve', 'ticker']))
    targets = {'mean_reversion': {'SPY': 5.}, 'seasonality_flows': {'SPY': 6.}, 'cash': {'SGOV': 9.}}
    sent = mod.post_trade_ledger(targets, ledger, set(), pd.Timestamp(DAY))
    assert sent.values.tolist() == [['mean_reversion', 'SPY', 5, DAY], ['seasonality_flows', 'SPY', 6, DAY],
                                    ['cash', 'SGOV', 9, DAY]]
    unsent = mod.post_trade_ledger(targets, ledger, {'SPY'}, pd.Timestamp(DAY))
    assert unsent.values.tolist() == [['mean_reversion', 'SPY', 4, DAY], ['seasonality_flows', 'SPY', 6, DAY],
                                      ['cash', 'SGOV', 9, DAY]]


# ========================================================== alert channel
def test_only_the_projects_own_alerts_reach_the_operator_channel(monkeypatch, capsys):
    mod = _live(monkeypatch, _prices(SPY=100))
    _sleeves(monkeypatch, mod, {'SPY': 0.})

    def noisy(_):
        warnings.warn("The 'generic' unit for NumPy timedelta is deprecated", DeprecationWarning)
        warnings.warn('some library will change', FutureWarning)
        warnings.warn('invalid value encountered in divide', RuntimeWarning)
        warnings.warn('array creation is deprecated',      # a UserWarning subclass in NumPy
                      getattr(getattr(np, 'exceptions', np), 'VisibleDeprecationWarning',
                              PendingDeprecationWarning))
        warnings.warn('treating final data date 2024-03-28 as a month-end: verify the calendar')
        warnings.warn('treating final data date 2024-03-28 as a month-end: verify the calendar')
        return None, 'HOLD'

    monkeypatch.setattr(mod, 'sleeve_tsmom', noisy)
    monkeypatch.setattr(mod, 'sleeve_xsec_etf', noisy)
    with warnings.catch_warnings():
        warnings.simplefilter('error')               # nothing may leak past the recorder either
        assert _run(mod, monkeypatch, '--equity', '1000', '--as-of', DAY) == 0
    alerts = [line for line in capsys.readouterr().out.splitlines() if '!!' in line]
    assert alerts == ['  !! treating final data date 2024-03-28 as a month-end: verify the calendar']


def test_real_month_end_calendar_alert_is_shown_once(monkeypatch, capsys):
    # 28 March 2024 is the month's last session only because Good Friday is a holiday:
    # the calendar asks the operator to verify, and both monthly sleeves raise it
    probe = module('scripts/live_targets.py')
    px = _panel(probe, '2024-03-28')
    mod = _live(monkeypatch, px)
    assert _run(mod, monkeypatch, '--equity', '100000', '--as-of', '2024-03-28') == 0
    alerts = [line for line in capsys.readouterr().out.splitlines() if '!!' in line]
    assert len(alerts) == 1 and 'treating final data date 2024-03-28 as a month-end' in alerts[0]


# ============================================================= exit codes
def test_refusals_exit_two_and_crashes_exit_three(monkeypatch, capsys):
    mod = module('scripts/live_targets.py')

    def refuse():
        raise ValueError('orders are unfunded')

    monkeypatch.setattr(mod, 'main', refuse)
    with pytest.raises(SystemExit) as exc:
        mod.cli()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert 'LIVE TARGETS REFUSED: orders are unfunded' in err and 'nothing printed above may be traded' in err

    def missing_cache():
        raise FileNotFoundError('data/close.csv')

    monkeypatch.setattr(mod, 'main', missing_cache)
    with pytest.raises(SystemExit) as exc:
        mod.cli()
    assert exc.value.code == 2

    def crash():
        raise KeyError('SPY')

    monkeypatch.setattr(mod, 'main', crash)
    with pytest.raises(SystemExit) as exc:
        mod.cli()
    assert exc.value.code == 3
    err = capsys.readouterr().err
    assert 'LIVE TARGETS CRASHED' in err and 'KeyError' in err
    monkeypatch.setattr(mod, 'main', lambda: None)
    mod.cli()                                        # a printed sheet is exit 0


@pytest.mark.parametrize('error,code', [(ValueError('bad cache'), 2), (RuntimeError('boom'), 3)])
def test_the_script_entry_point_never_exits_one(monkeypatch, capsys, error, code):
    from qcore import data

    def fail():
        raise error

    monkeypatch.setattr(data, 'load_prices', fail)
    monkeypatch.setattr(sys, 'argv', ['live_targets.py', '--as-of', DAY])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / 'scripts' / 'live_targets.py'), run_name='__main__')
    assert exc.value.code == code
    assert 'MOC ORDERS' not in capsys.readouterr().out
