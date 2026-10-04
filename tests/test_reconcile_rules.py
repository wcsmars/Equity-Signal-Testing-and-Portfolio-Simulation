"""Fill-reconciliation verdicts pinned at their boundaries (synthetic fills only).

These pin what the execution check decides: signed slippage, the modeled
order cost, the notional-weighted breach gate at assumed + 2 bps, the
commission review trigger and its label, the lines that withhold "ok", and
the exit codes that tell a breach (1) from invalid input (2) from a crash (3).
"""
import importlib.util
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['date', 'ticker', 'side', 'shares', 'price', 'commission', 'reference_close']


def module(relative='scripts/reconcile.py'):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem + '_reconcile_rules_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def _fills(tmp_path, rows, name='fills.csv'):
    path = tmp_path / name
    _frame(rows).to_csv(path, index=False)
    return path


def _run(mod, monkeypatch, path, *extra, entry='main'):
    monkeypatch.setattr(sys, 'argv', ['reconcile.py', str(path), *extra])
    try:
        getattr(mod, entry)()
    except SystemExit as exc:
        return exc.code
    return 0


# ============================================================ measurement
def test_slippage_is_signed_bps_of_the_reference_close():
    mod = module()
    out = mod.reconcile_fills(_frame([
        ['2024-01-02', 'SPY', 'BUY', 10., 100.10, 1., 100.],     # paid 10 bps above
        ['2024-01-02', 'SPY', 'SELL', 10., 99.90, 1., 100.],     # sold 10 bps below
        ['2024-01-02', 'QQQ', 'BUY', 10., 399.60, 1., 400.],     # bought 10 bps BELOW: better
    ]))
    assert out.slip_bps.tolist() == pytest.approx([10.0, 10.0, -10.0], rel=1e-9)


def test_modeled_commission_minimum_rate_cap_and_sell_fees():
    mod = module()
    out = mod.reconcile_fills(_frame([
        ['2024-01-02', 'SPY', 'BUY', 10., 100., 1., 100.],        # $0.05 -> $1 minimum
        ['2024-01-02', 'QQQ', 'BUY', 1000., 100., 5., 100.],      # 1000 x $0.005
        ['2024-01-02', 'PNY', 'BUY', 1000., 0.20, 2., 0.20],      # $5 capped at 1% of $200
        ['2024-01-02', 'IWM', 'SELL', 100., 100., 1., 100.],      # $1 + SEC and FINRA sell fees
    ]))
    assert out.comm_model.tolist() == pytest.approx(
        [1.0, 5.0, 2.0, 1.0 + 10_000 * 0.0000278 + 100 * 0.000166])
    # the reserve used for live sizing models a sale the same way
    live = module('scripts/live_targets.py')
    assert out.comm_model.iloc[3] == pytest.approx(
        live.execution_reserve({'IWM': -100.0}, pd.Series({'IWM': 100.0}), 0.0))


def test_partial_fills_of_one_order_share_one_minimum_commission():
    mod = module()
    out = mod.reconcile_fills(_frame([
        ['2024-01-02', 'SPY', 'BUY', 10., 100., 0.25, 100.],
        ['2024-01-02', 'SPY', 'BUY', 30., 100., 0.75, 100.],      # same order, second partial
        ['2024-01-02', 'SPY', 'SELL', 10., 100., 1.0, 100.],      # the other side is another order
        ['2024-01-03', 'SPY', 'BUY', 10., 100., 1.0, 100.],       # another day is another order
    ]))
    assert out.comm_model.iloc[:2].tolist() == pytest.approx([0.25, 0.75])
    assert out.comm_model.iloc[3] == pytest.approx(1.0)
    summary = mod.summarize(out, 3.0)
    assert summary['orders'] == 3 and summary['fills'] == 4
    assert summary['comm_gap'] == pytest.approx(-(1000 * 0.0000278 + 10 * 0.000166))
    assert not summary['commission_excess'] and not summary['commission_review_trigger']


def test_summary_is_dollar_slippage_over_traded_notional():
    mod = module()
    out = mod.reconcile_fills(_frame([
        ['2024-01-02', 'SPY', 'BUY', 100., 100.10, 1., 100.],     # +$10 on $10,000
        ['2024-01-02', 'QQQ', 'SELL', 10., 50.05, 1., 50.],       # -$0.50 on $500 (sold above)
    ]))
    summary = mod.summarize(out, 3.0)
    assert summary['notional'] == pytest.approx(10_500.0)
    assert summary['slip_usd'] == pytest.approx(9.5)
    assert summary['slip_bps_weighted'] == pytest.approx(9.5 / 10_500 * 1e4)
    assert summary['slip_bps_mean'] == pytest.approx(0.0, abs=1e-9)
    assert summary['slippage_breach'] and not summary['implausibly_favourable']


# ================================================================== gates
def _row(bps, ticker='SPY', commission=1.0, shares=10.):
    return ['2024-01-02', ticker, 'BUY', shares, 100. * (1 + bps / 1e4), commission, 100.]


def test_breach_exit_code_at_assumed_plus_two_bps(monkeypatch, tmp_path, capsys):
    mod = module()
    assert mod.TOLERANCE_BPS == 2.0
    inside = _fills(tmp_path, [_row(4.9, 'SPY'), _row(5.0, 'QQQ')])       # 4.95 bps
    assert _run(mod, monkeypatch, inside) == 0
    out = capsys.readouterr().out
    assert 'ok: slippage and commissions within the modeled assumptions' in out and 'BREACH' not in out
    outside = _fills(tmp_path, [_row(5.0, 'SPY'), _row(5.2, 'QQQ')])      # 5.1 > 3 + 2
    assert _run(mod, monkeypatch, outside) == 1
    out = capsys.readouterr().out
    assert 'BREACH: notional-weighted slippage' in out and 'ok:' not in out
    # the modeled assumption is configurable per liquidity tier
    assert _run(mod, monkeypatch, outside, '--slippage-assumed', '5') == 0
    assert _run(mod, monkeypatch, inside, '--slippage-assumed', '2') == 1   # 4.95 > 2 + 2
    capsys.readouterr()


def test_a_large_bad_fill_is_not_diluted_by_small_good_ones(monkeypatch, tmp_path, capsys):
    mod = module()
    rows = [['2026-07-01', 'SPY', 'BUY', 1000., 746.5058, 5., 745.76]]            # +10 bps on $746k
    rows += [['2026-07-01', 'XLE', 'BUY', 1., 52.81, 0.11, 52.81] for _ in range(9)]   # at the close
    out = mod.reconcile_fills(_frame(rows))
    assert out.slip_bps.mean() == pytest.approx(1.0, abs=1e-3)       # the old gate saw 1 bp
    assert mod.summarize(out, 3.0)['slip_bps_weighted'] == pytest.approx(9.99, abs=0.01)
    assert _run(mod, monkeypatch, _fills(tmp_path, rows)) == 1
    assert 'BREACH: notional-weighted slippage' in capsys.readouterr().out


def test_one_odd_lot_cannot_breach_on_its_own(monkeypatch, tmp_path, capsys):
    mod = module()
    rows = [['2026-07-01', 'SPY', 'BUY', 1000., 745.76, 5., 745.76],
            ['2026-07-01', 'XLE', 'BUY', 1., 53.13, 0.53, 52.81]]                # +60 bps on $53
    out = mod.reconcile_fills(_frame(rows))
    assert out.slip_bps.mean() > 30.0                                # the old gate saw 30 bps
    assert _run(mod, monkeypatch, _fills(tmp_path, rows)) == 0
    assert 'ok: slippage and commissions' in capsys.readouterr().out


def test_verdict_does_not_depend_on_how_an_order_was_split(monkeypatch, tmp_path):
    mod = module()
    whole = [['2024-01-02', 'SPY', 'BUY', 1000., 100.06, 5., 100.]]
    split = [['2024-01-02', 'SPY', 'BUY', 1., 100., 0.005, 100.]] * 9 + \
            [['2024-01-02', 'SPY', 'BUY', 991., 100.060545, 4.955, 100.]]
    a, b = (mod.summarize(mod.reconcile_fills(_frame(r)), 3.0) for r in (whole, split))
    assert a['slip_bps_weighted'] == pytest.approx(b['slip_bps_weighted'], abs=1e-3)
    assert a['slippage_breach'] and b['slippage_breach'] and a['orders'] == b['orders'] == 1
    assert a['comm_gap'] == pytest.approx(0.0, abs=1e-9) and b['comm_gap'] == pytest.approx(0.0, abs=1e-9)


def test_commissions_far_above_the_model_exit_one_and_are_never_ok(monkeypatch, tmp_path, capsys):
    mod = module()
    rows = [['2026-07-01', 'SPY', 'BUY', 26., 745.76, 50., 745.76],
            ['2026-07-01', 'XLE', 'SELL', 129., 52.81, 50., 52.81]]              # all at the close
    assert _run(mod, monkeypatch, _fills(tmp_path, rows)) == 1
    out = capsys.readouterr().out
    assert 'COMMISSION REVIEW TRIGGER: commissions paid exceed the modeled commissions and fees' in out
    assert 'ok:' not in out and 'notional-weighted slippage exceeds' not in out


def test_commission_gate_is_two_bps_of_traded_notional(monkeypatch, tmp_path, capsys):
    mod = module()
    # $10,000 traded, modeled $1: 2 bps of notional is $2 above the model
    at = lambda paid: _fills(tmp_path, [['2024-01-02', 'SPY', 'BUY', 100., 100., paid, 100.]])
    assert _run(mod, monkeypatch, at(3.01)) == 1
    assert 'COMMISSION REVIEW TRIGGER: commissions' in capsys.readouterr().out
    assert _run(mod, monkeypatch, at(2.99)) == 0
    out = capsys.readouterr().out
    assert 'REVIEW: commissions paid exceed the model by $1.99' in out
    assert 'not confirmed' in out and 'ok:' not in out and 'BREACH' not in out
    assert _run(mod, monkeypatch, at(1.00)) == 0
    assert 'ok: slippage and commissions' in capsys.readouterr().out
    assert _run(mod, monkeypatch, at(0.35)) == 0                     # cheaper than modeled is fine
    assert 'ok: slippage and commissions' in capsys.readouterr().out


def test_commission_trigger_is_labelled_and_separate_from_the_registered_slippage_rule(
        monkeypatch, tmp_path, capsys):
    mod = module()
    assert (mod.TOLERANCE_BPS, mod.COMMISSION_REVIEW_BPS) == (2.0, 2.0)
    costly = _fills(tmp_path, [['2024-01-02', 'SPY', 'BUY', 100., 100., 3.01, 100.]], 'costly.csv')
    slipped = _fills(tmp_path, [_row(9.0)], 'slipped.csv')
    # the commission gate exits 1 under its own label: it is a review trigger,
    # and the word reserved for the pre-registered slippage rule is not printed
    assert _run(mod, monkeypatch, costly) == 1
    out = capsys.readouterr().out
    assert 'COMMISSION REVIEW TRIGGER' in out and 'not the pre-registered slippage rule' in out
    assert 'BREACH' not in out and 'ok:' not in out
    assert _run(mod, monkeypatch, slipped) == 1
    out = capsys.readouterr().out
    assert 'BREACH: notional-weighted slippage' in out and 'COMMISSION REVIEW TRIGGER' not in out
    assert 'review trigger' in mod.__doc__ and 'not a pre-registered' in mod.__doc__
    # each gate has its own constant: moving one never moves the other
    monkeypatch.setattr(mod, 'COMMISSION_REVIEW_BPS', 50.0)
    assert _run(mod, monkeypatch, costly) == 0
    assert 'REVIEW: commissions paid exceed the model by $2.01' in capsys.readouterr().out
    assert _run(mod, monkeypatch, slipped) == 1
    monkeypatch.setattr(mod, 'COMMISSION_REVIEW_BPS', 2.0)
    monkeypatch.setattr(mod, 'TOLERANCE_BPS', 50.0)
    assert _run(mod, monkeypatch, costly) == 1
    assert _run(mod, monkeypatch, slipped) == 0
    capsys.readouterr()


def test_fills_that_beat_the_close_implausibly_never_breach_but_are_not_ok(monkeypatch, tmp_path, capsys):
    mod = module()
    assert _run(mod, monkeypatch, _fills(tmp_path, [_row(-4.9)])) == 0
    assert 'ok: slippage and commissions' in capsys.readouterr().out
    # a sale "400 bps above the close" is a wrong side or reference, not good execution
    rows = [['2024-01-02', 'SPY', 'SELL', 10., 104., 1., 100.]]
    assert _run(mod, monkeypatch, _fills(tmp_path, rows)) == 0
    out = capsys.readouterr().out
    assert 'REVIEW: fills beat the official close by 400.00 bps' in out
    assert 'not confirmed' in out and 'ok:' not in out and 'BREACH' not in out


# ============================================================ input errors
def test_missing_or_unreadable_fills_file_is_invalid_input_not_a_breach(monkeypatch, tmp_path, capsys):
    mod = module()
    assert _run(mod, monkeypatch, tmp_path / 'state' / 'fills.csv') == 2
    out = capsys.readouterr().out
    assert 'RECONCILIATION FAILED: cannot read fills file' in out and 'Traceback' not in out
    assert _run(mod, monkeypatch, tmp_path) == 2                     # a directory
    assert 'RECONCILIATION FAILED' in capsys.readouterr().out
    empty = tmp_path / 'empty.csv'
    empty.write_text('')
    assert _run(mod, monkeypatch, empty) == 2
    assert 'nonempty unique header' in capsys.readouterr().out
    header_only = tmp_path / 'header.csv'
    header_only.write_text(','.join(COLUMNS) + '\n')
    assert _run(mod, monkeypatch, header_only) == 2
    assert 'fills CSV is empty' in capsys.readouterr().out
    duplicate = tmp_path / 'duplicate.csv'
    duplicate.write_text('date,ticker,side,shares,price,price,commission,reference_close\n'
                         '2024-01-02,SPY,BUY,10,100,90,1,100\n')
    assert _run(mod, monkeypatch, duplicate) == 2
    assert 'nonempty unique header' in capsys.readouterr().out
    ragged = tmp_path / 'ragged.csv'
    ragged.write_text(','.join(COLUMNS) + '\n2024-01-02,SPY,BUY,10,100,1,100\n'
                      '2024-01-02,SPY,BUY,10,100,1,100,7,8,9\n')
    assert _run(mod, monkeypatch, ragged) == 2
    assert 'RECONCILIATION FAILED' in capsys.readouterr().out
    binary = tmp_path / 'binary.csv'
    binary.write_bytes(b'\xff\xfe\x00\x00date,ticker\n')
    assert _run(mod, monkeypatch, binary) == 2
    assert 'RECONCILIATION FAILED' in capsys.readouterr().out


def test_mixed_utc_offsets_are_invalid_input_not_a_crash(monkeypatch, tmp_path, capsys):
    mod = module()
    rows = [['2026-07-01T00:00:00+01:00', 'SPY', 'BUY', 26., 745.76, 1., 745.76],
            ['2026-07-01T00:00:00-04:00', 'SPY', 'BUY', 26., 745.76, 1., 745.76]]
    with pytest.raises(ValueError):
        mod.reconcile_fills(_frame(rows))
    assert _run(mod, monkeypatch, _fills(tmp_path, rows), entry='cli') == 2
    assert 'RECONCILIATION FAILED' in capsys.readouterr().out
    aware = [['2026-07-01T00:00:00-04:00', 'SPY', 'BUY', 26., 745.76, 1., 745.76]]
    with pytest.raises(ValueError, match='timezone-naive'):
        mod.reconcile_fills(_frame(aware))
    timed = [['2026-07-01 15:59:00', 'SPY', 'BUY', 26., 745.76, 1., 745.76]]
    with pytest.raises(ValueError, match='without time-of-day'):
        mod.reconcile_fills(_frame(timed))


def test_broker_yyyymmdd_dates_are_read_as_dates(monkeypatch, tmp_path, capsys):
    mod = module()
    row = lambda date: [date, 'SPY', 'BUY', 26., 745.76, 1., 745.76]
    assert str(mod.reconcile_fills(_frame([row(20260701)])).date.iloc[0]) == '2026-07-01'
    assert _run(mod, monkeypatch, _fills(tmp_path, [row(20260701)])) == 0   # read back as an integer
    assert '2026-07-01' in capsys.readouterr().out
    for bad in (2026070, 20261301, 20260701.5, float('nan'), -20260701, True):
        with pytest.raises(ValueError, match='fill dates|yyyyMMdd'):
            mod.reconcile_fills(_frame([row(bad)]))


def test_header_spacing_is_tolerated(monkeypatch, tmp_path, capsys):
    mod = module()
    path = tmp_path / 'spaced.csv'
    path.write_text('date, ticker, side, shares, price, commission, reference_close\n'
                    '2024-01-02, spy, buy, 10, 100, 1, 100\n')
    assert _run(mod, monkeypatch, path) == 0
    assert 'ok: slippage and commissions' in capsys.readouterr().out


# ============================================================= exit codes
def test_exit_codes_tell_breach_from_invalid_input_from_crash(monkeypatch, tmp_path, capsys):
    mod = module()
    assert (mod.EXIT_BREACH, mod.EXIT_INVALID, mod.EXIT_CRASH) == (1, 2, 3)
    good = _fills(tmp_path, [_row(0.0)], 'good.csv')
    breach = _fills(tmp_path, [_row(9.0)], 'breach.csv')
    invalid = _fills(tmp_path, [['2024-01-02', 'SPY', 'HOLD', 10., 100., 1., 100.]], 'invalid.csv')
    assert _run(mod, monkeypatch, good, entry='cli') == 0
    assert _run(mod, monkeypatch, breach, entry='cli') == 1
    assert _run(mod, monkeypatch, invalid, entry='cli') == 2
    capsys.readouterr()

    def crash(_):
        raise RuntimeError('unexpected')

    monkeypatch.setattr(mod, 'reconcile_fills', crash)
    assert _run(mod, monkeypatch, good, entry='cli') == 3
    captured = capsys.readouterr()
    assert 'RECONCILIATION CRASHED: no verdict' in captured.out and 'RuntimeError' in captured.err
    assert 'BREACH' not in captured.out and 'ok:' not in captured.out


@pytest.mark.parametrize('case,code', [('missing', 2), ('breach', 1), ('ok', 0), ('crash', 3)])
def test_the_script_entry_point_uses_those_codes(monkeypatch, tmp_path, capsys, case, code):
    path = {'missing': tmp_path / 'absent.csv',
            'breach': _fills(tmp_path, [_row(9.0)], 'breach.csv'),
            'ok': _fills(tmp_path, [_row(0.0)], 'ok.csv'),
            'crash': _fills(tmp_path, [_row(0.0)], 'crash.csv')}[case]
    if case == 'crash':
        def boom(*args, **kwargs):
            raise RuntimeError('unexpected')
        monkeypatch.setattr(pd, 'read_csv', boom)
    monkeypatch.setattr(sys, 'argv', ['reconcile.py', str(path)])
    try:
        runpy.run_path(str(ROOT / 'scripts' / 'reconcile.py'), run_name='__main__')
        exit_code = 0
    except SystemExit as exc:
        exit_code = exc.code
    assert exit_code == code
    capsys.readouterr()
