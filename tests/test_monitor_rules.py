"""The registered kill rules: pinned thresholds, rule boundaries, the judged
window, the cash benchmark and the exit status."""
import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASH_DAILY = 0.0002  # about 5% a year, so excess and total return clearly differ


def load_monitor():
    spec = importlib.util.spec_from_file_location("monitor_rules_test", ROOT / "scripts/monitor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patched(monkeypatch, cash):
    mod = load_monitor()
    monkeypatch.setattr(mod, "cash_benchmark", lambda idx: pd.Series(cash, index=idx))
    monkeypatch.setattr(mod, "benchmark_span", lambda: "patched")
    return mod


@pytest.fixture
def monitor(monkeypatch):
    return patched(monkeypatch, CASH_DAILY)


@pytest.fixture
def flat_cash_monitor(monkeypatch):
    return patched(monkeypatch, 0.0)


def run_all(mod, monkeypatch, capsys, tmp_path, columns, index, *extra):
    """main() on a live file whose last row is today; returns (exit status, stdout, stderr)."""
    frame = pd.DataFrame(columns, index=index)
    frame.index.name = "Date"
    path = tmp_path / "live.csv"
    frame.to_csv(path)
    monkeypatch.setattr(sys, "argv", ["monitor.py", str(path), "--as-of", str(index[-1].date()), *extra])
    try:
        mod.main()
        code = 0
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run(*args):
    return run_all(*args)[:2]


# ------------------------------------------------------------------ thresholds
def test_registered_thresholds_are_pinned(monitor):
    # Registered before live trading (the live kill rules: drawdown -15% /
    # 2y Sharpe -0.30 / underwater 32-42 months). Moving one is a
    # re-registration, itself a KILL-level event once live: change this test
    # only together with that decision.
    assert (monitor.KILL_MAX_DRAWDOWN, monitor.KILL_SHARPE_2Y, monitor.REVIEW_UNDERWATER_M,
            monitor.KILL_UNDERWATER_M, monitor.WINDOW_2Y) == (-0.15, -0.30, 32, 42, 504)


# -------------------------------------------------------------------- drawdown
@pytest.mark.parametrize("loss, status", [(-0.149, "ok"), (-0.151, "KILL")])
def test_drawdown_boundary_is_measured_from_the_running_peak(monitor, loss, status):
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.0, 0.10, loss, 0.0], index=idx)  # only -6% .. -7% from the start
    assert monitor.evaluate(r, since=idx[-1])[0] == (
        "current drawdown", f"{loss:.1%} (kill < -15%)", status)


def test_a_reverted_drawdown_breach_is_reported(monitor):
    idx = pd.bdate_range("2026-07-01", periods=30)
    r = pd.Series(0.0, index=idx)
    r.iloc[11] = -0.16
    r.iloc[12] = 0.86 / 0.84 - 1  # back to -14% the next session
    assert monitor.evaluate(r, since=idx[-1])[0] == (
        "current drawdown", "-14.0% (kill < -15%)", "ok")
    assert monitor.evaluate(r)[0] == (
        "current drawdown",
        f"-14.0% (kill < -15%); breached on 1 session(s), first {idx[11].date()}, "
        f"worst -16.0% on {idx[11].date()}",
        "KILL")
    assert monitor.evaluate(r, since=idx[11])[0][2] == "KILL"  # the window is inclusive
    assert monitor.evaluate(r, since=idx[12])[0][2] == "ok"
    with pytest.raises(ValueError, match="no session on or after"):
        monitor.evaluate(r, since=idx[-1] + pd.offsets.BDay(1))


def test_breach_history_names_the_first_and_the_worst_session(monitor):
    idx = pd.bdate_range("2026-07-01", periods=6)
    r = pd.Series([0.0, -0.16, -0.05, 0.0, 0.30, 0.0], index=idx)
    detail = monitor.evaluate(r)[0][1]
    assert detail == (f"0.0% (kill < -15%); breached on 3 session(s), first {idx[1].date()}, "
                      f"worst -20.2% on {idx[2].date()}")


# ---------------------------------------------------------------------- sharpe
def sharpe_block(target, n=504, amp=0.001):
    """n excess returns alternating +/- amp around a mean chosen so that the
    annualised Sharpe (sample standard deviation) is exactly `target`."""
    mean = target * amp * np.sqrt(n / (n - 1)) / np.sqrt(252)
    return mean + np.tile([amp, -amp], n // 2)


@pytest.mark.parametrize("target, status", [(-0.29, "ok"), (-0.31, "KILL")])
def test_two_year_sharpe_is_annualised_and_in_excess_of_cash(monitor, target, status):
    great_past = np.full(300, 0.002)  # outside the last 504-session window
    r = pd.Series(np.concatenate([great_past, sharpe_block(target) + CASH_DAILY]),
                  index=pd.bdate_range("2018-01-01", periods=804))
    # The total return's own Sharpe is about +2.9: only the excess over cash
    # is near the line, and only when annualised.
    assert monitor.evaluate(r, since=r.index[-1])[1] == (
        "rolling 2y excess Sharpe", f"{target:.2f} (kill < -0.3)", status)
    assert monitor.evaluate(r)[1][2] == status  # no earlier window is worse


def test_a_reverted_sharpe_breach_is_reported_and_needs_the_full_window(monitor):
    losing = np.tile([-0.001, 0.0001], 252)
    r = pd.Series(np.concatenate([losing, np.full(400, 0.002)]),
                  index=pd.bdate_range("2020-01-01", periods=904))
    assert monitor.evaluate(r, since=r.index[-1])[1][2] == "ok"
    _, detail, status = monitor.evaluate(r)[1]
    assert status == "KILL"
    assert f"first {r.index[503].date()}" in detail  # the first complete window
    assert monitor.evaluate(r.iloc[:503])[1][1:] == ("pending: 503/504 observations", "ok")


def test_sharpe_of_a_constant_excess_is_signed_infinity(flat_cash_monitor):
    idx = pd.bdate_range("2020-01-01", periods=504)

    def sharpe_row(level):  # binary fractions: the window's standard deviation is exactly zero
        return flat_cash_monitor.evaluate(pd.Series(level, index=idx), since=idx[-1])[1][1:]

    assert sharpe_row(-0.25) == ("-inf (kill < -0.3)", "KILL")
    assert sharpe_row(0.25) == ("inf (kill < -0.3)", "ok")
    assert sharpe_row(0.0) == ("nan (kill < -0.3)", "ok")  # exactly cash: undefined, not a breach


# ------------------------------------------------------------------ underwater
def spell(days):
    """Three observations: a peak, a dip the next day, and one `days` calendar days after the peak."""
    return pd.date_range("2020-01-01", periods=days + 1)[[0, 1, -1]]


@pytest.mark.parametrize("days, months, status", [
    (971, "31.9", "ok"), (974, "32.0", "ok"),            # 974 / 30.44 = 31.997
    (975, "32.0", "REVIEW"), (1278, "42.0", "REVIEW"),   # 1278 / 30.44 = 41.984
    (1279, "42.0", "KILL"), (1282, "42.1", "KILL"),
])
def test_underwater_reviews_at_32_months_and_kills_at_42(monitor, days, months, status):
    idx = spell(days)
    rows = monitor.evaluate(pd.Series([0.01, -0.001, 0.0], index=idx), since=idx[-1])
    assert rows[2] == ("months underwater", f"{months} (review >= 32, kill >= 42)", status)
    assert rows[0][2] == "ok"
    assert rows[1][1:] == ("pending: 3/504 observations", "ok")
    # A new high ends the spell.
    recovered = monitor.evaluate(pd.Series([0.01, -0.001, 0.002], index=idx), since=idx[-1])
    assert recovered[2] == ("months underwater", "0.0 (review >= 32, kill >= 42)", "ok")


def test_underwater_clock_restarts_at_the_latest_peak(monitor):
    idx = pd.date_range("2020-01-01", periods=1201)[[0, 1, 500, 501, 1200]]
    r = pd.Series([0.01, -0.001, 0.05, -0.001, 0.0], index=idx)
    # 700 days since the second peak (23.0 months), not 1,200 since the first (39.4).
    assert monitor.evaluate(r)[2] == ("months underwater", "23.0 (review >= 32, kill >= 42)", "ok")


def test_a_reverted_underwater_review_is_reported(monitor):
    idx = pd.bdate_range("2020-01-01", periods=760)
    r = pd.Series(0.0, index=idx)
    r.iloc[2] = -0.05
    r.iloc[741] = 0.10  # a new high after about 34 months
    assert monitor.evaluate(r, since=idx[-1])[2][2] == "ok"
    _, detail, status = monitor.evaluate(r)[2]
    assert status == "REVIEW"
    worst = (idx[740] - idx[1]).days / 30.44  # the last session before the new high
    assert worst >= 32
    first = idx[(idx - idx[1]).days >= 32 * 30.44][0]
    assert detail == ("0.0 (review >= 32, kill >= 42); breached on "
                      f"{741 - idx.get_loc(first)} session(s), first {first.date()}, "
                      f"worst {worst:.1f} on {idx[740].date()}")


# ---------------------------------------------------------------- exit status
HEALTHY, CRASHED = [0.01, -0.001, 0.0], [0.0, -0.2, 0.0]


def test_exit_statuses_are_distinct_and_keep_the_documented_meaning(monitor):
    # 0 / 1 / 2 are the ok / REVIEW / KILL contract shared with scripts/data_quality.py.
    assert (monitor.EXIT_OK, monitor.EXIT_REVIEW, monitor.EXIT_KILL) == (0, 1, 2)
    assert (monitor.EXIT_INVALID_INPUT, monitor.EXIT_CRASH) == (3, 4)


@pytest.mark.parametrize("columns, code, overall", [
    ({"tsmom_trend": HEALTHY, "ENSEMBLE": HEALTHY}, 0, "ok"),
    ({"tsmom_trend": HEALTHY, "ENSEMBLE": CRASHED}, 2, "KILL"),
    # Only the ENSEMBLE column is registered: a sleeve breach is information.
    ({"tsmom_trend": CRASHED, "ENSEMBLE": HEALTHY}, 0, "ok"),
    # Without an ENSEMBLE column every column is judged.
    ({"tsmom_trend": CRASHED, "xsec_etf_mom": HEALTHY}, 2, "KILL"),
])
def test_exit_status_follows_the_ensemble_column(monitor, monkeypatch, capsys, tmp_path,
                                                 columns, code, overall):
    index = monitor.nyse_bdays("2024-01-02", "2024-01-04")
    got, out = run(monitor, monkeypatch, capsys, tmp_path, columns, index)
    assert got == code
    assert out.rstrip().endswith(f"overall: {overall}")
    assert "mode: LIVE account" in out
    assert ("KILL (info)" in out) == ("ENSEMBLE" in columns and columns["tsmom_trend"] is CRASHED)


@pytest.mark.parametrize("end, code, overall", [
    ("2022-09-01", 0, "ok"),       # 973 days after the 2020-01-02 peak
    ("2022-09-06", 1, "REVIEW"),   # 978 days: 32.1 months
    ("2023-07-05", 2, "KILL"),     # 1280 days: 42.0 months
])
def test_underwater_spell_sets_review_then_kill_exit(flat_cash_monitor, monkeypatch, capsys,
                                                    tmp_path, end, code, overall):
    index = flat_cash_monitor.nyse_bdays("2020-01-02", end)
    # A peak, a dip, then noise that never regains the peak. The noise keeps
    # every 2y window's Sharpe near zero, so only the underwater rule can fire.
    returns = np.resize([0.0005, -0.0005], len(index))
    returns[:2] = [0.01, -0.001]
    got, out = run(flat_cash_monitor, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert (got, out.rstrip().splitlines()[-1]) == (code, f"overall: {overall}")


def test_live_file_is_judged_on_every_session_unless_since_narrows_it(monitor, monkeypatch,
                                                                       capsys, tmp_path):
    index = monitor.nyse_bdays("2024-01-02", "2024-01-12")
    returns = np.zeros(len(index))
    returns[2], returns[3] = -0.16, 0.86 / 0.84 - 1  # 2024-01-04 breach, reverted 2024-01-05
    code, out = run(monitor, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == 2
    assert f"judged: {len(index)} of {len(index)} session(s)" in out
    assert "first 2024-01-04, worst -16.0% on 2024-01-04" in out
    assert out.rstrip().endswith("overall: KILL")
    code, out = run(monitor, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index,
                    "--since", "2024-01-05")
    assert code == 0
    assert f"judged: {len(index) - 3} of {len(index)} session(s), 2024-01-05 .. 2024-01-12" in out
    assert out.rstrip().endswith("overall: ok")


def rule_row(out, column, rule):
    return next(line for line in out.splitlines() if column in line and rule in line)


def test_both_the_last_session_and_the_window_reading_are_printed(monitor, monkeypatch, capsys,
                                                                   tmp_path):
    index = monitor.nyse_bdays("2024-01-02", "2024-01-12")
    quiet = np.zeros(len(index))
    reverted, standing = quiet.copy(), quiet.copy()
    reverted[2], reverted[3] = -0.16, 0.86 / 0.84 - 1  # 2024-01-04 breach, -14% from 2024-01-05
    standing[2] = -0.16                                # still -16% at the last row
    window = "on any judged session (2024-01-02 .. 2024-01-12)"
    verdict = "<- verdict and exit status"

    # A reverted breach: the two readings differ and each is labelled.
    code, out = run(monitor, monkeypatch, capsys, tmp_path,
                    {"tsmom_trend": reverted, "ENSEMBLE": reverted}, index)
    assert code == 2
    assert out.rstrip().splitlines()[-3:] == [
        "at the last session (2024-01-12): ok", f"{window}: KILL  {verdict}", "overall: KILL"]
    assert rule_row(out, "ENSEMBLE", "current drawdown").endswith("KILL (last session ok)")
    assert rule_row(out, "tsmom_trend", "current drawdown").endswith("KILL (last session ok, info)")

    # A standing breach: both readings agree and no row carries the extra tag.
    code, out = run(monitor, monkeypatch, capsys, tmp_path, {"ENSEMBLE": standing}, index)
    assert code == 2
    assert out.rstrip().splitlines()[-3:] == [
        "at the last session (2024-01-12): KILL", f"{window}: KILL  {verdict}", "overall: KILL"]
    assert rule_row(out, "ENSEMBLE", "current drawdown").endswith(" KILL")
    assert "(last session" not in out

    # --since at the last row gives the last-row-only verdict.
    code, out = run(monitor, monkeypatch, capsys, tmp_path, {"ENSEMBLE": reverted}, index,
                    "--since", "2024-01-12")
    assert code == 0
    assert out.rstrip().splitlines()[-3:] == [
        "at the last session (2024-01-12): ok",
        f"on any judged session (2024-01-12 .. 2024-01-12): ok  {verdict}", "overall: ok"]

    # A sleeve's reverted breach is information: neither reading of the verdict moves.
    code, out = run(monitor, monkeypatch, capsys, tmp_path,
                    {"tsmom_trend": reverted, "ENSEMBLE": quiet}, index)
    assert code == 0
    assert out.rstrip().splitlines()[-3:] == [
        "at the last session (2024-01-12): ok", f"{window}: ok  {verdict}", "overall: ok"]
    assert rule_row(out, "tsmom_trend", "current drawdown").endswith("KILL (last session ok, info)")


def test_a_reverted_review_keeps_the_review_exit_and_shows_the_current_state(
        flat_cash_monitor, monkeypatch, capsys, tmp_path):
    index = flat_cash_monitor.nyse_bdays("2020-01-02", "2022-09-07")
    returns = np.resize([0.0005, -0.0005], len(index))
    returns[:2] = [0.01, -0.001]
    returns[-1] = 0.05  # a new high the session after the spell reached 32.1 months
    code, out = run(flat_cash_monitor, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == 1
    assert out.rstrip().splitlines()[-3:] == [
        "at the last session (2022-09-07): ok",
        "on any judged session (2020-01-02 .. 2022-09-07): REVIEW  <- verdict and exit status",
        "overall: REVIEW"]
    row = rule_row(out, "ENSEMBLE", "months underwater")
    assert "breached on 1 session(s), first 2022-09-06, worst 32.1 on 2022-09-06" in row
    assert row.endswith("REVIEW (last session ok)")


def test_an_unexpected_error_is_a_crash_status_not_review(monitor, monkeypatch, capsys, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(monitor, "evaluate", boom)
    index = monitor.nyse_bdays("2024-01-02", "2024-01-04")
    code, out, err = run_all(monitor, monkeypatch, capsys, tmp_path, {"ENSEMBLE": HEALTHY}, index)
    assert code == monitor.EXIT_CRASH == 4
    assert "no verdict: RuntimeError: boom" in out
    assert "overall" not in out
    assert "Traceback" in err


# -------------------------------------------------------------- cash benchmark
def rated_monitor(monkeypatch, irx):
    """The real benchmark code over an injected ^IRX series (annual %)."""
    mod = load_monitor()
    monkeypatch.setattr(sys.modules["qcore.backtest"], "_IRX_CACHE", [irx])
    return mod


def two_year_file(mod):
    """504 sessions earning 3% a year: a positive total-return Sharpe (about
    +3.8) that loses clearly to 5% cash."""
    index = mod.nyse_bdays("2021-01-04", "2023-12-29")[:504]
    returns = 0.03 / 252 + np.tile([0.0005, -0.0005], 252)
    return index, returns


def full_irx(mod, index, level=5.0):
    return pd.Series(level, index=mod.nyse_bdays("2020-12-01", index[-1]))


def test_two_year_sharpe_is_scored_against_the_cash_rate(monkeypatch, capsys, tmp_path):
    mod = load_monitor()
    index, returns = two_year_file(mod)
    mod = rated_monitor(monkeypatch, full_irx(mod, index))
    code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    cash = 1.049 ** (1 / 252) - 1  # 5.00% less the 10bp haircut
    expected = (0.03 / 252 - cash) / (0.0005 * np.sqrt(504 / 503)) * np.sqrt(252)
    assert expected < -2
    assert code == 2
    assert f"cash benchmark: ^IRX 2020-12-01 .. {index[-1].date()} less 10bp" in out
    assert f"{expected:.2f} (kill < -0.3)" in out
    assert out.rstrip().endswith("overall: KILL")


@pytest.mark.parametrize("damage, message", [
    (lambda irx: irx.iloc[:0], "cash benchmark unavailable"),
    (lambda irx: irx.iloc[:-30], "cash benchmark does not cover the returns"),  # stale cache
    (lambda irx: irx.loc["2021-02-01":], "cash benchmark does not cover the returns"),
    (lambda irx: irx.drop(irx.index[300:322]), "cash benchmark does not cover the returns"),
])
def test_missing_or_stale_cash_rate_is_an_input_error_not_a_verdict(monkeypatch, capsys, tmp_path,
                                                                    damage, message):
    mod = load_monitor()
    index, returns = two_year_file(mod)
    mod = rated_monitor(monkeypatch, damage(full_irx(mod, index)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the engine's 0% fallback warning must not be the only signal
        code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == mod.EXIT_INVALID_INPUT == 3
    assert "invalid monitor input" in out and message in out
    assert "overall" not in out and "kill < -0.3" not in out


def test_unpriced_first_rate_date_names_the_first_fully_priced_session(monkeypatch, capsys,
                                                                       tmp_path):
    # As in the shipped cache and backtest stream: the rate history starts on
    # the first return date, which therefore has no earlier print.
    mod = load_monitor()
    index = mod.nyse_bdays("2021-01-04", "2023-12-29")[:510]
    returns = 0.03 / 252 + np.tile([0.0005, -0.0005], 255)
    mod = rated_monitor(monkeypatch, pd.Series(5.0, index=index))
    code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == 3 and "overall" not in out
    assert f"1 of 510 dates in the judged 2y windows (first {index[0].date()})" in out
    assert f"--since {index[504].date()} or later" in out
    # One session earlier still needs the unpriced date; the named one does not.
    code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index,
                    "--since", str(index[503].date()))
    assert code == 3 and f"--since {index[504].date()} or later" in out
    code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index,
                    "--since", str(index[504].date()))
    assert code == 2 and out.rstrip().endswith("overall: KILL")

    # A stale cache prices no later window: --since cannot help and is not offered.
    stale = rated_monitor(monkeypatch, pd.Series(5.0, index=mod.nyse_bdays("2020-12-01", index[-40])))
    code, out = run(stale, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == 3 and "(refresh data/indices.csv)" in out and "--since" not in out

    # A hole in the history: the named session is counted from its LAST unpriced
    # date (index[32], the first return date after the missing prints).
    index = mod.nyse_bdays("2021-01-04", "2023-12-29")[:560]
    returns = 0.03 / 252 + np.tile([0.0005, -0.0005], 280)
    prints = pd.Series(5.0, index=mod.nyse_bdays("2020-12-01", index[-1])).drop(index[10:32])
    holed = rated_monitor(monkeypatch, prints)
    code, out = run(holed, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == 3 and f"--since {index[536].date()} or later" in out
    code, out = run(holed, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index,
                    "--since", str(index[536].date()))
    assert code == 2 and out.rstrip().endswith("overall: KILL")


def test_ordinary_gaps_in_the_cash_rate_are_tolerated(monkeypatch, capsys, tmp_path):
    mod = load_monitor()
    index, returns = two_year_file(mod)
    irx = full_irx(mod, index)
    # Thursday, Friday and Monday without a print: Tuesday is priced off a
    # six-day-old Wednesday print, inside MAX_RATE_AGE_DAYS = 7.
    irx = irx.drop(pd.to_datetime(["2021-03-11", "2021-03-12", "2021-03-15", "2022-06-01"]))
    mod = rated_monitor(monkeypatch, irx)
    assert mod.MAX_RATE_AGE_DAYS == 7
    code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == 2 and out.rstrip().endswith("overall: KILL")


def test_pending_sharpe_window_does_not_need_the_cash_rate(monkeypatch, capsys, tmp_path):
    mod = rated_monitor(monkeypatch, pd.Series(dtype=float))
    index = mod.nyse_bdays("2024-01-02", "2024-12-31")[:100]
    code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": np.full(100, 0.0001)}, index)
    assert code == 0
    assert "cash benchmark: not used yet (2y Sharpe pending)" in out
    assert "pending: 100/504 observations" in out
    assert out.rstrip().endswith("overall: ok")


def test_unreadable_cash_rate_is_an_input_error(monkeypatch, capsys, tmp_path):
    mod = load_monitor()
    index, returns = two_year_file(mod)

    def denied():
        raise PermissionError("indices.csv")
    monkeypatch.setattr(mod, "_irx_series", denied)
    code, out = run(mod, monkeypatch, capsys, tmp_path, {"ENSEMBLE": returns}, index)
    assert code == mod.EXIT_INVALID_INPUT
    assert "cash benchmark unreadable" in out and "overall" not in out


def test_cash_benchmark_matches_the_engine_rate_when_covered(monkeypatch):
    mod = load_monitor()
    index, _ = two_year_file(mod)
    mod = rated_monitor(monkeypatch, full_irx(mod, index, level=3.1))
    rf = mod.cash_benchmark(index)
    assert rf.index.equals(index)
    np.testing.assert_allclose(rf.to_numpy(), 1.03 ** (1 / 252) - 1, rtol=0, atol=1e-15)
