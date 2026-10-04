"""Saved result records are never replaced silently (qcore.records)."""
import json

import pandas as pd
import pytest

from qcore import records


def test_new_record_is_written(tmp_path):
    target = tmp_path / "results" / "a.json"
    assert records.save_record(target, "one", rebase=False, quiet=True) == target
    assert target.read_text() == "one"


def test_identical_record_is_left_alone(tmp_path):
    target = tmp_path / "a.json"
    target.write_text("same")
    before = target.stat().st_mtime_ns
    assert records.save_record(target, "same", rebase=False, quiet=True) == target
    assert target.stat().st_mtime_ns == before
    assert not (tmp_path / records.RECOMPUTED_DIR).exists()


def test_differing_record_is_kept_and_new_output_goes_to_recomputed(tmp_path, capsys):
    target = tmp_path / "a.json"
    target.write_text("old")
    written = records.save_record(target, "new", rebase=False)
    assert target.read_text() == "old"
    assert written == tmp_path / records.RECOMPUTED_DIR / "a.json"
    assert written.read_text() == "new"
    message = capsys.readouterr().out
    assert "kept existing record" in message and records.REBASE_FLAG in message
    # a later differing run replaces only the scratch copy
    assert records.save_record(target, "newer", rebase=False, quiet=True) == written
    assert written.read_text() == "newer" and target.read_text() == "old"


def test_rebase_replaces_the_record(tmp_path):
    target = tmp_path / "a.json"
    target.write_text("old")
    assert records.save_record(target, "new", rebase=True, quiet=True) == target
    assert target.read_text() == "new"
    assert not list(tmp_path.glob(".a.json.tmp-*"))


def test_rebase_comes_from_flag_or_environment(monkeypatch):
    monkeypatch.delenv(records.REBASE_ENV, raising=False)
    assert not records.rebase_requested([])
    assert not records.rebase_requested(["--sweep"])
    assert records.rebase_requested(["--sweep", records.REBASE_FLAG])
    monkeypatch.setenv(records.REBASE_ENV, "1")
    assert records.rebase_requested([])
    monkeypatch.setenv(records.REBASE_ENV, "0")
    assert not records.rebase_requested([])


def test_default_follows_the_environment(tmp_path, monkeypatch):
    target = tmp_path / "a.json"
    target.write_text("old")
    monkeypatch.setattr(records.sys, "argv", ["prog"])
    monkeypatch.delenv(records.REBASE_ENV, raising=False)
    assert records.save_record(target, "new", quiet=True) != target
    monkeypatch.setenv(records.REBASE_ENV, "1")
    assert records.save_record(target, "new", quiet=True) == target
    assert target.read_text() == "new"


def test_symlink_target_is_refused(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("x")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="non-regular"):
        records.save_record(link, "y", rebase=True, quiet=True)
    assert real.read_text() == "x"


def test_directory_target_is_refused(tmp_path):
    target = tmp_path / "a.json"
    target.mkdir()
    with pytest.raises(ValueError, match="non-regular"):
        records.save_record(target, "y", rebase=True, quiet=True)
    assert target.is_dir()


def test_failed_write_leaves_the_record_and_no_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "a.json"
    target.write_text("old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(records.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        records.save_record(target, "new", rebase=True, quiet=True)
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_record_is_replaced_by_rename_not_rewritten_in_place(tmp_path, monkeypatch):
    target = tmp_path / "a.json"
    target.write_text("old")
    calls = []
    real = records.os.replace
    monkeypatch.setattr(records.os, "replace", lambda s, d: (calls.append((s, d)), real(s, d))[1])
    records.save_record(target, "new", rebase=True, quiet=True)
    assert len(calls) == 1 and calls[0][1] == target and calls[0][0] != target
    assert target.read_text() == "new"


def test_scratch_copy_is_removed_once_the_record_matches_or_is_rebased(tmp_path):
    target = tmp_path / "a.json"
    other = tmp_path / "b.json"
    target.write_text("old")
    other.write_text("old")
    scratch = tmp_path / records.RECOMPUTED_DIR
    records.save_record(target, "new", rebase=False, quiet=True)
    records.save_record(other, "new", rebase=False, quiet=True)
    assert sorted(p.name for p in scratch.iterdir()) == ["a.json", "b.json"]
    # a later run that agrees with the record clears only its own stale copy
    records.save_record(target, "old", rebase=False, quiet=True)
    assert [p.name for p in scratch.iterdir()] == ["b.json"]
    # a rebase clears the other one, and the empty folder goes with it
    records.save_record(other, "new", rebase=True, quiet=True)
    assert other.read_text() == "new" and not scratch.exists()


def test_json_and_csv_helpers_match_the_plain_writers(tmp_path):
    payload = {"b": 1.5, "a": [1, 2]}
    records.save_json(tmp_path / "m.json", payload, quiet=True)
    assert (tmp_path / "m.json").read_text() == json.dumps(payload, indent=2)
    frame = pd.DataFrame({"name": ["x", "y"], "sharpe": [0.5, 0.25]})
    records.save_csv(tmp_path / "v.csv", frame, index=False, quiet=True)
    plain = tmp_path / "plain.csv"
    frame.to_csv(plain, index=False)
    assert (tmp_path / "v.csv").read_bytes() == plain.read_bytes()
    # unchanged on a second identical save
    assert records.save_csv(tmp_path / "v.csv", frame, index=False, quiet=True) == tmp_path / "v.csv"
