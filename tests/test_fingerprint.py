import hashlib
import subprocess

import pytest

from motion_proj.runtime.fingerprint import git_state


def _git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repository(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Motion Proj Test",
        "-c",
        "user.email=motion-proj-test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    return tmp_path


def test_git_state_reports_clean_repository(repository):
    state = git_state(str(repository))

    assert state["git_available"]
    assert not state["dirty"]
    assert not state["dirty_tracked"]
    assert not state["dirty_untracked"]
    assert state["untracked_count"] == 0
    assert state["dirty_diff_hash"] == hashlib.sha256(b"").hexdigest()


def test_git_state_reports_tracked_changes(repository):
    clean_hash = git_state(str(repository))["dirty_diff_hash"]
    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")

    state = git_state(str(repository))

    assert state["dirty"]
    assert state["dirty_tracked"]
    assert not state["dirty_untracked"]
    assert state["dirty_diff_hash"] != clean_hash


def test_git_state_fingerprints_untracked_contents(repository):
    clean_hash = git_state(str(repository))["dirty_diff_hash"]
    untracked = repository / "new.txt"
    untracked.write_text("first\n", encoding="utf-8")

    first = git_state(str(repository))
    repeated = git_state(str(repository))
    untracked.write_text("second\n", encoding="utf-8")
    changed = git_state(str(repository))

    assert first["dirty"]
    assert not first["dirty_tracked"]
    assert first["dirty_untracked"]
    assert first["untracked_count"] == 1
    assert first["dirty_diff_hash"] != clean_hash
    assert repeated["dirty_diff_hash"] == first["dirty_diff_hash"]
    assert changed["dirty_diff_hash"] != first["dirty_diff_hash"]


def test_git_state_ignores_excluded_untracked_files(repository):
    (repository / "ignored.log").write_text("ignored\n", encoding="utf-8")

    state = git_state(str(repository))

    assert not state["dirty"]
    assert state["untracked_count"] == 0


def test_git_state_fails_closed_outside_repository(tmp_path):
    state = git_state(str(tmp_path))

    assert not state["git_available"]
    assert state["commit"] == "unknown"
    assert state["dirty"]
    assert state["dirty_diff_hash"] == "unknown"
