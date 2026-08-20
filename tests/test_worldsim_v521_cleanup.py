from __future__ import annotations

import pytest

from scripts.cleanup_worldsim_v521_renders import RUNS_ROOT, assert_run_path


def test_cleanup_path_guard_accepts_only_child_run() -> None:
    assert assert_run_path(RUNS_ROOT / "fixture-run") == RUNS_ROOT / "fixture-run"
    with pytest.raises(RuntimeError, match="越界"):
        assert_run_path(RUNS_ROOT)
    with pytest.raises(RuntimeError, match="越界"):
        assert_run_path("/root/autodl-tmp/data")
