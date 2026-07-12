import os
import time
from pathlib import Path

import pytest

from scripts.watchdog import inspect_run


@pytest.mark.parametrize(
    ("marker", "status", "healthy"),
    [
        ("COMPLETE", "completed", True),
        ("FAILED", "failed", False),
        ("REJECTED", "rejected", True),
    ],
)
def test_terminal_state_ignores_stale_heartbeat(
    tmp_path: Path,
    marker: str,
    status: str,
    healthy: bool,
) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text("{}\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(heartbeat, (old, old))
    (tmp_path / marker).touch()

    report = inspect_run(tmp_path, min_free_gb=0, check_gpu=False)

    assert report["status"] == status
    assert report["terminal"] is True
    assert report["healthy"] is healthy
    assert "heartbeat stale" not in report["problems"]


def test_running_state_still_reports_stale_heartbeat(tmp_path: Path) -> None:
    report = inspect_run(tmp_path, min_free_gb=0, check_gpu=False)

    assert report["status"] == "running"
    assert report["terminal"] is False
    assert report["healthy"] is False
    assert "heartbeat stale" in report["problems"]


def test_conflicting_terminal_markers_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "COMPLETE").touch()
    (tmp_path / "FAILED").touch()

    report = inspect_run(tmp_path, min_free_gb=0, check_gpu=False)

    assert report["status"] == "invalid"
    assert report["terminal"] is True
    assert report["healthy"] is False
    assert report["problems"] == ["conflicting terminal markers: COMPLETE,FAILED"]
