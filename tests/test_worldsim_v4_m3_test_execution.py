from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from motion_proj.worldsim_v4.test_freeze import (
    TASK_ID,
    TestFreezeError as FreezeError,
    sha256_file,
)
from scripts.run_worldsim_v4_m3_test_exact_once import (
    attempt_payload,
    completion_payload,
    runtime_preflight,
    verify_all_assets,
)
from scripts import run_worldsim_v4_m3_test_exact_once as EXECUTOR


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def completed_fixture(tmp_path: Path, *, ready: bool) -> tuple[dict, dict, dict]:
    run = tmp_path / "run"
    run.mkdir()
    planned = {
        "ordinal": 1,
        "scene": "scene-0001",
        "attempt_id": "m3-test-01-scene0001-s0-r1",
        "run_dir": str(run),
    }
    provenance = {
        "freeze_sha256": "a" * 64,
        "freeze_commit": "b" * 40,
        "source_commit": "c" * 40,
        "relative_path": "V4_TEST_FREEZE.json",
    }
    write_json(run / "manifest.json", {"files": {}})
    write_json(run / "fingerprint.json", {"freeze": provenance["freeze_sha256"]})
    summary = {
        "task_id": TASK_ID,
        "scene": planned["scene"],
        "partition": "test",
        "status": "done" if ready else "abstain",
        "test_scene_attempted": True,
        "test_quality_read": ready,
        "test_attempt": {
            "attempt_id": planned["attempt_id"],
            "freeze_commit": provenance["freeze_commit"],
        },
        "project_git_head": provenance["freeze_commit"],
        "project_git_dirty": False,
    }
    write_json(run / "summary.json", summary)
    if ready:
        write_json(run / "test_read_started.json", {"state": "started"})
    status = {
        "task_id": TASK_ID,
        "scene": planned["scene"],
        "status": "done",
        "summary_sha256": sha256_file(run / "summary.json"),
        "manifest_sha256": sha256_file(run / "manifest.json"),
        "fingerprint_sha256": sha256_file(run / "fingerprint.json"),
        "test_quality_read": ready,
    }
    write_json(run / "status.json", status)
    inventory = {
        "scenes": {
            planned["scene"]: {"status": "ready" if ready else "abstain"}
        }
    }
    return planned, provenance, inventory


@pytest.mark.parametrize("ready", [False, True])
def test_completion_payload_matches_asset_read_contract(tmp_path: Path, ready: bool) -> None:
    planned, provenance, inventory = completed_fixture(tmp_path, ready=ready)
    completion = completion_payload(
        freeze={},
        provenance=provenance,
        planned=planned,
        inventory=inventory,
    )
    assert completion["state"] == "completed"
    assert completion["test_quality_read"] is ready
    assert attempt_payload(planned, provenance)["state"] == "started"


def test_ready_completion_requires_test_read_marker(tmp_path: Path) -> None:
    planned, provenance, inventory = completed_fixture(tmp_path, ready=True)
    (Path(planned["run_dir"]) / "test_read_started.json").unlink()
    with pytest.raises(FreezeError, match="marker"):
        completion_payload(
            freeze={},
            provenance=provenance,
            planned=planned,
            inventory=inventory,
        )


def test_all_assets_are_hashed_before_first_attempt(tmp_path: Path) -> None:
    scene = "scene-0001"
    rows = {}
    for label in ("checkpoint", "drivestudio_config", "registry"):
        path = tmp_path / label
        path.write_bytes(label.encode())
        rows[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    freeze = {"scene_order": [scene]}
    inventory = {"scene_order": [scene], "scenes": {scene: rows}}
    verify_all_assets(freeze, inventory)
    (tmp_path / "checkpoint").write_bytes(b"drift")
    with pytest.raises(FreezeError, match="content drift"):
        verify_all_assets(freeze, inventory)


def test_runtime_preflight_rejects_competing_gpu_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/run_worldsim_v4_m3_scene.py").write_text("# runner\n")
    freeze = {
        "runner_python": sys.executable,
        "resources": {
            "required_gpu": "NVIDIA GeForce RTX 3090",
            "minimum_disk_free_gib": 0,
            "maximum_gpu_used_at_attempt_start_mib": 2048,
        },
    }
    competing = False

    def fake_run(command, **kwargs):
        nonlocal competing
        if command[0] != "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if any("--query-gpu=" in argument for argument in command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="NVIDIA GeForce RTX 3090, 2\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="12345\n" if competing else "",
            stderr="",
        )

    monkeypatch.setattr(EXECUTOR.subprocess, "run", fake_run)
    runtime_preflight(freeze, project, tmp_path / "runs/ledger")
    competing = True
    with pytest.raises(FreezeError, match="competing"):
        runtime_preflight(freeze, project, tmp_path / "runs/ledger")
