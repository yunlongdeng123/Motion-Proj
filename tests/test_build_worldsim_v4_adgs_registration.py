from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_worldsim_v4_adgs_registration import (
    ADGSRegistrationError,
    registration_record,
    sha256_file,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _formal_run(tmp_path: Path, *, scene: str = "scene-0048") -> Path:
    run = tmp_path / "formal"
    checkpoint_root = run / "model" / "point_cloud" / "iteration_60000"
    checkpoint_root.mkdir(parents=True)
    files = {}
    for name, payload in {
        "point_cloud.ply": b"point-cloud",
        "deform.pth": b"deform",
        "env.pth": b"environment",
    }.items():
        path = checkpoint_root / name
        path.write_bytes(payload)
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    checkpoint = {"iteration": 60000, "root": str(checkpoint_root), "files": files}
    summary = {
        "schema_version": "worldsim_v4_adgs_summary_v1",
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "status": "done",
        "scene": scene,
        "mode": "formal",
        "checkpoint": checkpoint,
        "project_git": {"dirty": False},
        "training_started": True,
        "model_inference_started": False,
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    _write(run / "summary.json", summary)
    _write(
        run / "status.json",
        {
            "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
            "status": "done",
            "scene": scene,
            "mode": "formal",
            "summary_sha256": sha256_file(run / "summary.json"),
        },
    )
    _write(run / "fingerprint.json", {"checkpoint": checkpoint})
    _write(
        run / "manifest.json",
        {
            "schema_version": "worldsim_v4_adgs_run_manifest_v1",
            "status": "done",
            "checkpoint": checkpoint,
        },
    )
    _write(
        run / "stages" / "train_formal.json",
        {
            "stage": "train_formal",
            "status": "done",
            "return_code": 0,
            "duration_seconds": 123.5,
            "peak_gpu_memory_mib": 7000,
            "peak_cgroup_memory_bytes": 12_000_000_000,
        },
    )
    return run


def test_registration_record_binds_formal_checkpoint_and_terminal_evidence(tmp_path: Path) -> None:
    run = _formal_run(tmp_path)

    record = registration_record(run, expected_scene="scene-0048")

    assert record["state"] == "matched_formal_done"
    assert record["step"] == 60000
    assert record["duration_seconds"] == 123.5
    assert set(record["files"]) == {"point_cloud.ply", "deform.pth", "env.pth"}
    assert all(row["bytes"] > 0 for row in record["files"].values())


def test_registration_record_rejects_checkpoint_tamper(tmp_path: Path) -> None:
    run = _formal_run(tmp_path)
    (run / "model" / "point_cloud" / "iteration_60000" / "deform.pth").write_bytes(b"tampered")

    with pytest.raises(ADGSRegistrationError, match="checkpoint 内容漂移"):
        registration_record(run, expected_scene="scene-0048")


def test_registration_record_rejects_content_read_or_dirty_project(tmp_path: Path) -> None:
    run = _formal_run(tmp_path)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["heldout_content_read"] = True
    _write(summary_path, summary)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    status["summary_sha256"] = sha256_file(summary_path)
    _write(run / "status.json", status)

    with pytest.raises(ADGSRegistrationError, match="heldout_content_read"):
        registration_record(run, expected_scene="scene-0048")
