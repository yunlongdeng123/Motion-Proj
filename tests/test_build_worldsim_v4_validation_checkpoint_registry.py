from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.build_worldsim_v4_validation_checkpoint_registry import (
    TASK_ID,
    ValidationCheckpointRegistryError,
    build_registry,
    parse_run_bindings,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    scenes = [f"scene-{index:04d}" for index in range(1, 7)]
    config = {
        "schema_version": "worldsim_v4_streetgs_training_v1",
        "task_id": TASK_ID,
        "scenes": {scene: index for index, scene in enumerate(scenes)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runs = {}
    for scene in scenes:
        run = tmp_path / "runs" / scene
        checkpoint = run / "work" / "checkpoint_final.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(scene.encode())
        (checkpoint.parent / "config.yaml").write_text("data: {}\n", encoding="utf-8")
        checkpoint_row = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha(checkpoint),
            "step": 30000,
            "means_finite": True,
        }
        summary = {
            "status": "done",
            "scene": scene,
            "mode": "formal",
            "iterations": 30000,
            "checkpoint": checkpoint_row,
            "project_git": {"dirty": False},
            "test_quality_read": False,
        }
        write_json(run / "summary.json", summary)
        write_json(
            run / "status.json",
            {
                "status": "done",
                "task_id": TASK_ID,
                "scene": scene,
                "mode": "formal",
                "summary_sha256": sha(run / "summary.json"),
            },
        )
        write_json(
            run / "manifest.json",
            {
                "status": "done",
                "scene": scene,
                "mode": "formal",
                "artifacts": {"work_dirs_checkpoint": checkpoint_row},
                "test_quality_read": False,
            },
        )
        write_json(run / "fingerprint.json", {"scene": scene})
        runs[scene] = run
    return config_path, runs


def test_build_registry_binds_six_exact_formal_checkpoints(tmp_path: Path) -> None:
    config, runs = fixture(tmp_path)
    result = build_registry(config_path=config, run_bindings=runs)
    assert result["status"] == "done"
    assert result["scene_order"] == list(runs)
    assert len(result["checkpoints"]) == 6
    assert result["test_quality_read"] is False


def test_registry_rejects_dirty_formal_run(tmp_path: Path) -> None:
    config, runs = fixture(tmp_path)
    first = next(iter(runs.values()))
    summary = json.loads((first / "summary.json").read_text())
    summary["project_git"]["dirty"] = True
    write_json(first / "summary.json", summary)
    status = json.loads((first / "status.json").read_text())
    status["summary_sha256"] = sha(first / "summary.json")
    write_json(first / "status.json", status)
    with pytest.raises(ValidationCheckpointRegistryError, match="formal contract"):
        build_registry(config_path=config, run_bindings=runs)


def test_parse_run_bindings_rejects_duplicates() -> None:
    with pytest.raises(ValidationCheckpointRegistryError, match="duplicate"):
        parse_run_bindings(["scene-1=/a", "scene-1=/b"])
