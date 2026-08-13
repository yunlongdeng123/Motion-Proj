from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.build_worldsim_v4_m3_test_actor_registry import (
    M3TestActorRegistryError,
    bind_actor,
    resolve_scene_metadata,
)
from scripts.register_worldsim_v4_m3_test_checkpoints import (
    TASK_ID,
    TestCheckpointRegistryError as CheckpointRegistryError,
    build_registry,
    sha256_file,
)


SCENES = [f"scene-{index:04d}" for index in range(18)]


def test_resolve_test_metadata_and_bind_available_actor() -> None:
    token = "a" * 32
    cohort = {
        "freeze": {
            "scene_roles": {"test": SCENES},
            "scene_records": [
                {
                    "scene": scene,
                    "role": "test",
                    "actors": {"high_support": {"instance_token": token + str(index)}},
                    "continuous_clip": {
                        "status": "ready",
                        "actor_instance_token": token + str(index),
                        "start_index": 16,
                        "end_index": 22,
                        "duration_s": 3.0,
                        "sample_tokens": [str(value) for value in range(7)],
                    },
                }
                for index, scene in enumerate(SCENES)
            ],
        }
    }
    metadata = resolve_scene_metadata(cohort, SCENES[0])
    assert metadata["clip"]["end_index"] == 22
    registry = {
        "actors": [
            {
                "instance_token": metadata["instance_token"],
                "availability": "available",
                "rigid_model_index": 3,
                "checkpoint_tensor_slice": {"gaussian_count": 91},
            }
        ]
    }
    assert bind_actor(registry, metadata["instance_token"]) == {
        "status": "ready",
        "reason": None,
        "actor": {"model_index": 3, "gaussian_count": 91},
    }


def test_bind_unavailable_actor_becomes_explicit_abstain() -> None:
    token = "b" * 32
    result = bind_actor(
        {
            "actors": [
                {
                    "instance_token": token,
                    "availability": "unavailable_model_type",
                    "rigid_model_index": None,
                    "checkpoint_tensor_slice": {"gaussian_count": 0},
                }
            ]
        },
        token,
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "unavailable_model_type"


def test_resolve_test_metadata_rejects_non_continuous_clip() -> None:
    cohort = {
        "freeze": {
            "scene_roles": {"test": SCENES},
            "scene_records": [
                {
                    "scene": SCENES[0],
                    "role": "test",
                    "actors": {"high_support": {"instance_token": "token"}},
                    "continuous_clip": {
                        "status": "ready",
                        "actor_instance_token": "token",
                        "start_index": 16,
                        "end_index": 21,
                        "duration_s": 3.0,
                        "sample_tokens": [str(value) for value in range(6)],
                    },
                }
            ],
        }
    }
    with pytest.raises(M3TestActorRegistryError, match="continuous clip"):
        resolve_scene_metadata(cohort, SCENES[0])


def test_checkpoint_registry_accepts_exact_18_train_only_runs(tmp_path: Path) -> None:
    config_path = tmp_path / "reconstruction.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "worldsim_v4_streetgs_training_v1",
                "task_id": TASK_ID,
                "scenes": {scene: index for index, scene in enumerate(SCENES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    bindings = {}
    for index, scene in enumerate(SCENES):
        run = tmp_path / f"run-{index}"
        checkpoint_dir = run / "work_dirs/model"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "checkpoint_final.pth"
        checkpoint_path.write_bytes(f"checkpoint-{scene}".encode())
        source_config = checkpoint_dir / "config.yaml"
        source_config.write_text("data: {}\n", encoding="utf-8")
        checkpoint = {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "step": 30000,
            "model_keys": ["Background"],
            "gaussian_counts": {"Background": 1, "RigidNodes": None},
            "means_finite": True,
        }
        summary = {
            "status": "done",
            "task_id": TASK_ID,
            "scene": scene,
            "scene_index": index,
            "mode": "formal",
            "iterations": 30000,
            "checkpoint": checkpoint,
            "project_git": {"dirty": False},
            "model_inference_started": False,
            "test_quality_read": False,
        }
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        manifest = {"status": "done", "scene": scene, "mode": "formal", "test_quality_read": False, "artifacts": {"work_dirs_checkpoint": checkpoint}}
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "fingerprint.json").write_text("{}", encoding="utf-8")
        status = {"status": "done", "task_id": TASK_ID, "scene": scene, "mode": "formal", "summary_sha256": sha256_file(run / "summary.json")}
        (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
        bindings[scene] = run
    result = build_registry(config_path, bindings)
    assert result["scene_order"] == SCENES
    assert result["training_partition_only"] is True
    assert result["render_started"] is False
    assert result["test_quality_read"] is False

    bindings.pop(SCENES[-1])
    with pytest.raises(CheckpointRegistryError, match="scene set"):
        build_registry(config_path, bindings)
