from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from motion_proj.worldsim_v4.v33_replay import REQUIRED_STAGES
from scripts.build_worldsim_v4_validation_v33_replay_config import (
    TASK_ID,
    ValidationReplayConfigError,
    build_config,
)


def base() -> dict:
    return {
        "schema_version": "worldsim_v4_v33_replay_v1",
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "status": "running",
        "algorithm": {"required_stages": list(REQUIRED_STAGES)},
        "inputs": {"baseline_matrix": "configs/matrix.yaml"},
        "scene_source": {"split": "development", "checkpoint_source": "b0"},
        "frame_partition": {"test_quality_read": False},
    }


def registry() -> dict:
    return {
        "schema_version": "worldsim_v4_streetgs_checkpoint_registry_v1",
        "task_id": TASK_ID,
        "status": "done",
        "split": "validation",
        "partition_contract": "sample_index_mod_5",
        "checkpoints": {f"scene-{index}": {} for index in range(6)},
        "test_quality_read": False,
    }


def test_build_validation_replay_reuses_algorithm_but_not_development_bases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry()), encoding="utf-8")
    result = build_config(base=base(), registry_path=path, registry=registry())
    assert result["task_id"] == TASK_ID
    assert result["scene_source"]["split"] == "validation"
    assert result["inputs"]["checkpoint_registry"] == str(path)
    assert result["validation_protocol"]["validation_optimization_forbidden"] is True
    assert result["validation_protocol"]["within_scene_evaluation_partition"] == (
        "development"
    )


def test_build_validation_replay_rejects_non_validation_registry(tmp_path: Path) -> None:
    payload = registry()
    payload["split"] = "development"
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationReplayConfigError, match="registry"):
        build_config(base=base(), registry_path=path, registry=payload)
