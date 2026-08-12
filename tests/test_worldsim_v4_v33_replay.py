from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from motion_proj.worldsim_v4.v33_replay import (
    REQUIRED_STAGES,
    V33ReplayError,
    bind_actor_registry,
    resolve_scene_contracts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    processed = tmp_path / "processed"
    records = []
    scene_contract = {}
    checkpoints = {}
    scenes = [f"scene-{index:04d}" for index in range(1, 7)]
    for index, scene in enumerate(scenes, 1):
        scene_contract[scene] = {"scene_index": index, "official_split": "train"}
        scene_root = processed / f"{index:03d}"
        (scene_root / "instances").mkdir(parents=True)
        high = f"high-{index}"
        boundary = f"boundary-{index}"
        (scene_root / "instances" / "instances_info.json").write_text(
            json.dumps({"11": {"id": high}, "22": {"id": boundary}}),
            encoding="utf-8",
        )
        checkpoint_root = tmp_path / "runs" / scene
        checkpoint_root.mkdir(parents=True)
        checkpoint = checkpoint_root / "checkpoint_final.pth"
        checkpoint.write_bytes(f"checkpoint-{scene}".encode())
        (checkpoint_root / "config.yaml").write_text("data: {}\n", encoding="utf-8")
        checkpoints[scene] = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha(checkpoint),
            "scene_index": index,
        }
        records.append(
            {
                "scene": scene,
                "scene_index": index,
                "role": "development",
                "actors": {
                    "high_support": {"instance_token": high, "category": "vehicle.car"},
                    "difficult": {
                        "instance_token": boundary,
                        "category": "vehicle.car",
                    },
                },
                "continuous_clip": {"start_index": 1, "end_index": 7},
            }
        )
    cohort = {
        "freeze": {
            "scene_roles": {"development": scenes},
            "scene_records": records,
        }
    }
    cohort_path = project / "configs" / "cohort.yaml"
    cohort_path.write_text(yaml.safe_dump(cohort), encoding="utf-8")
    matrix = {
        "frame_partition": {
            "heldout": "sample_index_mod_5_eq_4",
            "development": "sample_index_mod_5_eq_2",
            "train": "remaining",
        },
        "scene_contract": scene_contract,
        "baselines": {
            "streetgs": {"checkpoints": checkpoints},
            "v33_frozen": {"implementation_commit": "e6663e1"},
        },
    }
    (project / "configs" / "matrix.yaml").write_text(
        yaml.safe_dump(matrix), encoding="utf-8"
    )
    config = {
        "algorithm": {
            "implementation_commit": "e6663e1",
            "required_stages": list(REQUIRED_STAGES),
            "base_rgb_immutable": True,
        },
        "inputs": {
            "cohort_config": "configs/cohort.yaml",
            "cohort_config_sha256": _sha(cohort_path),
            "baseline_matrix": "configs/matrix.yaml",
            "processed_root": str(processed),
        },
        "scene_source": {
            "split": "development",
            "high_support_role": "high_support",
            "boundary_support_role": "difficult",
        },
        "frame_partition": {
            "modulus": 5,
            "train_remainders": [0, 1, 3],
            "development_remainder": 2,
            "heldout_remainder": 4,
            "test_quality_read": False,
        },
        "gates": {"expected_scene_count": 6},
    }
    return config, project


def test_resolve_scene_contracts_uses_d0_actors_and_exact_bases(tmp_path: Path) -> None:
    config, project = _fixture(tmp_path)

    rows = resolve_scene_contracts(config, project_root=project)

    assert len(rows) == 6
    assert rows[0]["actors"]["high_support"]["dataset_instance_id"] == 11
    assert rows[0]["actors"]["boundary_support"]["dataset_instance_id"] == 22
    assert rows[0]["test_quality_read"] is False


def test_resolve_scene_contracts_rejects_checkpoint_tamper(tmp_path: Path) -> None:
    config, project = _fixture(tmp_path)
    matrix = yaml.safe_load((project / "configs" / "matrix.yaml").read_text())
    checkpoint = Path(matrix["baselines"]["streetgs"]["checkpoints"]["scene-0001"]["path"])
    checkpoint.write_bytes(b"tampered")

    try:
        resolve_scene_contracts(config, project_root=project)
    except V33ReplayError as error:
        assert "bytes 漂移" in str(error) or "SHA 漂移" in str(error)
    else:
        raise AssertionError("checkpoint tamper 必须 fail closed")


def test_validation_uses_independent_content_addressed_checkpoint_registry(
    tmp_path: Path,
) -> None:
    config, project = _fixture(tmp_path)
    cohort_path = project / "configs" / "cohort.yaml"
    cohort = yaml.safe_load(cohort_path.read_text(encoding="utf-8"))
    scenes = cohort["freeze"]["scene_roles"].pop("development")
    cohort["freeze"]["scene_roles"]["validation"] = scenes
    for row in cohort["freeze"]["scene_records"]:
        row["role"] = "validation"
    cohort_path.write_text(yaml.safe_dump(cohort), encoding="utf-8")
    config["inputs"]["cohort_config_sha256"] = _sha(cohort_path)
    config["scene_source"]["split"] = "validation"
    matrix = yaml.safe_load((project / "configs" / "matrix.yaml").read_text())
    registry = {
        "schema_version": "worldsim_v4_streetgs_checkpoint_registry_v1",
        "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
        "split": "validation",
        "partition_contract": "sample_index_mod_5",
        "test_quality_read": False,
        "checkpoints": matrix["baselines"]["streetgs"]["checkpoints"],
    }
    registry_path = project / "configs" / "validation_registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    config["inputs"]["checkpoint_registry"] = "configs/validation_registry.yaml"
    config["inputs"]["checkpoint_registry_sha256"] = _sha(registry_path)

    rows = resolve_scene_contracts(config, project_root=project)

    assert len(rows) == 6
    assert all(row["cohort_role"] == "validation" for row in rows)
    assert rows[0]["scene_index"] == 1


def test_validation_rejects_registry_scene_drift(tmp_path: Path) -> None:
    config, project = _fixture(tmp_path)
    cohort_path = project / "configs" / "cohort.yaml"
    cohort = yaml.safe_load(cohort_path.read_text(encoding="utf-8"))
    scenes = cohort["freeze"]["scene_roles"].pop("development")
    cohort["freeze"]["scene_roles"]["validation"] = scenes
    for row in cohort["freeze"]["scene_records"]:
        row["role"] = "validation"
    cohort_path.write_text(yaml.safe_dump(cohort), encoding="utf-8")
    config["inputs"]["cohort_config_sha256"] = _sha(cohort_path)
    config["scene_source"]["split"] = "validation"
    matrix = yaml.safe_load((project / "configs" / "matrix.yaml").read_text())
    checkpoints = matrix["baselines"]["streetgs"]["checkpoints"]
    checkpoints.pop(scenes[-1])
    registry_path = project / "configs" / "validation_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "worldsim_v4_streetgs_checkpoint_registry_v1",
                "split": "validation",
                "partition_contract": "sample_index_mod_5",
                "test_quality_read": False,
                "checkpoints": checkpoints,
            }
        ),
        encoding="utf-8",
    )
    config["inputs"]["checkpoint_registry"] = "configs/validation_registry.yaml"

    with pytest.raises(V33ReplayError, match="scene 集合"):
        resolve_scene_contracts(config, project_root=project)


def test_validation_rejects_registry_scene_index_drift(tmp_path: Path) -> None:
    config, project = _fixture(tmp_path)
    cohort_path = project / "configs" / "cohort.yaml"
    cohort = yaml.safe_load(cohort_path.read_text(encoding="utf-8"))
    scenes = cohort["freeze"]["scene_roles"].pop("development")
    cohort["freeze"]["scene_roles"]["validation"] = scenes
    for row in cohort["freeze"]["scene_records"]:
        row["role"] = "validation"
        row["scene_index"] = int(row["scene"].removeprefix("scene-"))
    cohort_path.write_text(yaml.safe_dump(cohort), encoding="utf-8")
    config["inputs"]["cohort_config_sha256"] = _sha(cohort_path)
    config["scene_source"]["split"] = "validation"
    matrix = yaml.safe_load((project / "configs" / "matrix.yaml").read_text())
    checkpoints = matrix["baselines"]["streetgs"]["checkpoints"]
    checkpoints[scenes[0]]["scene_index"] = 999
    registry_path = project / "configs" / "validation_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "worldsim_v4_streetgs_checkpoint_registry_v1",
                "split": "validation",
                "partition_contract": "sample_index_mod_5",
                "test_quality_read": False,
                "checkpoints": checkpoints,
            }
        ),
        encoding="utf-8",
    )
    config["inputs"]["checkpoint_registry"] = "configs/validation_registry.yaml"

    with pytest.raises(V33ReplayError, match="scene_index"):
        resolve_scene_contracts(config, project_root=project)


def test_bind_actor_registry_requires_available_high_actor(tmp_path: Path) -> None:
    config, project = _fixture(tmp_path)
    scene = resolve_scene_contracts(config, project_root=project)[0]
    registry = {
        "actors": [
            {
                "instance_token": "high-1",
                "processed_true_instance_id": 11,
                "availability": "unavailable_empty_slice",
                "rigid_model_index": 3,
            },
            {
                "instance_token": "boundary-1",
                "processed_true_instance_id": 22,
                "availability": "available",
                "rigid_model_index": 4,
            },
        ]
    }

    try:
        bind_actor_registry(scene, registry)
    except V33ReplayError as error:
        assert "high actor 不可用" in str(error)
    else:
        raise AssertionError("unavailable high actor 必须 fail closed")


def test_bind_actor_registry_can_retain_unavailable_high_as_abstain(
    tmp_path: Path,
) -> None:
    config, project = _fixture(tmp_path)
    scene = resolve_scene_contracts(config, project_root=project)[0]
    registry = {
        "actors": [
            {
                "instance_token": "high-1",
                "processed_true_instance_id": 11,
                "availability": "unavailable_initialization_filter",
                "rigid_model_index": None,
            },
            {
                "instance_token": "boundary-1",
                "processed_true_instance_id": 22,
                "availability": "unavailable_initialization_filter",
                "rigid_model_index": None,
            },
        ]
    }

    bound = bind_actor_registry(scene, registry, require_high_available=False)

    assert bound["actors"]["high_support"]["availability"] == (
        "unavailable_initialization_filter"
    )
