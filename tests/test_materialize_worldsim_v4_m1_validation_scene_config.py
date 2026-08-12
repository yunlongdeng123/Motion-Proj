from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.materialize_worldsim_v4_m1_validation_scene_config import (
    M1MaterializationError,
    materialize_abstain,
    materialize_ready,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def m1_config() -> dict:
    return {
        "schema_version": "worldsim_v4_m1_evidence_v1",
        "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
        "status": "development_frozen",
        "protocol": {
            "validation_scenes": ["scene-0071"],
            "test_quality_read": False,
        },
        "inputs": {"required_v33_arm": "O1_dual_opacity"},
        "evidence": {},
        "calibration": {},
        "evaluation": {},
        "immutability": {},
        "gates": {},
    }


def test_materialize_validation_abstention_retains_scene_denominator(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "m1.yaml"
    config_path.write_text(yaml.safe_dump(m1_config()), encoding="utf-8")
    bound_path = tmp_path / "bound.json"
    bound_path.write_text(
        json.dumps(
            {
                "schema_version": "worldsim_v4_v33_bound_scene_v1",
                "scene": "scene-0071",
                "cohort_role": "validation",
                "actors": {
                    "high_support": {
                        "availability": "unavailable_initialization_filter"
                    }
                },
                "test_quality_read": False,
            }
        ),
        encoding="utf-8",
    )
    result = materialize_abstain(
        m1_config=m1_config(),
        m1_config_path=config_path,
        bound_scene_path=bound_path,
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "ABSTAIN_NO_ACTOR"
    assert result["partition"] == "validation"
    assert result["validation_content_read"] is False


def test_materialize_validation_rejects_available_actor_abstention(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "m1.yaml"
    config_path.write_text(yaml.safe_dump(m1_config()), encoding="utf-8")
    bound_path = tmp_path / "bound.json"
    bound_path.write_text(
        json.dumps(
            {
                "schema_version": "worldsim_v4_v33_bound_scene_v1",
                "scene": "scene-0071",
                "cohort_role": "validation",
                "actors": {"high_support": {"availability": "available"}},
                "test_quality_read": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(M1MaterializationError, match="cannot abstain"):
        materialize_abstain(
            m1_config=m1_config(),
            m1_config_path=config_path,
            bound_scene_path=bound_path,
        )


def test_materialize_validation_ready_binds_exact_instance_artifacts(
    tmp_path: Path,
) -> None:
    config = m1_config()
    config_path = tmp_path / "m1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    source = tmp_path / "source.yaml"
    sidecar = tmp_path / "sidecar.npz"
    field = tmp_path / "field.npz"
    masks = tmp_path / "masks.json"
    checkpoint.write_bytes(b"checkpoint")
    source.write_text("data: {}\n", encoding="utf-8")
    sidecar.write_bytes(b"sidecar")
    field.write_bytes(b"field")
    masks.write_text("{}\n", encoding="utf-8")
    instance = {
        "schema_version": "worldsim_v4_v33_instance_field_v1",
        "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "source_config": str(source),
            "source_config_sha256": sha(source),
        },
        "scene": {
            "name": "scene-0071",
            "processed_scene_dir": "/processed/068",
        },
        "actors": {
            "high_support": {
                "instance_token": "actor",
                "dataset_instance_id": 11,
                "rigid_model_index": 3,
                "semantic_sidecar": str(sidecar),
                "semantic_sidecar_sha256": sha(sidecar),
            }
        },
        "runtimes": {"drivestudio_python": "/env/python"},
        "provenance": {
            "evaluation_partition": "development",
            "development_content_read": False,
            "heldout_content_read": False,
            "test_quality_read": False,
        },
    }
    instance_path = tmp_path / "instance.yaml"
    instance_path.write_text(yaml.safe_dump(instance), encoding="utf-8")
    run = tmp_path / "instance-run"
    (run / "instance_field").mkdir(parents=True)
    summary = {
        "arms": {
            "O1_dual_opacity": {
                "instance_field": str(field),
                "instance_field_sha256": sha(field),
                "evaluation": {"aggregate": {"boundary_f1": 0.4}, "rows": []},
            }
        },
        "evaluation_source": {
            "manifest": str(masks),
            "manifest_sha256": sha(masks),
            "partition": "development",
            "optimization_forbidden": True,
        },
    }
    (run / "instance_field" / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    stage = {
        "status": "done",
        "scene": "scene-0071",
        "config_sha256": sha(instance_path),
        "selected_arm": "O1_dual_opacity",
        "test_quality_read": False,
    }
    stage_path = run / "stage_summary.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    status = {
        "status": "done",
        "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
        "evaluation_partition": "development",
        "stage_summary_sha256": sha(stage_path),
    }
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")

    result = materialize_ready(
        m1_config=config,
        m1_config_path=config_path,
        instance_config_path=instance_path,
        instance_run=run,
    )

    assert result["status"] == "ready"
    assert result["partition"] == "validation"
    assert result["inputs"]["v33_o1_instance_field"]["sha256"] == sha(field)
    assert result["actors"]["high_support"]["rigid_model_index"] == 3
