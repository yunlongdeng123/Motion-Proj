from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from motion_proj.worldsim_v4.v33_replay import V33ReplayError
from scripts.materialize_worldsim_v4_v33_instance_config import build_instance_config


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[dict, dict, dict, Path, dict]:
    checkpoint = tmp_path / "checkpoint.pth"
    source_config = tmp_path / "source.yaml"
    registry = tmp_path / "registry.json"
    prompt = tmp_path / "prompt.json"
    masks = tmp_path / "masks.json"
    semantics = tmp_path / "semantics.json"
    sidecar = tmp_path / "high.npz"
    for path, payload in (
        (checkpoint, b"checkpoint"),
        (source_config, b"data: {}\n"),
        (registry, b"{}\n"),
        (prompt, b"{}\n"),
        (masks, b"{}\n"),
        (semantics, b"{}\n"),
        (sidecar, b"npz"),
    ):
        path.write_bytes(payload)
    semantic_config = {
        "schema_version": "worldsim_v4_v33_semantic_lift_v1",
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "source_config": str(source_config),
            "source_config_sha256": sha(source_config),
            "actor_registry": str(registry),
            "actor_registry_sha256": sha(registry),
        },
        "scene": {"name": "scene-0255", "frame_count": 10, "cameras": []},
        "split": {"development_frames": [2, 7], "heldout_frames": [4, 9]},
        "actors": {
            "high_support": {
                "instance_token": "high",
                "dataset_instance_id": 1,
                "rigid_model_index": 5,
                "sam_object_id": 1,
                "class_name": "vehicle.car",
            }
        },
        "provenance": {"actor_abstentions": {}},
    }
    semantic_config_path = tmp_path / "semantic.yaml"
    semantic_config_path.write_text(yaml.safe_dump(semantic_config), encoding="utf-8")
    summary = {
        "status": "done",
        "config_sha256": sha(semantic_config_path),
        "development_leaks": 0,
        "heldout_leaks": 0,
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "checkpoint_sha256_before": sha(checkpoint),
        "checkpoint_sha256_after": sha(checkpoint),
        "prompt_manifest": str(prompt),
        "prompt_manifest_sha256": sha(prompt),
        "mask_manifest": str(masks),
        "mask_manifest_sha256": sha(masks),
        "semantic_manifest": str(semantics),
        "semantic_manifest_sha256": sha(semantics),
        "actors": {
            "high_support": {"sidecar": str(sidecar), "sha256": sha(sidecar)}
        },
    }
    replay = {
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "algorithm": {"implementation_commit": "e6663e1"},
        "instance_field": {
            "formal_selected_arm": "O1_dual_opacity",
            "evaluation_partition": "development",
        },
        "runtimes": {
            "sam2_python": "/env/sam/bin/python",
            "drivestudio_python": "/env/ds/bin/python",
            "drivestudio_checkout": "/code/drivestudio",
        },
    }
    template = {
        "seed": 7,
        "sam2_fallback": {},
        "representation": {},
        "arms": {},
        "optimization": {"formal_selected_arm": "O1_dual_opacity"},
        "evaluation": {},
        "outputs": {},
    }
    return replay, template, semantic_config, semantic_config_path, summary


def test_materializes_development_only_instance_contract(tmp_path: Path) -> None:
    replay, template, config, config_path, summary = fixture(tmp_path)

    result = build_instance_config(
        replay=replay,
        template=template,
        semantic_config=config,
        semantic_config_path=config_path,
        semantic_summary=summary,
    )

    assert result["optimization"]["formal_selected_arm"] == "O1_dual_opacity"
    assert result["provenance"]["evaluation_partition"] == "development"
    assert result["actors"]["high_support"]["semantic_sidecar_sha256"] == summary[
        "actors"
    ]["high_support"]["sha256"]


def test_rejects_semantic_summary_without_no_test_proof(tmp_path: Path) -> None:
    replay, template, config, config_path, summary = fixture(tmp_path)
    summary.pop("test_quality_read")

    with pytest.raises(V33ReplayError, match="test_quality_read"):
        build_instance_config(
            replay=replay,
            template=template,
            semantic_config=config,
            semantic_config_path=config_path,
            semantic_summary=summary,
        )
