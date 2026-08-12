from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.build_worldsim_v33_s4_spatial_delta import (
    SUPPORTED_TASK_IDS,
    available_stack_ids,
)
from scripts.materialize_worldsim_v4_v33_spatial_config import build_spatial_config


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_available_stacks_fail_safe_without_optional_assets() -> None:
    assert available_stack_ids(has_background=False, has_actor=False) == [
        "base_only",
        "erase",
    ]
    assert available_stack_ids(has_background=True, has_actor=False) == [
        "base_only",
        "erase",
        "erase_background",
    ]
    assert available_stack_ids(has_background=False, has_actor=True) == [
        "base_only",
        "erase",
        "actor_override",
    ]


def test_spatial_builder_accepts_m1_validation_task() -> None:
    assert "WS-V4-M1-EVIDENCE-FIELD-01" in SUPPORTED_TASK_IDS


def test_materializes_erase_only_spatial_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    source = tmp_path / "source.yaml"
    registry = tmp_path / "registry.json"
    field = tmp_path / "instance_field.npz"
    target_mask = tmp_path / "target.npz"
    for path, payload in (
        (checkpoint, b"checkpoint"),
        (source, b"data: {}\n"),
        (registry, b"{}\n"),
        (field, b"npz"),
        (target_mask, b"mask"),
    ):
        path.write_bytes(payload)
    instance_config = {
        "schema_version": "worldsim_v4_v33_instance_field_v1",
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "source_config": str(source),
            "source_config_sha256": sha(source),
            "actor_registry": str(registry),
            "actor_registry_sha256": sha(registry),
        },
        "scene": {
            "name": "scene-0255",
            "processed_scene_dir": "/data/204",
            "cameras": [{"id": 0, "name": "CAM_FRONT"}],
        },
        "split": {"development_frames": [2, 7], "heldout_frames": [4, 9]},
        "actors": {
            "high_support": {
                "dataset_instance_id": 1,
                "instance_token": "high",
                "rigid_model_index": 5,
            }
        },
        "outputs": {"model_native_width": 800, "model_native_height": 450},
    }
    instance_config_path = tmp_path / "instance.yaml"
    instance_config_path.write_text(yaml.safe_dump(instance_config), encoding="utf-8")
    run = tmp_path / "instance-run"
    (run / "instance_field").mkdir(parents=True)
    inner = {
        "arms": {
            "O1_dual_opacity": {
                "instance_field": str(field),
                "instance_field_sha256": sha(field),
            }
        }
    }
    inner_path = run / "instance_field" / "summary.json"
    inner_path.write_text(json.dumps(inner))
    stage = {
        "status": "done",
        "evaluation_partition": "development",
        "config_sha256": sha(instance_config_path),
        "instance_summary_sha256": sha(inner_path),
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    (run / "stage_summary.json").write_text(json.dumps(stage))
    (run / "status.json").write_text(json.dumps({"status": "done"}))
    eval_mask_dir = run / "eval_targets" / "artifacts" / "masks"
    eval_mask_dir.mkdir(parents=True)
    (eval_mask_dir / "mask_manifest.json").write_text(
        json.dumps(
            {
                "masks": [
                    {
                        "role": "high_support",
                        "frame": 2,
                        "camera_id": 0,
                        "accepted": True,
                        "positive_pixels": 50,
                        "mask": str(target_mask),
                        "mask_sha256": sha(target_mask),
                    }
                ]
            }
        )
    )
    replay = {
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "algorithm": {"implementation_commit": "e6663e1"},
        "instance_field": {"formal_selected_arm": "O1_dual_opacity"},
        "spatial_delta": {
            "executable_stacks_without_scene_specific_assets": [
                "base_only",
                "erase",
            ]
        },
        "runtimes": {
            "drivestudio_checkout": "/code/drivestudio",
            "drivestudio_python": "/env/bin/python",
        },
    }
    template = {
        "seed": 0,
        "composition": {
            "order": ["ERASE", "INSERT_BACKGROUND", "INSERT_ACTOR", "RENDER_ONLY"],
            "stacks": ["base_only", "erase", "full"],
            "minimum_background_instance_opacity": 0.5,
        },
        "gates": {"erase_counts": {"Background": 1, "RigidNodes": 2}},
        "evaluation": {
            "edit_target_view": [91, 1],
            "development_views": [[5, 0]],
            "heldout_confirmation_views": [[20, 0]],
        },
    }

    config = build_spatial_config(
        replay=replay,
        template=template,
        instance_config=instance_config,
        instance_config_path=instance_config_path,
        instance_run=run,
    )

    assert config["composition"]["stacks"] == ["base_only", "erase"]
    assert "erase_counts" not in config["gates"]
    assert set(config["stage_abstentions"]) == {"roadpatch", "asset_harvester"}
    assert config["evaluation"]["edit_target_view"] == [2, 0]
    assert config["evaluation"]["heldout_confirmation_views"] == []
    assert config["resources"] == {}
    assert config["provenance"]["heldout_content_read"] is False
