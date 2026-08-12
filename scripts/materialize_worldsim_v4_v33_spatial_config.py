#!/usr/bin/env python3
"""物化逐场景 erase-only spatial delta；scene-0230 专属资产显式弃权。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.v33_replay import V33ReplayError, load_yaml, sha256_file


def _verified_file(path: str | Path, expected: str, label: str) -> Path:
    value = Path(path)
    if not value.is_file() or sha256_file(value) != expected:
        raise V33ReplayError(f"{label} 缺失或 SHA 漂移")
    return value


def build_spatial_config(
    *,
    replay: Mapping[str, Any],
    template: Mapping[str, Any],
    instance_config: Mapping[str, Any],
    instance_config_path: Path,
    instance_run: Path,
) -> dict[str, Any]:
    if instance_config.get("schema_version") != "worldsim_v4_v33_instance_field_v1":
        raise V33ReplayError("instance config schema 漂移")
    stage_path = instance_run / "stage_summary.json"
    status_path = instance_run / "status.json"
    inner_summary_path = instance_run / "instance_field" / "summary.json"
    eval_mask_path = instance_run / "eval_targets" / "artifacts" / "masks" / "mask_manifest.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    inner = json.loads(inner_summary_path.read_text(encoding="utf-8"))
    eval_masks = json.loads(eval_mask_path.read_text(encoding="utf-8"))
    if stage.get("status") != "done" or status.get("status") != "done":
        raise V33ReplayError("instance stage 尚未完成")
    if stage.get("evaluation_partition") != "development":
        raise V33ReplayError("instance stage 不是 development-only")
    if stage.get("config_sha256") != sha256_file(instance_config_path):
        raise V33ReplayError("instance stage/config SHA 漂移")
    if stage.get("instance_summary_sha256") != sha256_file(inner_summary_path):
        raise V33ReplayError("instance terminal summary SHA 漂移")
    if stage.get("heldout_content_read") is not False or stage.get(
        "test_quality_read"
    ) is not False:
        raise V33ReplayError("instance stage 读取了 sealed/test 内容")

    selected_arm = replay["instance_field"]["formal_selected_arm"]
    arm = inner.get("arms", {}).get(selected_arm)
    if arm is None:
        raise V33ReplayError(f"instance summary 缺少 frozen arm: {selected_arm}")
    field = _verified_file(
        arm["instance_field"], arm["instance_field_sha256"], "instance field"
    )
    inputs = instance_config["inputs"]
    for key in ("checkpoint", "source_config", "actor_registry"):
        _verified_file(inputs[key], inputs[f"{key}_sha256"], key)
    actor = instance_config["actors"].get("high_support")
    if actor is None:
        raise V33ReplayError("spatial delta 缺少 high_support actor")
    candidates = [
        row
        for row in eval_masks.get("masks", [])
        if row.get("role") == "high_support"
        and bool(row.get("accepted"))
        and int(row.get("positive_pixels", 0)) > 0
    ]
    candidates.sort(key=lambda row: (int(row["frame"]), int(row["camera_id"])))
    if not candidates:
        raise V33ReplayError("development eval masks 缺少 accepted high_support target")
    target = candidates[0]
    target_mask = _verified_file(target["mask"], target["mask_sha256"], "target mask")
    development_views = [
        [int(row["frame"]), int(row["camera_id"])] for row in candidates[1:3]
    ]

    gates = dict(template["gates"])
    gates.pop("erase_counts", None)
    stage_abstentions = {
        "roadpatch": {
            "status": "abstain",
            "reason": "scene_specific_v33_roadpatch_assets_not_available_under_v4_mod5",
        },
        "asset_harvester": {
            "status": "abstain",
            "reason": "scene_specific_v33_actor_asset_not_available_under_v4_mod5",
        },
    }
    return {
        "schema_version": "worldsim_v4_v33_spatial_delta_v1",
        "task_id": replay["task_id"],
        "seed": int(template["seed"]),
        "inputs": {
            "checkpoint": {
                "path": inputs["checkpoint"],
                "sha256": inputs["checkpoint_sha256"],
            },
            "source_config": {
                "path": inputs["source_config"],
                "sha256": inputs["source_config_sha256"],
            },
            "actor_registry": {
                "path": inputs["actor_registry"],
                "sha256": inputs["actor_registry_sha256"],
            },
            "s1_instance_field": {
                "path": str(field),
                "sha256": arm["instance_field_sha256"],
            },
            "s1_summary": {
                "path": str(inner_summary_path),
                "sha256": sha256_file(inner_summary_path),
            },
            "target_mask": {
                "path": str(target_mask),
                "sha256": target["mask_sha256"],
            },
        },
        "scene": {
            "name": instance_config["scene"]["name"],
            "processed_root": instance_config["scene"]["processed_scene_dir"],
            "model_native_width": instance_config["outputs"]["model_native_width"],
            "model_native_height": instance_config["outputs"]["model_native_height"],
            "cameras": [
                int(camera["id"]) for camera in instance_config["scene"]["cameras"]
            ],
            "development_frames": list(
                instance_config["split"]["development_frames"]
            ),
            "heldout_frames": list(instance_config["split"]["heldout_frames"]),
        },
        "actor": {
            "role": "high_support",
            "dataset_instance_id": int(actor["dataset_instance_id"]),
            "instance_token": str(actor["instance_token"]),
            "rigid_model_index": int(actor["rigid_model_index"]),
        },
        "composition": {
            **dict(template["composition"]),
            "stacks": list(
                replay["spatial_delta"][
                    "executable_stacks_without_scene_specific_assets"
                ]
            ),
        },
        "stage_abstentions": stage_abstentions,
        "evaluation": {
            **dict(template["evaluation"]),
            "edit_target_view": [int(target["frame"]), int(target["camera_id"])],
            "development_views": development_views,
            "heldout_confirmation_views": [],
        },
        "gates": gates,
        "resources": dict(template["resources"]),
        "runtimes": {
            "drivestudio_checkout": replay["runtimes"]["drivestudio_checkout"],
            "drivestudio_python": replay["runtimes"]["drivestudio_python"],
        },
        "provenance": {
            "algorithm_commit": replay["algorithm"]["implementation_commit"],
            "instance_config": str(instance_config_path),
            "instance_config_sha256": sha256_file(instance_config_path),
            "base_rgb_immutable": True,
            "development_content_read": True,
            "development_optimization_read": False,
            "heldout_content_read": False,
            "test_quality_read": False,
        },
    }


def atomic_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--instance-config", type=Path, required=True)
    parser.add_argument("--instance-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = load_yaml(args.replay_config)
    template_path = args.project_root / replay["spatial_delta"]["template"]
    if sha256_file(template_path) != replay["spatial_delta"]["template_sha256"]:
        raise V33ReplayError("spatial template SHA 漂移")
    config = build_spatial_config(
        replay=replay,
        template=load_yaml(template_path),
        instance_config=load_yaml(args.instance_config),
        instance_config_path=args.instance_config.resolve(),
        instance_run=args.instance_run.resolve(),
    )
    atomic_yaml(args.output, config)
    print(
        json.dumps(
            {
                "status": "done",
                "scene": config["scene"]["name"],
                "stacks": config["composition"]["stacks"],
                "stage_abstentions": config["stage_abstentions"],
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "test_quality_read": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
