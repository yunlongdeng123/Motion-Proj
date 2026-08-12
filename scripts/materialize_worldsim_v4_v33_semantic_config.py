#!/usr/bin/env python3
"""从冻结 replay/registry 合同物化单场景 V3.3 semantic-lift 配置。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.v33_replay import V33ReplayError, load_yaml, sha256_file


IMAGE_NAME = re.compile(r"^(?P<frame>\d+)_(?P<camera>\d+)\.jpg$")


def discover_scene_layout(
    processed_scene: Path, cameras: list[Mapping[str, Any]]
) -> int:
    expected_cameras = {int(camera["id"]) for camera in cameras}
    frames_by_camera = {camera: set() for camera in expected_cameras}
    image_dir = processed_scene / "images"
    for path in image_dir.glob("*.jpg"):
        match = IMAGE_NAME.match(path.name)
        if match is None:
            continue
        camera = int(match.group("camera"))
        if camera in frames_by_camera:
            frames_by_camera[camera].add(int(match.group("frame")))
    if not frames_by_camera or any(not frames for frames in frames_by_camera.values()):
        raise V33ReplayError("semantic cameras 缺少 RGB frames")
    reference = next(iter(frames_by_camera.values()))
    if any(frames != reference for frames in frames_by_camera.values()):
        raise V33ReplayError("semantic cameras 的 frame 集不一致")
    if reference != set(range(len(reference))):
        raise V33ReplayError("processed scene frame 索引必须从 0 连续")
    return len(reference)


def _verified_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise V33ReplayError(f"{label} 缺失或 SHA 漂移")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise V33ReplayError(f"{label} bytes 漂移")
    return path


def build_semantic_config(
    *,
    replay: Mapping[str, Any],
    template: Mapping[str, Any],
    bound_scene: Mapping[str, Any],
) -> dict[str, Any]:
    if bound_scene.get("schema_version") != "worldsim_v4_v33_bound_scene_v1":
        raise V33ReplayError("bound scene schema 漂移")
    if bound_scene.get("partition_contract") != "sample_index_mod_5":
        raise V33ReplayError("bound scene partition 漂移")
    if bound_scene.get("algorithm_commit") != replay["algorithm"]["implementation_commit"]:
        raise V33ReplayError("bound scene algorithm commit 漂移")
    if bound_scene.get("test_quality_read") is not False:
        raise V33ReplayError("semantic config 禁止读取 test quality")

    checkpoint = _verified_file(bound_scene["base_checkpoint"], "base checkpoint")
    registry = _verified_file(bound_scene["actor_registry"], "actor registry")
    source_config = Path(str(bound_scene["base_checkpoint"]["source_config"]))
    if (
        not source_config.is_file()
        or sha256_file(source_config)
        != bound_scene["base_checkpoint"]["source_config_sha256"]
    ):
        raise V33ReplayError("base source config 缺失或 SHA 漂移")
    instances = _verified_file(bound_scene["instances_info"], "instances_info")

    semantic = replay["semantic_lift"]
    cameras = [dict(camera) for camera in semantic["cameras"]]
    processed_scene = Path(str(bound_scene["processed_scene"]))
    frame_count = discover_scene_layout(processed_scene, cameras)
    partition = replay["frame_partition"]
    development = [
        frame
        for frame in range(frame_count)
        if frame % int(partition["modulus"])
        == int(partition["development_remainder"])
    ]
    heldout = [
        frame
        for frame in range(frame_count)
        if frame % int(partition["modulus"])
        == int(partition["heldout_remainder"])
    ]

    actors: dict[str, Any] = {}
    actor_abstentions: dict[str, Any] = {}
    for object_id, role in enumerate(("high_support", "boundary_support"), 1):
        actor = bound_scene["actors"][role]
        if actor["availability"] != "available":
            actor_abstentions[role] = {
                "instance_token": actor["instance_token"],
                "availability": actor["availability"],
                "reason": "actor_registry_unavailable",
            }
            continue
        if actor.get("rigid_model_index") is None:
            raise V33ReplayError(f"available actor 缺少 rigid_model_index: {role}")
        actors[role] = {
            "instance_token": str(actor["instance_token"]),
            "dataset_instance_id": int(actor["dataset_instance_id"]),
            "rigid_model_index": int(actor["rigid_model_index"]),
            "sam_object_id": object_id,
            "class_name": str(actor["class_name"]),
        }
    if "high_support" not in actors:
        raise V33ReplayError("high_support actor 必须可执行")

    return {
        "schema_version": "worldsim_v4_v33_semantic_lift_v1",
        "task_id": replay["task_id"],
        "seed": int(template["seed"]),
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": bound_scene["base_checkpoint"]["sha256"],
            "source_config": str(source_config),
            "source_config_sha256": bound_scene["base_checkpoint"][
                "source_config_sha256"
            ],
            "actor_registry": str(registry),
            "actor_registry_sha256": bound_scene["actor_registry"]["sha256"],
        },
        "scene": {
            "name": bound_scene["scene"],
            "processed_scene_dir": str(processed_scene),
            "instances_info_sha256": bound_scene["instances_info"]["sha256"],
            "frame_count": frame_count,
            "cameras": cameras,
        },
        "split": {
            "partition_contract": "sample_index_mod_5",
            "train_remainders": list(partition["train_remainders"]),
            "development_frames": development,
            "heldout_frames": heldout,
            "development_usage": "evaluation_only",
            "heldout_usage": "sealed",
            "optimization_forbidden": True,
        },
        "actors": actors,
        "prompts": dict(template["prompts"]),
        "sam2": dict(template["sam2"]),
        "lift": dict(template["lift"]),
        "outputs": dict(template["outputs"]),
        "runtimes": {
            "sam_python": replay["runtimes"]["sam2_python"],
            "drivestudio_python": replay["runtimes"]["drivestudio_python"],
            "drivestudio_checkout": replay["runtimes"]["drivestudio_checkout"],
        },
        "provenance": {
            "algorithm_commit": replay["algorithm"]["implementation_commit"],
            "base_rgb_immutable": True,
            "instances_info": str(instances),
            "actor_abstentions": actor_abstentions,
            "development_content_read": False,
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
    parser.add_argument("--bound-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay = load_yaml(args.replay_config)
    template_path = args.project_root / replay["semantic_lift"]["template"]
    if sha256_file(template_path) != replay["semantic_lift"]["template_sha256"]:
        raise V33ReplayError("semantic template SHA 漂移")
    template = load_yaml(template_path)
    bound_scene = json.loads(args.bound_scene.read_text(encoding="utf-8"))
    config = build_semantic_config(
        replay=replay, template=template, bound_scene=bound_scene
    )
    atomic_yaml(args.output, config)
    print(
        json.dumps(
            {
                "status": "done",
                "scene": config["scene"]["name"],
                "frame_count": config["scene"]["frame_count"],
                "actor_count": len(config["actors"]),
                "actor_abstentions": config["provenance"]["actor_abstentions"],
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
