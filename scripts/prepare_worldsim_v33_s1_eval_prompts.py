#!/usr/bin/env python
"""为固定 heldout 帧生成 evaluation-only SAM2.1 projected-box prompts。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import (
    project_box_prompt,
    sha256_file,
    validate_actor_identity_contract,
)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def annotations_by_frame(actor: dict[str, Any]) -> dict[int, dict[str, Any]]:
    values = actor["frame_annotations"]
    frames = values["frame_idx"]
    transforms = values["obj_to_world"]
    sizes = values["box_size"]
    if not (len(frames) == len(transforms) == len(sizes)):
        raise ValueError("actor frame_annotations 长度不一致")
    return {
        int(frame): {"obj_to_world": transform, "box_size": size}
        for frame, transform, size in zip(frames, transforms, sizes)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖 heldout prompt 目录: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    scene_dir = Path(config["scene"]["processed_scene_dir"])
    instances_path = scene_dir / "instances/instances_info.json"
    registry_path = Path(inputs["actor_registry"])
    checkpoint_path = Path(inputs["checkpoint"])
    for name, path, expected in (
        ("instances_info", instances_path, config["scene"]["instances_info_sha256"]),
        ("actor_registry", registry_path, inputs["actor_registry_sha256"]),
        ("checkpoint", checkpoint_path, inputs["checkpoint_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"{name} SHA 漂移")
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_by_token = {str(row["instance_token"]): row for row in registry["actors"]}
    actors: dict[str, dict[str, Any]] = {}
    for role, actor_config in config["actors"].items():
        dataset_id = str(int(actor_config["dataset_instance_id"]))
        dataset_actor = instances[dataset_id]
        token = str(actor_config["instance_token"])
        if token not in registry_by_token:
            raise RuntimeError(f"{role} actor registry token 缺失")
        validate_actor_identity_contract(
            role=role,
            actor_config=actor_config,
            dataset_instance=dataset_actor,
            registry_actor=registry_by_token[token],
        )
        actors[role] = {
            "config": actor_config,
            "annotations": annotations_by_frame(dataset_actor),
        }

    blocks: list[dict[str, Any]] = []
    heldout = [int(value) for value in config["split"]["heldout_frames"]]
    for camera in config["scene"]["cameras"]:
        camera_id = int(camera["id"])
        intrinsics_path = scene_dir / "intrinsics" / f"{camera_id}.txt"
        raw = np.loadtxt(intrinsics_path, dtype=np.float64).reshape(-1)
        intrinsics = np.asarray(
            [[raw[0], 0.0, raw[2]], [0.0, raw[1], raw[3]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        for frame in heldout:
            image_path = scene_dir / "images" / f"{frame:03d}_{camera_id}.jpg"
            extrinsics_path = scene_dir / "extrinsics" / f"{frame:03d}_{camera_id}.txt"
            if not image_path.is_file() or not extrinsics_path.is_file():
                raise FileNotFoundError(f"heldout view 缺失: {frame}/{camera_id}")
            with Image.open(image_path) as image:
                width, height = image.size
            c2w = np.loadtxt(extrinsics_path, dtype=np.float64).reshape(4, 4)
            projected_boxes: dict[str, list[float]] = {}
            prompts: list[dict[str, Any]] = []
            for role, actor in actors.items():
                annotation = actor["annotations"].get(frame)
                if annotation is None:
                    continue
                box = project_box_prompt(
                    obj_to_world=np.asarray(annotation["obj_to_world"]),
                    box_size=np.asarray(annotation["box_size"]),
                    camera_to_world=c2w,
                    intrinsics=intrinsics,
                    image_width=width,
                    image_height=height,
                    minimum_depth_m=0.1,
                    padding_fraction=0.08,
                    minimum_side_pixels=8.0,
                )
                if box is None:
                    continue
                values = [float(value) for value in box]
                projected_boxes[role] = values
                prompts.append(
                    {
                        "role": role,
                        "instance_token": actor["config"]["instance_token"],
                        "object_id": int(actor["config"]["sam_object_id"]),
                        "frame": frame,
                        "local_index": 0,
                        "box_xyxy": values,
                        "box_source": "processed_nuscenes_3d_box_projection",
                    }
                )
            if not prompts:
                continue
            block_dir = args.output_dir / "sam_inputs" / f"camera_{camera_id}" / f"frame_{frame:03d}"
            block_dir.mkdir(parents=True)
            (block_dir / "00000.jpg").symlink_to(image_path)
            blocks.append(
                {
                    "camera_id": camera_id,
                    "camera_name": camera["name"],
                    "frame": frame,
                    "video_dir": str(block_dir.relative_to(args.output_dir)),
                    "frames": [
                        {
                            "frame": frame,
                            "local_index": 0,
                            "image": str(image_path),
                            "image_sha256": sha256_file(image_path),
                            "extrinsics": str(extrinsics_path),
                            "extrinsics_sha256": sha256_file(extrinsics_path),
                            "width": width,
                            "height": height,
                            "projected_boxes": projected_boxes,
                        }
                    ],
                    "prompts": prompts,
                }
            )
    manifest = {
        "schema_version": "worldsim_v33_s1_heldout_prompt_manifest_v1",
        "task_id": config["task_id"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "scene": config["scene"]["name"],
        "evaluation_frames": heldout,
        "optimization_forbidden": True,
        "instances_info": str(instances_path),
        "instances_info_sha256": sha256_file(instances_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "actor_registry_sha256": sha256_file(registry_path),
        "actor_identity_contract": "validated",
        "actors": {
            role: {
                key: value
                for key, value in actor["config"].items()
                if key not in {"semantic_sidecar", "semantic_sidecar_sha256"}
            }
            for role, actor in actors.items()
        },
        "blocks": blocks,
        "block_count": len(blocks),
    }
    atomic_json(args.output_dir / "prompt_manifest.json", manifest)
    print(json.dumps({"status": "done", "blocks": len(blocks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
