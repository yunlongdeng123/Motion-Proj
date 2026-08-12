#!/usr/bin/env python
"""从冻结 nuScenes/DriveStudio 资产生成 train-only SAM2 box prompt manifest。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import (
    contiguous_blocks,
    project_box_prompt,
    sha256_file,
    validate_actor_identity_contract,
    validate_disjoint_split,
)


def resolve_optimization_frames(
    config: Mapping[str, Any], frame_count: int
) -> tuple[list[int], list[int], list[int]]:
    """返回 train/development/heldout，并保证开发帧不会进入语义优化。"""
    split = config["split"]
    heldout = sorted({int(value) for value in split["heldout_frames"]})
    development = sorted(
        {int(value) for value in split.get("development_frames", [])}
    )
    validate_disjoint_split(development, heldout)
    excluded = set(development) | set(heldout)
    if any(value < 0 or value >= frame_count for value in excluded):
        raise ValueError("development/heldout frame 超出 scene frame_count")
    train = [frame for frame in range(frame_count) if frame not in excluded]
    validate_disjoint_split(train, sorted(excluded))
    return train, development, heldout


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是 mapping: {path}")
    return payload


def annotation_by_frame(actor: dict[str, Any]) -> dict[int, dict[str, Any]]:
    values = actor["frame_annotations"]
    frame_indices = values["frame_idx"]
    object_to_world = values["obj_to_world"]
    box_sizes = values["box_size"]
    if not (len(frame_indices) == len(object_to_world) == len(box_sizes)):
        raise ValueError("actor frame annotation 字段长度不一致")
    return {
        int(frame): {
            "obj_to_world": obj_to_world,
            "box_size": box_size,
        }
        for frame, obj_to_world, box_size in zip(
            frame_indices, object_to_world, box_sizes
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.output_dir.exists():
        raise FileExistsError(f"S1 prompt output 已存在: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    scene_dir = Path(config["scene"]["processed_scene_dir"])
    instances_path = scene_dir / "instances/instances_info.json"
    registry_path = Path(config["inputs"]["actor_registry"])
    checkpoint_path = Path(config["inputs"]["checkpoint"])
    expected = config["inputs"]
    if sha256_file(instances_path) != config["scene"]["instances_info_sha256"]:
        raise RuntimeError("instances_info SHA 漂移")
    if sha256_file(registry_path) != expected["actor_registry_sha256"]:
        raise RuntimeError("actor registry SHA 漂移")
    if sha256_file(checkpoint_path) != expected["checkpoint_sha256"]:
        raise RuntimeError("D2 checkpoint SHA 漂移")

    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_by_token = {
        str(row["instance_token"]): row for row in registry["actors"]
    }
    frame_count = int(config["scene"]["frame_count"])
    train_frames, development, heldout = resolve_optimization_frames(
        config, frame_count
    )
    blocks = contiguous_blocks(train_frames)

    actor_rows: dict[str, dict[str, Any]] = {}
    for role, actor_config in config["actors"].items():
        dataset_id = str(int(actor_config["dataset_instance_id"]))
        actor_source = instances[dataset_id]
        token = str(actor_config["instance_token"])
        if token not in registry_by_token:
            raise RuntimeError(f"{role} actor registry token 不存在: {token}")
        validate_actor_identity_contract(
            role=role,
            actor_config=actor_config,
            dataset_instance=actor_source,
            registry_actor=registry_by_token[token],
        )
        actor_rows[role] = {
            "config": actor_config,
            "annotations": annotation_by_frame(actor_source),
        }

    manifest_blocks: list[dict[str, Any]] = []
    cameras = config["scene"]["cameras"]
    for camera in cameras:
        camera_id = int(camera["id"])
        intrinsics_path = scene_dir / "intrinsics" / f"{camera_id}.txt"
        values = np.loadtxt(intrinsics_path, dtype=np.float64).reshape(-1)
        if values.size < 4:
            raise RuntimeError(f"相机内参格式不合法: {intrinsics_path}")
        intrinsics = np.asarray(
            [[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        for block_index, frames in enumerate(blocks):
            block_dir = (
                args.output_dir
                / "sam_inputs"
                / f"camera_{camera_id}"
                / f"block_{block_index:03d}"
            )
            block_dir.mkdir(parents=True)
            frame_rows: list[dict[str, Any]] = []
            prompt_candidates: dict[str, list[dict[str, Any]]] = {
                role: [] for role in actor_rows
            }
            for local_index, frame in enumerate(frames):
                image_path = scene_dir / "images" / f"{frame:03d}_{camera_id}.jpg"
                extrinsics_path = scene_dir / "extrinsics" / f"{frame:03d}_{camera_id}.txt"
                if not image_path.is_file() or not extrinsics_path.is_file():
                    raise FileNotFoundError(f"S1 view 缺失: frame={frame} camera={camera_id}")
                with Image.open(image_path) as image:
                    width, height = image.size
                link = block_dir / f"{local_index:05d}.jpg"
                link.symlink_to(image_path)
                frame_row = {
                    "frame": frame,
                    "local_index": local_index,
                    "image": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "extrinsics": str(extrinsics_path),
                    "extrinsics_sha256": sha256_file(extrinsics_path),
                    "width": width,
                    "height": height,
                    "projected_boxes": {},
                }
                frame_rows.append(frame_row)
                c2w = np.loadtxt(extrinsics_path, dtype=np.float64).reshape(4, 4)
                for role, actor in actor_rows.items():
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
                        minimum_depth_m=float(config["prompts"]["minimum_depth_m"]),
                        padding_fraction=float(config["prompts"]["padding_fraction"]),
                        minimum_side_pixels=float(config["prompts"]["minimum_side_pixels"]),
                    )
                    if box is not None:
                        frame_row["projected_boxes"][role] = [
                            float(value) for value in box
                        ]
                        prompt_candidates[role].append(
                            {
                                "role": role,
                                "instance_token": actor["config"]["instance_token"],
                                "object_id": int(actor["config"]["sam_object_id"]),
                                "frame": frame,
                                "local_index": local_index,
                                "box_xyxy": [float(value) for value in box],
                                "box_source": "processed_nuscenes_3d_box_projection",
                            }
                        )
            prompts = [values[0] for values in prompt_candidates.values() if values]
            if prompts:
                manifest_blocks.append(
                    {
                        "camera_id": camera_id,
                        "camera_name": camera["name"],
                        "block_index": block_index,
                        # 使用相对路径，保证原子发布目录改名后 manifest 仍可复现。
                        "video_dir": str(block_dir.relative_to(args.output_dir)),
                        "frames": frame_rows,
                        "prompts": prompts,
                    }
                )

    manifest = {
        "schema_version": "worldsim_v32_s1_prompt_manifest_v1",
        "task_id": config["task_id"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "scene": config["scene"]["name"],
        "train_frames": train_frames,
        "development_frames": development,
        "heldout_frames": heldout,
        "development_excluded": True,
        "heldout_excluded": True,
        "instances_info": str(instances_path),
        "instances_info_sha256": sha256_file(instances_path),
        "intrinsics_sha256": {
            str(int(camera["id"])): sha256_file(
                scene_dir / "intrinsics" / f"{int(camera['id'])}.txt"
            )
            for camera in cameras
        },
        "actors": {
            role: {
                key: value
                for key, value in actor["config"].items()
                if key not in {"notes"}
            }
            for role, actor in actor_rows.items()
        },
        "blocks": manifest_blocks,
        "block_count": len(manifest_blocks),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "actor_registry_sha256": sha256_file(registry_path),
        "actor_identity_contract": "validated",
    }
    atomic_json(args.output_dir / "prompt_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "done",
                "block_count": len(manifest_blocks),
                "train_frames": len(train_frames),
                "heldout_frames": len(heldout),
                "manifest": str(args.output_dir / "prompt_manifest.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
