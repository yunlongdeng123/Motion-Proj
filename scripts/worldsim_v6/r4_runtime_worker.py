#!/usr/bin/env python3
"""在单个全新进程内执行一次 SceneIR deterministic runtime episode。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v6.sceneir import load_sceneir
from motion_proj.worldsim_v6.sceneir_adapters import quaternion_to_matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _fixed(value: float, scale: int) -> int:
    return int(
        (Decimal(str(value)) * Decimal(scale)).to_integral_value(rounding=ROUND_HALF_EVEN)
    )


def _vector_fixed(values: list[float], scale: int) -> list[int]:
    return [_fixed(value, scale) for value in values]


def _transform_index(document: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result = {}
    for row in document["transforms"]:
        key = (row["src_frame"], int(row["timestamp_us"]))
        if key in result:
            raise RuntimeError(f"重复 transform：{key}")
        result[key] = row
    return result


def _states(
    document: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    episode = config["episode"]
    sensor_cfg = config["sensor"]
    label_cfg = config["labels"]
    action_cfg = config["action"]
    timestamps = [
        int(episode["initial_timestamp_us"]) + index * int(episode["step_delta_us"])
        for index in range(int(episode["step_count"]))
    ]
    available = set(document["support"][0]["observed_timestamp_us"])
    if not set(timestamps) <= available:
        raise RuntimeError("runtime timestamp 超出 observed support")
    initial_sensor_um = _vector_fixed(sensor_cfg["initial_translation_m"], 1_000_000)
    delta_sensor_um = _vector_fixed(action_cfg["sensor_delta_per_step_m"], 1_000_000)
    collision_mm = _fixed(label_cfg["collision_radius_m"], 1_000)
    near_mm = _fixed(label_cfg["near_range_m"], 1_000)
    mid_mm = _fixed(label_cfg["mid_range_m"], 1_000)
    view_range_mm = _fixed(sensor_cfg["view_range_m"], 1_000)
    transforms = _transform_index(document)
    world_steps = []
    label_steps = []
    chunk_steps = []
    trajectory = {actor["id"]: [] for actor in document["actors"]}
    actor_chunks = {actor["id"]: sorted(actor["chunk_ids"]) for actor in document["actors"]}
    for step_index, timestamp_us in enumerate(timestamps):
        sensor_um = [
            initial_sensor_um[axis] + step_index * delta_sensor_um[axis] for axis in range(3)
        ]
        actor_states = []
        actor_labels = []
        selected_chunks = list(document["static_world"]["chunk_ids"])
        for actor in sorted(document["actors"], key=lambda row: row["id"]):
            transform = transforms[(actor["canonical_frame"], timestamp_us)]
            translation_um = _vector_fixed(transform["translation_m"], 1_000_000)
            rotation_q1e9 = _vector_fixed(transform["rotation_wxyz"], 1_000_000_000)
            state = {
                "actor_id": actor["id"],
                "translation_um": translation_um,
                "rotation_wxyz_q1e9": rotation_q1e9,
            }
            actor_states.append(state)
            trajectory[actor["id"]].append({"timestamp_us": timestamp_us, **state})
            squared_um = sum(
                (translation_um[axis] - sensor_um[axis]) ** 2 for axis in range(3)
            )
            distance_mm = math.isqrt(squared_um) // 1000
            if distance_mm < near_mm:
                bucket = "near"
            elif distance_mm < mid_mm:
                bucket = "mid"
            else:
                bucket = "far"
            actor_labels.append(
                {
                    "actor_id": actor["id"],
                    "distance_mm": distance_mm,
                    "range_bucket": bucket,
                    "collision_proxy": distance_mm <= collision_mm,
                    "observed_support": True,
                }
            )
            if distance_mm <= view_range_mm:
                selected_chunks.extend(actor_chunks[actor["id"]])
        world_steps.append(
            {
                "step_index": step_index,
                "timestamp_us": timestamp_us,
                "sensor_translation_um": sensor_um,
                "actors": actor_states,
            }
        )
        label_steps.append(
            {
                "step_index": step_index,
                "timestamp_us": timestamp_us,
                "actors": actor_labels,
                "collision_actor_ids": sorted(
                    row["actor_id"] for row in actor_labels if row["collision_proxy"]
                ),
            }
        )
        chunk_steps.append(
            {
                "step_index": step_index,
                "timestamp_us": timestamp_us,
                "chunk_ids": sorted(set(selected_chunks)),
            }
        )
    common = {
        "schema_version": "worldsim_v6.r4_runtime_output.v1",
        "episode_id": episode["id"],
        "seed": int(config["seed"]),
    }
    return (
        {**common, "action": action_cfg, "sensor": sensor_cfg, "steps": world_steps},
        {**common, "steps": label_steps},
        {**common, "steps": chunk_steps},
        {
            **common,
            "actors": [
                {"actor_id": actor_id, "states": trajectory[actor_id]}
                for actor_id in sorted(trajectory)
            ],
        },
    )


def _rasterize(
    document: Mapping[str, Any],
    blobs: Mapping[str, np.ndarray],
    world_state: Mapping[str, Any],
    chunk_selection: Mapping[str, Any],
) -> np.ndarray:
    sensor = world_state["sensor"]
    width, height = (int(value) for value in sensor["resolution_px"])
    extent = float(sensor["extent_m"])
    stride = int(sensor["primitive_stride"])
    chunk_by_id = {row["id"]: row for row in document["chunks"]}
    transforms = _transform_index(document)
    frames = []
    for world_step, chunk_step in zip(world_state["steps"], chunk_selection["steps"]):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        sensor_m = np.asarray(world_step["sensor_translation_um"], dtype=np.float64) / 1_000_000.0
        point_rows = []
        for chunk_id in chunk_step["chunk_ids"]:
            chunk = chunk_by_id[chunk_id]
            means = blobs[chunk["arrays"]["means_m"]["sha256"]]
            features = blobs[chunk["arrays"]["features_dc"]["sha256"]]
            source_indices = blobs[chunk["arrays"]["source_indices"]["sha256"]]
            keep = np.remainder(source_indices, stride) == 0
            means = means[keep].astype(np.float32, copy=False)
            features = features[keep].astype(np.float32, copy=False)
            source_indices = source_indices[keep].astype(np.int64, copy=False)
            role = 0
            if chunk["role"] == "actor":
                transform = transforms[(chunk["frame_id"], int(world_step["timestamp_us"]))]
                rotation = quaternion_to_matrix(
                    np.asarray(transform["rotation_wxyz"], dtype=np.float32)
                )
                translation = np.asarray(transform["translation_m"], dtype=np.float32)
                means = means @ rotation.T + translation
                role = 1
            colors = np.rint(
                np.clip(features * np.float32(0.28209479177387814) + 0.5, 0.0, 1.0)
                * 255.0
            ).astype(np.uint8)
            if role == 1:
                colors[:, 0] = np.maximum(colors[:, 0], 192)
            point_rows.append((means, colors, source_indices, role))
        for means, colors, source_indices, role in sorted(point_rows, key=lambda row: row[3]):
            x = means[:, 0].astype(np.float64) - sensor_m[0]
            y = means[:, 1].astype(np.float64) - sensor_m[1]
            u = np.floor((x / extent + 0.5) * width).astype(np.int64)
            v = np.floor((0.5 - y / extent) * height).astype(np.int64)
            visible = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            order = np.argsort(source_indices[visible], kind="stable")
            for pixel_x, pixel_y, color in zip(u[visible][order], v[visible][order], colors[visible][order]):
                image[pixel_y, pixel_x] = color
        center_x, center_y = width // 2, height // 2
        image[max(0, center_y - 1) : center_y + 2, max(0, center_x - 1) : center_x + 2] = [0, 128, 255]
        frames.append(image)
    return np.stack(frames, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    package = args.package.resolve()
    manifest_before = _sha256(package / "MANIFEST.json")
    document, blobs = load_sceneir(package)
    if document["content_sha256"] != config["source"]["sceneir_content_sha256"]:
        raise RuntimeError("SceneIR content hash 漂移")
    world_state, labels, chunk_selection, actor_trajectory = _states(document, config)
    _write_json(output / "WORLD_STATE.json", world_state)
    _write_json(output / "LABELS.json", labels)
    _write_json(output / "CHUNK_SELECTION.json", chunk_selection)
    _write_json(output / "ACTOR_TRAJECTORY.json", actor_trajectory)
    rgb = _rasterize(document, blobs, world_state, chunk_selection)
    np.save(output / "RGB.npy", rgb, allow_pickle=False)
    manifest_after = _sha256(package / "MANIFEST.json")
    if manifest_before != manifest_after:
        raise RuntimeError("SceneIR package 在 runtime 前后漂移")
    outputs = {
        name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
        for name in config["determinism_gate"]["exact_files"]
    }
    _write_json(
        output / "RUNTIME_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r4_runtime_audit.v1",
            "sceneir_content_sha256": document["content_sha256"],
            "package_manifest_sha256_before": manifest_before,
            "package_manifest_sha256_after": manifest_after,
            "chunk_count": len(document["chunks"]),
            "actor_count": len(document["actors"]),
            "primitive_count": sum(row["primitive_count"] for row in document["chunks"]),
            "unique_blob_count": len(blobs),
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "rgb_renderer": "fixed_order_cpu_topdown_chunk_diagnostic_v0",
            "training_started": False,
            "confirmation_content_read": False,
            "outputs": outputs,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
