#!/usr/bin/env python
"""把旧 O0 扁平 occupancy 拆为 base evidence + 稀疏 per-instance OBB layers。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.observation_evidence import (
    RAY_FREE,
    STATIC_OCCUPIED,
    UNKNOWN,
    ObservationEvidenceFrame,
)
from motion_proj.resim.safety_geometry import (
    GridSpec,
    OrientedBox,
    voxelize_aabb_baseline,
)
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _yaw(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _frame_boxes(raw: dict, frame: int, world_to_grid: np.ndarray) -> list[tuple[int, OrientedBox]]:
    values = []
    for true_id, actor in raw.items():
        if not actor.get("class_name", "").startswith("vehicle"):
            continue
        annotations = actor["frame_annotations"]
        try:
            index = [int(value) for value in annotations["frame_idx"]].index(frame)
        except ValueError:
            continue
        object_to_world = np.asarray(annotations["obj_to_world"][index], dtype=float)
        object_to_grid = world_to_grid @ object_to_world
        values.append(
            (
                int(true_id) + 1,
                OrientedBox(
                    tuple(object_to_grid[:3, 3]),
                    tuple(float(value) for value in annotations["box_size"][index]),
                    _yaw(object_to_grid),
                    int(true_id) + 1,
                ),
            )
        )
    return values


def _sparse_layers(layers: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actor_ids, offsets, flat = [], [0], []
    for actor_id, mask in sorted(layers.items()):
        indices = np.flatnonzero(mask).astype(np.int32)
        actor_ids.append(actor_id)
        flat.append(indices)
        offsets.append(offsets[-1] + len(indices))
    return (
        np.asarray(actor_ids, dtype=np.int32),
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(flat) if flat else np.asarray([], dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/resim/v71/observation_evidence_v2.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"evidence output 已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    grid_cfg = config["grid"]
    grid = GridSpec(
        tuple(grid_cfg["minimum"]), tuple(grid_cfg["maximum"]),
        float(grid_cfg["voxel_size"]),
    )
    processed_root = Path(config["processed_root"])
    legacy_root = Path(config["legacy_root"])
    start, end = (int(value) for value in config["frames"])
    global_summary = {
        "schema_version": "observation-evidence-v2-build",
        "task_id": "V7-H1-11B",
        "config_sha256": file_fingerprint(str(args.config)),
        "scenes": {},
    }
    for scene_id in config["scenes"]:
        scene_out = output_root / scene_id
        scene_out.mkdir()
        raw_path = processed_root / scene_id / "instances" / "instances_info.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        stats = defaultdict(lambda: defaultdict(int))
        scene_counts = defaultdict(int)
        frame_hashes = {}
        for frame in range(start, end + 1):
            legacy_path = legacy_root / scene_id / f"frame_{frame:03d}.npz"
            legacy = np.load(legacy_path, allow_pickle=False)
            evidence = ObservationEvidenceFrame.from_legacy_o0(
                semantics=legacy["semantics"],
                mask_lidar=legacy["mask_lidar"],
                instance_id=legacy["instance_id"],
                grid=grid,
            )
            lidar_to_world = np.loadtxt(
                processed_root / scene_id / "lidar_pose" / f"{frame:03d}.txt"
            ).reshape(4, 4)
            world_to_grid = np.linalg.inv(lidar_to_world)
            # 丢弃旧 AABB layer，按 raw annotation 重新生成 OBB layer。
            edited = ObservationEvidenceFrame(
                grid, evidence.base_state, evidence.observed_count,
                evidence.observation_age_frames, {}, evidence.source,
            )
            coarse_voxels = 0
            overlap_count = np.zeros(grid.shape, dtype=np.uint8)
            for actor_id, box in _frame_boxes(raw, frame, world_to_grid):
                edited = edited.with_actor_box(actor_id, box)
                layer = edited.dynamic_instance_layers[actor_id]
                coarse_voxels += int(voxelize_aabb_baseline(box, grid).sum())
                overlap_count += layer.astype(np.uint8)
                actor_stats = stats[actor_id]
                actor_stats["frames"] += 1
                actor_stats["obb_voxels"] += int(layer.sum())
                actor_stats["base_unknown_voxels"] += int(
                    np.count_nonzero(layer & (edited.base_state == UNKNOWN))
                )
                actor_stats["base_free_voxels"] += int(
                    np.count_nonzero(layer & (edited.base_state == RAY_FREE))
                )
                actor_stats["base_static_voxels"] += int(
                    np.count_nonzero(layer & (edited.base_state == STATIC_OCCUPIED))
                )
            composite, composite_ids = edited.composite()
            actor_ids, offsets, flat_indices = _sparse_layers(
                dict(edited.dynamic_instance_layers)
            )
            output_path = scene_out / f"frame_{frame:03d}.npz"
            np.savez_compressed(
                output_path,
                base_state=edited.base_state,
                observed_count=edited.observed_count,
                observation_age_frames=edited.observation_age_frames,
                dynamic_actor_ids=actor_ids,
                dynamic_layer_offsets=offsets,
                dynamic_layer_flat_indices=flat_indices,
                composite_semantics=composite,
                composite_instance_id=composite_ids,
            )
            frame_hashes[f"{frame:03d}"] = file_fingerprint(str(output_path))
            scene_counts["frames"] += 1
            scene_counts["obb_voxels"] += sum(
                int(mask.sum()) for mask in edited.dynamic_instance_layers.values()
            )
            scene_counts["coarse_aabb_voxels"] += coarse_voxels
            scene_counts["overlap_voxels"] += int(np.count_nonzero(overlap_count > 1))
            scene_counts["base_unknown_voxels"] += int(
                np.count_nonzero(edited.base_state == UNKNOWN)
            )
            scene_counts["base_free_voxels"] += int(
                np.count_nonzero(edited.base_state == RAY_FREE)
            )
            scene_counts["base_static_voxels"] += int(
                np.count_nonzero(edited.base_state == STATIC_OCCUPIED)
            )
        actor_summary = {}
        for actor_id, value in sorted(stats.items()):
            value = dict(value)
            volume = value["obb_voxels"]
            value["base_unknown_ratio"] = (
                value["base_unknown_voxels"] / volume if volume else None
            )
            actor_summary[str(actor_id)] = value
        summary = {
            "scene_id": scene_id,
            "counts": dict(scene_counts),
            "coarse_to_oriented_voxel_ratio": (
                scene_counts["coarse_aabb_voxels"] / scene_counts["obb_voxels"]
                if scene_counts["obb_voxels"] else None
            ),
            "actors": actor_summary,
            "frame_artifact_sha256": frame_hashes,
            "source_raw_sha256": file_fingerprint(str(raw_path)),
        }
        summary["scene_evidence_sha256"] = canonical_sha256(summary)
        atomic_write_json(str(scene_out / "summary.json"), summary)
        global_summary["scenes"][scene_id] = summary
    global_summary["evidence_set_sha256"] = canonical_sha256(global_summary["scenes"])
    atomic_write_json(str(output_root / "summary.json"), global_summary)
    print(json.dumps(global_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
