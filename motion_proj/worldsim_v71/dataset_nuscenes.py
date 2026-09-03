"""nuScenes V7.1 Actor tracklet 数据入口。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from motion_proj.worldsim_v7.nuscenes_actor_surface import (
    build_selected_index,
    compile_nuscenes_scene,
)


RIGID_NUSCENES_PREFIXES = (
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "vehicle.construction",
    "vehicle.trailer",
)


def freeze_source_split(
    dataset_root: Path,
    *,
    prior_scene_names: Iterable[str],
    train_count: int = 120,
    selection_count: int = 20,
    final_count: int = 20,
    minimum_lidar_frames: int = 9,
) -> dict[str, Any]:
    """只按 metadata 与文件存在性冻结角色，不读取 Actor 几何质量。"""
    metadata_root = dataset_root / "v1.0-trainval"
    scenes = json.loads((metadata_root / "scene.json").read_text(encoding="utf-8"))
    samples = {
        str(row["token"]): str(row["scene_token"])
        for row in json.loads((metadata_root / "sample.json").read_text(encoding="utf-8"))
    }
    frame_counts: Counter[str] = Counter()
    for row in json.loads((metadata_root / "sample_data.json").read_text(encoding="utf-8")):
        if not bool(row["is_key_frame"]) or not str(row["filename"]).startswith("samples/LIDAR_TOP/"):
            continue
        if (dataset_root / str(row["filename"])).is_file():
            frame_counts[samples[str(row["sample_token"])]] += 1
    available = [
        str(row["name"])
        for row in scenes
        if frame_counts[str(row["token"])] >= int(minimum_lidar_frames)
    ]
    prior = set(str(name) for name in prior_scene_names)
    never_used = [name for name in available if name not in prior]
    if len(never_used) < selection_count + final_count:
        raise RuntimeError("可用且未被 V7 使用的 scene 不足以冻结 selection/final")
    final = never_used[:final_count]
    selection = never_used[final_count : final_count + selection_count]
    reserved = set(final) | set(selection)
    train = [name for name in available if name not in reserved][:train_count]
    if len(train) < train_count:
        raise RuntimeError("可用 scene 不足以冻结 V7.1 train")
    return {
        "schema_version": "worldsim_v71.source_split.v1",
        "selection_rule": "official_scene_order_available_lidar_prior_v7_excluded_from_selection_final",
        "dataset_root": str(dataset_root),
        "minimum_lidar_frames": int(minimum_lidar_frames),
        "roles": {"train": train, "selection": selection, "source_final": final},
        "role_counts": {"train": len(train), "selection": len(selection), "source_final": len(final)},
        "roles_disjoint": len(set(train) | set(selection) | set(final)) == len(train) + len(selection) + len(final),
        "quality_read": False,
    }


def build_v71_index(dataset_root: Path, split: Mapping[str, Any]) -> dict[str, Any]:
    roles = {str(role): list(names) for role, names in split["roles"].items()}
    return build_selected_index(dataset_root, roles, list(RIGID_NUSCENES_PREFIXES))


def compile_source_scene(
    scene_name: str,
    index: Mapping[str, Any],
    actor_config: Mapping[str, Any],
    compiler_config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows, diagnostics = compile_nuscenes_scene(
        scene_name,
        index["scenes"][scene_name],
        Path(index["dataset_root"]),
        actor_config,
        compiler_config,
        device,
        include_diagnostics=True,
    )
    return [
        {"scene_name": scene_name, "row": row, "diagnostics": diagnostics[row["track_id"]]}
        for row in rows
    ]
