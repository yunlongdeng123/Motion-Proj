#!/usr/bin/env python
"""从 DriveStudio 初始化顺序构建 PILOT-3 actor registry。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.actor_registry import build_actor_registry
from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("world_state_v1.yaml 必须是 object")
    return config


def _occupancy_frame_counts(root: Path, scene_id: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for path in sorted((root / scene_id).glob("frame_*.npz")):
        instance_ids = np.load(path)["instance_id"]
        for value in np.unique(instance_ids):
            if int(value) > 0:
                counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def build_scene(scene: dict, config: dict) -> dict:
    drivestudio_root = Path(config["drivestudio_root"])
    sys.path.insert(0, str(drivestudio_root))
    from datasets.driving_dataset import DrivingDataset

    checkpoint = Path(scene["checkpoint"])
    train_config = OmegaConf.load(checkpoint.parent / "config.yaml")
    dataset = DrivingDataset(data_cfg=train_config.data)
    init = dataset.get_init_objects(
        cur_node_type="RigidNodes",
        instance_max_pts=int(train_config.model.RigidNodes.init.instance_max_pts),
        only_moving=bool(train_config.model.RigidNodes.init.only_moving),
        traj_length_thres=float(train_config.model.RigidNodes.init.traj_length_thres),
    )
    checkpoint_state = torch.load(checkpoint, map_location="cpu")
    rigid_state = checkpoint_state["models"]["RigidNodes"]
    checkpoint_count = int(rigid_state["instances_trans"].shape[1])
    checkpoint_sizes = rigid_state["instances_size"].cpu().numpy()
    checkpoint_sha256 = file_fingerprint(str(checkpoint))
    true_ids = dataset.pixel_source.instances_true_id.cpu().numpy()
    raw_path = (
        Path(config["processed_root"])
        / scene["scene_id"]
        / "instances"
        / "instances_info.json"
    )
    with raw_path.open(encoding="utf-8") as handle:
        raw_instances = json.load(handle)
    occupancy_counts = _occupancy_frame_counts(
        Path(config["occupancy_root"]), scene["scene_id"]
    )

    actors = []
    size_errors = []
    for model_index, (dataset_column, value) in enumerate(init.items()):
        true_id = int(true_ids[int(dataset_column)])
        raw = raw_instances[str(true_id)]
        frames = [int(frame) for frame in raw["frame_annotations"]["frame_idx"]]
        dimensions = value["size"].cpu().tolist()
        size_error = float(
            np.max(np.abs(np.asarray(dimensions) - checkpoint_sizes[model_index]))
        )
        size_errors.append(size_error)
        actors.append(
            {
                "true_instance_id": true_id,
                "dataset_instance_column": int(dataset_column),
                "rigid_model_index": model_index,
                "occupancy_instance_id": true_id + 1,
                "limited_label_id": true_id + 1,
                "class_name": raw["class_name"],
                "canonical_dimensions_lwh": dimensions,
                "first_frame": min(frames),
                "last_frame": max(frames),
                "active_frame_count": int(value["frame_info"].sum().item()),
                "lidar_point_count": int(value["num_pts"]),
                "occupancy_observed_frame_count": occupancy_counts.get(true_id + 1, 0),
                "checkpoint_size_max_abs_error": size_error,
            }
        )
    registry = build_actor_registry(
        scene_id=scene["scene_id"],
        checkpoint_sha256=checkpoint_sha256,
        actors=actors,
        checkpoint_num_instances=checkpoint_count,
    )
    registry.update(
        {
            "cohort_id": scene["cohort_id"],
            "b0_run": scene["b0_run"],
            "source_config_sha256": file_fingerprint(str(checkpoint.parent / "config.yaml")),
            "max_checkpoint_size_abs_error": max(size_errors, default=0.0),
        }
    )
    # 补充 provenance 后重新绑定最终 registry 内容。
    registry.pop("actor_registry_sha256")
    registry["actor_registry_sha256"] = canonical_sha256(registry)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/v71/world_state_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    output_root = args.output_root or Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"registry output 已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    registries = []
    for scene in config["scenes"]:
        registry = build_scene(scene, config)
        scene_dir = output_root / scene["scene_id"]
        scene_dir.mkdir()
        atomic_write_json(str(scene_dir / "actor_registry.json"), registry)
        registries.append(registry)
    summary = {
        "schema_version": 1,
        "task_id": "V7-H1-11A",
        "split": config["split"],
        "config_sha256": file_fingerprint(str(args.config)),
        "scene_count": len(registries),
        "actor_count_by_scene": {
            registry["scene_id"]: registry["actor_count"] for registry in registries
        },
        "registry_hash_by_scene": {
            registry["scene_id"]: registry["actor_registry_sha256"]
            for registry in registries
        },
        "minimum_two_actors_per_scene": all(
            registry["actor_count"] >= 2 for registry in registries
        ),
    }
    summary["combined_registry_sha256"] = canonical_sha256(
        summary["registry_hash_by_scene"]
    )
    atomic_write_json(str(output_root / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
