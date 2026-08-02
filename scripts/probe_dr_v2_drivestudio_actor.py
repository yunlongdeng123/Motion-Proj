#!/usr/bin/env python
"""Audit whether the frozen M2 actor reaches DriveStudio RigidNodes init/checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instance-token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drivestudio-root", type=Path, default=Path("/root/autodl-tmp/third_party/drivestudio"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset

    config = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    dataset = DrivingDataset(data_cfg=config.data)
    scene_root = Path(config.data.data_root) / f"{int(config.data.scene_idx):03d}"
    processed = json.loads((scene_root / "instances" / "instances_info.json").read_text())
    token_matches = [int(true_id) for true_id, row in processed.items() if row.get("id") == args.instance_token]
    if len(token_matches) != 1:
        raise RuntimeError(f"selected token must resolve to one processed id, got {token_matches}")
    true_id = token_matches[0]
    true_ids = [int(value) for value in dataset.pixel_source.instances_true_id.cpu().tolist()]
    columns = [index for index, value in enumerate(true_ids) if value == true_id]
    if len(columns) != 1:
        raise RuntimeError(f"processed id must resolve to one visible dataset column, got {columns}")
    column = columns[0]
    frame_mask = dataset.pixel_source.per_frame_instance_mask[:, column].bool()
    translations = dataset.pixel_source.instances_pose[:, column, :3, 3][frame_mask]
    trajectory_length = float(
        torch.linalg.vector_norm(translations[1:] - translations[:-1], dim=-1).sum().item()
    ) if len(translations) > 1 else 0.0
    init_cfg = config.model.RigidNodes.init
    init = dataset.get_init_objects(
        cur_node_type="RigidNodes",
        instance_max_pts=int(init_cfg.instance_max_pts),
        only_moving=bool(init_cfg.only_moving),
        traj_length_thres=float(init_cfg.traj_length_thres),
    )
    ordered_columns = [int(value) for value in init.keys()]
    model_indices = [index for index, value in enumerate(ordered_columns) if value == column]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    rigid = checkpoint["models"].get("RigidNodes")
    checkpoint_instance_count = int(rigid["instances_trans"].shape[1]) if rigid else 0
    point_ids = rigid["points_ids"].reshape(-1) if rigid else torch.empty(0, dtype=torch.long)
    model_index = model_indices[0] if len(model_indices) == 1 else None
    gaussian_count = int((point_ids == model_index).sum().item()) if model_index is not None else 0
    available = len(model_indices) == 1 and model_index < checkpoint_instance_count and gaussian_count > 0
    reasons = []
    if bool(init_cfg.only_moving) and trajectory_length <= float(init_cfg.traj_length_thres):
        reasons.append("selected_actor_below_native_moving_threshold")
    if not model_indices:
        reasons.append("selected_actor_absent_from_rigid_init_order")
    if model_index is not None and model_index >= checkpoint_instance_count:
        reasons.append("selected_actor_model_index_absent_from_checkpoint")
    if model_index is not None and gaussian_count == 0:
        reasons.append("selected_actor_checkpoint_slice_empty")
    payload = {
        "schema_version": 1,
        "status": "available" if available else "missing",
        "instance_token": args.instance_token,
        "processed_true_instance_id": true_id,
        "dataset_instance_column": column,
        "rigid_model_index": model_index,
        "checkpoint_gaussian_count": gaussian_count,
        "checkpoint_instance_count": checkpoint_instance_count,
        "processed_active_frame_count": int(frame_mask.sum().item()),
        "processed_trajectory_length_m": trajectory_length,
        "native_only_moving": bool(init_cfg.only_moving),
        "native_trajectory_length_threshold_m": float(init_cfg.traj_length_thres),
        "reasons": reasons,
        "checkpoint": str(args.checkpoint),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if not available:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
