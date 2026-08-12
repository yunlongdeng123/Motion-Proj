#!/usr/bin/env python
"""Build the M3 token-first actor registry from a trained DriveStudio checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.dynamic_editing_v2.drivestudio_registry import (
    build_drivestudio_registry,
    canonical_sha256,
    require_token,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def raw_chains(metadata: Path, tokens: set[str]) -> dict[str, dict]:
    result = {}
    # The frozen nuScenes instance table is only about 16 MB.  Reading it with
    # the standard library keeps this audit helper inside the native
    # DriveStudio environment instead of adding a non-upstream dependency.
    rows = json.loads((metadata / "instance.json").read_text(encoding="utf-8"))
    for row in rows:
        if row["token"] in tokens:
            result[row["token"]] = {
                "first_annotation_token": row["first_annotation_token"],
                "last_annotation_token": row["last_annotation_token"],
                "nbr_annotations": int(row["nbr_annotations"]),
            }
    return result


def add_requested_unavailable_actors(
    registry: dict,
    *,
    requested_tokens: list[str],
    processed_instances: dict[str, dict],
    raw_instance_chains: dict[str, dict],
    dataset_true_ids: list[int],
    dataset_model_types: list[int],
    ordered_init_columns: list[int],
    rigid_model_type: int,
) -> dict:
    """把被 dataset/model 初始化过滤的冻结 actor 显式保留为不可用行。"""
    result = copy.deepcopy(registry)
    actors = list(result["actors"])
    existing_tokens = {str(actor["instance_token"]) for actor in actors}
    true_id_columns: dict[int, list[int]] = {}
    for column, raw_true_id in enumerate(dataset_true_ids):
        true_id_columns.setdefault(int(raw_true_id), []).append(column)
    init_columns = {int(value) for value in ordered_init_columns}
    requested_set = {str(value) for value in requested_tokens}

    for token in dict.fromkeys(str(value) for value in requested_tokens):
        if token in existing_tokens:
            continue
        matches = [
            (int(true_id), row)
            for true_id, row in processed_instances.items()
            if str(row.get("id")) == token
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"requested token 必须在 processed instances 精确命中一次：{token}"
            )
        true_id, processed = matches[0]
        columns = true_id_columns.get(true_id, [])
        if len(columns) > 1:
            raise RuntimeError(f"requested token 的 dataset column 不唯一：{token}")
        column = columns[0] if columns else None
        model_type = int(dataset_model_types[column]) if column is not None else None
        if column is not None and column in init_columns:
            raise RuntimeError(
                f"requested token 已进入初始化却未进入 registry：{token}"
            )
        if column is None:
            availability = "unavailable_dataset_filter"
        elif model_type != int(rigid_model_type):
            availability = "unavailable_model_type"
        else:
            availability = "unavailable_initialization_filter"
        chain = raw_instance_chains.get(token)
        if chain is None:
            raise RuntimeError(f"requested token 缺少 raw annotation chain：{token}")
        frames = [
            int(value) for value in processed["frame_annotations"]["frame_idx"]
        ]
        actors.append(
            {
                "instance_token": token,
                "raw_annotation_chain": {
                    "first_annotation_token": str(chain["first_annotation_token"]),
                    "last_annotation_token": str(chain["last_annotation_token"]),
                    "nbr_annotations": int(chain["nbr_annotations"]),
                },
                "processed_true_instance_id": true_id,
                "dataset_instance_column": column,
                "rigid_model_index": None,
                "availability": availability,
                "unavailable_reason": {
                    "dataset_model_type": model_type,
                    "rigid_model_type": int(rigid_model_type),
                    "entered_rigid_initialization": False,
                },
                "checkpoint_tensor_slice": {
                    "selector": None,
                    "gaussian_count": 0,
                    "flat_index_ranges_half_open": [],
                    "flat_indices_sha256": hashlib.sha256(b"[]").hexdigest(),
                },
                "class_name": str(processed["class_name"]),
                "first_processed_frame": min(frames),
                "last_processed_frame": max(frames),
                "processed_frame_count": len(frames),
            }
        )
        existing_tokens.add(token)

    result["actors"] = actors
    result["actor_count"] = len(actors)
    result["available_actor_count"] = sum(
        actor["availability"] == "available" for actor in actors
    )
    result["empty_checkpoint_actor_count"] = sum(
        actor["availability"] == "unavailable_empty_checkpoint_slice"
        for actor in actors
    )
    result["requested_unavailable_actor_count"] = sum(
        str(actor["instance_token"]) in requested_set
        and actor["availability"] != "available"
        for actor in actors
    )
    result.pop("actor_registry_sha256", None)
    result["actor_registry_sha256"] = canonical_sha256(result)
    return result


def rigid_checkpoint_contract(
    rigid: Optional[dict], *, ordered_init_columns: list[int]
) -> tuple[list[int], int]:
    """Return the exact RigidNodes identity tensors, allowing a truly empty model."""
    if rigid is None:
        if ordered_init_columns:
            raise RuntimeError(
                "checkpoint has no RigidNodes state but initialization is nonempty"
            )
        return [], 0
    return (
        [int(value) for value in rigid["points_ids"].reshape(-1).tolist()],
        int(rigid["instances_trans"].shape[1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--drivestudio-root",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio"),
    )
    parser.add_argument(
        "--raw-metadata",
        type=Path,
        default=Path("/root/autodl-tmp/data/nuscenes/v1.0-trainval"),
    )
    parser.add_argument("--scene-name", default="scene-0230")
    parser.add_argument("--selected-token", required=True)
    parser.add_argument("--requested-token", action="append", default=[])
    parser.add_argument("--allow-missing-selected", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.base.scene_dataset import ModelType
    from datasets.driving_dataset import DrivingDataset

    config_path = args.checkpoint.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = OmegaConf.load(config_path)
    dataset = DrivingDataset(data_cfg=config.data)
    init = dataset.get_init_objects(
        cur_node_type="RigidNodes",
        instance_max_pts=int(config.model.RigidNodes.init.instance_max_pts),
        only_moving=bool(config.model.RigidNodes.init.only_moving),
        traj_length_thres=float(config.model.RigidNodes.init.traj_length_thres),
    )
    scene_path = Path(config.data.data_root) / f"{int(config.data.scene_idx):03d}"
    processed_instances = json.loads(
        (scene_path / "instances" / "instances_info.json").read_text()
    )
    tokens = {str(row["id"]) for row in processed_instances.values()}
    chains = raw_chains(args.raw_metadata, tokens)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    rigid = checkpoint["models"].get("RigidNodes")
    ordered_init_columns = [int(value) for value in init.keys()]
    point_ids, instance_count = rigid_checkpoint_contract(
        rigid, ordered_init_columns=ordered_init_columns
    )
    registry = build_drivestudio_registry(
        scene_id=str(int(config.data.scene_idx)),
        scene_name=args.scene_name,
        checkpoint_sha256=sha256_file(args.checkpoint),
        processed_instances=processed_instances,
        raw_instance_chains=chains,
        dataset_true_ids=dataset.pixel_source.instances_true_id.cpu().tolist(),
        ordered_init_columns=ordered_init_columns,
        checkpoint_point_ids=point_ids,
        checkpoint_instance_count=instance_count,
    )
    registry = add_requested_unavailable_actors(
        registry,
        requested_tokens=[*args.requested_token, args.selected_token],
        processed_instances=processed_instances,
        raw_instance_chains=chains,
        dataset_true_ids=dataset.pixel_source.instances_true_id.cpu().tolist(),
        dataset_model_types=dataset.pixel_source.instances_model_types.cpu().tolist(),
        ordered_init_columns=ordered_init_columns,
        rigid_model_type=int(ModelType.RigidNodes),
    )
    selected = dict(
        require_token(
            registry,
            args.selected_token,
            require_nonempty=not args.allow_missing_selected,
        )
    )
    registry["selected_smoke_actor"] = selected
    registry["source"] = {
        "checkpoint": str(args.checkpoint),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "processed_scene": str(scene_path),
        "raw_metadata": str(args.raw_metadata),
    }
    registry.pop("actor_registry_sha256")
    registry["actor_registry_sha256"] = canonical_sha256(registry)
    atomic_json(args.output, registry)
    print(
        json.dumps(
            {
                "status": "done",
                "actor_count": registry["actor_count"],
                "selected_token": args.selected_token,
                "selected_model_index": selected["rigid_model_index"],
                "selected_availability": selected["availability"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
