#!/usr/bin/env python
"""Build the M3 token-first actor registry from a trained DriveStudio checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.drivestudio_root))
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
    if rigid is None:
        raise RuntimeError("checkpoint has no RigidNodes state")
    point_ids = rigid["points_ids"].reshape(-1).tolist()
    instance_count = int(rigid["instances_trans"].shape[1])
    registry = build_drivestudio_registry(
        scene_id=str(int(config.data.scene_idx)),
        scene_name=args.scene_name,
        checkpoint_sha256=sha256_file(args.checkpoint),
        processed_instances=processed_instances,
        raw_instance_chains=chains,
        dataset_true_ids=dataset.pixel_source.instances_true_id.cpu().tolist(),
        ordered_init_columns=[int(value) for value in init.keys()],
        checkpoint_point_ids=point_ids,
        checkpoint_instance_count=instance_count,
    )
    selected = dict(require_token(registry, args.selected_token))
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
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
