#!/usr/bin/env python
"""导入 Asset Harvester PLY，冻结为 StreetGS actor-local sidecar。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.asset_harvester_adapter import (
    canonicalize_asset_harvester_ply,
)
from motion_proj.worldsim_v32.actor_asset_schema import validate_actor_asset
from motion_proj.worldsim_v32.semantic_schema import array_sha256, sha256_file


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--sample", choices=["high_support_1view", "high_support_2view"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
    if sha256_file(args.input_manifest) != config["inputs"]["input_manifest_sha256"]:
        raise RuntimeError("S3 import input manifest SHA 漂移")
    ply_spec = inference["plys"][args.sample]
    ply_path = Path(ply_spec["path"])
    if sha256_file(ply_path) != ply_spec["sha256"]:
        raise RuntimeError("S3 import PLY SHA 漂移")
    sample = next(row for row in inputs["samples"] if row["sample"] == args.sample)
    sizes = np.asarray([row["box_size_source"] for row in sample["views"]], dtype=np.float64)
    if not np.allclose(sizes, sizes[0], rtol=0, atol=1e-6):
        raise RuntimeError("S3 source views actor box_size 不一致")
    target_lwh = sizes[0]
    fit = canonicalize_asset_harvester_ply(
        path=ply_path,
        target_lwh=target_lwh,
        orientation_y_degrees=float(config["adapter"]["orientation_y_degrees"]),
        support_sigma=float(config["adapter"]["bounds_support_sigma"]),
    )
    arrays = {
        "means": fit["means"],
        "scales": fit["scales"],
        "quats": fit["quats"],
        "rgb": fit["rgb"],
        "opacity": fit["opacity"],
        "T_actor_asset": fit["T_actor_asset"],
        "bounds_lower": fit["bounds_lower"],
        "bounds_upper": fit["bounds_upper"],
        "target_lwh": fit["target_lwh"],
        "scale_xyz": fit["scale_xyz"],
    }
    validate_actor_asset(arrays)
    asset_path = args.output_dir / "actor_asset.npz"
    atomic_npz(asset_path, **arrays)
    with np.load(asset_path, allow_pickle=False) as reloaded:
        reload_hashes = {name: array_sha256(reloaded[name]) for name in reloaded.files}
    expected_hashes = {name: array_sha256(value) for name, value in arrays.items()}
    if reload_hashes != expected_hashes:
        raise RuntimeError("S3 actor asset reload 不精确")
    manifest = {
        "schema_version": "worldsim_v32_actor_asset_v1",
        "task_id": config["task_id"],
        "status": "done",
        "sample": args.sample,
        "instance_token": config["actor"]["instance_token"],
        "dataset_instance_id": int(config["actor"]["dataset_instance_id"]),
        "rigid_model_index": int(config["actor"]["rigid_model_index"]),
        "generation_provenance": "GENERATED_ACTOR",
        "implementation": {
            "importer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "adapter": {
                "path": str(
                    PROJECT / "motion_proj/worldsim_v32/asset_harvester_adapter.py"
                ),
                "sha256": sha256_file(
                    PROJECT / "motion_proj/worldsim_v32/asset_harvester_adapter.py"
                ),
            },
            "actor_asset_schema": {
                "path": str(
                    PROJECT / "motion_proj/worldsim_v32/actor_asset_schema.py"
                ),
                "sha256": sha256_file(
                    PROJECT / "motion_proj/worldsim_v32/actor_asset_schema.py"
                ),
            },
        },
        "camera_source": inputs["camera_source"],
        "source_views": sample["views"],
        "source_ply": ply_spec,
        "asset": {
            "path": str(asset_path),
            "bytes": asset_path.stat().st_size,
            "sha256": sha256_file(asset_path),
            "gaussian_count": int(len(arrays["means"])),
            "array_sha256": expected_hashes,
        },
        "coordinate_contract": {
            "native": "asset_harvester_compatible_ply_after_internal_x180",
            "orientation": "official_nurec_y_rotation",
            "orientation_y_degrees": float(config["adapter"]["orientation_y_degrees"]),
            "T_actor_asset": arrays["T_actor_asset"].tolist(),
            "actor_axes": "x=length,y=width,z=height",
            "target_lwh_m": target_lwh.tolist(),
            "bounds_support_sigma": float(config["adapter"]["bounds_support_sigma"]),
            "bounds_lower_m": arrays["bounds_lower"].tolist(),
            "bounds_upper_m": arrays["bounds_upper"].tolist(),
        },
        "reload_exact": True,
    }
    atomic_json(args.output_dir / "actor_asset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
