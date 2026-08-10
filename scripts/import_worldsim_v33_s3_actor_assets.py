#!/usr/bin/env python3
"""把自动 1/2/4-view 官方 PLY 导入确定性的 StreetGS actor sidecars。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.actor_asset_schema import validate_actor_asset
from motion_proj.worldsim_v32.asset_harvester_adapter import (
    canonicalize_asset_harvester_ply,
)
from motion_proj.worldsim_v32.semantic_schema import array_sha256
from motion_proj.worldsim_v33.view_selection import atomic_save_deterministic_npz


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, Any]:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    sources = {
        "config": config_path,
        "importer": Path(__file__).resolve(),
        "view_selection": PROJECT / "motion_proj" / "worldsim_v33" / "view_selection.py",
        "adapter": PROJECT / "motion_proj" / "worldsim_v32" / "asset_harvester_adapter.py",
        "schema": PROJECT / "motion_proj" / "worldsim_v32" / "actor_asset_schema.py",
    }
    report = {}
    for role, source in sources.items():
        target = snapshot / source.name
        shutil.copy2(source, target)
        report[role] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--inference-manifest-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"S3 import run-dir 非空: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_dir / "artifacts"
    assets_dir = artifacts / "actor_assets"
    assets_dir.mkdir(parents=True)
    started = time.time()
    atomic_json(args.run_dir / "status.json", {"state": "running", "started_unix": started})

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_sha = sha256_file(args.input_manifest)
    inference_sha = sha256_file(args.inference_manifest)
    if input_sha != args.input_manifest_sha256:
        raise RuntimeError("S3 import input manifest SHA 漂移")
    if inference_sha != args.inference_manifest_sha256:
        raise RuntimeError("S3 import inference manifest SHA 漂移")
    inputs = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
    if inputs["role"] != inference["role"]:
        raise RuntimeError("S3 import role 错配")
    selection_path = args.input_manifest.parent / inputs["selection_manifest"]
    if sha256_file(selection_path) != inputs["selection_manifest_sha256"]:
        raise RuntimeError("S3 import selection manifest SHA 漂移")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    actor = inputs["actor"]
    ah = config["asset_harvester"]
    imported = {}
    for sample in sorted(inputs["samples"], key=lambda row: int(row["view_count"])):
        name = sample["sample"]
        view_count = int(sample["view_count"])
        ply_spec = inference["plys"][name]
        ply_path = Path(ply_spec["path"])
        if sha256_file(ply_path) != ply_spec["sha256"]:
            raise RuntimeError(f"S3 {name} PLY SHA 漂移")
        sizes = np.asarray(
            [row["box_size_source"] for row in sample["views"]], dtype=np.float64
        )
        if not np.allclose(sizes, sizes[0], rtol=0, atol=1e-6):
            raise RuntimeError(f"S3 {name} source actor LWH 不一致")
        fit = canonicalize_asset_harvester_ply(
            path=ply_path,
            target_lwh=sizes[0],
            orientation_y_degrees=float(ah["adapter"]["orientation_y_degrees"]),
            support_sigma=float(ah["adapter"]["bounds_support_sigma"]),
        )
        arrays = {
            key: fit[key]
            for key in (
                "means",
                "scales",
                "quats",
                "rgb",
                "opacity",
                "T_actor_asset",
                "bounds_lower",
                "bounds_upper",
                "target_lwh",
                "scale_xyz",
            )
        }
        validate_actor_asset(arrays)
        output_dir = assets_dir / name
        output_dir.mkdir()
        asset_path = output_dir / "actor_asset.npz"
        repeat_path = output_dir / "actor_asset.repeat.npz"
        atomic_save_deterministic_npz(asset_path, arrays)
        atomic_save_deterministic_npz(repeat_path, dict(reversed(list(arrays.items()))))
        if asset_path.read_bytes() != repeat_path.read_bytes():
            raise RuntimeError(f"S3 {name} actor NPZ 非 byte-exact")
        repeat_path.unlink()
        with np.load(asset_path, allow_pickle=False) as payload:
            reloaded = {key: payload[key].copy() for key in payload.files}
        validate_actor_asset(reloaded)
        expected_array_hashes = {key: array_sha256(value) for key, value in arrays.items()}
        if {key: array_sha256(value) for key, value in reloaded.items()} != expected_array_hashes:
            raise RuntimeError(f"S3 {name} actor NPZ reload 非 exact")
        set_evidence = selection["selected_sets"][str(view_count)]
        selected_views = [
            {
                key: row[key]
                for key in (
                    "frame",
                    "camera_id",
                    "camera_name",
                    "view_score",
                    "view_score_components",
                    "yaw_degrees",
                    "mask_confidence",
                    "occlusion_score",
                    "sharpness_laplacian_variance",
                    "visible_fraction",
                    "truncation_score",
                )
            }
            for row in set_evidence["selected_views"]
        ]
        manifest = {
            "schema_version": "worldsim_v33_actor_asset_v1",
            "task_id": config["task_id"],
            "state": "completed",
            "sample": name,
            "role": inputs["role"],
            "instance_token": actor["instance_token"],
            "dataset_instance_id": int(actor["dataset_instance_id"]),
            "rigid_model_index": int(actor["rigid_model_index"]),
            "generation_provenance": "GENERATED_ACTOR",
            "camera_source": inputs["camera_source"],
            "view_selection": {
                "selection_policy": "automatic_train_only_quality_plus_yaw_time_camera_diversity_v1",
                "candidate_views": int(selection["candidate_count_before_d2"]),
                "eligible_views": int(selection["eligible_count"]),
                "selected_views": selected_views,
                "view_score": [float(row["view_score"]) for row in selected_views],
                "set_score": float(set_evidence["set_score"]),
                "set_score_components": {
                    key: float(set_evidence[key])
                    for key in (
                        "quality_sum",
                        "yaw_diversity",
                        "temporal_diversity",
                        "camera_diversity",
                    )
                },
                "yaw_distribution_degrees": [
                    float(row["yaw_degrees"]) for row in selected_views
                ],
                "mask_quality": [float(row["mask_confidence"]) for row in selected_views],
                "occlusion": [float(row["occlusion_score"]) for row in selected_views],
                "sharpness": [
                    float(row["sharpness_laplacian_variance"])
                    for row in selected_views
                ],
                "heldout_read": False,
                "reserved_development_read": False,
            },
            "source_views": sample["views"],
            "source_ply": ply_spec,
            "asset": {
                "path": str(asset_path),
                "bytes": asset_path.stat().st_size,
                "sha256": sha256_file(asset_path),
                "gaussian_count": int(len(arrays["means"])),
                "array_sha256": expected_array_hashes,
                "deterministic_reserialization": True,
                "reload_exact": True,
            },
            "coordinate_contract": {
                "native": "asset_harvester_compatible_ply_after_internal_x180",
                "orientation": "official_nurec_y_rotation",
                "orientation_y_degrees": float(ah["adapter"]["orientation_y_degrees"]),
                "T_actor_asset": arrays["T_actor_asset"].tolist(),
                "actor_axes": "x=length,y=width,z=height",
                "target_lwh_m": arrays["target_lwh"].tolist(),
                "bounds_support_sigma": float(ah["adapter"]["bounds_support_sigma"]),
                "bounds_lower_m": arrays["bounds_lower"].tolist(),
                "bounds_upper_m": arrays["bounds_upper"].tolist(),
            },
            "lineage": {
                "input_manifest_sha256": input_sha,
                "selection_manifest_sha256": inputs["selection_manifest_sha256"],
                "inference_manifest_sha256": inference_sha,
            },
        }
        manifest_path = output_dir / "actor_asset_manifest.json"
        atomic_json(manifest_path, manifest)
        imported[name] = {
            "view_count": view_count,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "asset": manifest["asset"],
        }

    snapshot = snapshot_sources(args.run_dir, args.config.resolve())
    summary = {
        "task_id": config["task_id"],
        "state": "completed",
        "role": inputs["role"],
        "input_manifest_sha256": input_sha,
        "inference_manifest_sha256": inference_sha,
        "assets": imported,
        "source_snapshot": snapshot,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "status.json",
        {
            "state": "completed",
            "started_unix": started,
            "completed_unix": time.time(),
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "asset_manifest_sha256": {
                name: row["manifest_sha256"] for name, row in imported.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
