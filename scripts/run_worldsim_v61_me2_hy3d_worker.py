#!/usr/bin/env python3
"""WorldSim V6.1 ME-2：隔离 Hunyuan3D backend namespace 的 GPU worker。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def batches(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[index : index + int(size)] for index in range(0, len(values), int(size))]


def export_rows(
    meshes: list[trimesh.Trimesh],
    units: list[dict[str, Any]],
    arm: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    if len(meshes) != len(units):
        raise RuntimeError(f"mesh batch 数漂移: {len(meshes)} != {len(units)}")
    rows = []
    for mesh, unit in zip(meshes, units, strict=True):
        arm_root = output_root / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        path = arm_root / f"{unit['unit_id']}.glb"
        mesh.export(path)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if not vertices.size or not faces.size or not np.isfinite(vertices).all():
            raise RuntimeError(f"生成 mesh 无效: {arm}/{unit['unit_id']}")
        rows.append(
            {
                "schema_version": "worldsim_v61.me2_worker_asset.v1",
                "arm": arm,
                "unit_id": unit["unit_id"],
                "seed": int(unit["seed"]),
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "vertices": int(vertices.shape[0]),
                "faces": int(faces.shape[0]),
                "watertight": bool(mesh.is_watertight),
                "connected_components": len(mesh.split(only_watertight=False)),
            }
        )
    return rows


def run_image_base(plan: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    repo = Path(plan["repo"])
    sys.path.insert(0, str(repo / "hy3dshape"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        plan["model_root"],
        device="cuda",
        dtype=torch.float16,
        subfolder="hunyuan3d-dit-v2-1",
    )
    rows: list[dict[str, Any]] = []
    for group in batches(plan["units"], int(plan["batch_size"])):
        generators = [
            torch.Generator(device="cuda").manual_seed(int(unit["seed"])) for unit in group
        ]
        meshes = pipeline(
            image=[unit["image_path"] for unit in group],
            num_inference_steps=int(plan["num_inference_steps"]),
            guidance_scale=float(plan["guidance_scale"]),
            octree_resolution=int(plan["octree_resolution"]),
            mc_level=float(plan["mc_level"]),
            generator=generators,
            enable_pbar=False,
        )
        rows.extend(export_rows(list(meshes), group, "A0-image", output_root))
    return rows


def run_omni(plan: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    repo = Path(plan["repo"])
    sys.path.insert(0, str(repo))
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline

    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        plan["model_root"], fast_decode=False
    )
    rows: list[dict[str, Any]] = []
    for arm, control_name in (
        ("A1-bbox", "bbox"),
        ("A2-point", "point"),
        ("A3-voxel", "voxel"),
    ):
        for group in batches(plan["units"], int(plan["batch_size"])):
            controls = [
                np.asarray(np.load(unit["controls_path"], allow_pickle=False)[control_name])
                for unit in group
            ]
            control = torch.as_tensor(np.stack(controls), dtype=pipeline.dtype, device=pipeline.device)
            generators = [
                torch.Generator(device="cuda").manual_seed(int(unit["seed"])) for unit in group
            ]
            kwargs = {control_name: control}
            result = pipeline(
                image=[unit["image_path"] for unit in group],
                num_inference_steps=int(plan["num_inference_steps"]),
                octree_resolution=int(plan["octree_resolution"]),
                mc_level=float(plan["mc_level"]),
                guidance_scale=float(plan["guidance_scale"]),
                generator=generators,
                fast_decode=False,
                **kwargs,
            )
            rows.extend(export_rows(list(result["shapes"][0]), group, arm, output_root))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    started = time.monotonic()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    random.seed(int(plan["global_seed"]))
    np.random.seed(int(plan["global_seed"]))
    torch.manual_seed(int(plan["global_seed"]))
    torch.cuda.manual_seed_all(int(plan["global_seed"]))
    if plan["backend"] == "image_base":
        rows = run_image_base(plan, args.output_root)
    elif plan["backend"] == "omni":
        rows = run_omni(plan, args.output_root)
    else:
        raise RuntimeError(f"未知 backend: {plan['backend']}")
    torch.cuda.synchronize()
    report = {
        "schema_version": "worldsim_v61.me2_worker_report.v1",
        "backend": plan["backend"],
        "assets": rows,
        "asset_count": len(rows),
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
