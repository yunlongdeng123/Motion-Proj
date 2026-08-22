#!/usr/bin/env python3
"""WorldSim V6.1 P4：Hunyuan3D-Omni 单卡 voxel-control capability smoke。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import trimesh
import yaml


TASK_ID = "WS-V61-P4-HY3D-OMNI-3090-SMOKE-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class P4SmokeError(RuntimeError):
    """P4 source、environment 或 capability smoke 合同失败。"""


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


def verify_files(root: Path, files: Mapping[str, str]) -> None:
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise P4SmokeError(f"冻结文件漂移: {path}")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def resolve_runs_uri(uri: str) -> Path:
    if not uri.startswith("runs://"):
        raise P4SmokeError("只接受 runs URI")
    relative = Path(uri.removeprefix("runs://"))
    if ".." in relative.parts:
        raise P4SmokeError("runs URI 不得包含上级路径")
    return (RUNS_ROOT / relative).resolve()


def normalized_voxel_surface(path: Path, count: int) -> torch.Tensor:
    """精确复用官方 voxel demo 的旋转、归一化和表面采样。"""
    mesh = trimesh.load(path, force="mesh")
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(angle=np.radians(-90), direction=[1, 0, 0])
    )
    bounds = mesh.bounds
    center = (bounds[1] + bounds[0]) / 2
    scale = (bounds[1] - bounds[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale * 2.0 * 0.9999)
    surface = mesh.sample(int(count))
    return torch.as_tensor(surface, dtype=torch.float32).unsqueeze(0)


def run(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if git(repo_root, "status", "--porcelain"):
        raise P4SmokeError("正式 P4 要求 motion_proj 工作树干净")
    source_commit = git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise P4SmokeError("P4 task_id 漂移")

    me1_root = resolve_runs_uri(config["sources"]["me1_run"])
    official_root = Path(config["sources"]["official_repo"]).resolve()
    model_root = Path(config["sources"]["model_root"]).resolve()
    verify_files(me1_root, config["sources"]["me1_files"])
    verify_files(official_root, config["sources"]["official_files"])
    verify_files(model_root, config["sources"]["model_files"])
    if git(official_root, "rev-parse", "HEAD") != config["sources"]["official_git_commit"]:
        raise P4SmokeError("official repo commit 漂移")
    me1_gate = json.loads((me1_root / "ME1_GATE.json").read_text(encoding="utf-8"))
    if not me1_gate.get("passed"):
        raise P4SmokeError("ME-1 未解锁 P4")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != config["environment"]["python_major_minor"]:
        raise P4SmokeError("Python major/minor 漂移")
    if torch.__version__ != config["environment"]["torch_version"]:
        raise P4SmokeError(f"torch version 漂移: {torch.__version__}")
    if not torch.cuda.is_available():
        raise P4SmokeError("CUDA 不可用")
    hf_home = Path(config["environment"]["hf_home"])
    dino_blobs = list(hf_home.glob("hub/models--facebook--dinov2-large/blobs/*"))
    if not any(path.is_file() and sha256_file(path) == config["sources"]["dino_model_sha256"] for path in dino_blobs):
        raise P4SmokeError("冻结 DINOv2-large blob 缺失或漂移")

    run_root.mkdir(parents=True, exist_ok=True)
    disk_free_gib = shutil.disk_usage(run_root).free / 1024**3
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__voxel-smoke-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        sys.path.insert(0, str(official_root))
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline

        random.seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.set_device(int(config["resources"]["gpu"]))
        torch.cuda.reset_peak_memory_stats()
        pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
            str(model_root), fast_decode=bool(config["inference"]["fast_decode"])
        )
        image_path = official_root / config["inference"]["input_image"]
        voxel_path = official_root / config["inference"]["input_voxel"]
        surface = normalized_voxel_surface(voxel_path, int(config["inference"]["surface_points"]))
        surface = surface.to(pipeline.device, dtype=pipeline.dtype)
        result = pipeline(
            image=str(image_path),
            voxel=surface,
            num_inference_steps=int(config["inference"]["num_inference_steps"]),
            octree_resolution=int(config["inference"]["octree_resolution"]),
            mc_level=float(config["inference"]["mc_level"]),
            guidance_scale=float(config["inference"]["guidance_scale"]),
            generator=torch.Generator("cuda").manual_seed(int(config["seed"])),
        )
        mesh = result["shapes"][0][0]
        sampled = result["sampled_point"][0].detach().float().cpu().numpy()
        mesh_path = run_dir / "VOXEL_SMOKE_MESH.glb"
        point_path = run_dir / "VOXEL_SMOKE_POINTS.ply"
        mesh.export(mesh_path)
        trimesh.points.PointCloud(sampled).export(point_path)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak_gpu_gib = torch.cuda.max_memory_allocated() / 1024**3
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        checks = {
            "me1_authority_passed": True,
            "official_source_exact": True,
            "model_and_dino_weights_exact": True,
            "python_and_torch_exact": True,
            "mesh_vertices_and_faces_nonzero": bool(vertices.shape[0] and faces.shape[0]),
            "mesh_finite": bool(np.all(np.isfinite(vertices))),
            "sampled_points_nonzero_and_finite": bool(sampled.size and np.all(np.isfinite(sampled))),
            "gpu_memory_within_budget": peak_gpu_gib <= float(config["gate"]["maximum_gpu_memory_gib"]),
            "wall_within_budget": elapsed <= float(config["gate"]["maximum_wall_seconds"]),
            "disk_free_within_budget": disk_free_gib >= float(config["gate"]["minimum_disk_free_gib"]),
            "offline_formal_run": os.environ.get("HF_HUB_OFFLINE") == "1",
            "no_ema_fast_decode_or_sweep": (
                not bool(config["inference"]["use_ema"])
                and not bool(config["inference"]["fast_decode"])
            ),
            "no_training_or_confirmation": True,
        }
        gate = {
            "schema_version": "worldsim_v61.p4_hy3d_smoke_gate.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "checks": checks,
            "passed": all(checks.values()),
        }
        write_json(run_dir / "P4_GATE.json", gate)
        capability = {
            "schema_version": "worldsim_v61.p4_hy3d_capability.v1",
            "control_type": "voxel",
            "mesh_vertices": int(vertices.shape[0]),
            "mesh_faces": int(faces.shape[0]),
            "mesh_watertight": bool(mesh.is_watertight),
            "mesh_connected_components": len(mesh.split(only_watertight=False)),
            "sampled_point_count": int(sampled.shape[0]),
            "mesh_bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
            "seed": int(config["seed"]),
            "num_inference_steps": int(config["inference"]["num_inference_steps"]),
            "octree_resolution": int(config["inference"]["octree_resolution"]),
            "guidance_scale": float(config["inference"]["guidance_scale"]),
            "mesh_sha256": sha256_file(mesh_path),
            "points_sha256": sha256_file(point_path),
        }
        write_json(run_dir / "CAPABILITY.json", capability)
        resource = {
            "schema_version": "worldsim_v61.p4_resource_audit.v1",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "peak_gpu_memory_gib": peak_gpu_gib,
            "wall_seconds": elapsed,
            "disk_free_gib_at_start": disk_free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        }
        write_json(run_dir / "RESOURCE_AUDIT.json", resource)
        summary = {
            "schema_version": "worldsim_v61.p4_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "source_commit": source_commit,
            "status": "done" if gate["passed"] else "rejected",
            "hypothesis_outcome": "accepted_3090_voxel_capability" if gate["passed"] else "rejected_3090_voxel_capability",
            "next": "WS-V61-ME2-HY3D-OCC-ACTOR-01" if gate["passed"] else "stop_hy3d_before_actor_experiment",
            "failure_ledger_delta": "none" if gate["passed"] else "required",
            "claim_boundary": config["claim_boundary"],
        }
        write_json(run_dir / "SUMMARY.json", summary)
        manifest = {
            "schema_version": "worldsim_v61.p4_manifest.v1",
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "official_git_commit": config["sources"]["official_git_commit"],
            "model_revision": config["sources"]["model_revision"],
            "dino_revision": config["sources"]["dino_revision"],
            "artifacts": {
                name: sha256_file(run_dir / name)
                for name in ("P4_GATE.json", "CAPABILITY.json", "RESOURCE_AUDIT.json", "SUMMARY.json")
            },
        }
        write_json(run_dir / "MANIFEST.json", manifest)
        write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.p4_terminal.v1",
                "task_id": TASK_ID,
                "status": summary["status"],
                "canonical": bool(gate["passed"]),
                "run_uri": f"run://worldsim_v61/{TASK_ID}/{run_dir.name}",
            },
        )
        return run_dir
    except Exception as error:
        write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.p4_terminal.v1",
                "task_id": TASK_ID,
                "status": "failed",
                "canonical": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", default=Path("/root/autodl-tmp/runs/worldsim_v61"), type=Path)
    args = parser.parse_args()
    print(run(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
