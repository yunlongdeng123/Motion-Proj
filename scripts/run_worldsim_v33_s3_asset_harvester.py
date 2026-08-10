#!/usr/bin/env python3
"""对冻结的自动 1/2/4-view 输入运行官方 NVIDIA Asset Harvester。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.run_worldsim_v32_s3_asset_harvester import (
    ResourceMonitor,
    validate_hf_snapshot,
)


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


def verify_file(spec: dict[str, str], role: str) -> Path:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{role} 不存在: {path}")
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"{role} SHA 漂移: {actual} != {spec['sha256']}")
    return path


def snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, Any]:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    sources = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
        "resource_monitor": PROJECT / "scripts" / "run_worldsim_v32_s3_asset_harvester.py",
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
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"S3 inference run-dir 非空: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_dir / "artifacts"
    artifacts.mkdir()
    started = time.time()
    atomic_json(args.run_dir / "status.json", {"state": "running", "started_unix": started})

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_s3_viewselect_v1":
        raise ValueError("S3 inference config schema 漂移")
    actual_input_sha = sha256_file(args.input_manifest)
    if actual_input_sha != args.input_manifest_sha256:
        raise RuntimeError("S3 input manifest SHA 与 CLI 冻结值不一致")
    inputs = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if not inputs.get("heldout_excluded") or inputs.get("heldout_read"):
        raise RuntimeError("S3 inference 输入未证明 heldout 隔离")
    sample_rows = list(inputs["samples"])
    counts = sorted(int(row["view_count"]) for row in sample_rows)
    if counts != [1, 2, 4]:
        raise RuntimeError(f"S3 auto sample count 漂移: {counts}")
    sample_names = {row["sample"] for row in sample_rows}
    samples_dir = args.input_manifest.parent / inputs["samples_dir"]
    if not samples_dir.is_dir():
        raise FileNotFoundError(samples_dir)
    for row in sample_rows:
        sample_dir = samples_dir / row["sample"]
        if not sample_dir.is_dir() or len(list(sample_dir.glob("*.jpeg"))) != int(
            row["view_count"]
        ):
            raise RuntimeError(f"S3 sample 图像数量漂移: {row['sample']}")
        for view in row["views"]:
            image = args.input_manifest.parent / view["image"]
            mask = args.input_manifest.parent / view["mask"]
            if sha256_file(image) != view["image_sha256"]:
                raise RuntimeError(f"S3 crop image SHA 漂移: {image}")
            if sha256_file(mask) != view["mask_sha256"]:
                raise RuntimeError(f"S3 crop mask SHA 漂移: {mask}")

    ah = config["asset_harvester"]
    source = Path(ah["checkout"])
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != ah["commit"]:
        raise RuntimeError(f"Asset Harvester source commit 漂移: {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("Asset Harvester source tree 非 clean")
    weights = ah["weights"]
    diffusion = verify_file(weights["diffusion"], "diffusion")
    lifting = verify_file(weights["lifting"], "lifting")
    camera = verify_file(weights["camera_estimator"], "camera estimator")
    hf_home = Path(ah["hf_home"])
    transitive = {
        name: validate_hf_snapshot(hf_home, spec)
        for name, spec in ah["transitive_models"].items()
    }
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_ENDPOINT"] = str(ah["hf_endpoint"])
    if bool(ah["offline"]):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"

    if not torch.cuda.is_available():
        raise RuntimeError("S3 Asset Harvester 正式推理需要 CUDA")
    torch.cuda.set_device(0)
    torch.cuda.init()
    inference = ah["inference"]
    seed = int(inference["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(0)
    resources = ResourceMonitor()
    resources.start()
    official_output = artifacts / "asset_harvester"
    official_argv = [
        str(source / "run_inference.py"),
        "--diffusion_checkpoint",
        str(diffusion),
        "--ahc_checkpoint",
        str(camera),
        "--image_dir",
        str(samples_dir),
        "--lifting_checkpoint",
        str(lifting),
        "--output_dir",
        str(official_output),
        "--num_steps",
        str(int(inference["num_steps"])),
        "--cfg_scale",
        str(float(inference["cfg_scale"])),
        "--precision",
        str(inference["precision"]),
    ]
    if bool(inference["offload_model_to_cpu"]):
        official_argv.append("--offload_model_to_cpu")
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(source))
    inference_started = time.monotonic()
    try:
        os.chdir(source)
        sys.argv = official_argv
        runpy.run_path(str(source / "run_inference.py"), run_name="__main__")
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        resource_summary = resources.finish()
    if resource_summary["memory_pressure_observed"]:
        raise RuntimeError("S3 Asset Harvester cgroup 连续达到 90% 停止门")
    if resource_summary["memory_events_delta"].get("oom", 0) or resource_summary[
        "memory_events_delta"
    ].get("oom_kill", 0):
        raise RuntimeError("S3 Asset Harvester 发生 cgroup OOM")

    plys = {}
    for sample in sorted(sample_names):
        path = official_output / sample / "gaussians.ply"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"S3 {sample} 未生成非空 PLY")
        plys[sample] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(official_output.rglob("*"))
        if path.is_file()
    ]
    snapshot = snapshot_sources(args.run_dir, args.config.resolve())
    summary = {
        "schema_version": "worldsim_v33_s3_asset_harvester_inference_v1",
        "task_id": config["task_id"],
        "state": "completed",
        "source_commit": head,
        "source_tree_clean": True,
        "config_sha256": sha256_file(args.config),
        "input_manifest": str(args.input_manifest),
        "input_manifest_sha256": actual_input_sha,
        "selection_manifest_sha256": inputs["selection_manifest_sha256"],
        "role": inputs["role"],
        "seed": seed,
        "generation_provenance": "GENERATED_ACTOR",
        "heldout_read": False,
        "official_argv": official_argv,
        "transitive_models": transitive,
        "hf_offline": bool(ah["offline"]),
        "plys": plys,
        "files": files,
        "runtime": {
            "wall_seconds": time.monotonic() - inference_started,
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
            **resource_summary,
        },
        "source_snapshot": snapshot,
    }
    manifest_path = artifacts / "inference_manifest.json"
    atomic_json(manifest_path, summary)
    atomic_json(
        args.run_dir / "summary.json",
        {
            "task_id": config["task_id"],
            "state": "completed",
            "role": inputs["role"],
            "sample_names": sorted(sample_names),
            "ply_sha256": {name: spec["sha256"] for name, spec in plys.items()},
            "inference_manifest_sha256": sha256_file(manifest_path),
            "wall_seconds": summary["runtime"]["wall_seconds"],
            "peak_nvidia_memory_mib": summary["runtime"]["peak_nvidia_memory_mib_sampled"],
        },
    )
    atomic_json(
        args.run_dir / "status.json",
        {
            "state": "completed",
            "started_unix": started,
            "completed_unix": time.time(),
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "inference_manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
