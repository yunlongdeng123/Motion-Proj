#!/usr/bin/env python
"""验收 Asset Harvester 独立环境并写入 formal run 环境证据。"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file
from scripts.run_worldsim_v32_s3_asset_harvester import validate_hf_snapshot


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-transitive-models", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = Path(config["source"]["checkout"])
    if git_head(source) != config["source"]["commit"]:
        raise RuntimeError("S3 environment Asset Harvester source commit 漂移")
    gsplat = Path("/root/autodl-tmp/third_party/worldsim_v32/gsplat-b60e917")
    expected_gsplat = "b60e917c95afc449c5be33a634f1f457e116ff5e"
    if git_head(gsplat) != expected_gsplat:
        raise RuntimeError("S3 environment gsplat source commit 漂移")
    glm_status = subprocess.check_output(
        ["git", "-C", str(gsplat), "submodule", "status", "--recursive"],
        text=True,
    ).strip()
    if not glm_status.lstrip("-+ ").startswith(
        "33b4a621a697a305bc3a7610d290677b96beb181"
    ):
        raise RuntimeError(f"S3 environment glm submodule 漂移: {glm_status}")
    for name, spec in config["weights"].items():
        path = Path(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"S3 environment weight SHA 漂移: {name}")
    input_manifest = Path(config["inputs"]["input_manifest"])
    if sha256_file(input_manifest) != config["inputs"]["input_manifest_sha256"]:
        raise RuntimeError("S3 environment input manifest SHA 漂移")

    if not torch.cuda.is_available():
        raise RuntimeError("S3 environment CUDA 不可见")
    tensor = torch.arange(16, device="cuda", dtype=torch.float32)
    cuda_smoke_sum = float((tensor * tensor).sum().item())
    from gsplat.cuda._backend import _C  # noqa: F401

    for module in (
        "asset_harvester",
        "diffusers",
        "transformers",
        "xformers",
        "lpips",
        "decord",
    ):
        importlib.import_module(module)
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    pip_check_text = (pip_check.stdout + pip_check.stderr).strip()
    known_decord_warning = "decord 0.6.0 is not supported on this platform"
    if pip_check.returncode != 0 and pip_check_text != known_decord_warning:
        raise RuntimeError(f"S3 environment pip check 失败:\n{pip_check_text}")
    pip_check_status = (
        "passed"
        if pip_check.returncode == 0
        else "passed_with_imported_decord_platform_metadata_warning"
    )
    cli = subprocess.run(
        [sys.executable, str(source / "run_inference.py"), "--help"],
        cwd=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if cli.returncode != 0 or "--diffusion_checkpoint" not in cli.stdout:
        raise RuntimeError(f"S3 environment official CLI import 失败:\n{cli.stderr}")
    package_names = [
        "asset-harvester",
        "torch",
        "torchvision",
        "gsplat",
        "diffusers",
        "transformers",
        "xformers",
        "huggingface-hub",
    ]
    setup_logs = []
    for path in sorted((args.run_dir / "logs").glob("environment_setup*.log")):
        setup_logs.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    transitive_models = {}
    if args.require_transitive_models:
        hf_home = Path(config["runtime"]["hf_home"])
        transitive_models = {
            name: validate_hf_snapshot(hf_home, spec)
            for name, spec in config["transitive_models"].items()
        }
    result = {
        "schema_version": "worldsim_v32_s3_environment_v1",
        "task_id": config["task_id"],
        "status": "done",
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_smoke_sum": cuda_smoke_sum,
        "gsplat_cuda_extension": "imported",
        "asset_harvester_commit": git_head(source),
        "gsplat_commit": git_head(gsplat),
        "glm_submodule_status": glm_status,
        "input_manifest_sha256": sha256_file(input_manifest),
        "weights": {
            name: {
                "path": spec["path"],
                "bytes": Path(spec["path"]).stat().st_size,
                "sha256": sha256_file(Path(spec["path"])),
            }
            for name, spec in config["weights"].items()
        },
        "official_cli_help": "passed",
        "pip_check": pip_check_status,
        "pip_check_output": pip_check_text,
        "setup_transport": {
            "path": "/tmp/asset_harvester_setup_transport_patch.sh",
            "sha256": sha256_file(
                Path("/tmp/asset_harvester_setup_transport_patch.sh")
            ),
        },
        "setup_logs": setup_logs,
        "transitive_models": transitive_models,
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(args.run_dir / "environment_setup_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
