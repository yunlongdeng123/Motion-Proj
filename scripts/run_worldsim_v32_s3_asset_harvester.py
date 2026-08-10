#!/usr/bin/env python
"""冻结输入与随机种子后，调用官方 Asset Harvester image-to-3D。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file


GIB = 1024**3


def read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None if value == "max" else int(value)


def read_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    return {
        key: int(value)
        for key, value in (
            line.split() for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


class ResourceMonitor:
    """在正式推理期间采样 GPU 与 cgroup，留下单卡资源证据。"""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.peak_nvidia_mib = 0
        self.peak_cgroup_current_bytes = 0
        self.cgroup_limit_bytes = read_int(Path("/sys/fs/cgroup/memory.max"))
        self.events_before = read_memory_events()
        self.disk_free_before_bytes = shutil.disk_usage("/root/autodl-tmp").free
        self.consecutive_memory_pressure_samples = 0
        self.memory_pressure_observed = False

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            current = read_int(Path("/sys/fs/cgroup/memory.current")) or 0
            self.peak_cgroup_current_bytes = max(
                self.peak_cgroup_current_bytes, current
            )
            if self.cgroup_limit_bytes and current >= 0.9 * self.cgroup_limit_bytes:
                self.consecutive_memory_pressure_samples += 1
                if self.consecutive_memory_pressure_samples >= 2:
                    self.memory_pressure_observed = True
            else:
                self.consecutive_memory_pressure_samples = 0
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                for line in raw.splitlines():
                    self.peak_nvidia_mib = max(
                        self.peak_nvidia_mib, int(line.strip())
                    )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def start(self) -> None:
        if self.disk_free_before_bytes < 20 * GIB:
            raise RuntimeError("S3 disk free 低于冻结的 20 GiB 停止门")
        self._thread.start()

    def finish(self) -> dict[str, object]:
        self._stop.set()
        self._thread.join(timeout=10)
        events_after = read_memory_events()
        keys = set(self.events_before) | set(events_after)
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_nvidia_memory_mib_sampled": self.peak_nvidia_mib,
            "peak_cgroup_current_bytes_sampled": self.peak_cgroup_current_bytes,
            "cgroup_memory_max_bytes": self.cgroup_limit_bytes,
            "memory_pressure_observed": self.memory_pressure_observed,
            "memory_events_before": self.events_before,
            "memory_events_after": events_after,
            "memory_events_delta": {
                key: events_after.get(key, 0) - self.events_before.get(key, 0)
                for key in sorted(keys)
            },
            "disk_free_before_bytes": self.disk_free_before_bytes,
            "disk_free_after_bytes": shutil.disk_usage("/root/autodl-tmp").free,
        }


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def verify_file(spec: dict[str, str], name: str) -> Path:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"S3 {name} SHA 漂移: {actual}")
    return path


def validate_hf_snapshot(hf_home: Path, spec: dict[str, str]) -> dict[str, object]:
    """验证官方推理的传递模型已在镜像缓存中冻结到指定 revision。"""
    repo_id = spec["repo_id"]
    revision = spec["revision"]
    cache_name = "models--" + repo_id.replace("/", "--")
    repo_cache = hf_home / "hub" / cache_name
    main_ref = repo_cache / "refs" / "main"
    if not main_ref.is_file():
        raise FileNotFoundError(f"S3 HF main ref 缺失: {main_ref}")
    cached_revision = main_ref.read_text(encoding="utf-8").strip()
    if cached_revision != revision:
        raise RuntimeError(
            f"S3 HF revision 漂移: {repo_id} {cached_revision} != {revision}"
        )
    snapshot = repo_cache / "snapshots" / revision
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    if not files:
        raise RuntimeError(f"S3 HF snapshot 为空: {snapshot}")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot": str(snapshot),
        "file_count": len(files),
        "resolved_bytes": sum(path.stat().st_size for path in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"S3 inference output 已存在: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    expected_manifest_sha = config["inputs"]["input_manifest_sha256"]
    actual_manifest_sha = sha256_file(args.input_manifest)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError("S3 input manifest SHA 漂移")
    if manifest["instance_token"] != config["actor"]["instance_token"]:
        raise RuntimeError("S3 input/actor instance token 错配")
    if not manifest.get("heldout_excluded"):
        raise RuntimeError("S3 input 未证明 heldout 排除")

    source = Path(config["source"]["checkout"])
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["source"]["commit"]:
        raise RuntimeError(f"Asset Harvester source commit 漂移: {head}")
    diffusion = verify_file(config["weights"]["diffusion"], "diffusion")
    lifting = verify_file(config["weights"]["lifting"], "lifting")
    camera = verify_file(config["weights"]["camera_estimator"], "camera estimator")
    runtime = config["runtime"]
    hf_home = Path(runtime["hf_home"])
    transitive_models = {
        name: validate_hf_snapshot(hf_home, spec)
        for name, spec in config["transitive_models"].items()
    }
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_ENDPOINT"] = str(runtime["hf_endpoint"])
    if bool(runtime["offline"]):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"
    samples_dir = Path(manifest["samples_dir"])
    if not samples_dir.is_dir():
        raise FileNotFoundError(samples_dir)
    sample_names = {row["sample"] for row in manifest["samples"]}
    if sample_names != {"high_support_1view", "high_support_2view"}:
        raise RuntimeError(f"S3 sample set 漂移: {sorted(sample_names)}")

    if not torch.cuda.is_available():
        raise RuntimeError("Asset Harvester 正式推理需要可见 CUDA GPU")
    # PyTorch 2.10 的峰值显存计数器要求 CUDA 上下文已经显式初始化。
    # manual_seed_all 会延迟排队操作，不会建立上下文。
    torch.cuda.set_device(0)
    torch.cuda.init()
    seed = int(config["inference"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    resources = ResourceMonitor()
    resources.start()
    started = time.monotonic()

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
        str(args.output_dir),
        "--num_steps",
        str(int(config["inference"]["num_steps"])),
        "--cfg_scale",
        str(float(config["inference"]["cfg_scale"])),
        "--precision",
        str(config["inference"]["precision"]),
    ]
    if bool(config["inference"]["offload_model_to_cpu"]):
        official_argv.append("--offload_model_to_cpu")
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(source))
    try:
        os.chdir(source)
        sys.argv = official_argv
        runpy.run_path(str(source / "run_inference.py"), run_name="__main__")
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        resource_summary = resources.finish()
    if resource_summary["memory_pressure_observed"]:
        raise RuntimeError("S3 cgroup memory 连续两次达到 90% 停止门")
    oom_delta = resource_summary["memory_events_delta"].get("oom", 0)
    oom_kill_delta = resource_summary["memory_events_delta"].get("oom_kill", 0)
    if oom_delta or oom_kill_delta:
        raise RuntimeError(
            f"S3 cgroup OOM 事件增加: oom={oom_delta}, oom_kill={oom_kill_delta}"
        )

    plys = {
        name: args.output_dir / name / "gaussians.ply"
        for name in sorted(sample_names)
    }
    for name, path in plys.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"S3 {name} 未生成非空 gaussians.ply")
    files = []
    for path in sorted(value for value in args.output_dir.rglob("*") if value.is_file()):
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    summary = {
        "schema_version": "worldsim_v32_s3_asset_harvester_inference_v1",
        "task_id": config["task_id"],
        "status": "done",
        "source_commit": head,
        "implementation": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            }
        },
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": actual_manifest_sha,
        "seed": seed,
        "official_argv": official_argv,
        "camera_source": manifest["camera_source"],
        "generation_provenance": "GENERATED_ACTOR",
        "transitive_models": transitive_models,
        "hf_offline": bool(runtime["offline"]),
        "plys": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in plys.items()
        },
        "files": files,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
            **resource_summary,
        },
    }
    atomic_json(args.output_dir / "inference_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
