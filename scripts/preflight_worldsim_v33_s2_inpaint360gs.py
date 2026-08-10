#!/usr/bin/env python3
"""对官方 Inpaint360GS 做可复现的单卡与输入契约预检，不修改第三方源码。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import yaml


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


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"run-dir 必须为空: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_s2_inpaint360gs_preflight_v1":
        raise ValueError("Inpaint360GS preflight config schema 漂移")
    checkout = Path(config["upstream"]["checkout"])
    commit = command("git", "rev-parse", "HEAD", cwd=checkout)
    dirty = command("git", "status", "--porcelain", cwd=checkout)
    license_sha = sha256_file(config["upstream"]["license"])
    source_sha = sha256_file(config["source"]["checkpoint"])
    gpu_text = command(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    gpu_name, memory_mib, driver = [value.strip() for value in gpu_text.split(",")]

    environments = {
        "main": {
            "path": config["host"]["main_env"],
            "present": Path(config["host"]["main_env"]).is_dir(),
        },
        "lama": {
            "path": config["host"]["lama_env"],
            "present": Path(config["host"]["lama_env"]).is_dir(),
        },
    }
    weights = {
        role: {
            "path": str(checkout / relative),
            "present": (checkout / relative).is_file(),
            **(
                {
                    "sha256": sha256_file(checkout / relative),
                    "bytes": (checkout / relative).stat().st_size,
                }
                if (checkout / relative).is_file()
                else {}
            ),
        }
        for role, relative in config["required_weights"].items()
    }
    source_code = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in checkout.rglob("*.py")
        if path.stat().st_size < 2_000_000
    )
    adapter_markers = {
        marker: marker.lower() in source_code.lower()
        for marker in ("DriveStudio", "StreetGS", "checkpoint_final.pth")
    }
    official_contract_markers = {
        marker: marker.lower() in source_code.lower()
        for marker in ("sparse/0", "point_cloud.ply", "images_inpaint_unseen_virtual")
    }
    checks = {
        "commit_exact": commit == config["upstream"]["commit"],
        "checkout_clean": dirty == "",
        "license_exact": license_sha == config["upstream"]["license_sha256"],
        "source_checkpoint_exact": source_sha == config["source"]["checkpoint_sha256"],
        "gpu_exact": gpu_name == config["host"]["expected_gpu"],
        "gpu_memory_exact": int(memory_mib) == int(config["host"]["expected_memory_mib"]),
        "main_environment_present": environments["main"]["present"],
        "lama_environment_present": environments["lama"]["present"],
        "all_required_weights_present": all(row["present"] for row in weights.values()),
        "official_streetgs_adapter_present": any(adapter_markers.values()),
        "official_input_contract_detected": all(official_contract_markers.values()),
        "documented_gpu_matches_host": config["upstream"]["documented_gpu"] in gpu_name,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    status = (
        "ready_for_official_single_gpu_run"
        if not blockers
        else config["decision"]["missing_prerequisite_status"]
    )
    report = {
        "task_id": config["task_id"],
        "status": status,
        "official_execution_attempted": False,
        "reason": (
            "所有官方前置条件已满足"
            if not blockers
            else "官方运行前置条件或 StreetGS 输入适配缺失；按冻结协议不得临时改写上游方法"
        ),
        "blockers": blockers,
        "checks": checks,
        "upstream": {
            "checkout": str(checkout),
            "commit": commit,
            "dirty": dirty,
            "license_sha256": license_sha,
            "documented_gpu": config["upstream"]["documented_gpu"],
            "documented_cuda": config["upstream"]["documented_cuda"],
            "documented_lama_cuda": config["upstream"]["documented_lama_cuda"],
            "official_finetune_iterations": config["official_input_contract"]["finetune_iterations"],
        },
        "host": {"gpu": gpu_name, "memory_mib": int(memory_mib), "driver": driver},
        "environments": environments,
        "weights": weights,
        "adapter_markers": adapter_markers,
        "official_contract_markers": official_contract_markers,
        "source": {
            "checkpoint": config["source"]["checkpoint"],
            "checkpoint_sha256": source_sha,
            "schema": config["source"]["schema"],
            "processed_root": config["source"]["processed_root"],
        },
        "decision_contract": config["decision"],
    }
    artifacts = args.run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_json(artifacts / "preflight.json", report)
    snapshot = args.run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in (args.config.resolve(), Path(__file__).resolve()):
        shutil.copy2(source, snapshot / source.name)
    summary = {
        "task_id": config["task_id"],
        "status": status,
        "blockers": blockers,
        "official_execution_attempted": False,
        "config_sha256": sha256_file(args.config),
        "preflight_sha256": sha256_file(artifacts / "preflight.json"),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "status.json",
        {
            "state": "completed",
            "outcome": status,
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
