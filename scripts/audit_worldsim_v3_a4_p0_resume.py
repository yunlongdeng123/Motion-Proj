#!/usr/bin/env python
"""在无 torch 进程中执行 A4-P0 read-only resume dry-run。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


_ACTIVE_RUN_DIR: Path | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def gpu_rows() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_resume_audit(run_dir: Path) -> dict:
    started = time.perf_counter()
    if "torch" in sys.modules:
        raise RuntimeError("A4-P0 resume auditor imported torch")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before_gpu = gpu_rows()
    if before_gpu:
        raise RuntimeError(f"A4-P0 resume auditor GPU not idle: {before_gpu}")
    actions = {}
    for name in ("inventory", "runtime_probe", "aggregate"):
        path = run_dir / "stages" / f"{name}.json"
        actual = sha256_file(path)
        expected = manifest["stage_hashes"][name]
        if actual != expected:
            raise RuntimeError(f"A4-P0 resume stage hash drift: {name}")
        actions[name] = {
            "action": "reuse_completed_stage",
            "path": str(path.relative_to(run_dir)),
            "sha256": actual,
        }
    media_suffixes = {".pth", ".png", ".jpg", ".jpeg", ".mp4"}
    forbidden = [str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.suffix.lower() in media_suffixes]
    after_gpu = gpu_rows()
    return {
        "status": "done",
        "stage": "resume_audit",
        "actions": actions,
        "torch_imported": "torch" in sys.modules,
        "gpu_compute_rows_before": before_gpu,
        "gpu_compute_rows_after": after_gpu,
        "gpu_launch_observed": bool(before_gpu or after_gpu),
        "forbidden_checkpoint_or_media_outputs": forbidden,
        "dry_run_seconds": time.perf_counter() - started,
        "minimum_rerun_unit": "none_all_completed_stages_reusable",
    }


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    _ACTIVE_RUN_DIR = args.run_dir
    payload = build_resume_audit(args.run_dir)
    stage_path = args.run_dir / "stages" / "resume_audit.json"
    atomic_json(stage_path, payload)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage_hashes"]["resume_audit"] = sha256_file(stage_path)
    atomic_json(manifest_path, manifest, replace=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.exists():
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P0_RESUME_AUDIT_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
