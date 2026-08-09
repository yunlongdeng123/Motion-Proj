#!/usr/bin/env python
"""在无 torch/GPU 进程中执行 A4-P5 read-only resume dry-run。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


from scripts.run_worldsim_v3_a4_p5_registry import (
    atomic_json,
    canonical_sha256,
    nvidia_compute_rows,
    sha256_file,
)


_ACTIVE_RUN_DIR: Path | None = None


def build_resume_audit(run_dir: Path) -> dict:
    started = time.perf_counter()
    if "torch" in sys.modules:
        raise RuntimeError("A4-P5 resume auditor imported torch")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    before_gpu = nvidia_compute_rows()
    if before_gpu:
        raise RuntimeError(f"A4-P5 resume auditor GPU not idle: {before_gpu}")
    actions = {}
    for name in ("input_audit", "registry_materialize", "reload_smoke", "aggregate"):
        path = run_dir / "stages" / f"{name}.json"
        actual = sha256_file(path)
        if actual != manifest["stage_hashes"][name]:
            raise RuntimeError(f"A4-P5 resume stage hash drift: {name}")
        actions[name] = {
            "action": "reuse_completed_stage",
            "path": str(path.relative_to(run_dir)),
            "sha256": actual,
        }
    materialize = json.loads(
        (run_dir / "stages" / "registry_materialize.json").read_text(encoding="utf-8")
    )
    registry_path = run_dir / materialize["registry"]["path"]
    if sha256_file(registry_path) != materialize["registry"]["sha256"]:
        raise RuntimeError("A4-P5 deployment registry file hash drift")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected_canonical = registry.pop("registry_sha256")
    if canonical_sha256(registry) != expected_canonical:
        raise RuntimeError("A4-P5 deployment registry canonical hash drift")
    forbidden_suffixes = {".pth", ".png", ".jpg", ".jpeg", ".mp4"}
    forbidden = [
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.suffix.lower() in forbidden_suffixes
    ]
    after_gpu = nvidia_compute_rows()
    return {
        "status": "done",
        "stage": "resume_audit",
        "actions": actions,
        "deployment_registry_sha256": sha256_file(registry_path),
        "deployment_registry_canonical_sha256": expected_canonical,
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
                        "code": "A4_P5_RESUME_AUDIT_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
