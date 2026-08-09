#!/usr/bin/env python
"""WorldSim V3 A2-I0 ancestry instrumentation 正式控制器。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path("/root/autodl-tmp/motion_proj")
MOTIONPROJ_PYTHON = Path("/root/autodl-tmp/envs/motionproj/bin/python")
DRIVESTUDIO_PYTHON = Path("/root/autodl-tmp/envs/drivestudio/bin/python")

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v3.gaussian_ancestry import (
    validate_a2_instrumentation_contract,
)
from scripts.run_worldsim_v3_a0_scene import run_stage
from scripts.run_worldsim_v3_a0_smoke import atomic_json, command_output, now
from scripts.prepare_worldsim_v3_a2_drivestudio import (
    sha256_file,
    verify_patched_tree,
)


_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def source_snapshot(run_dir: Path, sources: tuple[Path, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source in sources:
        relative = source.relative_to(PROJECT)
        destination = run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256_file(source)
    return hashes


def environment(drivestudio_root: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "PYTHONPATH": f"{PROJECT}:{drivestudio_root}",
            "WANDB_MODE": "disabled",
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
        }
    )
    return value


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a2_instrumentation_v1.yaml",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    contract = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_a2_instrumentation_contract(contract)
    drivestudio_root = Path(contract["drivestudio"]["patched_worktree"])
    compatibility_patch = PROJECT / contract["drivestudio"][
        "compatibility_patch"
    ]
    instrumentation_patch = PROJECT / contract["drivestudio"][
        "instrumentation_patch"
    ]
    worktree_evidence = verify_patched_tree(
        drivestudio_root,
        compatibility_patch,
        instrumentation_patch,
    )

    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.run_dir / "resolved.yaml")
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    atomic_json(args.run_dir / "artifacts/worktree.json", worktree_evidence)

    sources = (
        PROJECT / "scripts/run_worldsim_v3_a2_instrumentation.py",
        PROJECT / "scripts/prepare_worldsim_v3_a2_drivestudio.py",
        PROJECT / "scripts/smoke_worldsim_v3_a2_ancestry.py",
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        PROJECT / "motion_proj/worldsim_v3/gaussian_ancestry.py",
        args.config,
        compatibility_patch,
        instrumentation_patch,
    )
    source_hashes = source_snapshot(args.run_dir, sources)
    commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    resolved_sha = sha256_file(args.run_dir / "resolved.yaml")
    fingerprint = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "component": "A2-I0 ancestry instrumentation",
        "project_commit": commit,
        "project_status": command_output(
            "git", "status", "--short", cwd=PROJECT
        ).splitlines(),
        "source_hashes": source_hashes,
        "resolved_config_sha256": resolved_sha,
        "upstream_commit": worktree_evidence["upstream_commit"],
        "compatibility_patch_sha256": worktree_evidence[
            "compatibility_patch_sha256"
        ],
        "instrumentation_patch_sha256": worktree_evidence[
            "instrumentation_patch_sha256"
        ],
    }
    atomic_json(args.run_dir / "fingerprint.json", fingerprint)

    smoke_output = args.run_dir / "artifacts/module_off_equivalence.json"
    command = [
        str(DRIVESTUDIO_PYTHON),
        str(PROJECT / "scripts/smoke_worldsim_v3_a2_ancestry.py"),
        "--config",
        str(args.run_dir / "resolved.yaml"),
        "--output",
        str(smoke_output),
    ]
    started_at = now()
    manifest = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "component": "A2-I0 ancestry instrumentation formal smoke",
        "status": "running",
        "seed": int(contract["seed"]),
        "split": "synthetic RigidNodes deterministic refinement contract",
        "project_commit": commit,
        "drivestudio_root": str(drivestudio_root),
        "command": command,
        "started_at": started_at,
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    environment_evidence = {
        "motionproj_python": str(MOTIONPROJ_PYTHON),
        "drivestudio_python": str(DRIVESTUDIO_PYTHON),
        "drivestudio_python_version": command_output(
            str(DRIVESTUDIO_PYTHON), "--version"
        ),
        "nvidia_smi": command_output("nvidia-smi", "-L"),
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max")
        .read_text(encoding="utf-8")
        .strip(),
    }
    atomic_json(args.run_dir / "environment/environment.json", environment_evidence)

    def validate() -> tuple[bool, dict[str, object]]:
        if not smoke_output.is_file():
            return False, {"output": str(smoke_output), "exists": False}
        payload = json.loads(smoke_output.read_text(encoding="utf-8"))
        ok = (
            payload.get("status") == "done"
            and payload.get("native_tensor_bitwise_equal") is True
            and payload.get("off_native_checkpoint_keys_equal_on") is True
            and payload.get("off_has_ancestry_checkpoint_key") is False
            and payload.get("on_has_ancestry_checkpoint_key") is True
            and payload.get("on_ancestry", {}).get("live_gaussians")
            == payload.get("on_gaussian_count")
        )
        return ok, {
            "output": str(smoke_output),
            "exists": True,
            "sha256": sha256_file(smoke_output),
            "status": payload.get("status"),
            "native_tensor_bitwise_equal": payload.get(
                "native_tensor_bitwise_equal"
            ),
        }

    stage = run_stage(
        run_dir=args.run_dir,
        stage="a2_i0_module_off_equivalence",
        command=command,
        cwd=drivestudio_root,
        environment=environment(drivestudio_root),
        validate=validate,
        timeout_seconds=args.timeout_seconds,
    )
    smoke = json.loads(smoke_output.read_text(encoding="utf-8"))
    completed_at = now()
    stage_manifest = {
        "schema_version": 1,
        "stage": "a2_i0_module_off_equivalence",
        "status": "done",
        "input_hashes": {
            "resolved_config_sha256": resolved_sha,
            "instrumentation_patch_sha256": worktree_evidence[
                "instrumentation_patch_sha256"
            ],
            "source_snapshot_sha256": source_hashes,
        },
        "output_hashes": {
            "module_off_equivalence_sha256": sha256_file(smoke_output),
            "stage_log_sha256": sha256_file(
                args.run_dir / "logs/a2_i0_module_off_equivalence.log"
            ),
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "return_code": stage["return_code"],
        "resource_peak": {
            "gpu_memory_mib": stage["peak_gpu_memory_mib_sampled"],
            "cgroup_memory_bytes": stage["peak_cgroup_memory_bytes"],
        },
        "resume_safe": True,
        "invalidation_dependency": [
            "resolved_config_sha256",
            "instrumentation_patch_sha256",
            "source_snapshot_sha256",
        ],
    }
    atomic_json(args.run_dir / "stage_manifest.json", stage_manifest)
    shutil.copy2(smoke_output, args.run_dir / "metrics.json")
    artifacts = {
        "worktree": {
            "path": str(args.run_dir / "artifacts/worktree.json"),
            "sha256": sha256_file(args.run_dir / "artifacts/worktree.json"),
        },
        "module_off_equivalence": {
            "path": str(smoke_output),
            "sha256": sha256_file(smoke_output),
        },
        "stage_manifest": {
            "path": str(args.run_dir / "stage_manifest.json"),
            "sha256": sha256_file(args.run_dir / "stage_manifest.json"),
        },
    }
    atomic_json(args.run_dir / "artifacts.json", artifacts)
    summary = {
        "status": "done",
        "task_id": contract["task_id"],
        "component": "A2-I0 ancestry instrumentation",
        "seed": int(contract["seed"]),
        "split": manifest["split"],
        "resolved_config_sha256": resolved_sha,
        "instrumentation_patch_sha256": worktree_evidence[
            "instrumentation_patch_sha256"
        ],
        "module_off_equivalence": smoke,
        "resources": {
            "duration_seconds": stage["duration_seconds"],
            "peak_gpu_memory_mib_sampled": stage[
                "peak_gpu_memory_mib_sampled"
            ],
            "peak_cgroup_memory_bytes": stage[
                "peak_cgroup_memory_bytes"
            ],
        },
        "completed_at": completed_at,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    (args.run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# A2-I0 ancestry instrumentation",
                "",
                "- 状态：`done`",
                f"- 配置 SHA-256：`{resolved_sha}`",
                "- module-off 原生 tensor：逐位相等",
                "- module-off checkpoint 键：与原生一致",
                "- split/clone/prune lineage：通过",
                "- ancestry checkpoint roundtrip：通过",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest["status"] = "done"
    manifest["completed_at"] = completed_at
    atomic_json(args.run_dir / "manifest.json", manifest)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": completed_at, "failure": None},
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if (
            _ACTIVE_RUN_DIR is not None
            and _ACTIVE_RUN_DIR.is_dir()
            and not _TERMINAL_FINAL
        ):
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {
                        "code": "A2_INSTRUMENTATION_RUN_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise
