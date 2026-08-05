#!/usr/bin/env python
"""Register and evaluate reusable scene-0230/0242 checkpoints for V3 A0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from scripts.run_worldsim_v3_a0_scene import (
    PATCHED_DRIVESTUDIO,
    PROJECT,
    TASK_ID,
    build_eval_command,
    checkpoint_contract,
    common_environment,
    compact_actor,
    run_stage,
)
from scripts.run_worldsim_v3_a0_smoke import (
    DRIVESTUDIO_ENV,
    atomic_json,
    command_output,
    now,
    sha256_file,
)


PRESETS: dict[str, dict[str, object]] = {
    "scene-0230": {
        "scene_index": 179,
        "source_run": Path(
            "/root/autodl-tmp/runs/dynamic_editing_v2/"
            "DR-V2-M5-STRESS-3SCENE-01/"
            "20260802T183100Z__scene0230-heldout-s0-r7"
        ),
        "checkpoint_rel": Path(
            "work_dirs/m5_stress/scene0230_m5_heldout_s0/"
            "checkpoint_final.pth"
        ),
        "sha256": "24a39f27dfeed36bbdb01ee14211aec51b414e6ab0e61915b71c1dddcdf61e49",
        "bytes": 398_652_534,
        "high_token": "af663976db5e412e83db033d309c5c29",
        "boundary_token": "18c7f0c5fa6b49449f71c9dbae5c31d4",
        "expected_high_gaussians": 4_747,
        "expected_boundary_gaussians": 1_914,
    },
    "scene-0242": {
        "scene_index": 191,
        "source_run": Path(
            "/root/autodl-tmp/runs/dynamic_editing_v2/"
            "DR-V2-M5-STRESS-3SCENE-01/"
            "20260802T180500Z__scene0242-heldout-s0-r5"
        ),
        "checkpoint_rel": Path(
            "work_dirs/m5_stress/scene0242_m5_heldout_s0/"
            "checkpoint_final.pth"
        ),
        "sha256": "16179d8f99becb86b6893a18ff036af72d78c9897f7aa2b0e297b735dd6c5fda",
        "bytes": 306_034_934,
        "high_token": "40f087d8d9d74c10ae7dc8a2f34e0df5",
        "boundary_token": "2c820a798ad943a299a83ec2dd494dd9",
        "expected_high_gaussians": 6_939,
        "expected_boundary_gaussians": None,
    },
}

_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def normalized_config(path: Path) -> dict[str, object]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"config root is not a mapping: {path}")
    payload = dict(payload)
    payload.pop("log_dir", None)
    data = dict(payload["data"])
    data.pop("scene_idx", None)
    payload["data"] = data
    return payload


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialize_checkpoint(
    source_checkpoint: Path,
    destination_dir: Path,
) -> tuple[Path, Path]:
    destination_dir.mkdir(parents=True, exist_ok=False)
    destination_checkpoint = destination_dir / "checkpoint_final.pth"
    os.link(source_checkpoint, destination_checkpoint)
    source_config = source_checkpoint.parent / "config.yaml"
    config = OmegaConf.load(source_config)
    config.log_dir = str(destination_dir)
    destination_config = destination_dir / "config.yaml"
    OmegaConf.save(config, destination_config)
    return destination_checkpoint, destination_config


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", choices=sorted(PRESETS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--eval-timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    preset = PRESETS[args.scene_name]
    source_run = Path(preset["source_run"])
    source_checkpoint = source_run / Path(preset["checkpoint_rel"])
    source_config = source_checkpoint.parent / "config.yaml"
    source_train_stage = source_run / "stages/train_heldout_30000.json"
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )

    for path in (source_checkpoint, source_config, source_train_stage, args.reference_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if source_checkpoint.stat().st_size != preset["bytes"]:
        raise RuntimeError("source checkpoint byte contract changed")
    if sha256_file(source_checkpoint) != preset["sha256"]:
        raise RuntimeError("source checkpoint SHA-256 contract changed")
    source_stage = json.loads(source_train_stage.read_text(encoding="utf-8"))
    if source_stage.get("status") != "done":
        raise RuntimeError(f"source training stage is not done: {source_stage}")

    source_signature = normalized_config(source_config)
    reference_signature = normalized_config(args.reference_config)
    if source_signature != reference_signature:
        raise RuntimeError("source config differs beyond scene_idx/log_dir")
    signature_sha256 = canonical_sha256(source_signature)

    source_snapshot_files = (
        PROJECT / "scripts/register_worldsim_v3_a0_reuse.py",
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/prepare_worldsim_v3_drivestudio.py",
        PROJECT / "scripts/build_dr_v2_drivestudio_registry.py",
        PROJECT / "motion_proj/worldsim_v3/drivestudio_compat.py",
        PROJECT / "motion_proj/dynamic_editing_v2/drivestudio_registry.py",
        PROJECT / "compatibility/DriveStudio-2026-08-05.patch",
    )
    for source in source_snapshot_files:
        destination = args.run_dir / "source_snapshot" / source.relative_to(
            PROJECT
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    verify = subprocess.run(
        [
            "/root/miniconda3/bin/python",
            str(PROJECT / "scripts/prepare_worldsim_v3_drivestudio.py"),
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compatibility = json.loads(verify.stdout)
    checkpoint, materialized_config = materialize_checkpoint(
        source_checkpoint,
        args.run_dir / "assets/native_checkpoint",
    )
    source_stat = source_checkpoint.stat()
    destination_stat = checkpoint.stat()
    if source_stat.st_ino != destination_stat.st_ino:
        raise RuntimeError("checkpoint reuse must be a same-filesystem hardlink")
    checkpoint_info = checkpoint_contract(checkpoint)
    if checkpoint_info.get("step") != 30_000:
        raise RuntimeError(f"checkpoint step mismatch: {checkpoint_info}")

    reuse_audit = {
        "status": "done",
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": preset["sha256"],
        "source_checkpoint_bytes": preset["bytes"],
        "source_config": str(source_config),
        "reference_config": str(args.reference_config),
        "normalized_config_sha256": signature_sha256,
        "only_allowed_config_differences": ["data.scene_idx", "log_dir"],
        "hardlink": {
            "destination": str(checkpoint),
            "inode": source_stat.st_ino,
            "link_count": destination_stat.st_nlink,
        },
        "source_training": source_stage,
        "patched_drivestudio": compatibility,
    }
    atomic_json(args.run_dir / "stages/reuse_audit.json", reuse_audit)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": f"{args.scene_name} A0 native checkpoint reuse and held-out eval",
        "scene_name": args.scene_name,
        "scene_index": preset["scene_index"],
        "actors": {
            "high-support": preset["high_token"],
            "boundary-support": preset["boundary_token"],
        },
        "seed": 0,
        "test_image_stride": 10,
        "project_commit": command_output("git", "rev-parse", "HEAD", cwd=PROJECT),
        "project_status": command_output(
            "git", "status", "--short", cwd=PROJECT
        ).splitlines(),
        "reuse_audit": reuse_audit,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    environment = common_environment()
    registry = args.run_dir / "artifacts/actor_registry.json"
    registry_command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PROJECT / "scripts/build_dr_v2_drivestudio_registry.py"),
        "--checkpoint",
        str(checkpoint),
        "--drivestudio-root",
        str(PATCHED_DRIVESTUDIO),
        "--scene-name",
        args.scene_name,
        "--selected-token",
        str(preset["high_token"]),
        "--output",
        str(registry),
    ]
    run_stage(
        run_dir=args.run_dir,
        stage="actor_registry",
        command=registry_command,
        cwd=PROJECT,
        environment=environment,
        validate=lambda: (
            registry.is_file() and registry.stat().st_size > 0,
            {"registry": str(registry)},
        ),
        timeout_seconds=900,
    )
    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    by_token = {
        row["instance_token"]: row
        for row in registry_payload.get("actors", [])
    }
    selected = {
        "high-support": compact_actor(by_token.get(str(preset["high_token"]))),
        "boundary-support": compact_actor(
            by_token.get(str(preset["boundary_token"]))
        ),
    }
    high = selected["high-support"]
    boundary = selected["boundary-support"]
    if high is None or high["gaussian_count"] != preset["expected_high_gaussians"]:
        raise RuntimeError(f"high-support registry contract changed: {high}")
    expected_boundary = preset["expected_boundary_gaussians"]
    if expected_boundary is not None and (
        boundary is None or boundary["gaussian_count"] != expected_boundary
    ):
        raise RuntimeError(f"boundary-support registry contract changed: {boundary}")
    if expected_boundary is None and boundary is not None:
        raise RuntimeError(f"boundary-support should ABSTAIN: {boundary}")
    atomic_json(
        args.run_dir / "stages/selected_actors.json",
        {
            "status": "done",
            "registry_sha256": sha256_file(registry),
            "selected": selected,
        },
    )

    eval_command = build_eval_command(checkpoint)
    eval_dir = checkpoint.parent / "metrics_eval"
    video_dir = checkpoint.parent / "videos_eval"
    eval_stage = run_stage(
        run_dir=args.run_dir,
        stage="eval_heldout",
        command=eval_command,
        cwd=PATCHED_DRIVESTUDIO,
        environment=environment,
        validate=lambda: (
            bool(list(eval_dir.glob("images_test_*.json")))
            and bool(list(video_dir.glob("test_set_*.mp4"))),
            {
                "metric_files": [str(path) for path in sorted(eval_dir.glob("*.json"))],
                "video_files": [str(path) for path in sorted(video_dir.glob("*.mp4"))],
            },
        ),
        timeout_seconds=args.eval_timeout_seconds,
    )
    metric_files = sorted(eval_dir.glob("images_test_*.json"))
    heldout_metrics = json.loads(metric_files[-1].read_text(encoding="utf-8"))
    eval_stage["heldout_metrics"] = heldout_metrics
    atomic_json(args.run_dir / "stages/eval_heldout.json", eval_stage)

    summary = {
        "status": "done",
        "scene_name": args.scene_name,
        "scene_index": preset["scene_index"],
        "checkpoint": checkpoint_info,
        "checkpoint_reuse": reuse_audit["hardlink"],
        "normalized_config_sha256": signature_sha256,
        "source_training_resources": {
            key: source_stage.get(key)
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib",
                "peak_cgroup_memory_bytes",
            )
        },
        "registry": str(registry),
        "registry_sha256": sha256_file(registry),
        "selected_actors": selected,
        "heldout_metrics": heldout_metrics,
        "eval_resources": {
            key: eval_stage[key]
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib_sampled",
                "peak_gpu_memory_mib_torch_log",
                "peak_cgroup_memory_bytes",
            )
        },
        "completed_at": now(),
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
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
                        "code": "A0_REUSE_EVAL_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise
