#!/usr/bin/env python
"""Run a controlled WorldSim V3 A1 calibration ablation for one scene."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from scripts.run_worldsim_v3_a0_scene import (
    checkpoint_contract,
    common_environment,
    compact_actor,
    run_stage,
)
from scripts.run_worldsim_v3_a0_smoke import (
    DRIVESTUDIO_ENV,
    PATCHED_DRIVESTUDIO,
    PROJECT,
    atomic_json,
    command_output,
    now,
    sha256_file,
)


TASK_ID = "WS-V3-A1-CALIBRATION-01"
VARIANTS = ("c0-off", "c2-factorized-isp", "c3-bounded-pose")
SCENES = {
    "scene-0230": {
        "scene_index": 179,
        "high-support": "af663976db5e412e83db033d309c5c29",
        "boundary-support": "18c7f0c5fa6b49449f71c9dbae5c31d4",
        "source_config": Path(
            "/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/"
            "20260805T171624Z__scene0230-reuse-eval-s0-r1/assets/"
            "native_checkpoint/config.yaml"
        ),
    },
    "scene-0242": {
        "scene_index": 191,
        "high-support": "40f087d8d9d74c10ae7dc8a2f34e0df5",
        "boundary-support": None,
        "source_config": Path(
            "/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/"
            "20260805T171914Z__scene0242-reuse-eval-s0-r1/assets/"
            "native_checkpoint/config.yaml"
        ),
    },
    "scene-0255": {
        "scene_index": 204,
        "high-support": "f4aa30b8d0b44e2381a4abeafbe17642",
        "boundary-support": "80c08b992f1d47359de644be24f491df",
        "source_config": Path(
            "/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/"
            "20260805T162355Z__scene0255-native30k-s0-r1/work_dirs/"
            "worldsim_v3/scene0255_a0_native_heldout_s0/config.yaml"
        ),
    },
}
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def build_train_command(
    run_dir: Path, config: Path, scene_name: str, variant: str, num_iters: int
) -> tuple[list[str], Path]:
    run_name = (
        f"{scene_name.replace('-', '')}_a1_{variant.replace('-', '_')}_"
        f"s0_i{num_iters}"
    )
    checkpoint = (
        run_dir / "work_dirs" / "worldsim_v3_a1" / run_name / "checkpoint_final.pth"
    )
    return (
        [
            str(DRIVESTUDIO_ENV / "bin/python"),
            str(PATCHED_DRIVESTUDIO / "tools/train.py"),
            "--config_file",
            str(config),
            "--output_root",
            str(run_dir / "work_dirs"),
            "--project",
            "worldsim_v3_a1",
            "--run_name",
            run_name,
        ],
        checkpoint,
    )


def calibration_contract(checkpoint: Path, variant: str) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu")
    models = payload.get("models", {})
    affine = models.get("Affine")
    pose = models.get("CamPose")
    result: dict[str, object] = {
        "variant": variant,
        "affine_present": affine is not None,
        "pose_present": pose is not None,
        "affine_keys": sorted(affine) if affine is not None else [],
        "pose_keys": sorted(pose) if pose is not None else [],
    }
    if affine is not None:
        result["affine_parameter_count"] = int(
            sum(value.numel() for value in affine.values() if torch.is_tensor(value))
        )
        camera_embedding = affine.get("camera_embedding.weight")
        if camera_embedding is not None:
            result["camera_embedding_abs_max"] = float(
                camera_embedding.detach().abs().max()
            )
    if pose is not None:
        result["pose_parameter_count"] = int(
            sum(value.numel() for value in pose.values() if torch.is_tensor(value))
        )
        raw = pose.get("embeds.weight")
        if raw is not None and variant == "c3-bounded-pose":
            translation = raw[:, :3]
            rotation = raw[:, 3:]
            translation_norm = torch.linalg.vector_norm(translation, dim=-1)
            rotation_norm = torch.linalg.vector_norm(rotation, dim=-1)
            bounded_translation = 0.15 * torch.tanh(translation_norm)
            bounded_rotation = torch.deg2rad(torch.tensor(2.0)) * torch.tanh(
                rotation_norm
            )
            result["bounded_translation_max_m"] = float(
                bounded_translation.max()
            )
            result["bounded_rotation_max_deg"] = float(
                torch.rad2deg(bounded_rotation.max())
            )
    expected = {
        "c0-off": (False, False),
        "c2-factorized-isp": (True, True),
        "c3-bounded-pose": (True, True),
    }[variant]
    result["expected_modules_present"] = (
        result["affine_present"],
        result["pose_present"],
    ) == expected
    del payload, models, affine, pose
    gc.collect()
    return result


def provenance_contract(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"exists": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    background = payload.get("background_lidar_sample") or {}
    instances = payload.get("instance_lidar_samples") or {}
    return {
        "exists": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "truth_tier": payload.get("truth_tier"),
        "background_point_count": background.get("point_count"),
        "background_points_sha256": background.get("points_sha256"),
        "instance_count": len(instances),
        "instance_point_count": sum(
            int(row.get("point_count", 0)) for row in instances.values()
        ),
        "initialized_gaussians": payload.get("initialized_gaussians"),
        "limitations": payload.get("limitations"),
    }


def materialize(
    *, source: Path, variant: str, num_iters: int, output: Path, log: Path
) -> list[str]:
    command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PROJECT / "scripts/materialize_worldsim_v3_a1_config.py"),
        "--source-config",
        str(source),
        "--variant",
        variant,
        "--num-iters",
        str(num_iters),
        "--output",
        str(output),
    ]
    with log.open("xb") as stream:
        process = subprocess.run(
            command,
            cwd=PROJECT,
            env=common_environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0 or not output.is_file():
        raise RuntimeError(f"config materialization failed with {process.returncode}")
    atomic_json(
        output.parent.parent / "stages/materialize_config.json",
        {
            "status": "done",
            "command": command,
            "source_config": str(source),
            "source_config_sha256": sha256_file(source),
            "output_config": str(output),
            "output_config_sha256": sha256_file(output),
        },
    )
    return command


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene", choices=sorted(SCENES), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--num-iters", type=int, default=30_000)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--train-timeout-seconds", type=float, default=10_800)
    parser.add_argument("--eval-timeout-seconds", type=float, default=1_800)
    args = parser.parse_args()
    if args.num_iters <= 0:
        raise ValueError("num-iters must be positive")
    if args.formal and args.num_iters != 30_000:
        raise ValueError("formal A1 runs must use the frozen 30000-iteration budget")
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    scene = SCENES[args.scene]
    source_config = scene["source_config"]
    if not isinstance(source_config, Path) or not source_config.is_file():
        raise FileNotFoundError(source_config)
    _ACTIVE_RUN_DIR = args.run_dir
    for name in (
        "artifacts",
        "environment",
        "logs",
        "source_snapshot",
        "stages",
    ):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )

    sources = (
        PROJECT / "scripts/run_worldsim_v3_a1_scene.py",
        PROJECT / "scripts/materialize_worldsim_v3_a1_config.py",
        PROJECT / "scripts/eval_worldsim_v3_a0_actor_metrics.py",
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        PROJECT / "scripts/build_dr_v2_drivestudio_registry.py",
        PROJECT / "motion_proj/worldsim_v3/calibration.py",
        PROJECT / "motion_proj/worldsim_v3/trainer.py",
        PROJECT / "motion_proj/worldsim_v3/actor_metrics.py",
    )
    for source in sources:
        destination = args.run_dir / "source_snapshot" / source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    materialized_config = args.run_dir / "artifacts/a1_config.yaml"
    materialize_command = materialize(
        source=source_config,
        variant=args.variant,
        num_iters=args.num_iters,
        output=materialized_config,
        log=args.run_dir / "logs/materialize_config.log",
    )
    train_command, checkpoint = build_train_command(
        args.run_dir,
        materialized_config,
        args.scene,
        args.variant,
        args.num_iters,
    )
    provenance_path = args.run_dir / "artifacts/init_provenance.json"
    environment = common_environment()
    environment["WORLDSIM_V3_INIT_PROVENANCE"] = str(provenance_path)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": "A1 formal calibration ablation" if args.formal else "A1 engineering smoke",
        "scene_name": args.scene,
        "scene_index": scene["scene_index"],
        "variant": args.variant,
        "seed": 0,
        "test_image_stride": 10,
        "num_iters": args.num_iters,
        "formal": args.formal,
        "rolling_shutter": "not_supported",
        "actors": {
            "high-support": scene["high-support"],
            "boundary-support": scene["boundary-support"],
        },
        "source_config": str(source_config),
        "source_config_sha256": sha256_file(source_config),
        "materialize_command": materialize_command,
        "materialized_config_sha256": sha256_file(materialized_config),
        "project_commit": command_output("git", "rev-parse", "HEAD", cwd=PROJECT),
        "project_status": command_output(
            "git", "status", "--short", cwd=PROJECT
        ).splitlines(),
        "train_command": train_command,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    stage_name = f"train_{args.num_iters}"
    train_stage = run_stage(
        run_dir=args.run_dir,
        stage=stage_name,
        command=train_command,
        cwd=PATCHED_DRIVESTUDIO,
        environment=environment,
        validate=lambda: (
            checkpoint.is_file()
            and checkpoint.stat().st_size > 0
            and provenance_path.is_file()
            and provenance_path.stat().st_size > 0,
            {
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size
                if checkpoint.is_file()
                else 0,
                "init_provenance": str(provenance_path),
                "init_provenance_exists": provenance_path.is_file(),
            },
        ),
        timeout_seconds=args.train_timeout_seconds,
    )
    checkpoint_info = checkpoint_contract(checkpoint)
    if checkpoint_info.get("step") != args.num_iters:
        raise RuntimeError(f"checkpoint step mismatch: {checkpoint_info}")
    calibration_info = calibration_contract(checkpoint, args.variant)
    if not calibration_info["expected_modules_present"]:
        raise RuntimeError(f"calibration module contract failed: {calibration_info}")
    provenance_info = provenance_contract(provenance_path)
    if (
        not provenance_info["exists"]
        or provenance_info["truth_tier"] != "exact_runtime_initialization_inputs"
        or not provenance_info["background_points_sha256"]
    ):
        raise RuntimeError(f"initialization provenance contract failed: {provenance_info}")
    train_stage.update(
        {
            "checkpoint_contract": checkpoint_info,
            "calibration_contract": calibration_info,
            "provenance_contract": provenance_info,
        }
    )
    atomic_json(args.run_dir / f"stages/{stage_name}.json", train_stage)

    base_summary: dict[str, object] = {
        "status": "done",
        "task_id": TASK_ID,
        "scene_name": args.scene,
        "scene_index": scene["scene_index"],
        "variant": args.variant,
        "num_iters": args.num_iters,
        "formal": args.formal,
        "checkpoint": checkpoint_info,
        "calibration": calibration_info,
        "initialization_provenance": provenance_info,
        "train_resources": {
            key: train_stage[key]
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib_sampled",
                "peak_gpu_memory_mib_torch_log",
                "peak_cgroup_memory_bytes",
            )
        },
    }
    if not args.formal:
        base_summary["completed_at"] = now()
        atomic_json(args.run_dir / "summary.json", base_summary)
        atomic_json(
            args.run_dir / "terminal.json",
            {"status": "done", "updated_at": now(), "failure": None},
        )
        _TERMINAL_FINAL = True
        print(json.dumps(base_summary, indent=2, sort_keys=True))
        return

    registry = args.run_dir / "artifacts/actor_registry.json"
    registry_command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PROJECT / "scripts/build_dr_v2_drivestudio_registry.py"),
        "--checkpoint",
        str(checkpoint),
        "--drivestudio-root",
        str(PATCHED_DRIVESTUDIO),
        "--scene-name",
        args.scene,
        "--selected-token",
        str(scene["high-support"]),
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
        row["instance_token"]: row for row in registry_payload.get("actors", [])
    }
    selected = {
        "high-support": compact_actor(by_token.get(str(scene["high-support"]))),
        "boundary-support": compact_actor(
            by_token.get(str(scene["boundary-support"]))
        )
        if scene["boundary-support"] is not None
        else None,
    }
    if selected["high-support"] is None:
        raise RuntimeError("high-support actor is missing from registry")

    eval_command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PATCHED_DRIVESTUDIO / "tools/eval.py"),
        "--resume_from",
        str(checkpoint),
        "--render_video_postfix",
        f"a1_{args.variant}",
        "render.render_test=true",
        "render.render_full=false",
        "render.render_novel=null",
    ]
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

    evaluator_source = dict(base_summary)
    evaluator_source.update(
        {
            "registry": str(registry),
            "registry_sha256": sha256_file(registry),
            "selected_actors": selected,
            "heldout_metrics": heldout_metrics,
        }
    )
    evaluator_source_path = args.run_dir / "artifacts/evaluator_source_summary.json"
    atomic_json(evaluator_source_path, evaluator_source)
    actor_output = args.run_dir / "artifacts/actor_metrics"
    actor_command = [
        str(DRIVESTUDIO_ENV / "bin/python"),
        str(PROJECT / "scripts/eval_worldsim_v3_a0_actor_metrics.py"),
        "--source-summary",
        str(evaluator_source_path),
        "--output-dir",
        str(actor_output),
        "--drivestudio-root",
        str(PATCHED_DRIVESTUDIO),
    ]

    def validate_actor_metrics() -> tuple[bool, dict[str, object]]:
        path = actor_output / "summary.json"
        if not path.is_file():
            return False, {"summary": str(path), "exists": False}
        payload = json.loads(path.read_text(encoding="utf-8"))
        roles = payload.get("roles", {})
        valid = {"done", "ABSTAIN"}
        ok = (
            payload.get("status") == "done"
            and set(roles) == {"high-support", "boundary-support"}
            and all(row.get("status") in valid for row in roles.values())
            and payload.get("checkpoint_sha256_before")
            == payload.get("checkpoint_sha256_after")
        )
        return ok, {
            "summary": str(path),
            "role_statuses": {
                key: value.get("status") for key, value in roles.items()
            },
            "checkpoint_unchanged": payload.get("checkpoint_sha256_before")
            == payload.get("checkpoint_sha256_after"),
        }

    actor_stage = run_stage(
        run_dir=args.run_dir,
        stage="actor_metrics_heldout",
        command=actor_command,
        cwd=PROJECT,
        environment=environment,
        validate=validate_actor_metrics,
        timeout_seconds=args.eval_timeout_seconds,
    )
    actor_metrics = json.loads(
        (actor_output / "summary.json").read_text(encoding="utf-8")
    )
    base_summary.update(
        {
            "registry": str(registry),
            "registry_sha256": sha256_file(registry),
            "selected_actors": selected,
            "heldout_metrics": heldout_metrics,
            "actor_metrics": actor_metrics,
            "eval_resources": {
                key: eval_stage[key]
                for key in (
                    "duration_seconds",
                    "peak_gpu_memory_mib_sampled",
                    "peak_gpu_memory_mib_torch_log",
                    "peak_cgroup_memory_bytes",
                )
            },
            "actor_metric_resources": {
                key: actor_stage[key]
                for key in (
                    "duration_seconds",
                    "peak_gpu_memory_mib_sampled",
                    "peak_gpu_memory_mib_torch_log",
                    "peak_cgroup_memory_bytes",
                )
            },
            "completed_at": now(),
        }
    )
    atomic_json(args.run_dir / "summary.json", base_summary)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    _TERMINAL_FINAL = True
    print(json.dumps(base_summary, indent=2, sort_keys=True))


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
                        "code": "A1_SCENE_RUN_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise
