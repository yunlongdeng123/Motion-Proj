#!/usr/bin/env python
"""为 M5 压力测试准备并训练一个含 held-out 切分的 DriveStudio 场景。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable


CAMERAS = (0, 1, 2)
EXPECTED_TIMESTEPS = 196


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def int_file(path: str) -> int | None:
    raw = Path(path).read_text(encoding="utf-8").strip()
    return None if raw == "max" else int(raw)


def memory_events() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split()
            for line in Path("/sys/fs/cgroup/memory.events")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }


def gpu_sample() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    return {
        "name": fields[0],
        "driver": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
    }


def resource_sample(stage: str, event: str) -> dict:
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "timestamp": now(),
        "stage": stage,
        "event": event,
        "memory_current_bytes": int_file("/sys/fs/cgroup/memory.current"),
        "memory_max_bytes": int_file("/sys/fs/cgroup/memory.max"),
        "memory_events": memory_events(),
        "disk_free_bytes": disk.free,
        "gpu": gpu_sample(),
    }


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def validate_processed(scene_root: Path) -> dict:
    counts = {
        "images": len(list((scene_root / "images").glob("*.jpg"))),
        "lidar": len(list((scene_root / "lidar").glob("*.bin"))),
        "lidar_pose": len(list((scene_root / "lidar_pose").glob("*.txt"))),
        "extrinsics": len(list((scene_root / "extrinsics").glob("*.txt"))),
        "sky_masks": len(list((scene_root / "sky_masks").glob("*.png"))),
        "dynamic_all": len(list((scene_root / "dynamic_masks/all").glob("*.png"))),
        "dynamic_human": len(list((scene_root / "dynamic_masks/human").glob("*.png"))),
        "dynamic_vehicle": len(list((scene_root / "dynamic_masks/vehicle").glob("*.png"))),
    }
    counts["instances_info"] = int(
        (scene_root / "instances" / "instances_info.json").is_file()
    )
    return counts


def processed_ready(counts: dict) -> bool:
    return (
        counts["images"] == EXPECTED_TIMESTEPS * 6
        and counts["lidar"] == EXPECTED_TIMESTEPS
        and counts["lidar_pose"] == EXPECTED_TIMESTEPS
        and counts["extrinsics"] == EXPECTED_TIMESTEPS * 6
        and counts["dynamic_all"] == EXPECTED_TIMESTEPS * 6
        and counts["dynamic_human"] == EXPECTED_TIMESTEPS * 6
        and counts["dynamic_vehicle"] == EXPECTED_TIMESTEPS * 6
        and counts["instances_info"] == 1
    )


def run_stage(
    *,
    run_dir: Path,
    stage: str,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    validate: Callable[[], tuple[bool, dict]],
    gpu_idle_required: bool = False,
    timeout_seconds: float | None = None,
) -> dict:
    stage_path = run_dir / "stages" / f"{stage}.json"
    log_path = run_dir / "logs" / f"{stage}.log"
    if stage_path.exists() or log_path.exists():
        raise FileExistsError(f"拒绝覆盖 M5 stage: {stage}")
    pre = resource_sample(stage, "preflight")
    append_jsonl(run_dir / "resource.jsonl", pre)
    if pre["disk_free_bytes"] < 45 * 2**30:
        raise RuntimeError(f"{stage}: 启动磁盘必须至少 45 GiB")
    if gpu_idle_required and pre["gpu"]["memory_used_mib"] > 2048:
        raise RuntimeError(f"{stage}: GPU 启动时非空闲: {pre['gpu']}")
    baseline_events = pre["memory_events"]
    started = time.monotonic()
    peak_gpu = int(pre["gpu"]["memory_used_mib"])
    peak_memory = int(pre["memory_current_bytes"] or 0)
    over_memory = 0
    stop_reason = None
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(10)
            current = resource_sample(stage, "running")
            append_jsonl(run_dir / "resource.jsonl", current)
            peak_gpu = max(peak_gpu, int(current["gpu"]["memory_used_mib"] or 0))
            peak_memory = max(peak_memory, int(current["memory_current_bytes"] or 0))
            maximum = current["memory_max_bytes"]
            used = current["memory_current_bytes"]
            over_memory = (
                over_memory + 1
                if maximum and used and used / maximum >= 0.90
                else 0
            )
            events = current["memory_events"]
            if over_memory >= 2:
                stop_reason = "memory.current/memory.max >= 0.90 twice"
            elif (
                events.get("oom", 0) > baseline_events.get("oom", 0)
                or events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0)
            ):
                stop_reason = "cgroup oom event increased"
            elif current["disk_free_bytes"] < 15 * 2**30:
                stop_reason = "disk free below 15 GiB"
            elif timeout_seconds and time.monotonic() - started > timeout_seconds:
                stop_reason = f"stage timeout after {timeout_seconds:.0f}s"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        return_code = process.wait()
    output_ok, evidence = validate()
    payload = {
        "stage": stage,
        "status": (
            "done"
            if return_code == 0 and stop_reason is None and output_ok
            else "blocked"
        ),
        "return_code": return_code,
        "stop_reason": stop_reason,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "log": str(log_path),
        "output_validation": evidence,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_cgroup_memory_bytes": peak_memory,
    }
    atomic_json(stage_path, payload)
    append_jsonl(run_dir / "resource.jsonl", resource_sample(stage, "completed"))
    if payload["status"] != "done":
        raise RuntimeError(f"M5 stage {stage} blocked: {payload}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--high-token", required=True)
    parser.add_argument("--boundary-token", required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/root/autodl-tmp/motion_proj")
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio"),
    )
    parser.add_argument(
        "--processed-base",
        type=Path,
        default=Path("/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed"),
    )
    parser.add_argument("--test-image-stride", type=int, default=10)
    args = parser.parse_args()

    if args.run_dir.exists():
        raise FileExistsError(f"M5 scene run 已存在: {args.run_dir}")
    for directory in (
        "artifacts",
        "environment",
        "logs",
        "source_snapshot",
        "stages",
    ):
        (args.run_dir / directory).mkdir(parents=True, exist_ok=True)
    for source in (
        args.project_root / "scripts" / "run_dr_v2_m5_scene_train.py",
        args.project_root / "scripts" / "prepare_dr_v2_drivestudio_scene.py",
        args.project_root / "scripts" / "build_adgs_nuscenes_assets.py",
        args.project_root / "scripts" / "preprocess_dr_v2_nuscenes_single.py",
        args.project_root / "scripts" / "build_dr_v2_sky_masks.py",
        args.project_root / "scripts" / "build_dr_v2_drivestudio_registry.py",
        args.project_root
        / "motion_proj"
        / "dynamic_editing_v2"
        / "drivestudio_registry.py",
    ):
        destination = args.run_dir / "source_snapshot" / source.relative_to(args.project_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    raw_root = Path(
        f"/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_raw_{args.scene_name.replace('-', '')}"
    )
    processed_root = Path(str(args.processed_base) + "_10Hz") / "trainval"
    scene_root = processed_root / f"{args.scene_index:03d}"
    raw_manifest = (
        Path("/root/autodl-tmp/data/dynamic_editing_v2/manifests")
        / f"{args.scene_name}_raw_manifest.json"
    )
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    project_commit = subprocess.check_output(
        ["git", "-C", str(args.project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    contract = {
        "schema_version": 1,
        "task_id": "DR-V2-M5-STRESS-3SCENE-01",
        "component": "held-out DriveStudio/StreetGS scene training",
        "scene_name": args.scene_name,
        "scene_index": args.scene_index,
        "actors": {
            "high-support": args.high_token,
            "boundary-support": args.boundary_token,
        },
        "test_image_stride": args.test_image_stride,
        "heldout_truth_policy": "every 10th image excluded before optimization",
        "project_commit": project_commit,
        "upstream_commit": upstream_commit,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", contract)
    atomic_json(args.run_dir / "resolved.json", contract)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )

    common_env = os.environ.copy()
    common_env.update(
        {
            "PYTHONPATH": str(args.upstream_root),
            "WANDB_MODE": "disabled",
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
            "TMPDIR": "/root/autodl-tmp/tmp",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
    )

    try:
        if raw_manifest.is_file() and json.loads(raw_manifest.read_text()).get("complete"):
            atomic_json(
                args.run_dir / "stages" / "raw_prepare.json",
                {
                    "stage": "raw_prepare",
                    "status": "done",
                    "reuse": True,
                    "manifest": str(raw_manifest),
                    "manifest_sha256": sha256_file(raw_manifest),
                },
            )
        else:
            run_stage(
                run_dir=args.run_dir,
                stage="raw_prepare",
                command=[
                    "/root/autodl-tmp/envs/motionproj/bin/python",
                    str(args.project_root / "scripts/prepare_dr_v2_drivestudio_scene.py"),
                    "--scene-name",
                    args.scene_name,
                    "--scene-index",
                    str(args.scene_index),
                    "--out-root",
                    str(raw_root),
                    "--manifest-dir",
                    str(raw_manifest.parent),
                    "--workers",
                    "1",
                ],
                cwd=args.project_root,
                environment=common_env,
                validate=lambda: (
                    raw_manifest.is_file()
                    and json.loads(raw_manifest.read_text()).get("complete") is True,
                    {"manifest": str(raw_manifest)},
                ),
                timeout_seconds=3600,
            )

        counts = validate_processed(scene_root)
        if processed_ready(counts):
            atomic_json(
                args.run_dir / "stages" / "preprocess.json",
                {
                    "stage": "preprocess",
                    "status": "done",
                    "reuse": True,
                    "scene_root": str(scene_root),
                    "counts": counts,
                },
            )
        else:
            run_stage(
                run_dir=args.run_dir,
                stage="preprocess",
                command=[
                    "/root/autodl-tmp/envs/drivestudio/bin/python",
                    str(args.project_root / "scripts/preprocess_dr_v2_nuscenes_single.py"),
                    "--data-root",
                    str(raw_root),
                    "--target-dir",
                    str(args.processed_base),
                    "--scene-index",
                    str(args.scene_index),
                ],
                cwd=args.upstream_root,
                environment=common_env,
                validate=lambda: (
                    processed_ready(validate_processed(scene_root)),
                    validate_processed(scene_root),
                ),
                timeout_seconds=3600,
            )

        counts = validate_processed(scene_root)
        if counts["sky_masks"] == EXPECTED_TIMESTEPS * len(CAMERAS):
            atomic_json(
                args.run_dir / "stages" / "sky_masks.json",
                {
                    "stage": "sky_masks",
                    "status": "done",
                    "reuse": True,
                    "scene_root": str(scene_root),
                    "counts": counts,
                },
            )
        else:
            run_stage(
                run_dir=args.run_dir,
                stage="sky_masks",
                command=[
                    "/root/autodl-tmp/envs/drivestudio/bin/python",
                    str(args.project_root / "scripts/build_dr_v2_sky_masks.py"),
                    "--scene-root",
                    str(scene_root),
                    "--revision",
                    "2c6f153e4c23c229e2fa2b188eb250607e030cd8",
                    "--manifest",
                    str(args.run_dir / "environment/sky_mask_model_and_files.json"),
                ],
                cwd=args.project_root,
                environment=common_env,
                validate=lambda: (
                    validate_processed(scene_root)["sky_masks"]
                    == EXPECTED_TIMESTEPS * len(CAMERAS),
                    validate_processed(scene_root),
                ),
                gpu_idle_required=True,
                timeout_seconds=3600,
            )

        run_name = f"{args.scene_name.replace('-', '')}_m5_heldout_s0"
        log_dir = args.run_dir / "work_dirs" / "m5_stress" / run_name
        checkpoint = log_dir / "checkpoint_final.pth"
        train_command = [
            "/root/autodl-tmp/envs/drivestudio/bin/python",
            "tools/train.py",
            "--config_file",
            "configs/streetgs.yaml",
            "--output_root",
            str(args.run_dir / "work_dirs"),
            "--project",
            "m5_stress",
            "--run_name",
            run_name,
            "dataset=nuscenes/3cams",
            f"data.data_root={processed_root}",
            f"data.scene_idx={args.scene_index}",
            "data.start_timestep=0",
            "data.end_timestep=-1",
            "data.pixel_source.load_smpl=false",
            f"data.pixel_source.test_image_stride={args.test_image_stride}",
            "trainer.optim.num_iters=30000",
            "logging.saveckpt_freq=30000",
            "render.render_full=false",
            "render.render_test=false",
            "render.render_novel=null",
        ]
        train_stage = run_stage(
            run_dir=args.run_dir,
            stage="train_heldout_30000",
            command=train_command,
            cwd=args.upstream_root,
            environment=common_env,
            validate=lambda: (
                checkpoint.is_file() and checkpoint.stat().st_size > 0,
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size
                    if checkpoint.is_file()
                    else 0,
                },
            ),
            gpu_idle_required=True,
            timeout_seconds=7200,
        )
        train_stage["checkpoint"] = str(checkpoint)
        train_stage["checkpoint_bytes"] = checkpoint.stat().st_size
        train_stage["checkpoint_sha256"] = sha256_file(checkpoint)
        atomic_json(args.run_dir / "stages/train_heldout_30000.json", train_stage)

        registry = args.run_dir / "artifacts" / "actor_registry.json"
        registry_log = args.run_dir / "logs" / "registry.log"
        with registry_log.open("xb") as log:
            result = subprocess.run(
                [
                    "/root/autodl-tmp/envs/drivestudio/bin/python",
                    str(args.project_root / "scripts/build_dr_v2_drivestudio_registry.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--scene-name",
                    args.scene_name,
                    "--selected-token",
                    args.high_token,
                    "--output",
                    str(registry),
                ],
                cwd=args.project_root,
                env=common_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        registry_payload = json.loads(registry.read_text(encoding="utf-8"))
        by_token = {
            row["instance_token"]: row for row in registry_payload.get("actors", [])
        }
        selected = {
            role: by_token.get(token)
            for role, token in (
                ("high-support", args.high_token),
                ("boundary-support", args.boundary_token),
            )
        }
        if result.returncode != 0 or selected["high-support"] is None:
            raise RuntimeError("actor registry 构建失败或 high-support token 缺失")
        atomic_json(
            args.run_dir / "stages" / "registry.json",
            {
                "stage": "registry",
                "status": "done",
                "return_code": result.returncode,
                "registry": str(registry),
                "registry_sha256": sha256_file(registry),
                "selected": selected,
            },
        )
        summary = {
            "status": "done",
            "scene_name": args.scene_name,
            "scene_index": args.scene_index,
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "registry": str(registry),
            "registry_sha256": sha256_file(registry),
            "selected_actors": selected,
            "test_image_stride": args.test_image_stride,
            "truth_tier_a_images": len(range(args.test_image_stride, EXPECTED_TIMESTEPS, args.test_image_stride))
            * len(CAMERAS),
        }
        atomic_json(args.run_dir / "summary.json", summary)
        atomic_json(
            args.run_dir / "terminal.json",
            {"status": "done", "updated_at": now(), "failure": None},
        )
        print(json.dumps(summary, sort_keys=True))
    except BaseException as error:
        atomic_json(
            args.run_dir / "terminal.json",
            {
                "status": "blocked",
                "updated_at": now(),
                "failure": {
                    "code": "M5_SCENE_TRAINING_FAILED",
                    "detail": f"{type(error).__name__}: {error}",
                },
            },
        )
        raise


if __name__ == "__main__":
    main()
