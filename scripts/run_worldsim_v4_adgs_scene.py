#!/usr/bin/env python3
"""资源守卫下预处理或训练单个 WorldSim V4 matched AD-GS scene。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v4/adgs_training_v1.yaml",
    "configs/worldsim_v4/adgs_nuscenes_v4.py",
    "configs/worldsim_v4/adgs_environment_v1.yaml",
    "scripts/run_worldsim_v4_adgs_scene.py",
    "scripts/prepare_worldsim_v4_adgs.py",
    "scripts/smoke_worldsim_v4_adgs_env.py",
    "compatibility/AD-GS-2026-07-27.patch",
    "tests/test_run_worldsim_v4_adgs_scene.py",
    "tests/test_prepare_worldsim_v4_adgs.py",
)
ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None


class ADGSRunError(RuntimeError):
    """AD-GS matched scene 合同或资源门失败。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, path)


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(payload))


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ADGSRunError("config 必须为 mapping")
    return value


def git(root: Path, *args: str) -> str:
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise ADGSRunError(process.stderr.strip())
    return process.stdout.rstrip()


def validate_config(config: Mapping[str, Any], project_root: Path) -> None:
    if config.get("schema_version") != "worldsim_v4_adgs_training_v1" or config.get("task_id") != TASK_ID or config.get("status") != "running":
        raise ADGSRunError("AD-GS config schema/task/status 漂移")
    data = config.get("data", {})
    if data.get("materialized_partitions") != ["train"] or data.get("frame_partition") != {
        "modulus": 5,
        "train_remainders": [0, 1, 3],
        "development_remainder": 2,
        "heldout_remainder": 4,
    } or int(data.get("expected_train_images", -1)) != 354:
        raise ADGSRunError("AD-GS train-only partition 合同漂移")
    training = config.get("training", {})
    if training.get("seed") != 0 or training.get("modes") != {"profile100": 100, "formal": 60000} or training.get("disable_test_evaluation") is not True:
        raise ADGSRunError("AD-GS seed/iteration/no-eval 合同漂移")
    expected_scenes = {"scene-0230": 179, "scene-0242": 191, "scene-0255": 204, "scene-0048": 45, "scene-0994": 752, "scene-0139": 110}
    if config.get("scenes") != expected_scenes:
        raise ADGSRunError("AD-GS 六场景合同漂移")
    implementation = config["implementation"]
    patch = project_root / implementation["compatibility_patch"]
    if not patch.is_file() or sha256_file(patch) != implementation["compatibility_patch_sha256"]:
        raise ADGSRunError("AD-GS compatibility patch 漂移")


def audit_sources(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    implementation = config["implementation"]
    root = Path(implementation["root"])
    if git(root, "rev-parse", "HEAD") != implementation["commit"]:
        raise ADGSRunError("AD-GS commit 漂移")
    status_lines = [line for line in git(root, "status", "--short").splitlines() if line]
    changed = sorted(line[3:] for line in status_lines)
    if changed != sorted(implementation["expected_modified_files"]):
        raise ADGSRunError(f"AD-GS modified file set 漂移：{changed}")
    patch = project_root / implementation["compatibility_patch"]
    reverse = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", "--reverse", "--unidiff-zero", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    if reverse.returncode != 0:
        raise ADGSRunError("AD-GS compatibility patch reverse-check 失败")
    dependencies = {}
    for name in ("depth_anything", "cotracker"):
        row = config["dependencies"][name]
        dependency_root = Path(row["root"])
        checkpoint = Path(row["checkpoint"])
        if git(dependency_root, "rev-parse", "HEAD") != row["commit"] or git(dependency_root, "status", "--porcelain"):
            raise ADGSRunError(f"{name} source commit/status 漂移")
        actual = {"bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)} if checkpoint.is_file() else None
        if actual != {"bytes": row["bytes"], "sha256": row["sha256"]}:
            raise ADGSRunError(f"{name} checkpoint 漂移：{actual}")
        dependencies[name] = {"root": str(dependency_root), "commit": row["commit"], "checkpoint": {"path": str(checkpoint), **actual}}
    environment = Path(implementation["environment"])
    if not (environment / "bin/python").is_file():
        raise ADGSRunError("AD-GS environment 缺失")
    return {"implementation": {"root": str(root), "commit": implementation["commit"], "modified_files": changed, "patch_sha256": sha256_file(patch)}, "dependencies": dependencies, "environment": str(environment)}


def scene_source(config: Mapping[str, Any], scene: str) -> Path:
    return Path(config["data"]["source_root"]) / f"{int(config['scenes'][scene]):03d}"


def scene_destination(config: Mapping[str, Any], scene: str) -> Path:
    return Path(config["data"]["processed_root"]) / scene


def build_preprocess_commands(config: Mapping[str, Any], project_root: Path, scene: str) -> list[tuple[str, list[str], Path]]:
    implementation = config["implementation"]
    adgs = Path(implementation["root"])
    python = str(Path(implementation["environment"]) / "bin/python")
    destination = scene_destination(config, scene)
    depth = config["dependencies"]["depth_anything"]
    preprocess = config["preprocess"]
    return [
        (
            "adapter",
            [python, str(project_root / "scripts/prepare_worldsim_v4_adgs.py"), "--source", str(scene_source(config, scene)), "--destination", str(destination), "--partitions", "train"],
            project_root,
        ),
        (
            "depth",
            [python, str(adgs / "scripts/run-dpt.py"), "--img-path", str(destination / "image"), "--outdir", str(destination / "depth"), "--encoder", str(preprocess["depth_encoder"]), "--checkpoint", str(depth["checkpoint"]), "--pred-only", "--grayscale"],
            adgs,
        ),
        ("segment_points", [python, str(adgs / "scripts/segment_pcd.py"), str(destination)], adgs),
        (
            "flow",
            [python, str(adgs / "scripts/flow.py"), str(destination), "--device", "cuda:0", "--downsample", "1", "--step", str(preprocess["flow_step"]), "--seed", str(preprocess["seed"])],
            adgs,
        ),
    ]


def build_train_command(config: Mapping[str, Any], project_root: Path, scene: str, mode: str, run_dir: Path) -> tuple[list[str], Path, int]:
    iterations = int(config["training"]["modes"][mode])
    implementation = config["implementation"]
    python = str(Path(implementation["environment"]) / "bin/python")
    model_root = run_dir / "model"
    command = [
        python,
        "train.py",
        "--config",
        str(project_root / implementation["model_config"]),
        "--source_path",
        str(scene_destination(config, scene)),
        "--model_path",
        str(model_root),
        "--data_device",
        str(config["training"]["data_device"]),
        "--iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--disable_test_evaluation",
        "--quiet",
    ]
    return command, model_root, iterations


def memory_value(path: str) -> int | None:
    raw = Path(path).read_text(encoding="utf-8").strip()
    return None if raw == "max" else int(raw)


def memory_events() -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines())}


def resource_sample(stage: str, event: str) -> dict[str, Any]:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    fields = [value.strip() for value in process.stdout.strip().split(",")]
    return {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "event": event,
        "memory_current_bytes": memory_value("/sys/fs/cgroup/memory.current"),
        "memory_max_bytes": memory_value("/sys/fs/cgroup/memory.max"),
        "memory_events": memory_events(),
        "disk_free_bytes": shutil.disk_usage("/root/autodl-tmp").free,
        "gpu": {"name": fields[0], "driver": fields[1], "memory_total_mib": int(fields[2]), "memory_used_mib": int(fields[3]), "utilization_percent": int(fields[4])},
    }


def preflight_resource(sample: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    limits = config["resources"]
    gpu = sample["gpu"]
    if gpu["memory_total_mib"] < int(limits["minimum_gpu_memory_mib"]) or gpu["memory_used_mib"] > int(limits["maximum_gpu_used_at_start_mib"]):
        raise ADGSRunError(f"GPU resource gate 未通过：{gpu}")
    maximum = sample["memory_max_bytes"]
    if maximum is None or maximum < int(limits["minimum_cgroup_memory_gib"]) * 2**30:
        raise ADGSRunError("cgroup memory gate 未通过")
    if sample["disk_free_bytes"] < int(limits["minimum_disk_free_at_start_gib"]) * 2**30:
        raise ADGSRunError("disk resource gate 未通过")


def environment(config: Mapping[str, Any], project_root: Path) -> dict[str, str]:
    value = os.environ.copy()
    depth_root = config["dependencies"]["depth_anything"]["root"]
    cotracker_root = config["dependencies"]["cotracker"]["root"]
    value.update(
        {
            "PYTHONPATH": f"{project_root}:{depth_root}:{cotracker_root}",
            "COTRACKER_REPO": cotracker_root,
            "COTRACKER_CHECKPOINT": config["dependencies"]["cotracker"]["checkpoint"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "TMPDIR": "/root/autodl-tmp/tmp",
        }
    )
    return value


def run_stage(run_dir: Path, name: str, command: Sequence[str], cwd: Path, env: Mapping[str, str], config: Mapping[str, Any], timeout_key: str) -> dict[str, Any]:
    global ACTIVE_PROCESS
    pre = resource_sample(name, "preflight")
    append_jsonl(run_dir / "resource.jsonl", pre)
    preflight_resource(pre, config)
    log_path = run_dir / "logs" / f"{name}.log"
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    baseline_events = pre["memory_events"]
    peak_gpu = pre["gpu"]["memory_used_mib"]
    peak_memory = int(pre["memory_current_bytes"] or 0)
    high_count = 0
    stop_reason = None
    with log_path.open("xb") as log:
        ACTIVE_PROCESS = subprocess.Popen(list(command), cwd=str(cwd), env=dict(env), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while ACTIVE_PROCESS.poll() is None:
            time.sleep(float(config["resources"]["sample_interval_seconds"]))
            sample = resource_sample(name, "running")
            append_jsonl(run_dir / "resource.jsonl", sample)
            peak_gpu = max(peak_gpu, sample["gpu"]["memory_used_mib"])
            peak_memory = max(peak_memory, int(sample["memory_current_bytes"] or 0))
            maximum, current = sample["memory_max_bytes"], sample["memory_current_bytes"]
            high_count = high_count + 1 if maximum and current and current / maximum >= float(config["resources"]["stop_cgroup_ratio"]) else 0
            events = sample["memory_events"]
            if high_count >= int(config["resources"]["stop_cgroup_consecutive_samples"]):
                stop_reason = "cgroup_memory_ratio"
            elif events.get("oom", 0) > baseline_events.get("oom", 0) or events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0):
                stop_reason = "cgroup_oom_event"
            elif sample["disk_free_bytes"] < int(config["resources"]["stop_disk_free_gib"]) * 2**30:
                stop_reason = "disk_free"
            elif time.monotonic() - started > float(config["resources"]["timeout_seconds"][timeout_key]):
                stop_reason = "timeout"
            if stop_reason:
                os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
                try:
                    ACTIVE_PROCESS.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(ACTIVE_PROCESS.pid, signal.SIGKILL)
                break
        return_code = ACTIVE_PROCESS.wait()
        ACTIVE_PROCESS = None
    elapsed = time.monotonic() - started
    post = resource_sample(name, "completed")
    append_jsonl(run_dir / "resource.jsonl", post)
    result = {"stage": name, "status": "done" if return_code == 0 and stop_reason is None else "blocked", "started_at_utc": started_wall, "finished_at_utc": post["at_utc"], "duration_seconds": elapsed, "command": list(command), "cwd": str(cwd), "return_code": return_code, "stop_reason": stop_reason, "peak_gpu_memory_mib": peak_gpu, "peak_cgroup_memory_bytes": peak_memory}
    write_json(run_dir / "stages" / f"{name}.json", result)
    if result["status"] != "done":
        raise ADGSRunError(f"stage {name} blocked: rc={return_code} stop={stop_reason}")
    return result


def validate_processed(config: Mapping[str, Any], scene: str) -> dict[str, Any]:
    root = scene_destination(config, scene)
    manifest_path = root / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise ADGSRunError("AD-GS adapter manifest 缺失")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("included_partitions") != ["train"] or manifest.get("partition_image_counts") != {"development": 0, "heldout": 0, "train": 354} or manifest.get("image_count") != 354:
        raise ADGSRunError("AD-GS processed partition/count 漂移")
    meta = np.load(root / "meta.npz", allow_pickle=False)
    if meta["is_val_list"].shape != (354,) or bool(meta["is_val_list"].any()):
        raise ADGSRunError("AD-GS train-only meta 含 validation flag")
    counts = {
        "image": len(list((root / "image").glob("*.png"))),
        "semantic": len(list((root / "semantic").glob("*.npy"))),
        "sky": len(list((root / "sky").glob("*.npy"))),
        "depth": len(list((root / "depth").glob("*.npy"))),
        "flow": len(list((root / "flow").glob("*.npz"))),
    }
    if any(counts[name] != 354 for name in ("image", "semantic", "sky", "depth")) or not 0 < counts["flow"] <= 354:
        raise ADGSRunError(f"AD-GS processed file count 漂移：{counts}")
    for required in ("points3d.ply", "partition.json"):
        if not (root / required).is_file() or (root / required).stat().st_size == 0:
            raise ADGSRunError(f"AD-GS processed artifact 缺失：{required}")
    return {"root": str(root), "counts": counts, "adapter_manifest_sha256": sha256_file(manifest_path), "meta_sha256": sha256_file(root / "meta.npz"), "points3d_sha256": sha256_file(root / "points3d.ply"), "partitions_materialized": ["train"], "development_content_read": False, "heldout_content_read": False}


def checkpoint_contract(model_root: Path, iterations: int) -> dict[str, Any]:
    root = model_root / "point_cloud" / f"iteration_{iterations}"
    files = {}
    for name in ("point_cloud.ply", "deform.pth", "env.pth"):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ADGSRunError(f"AD-GS checkpoint file 缺失：{path}")
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"iteration": iterations, "root": str(root), "files": files}


def finalize_manifest(run_dir: Path, status: str, checkpoint: Mapping[str, Any] | None = None) -> None:
    artifacts = {
        str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json" and not path.is_relative_to(run_dir / "model")
    }
    write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_adgs_run_manifest_v1", "task_id": TASK_ID, "status": status, "artifacts": artifacts, "checkpoint": checkpoint, "test_quality_read": False})


def run(config_path: Path, run_dir: Path, project_root: Path, scene: str, mode: str) -> dict[str, Any]:
    config_path, run_dir, project_root = config_path.resolve(), run_dir.resolve(), project_root.resolve()
    if run_dir.exists():
        raise ADGSRunError(f"run 目录已存在，禁止复用：{run_dir}")
    config = load_config(config_path)
    validate_config(config, project_root)
    if scene not in config["scenes"] or mode not in ("preprocess", "profile100", "formal"):
        raise ADGSRunError("scene/mode 未冻结")
    if git(project_root, "status", "--porcelain"):
        raise ADGSRunError("project 必须 clean 后才能启动 AD-GS run")
    source_audit = audit_sources(config, project_root)
    source = scene_source(config, scene)
    if not source.is_dir():
        raise ADGSRunError(f"DriveStudio source scene 缺失：{source}")
    destination = scene_destination(config, scene)
    if mode == "preprocess" and destination.exists():
        raise ADGSRunError(f"preprocess target 已存在，禁止覆盖：{destination}")
    if mode != "preprocess" and not destination.is_dir():
        raise ADGSRunError(f"matched processed scene 缺失：{destination}")
    for folder in ("artifacts", "logs", "source_snapshot", "stages", "model"):
        (run_dir / folder).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    write_json(run_dir / "artifacts/source_audit.json", source_audit)
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source_file = project_root / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        snapshots[relpath] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    env = environment(config, project_root)
    checkpoint = None
    if mode == "preprocess":
        for stage_name, command, cwd in build_preprocess_commands(config, project_root, scene):
            run_stage(run_dir, stage_name, command, cwd, env, config, stage_name)
    else:
        command, model_root, iterations = build_train_command(config, project_root, scene, mode, run_dir)
        run_stage(run_dir, f"train_{mode}", command, Path(config["implementation"]["root"]), env, config, mode)
        checkpoint = checkpoint_contract(model_root, iterations)
        write_json(run_dir / "artifacts/checkpoint.json", checkpoint)
    data_audit = validate_processed(config, scene)
    write_json(run_dir / "artifacts/data_audit.json", data_audit)
    project_git = {"head": git(project_root, "rev-parse", "HEAD"), "branch": git(project_root, "branch", "--show-current"), "dirty": bool(git(project_root, "status", "--porcelain"))}
    fingerprint = {"config_sha256": sha256_file(config_path), "source_audit_sha256": sha256_file(run_dir / "artifacts/source_audit.json"), "data_audit_sha256": sha256_file(run_dir / "artifacts/data_audit.json"), "source_snapshots": snapshots, "checkpoint": checkpoint, "project_git": project_git}
    write_json(run_dir / "fingerprint.json", fingerprint)
    finished = datetime.now(timezone.utc).isoformat()
    append_jsonl(run_dir / "events.jsonl", {"at_utc": finished, "event": "adgs_scene_complete", "status": "done", "scene": scene, "mode": mode})
    summary = {"schema_version": "worldsim_v4_adgs_summary_v1", "task_id": TASK_ID, "status": "done", "scene": scene, "mode": mode, "finished_at_utc": finished, "checkpoint": checkpoint, "data_audit": data_audit, "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "project_git": project_git, "training_started": mode != "preprocess", "model_inference_started": mode == "preprocess", "development_content_read": False, "heldout_content_read": False, "test_quality_read": False}
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "scene": scene, "mode": mode, "finished_at_utc": finished, "summary_sha256": sha256_file(run_dir / "summary.json")})
    finalize_manifest(run_dir, "done", checkpoint)
    return summary


def record_blocked(config_path: Path, run_dir: Path, project_root: Path, scene: str, mode: str, error: BaseException) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    finished = datetime.now(timezone.utc).isoformat()
    event = {"at_utc": finished, "event": "adgs_scene_blocked", "scene": scene, "mode": mode, "error_type": type(error).__name__, "message": str(error)}
    append_jsonl(run_dir / "events.jsonl", event)
    write_json(run_dir / "fingerprint.json", {"config_sha256": sha256_file(config_path) if config_path.is_file() else None, "error": event})
    summary = {"schema_version": "worldsim_v4_adgs_summary_v1", "task_id": TASK_ID, "status": "blocked", "scene": scene, "mode": mode, "finished_at_utc": finished, "reason": "adgs_scene_failed", "error": event, "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "training_started": mode in ("profile100", "formal") and (run_dir / "resource.jsonl").exists(), "development_content_read": False, "heldout_content_read": False, "test_quality_read": False}
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "blocked", "scene": scene, "mode": mode, "finished_at_utc": finished, "summary_sha256": sha256_file(run_dir / "summary.json")})
    finalize_manifest(run_dir, "blocked")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", choices=("preprocess", "profile100", "formal"), required=True)
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    try:
        summary = run(args.config, args.run_dir, args.project_root, args.scene, args.mode)
    except BaseException as error:
        global ACTIVE_PROCESS
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
            ACTIVE_PROCESS.wait(timeout=30)
            ACTIVE_PROCESS = None
        if not existed_before:
            record_blocked(args.config.resolve(), args.run_dir.resolve(), args.project_root.resolve(), args.scene, args.mode, error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
