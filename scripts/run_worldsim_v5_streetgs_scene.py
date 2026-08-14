#!/usr/bin/env python3
"""在冻结 fresh-development 输入上训练单场景 V5 StreetGS base。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from scripts import run_worldsim_v4_streetgs_scene as legacy


TASK_ID = "WS-V5-M1-STRUCTURED-OWNERSHIP-01"
SCHEMA_VERSION = "worldsim_v5_streetgs_training_v1"
COHORT_SCHEMA = "worldsim_v5_nuscenes_fresh_cohort_v1"
COHORT_SHA256 = "553373159023218b44615be27aeeb5533a6c585be276e06425235fe09b6b48b1"
SCENE_CONTRACT = {
    "scene-0471": 382,
    "scene-1087": 827,
    "scene-0379": 296,
    "scene-0998": 756,
    "scene-0359": 276,
    "scene-0875": 663,
    "scene-0535": 425,
    "scene-0436": 350,
}
FRAME_CONTRACT = {
    "scene-0471": 196,
    "scene-1087": 196,
    "scene-0379": 191,
    "scene-0998": 196,
    "scene-0359": 196,
    "scene-0875": 196,
    "scene-0535": 201,
    "scene-0436": 196,
}
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v5/m1_development_reconstruction_v1.yaml",
    "configs/worldsim_v5/m1_structured_ownership_v1.yaml",
    "configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml",
    "scripts/run_worldsim_v5_streetgs_scene.py",
    "scripts/preprocess_worldsim_v5_m1_development.py",
    "compatibility/DriveStudio-2026-08-13-m3-test-timeline.patch",
    "tests/test_run_worldsim_v5_streetgs_scene.py",
)
ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None


class V5StreetGSTrainingError(RuntimeError):
    """V5 StreetGS base 合同、输入或资源门失败。"""


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V5StreetGSTrainingError(f"配置根节点必须为 mapping：{path}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    legacy._write_json(path, payload)


def _git(path: Path, *args: str) -> str:
    try:
        return legacy._git(path, *args)
    except Exception as error:
        raise V5StreetGSTrainingError(str(error)) from error


def sha256_file(path: Path) -> str:
    return legacy.sha256_file(path)


def expected_scene_frames(config: Mapping[str, Any], scene: str) -> int:
    try:
        value = int(config["data"]["expected_frames_by_scene"][scene])
    except (KeyError, TypeError, ValueError) as error:
        raise V5StreetGSTrainingError(f"scene frame contract 缺失：{scene}") from error
    if value <= 0:
        raise V5StreetGSTrainingError(f"scene frame contract 非法：{scene}: {value}")
    return value


def _verified_preprocess_artifact(
    config: Mapping[str, Any], scene: str, *, verify_payload: bool
) -> dict[str, Any]:
    row = config["preprocess_binding"]["scene_artifacts"][scene]
    path = Path(str(row["path"]))
    if not path.is_file() or sha256_file(path) != row.get("file_sha256"):
        raise V5StreetGSTrainingError(f"preprocess scene artifact 漂移：{scene}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "worldsim_v5_m1_processed_scene_v1"
        or payload.get("task_id") != TASK_ID
        or payload.get("status") != "done"
        or payload.get("scene_name") != scene
        or int(payload.get("scene_index", -1)) != SCENE_CONTRACT[scene]
        or payload.get("quality_read") is not False
        or payload.get("training_started") is not False
        or payload.get("model_inference_started") is not False
        or payload.get("inventory_sha256") != row.get("inventory_sha256")
    ):
        raise V5StreetGSTrainingError(f"preprocess scene artifact 合同漂移：{scene}")
    expected_output = (
        Path(config["data"]["processed_root"])
        / legacy.scene_directory_name(SCENE_CONTRACT[scene])
    ).resolve()
    if Path(str(payload.get("output"))).resolve() != expected_output:
        raise V5StreetGSTrainingError(f"preprocess output identity 漂移：{scene}")
    inventory = payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise V5StreetGSTrainingError(f"preprocess inventory 为空：{scene}")
    if verify_payload:
        expected_paths = []
        total_bytes = 0
        for item in inventory:
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise V5StreetGSTrainingError(f"preprocess inventory 路径非法：{scene}")
            target = expected_output / relative
            if (
                not target.is_file()
                or target.stat().st_size != int(item["bytes"])
                or sha256_file(target) != item["sha256"]
            ):
                raise V5StreetGSTrainingError(
                    f"processed payload 漂移：{scene}/{relative.as_posix()}"
                )
            expected_paths.append(relative.as_posix())
            total_bytes += int(item["bytes"])
        observed_paths = sorted(
            path.relative_to(expected_output).as_posix()
            for path in expected_output.rglob("*")
            if path.is_file()
        )
        if sorted(expected_paths) != observed_paths:
            raise V5StreetGSTrainingError(f"processed payload 文件集合漂移：{scene}")
    else:
        total_bytes = sum(int(item["bytes"]) for item in inventory)
    return {
        "artifact": str(path),
        "artifact_sha256": row["file_sha256"],
        "inventory_sha256": row["inventory_sha256"],
        "file_count": len(inventory),
        "total_bytes": total_bytes,
        "payload_rehashed": verify_payload,
    }


def validate_config(
    config: Mapping[str, Any], project_root: Path, *, verify_payload: bool = False
) -> dict[str, Any]:
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("task_id") != TASK_ID
        or config.get("status") != "running"
        or config.get("phase") != "development_base_reconstruction"
    ):
        raise V5StreetGSTrainingError("V5 StreetGS config schema/task/status/phase 漂移")
    implementation = config.get("implementation", {})
    patch = project_root / str(implementation.get("compatibility_patch"))
    if not patch.is_file() or sha256_file(patch) != implementation.get(
        "compatibility_patch_sha256"
    ):
        raise V5StreetGSTrainingError("DriveStudio compatibility patch 漂移")
    if config.get("scenes") != SCENE_CONTRACT:
        raise V5StreetGSTrainingError("fresh development scenes 漂移")
    observed_frames = {
        scene: expected_scene_frames(config, scene) for scene in SCENE_CONTRACT
    }
    if observed_frames != FRAME_CONTRACT:
        raise V5StreetGSTrainingError("fresh development frame contract 漂移")
    data = config.get("data", {})
    if (
        data.get("processed_root")
        != "/root/autodl-tmp/data/worldsim_v5/drivestudio_processed_10Hz/trainval"
        or data.get("dataset_config") != "nuscenes/3cams"
        or int(data.get("expected_cameras", -1)) != 6
        or int(data.get("test_image_stride", -1)) != 0
        or data.get("frame_partition")
        != {
            "modulus": 5,
            "development_remainder": 2,
            "heldout_remainder": 4,
            "train_remainders": [0, 1, 3],
        }
    ):
        raise V5StreetGSTrainingError("V5 processed data/partition 合同漂移")
    if config.get("training", {}).get("seed") != 0 or config.get(
        "training", {}
    ).get("modes") != {"profile100": 100, "formal": 30000}:
        raise V5StreetGSTrainingError("seed/iteration 合同漂移")
    cohort = config.get("fresh_cohort_binding", {})
    cohort_path = project_root / str(cohort.get("config"))
    if (
        cohort.get("cohort_sha256") != COHORT_SHA256
        or not cohort_path.is_file()
        or sha256_file(cohort_path) != cohort.get("config_sha256")
    ):
        raise V5StreetGSTrainingError("fresh cohort binding 漂移")
    cohort_payload = _load_yaml(cohort_path)
    if (
        cohort_payload.get("schema_version") != COHORT_SCHEMA
        or cohort_payload.get("status") != "done"
        or cohort_payload.get("freeze", {}).get("cohort_sha256") != COHORT_SHA256
        or cohort_payload.get("freeze", {}).get("scene_roles", {}).get("development")
        != list(SCENE_CONTRACT)
    ):
        raise V5StreetGSTrainingError("fresh cohort payload 漂移")
    preprocessing = config.get("preprocess_binding", {})
    preprocess_run = Path(str(preprocessing.get("run")))
    summary_path = preprocess_run / "summary.json"
    if (
        not summary_path.is_file()
        or sha256_file(summary_path) != preprocessing.get("summary_sha256")
    ):
        raise V5StreetGSTrainingError("preprocess summary 漂移")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != "worldsim_v5_m1_preprocess_summary_v1"
        or summary.get("task_id") != TASK_ID
        or summary.get("stage") != "development_preprocess"
        or summary.get("status") != "done"
        or summary.get("processed_root") != data.get("processed_root")
        or summary.get("quality_read") is not False
        or summary.get("training_started") is not False
        or summary.get("model_inference_started") is not False
        or int(summary.get("scene_count", -1)) != len(SCENE_CONTRACT)
        or summary.get("project_git", {}).get("head")
        != preprocessing.get("source_commit")
        or summary.get("project_git", {}).get("dirty") is not False
    ):
        raise V5StreetGSTrainingError("preprocess summary 合同漂移")
    scene_artifacts = preprocessing.get("scene_artifacts")
    if not isinstance(scene_artifacts, Mapping) or set(scene_artifacts) != set(
        SCENE_CONTRACT
    ):
        raise V5StreetGSTrainingError("preprocess scene artifact 集合漂移")
    bindings = {
        scene: _verified_preprocess_artifact(
            config, scene, verify_payload=verify_payload
        )
        for scene in SCENE_CONTRACT
    }
    restrictions = config.get("restrictions", {})
    if (
        restrictions.get("validation_quality_read") is not False
        or restrictions.get("test_quality_read") is not False
        or restrictions.get("post_train_render") is not False
    ):
        raise V5StreetGSTrainingError("quality/render restriction 漂移")
    return {"scene_count": len(SCENE_CONTRACT), "preprocess_bindings": bindings}


def build_train_command(
    config: Mapping[str, Any], scene: str, mode: str, run_dir: Path
) -> tuple[list[str], Path, int]:
    iterations = int(config["training"]["modes"][mode])
    scene_index = int(config["scenes"][scene])
    project = "worldsim_v5_streetgs"
    run_name = f"{scene.replace('-', '')}_{mode}_s0"
    output_root = run_dir / "work_dirs"
    checkpoint = output_root / project / run_name / "checkpoint_final.pth"
    partition = config["data"]["frame_partition"]
    excluded = [
        int(partition["development_remainder"]),
        int(partition["heldout_remainder"]),
    ]
    command = [
        str(Path(config["implementation"]["environment"]) / "bin/python"),
        "tools/train.py",
        "--config_file",
        str(config["implementation"]["config_file"]),
        "--output_root",
        str(output_root),
        "--project",
        project,
        "--run_name",
        run_name,
        f"dataset={config['data']['dataset_config']}",
        f"data.data_root={config['data']['processed_root']}",
        f"data.scene_idx={scene_index}",
        f"data.start_timestep={config['data']['start_timestep']}",
        f"data.end_timestep={config['data']['end_timestep']}",
        f"data.pixel_source.load_smpl={str(bool(config['data']['load_smpl'])).lower()}",
        f"data.pixel_source.test_image_stride={int(config['data']['test_image_stride'])}",
        f"+data.pixel_source.partition_modulus={int(partition['modulus'])}",
        "+data.pixel_source.excluded_remainders="
        + json.dumps(excluded, separators=(",", ":")),
        f"trainer.optim.num_iters={iterations}",
        f"logging.saveckpt_freq={iterations}",
        "logging.vis_freq=-1",
        "render.render_full=false",
        "render.render_test=false",
        "render.render_novel=null",
    ]
    return command, checkpoint, iterations


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    legacy.append_jsonl(path, payload)


def run(
    config_path: Path,
    run_dir: Path,
    project_root: Path,
    scene: str,
    mode: str,
) -> dict[str, Any]:
    global ACTIVE_PROCESS
    config_path, run_dir, project_root = (
        config_path.resolve(),
        run_dir.resolve(),
        project_root.resolve(),
    )
    if run_dir.exists():
        raise V5StreetGSTrainingError(f"run 目录已存在，禁止复用：{run_dir}")
    config = _load_yaml(config_path)
    validated = validate_config(config, project_root, verify_payload=False)
    if scene not in SCENE_CONTRACT or mode not in config["training"]["modes"]:
        raise V5StreetGSTrainingError("scene/mode 未冻结")
    upstream = Path(config["implementation"]["upstream_root"])
    if (
        _git(upstream, "rev-parse", "HEAD")
        != config["implementation"]["upstream_commit"]
        or _git(upstream, "status", "--short")
        != config["implementation"]["expected_git_status"]
    ):
        raise V5StreetGSTrainingError("patched DriveStudio HEAD/status 漂移")
    patch = project_root / config["implementation"]["compatibility_patch"]
    reverse = subprocess.run(
        ["git", "-C", str(upstream), "apply", "--reverse", "--check", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    if reverse.returncode != 0:
        raise V5StreetGSTrainingError("DriveStudio patch reverse-check 失败")
    scene_index = SCENE_CONTRACT[scene]
    processed = Path(config["data"]["processed_root"]) / legacy.scene_directory_name(
        scene_index
    )
    data_validation = legacy.validate_processed_scene(
        processed, expected_scene_frames(config, scene), 6
    )
    preprocess_binding = _verified_preprocess_artifact(
        config, scene, verify_payload=True
    )
    for name in ("artifacts", "logs", "source_snapshot", "stages", "work_dirs"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    stage_name = f"train_{mode}"
    pre = legacy.resource_sample(stage_name, "preflight")
    _append_jsonl(run_dir / "resource.jsonl", pre)
    legacy.preflight_resource(pre, config)
    command, checkpoint, iterations = build_train_command(
        config, scene, mode, run_dir
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{project_root}:{upstream}",
            "WANDB_MODE": "disabled",
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
            "TMPDIR": "/root/autodl-tmp/tmp",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
    )
    log_path = run_dir / "logs" / f"{stage_name}.log"
    baseline_events = pre["memory_events"]
    peak_gpu = int(pre["gpu"]["memory_used_mib"])
    peak_memory = int(pre["memory_current_bytes"] or 0)
    over_memory = 0
    stop_reason = None
    started = time.monotonic()
    with log_path.open("xb") as log:
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=upstream,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        interval = float(config["resources"]["sample_interval_seconds"])
        timeout = float(config["resources"]["timeout_seconds"][mode])
        while ACTIVE_PROCESS.poll() is None:
            time.sleep(interval)
            sample = legacy.resource_sample(stage_name, "running")
            _append_jsonl(run_dir / "resource.jsonl", sample)
            peak_gpu = max(peak_gpu, int(sample["gpu"]["memory_used_mib"]))
            peak_memory = max(
                peak_memory, int(sample["memory_current_bytes"] or 0)
            )
            maximum, current = (
                sample["memory_max_bytes"],
                sample["memory_current_bytes"],
            )
            over_memory = (
                over_memory + 1
                if maximum
                and current
                and current / maximum
                >= float(config["resources"]["stop_cgroup_ratio"])
                else 0
            )
            events = sample["memory_events"]
            if over_memory >= int(
                config["resources"]["stop_cgroup_consecutive_samples"]
            ):
                stop_reason = "cgroup_memory_ratio"
            elif events.get("oom", 0) > baseline_events.get(
                "oom", 0
            ) or events.get("oom_kill", 0) > baseline_events.get("oom_kill", 0):
                stop_reason = "cgroup_oom_event"
            elif int(sample["disk_free_bytes"]) < int(
                config["resources"]["stop_disk_free_gib"]
            ) * 2**30:
                stop_reason = "disk_free"
            elif time.monotonic() - started > timeout:
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
    _append_jsonl(
        run_dir / "resource.jsonl",
        legacy.resource_sample(stage_name, "completed"),
    )
    invalid_config = "invalid configuration argument" in log_path.read_text(
        encoding="utf-8", errors="replace"
    )
    if return_code != 0 or stop_reason or invalid_config:
        raise V5StreetGSTrainingError(
            f"training blocked: rc={return_code} stop={stop_reason} "
            f"invalid_config={invalid_config}"
        )
    checkpoint_row = legacy.checkpoint_contract(checkpoint, iterations)
    stage = {
        "stage": stage_name,
        "status": "done",
        "scene": scene,
        "scene_index": scene_index,
        "mode": mode,
        "iterations": iterations,
        "duration_seconds": elapsed,
        "command": command,
        "checkpoint": checkpoint_row,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_cgroup_memory_bytes": peak_memory,
        "stop_reason": None,
        "invalid_configuration_observed": False,
    }
    _write_json(run_dir / "stages" / f"{stage_name}.json", stage)
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    fingerprint = {
        "config_sha256": sha256_file(config_path),
        "data_validation": data_validation,
        "preprocess_binding": preprocess_binding,
        "upstream": {
            "head": _git(upstream, "rev-parse", "HEAD"),
            "status": _git(upstream, "status", "--short"),
            "diff_sha256": hashlib.sha256(
                _git(upstream, "diff", "--binary").encode()
            ).hexdigest(),
        },
        "checkpoint_sha256": checkpoint_row["sha256"],
        "source_snapshots": snapshots,
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        run_dir / "events.jsonl",
        {
            "at_utc": now,
            "event": "v5_streetgs_training_complete",
            "scene": scene,
            "mode": mode,
            "status": "done",
        },
    )
    summary = {
        "schema_version": "worldsim_v5_streetgs_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": "development_base_reconstruction",
        "scene": scene,
        "scene_index": scene_index,
        "mode": mode,
        "iterations": iterations,
        "finished_at_utc": now,
        "duration_seconds": elapsed,
        "checkpoint": checkpoint_row,
        "resources": {
            "peak_gpu_memory_mib": peak_gpu,
            "peak_cgroup_memory_bytes": peak_memory,
        },
        "preprocess_inventory_sha256": preprocess_binding["inventory_sha256"],
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "project_git": {
            "head": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "dirty": bool(_git(project_root, "status", "--porcelain")),
        },
        "training_started": True,
        "model_inference_started": False,
        "validation_quality_read": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "scene": scene,
            "mode": mode,
            "finished_at_utc": now,
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and not path.is_relative_to(run_dir / "work_dirs")
    }
    artifacts["work_dirs_checkpoint"] = checkpoint_row
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_streetgs_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": scene,
            "mode": mode,
            "artifacts": artifacts,
            "validation_quality_read": False,
            "test_quality_read": False,
        },
    )
    gc.collect()
    return summary


def record_blocked(
    config_path: Path,
    run_dir: Path,
    project_root: Path,
    scene: str,
    mode: str,
    error: BaseException,
) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "at_utc": now,
        "event": "v5_streetgs_training_blocked",
        "scene": scene,
        "mode": mode,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    _append_jsonl(run_dir / "events.jsonl", event)
    fingerprint = {
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "project_head": _git(project_root, "rev-parse", "HEAD"),
        "error": event,
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    summary = {
        "schema_version": "worldsim_v5_streetgs_summary_v1",
        "task_id": TASK_ID,
        "status": "blocked",
        "scene": scene,
        "mode": mode,
        "finished_at_utc": now,
        "reason": "streetgs_training_failed",
        "error": event,
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "training_started": (run_dir / "resource.jsonl").exists(),
        "model_inference_started": False,
        "validation_quality_read": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "blocked",
            "scene": scene,
            "mode": mode,
            "finished_at_utc": now,
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and not path.is_relative_to(run_dir / "work_dirs")
    }
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_streetgs_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "blocked",
            "scene": scene,
            "mode": mode,
            "artifacts": artifacts,
            "validation_quality_read": False,
            "test_quality_read": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    parser.add_argument("--scene", required=True, choices=list(SCENE_CONTRACT))
    parser.add_argument("--mode", required=True, choices=["profile100", "formal"])
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    try:
        result = run(
            args.config, args.run_dir, args.project_root, args.scene, args.mode
        )
    except BaseException as error:
        global ACTIVE_PROCESS
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
            ACTIVE_PROCESS.wait(timeout=30)
            ACTIVE_PROCESS = None
        if not existed_before:
            record_blocked(
                args.config.resolve(),
                args.run_dir.resolve(),
                args.project_root.resolve(),
                args.scene,
                args.mode,
                error,
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
