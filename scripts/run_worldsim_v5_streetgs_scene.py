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
STATIC_SNAPSHOT_RELPATHS = (
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


def load_training_config(
    config_path: Path, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw = _load_yaml(config_path)
    if raw.get("schema_version") != "worldsim_v5_streetgs_training_binding_v1":
        return raw, None
    if (
        raw.get("task_id") != TASK_ID
        or raw.get("status") != "running"
        or raw.get("phase") != "development_base_reconstruction"
    ):
        raise V5StreetGSTrainingError("sky-bound training config 合同漂移")
    base_binding = raw.get("base_config", {})
    base_path = Path(str(base_binding.get("path", "")))
    if not base_path.is_absolute():
        base_path = project_root / base_path
    if not base_path.is_file() or sha256_file(base_path) != base_binding.get("sha256"):
        raise V5StreetGSTrainingError("base reconstruction config binding 漂移")
    base = _load_yaml(base_path)
    if base.get("schema_version") != SCHEMA_VERSION or "sky_mask_binding" in base:
        raise V5StreetGSTrainingError("base reconstruction config schema/immutability 漂移")
    resolved = dict(base)
    resolved["sky_mask_binding"] = raw.get("sky_mask_binding")
    if "profile100_binding" in raw:
        resolved["profile100_binding"] = raw.get("profile100_binding")
    binding = {
        "overlay_path": str(config_path),
        "overlay_sha256": sha256_file(config_path),
        "base_path": str(base_path),
        "base_sha256": base_binding["sha256"],
    }
    return resolved, binding


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
    config: Mapping[str, Any],
    scene: str,
    *,
    verify_payload: bool,
    allowed_extra_paths: set[str] | None = None,
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
        extras = set(observed_paths) - set(expected_paths)
        missing = set(expected_paths) - set(observed_paths)
        if missing or extras != (allowed_extra_paths or set()):
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


def _verified_sky_mask_artifact(
    config: Mapping[str, Any], scene: str, *, verify_payload: bool
) -> dict[str, Any]:
    binding = config["sky_mask_binding"]
    row = binding["scene_runs"][scene]
    run_dir = Path(str(row["run"]))
    summary_path = run_dir / "summary.json"
    run_manifest_path = run_dir / "manifest.json"
    artifact_path = run_dir / "artifacts/sky_mask_manifest.json"
    expected_hashes = {
        summary_path: row["summary_sha256"],
        run_manifest_path: row["run_manifest_sha256"],
        artifact_path: row["artifact_sha256"],
    }
    for path, expected_sha in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise V5StreetGSTrainingError(f"sky-mask run artifact 漂移：{scene}/{path.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_frames = FRAME_CONTRACT[scene]
    expected_masks = expected_frames * len(binding["camera_ids"])
    if (
        summary.get("schema_version") != "worldsim_v5_sky_mask_summary_v1"
        or summary.get("task_id") != TASK_ID
        or summary.get("status") != "done"
        or summary.get("scene") != scene
        or int(summary.get("mask_count", -1)) != expected_masks
        or summary.get("segmentation_inference_started") is not True
        or summary.get("method_inference_started") is not False
        or summary.get("network_accessed") is not False
        or summary.get("test_quality_read") is not False
    ):
        raise V5StreetGSTrainingError(f"sky-mask summary 合同漂移：{scene}")
    if (
        artifact.get("schema_version") != "worldsim_v5_sky_mask_manifest_v1"
        or artifact.get("task_id") != TASK_ID
        or artifact.get("status") != "done"
        or artifact.get("scene") != scene
        or int(artifact.get("scene_index", -1)) != SCENE_CONTRACT[scene]
        or int(artifact.get("expected_timesteps", -1)) != expected_frames
        or int(artifact.get("mask_count", -1)) != expected_masks
        or artifact.get("model", {}).get("revision")
        != binding["model_revision"]
        or artifact.get("network_accessed") is not False
        or artifact.get("test_quality_read") is not False
    ):
        raise V5StreetGSTrainingError(f"sky-mask manifest 合同漂移：{scene}")
    files = artifact.get("files")
    if not isinstance(files, list) or len(files) != expected_masks:
        raise V5StreetGSTrainingError(f"sky-mask file inventory 漂移：{scene}")
    scene_root = Path(config["data"]["processed_root"]) / legacy.scene_directory_name(
        SCENE_CONTRACT[scene]
    )
    sky_root = scene_root / "sky_masks"
    relative_paths = {f"sky_masks/{item['mask']}" for item in files}
    expected_names = {str(item["mask"]) for item in files}
    if verify_payload:
        for item in files:
            path = sky_root / str(item["mask"])
            if (
                not path.is_file()
                or path.stat().st_size != int(item["bytes"])
                or sha256_file(path) != item["sha256"]
            ):
                raise V5StreetGSTrainingError(
                    f"sky-mask payload 漂移：{scene}/{item['mask']}"
                )
        observed_names = {path.name for path in sky_root.glob("*.png")}
        if observed_names != expected_names:
            raise V5StreetGSTrainingError(f"sky-mask 输出集合漂移：{scene}")
    return {
        "run": str(run_dir),
        "summary_sha256": row["summary_sha256"],
        "run_manifest_sha256": row["run_manifest_sha256"],
        "artifact_sha256": row["artifact_sha256"],
        "mask_count": expected_masks,
        "relative_paths": relative_paths,
        "payload_rehashed": verify_payload,
    }


def _verified_profile100_cohort(
    config: Mapping[str, Any], *, verify_payload: bool
) -> dict[str, Any] | None:
    binding = config.get("profile100_binding")
    if binding is None:
        return None
    if (
        not isinstance(binding, Mapping)
        or binding.get("source_commit")
        != "200ece4ebe59031b5546f285d2251482446ab162"
        or int(binding.get("scene_count", -1)) != len(SCENE_CONTRACT)
        or not isinstance(binding.get("scene_runs"), Mapping)
        or set(binding["scene_runs"]) != set(SCENE_CONTRACT)
    ):
        raise V5StreetGSTrainingError("profile100 cohort binding 合同漂移")
    verified: dict[str, Any] = {}
    for scene in SCENE_CONTRACT:
        row = binding["scene_runs"][scene]
        run_dir = Path(str(row.get("run", "")))
        paths = {
            "summary": run_dir / "summary.json",
            "status": run_dir / "status.json",
            "fingerprint": run_dir / "fingerprint.json",
            "manifest": run_dir / "manifest.json",
        }
        expected_hashes = {
            "summary": row.get("summary_sha256"),
            "status": row.get("status_sha256"),
            "fingerprint": row.get("fingerprint_sha256"),
            "manifest": row.get("run_manifest_sha256"),
        }
        for name, path in paths.items():
            if not path.is_file() or sha256_file(path) != expected_hashes[name]:
                raise V5StreetGSTrainingError(
                    f"profile100 run artifact 漂移：{scene}/{name}"
                )
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        checkpoint = Path(str(summary.get("checkpoint", {}).get("path", "")))
        if (
            summary.get("schema_version") != "worldsim_v5_streetgs_summary_v1"
            or summary.get("task_id") != TASK_ID
            or summary.get("status") != "done"
            or summary.get("scene") != scene
            or summary.get("mode") != "profile100"
            or int(summary.get("iterations", -1)) != 100
            or int(summary.get("checkpoint", {}).get("step", -1)) != 100
            or summary.get("checkpoint", {}).get("means_finite") is not True
            or summary.get("checkpoint", {}).get("sha256")
            != row.get("checkpoint_sha256")
            or int(summary.get("checkpoint", {}).get("bytes", -1))
            != int(row.get("checkpoint_bytes", -2))
            or summary.get("project_git")
            != {
                "head": binding["source_commit"],
                "branch": "research/worldsim-v5-structdelta",
                "dirty": False,
            }
            or summary.get("training_started") is not True
            or summary.get("model_inference_started") is not False
            or summary.get("validation_quality_read") is not False
            or summary.get("test_quality_read") is not False
            or status.get("status") != "done"
            or status.get("summary_sha256") != expected_hashes["summary"]
            or manifest.get("schema_version")
            != "worldsim_v5_streetgs_run_manifest_v1"
            or manifest.get("status") != "done"
            or manifest.get("scene") != scene
            or manifest.get("mode") != "profile100"
            or not checkpoint.is_file()
            or not checkpoint.resolve().is_relative_to(run_dir.resolve())
            or checkpoint.stat().st_size != int(row.get("checkpoint_bytes", -1))
        ):
            raise V5StreetGSTrainingError(f"profile100 run 合同漂移：{scene}")
        if verify_payload and sha256_file(checkpoint) != row["checkpoint_sha256"]:
            raise V5StreetGSTrainingError(f"profile100 checkpoint 漂移：{scene}")
        verified[scene] = {
            "run": str(run_dir),
            "summary_sha256": expected_hashes["summary"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_bytes": int(row["checkpoint_bytes"]),
            "payload_rehashed": verify_payload,
        }
    return {
        "source_commit": binding["source_commit"],
        "scene_count": len(verified),
        "scene_runs": verified,
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
    sky_binding = config.get("sky_mask_binding", {})
    if (
        sky_binding.get("model_revision")
        != "2c6f153e4c23c229e2fa2b188eb250607e030cd8"
        or sky_binding.get("camera_ids") != [0, 1, 2]
        or int(sky_binding.get("total_mask_count", -1)) != 4704
        or not isinstance(sky_binding.get("scene_runs"), Mapping)
        or set(sky_binding["scene_runs"]) != set(SCENE_CONTRACT)
    ):
        raise V5StreetGSTrainingError("sky-mask binding 合同漂移")
    sky_bindings = {
        scene: _verified_sky_mask_artifact(
            config, scene, verify_payload=verify_payload
        )
        for scene in SCENE_CONTRACT
    }
    bindings = {
        scene: _verified_preprocess_artifact(
            config,
            scene,
            verify_payload=verify_payload,
            allowed_extra_paths=sky_bindings[scene]["relative_paths"],
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
    profile100_binding = _verified_profile100_cohort(
        config, verify_payload=verify_payload
    )
    return {
        "scene_count": len(SCENE_CONTRACT),
        "preprocess_bindings": bindings,
        "sky_mask_bindings": sky_bindings,
        "profile100_binding": profile100_binding,
    }


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
    config, config_binding = load_training_config(config_path, project_root)
    validated = validate_config(config, project_root, verify_payload=False)
    if scene not in SCENE_CONTRACT or mode not in config["training"]["modes"]:
        raise V5StreetGSTrainingError("scene/mode 未冻结")
    profile100_gate = validated["profile100_binding"]
    if mode == "formal":
        profile100_gate = _verified_profile100_cohort(
            config, verify_payload=True
        )
        if profile100_gate is None:
            raise V5StreetGSTrainingError("formal run 缺少 8-scene profile100 gate")
    elif profile100_gate is not None:
        raise V5StreetGSTrainingError("profile100 run 不得使用 formal-bound config")
    if _git(project_root, "status", "--porcelain"):
        raise V5StreetGSTrainingError("正式 StreetGS run 要求 clean project worktree")
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
    sky_mask_binding = _verified_sky_mask_artifact(
        config, scene, verify_payload=True
    )
    preprocess_binding = _verified_preprocess_artifact(
        config,
        scene,
        verify_payload=True,
        allowed_extra_paths=sky_mask_binding["relative_paths"],
    )
    for name in ("artifacts", "logs", "source_snapshot", "stages", "work_dirs"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
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
        _write_json(
            run_dir / "stages" / "training_process_started.json",
            {
                "process_started": True,
                "scene": scene,
                "mode": mode,
                "pid": ACTIVE_PROCESS.pid,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            },
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
    try:
        config_relpath = config_path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise V5StreetGSTrainingError("训练 config 必须位于 project root") from error
    snapshot_relpaths = [config_relpath, *STATIC_SNAPSHOT_RELPATHS]
    if config_binding is not None:
        base_path = Path(config_binding["base_path"])
        try:
            base_relpath = base_path.relative_to(project_root).as_posix()
        except ValueError as error:
            raise V5StreetGSTrainingError(
                "base reconstruction config 必须位于 project root"
            ) from error
        snapshot_relpaths.append(base_relpath)
    for relpath in dict.fromkeys(snapshot_relpaths):
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
        "config_binding": config_binding,
        "data_validation": data_validation,
        "preprocess_binding": preprocess_binding,
        "sky_mask_binding": {
            key: value
            for key, value in sky_mask_binding.items()
            if key != "relative_paths"
        },
        "profile100_gate": profile100_gate,
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
        "sky_mask_artifact_sha256": sky_mask_binding["artifact_sha256"],
        "profile100_gate_source_commit": (
            profile100_gate["source_commit"]
            if profile100_gate is not None
            else None
        ),
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
        "training_started": (
            run_dir / "stages" / "training_process_started.json"
        ).exists(),
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
