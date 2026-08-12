#!/usr/bin/env python3
"""盘点或执行 WorldSim V4 matched baselines。

B0 的第一阶段只使用 ``audit``：它不会训练或读取 test quality，只把当前可执行性写入
不可变 run。后续训练 runner 会回填 checkpoint 路径，再由同一入口重新审计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
REQUIRED_METHODS = ("streetgs", "v33_frozen", "ad_gs")
V33_REQUIRED_STAGES = (
    "semantic_lift",
    "instance_field",
    "roadpatch",
    "asset_harvester",
    "spatial_delta",
    "semantic_render",
)
V33_ABSTAIN_STAGES = ("roadpatch", "asset_harvester", "semantic_render")
V33_REQUIRED_FILES = ("scene_chain.json", "render_manifest.json", "metrics.json")
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v4/baseline_matrix_v1.yaml",
    "configs/worldsim_v4/metrics_v1.yaml",
    "configs/worldsim_v4/v33_replay_v1.yaml",
    "motion_proj/worldsim_v4/evaluator.py",
    "motion_proj/worldsim_v4/region_masks.py",
    "motion_proj/worldsim_v4/baseline_scene_evaluator.py",
    "motion_proj/worldsim_v4/statistics.py",
    "motion_proj/worldsim_v4/engineering_metrics.py",
    "motion_proj/worldsim_v4/v33_replay.py",
    "motion_proj/worldsim_v4/semantic_split.py",
    "motion_proj/worldsim_v33/evaluation_partition.py",
    "scripts/build_worldsim_v4_v33_actor_registry.py",
    "scripts/materialize_worldsim_v4_v33_replay.py",
    "scripts/materialize_worldsim_v4_v33_semantic_config.py",
    "scripts/materialize_worldsim_v4_v33_instance_config.py",
    "scripts/run_worldsim_v4_v33_semantic_lift.py",
    "scripts/run_worldsim_v4_v33_instance_field.py",
    "scripts/finalize_worldsim_v33_s1.py",
    "scripts/materialize_worldsim_v4_v33_spatial_config.py",
    "scripts/build_worldsim_v33_s4_spatial_delta.py",
    "scripts/evaluate_worldsim_v33_s4_spatial_delta.py",
    "scripts/run_worldsim_v4_v33_spatial_delta.py",
    "scripts/finalize_worldsim_v4_v33_scene_chain.py",
    "scripts/build_worldsim_v4_v33_registration.py",
    "scripts/prepare_worldsim_v32_s1_prompts.py",
    "scripts/validate_worldsim_v32_s1.py",
    "scripts/build_worldsim_v32_sam_masks.py",
    "scripts/lift_worldsim_v32_semantics.py",
    "scripts/finalize_worldsim_v32_s1.py",
    "scripts/prepare_worldsim_v33_s1_eval_prompts.py",
    "scripts/build_worldsim_v33_s1_eval_masks.py",
    "scripts/finalize_worldsim_v33_s1_eval_targets.py",
    "scripts/run_worldsim_v33_s1_instance_field.py",
    "scripts/run_worldsim_v4_baselines.py",
    "tests/test_worldsim_v4_evaluator.py",
    "tests/test_worldsim_v4_region_masks.py",
    "tests/test_worldsim_v4_baseline_scene_evaluator.py",
    "tests/test_worldsim_v4_statistics.py",
    "tests/test_worldsim_v4_engineering_metrics.py",
    "tests/test_materialize_worldsim_v4_v33_replay.py",
    "tests/test_materialize_worldsim_v4_v33_semantic_config.py",
    "tests/test_materialize_worldsim_v4_v33_instance_config.py",
    "tests/test_run_worldsim_v4_v33_semantic_lift.py",
    "tests/test_run_worldsim_v4_v33_instance_field.py",
    "tests/test_worldsim_v4_v33_spatial_delta.py",
    "tests/test_run_worldsim_v4_v33_spatial_delta.py",
    "tests/test_finalize_worldsim_v4_v33_scene_chain.py",
    "tests/test_build_worldsim_v4_v33_registration.py",
    "tests/test_build_worldsim_v4_v33_actor_registry.py",
    "tests/test_prepare_worldsim_v32_s1_prompts.py",
    "tests/test_worldsim_v4_semantic_split.py",
    "tests/test_worldsim_v33_evaluation_partition.py",
    "tests/test_worldsim_v4_v33_replay.py",
)


class BaselineAuditError(RuntimeError):
    """B0 baseline matrix 不满足冻结合同。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BaselineAuditError(f"配置根节点必须为 mapping：{path}")
    return value


def _git(path: Path, *args: str) -> str | None:
    process = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
    return process.stdout.rstrip() if process.returncode == 0 else None


def _path_state(value: Any, *, kind: str = "file") -> dict[str, Any]:
    if not value:
        return {"path": None, "exists": False, "state": "not_registered"}
    path = Path(str(value))
    exists = path.is_file() if kind == "file" else path.is_dir()
    row: dict[str, Any] = {"path": str(path), "exists": exists, "state": "present" if exists else "missing"}
    if exists and kind == "file":
        row.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    return row


def _checkpoint_registration_state(value: Any) -> dict[str, Any]:
    """Fail closed unless every AD-GS checkpoint component is byte exact."""
    required_files = ("point_cloud.ply", "deform.pth", "env.pth")
    if not isinstance(value, Mapping):
        return {
            "state": "not_registered",
            "all_files_exact": False,
            "files": {},
        }
    configured_files = value.get("files")
    if not isinstance(configured_files, Mapping) or set(configured_files) != set(required_files):
        return {
            "state": "invalid_registration",
            "all_files_exact": False,
            "files": {},
        }
    run = Path(str(value.get("run", "")))
    run_state = _path_state(run, kind="dir") if value.get("run") else {"path": None, "exists": False, "state": "not_registered"}
    evidence: dict[str, Any] = {}
    for name, field in (("fingerprint.json", "fingerprint_sha256"), ("manifest.json", "manifest_sha256")):
        actual = _path_state(run / name) if run_state["exists"] else _path_state(None)
        expected_sha256 = value.get(field)
        evidence[name] = {
            **actual,
            "expected_sha256": expected_sha256,
            "sha256_exact": actual.get("sha256") == expected_sha256,
        }
    files: dict[str, Any] = {}
    for name in required_files:
        configured = configured_files[name]
        if not isinstance(configured, Mapping):
            files[name] = {"state": "invalid_registration", "exact": False}
            continue
        actual = _path_state(configured.get("path"))
        expected_bytes = configured.get("bytes")
        expected_sha256 = configured.get("sha256")
        bytes_exact = actual.get("bytes") == expected_bytes
        sha256_exact = actual.get("sha256") == expected_sha256
        inside_run = bool(
            actual["exists"]
            and run_state["exists"]
            and Path(str(actual["path"])).resolve().is_relative_to(run.resolve())
        )
        files[name] = {
            **actual,
            "expected_bytes": expected_bytes,
            "expected_sha256": expected_sha256,
            "bytes_exact": bytes_exact,
            "sha256_exact": sha256_exact,
            "inside_run": inside_run,
            "exact": actual["exists"] and bytes_exact and sha256_exact and inside_run,
        }
    all_files_exact = all(row.get("exact", False) for row in files.values())
    evidence_exact = all(row["exists"] and row["sha256_exact"] for row in evidence.values())
    formal_step_exact = value.get("step") == 60000
    executable_exact = all_files_exact and evidence_exact and formal_step_exact
    return {
        "state": "executable_exact" if executable_exact else "checkpoint_mismatch",
        "executable_exact": executable_exact,
        "all_files_exact": all_files_exact,
        "run": run_state,
        "step": value.get("step"),
        "formal_step_exact": formal_step_exact,
        "fingerprint_sha256": value.get("fingerprint_sha256"),
        "manifest_sha256": value.get("manifest_sha256"),
        "evidence_exact": evidence_exact,
        "evidence": evidence,
        "files": files,
    }


def _single_checkpoint_registration_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"path": None, "exists": False, "state": "not_registered", "exact": False}
    actual = _path_state(value.get("path"))
    expected_bytes = value.get("bytes")
    expected_sha256 = value.get("sha256")
    bytes_exact = actual.get("bytes") == expected_bytes
    sha256_exact = actual.get("sha256") == expected_sha256
    exact = actual["exists"] and bytes_exact and sha256_exact
    return {
        **actual,
        "state": "executable_exact" if exact else actual["state"],
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "bytes_exact": bytes_exact,
        "sha256_exact": sha256_exact,
        "exact": exact,
    }


def _v33_chain_registration_state(
    value: Any, *, expected_scene: str
) -> dict[str, Any]:
    """只有完整、内容寻址且语义闭环的逐场景链才计入 V3.3 coverage。"""
    if not isinstance(value, Mapping):
        return {
            "state": "not_registered",
            "executable_exact": False,
            "files": {},
        }
    configured_files = value.get("files")
    if not isinstance(configured_files, Mapping) or set(configured_files) != set(
        V33_REQUIRED_FILES
    ):
        return {
            "state": "invalid_registration",
            "executable_exact": False,
            "files": {},
        }

    run = Path(str(value.get("run", "")))
    run_state = (
        _path_state(run, kind="dir")
        if value.get("run")
        else {"path": None, "exists": False, "state": "not_registered"}
    )
    evidence: dict[str, Any] = {}
    for name, field in (
        ("summary.json", "summary_sha256"),
        ("manifest.json", "manifest_sha256"),
        ("status.json", "status_sha256"),
    ):
        actual = _path_state(run / name) if run_state["exists"] else _path_state(None)
        expected_sha256 = value.get(field)
        evidence[name] = {
            **actual,
            "expected_sha256": expected_sha256,
            "sha256_exact": actual.get("sha256") == expected_sha256,
        }

    files: dict[str, Any] = {}
    for name in V33_REQUIRED_FILES:
        configured = configured_files[name]
        if not isinstance(configured, Mapping):
            files[name] = {"state": "invalid_registration", "exact": False}
            continue
        actual = _path_state(configured.get("path"))
        bytes_exact = actual.get("bytes") == configured.get("bytes")
        sha256_exact = actual.get("sha256") == configured.get("sha256")
        inside_run = bool(
            actual["exists"]
            and run_state["exists"]
            and Path(str(actual["path"])).resolve().is_relative_to(run.resolve())
        )
        files[name] = {
            **actual,
            "expected_bytes": configured.get("bytes"),
            "expected_sha256": configured.get("sha256"),
            "bytes_exact": bytes_exact,
            "sha256_exact": sha256_exact,
            "inside_run": inside_run,
            "exact": actual["exists"] and bytes_exact and sha256_exact and inside_run,
        }

    all_files_exact = all(row.get("exact", False) for row in files.values())
    evidence_exact = all(row["exists"] and row["sha256_exact"] for row in evidence.values())
    chain_semantics_exact = False
    terminal_exact = False
    stage_states: dict[str, Any] = {}
    if all_files_exact:
        chain = json.loads(Path(files["scene_chain.json"]["path"]).read_text(encoding="utf-8"))
        stages = chain.get("stages", {}) if isinstance(chain, Mapping) else {}
        for stage in V33_REQUIRED_STAGES:
            row = stages.get(stage, {}) if isinstance(stages, Mapping) else {}
            status = row.get("status") if isinstance(row, Mapping) else None
            reason = row.get("reason") if isinstance(row, Mapping) else None
            valid = status == "done" or (
                stage in V33_ABSTAIN_STAGES
                and status == "abstain"
                and isinstance(reason, str)
                and bool(reason.strip())
            )
            stage_states[stage] = {"status": status, "reason": reason, "valid": valid}
        render = json.loads(
            Path(files["render_manifest.json"]["path"]).read_text(encoding="utf-8")
        )
        metrics = json.loads(
            Path(files["metrics.json"]["path"]).read_text(encoding="utf-8")
        )
        render_rows = render.get("rows", []) if isinstance(render, Mapping) else []
        metric_rows = metrics.get("rows", []) if isinstance(metrics, Mapping) else []
        evaluation_exact = bool(
            render.get("scene") == expected_scene
            and render.get("split") == "development"
            and render.get("test_quality_read") is False
            and isinstance(render_rows, list)
            and bool(render_rows)
            and metrics.get("scene") == expected_scene
            and metrics.get("split") == "development"
            and metrics.get("test_quality_read") is False
            and isinstance(metric_rows, list)
            and bool(metric_rows)
        )
        chain_semantics_exact = bool(
            chain.get("schema_version") == "worldsim_v4_v33_scene_chain_v1"
            and chain.get("scene") == expected_scene
            and chain.get("algorithm_commit") == value.get("algorithm_commit")
            and chain.get("base_checkpoint_sha256")
            == value.get("base_checkpoint_sha256")
            and chain.get("partition_contract") == "sample_index_mod_5"
            and chain.get("test_quality_read") is False
            and set(stages) == set(V33_REQUIRED_STAGES)
            and all(row["valid"] for row in stage_states.values())
            and evaluation_exact
        )
    if evidence_exact:
        terminal = json.loads(Path(evidence["status.json"]["path"]).read_text(encoding="utf-8"))
        terminal_exact = bool(
            terminal.get("scene") == expected_scene
            and terminal.get("status", terminal.get("state")) in {"done", "completed"}
            and terminal.get("test_quality_read") is False
        )
    executable_exact = bool(
        all_files_exact and evidence_exact and chain_semantics_exact and terminal_exact
    )
    return {
        "state": "executable_exact" if executable_exact else "scene_chain_mismatch",
        "executable_exact": executable_exact,
        "run": run_state,
        "scene": expected_scene,
        "algorithm_commit": value.get("algorithm_commit"),
        "base_checkpoint_sha256": value.get("base_checkpoint_sha256"),
        "all_files_exact": all_files_exact,
        "evidence_exact": evidence_exact,
        "chain_semantics_exact": chain_semantics_exact,
        "terminal_exact": terminal_exact,
        "stage_states": stage_states,
        "evidence": evidence,
        "files": files,
    }


def _validate_metrics(config: Mapping[str, Any]) -> None:
    image = config.get("image", {})
    if image.get("primary") != ["psnr", "ssim", "lpips_alex"]:
        raise BaselineAuditError("主图像指标必须冻结为 PSNR/SSIM/LPIPS-Alex")
    if image.get("regions") != ["global", "static", "actor", "boundary", "edit_roi"]:
        raise BaselineAuditError("图像区域顺序/集合漂移")
    if image.get("region_protocol") != {
        "actor": "drivestudio_dynamic_masks_all_nonzero",
        "static": "not_actor_and_not_egocar",
        "boundary": "dynamic_mask_morphological_band_l1_radius_3px",
        "baseline_edit_roi": "empty_undefined",
    }:
        raise BaselineAuditError("图像区域生成协议漂移")
    statistics = config.get("statistics", {})
    if statistics.get("unit") != "scene" or statistics.get("denominator_policy") != "retain_failed_blocked_abstain":
        raise BaselineAuditError("scene-level 统计或 denominator 合同漂移")


def audit_matrix(matrix_path: Path, project_root: Path) -> dict[str, Any]:
    matrix_path = matrix_path.resolve()
    project_root = project_root.resolve()
    matrix = _load_yaml(matrix_path)
    if matrix.get("task_id") != TASK_ID or matrix.get("status") != "running":
        raise BaselineAuditError("B0 matrix 必须是 running 状态且 task_id 精确")
    metrics_path = project_root / str(matrix["metrics_config"])
    cohort_path = project_root / str(matrix["cohort_config"])
    metrics = _load_yaml(metrics_path)
    cohort = _load_yaml(cohort_path)
    _validate_metrics(metrics)
    scenes = list(matrix.get("scene_contract", {}))
    frozen_scenes = list(cohort.get("freeze", {}).get("scene_roles", {}).get("development", []))
    if len(scenes) != 6 or set(scenes) != set(frozen_scenes):
        raise BaselineAuditError("B0 scene_contract 必须精确匹配 D0 的 6 development scenes")
    if matrix.get("resolution_contract") != {
        "sensor_rgb": [1600, 900],
        "source_config_downscale": 2,
        "model_native_render": [800, 450],
        "metric_resolution": [800, 450],
    }:
        raise BaselineAuditError("B0 resolution contract 漂移")
    baselines = matrix.get("baselines", {})
    if set(baselines) != set(REQUIRED_METHODS):
        raise BaselineAuditError("Tier A baseline 必须且仅包含 V3.3/StreetGS/AD-GS")

    street = baselines["streetgs"]
    street_root = Path(street["implementation_root"])
    street_env = Path(street["environment"])
    street_runtime = {
        "implementation": _path_state(street_root, kind="dir"),
        "environment": _path_state(street_env, kind="dir"),
        "actual_commit": _git(street_root, "rev-parse", "HEAD") if street_root.is_dir() else None,
        "expected_commit": street["implementation_commit"],
    }
    street_runtime["commit_exact"] = street_runtime["actual_commit"] == street_runtime["expected_commit"]
    street_runtime_ready = bool(
        street_runtime["implementation"]["exists"]
        and street_runtime["environment"]["exists"]
        and street_runtime["commit_exact"]
    )
    street_scenes: dict[str, Any] = {}
    for scene in scenes:
        checkpoint = _single_checkpoint_registration_state(street.get("checkpoints", {}).get(scene))
        checkpoint_exact = checkpoint["exact"]
        checkpoint["checkpoint_exact"] = checkpoint_exact
        checkpoint["runtime_ready"] = street_runtime_ready
        checkpoint["exact"] = street_runtime_ready and checkpoint_exact
        checkpoint["state"] = "executable" if checkpoint["exact"] else checkpoint["state"]
        street_scenes[scene] = checkpoint

    v33 = baselines["v33_frozen"]
    v33_source = _path_state(v33["source_run"], kind="dir")
    v33_registrations = v33.get("executable_scene_chains", {})
    if not isinstance(v33_registrations, Mapping):
        v33_registrations = {}
    v33_scenes: dict[str, Any] = {}
    for scene in scenes:
        registration = _v33_chain_registration_state(
            v33_registrations.get(scene), expected_scene=scene
        )
        legacy_exact = bool(
            scene in v33.get("legacy_executable_scenes", [])
            and scene == "scene-0230"
            and v33_source["exists"]
        )
        if legacy_exact:
            v33_scenes[scene] = {
                "state": "executable_frozen_chain",
                "source_run": v33_source["path"],
                "legacy_canonical": True,
                "registration": registration,
            }
        else:
            v33_scenes[scene] = {
                "state": (
                    "executable_frozen_chain"
                    if registration["executable_exact"]
                    else "missing_scene_chain"
                ),
                "source_run": registration.get("run", {}).get("path"),
                "legacy_canonical": False,
                "registration": registration,
            }

    adgs = baselines["ad_gs"]
    adgs_root = Path(adgs["implementation_root"])
    adgs_env = Path(adgs["environment"])
    adgs_patch = project_root / str(adgs.get("compatibility_patch", ""))
    historical = _path_state(adgs["historical_metrics"])
    historical_scenes: list[str] = []
    if historical["exists"]:
        payload = json.loads(Path(historical["path"]).read_text(encoding="utf-8"))
        historical_scenes = sorted(row["scene"] for row in payload.get("scenes", []))
    adgs_runtime = {
        "implementation": _path_state(adgs_root, kind="dir"),
        "environment": _path_state(adgs_env, kind="dir"),
        "actual_commit": _git(adgs_root, "rev-parse", "HEAD") if adgs_root.is_dir() else None,
        "expected_commit": adgs["implementation_commit"],
        "compatibility_patch": _path_state(adgs_patch),
        "expected_compatibility_patch_sha256": adgs.get("compatibility_patch_sha256"),
        "historical_metrics": historical,
        "historical_scenes": historical_scenes,
        "historical_metrics_count_as_executable": False,
    }
    adgs_runtime["commit_exact"] = adgs_runtime["actual_commit"] == adgs_runtime["expected_commit"]
    actual_modified_files = sorted(
        line[3:]
        for line in (_git(adgs_root, "status", "--short") or "").splitlines()
        if len(line) > 3
    )
    adgs_runtime["actual_modified_files"] = actual_modified_files
    adgs_runtime["expected_modified_files"] = sorted(adgs.get("expected_modified_files", []))
    adgs_runtime["modified_files_exact"] = actual_modified_files == adgs_runtime["expected_modified_files"]
    adgs_runtime["compatibility_patch_exact"] = bool(
        adgs_runtime["compatibility_patch"].get("sha256")
        == adgs_runtime["expected_compatibility_patch_sha256"]
    )
    reverse_check = subprocess.run(
        ["git", "-C", str(adgs_root), "apply", "--reverse", "--check", "--unidiff-zero", str(adgs_patch)],
        capture_output=True,
        text=True,
        check=False,
    ) if adgs_root.is_dir() and adgs_patch.is_file() else None
    adgs_runtime["compatibility_patch_reverse_check"] = bool(reverse_check and reverse_check.returncode == 0)
    runtime_ready = bool(
        adgs_runtime["implementation"]["exists"]
        and adgs_runtime["environment"]["exists"]
        and adgs_runtime["commit_exact"]
        and adgs_runtime["modified_files_exact"]
        and adgs_runtime["compatibility_patch_exact"]
        and adgs_runtime["compatibility_patch_reverse_check"]
    )
    registrations = adgs.get("executable_checkpoints", {})
    if not isinstance(registrations, Mapping):
        registrations = {}
    adgs_scenes: dict[str, Any] = {}
    for scene in scenes:
        checkpoint = _checkpoint_registration_state(registrations.get(scene))
        executable = runtime_ready and checkpoint.get("executable_exact", False)
        adgs_scenes[scene] = {
            "state": "executable" if executable else "historical_metrics_only" if scene in historical_scenes else "missing_training_required",
            "historical_metric_present": scene in historical_scenes,
            "runtime_ready": runtime_ready,
            "checkpoint": checkpoint,
        }

    inventory = {
        "schema_version": "worldsim_v4_b0_inventory_v1",
        "task_id": TASK_ID,
        "split": "development",
        "test_split_access": False,
        "scenes": scenes,
        "methods": {
            "streetgs": {"runtime": street_runtime, "scenes": street_scenes},
            "v33_frozen": {
                "runtime": {
                    "source_run": v33_source,
                    "canonical_read_only": True,
                    "algorithm_commit": v33.get("implementation_commit"),
                },
                "scenes": v33_scenes,
            },
            "ad_gs": {"runtime": adgs_runtime, "scenes": adgs_scenes},
        },
    }
    executable_counts = {
        "streetgs": sum(row["exact"] for row in street_scenes.values()),
        "v33_frozen": sum(row["state"] == "executable_frozen_chain" for row in v33_scenes.values()),
        "ad_gs": sum(row["state"] == "executable" for row in adgs_scenes.values()),
    }
    required = int(matrix["completion_gate"]["required_scene_count_per_method"])
    inventory["executable_scene_counts"] = executable_counts
    inventory["completion_gate"] = {
        "required_scene_count_per_method": required,
        "passed": all(value == required for value in executable_counts.values()),
    }
    return inventory


def run_audit(matrix_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    matrix_path = matrix_path.resolve()
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise BaselineAuditError(f"run 目录已存在，禁止复用：{run_dir}")
    inventory = audit_matrix(matrix_path, project_root)
    terminal = "done" if inventory["completion_gate"]["passed"] else "blocked"
    now = datetime.now(timezone.utc).isoformat()
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    _write_json(run_dir / "artifacts" / "baseline_inventory.json", inventory)
    shutil.copy2(matrix_path, run_dir / "resolved.yaml")
    snapshots: dict[str, Any] = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        if not source.is_file():
            raise BaselineAuditError(f"source snapshot 缺失：{relpath}")
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    fingerprint = {
        "schema_version": "worldsim_v4_b0_fingerprint_v1",
        "matrix_sha256": sha256_file(matrix_path),
        "metrics_sha256": sha256_file(project_root / _load_yaml(matrix_path)["metrics_config"]),
        "cohort_sha256": sha256_file(project_root / _load_yaml(matrix_path)["cohort_config"]),
        "inventory_sha256": sha256_file(run_dir / "artifacts" / "baseline_inventory.json"),
        "source_snapshots": snapshots,
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    _write_json(run_dir / "events.jsonl", {"at_utc": now, "event": "baseline_inventory_complete", "status": terminal})
    summary = {
        "schema_version": "worldsim_v4_b0_summary_v1",
        "task_id": TASK_ID,
        "status": terminal,
        "reason": None if terminal == "done" else "matched_baseline_assets_incomplete",
        "finished_at_utc": now,
        "executable_scene_counts": inventory["executable_scene_counts"],
        "required_scene_count_per_method": inventory["completion_gate"]["required_scene_count_per_method"],
        "inventory_sha256": fingerprint["inventory_sha256"],
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "project_git": {"head": _git(project_root, "rev-parse", "HEAD"), "branch": _git(project_root, "branch", "--show-current"), "dirty": bool(_git(project_root, "status", "--porcelain"))},
        "training_started": False,
        "model_inference_started": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": terminal, "finished_at_utc": now, "summary_sha256": sha256_file(run_dir / "summary.json")})
    artifacts = {
        str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_b0_run_manifest_v1", "task_id": TASK_ID, "status": terminal, "artifacts": artifacts, "test_quality_read": False})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldSim V4 matched baseline 入口")
    parser.add_argument("--mode", choices=["audit"], default="audit")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.matrix, args.run_dir, args.project_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
