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
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v4/baseline_matrix_v1.yaml",
    "configs/worldsim_v4/metrics_v1.yaml",
    "motion_proj/worldsim_v4/evaluator.py",
    "motion_proj/worldsim_v4/region_masks.py",
    "motion_proj/worldsim_v4/baseline_scene_evaluator.py",
    "motion_proj/worldsim_v4/statistics.py",
    "motion_proj/worldsim_v4/engineering_metrics.py",
    "scripts/run_worldsim_v4_baselines.py",
    "tests/test_worldsim_v4_evaluator.py",
    "tests/test_worldsim_v4_region_masks.py",
    "tests/test_worldsim_v4_baseline_scene_evaluator.py",
    "tests/test_worldsim_v4_statistics.py",
    "tests/test_worldsim_v4_engineering_metrics.py",
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
    return process.stdout.strip() if process.returncode == 0 else None


def _path_state(value: Any, *, kind: str = "file") -> dict[str, Any]:
    if not value:
        return {"path": None, "exists": False, "state": "not_registered"}
    path = Path(str(value))
    exists = path.is_file() if kind == "file" else path.is_dir()
    row: dict[str, Any] = {"path": str(path), "exists": exists, "state": "present" if exists else "missing"}
    if exists and kind == "file":
        row.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    return row


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
    street_scenes = {scene: _path_state(street.get("checkpoints", {}).get(scene, {}).get("path")) for scene in scenes}

    v33 = baselines["v33_frozen"]
    v33_source = _path_state(v33["source_run"], kind="dir")
    v33_scenes = {
        scene: {
            "state": "executable_frozen_chain" if scene in v33.get("executable_scenes", []) and v33_source["exists"] else "missing_scene_chain",
            "source_run": v33_source["path"] if scene in v33.get("executable_scenes", []) else None,
        }
        for scene in scenes
    }

    adgs = baselines["ad_gs"]
    adgs_root = Path(adgs["implementation_root"])
    adgs_env = Path(adgs["environment"])
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
        "historical_metrics": historical,
        "historical_scenes": historical_scenes,
        "historical_metrics_count_as_executable": False,
    }
    executable = set(adgs.get("executable_checkpoints", [])) if adgs_root.is_dir() and adgs_env.is_dir() else set()
    adgs_scenes = {
        scene: {
            "state": "executable" if scene in executable else "historical_metrics_only" if scene in historical_scenes else "missing_training_required",
            "historical_metric_present": scene in historical_scenes,
        }
        for scene in scenes
    }

    inventory = {
        "schema_version": "worldsim_v4_b0_inventory_v1",
        "task_id": TASK_ID,
        "split": "development",
        "test_split_access": False,
        "scenes": scenes,
        "methods": {
            "streetgs": {"runtime": street_runtime, "scenes": street_scenes},
            "v33_frozen": {"runtime": {"source_run": v33_source, "canonical_read_only": True}, "scenes": v33_scenes},
            "ad_gs": {"runtime": adgs_runtime, "scenes": adgs_scenes},
        },
    }
    executable_counts = {
        "streetgs": sum(row["exists"] for row in street_scenes.values()),
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
