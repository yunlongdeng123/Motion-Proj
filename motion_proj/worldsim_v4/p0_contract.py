from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "worldsim_v4_p0_scope_v1"
TASK_ID = "WS-V4-P0-SCOPE-PAPER-FREEZE-01"
ALLOWED_TASK_STATES = {"pending", "running", "blocked", "done", "rejected"}
REQUIRED_METHOD_KEYS = {
    "base_asset",
    "evidence_state",
    "multi_view_update",
    "temporal_memory",
    "calibration",
    "authenticity",
    "repair_risk",
    "temporal_transform",
    "reversible_delta",
}
REQUIRED_BASELINES = {"v33_frozen", "streetgs", "ad_gs"}
REQUIRED_IMAGE_METRICS = {"psnr", "ssim", "lpips_alex"}
REQUIRED_REGIONS = {"global", "static", "actor", "boundary", "edit_roi"}
REQUIRED_ENGINEERING_METRICS = {
    "wall",
    "peak_nvidia_vram",
    "peak_cgroup_ram",
    "asset_bytes",
    "cold_load",
    "fps",
    "pipeline_success_rate",
    "valid_edit_yield",
    "retry_amplification",
    "resume_efficiency",
}


class P0ContractError(RuntimeError):
    """V4 P0 的 paper-first 冻结合同不满足。"""


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P0ContractError("P0 配置根节点必须是 mapping")
    return payload


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise P0ContractError(f"{key} 必须是 mapping")
    return value


def _require_keys(record: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(record))
    if missing:
        raise P0ContractError(f"{label} 缺少字段：{missing}")


def _git(project_root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise P0ContractError(
            f"git {' '.join(args)} 失败：{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def _is_ancestor(project_root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    proc = subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def audit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise P0ContractError(f"schema_version 必须为 {SCHEMA_VERSION}")
    if config.get("task_id") != TASK_ID:
        raise P0ContractError(f"task_id 必须为 {TASK_ID}")
    if config.get("status") != "done":
        raise P0ContractError("P0 仅在全部冻结门通过后标记 done")

    project = _require_mapping(config, "project")
    if project.get("branch") != "research/worldsim-v4-evidelta":
        raise P0ContractError("V4 分支必须冻结为 research/worldsim-v4-evidelta")
    for key in ("head_at_start", "v33_closeout_commit", "plan_sha256"):
        value = project.get(key)
        if not isinstance(value, str) or len(value) not in {7, 40, 64}:
            raise P0ContractError(f"project.{key} 不是合法冻结标识")
    if project.get("v33_canonical_read_only") is not True:
        raise P0ContractError("V3.3 canonical 必须只读")

    method = _require_mapping(config, "method_schema")
    _require_keys(method, REQUIRED_METHOD_KEYS, "method_schema")
    if method["calibration"].get("fit_split") != "development":
        raise P0ContractError("概率校准只能在 development 拟合")
    if method["reversible_delta"].get("base_immutable") is not True:
        raise P0ContractError("delta 合同必须保持 base immutable")
    if method["reversible_delta"].get("rollback_render_sha_exact") is not True:
        raise P0ContractError("delta 合同必须要求 rollback render SHA exact")

    datasets = _require_mapping(config, "datasets")
    nuscenes = _require_mapping(datasets, "nuscenes")
    counts = nuscenes.get("scene_counts")
    if counts != {"development": 6, "validation": 6, "test": 18}:
        raise P0ContractError("nuScenes split 必须为 6/6/18")
    if sum(counts.values()) != 30 or nuscenes.get("scene_disjoint") is not True:
        raise P0ContractError("nuScenes 必须是 30 个 scene-disjoint 场景")
    if nuscenes.get("selection_uses_model_results") is not False:
        raise P0ContractError("nuScenes cohort 不得使用模型结果选场景")
    if nuscenes.get("test_read_count") != 1:
        raise P0ContractError("test 只允许读取一次")
    kitti = _require_mapping(datasets, "kitti")
    if kitti.get("download_allowed") is not False:
        raise P0ContractError("KITTI 禁止下载")
    if kitti.get("method_threshold_source") != "frozen_nuscenes":
        raise P0ContractError("KITTI 阈值必须来自冻结 nuScenes")

    baselines = _require_mapping(config, "baselines")
    _require_keys(baselines, REQUIRED_BASELINES, "baselines")
    if any(not item.get("same_split") for item in baselines.values() if item.get("tier") == "A"):
        raise P0ContractError("Tier A baseline 必须使用相同 split")

    metrics = _require_mapping(config, "metrics")
    if set(metrics.get("image", {}).get("primary", [])) != REQUIRED_IMAGE_METRICS:
        raise P0ContractError("图像主指标必须且仅冻结 PSNR/SSIM/LPIPS-Alex")
    if set(metrics.get("image", {}).get("regions", [])) != REQUIRED_REGIONS:
        raise P0ContractError("图像区域合同不完整")
    if not REQUIRED_ENGINEERING_METRICS.issubset(set(metrics.get("engineering", []))):
        raise P0ContractError("工程指标合同不完整")
    if metrics.get("statistics", {}).get("unit") != "scene":
        raise P0ContractError("统计单位必须是 scene")
    if metrics.get("statistics", {}).get("include_failed_blocked_abstain") is not True:
        raise P0ContractError("failed/blocked/abstain 必须保留在 denominator")

    tasks = _require_mapping(config, "task_registry")
    invalid = sorted(
        task_id for task_id, state in tasks.items() if state not in ALLOWED_TASK_STATES
    )
    if invalid:
        raise P0ContractError(f"任务状态非法：{invalid}")

    gates = _require_mapping(config, "gates")
    required_true = {
        "paper_claim_frozen",
        "math_schema_frozen",
        "baseline_matrix_frozen",
        "metrics_schema_frozen",
        "dataset_protocol_frozen",
        "kitti_layout_audited",
        "no_training",
        "no_model_inference",
        "no_weight_download",
        "d0_authorized",
    }
    failed = sorted(key for key in required_true if gates.get(key) is not True)
    if failed:
        raise P0ContractError(f"P0 必须门未通过：{failed}")
    for key in ("d1_authorized", "b0_authorized", "m1_authorized", "m2_authorized", "m3_authorized", "test_authorized"):
        if gates.get(key) is not False:
            raise P0ContractError(f"P0 收口时 {key} 必须为 false")

    sources = config.get("literature")
    if not isinstance(sources, list) or len(sources) < 12:
        raise P0ContractError("P0 至少登记 12 个一手文献/官方项目")
    for index, item in enumerate(sources):
        if not isinstance(item, Mapping):
            raise P0ContractError(f"literature[{index}] 必须是 mapping")
        _require_keys(item, {"name", "primary_url", "role", "execution_state"}, f"literature[{index}]")
        if not str(item["primary_url"]).startswith("https://"):
            raise P0ContractError(f"literature[{index}] 必须使用 HTTPS 一手来源")

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done",
        "literature_count": len(sources),
        "task_count": len(tasks),
        "nuscenes_scene_count": sum(counts.values()),
        "kitti_layout_state": kitti.get("layout_state"),
        "gates": dict(gates),
    }


def audit_repository(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    project = _require_mapping(config, "project")
    branch = _git(project_root, "branch", "--show-current")
    if branch != project["branch"]:
        raise P0ContractError(f"当前分支 {branch} != {project['branch']}")
    if not _is_ancestor(project_root, project["head_at_start"]):
        raise P0ContractError("P0 起始 HEAD 不在当前历史中")
    if not _is_ancestor(project_root, project["v33_closeout_commit"]):
        raise P0ContractError("V3.3 收口提交不在当前历史中")
    plan = project_root / project["plan_path"]
    if not plan.is_file():
        raise P0ContractError(f"V4 计划不存在：{plan}")
    if sha256_file(plan) != project["plan_sha256"]:
        raise P0ContractError("V4 计划 SHA-256 不一致")
    kitti_root = Path(config["datasets"]["kitti"]["root"])
    expected_present = config["datasets"]["kitti"]["root_present_at_p0"]
    if kitti_root.exists() is not expected_present:
        raise P0ContractError("KITTI 根目录状态与 P0 冻结事实不一致")
    return {
        "head": _git(project_root, "rev-parse", "HEAD"),
        "branch": branch,
        "dirty": _git(project_root, "status", "--porcelain") != "",
        "head_at_start_is_ancestor": True,
        "v33_closeout_is_ancestor": True,
        "plan_sha256_exact": True,
        "kitti_root_present": kitti_root.exists(),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_audit(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    if run_dir.exists():
        raise P0ContractError(f"run 目录已存在，禁止复用：{run_dir}")
    config = load_config(config_path)
    summary = audit_config(config)
    summary["repository"] = audit_repository(config, project_root)
    now = datetime.now(timezone.utc).isoformat()
    summary.update(
        {
            "finished_at_utc": now,
            "config_sha256": sha256_file(config_path),
            "run_dir": str(run_dir),
        }
    )

    run_dir.mkdir(parents=True)
    snapshot_relpaths = (
        "motion_proj/worldsim_v4/p0_contract.py",
        "scripts/audit_worldsim_v4_start.py",
        "tests/test_worldsim_v4_p0_contract.py",
        "configs/worldsim_v4/p0_scope_v1.yaml",
        "docs/WORLDSIM_V4_EVIDELTA_GS_PLAN.md",
    )
    snapshots: dict[str, Any] = {}
    for relpath in snapshot_relpaths:
        source = project_root / relpath
        if not source.is_file():
            raise P0ContractError(f"source snapshot 文件不存在：{source}")
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    summary["source_snapshots"] = snapshots
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    _write_json(run_dir / "summary.json", summary)
    status = {
        "task_id": TASK_ID,
        "status": "done",
        "finished_at_utc": now,
        "summary_sha256": sha256_file(run_dir / "summary.json"),
    }
    _write_json(run_dir / "status.json", status)
    artifacts = {
        name: {"bytes": (run_dir / name).stat().st_size, "sha256": sha256_file(run_dir / name)}
        for name in ("resolved.yaml", "summary.json", "status.json")
    }
    manifest = {
        "schema_version": "worldsim_v4_p0_run_manifest_v1",
        "task_id": TASK_ID,
        "status": "done",
        "artifacts": artifacts,
        "source_snapshots": snapshots,
        "no_training": True,
        "no_model_inference": True,
        "no_weight_download": True,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return summary
