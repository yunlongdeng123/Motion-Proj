"""V5.1 M1-only 的冻结协议校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


V5_CANONICAL_HEAD = "44d0e4a2468112b89a454992ecd9177d65184067"
V51_BRANCH = "research/worldsim-v5.1-m1"
AUTHORIZED_FIRST_ROUND = (
    "WS-V51-P0-M1-SCOPE-FREEZE-01",
    "WS-V51-D0-DEV-ROLE-FREEZE-01",
    "WS-V51-M1-A-UNARY-OBSERVABILITY-01",
)
DEVELOPMENT_ROLE_ORDER = (
    "scene-0471",
    "scene-1087",
    "scene-0379",
    "scene-0998",
    "scene-0359",
    "scene-0875",
    "scene-0535",
    "scene-0436",
)


class ProtocolError(RuntimeError):
    """冻结协议、输入身份或研究边界发生漂移。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"YAML 顶层必须是 mapping: {path}")
    return payload


def _expect_file(project: Path, spec: Mapping[str, Any]) -> Path:
    path = project / str(spec["path"])
    if not path.is_file():
        raise ProtocolError(f"冻结文件缺失: {path}")
    observed = sha256_file(path)
    if observed != spec["file_sha256"]:
        raise ProtocolError(f"冻结文件 SHA 漂移: {path}")
    return path


def validate_scope(project: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v51_p0_m1_scope_v1":
        raise ProtocolError("P0 schema 漂移")
    if config.get("task_id") != AUTHORIZED_FIRST_ROUND[0] or config.get("status") != "done":
        raise ProtocolError("P0 task/status 漂移")
    project_cfg = config["project"]
    if project_cfg.get("branch") != V51_BRANCH:
        raise ProtocolError("V5.1 branch 漂移")
    if project_cfg.get("v5_canonical_head") != V5_CANONICAL_HEAD:
        raise ProtocolError("V5 canonical HEAD 漂移")
    plan = project / project_cfg["normative_plan"]["path"]
    if sha256_file(plan) != project_cfg["normative_plan"]["sha256"]:
        raise ProtocolError("V5.1 normative plan SHA 漂移")
    authorization = config["first_round_authorization"]
    if tuple(authorization["tasks"]) != AUTHORIZED_FIRST_ROUND:
        raise ProtocolError("第一轮授权集合或顺序漂移")
    if not authorization.get("later_stages_locked"):
        raise ProtocolError("后续 Stage 未锁定")
    if authorization.get("m2_status") != "pending" or authorization.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    locks = config["data_locks"]
    for name in (
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
        "sam_threshold_search",
        "base_reconstruction_change",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"数据/方法锁漂移: {name}")
    cohort_path = _expect_file(project, locks["source_cohort"])
    cohort = load_yaml(cohort_path)
    observed_cohort_sha = cohort.get("freeze", {}).get("cohort_sha256")
    if observed_cohort_sha != locks["source_cohort"]["cohort_sha256"]:
        raise ProtocolError("cohort identity 漂移")
    if not config.get("failure_ledger_refs"):
        raise ProtocolError("P0 缺 failure_ledger_refs")
    return {
        "task_id": config["task_id"],
        "normative_plan_sha256": project_cfg["normative_plan"]["sha256"],
        "cohort_file_sha256": locks["source_cohort"]["file_sha256"],
        "cohort_sha256": observed_cohort_sha,
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
    }


def validate_development_roles(
    project: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v51_development_roles_v1":
        raise ProtocolError("development roles schema 漂移")
    if config.get("task_id") != AUTHORIZED_FIRST_ROUND[1] or config.get("status") != "done":
        raise ProtocolError("development roles task/status 漂移")
    cohort_path = _expect_file(project, config["source_cohort"])
    cohort = load_yaml(cohort_path)
    source_development = tuple(cohort["freeze"]["scene_roles"]["development"])
    roles = config["roles"]
    frozen = tuple(
        list(roles["historical_diagnostic"]["scenes"])
        + list(roles["screening"]["scenes"])
        + list(roles["development_confirmation"]["scenes"])
    )
    if frozen != DEVELOPMENT_ROLE_ORDER or frozen != source_development:
        raise ProtocolError("H/S/C 场景身份或顺序漂移")
    if len(set(frozen)) != len(frozen):
        raise ProtocolError("H/S/C 场景不互斥")
    clean = config["clean_cohorts"]
    validation = cohort["freeze"]["scene_roles"]["validation"]
    test = cohort["freeze"]["scene_roles"]["test"]
    if clean["validation"]["expected_scene_count"] != len(validation):
        raise ProtocolError("validation 分母漂移")
    if clean["test"]["expected_scene_count"] != len(test):
        raise ProtocolError("test 分母漂移")
    if clean["validation"].get("quality_read") is not False:
        raise ProtocolError("validation quality 已被错误解锁")
    if clean["test"].get("quality_read") is not False:
        raise ProtocolError("test quality 已被错误解锁")
    if not config.get("failure_ledger_refs"):
        raise ProtocolError("development roles 缺 failure_ledger_refs")
    return {
        "task_id": config["task_id"],
        "historical_diagnostic": list(roles["historical_diagnostic"]["scenes"]),
        "screening": list(roles["screening"]["scenes"]),
        "development_confirmation": list(roles["development_confirmation"]["scenes"]),
        "validation_scene_count": len(validation),
        "test_scene_count": len(test),
        "validation_quality_read": False,
        "test_quality_read": False,
    }


def verify_manifest_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ProtocolError(f"canonical manifest inventory 为空: {run_dir}")
    total_bytes = 0
    for record in inventory:
        path = run_dir / record["path"]
        if not path.is_file():
            raise ProtocolError(f"canonical artifact 缺失: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ProtocolError(f"canonical artifact bytes 漂移: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ProtocolError(f"canonical artifact SHA 漂移: {path}")
        total_bytes += int(record["bytes"])
    return {"file_count": len(inventory), "total_bytes": total_bytes}


def verify_canonical_run(scene: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["path"])
    if not run_dir.is_dir():
        raise ProtocolError(f"canonical run 缺失: {run_dir}")
    for relative, expected in spec["hashes"].items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"canonical 绑定漂移: {path}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "gaussian_count": int(spec["gaussian_count"]),
        "evidence_view_count": int(spec["evidence_view_count"]),
        "accepted_evaluation_view_count": int(spec["accepted_evaluation_view_count"]),
        "abstained_evaluation_view_count": int(spec["abstained_evaluation_view_count"]),
    }
    if summary.get("scene") != scene or summary.get("status") != "done":
        raise ProtocolError(f"canonical summary scene/status 漂移: {scene}")
    if status.get("status") != "done" or manifest.get("status") != "done":
        raise ProtocolError(f"canonical terminal 漂移: {scene}")
    for name, value in expected.items():
        if int(summary.get(name, -1)) != value:
            raise ProtocolError(f"canonical summary 分母漂移: {scene}/{name}")
    if summary.get("validation_quality_read") is not False or summary.get("heldout_quality_read") is not False:
        raise ProtocolError(f"canonical run 越界读取: {scene}")
    checkpoint = summary.get("checkpoint_sha256_before")
    if checkpoint != spec["checkpoint_sha256"] or summary.get("checkpoint_sha256_after") != checkpoint:
        raise ProtocolError(f"canonical checkpoint identity 漂移: {scene}")
    inventory = verify_manifest_inventory(run_dir, manifest)
    return {
        "scene": scene,
        "run_id": spec["run_id"],
        "path": str(run_dir),
        "summary_sha256": spec["hashes"]["summary.json"],
        "manifest_sha256": spec["hashes"]["manifest.json"],
        "checkpoint_sha256": checkpoint,
        "inventory": inventory,
    }
