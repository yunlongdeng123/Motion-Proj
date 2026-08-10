from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "worldsim_v33_p0_sources_v1"
TASK_ID = "WS-V33-P0-ROUTE-SOTA-AUDIT-01"
EXECUTION_STATES = {
    "executable",
    "audit_only",
    "license_blocked",
    "weights_blocked",
    "source_not_released",
}
REQUIRED_SOURCE_FIELDS = {
    "name",
    "official_url",
    "paper_url",
    "commit",
    "tree_sha",
    "license",
    "license_sha256",
    "weights",
    "weights_revision",
    "weights_sha256",
    "python",
    "torch",
    "cuda",
    "single_3090",
    "input_schema",
    "output_schema",
    "execution_state",
}


class SourceAuditError(RuntimeError):
    """P0 source/provenance 合同不满足。"""


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SourceAuditError("P0 配置根节点必须是 mapping")
    return data


def _git(checkout: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SourceAuditError(
            f"git {' '.join(args)} 失败：{checkout}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _audit_source(key: str, record: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_SOURCE_FIELDS - set(record))
    if missing:
        raise SourceAuditError(f"source={key} 缺少字段：{missing}")

    state = record["execution_state"]
    if state not in EXECUTION_STATES:
        raise SourceAuditError(f"source={key} execution_state 非法：{state}")

    checkout_raw = record.get("checkout")
    result: dict[str, Any] = {
        "execution_state": state,
        "checkout_present": False,
        "commit_exact": None,
        "tree_exact": None,
        "license_exact": None,
        "clean": None,
    }
    if checkout_raw is None:
        if record["commit"] is not None or record["tree_sha"] is not None:
            raise SourceAuditError(f"source={key} 无 checkout 时 commit/tree 必须为空")
        if state not in {"audit_only", "source_not_released"}:
            raise SourceAuditError(f"source={key} 无 checkout 不能标记为 {state}")
        return result

    checkout = Path(checkout_raw)
    if not (checkout / ".git").exists():
        raise SourceAuditError(f"source={key} checkout 不存在或不是 Git 仓库：{checkout}")

    actual_commit = _git(checkout, "rev-parse", "HEAD")
    actual_tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    if actual_commit != record["commit"]:
        raise SourceAuditError(
            f"source={key} commit 不一致：{actual_commit} != {record['commit']}"
        )
    if actual_tree != record["tree_sha"]:
        raise SourceAuditError(
            f"source={key} tree 不一致：{actual_tree} != {record['tree_sha']}"
        )

    dirty = _git(checkout, "status", "--porcelain")
    result.update(
        {
            "checkout_present": True,
            "commit_exact": True,
            "tree_exact": True,
            "clean": dirty == "",
        }
    )
    if dirty:
        raise SourceAuditError(f"source={key} checkout 非 clean")

    license_path_raw = record.get("license_path")
    if license_path_raw is None:
        if record["license_sha256"] is not None:
            raise SourceAuditError(f"source={key} 无 license_path 却声明 license SHA")
        return result

    license_path = checkout / license_path_raw
    if not license_path.is_file():
        raise SourceAuditError(f"source={key} license 文件不存在：{license_path}")
    actual_license_sha = sha256_file(license_path)
    if actual_license_sha != record["license_sha256"]:
        raise SourceAuditError(
            f"source={key} license SHA 不一致：{actual_license_sha}"
        )
    result["license_exact"] = True
    return result


def audit_config(
    config: Mapping[str, Any], *, verify_large_assets: bool = False
) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SourceAuditError(f"schema_version 必须为 {SCHEMA_VERSION}")
    if config.get("task_id") != TASK_ID:
        raise SourceAuditError(f"task_id 必须为 {TASK_ID}")
    if config.get("status") != "done":
        raise SourceAuditError("P0 事实配置只允许在全部门通过后标记 done")

    sources = config.get("sources")
    if not isinstance(sources, dict) or len(sources) < 9:
        raise SourceAuditError("P0 至少必须登记 9 个计划内 SOTA source")
    source_results = {key: _audit_source(key, value) for key, value in sources.items()}

    gates = config.get("gates")
    if not isinstance(gates, dict):
        raise SourceAuditError("缺少 gates")
    required_true = {
        "v32_canonical_immutable",
        "no_training",
        "no_model_inference",
        "no_large_weight_download",
        "s1_authorized",
    }
    failed = sorted(key for key in required_true if gates.get(key) is not True)
    if failed:
        raise SourceAuditError(f"P0 必须门未通过：{failed}")
    for key in ("s2_authorized", "s3_authorized", "s4_authorized", "s5_authorized"):
        if gates.get(key) is not False:
            raise SourceAuditError(f"P0 收口时 {key} 必须为 false")

    asset_results: dict[str, Any] = {}
    for key, record in config.get("v32_canonical", {}).get("assets", {}).items():
        path = Path(record["path"])
        if not path.is_file():
            raise SourceAuditError(f"V3.2 canonical 资产不存在：{key}: {path}")
        item = {"present": True, "bytes": path.stat().st_size, "sha256_exact": None}
        if item["bytes"] != int(record["bytes"]):
            raise SourceAuditError(f"V3.2 canonical 资产 bytes 不一致：{key}")
        if verify_large_assets:
            actual = sha256_file(path)
            if actual != record["sha256"]:
                raise SourceAuditError(f"V3.2 canonical 资产 SHA 不一致：{key}")
            item["sha256_exact"] = True
        asset_results[key] = item

    counts = Counter(record["execution_state"] for record in sources.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done",
        "source_count": len(sources),
        "execution_state_counts": dict(sorted(counts.items())),
        "sources": source_results,
        "v32_assets": asset_results,
        "large_assets_verified": verify_large_assets,
        "gates": dict(gates),
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
    verify_large_assets: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise SourceAuditError(f"run 目录已存在，禁止复用：{run_dir}")

    config = load_config(config_path)
    summary = audit_config(config, verify_large_assets=verify_large_assets)
    now = datetime.now(timezone.utc).isoformat()
    summary["finished_at_utc"] = now
    summary["config_sha256"] = sha256_file(config_path)
    summary["run_dir"] = str(run_dir)

    run_dir.mkdir(parents=True)
    if project_root is None:
        candidates = [config_path.parent, *config_path.parents]
        project_root_path = next(
            (candidate for candidate in candidates if (candidate / ".git").exists()),
            None,
        )
        if project_root_path is None:
            raise SourceAuditError("无法从 config 路径定位项目 Git 根目录")
    else:
        project_root_path = Path(project_root).resolve()
    snapshot_records: dict[str, Any] = {}
    snapshot_relpaths = (
        "motion_proj/worldsim_v33/source_audit.py",
        "scripts/audit_worldsim_v33_sources.py",
        "tests/test_worldsim_v33_source_audit.py",
    )
    for relpath in snapshot_relpaths:
        source = project_root_path / relpath
        if not source.is_file():
            raise SourceAuditError(f"source snapshot 文件不存在：{source}")
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshot_records[relpath] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    summary["project_git"] = {
        "head": _git(project_root_path, "rev-parse", "HEAD"),
        "branch": _git(project_root_path, "branch", "--show-current"),
        "dirty": _git(project_root_path, "status", "--porcelain") != "",
    }
    summary["source_snapshots"] = snapshot_records
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    _write_json(run_dir / "summary.json", summary)
    summary_sha = sha256_file(run_dir / "summary.json")
    status = {
        "task_id": TASK_ID,
        "status": "done",
        "finished_at_utc": now,
        "summary_sha256": summary_sha,
    }
    _write_json(run_dir / "status.json", status)

    artifacts = {}
    for name in ("resolved.yaml", "summary.json", "status.json"):
        path = run_dir / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "worldsim_v33_p0_run_manifest_v1",
        "task_id": TASK_ID,
        "status": "done",
        "artifacts": artifacts,
        "source_snapshots": snapshot_records,
        "no_training": True,
        "no_model_inference": True,
        "no_weight_download": True,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return summary
