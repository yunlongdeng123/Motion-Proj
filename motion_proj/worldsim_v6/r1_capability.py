"""WorldSim V6 R1 前端能力审计。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_MATRIX_FIELDS = {
    "paper",
    "official_source",
    "commit",
    "license",
    "weights",
    "input_schema",
    "output_schema",
    "gpu_requirement",
    "local_status",
    "adapter_cost",
    "selected_role",
}


class R1AuditError(RuntimeError):
    """R1 配置或能力证据不满足合同。"""


def _repo_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _frontend_facts(local: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    frontend = local.get("frontends", {}).get(candidate_id, {})
    repo = local.get("third_party", {}).get(frontend.get("repo_key"), {})
    env = local.get("envs", {}).get(frontend.get("env_key"), {})
    checkpoint = local.get("checkpoints", {}).get(frontend.get("checkpoint_key"), {})
    return {
        "repo_exists": bool(repo.get("exists")),
        "repo_commit": repo.get("commit"),
        "repo_clean": repo.get("clean"),
        "required_files_complete": bool(repo.get("required_files"))
        and all(repo.get("required_files", {}).values()),
        "license_files": repo.get("license_files", []),
        "env_python_exists": bool(env.get("exists")),
        "checkpoint_exists": bool(checkpoint.get("exists")),
        "checkpoint_bytes": checkpoint.get("bytes"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "checkpoint_count": int(frontend.get("checkpoint_count", 0)),
        "input_ready": bool(frontend.get("input_ready")),
        "base_nuscenes_ready": bool(frontend.get("base_nuscenes_ready")),
        "historical_run_count": int(frontend.get("historical_run_count", 0)),
    }


def _status_for(
    candidate: Mapping[str, Any], facts: Mapping[str, Any], gpu: Mapping[str, Any]
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    expected_commit = candidate.get("expected_commit")
    repo_exact = facts["repo_exists"] and facts["repo_commit"] == expected_commit
    required_files = facts["required_files_complete"] or not candidate.get("required_files")
    min_vram = int(candidate.get("gpu_requirement", {}).get("minimum_vram_mib", 0))
    gpu_ready = bool(gpu.get("available")) and int(gpu.get("total_vram_mib") or 0) >= min_vram
    remote_weights = bool(candidate.get("weights", {}).get("remote_available"))

    if not repo_exact:
        blockers.append("exact_source_checkout_missing")
    if not required_files:
        blockers.append("required_source_files_missing")
    if not gpu_ready:
        blockers.append("gpu_requirement_unsatisfied")

    kind = candidate["kind"]
    if kind == "optimization":
        if not facts["env_python_exists"]:
            blockers.append("runtime_environment_missing")
        if facts["checkpoint_count"] < int(candidate.get("minimum_checkpoint_count", 1)):
            blockers.append("executable_checkpoint_missing")
        if repo_exact and required_files and facts["env_python_exists"] and not blockers:
            return "executable", []
        return "unavailable", blockers

    if kind == "audit_only":
        if not facts["checkpoint_exists"]:
            blockers.append("model_checkpoint_missing")
        if not facts["input_ready"]:
            blockers.append("licensed_input_contract_missing")
        return "audit_only", blockers

    if not facts["env_python_exists"]:
        blockers.append("runtime_environment_missing")
    if not facts["checkpoint_exists"]:
        blockers.append("local_weights_missing")
    if not facts["input_ready"]:
        blockers.append("native_input_schema_missing")
    if (
        repo_exact
        and required_files
        and facts["env_python_exists"]
        and facts["checkpoint_exists"]
        and facts["input_ready"]
        and gpu_ready
    ):
        return "executable", []
    if repo_exact and required_files and remote_weights and gpu_ready and candidate.get("adapter_possible"):
        return "adaptable", blockers
    if candidate.get("allow_frozen_output_adapter") and facts["historical_run_count"] > 0:
        return "adaptable_from_frozen_outputs", blockers
    return "unavailable", blockers


def build_matrix(config: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
    """从声明式候选配置和本机事实生成不含绝对路径的审计矩阵。"""
    if config.get("schema_version") != "worldsim_v6.r1_frontend_capability.v1":
        raise R1AuditError("R1 config schema 漂移")
    rows = []
    gpu = local.get("gpu", {})
    for candidate in config.get("candidates", []):
        candidate_id = candidate["id"]
        facts = _frontend_facts(local, candidate_id)
        status, blockers = _status_for(candidate, facts, gpu)
        license_record = dict(candidate["license"])
        license_record["checkout_files"] = facts["license_files"]
        weight_record = dict(candidate["weights"])
        weight_record.update(
            {
                "local_exists": facts["checkpoint_exists"],
                "local_bytes": facts["checkpoint_bytes"],
                "local_sha256": facts["checkpoint_sha256"],
            }
        )
        row = {
            "candidate_id": candidate_id,
            "kind": candidate["kind"],
            "paper": candidate["paper"],
            "official_source": candidate["official_source"],
            "commit": facts["repo_commit"] or candidate.get("expected_commit"),
            "license": license_record,
            "weights": weight_record,
            "input_schema": candidate["input_schema"],
            "output_schema": candidate["output_schema"],
            "gpu_requirement": candidate["gpu_requirement"],
            "local_status": status,
            "adapter_cost": candidate["adapter_cost"],
            "selected_role": candidate.get("selected_role") if status.startswith(("executable", "adaptable")) else "none",
            "audit": {
                "repo_exact": facts["repo_exists"]
                and facts["repo_commit"] == candidate.get("expected_commit"),
                "required_files_complete": facts["required_files_complete"],
                "env_python_exists": facts["env_python_exists"],
                "checkpoint_count": facts["checkpoint_count"],
                "input_ready": facts["input_ready"],
                "base_nuscenes_ready": facts["base_nuscenes_ready"],
                "historical_run_count": facts["historical_run_count"],
                "blockers": blockers,
            },
        }
        if not REQUIRED_MATRIX_FIELDS.issubset(row):
            raise R1AuditError(f"{candidate_id} 输出字段不完整")
        rows.append(row)

    optimization = [row for row in rows if row["kind"] == "optimization" and row["local_status"] == "executable"]
    feed_forward = [
        row
        for row in rows
        if row["kind"] == "feed_forward"
        and row["local_status"].startswith(("executable", "adaptable"))
    ]
    return {
        "schema_version": "worldsim_v6.frontend_capability_matrix.v1",
        "task_id": "WS-V6-R1-FRONTEND-CAPABILITY-01",
        "quality_data_read": False,
        "training_started": False,
        "model_inference_started": False,
        "gpu": {
            "available": gpu.get("available"),
            "name": gpu.get("name"),
            "total_vram_mib": gpu.get("total_vram_mib"),
            "compute_capability": gpu.get("compute_capability"),
        },
        "frontends": rows,
        "gate": {
            "optimization_executable": len(optimization),
            "feed_forward_executable_or_adaptable": len(feed_forward),
            "passed": bool(optimization and feed_forward),
        },
    }


def run_audit(
    repo_root: Path,
    config_path: Path,
    local_manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """生成一次不可覆盖的 R1 capability run。"""
    if run_dir.exists():
        raise R1AuditError(f"run 目录已存在：{run_dir}")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    source_status = _git(repo_root, "status", "--porcelain")
    if source_status:
        raise R1AuditError("正式 R1 run 禁止使用 dirty source；先提交协议与实现")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    local = yaml.safe_load(local_manifest_path.read_text(encoding="utf-8"))
    matrix = build_matrix(config, local)
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "FRONTEND_CAPABILITY_MATRIX.json", matrix)
    environment = {
        "schema_version": "worldsim_v6.r1_environment_snapshot.v1",
        "gpu": matrix["gpu"],
        "disk": local.get("disk"),
        "datasets": {
            key: {field: value for field, value in row.items() if field != "path"}
            for key, row in local.get("datasets", {}).items()
        },
        "quality_data_read": False,
    }
    _write_json(run_dir / "ENVIRONMENT_SNAPSHOT.json", environment)
    status = "done" if matrix["gate"]["passed"] else "blocked"
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = {
        "schema_version": "worldsim_v6.r1_summary.v1",
        "task_id": matrix["task_id"],
        "status": status,
        "finished_at_utc": now,
        "source_commit": source_commit,
        "source_dirty": False,
        "config_sha256": _sha256(config_path),
        "local_manifest_sha256": _sha256(local_manifest_path),
        "gate": matrix["gate"],
        "quality_data_read": False,
        "training_started": False,
        "model_inference_started": False,
        "next_task": "WS-V6-R2-SCENEIR-V0-01" if status == "done" else "search_next_feed_forward_candidate",
    }
    _write_json(run_dir / "summary.json", summary)
    artifacts = []
    for path in sorted(run_dir.glob("*.json")):
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    manifest = {
        "schema_version": "worldsim_v6.r1_artifact_manifest.v1",
        "task_id": matrix["task_id"],
        "artifacts": artifacts,
    }
    _write_json(run_dir / "manifest.json", manifest)
    terminal = {
        "schema_version": "worldsim_v6.terminal.v1",
        "task_id": matrix["task_id"],
        "status": status,
        "finished_at_utc": now,
        "summary_sha256": _sha256(run_dir / "summary.json"),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
    }
    _write_json(run_dir / "terminal.json", terminal)
    return summary


def default_run_dir(repo_root: Path) -> Path:
    """生成 UTC 唯一 run 路径。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        repo_root.parent
        / "runs/worldsim_v6/WS-V6-R1-FRONTEND-CAPABILITY-01"
        / f"{timestamp}__r1-capability-s0-r1"
    )


def main() -> int:
    import argparse

    from motion_proj.worldsim_v6.capabilities import (
        discover_local_capabilities,
        write_local_capabilities,
    )

    parser = argparse.ArgumentParser(description="运行 WorldSim V6 R1 capability audit")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--local-manifest", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    repo_root = _repo_root()
    config_path = args.config or repo_root / "configs/worldsim_v6/r1_frontend_capability_v1.yaml"
    local_path = args.local_manifest or repo_root / ".local/worldsim_v6/capabilities.local.yaml"
    write_local_capabilities(local_path, discover_local_capabilities(repo_root))
    summary = run_audit(repo_root, config_path, local_path, args.run_dir or default_run_dir(repo_root))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
