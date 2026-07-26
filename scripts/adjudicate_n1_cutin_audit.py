#!/usr/bin/env python
"""不可变登记第四次 receiver-centric N1 人工评测的用户 reject。

本工具只校验、复制并聚合用户已经填写的 verdict；它不会修改 parent run、
不会推断人工判断，也不会授权或启动 N2。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import generate_run_id, utc_now
from scripts.validate_n1_cutin_review import COMPONENT_FIELDS, validate


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 不是 JSON object")
        rows.append(value)
    return rows


def _unique_terminal(parent: Path) -> str:
    markers = [
        name
        for name in ("COMPLETE", "REJECTED", "FAILED", "AWAITING_HUMAN_REVIEW")
        if (parent / name).is_file()
    ]
    if markers != ["AWAITING_HUMAN_REVIEW"]:
        raise RuntimeError(f"父 run 终态不唯一或不是 AWAITING_HUMAN_REVIEW: {markers}")
    return markers[0]


def _component_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
        for field in COMPONENT_FIELDS
    }


def _parent_snapshot(parent: Path, immutable_set_sha: str) -> dict[str, str]:
    """记录 parent 的不可变输入指纹，供独立裁决 run 审计。"""
    return {
        "audit_immutable_artifact_set_sha256": immutable_set_sha,
        "event_pool_file_sha256": file_fingerprint(str(parent / "event_pool.json")),
        "manifest_file_sha256": file_fingerprint(str(parent / "manifest.json")),
        "resolved_config_file_sha256": file_fingerprint(str(parent / "resolved.yaml")),
        "summary_file_sha256": file_fingerprint(str(parent / "summary.json")),
    }


def _require_expected_counts(config: dict, validation: dict) -> None:
    expected_count = int(config["expected_review_record_count"])
    if validation["reviewed_item_count"] != expected_count:
        raise RuntimeError(
            "已审条数与冻结第四轮结果不一致: "
            f"{validation['reviewed_item_count']} != {expected_count}"
        )
    expected_counts = {
        str(key): int(value)
        for key, value in config["expected_overall_counts"].items()
    }
    actual_counts = {
        str(key): int(value) for key, value in validation["overall_counts"].items()
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"人工总体计数与冻结第四轮结果不一致: {actual_counts} != {expected_counts}"
        )


def adjudicate(config_path: Path, output_root: Path | None = None) -> Path:
    config = _load_yaml(config_path)
    parent = Path(config["parent_run"])
    _unique_terminal(parent)

    parent_summary = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    event_pool = json.loads((parent / "event_pool.json").read_text(encoding="utf-8"))
    stored_pool_sha = event_pool.get("event_pool_sha256")
    unhashed_pool = dict(event_pool)
    unhashed_pool.pop("event_pool_sha256", None)
    if canonical_sha256(unhashed_pool) != stored_pool_sha:
        raise RuntimeError("父 event_pool canonical SHA256 不匹配")
    if stored_pool_sha != parent_summary.get("event_pool_sha256"):
        raise RuntimeError("父 summary/event_pool SHA256 不一致")

    audit_manifest = json.loads(
        (parent / "audit" / "audit_manifest.json").read_text(encoding="utf-8")
    )
    immutable_set_sha = audit_manifest["immutable_artifact_set_sha256"]
    if immutable_set_sha != str(config["parent_immutable_artifact_set_sha256"]):
        raise RuntimeError("父 audit immutable artifact set SHA256 不匹配")

    review_path = Path(config["completed_review_file"])
    actual_review_sha = file_fingerprint(str(review_path)).lower()
    expected_review_sha = str(config["completed_review_sha256"]).lower()
    if actual_review_sha != expected_review_sha:
        raise RuntimeError(
            f"人审文件 SHA256 不匹配: {actual_review_sha} != {expected_review_sha}"
        )
    validation = validate(parent, review_path)
    _require_expected_counts(config, validation)
    if validation["all_human_gates_passed"]:
        raise RuntimeError("人审 gate 实际通过，拒绝使用 reject-only adjudication 配置")

    decision = config["decision"]
    if decision["terminal_status"] != "REJECTED":
        raise RuntimeError("本 adjudication 配置只允许登记用户已声明的 REJECTED")
    if validation["recommended_research_verdict"] != decision["research_verdict"]:
        raise RuntimeError("validator 建议与冻结 decision 不一致")

    rows = _read_jsonl(review_path)
    overall_counts = Counter(str(row["overall_verdict"]) for row in rows)
    failure_counts = Counter(code for row in rows for code in row.get("failure_codes", []))
    reviewer_counts = Counter(str(row["reviewer"]) for row in rows)
    parent_snapshot = _parent_snapshot(parent, immutable_set_sha)
    data_fingerprint = canonical_sha256(
        {
            "parent_event_pool_sha256": stored_pool_sha,
            "parent_snapshot": parent_snapshot,
            "completed_review_sha256": actual_review_sha,
            "decision": decision,
        }
    )
    config_sha = file_fingerprint(str(config_path))
    code = git_state(str(Path(config["repo_root"])))
    if code["dirty"]:
        raise RuntimeError("正式 adjudication 必须在 clean git worktree 上运行")

    run_id = generate_run_id(
        config["task_id"], "human-audit-reject-v1", int(config["seed"]), config_sha
    )
    run_dir = (output_root or Path(config["run_root"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    audit_summary = {
        "schema_version": config["schema_version"],
        "task_id": config["task_id"],
        "run_id": run_id,
        "parent_run": str(parent),
        "parent_event_pool_sha256": stored_pool_sha,
        "parent_machine_gate_passed": bool(parent_summary["machine_gate_passed"]),
        "parent_snapshot": parent_snapshot,
        "review_file_sha256": actual_review_sha,
        "review_record_count": len(rows),
        "overall_counts": dict(sorted(overall_counts.items())),
        "component_counts": _component_counts(rows),
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "reviewer_field_counts_as_received": dict(sorted(reviewer_counts.items())),
        "determinate_count": validation["determinate_count"],
        "audited_positive_precision": validation["precision"],
        "wilson_95_lower_bound": validation["wilson_95_lower_bound"],
        "uncertain_fraction": validation["uncertain_fraction"],
        "human_gate_checks": validation["gate_checks"],
        "all_human_gates_passed": validation["all_human_gates_passed"],
        "decision_source": "user-declared; completed review file ratified by user",
        "terminal_status": decision["terminal_status"],
        "research_verdict": decision["research_verdict"],
        "n2_authorized": False,
    }
    ended_at = utc_now()
    manifest = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "run_id": run_id,
        "command": list(sys.argv),
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_sha,
        "data_fingerprint": data_fingerprint,
        "parent_run": str(parent),
        "parent_code_commit": json.loads(
            (parent / "manifest.json").read_text(encoding="utf-8")
        )["code_commit"],
        "parent_snapshot": parent_snapshot,
        "completed_review_sha256": actual_review_sha,
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "started_at": started_at,
        "ended_at": ended_at,
        "terminal_status": decision["terminal_status"],
        "exit_reason": decision["research_verdict"],
        "n2_authorized": False,
    }
    copied_lines = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_text(str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))
    atomic_write_text(str(run_dir / "review_records.jsonl"), copied_lines)
    atomic_write_json(str(run_dir / "review_validation.json"), validation)
    atomic_write_json(str(run_dir / "audit_summary.json"), audit_summary)
    atomic_write_json(str(run_dir / "summary.json"), audit_summary)
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(audit_summary, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_text(str(run_dir / decision["terminal_status"]), decision["research_verdict"] + "\n")
    print(json.dumps({"run_dir": str(run_dir), **audit_summary}, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/n1_cutin_audit_rejection_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    adjudicate(args.config, args.output_root)


if __name__ == "__main__":
    main()
