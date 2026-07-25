#!/usr/bin/env python
"""校验并不可变登记 N1-EVENT-FULL-01 的用户人审 reject。

本工具只聚合用户提供的 verdict，不生成、推断或代填任何人工判断，也不改写父 run。
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

REVIEW_FIELDS = {"verdict", "reviewer", "notes"}


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 不是 JSON object")
        rows.append(value)
    return rows


def _without_review_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in REVIEW_FIELDS}


def _unique_terminal(parent: Path) -> str:
    markers = [
        name
        for name in ("COMPLETE", "REJECTED", "FAILED", "AWAITING_HUMAN_REVIEW")
        if (parent / name).is_file()
    ]
    if markers != ["COMPLETE"]:
        raise RuntimeError(f"父 run 终态不唯一或不是 COMPLETE: {markers}")
    return markers[0]


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

    template_path = Path(config["original_review_template"])
    completed_path = Path(config["completed_review_file"])
    expected_completed_sha = str(config["completed_review_sha256"]).lower()
    actual_completed_sha = file_fingerprint(str(completed_path)).lower()
    if actual_completed_sha != expected_completed_sha:
        raise RuntimeError(
            f"人审文件 SHA256 不匹配: {actual_completed_sha} != {expected_completed_sha}"
        )
    template = _read_jsonl(template_path)
    completed = _read_jsonl(completed_path)
    if len(template) != len(completed):
        raise RuntimeError("人审模板与完成文件行数不一致")
    template_by_id = {row["event_id"]: row for row in template}
    completed_by_id = {row["event_id"]: row for row in completed}
    if len(template_by_id) != len(template) or len(completed_by_id) != len(completed):
        raise RuntimeError("人审文件存在重复 event_id")
    expected_ids = {
        row["event_id"] for row in event_pool["evaluation"]["positives"]
    }
    if set(template_by_id) != expected_ids or set(completed_by_id) != expected_ids:
        raise RuntimeError("人审 event_id 集合与父 event pool positive 不一致")

    allowed = set(config["allowed_verdicts"])
    for event_id in sorted(expected_ids):
        source = template_by_id[event_id]
        reviewed = completed_by_id[event_id]
        if _without_review_fields(source) != _without_review_fields(reviewed):
            raise RuntimeError(f"人审文件改动了非 review 字段: {event_id}")
        verdict = str(reviewed.get("verdict", "")).strip()
        reviewer = str(reviewed.get("reviewer", "")).strip()
        notes = str(reviewed.get("notes", "")).strip()
        if verdict not in allowed:
            raise RuntimeError(f"非法或缺失 verdict: {event_id}={verdict!r}")
        if not reviewer or not notes:
            raise RuntimeError(f"reviewer/notes 未完整填写: {event_id}")

    verdict_counts = Counter(row["verdict"] for row in completed)
    reviewer_counts = Counter(row["reviewer"] for row in completed)
    determinate = verdict_counts["TRUE_POSITIVE"] + verdict_counts["FALSE_POSITIVE"]
    precision = (
        verdict_counts["TRUE_POSITIVE"] / determinate if determinate else None
    )
    decision = config["decision"]
    if decision["terminal_status"] != "REJECTED":
        raise RuntimeError("本 adjudication 配置只允许登记用户已声明的 REJECTED")

    config_sha = file_fingerprint(str(config_path))
    code = git_state(str(Path(config["repo_root"])))
    if code["dirty"]:
        raise RuntimeError("正式 adjudication 必须在 clean git worktree 上运行")
    run_id = generate_run_id(
        config["task_id"], "human-audit-reject-v1", int(config["seed"]), config_sha
    )
    run_dir = (output_root or Path(config["run_root"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    audit_summary = {
        "schema_version": config["schema_version"],
        "task_id": config["task_id"],
        "run_id": run_id,
        "parent_run": str(parent),
        "parent_event_pool_sha256": stored_pool_sha,
        "review_record_count": len(completed),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "reviewer_field_counts_as_received": dict(sorted(reviewer_counts.items())),
        "determinate_count": determinate,
        "audited_positive_precision": precision,
        "decision_source": "user-declared; file contents ratified by user",
        "threshold_note": (
            "父审计提示词未预注册聚合阈值；本 run 不补造阈值，"
            "只登记用户明确声明的第二次 N1 REJECTED。"
        ),
        "terminal_status": decision["terminal_status"],
        "research_verdict": decision["research_verdict"],
        "n2_authorized": False,
    }
    data_fingerprint = canonical_sha256(
        {
            "parent_event_pool_sha256": stored_pool_sha,
            "template_sha256": file_fingerprint(str(template_path)),
            "completed_review_sha256": actual_completed_sha,
            "decision": decision,
        }
    )
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
        "parent_code_dirty": json.loads(
            (parent / "manifest.json").read_text(encoding="utf-8")
        ).get("code_dirty"),
        "completed_review_sha256": actual_completed_sha,
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "started_at": utc_now(),
        "ended_at": utc_now(),
        "terminal_status": decision["terminal_status"],
        "exit_reason": decision["research_verdict"],
    }
    copied_lines = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in completed
    )
    atomic_write_text(str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))
    atomic_write_text(str(run_dir / "review_records.jsonl"), copied_lines)
    atomic_write_json(str(run_dir / "audit_summary.json"), audit_summary)
    atomic_write_json(str(run_dir / "summary.json"), audit_summary)
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(audit_summary, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_text(
        str(run_dir / decision["terminal_status"]),
        decision["research_verdict"] + "\n",
    )
    print(json.dumps({"run_dir": str(run_dir), **audit_summary}, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/n1_fulldomain_audit_rejection_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    adjudicate(args.config, args.output_root)


if __name__ == "__main__":
    main()
