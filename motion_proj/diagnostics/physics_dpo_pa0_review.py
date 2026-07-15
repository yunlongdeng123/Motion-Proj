"""SAP-DPO PA0 人工复核的只读聚合与不可变决策证据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from ..runtime.atomic import atomic_directory, atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json


PROTOCOL_VERSION = "sap-dpo-pa0-review-v1"
REVIEW_VALUES = frozenset({"valid", "invalid", "uncertain"})
SOURCE_FILES = (
    "manifest.json",
    "resolved.yaml",
    "summary.json",
    "machine_summary.json",
    "reviews.template.jsonl",
    "reviews.jsonl",
)


class ReviewValidationError(ValueError):
    """源 review 包不完整或不再能与原始 case 一一对应。"""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewValidationError(f"{label} 必须是 JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewValidationError(f"无法解析 {path}: {exc}") from exc
    return dict(_mapping(value, label=str(path)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewValidationError(f"无法解析 {path}:{line_no}: {exc}") from exc
        rows.append(dict(_mapping(value, label=f"{path}:{line_no}")))
    return rows


def _required_path(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise ReviewValidationError(f"缺少源文件: {path}")
    return path


def _dig(value: Mapping[str, Any], path: str, *, label: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ReviewValidationError(f"{label} 缺少字段: {path}")
        current = current[component]
    return current


def _as_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReviewValidationError(f"{label} 必须是 bool")
    return value


def _validate_review_rows(
    *,
    role: str,
    template_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    required_reviews: int,
    minimum_valid_reviews: int,
) -> dict[str, Any]:
    if len(template_rows) != required_reviews:
        raise ReviewValidationError(
            f"{role} template 行数必须为 {required_reviews}，实际为 {len(template_rows)}"
        )
    if len(review_rows) != required_reviews:
        raise ReviewValidationError(
            f"{role} review 行数必须为 {required_reviews}，实际为 {len(review_rows)}"
        )

    template_ids = [str(row.get("case_id", "")) for row in template_rows]
    review_ids = [str(row.get("case_id", "")) for row in review_rows]
    if any(not case_id for case_id in template_ids + review_ids):
        raise ReviewValidationError(f"{role} 存在空 case_id")
    if len(template_ids) != len(set(template_ids)):
        raise ReviewValidationError(f"{role} template 存在重复 case_id")
    if len(review_ids) != len(set(review_ids)):
        raise ReviewValidationError(f"{role} review 存在重复 case_id")
    if set(template_ids) != set(review_ids):
        raise ReviewValidationError(f"{role} review case_id 与 template 不一致")

    counts = {name: 0 for name in sorted(REVIEW_VALUES)}
    reviewers: set[str] = set()
    decisions: list[dict[str, Any]] = []
    for row in review_rows:
        verdict = row.get("verdict")
        if verdict not in REVIEW_VALUES:
            raise ReviewValidationError(f"{role}:{row['case_id']} verdict 无效: {verdict!r}")
        reviewer = row.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ReviewValidationError(f"{role}:{row['case_id']} 缺少 reviewer")
        counts[str(verdict)] += 1
        reviewers.add(reviewer)
        decisions.append({
            "case_id": str(row["case_id"]),
            "verdict": str(verdict),
            "reviewer": reviewer,
            "notes": row.get("notes"),
            "raw_row_sha256": sha256_json(row),
        })

    valid_count = counts["valid"]
    return {
        "required_reviews": required_reviews,
        "review_count": len(review_rows),
        "case_ids_match_template": True,
        "verdict_counts": counts,
        "decisive_count": valid_count + counts["invalid"],
        "valid_count": valid_count,
        "minimum_valid_reviews": minimum_valid_reviews,
        "review_pass": valid_count >= minimum_valid_reviews,
        "reviewer_ids": sorted(reviewers),
        "records": decisions,
    }


def _validate_p0_machine(machine: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    candidate = str(settings["candidate"])
    checks = _mapping(_dig(machine, f"machine_checks.{candidate}.checks", label="P0 machine summary"), label="P0 checks")
    required_checks = [str(item) for item in settings["required_machine_checks"]]
    check_values = {
        name: _as_bool(checks.get(name), label=f"P0 {candidate} machine check {name}")
        for name in required_checks
    }
    audit = _mapping(_dig(machine, f"generated_track_audit.{candidate}", label="P0 machine summary"), label="P0 audit")
    zero_fields = [str(item) for item in settings["required_zero_audit_fields"]]
    zero_values = {
        field: _dig(audit, field, label=f"P0 {candidate} audit")
        for field in zero_fields
    }
    for field, value in zero_values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != 0:
            raise ReviewValidationError(f"P0 {candidate} audit {field} 必须为 0，实际为 {value!r}")

    machine_pass = _as_bool(
        _dig(machine, f"machine_checks.{candidate}.machine_pass", label="P0 machine summary"),
        label=f"P0 {candidate} machine_pass",
    )
    no_future_gt = machine.get("uses_future_gt") is False
    no_adapter = machine.get("base_generation_adapter_loaded") is False
    return {
        "candidate": candidate,
        "machine_pass": machine_pass,
        "required_checks": check_values,
        "required_zero_audit_fields": zero_values,
        "uses_future_gt": machine.get("uses_future_gt"),
        "base_generation_adapter_loaded": machine.get("base_generation_adapter_loaded"),
        "machine_requirements_pass": machine_pass and all(check_values.values()) and no_future_gt and no_adapter,
    }


def _validate_e0_machine(machine: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(machine.get("decision"), label="E0 decision")
    checks = _mapping(decision.get("checks"), label="E0 decision checks")
    required_checks = [str(item) for item in settings["required_machine_checks"]]
    check_values = {
        name: _as_bool(checks.get(name), label=f"E0 machine check {name}")
        for name in required_checks
    }
    machine_pass = _as_bool(machine.get("machine_pass"), label="E0 machine_pass")
    decision_pass = _as_bool(decision.get("machine_pass"), label="E0 decision.machine_pass")
    no_future_gt = machine.get("uses_future_gt") is False
    no_fallback = machine.get("fallback_used") is False
    return {
        "machine_pass": machine_pass,
        "decision_machine_pass": decision_pass,
        "required_checks": check_values,
        "uses_future_gt": machine.get("uses_future_gt"),
        "fallback_used": machine.get("fallback_used"),
        "machine_requirements_pass": (
            machine_pass and decision_pass and all(check_values.values()) and no_future_gt and no_fallback
        ),
    }


def _source_evidence(role: str, settings: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    root = Path(str(settings["path"])).expanduser()
    files = {name: _required_path(root, name) for name in SOURCE_FILES}
    manifest = _read_json(files["manifest.json"])
    expected_run_id = str(settings["run_id"])
    if manifest.get("run_id") != expected_run_id:
        raise ReviewValidationError(
            f"{role} manifest run_id 不匹配: {manifest.get('run_id')!r} != {expected_run_id!r}"
        )
    template_rows = _read_jsonl(files["reviews.template.jsonl"])
    review_rows = _read_jsonl(files["reviews.jsonl"])
    review = _validate_review_rows(
        role=role,
        template_rows=template_rows,
        review_rows=review_rows,
        required_reviews=int(settings["required_reviews"]),
        minimum_valid_reviews=int(settings["minimum_valid_reviews"]),
    )
    machine = _read_json(files["machine_summary.json"])
    machine_result = (
        _validate_p0_machine(machine, settings)
        if role == "p0"
        else _validate_e0_machine(machine, settings)
    )
    review_bytes = files["reviews.jsonl"].read_bytes()
    source_hashes = {name: file_fingerprint(str(path)) for name, path in files.items()}
    source_hashes["reviews.jsonl"] = hashlib.sha256(review_bytes).hexdigest()
    return {
        "role": role,
        "source_run": str(root),
        "source_run_id": expected_run_id,
        "source_manifest_status": manifest.get("status"),
        "source_files_sha256": source_hashes,
        "machine": machine_result,
        "review": review,
        "pass": bool(machine_result["machine_requirements_pass"] and review["review_pass"]),
    }, review_bytes


def aggregate_pa0_reviews(
    config: Mapping[str, Any],
    *,
    work_dir: str | Path | None = None,
    command: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """只读源 review，创建一个新的 PA0 结论目录；绝不修改 P0/E0 历史 run。"""
    cfg = dict(config)
    protocol = str(cfg.get("protocol", PROTOCOL_VERSION))
    if protocol != PROTOCOL_VERSION:
        raise ReviewValidationError(f"不支持的 PA0 protocol: {protocol}")
    run_id = str(cfg["run_id"])
    target = Path(work_dir if work_dir is not None else str(cfg["work_dir"])).expanduser()
    if target.exists():
        raise FileExistsError(f"PA0 decision run 已存在: {target}")
    sources = _mapping(cfg.get("sources"), label="sources")
    if set(sources) != {"p0", "e0"}:
        raise ReviewValidationError("sources 必须恰好包含 p0 和 e0")

    p0, p0_review_bytes = _source_evidence("p0", _mapping(sources["p0"], label="sources.p0"))
    e0, e0_review_bytes = _source_evidence("e0", _mapping(sources["e0"], label="sources.e0"))
    status = "done" if p0["pass"] and e0["pass"] else "blocked"
    next_gate = "PA1-HORIZON-01" if status == "done" else "blocked: 禁止生成 preference 数据"
    resolved = json.loads(json.dumps(cfg, ensure_ascii=False, default=str))
    config_fingerprint = sha256_json(resolved)
    started_at = utc_now()
    decision = {
        "task_id": str(cfg["task_id"]),
        "protocol": protocol,
        "status": status,
        "review_provenance": str(cfg.get("review_provenance", "user_provided; reviewer IDs preserved verbatim")),
        "p0": p0,
        "e0": e0,
        "all_requirements_pass": status == "done",
        "next_gate": next_gate,
    }
    summary = {
        "status": status,
        "task_id": str(cfg["task_id"]),
        "protocol": protocol,
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "p0_valid_reviews": p0["review"]["valid_count"],
        "p0_minimum_valid_reviews": p0["review"]["minimum_valid_reviews"],
        "p0_pass": p0["pass"],
        "e0_valid_reviews": e0["review"]["valid_count"],
        "e0_minimum_valid_reviews": e0["review"]["minimum_valid_reviews"],
        "e0_pass": e0["pass"],
        "next_gate": next_gate,
        "decision_fingerprint": sha256_json(decision),
    }
    manifest = {
        "run_id": run_id,
        "task_id": str(cfg["task_id"]),
        "protocol": protocol,
        "command": list(command if command is not None else sys.argv),
        "config_fingerprint": config_fingerprint,
        "git": git_state("."),
        "environment": environment_fingerprint(),
        "status": "running",
        "started_at": started_at,
        "sources": {
            role: {
                "run_id": evidence["source_run_id"],
                "path": evidence["source_run"],
                "files_sha256": evidence["source_files_sha256"],
            }
            for role, evidence in (("p0", p0), ("e0", e0))
        },
    }

    with atomic_directory(str(target)) as tmp_dir:
        output = Path(tmp_dir)
        atomic_write_text(str(output / "resolved.yaml"), OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True))
        atomic_write_json(str(output / "manifest.json"), manifest)
        (output / "reviews.p0.jsonl").write_bytes(p0_review_bytes)
        (output / "reviews.e0.jsonl").write_bytes(e0_review_bytes)
        if file_fingerprint(str(output / "reviews.p0.jsonl")) != p0["source_files_sha256"]["reviews.jsonl"]:
            raise ReviewValidationError("复制后的 P0 review SHA256 不匹配")
        if file_fingerprint(str(output / "reviews.e0.jsonl")) != e0["source_files_sha256"]["reviews.jsonl"]:
            raise ReviewValidationError("复制后的 E0 review SHA256 不匹配")
        atomic_write_json(str(output / "review_decision.json"), decision)
        metrics = JsonlMetrics(str(output / "metrics.jsonl"))
        metrics.append(0, {"event": "p0_review_aggregated", "pass": p0["pass"], "valid_count": p0["review"]["valid_count"]})
        metrics.append(1, {"event": "e0_review_aggregated", "pass": e0["pass"], "valid_count": e0["review"]["valid_count"]})
        metrics.append(2, {"event": "pa0_decision", "status": status, "next_gate": next_gate})
        atomic_write_json(str(output / "summary.json"), summary)
        manifest.update({"status": status, "ended_at": utc_now(), "exit_reason": next_gate})
        atomic_write_json(str(output / "manifest.json"), manifest)
        atomic_write_text(str(output / "COMPLETE"), sha256_json(summary) + "\n")
    return target, summary


def _load_config(path: str) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return dict(_mapping(resolved, label="PA0 config"))


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合 SAP-DPO PA0 的 P-UNC/E0 人工复核")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()
    run_dir, summary = aggregate_pa0_reviews(_load_config(args.config), work_dir=args.work_dir)
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
