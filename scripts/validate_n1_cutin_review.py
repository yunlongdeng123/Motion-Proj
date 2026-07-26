#!/usr/bin/env python
"""校验并汇总第四版 receiver-centric N1 人审；不代填 verdict、不启动 N2。"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.fingerprint import file_fingerprint


COMPONENT_FIELDS = (
    "subject_maneuver_verdict",
    "receiver_corridor_verdict",
    "receiver_relation_verdict",
    "temporal_persistence_verdict",
)
COMPONENT_VALUES = {"VALID", "INVALID", "UNCERTAIN"}
OVERALL_VALUES = {"TRUE_POSITIVE", "FALSE_POSITIVE", "UNCERTAIN"}
REVIEW_FIELDS = {
    *COMPONENT_FIELDS,
    "overall_verdict",
    "failure_codes",
    "reviewer",
    "notes",
}
FAILURE_CODES = {
    "SUBJECT_IDENTITY_MISMATCH",
    "SUBJECT_NO_LATERAL_MANEUVER",
    "ROUTE_CONTINUATION",
    "NORMAL_TURN",
    "MAP_MATCH_JITTER",
    "INTERPOLATION_ONLY",
    "WRONG_BRANCH",
    "OPPOSITE_OR_CROSS_TRAFFIC",
    "RECEIVER_INVALID",
    "RECEIVER_ON_SOURCE_STREAM",
    "GAP_INVALID",
    "PATH_NOT_CLEAR",
    "IDENTITY_NOT_PERSISTENT",
    "INSUFFICIENT_VISUAL_EVIDENCE",
    "OTHER",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 不是 JSON object")
        rows.append(value)
    return rows


def wilson_lower_bound(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (center - margin) / denominator


def _non_review(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in REVIEW_FIELDS}


def _validate_v1(run_dir: Path, review_file: Path) -> dict:
    if not (run_dir / "AWAITING_HUMAN_REVIEW").is_file():
        raise RuntimeError("候选 run 不处于 AWAITING_HUMAN_REVIEW")
    config = yaml.safe_load((run_dir / "resolved.yaml").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if config["human_audit"].get("review_schema") != "receiver_cutin":
        raise RuntimeError("resolved.yaml 不是 receiver_cutin 人审 schema")
    if summary.get("task_id") != "N1-EVENT-CUTIN-01":
        raise RuntimeError("summary task_id 不是第四版 N1")
    if not summary.get("formal") or not summary.get("audit_ready"):
        raise RuntimeError("只允许校验 formal 且 audit_ready 的候选 run")
    if summary.get("n2_authorized") is not False:
        raise RuntimeError("候选 run 的 N2 fail-closed 字段异常")

    audit_dir = run_dir / "audit"
    manifest = json.loads(
        (audit_dir / "audit_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != "n1-receiver-cutin-human-audit-pack-v1":
        raise RuntimeError("audit manifest schema 不匹配")
    file_hashes = manifest["immutable_file_hashes"]
    for relative, expected in file_hashes.items():
        actual = file_fingerprint(str(audit_dir / relative))
        if actual != expected:
            raise RuntimeError(f"审计证据 hash 不匹配: {relative}")
    if canonical_sha256(file_hashes) != manifest["immutable_artifact_set_sha256"]:
        raise RuntimeError("immutable_artifact_set_sha256 不匹配")
    event_pool = json.loads((run_dir / "event_pool.json").read_text(encoding="utf-8"))
    if manifest["event_pool_sha256"] != event_pool["event_pool_sha256"]:
        raise RuntimeError("audit/event_pool SHA256 链断裂")

    template = _read_jsonl(audit_dir / "review_template.jsonl")
    reviewed = _read_jsonl(review_file)
    if len(template) != len(reviewed):
        raise ValueError("review 行数与模板不一致")
    template_by_id = {row["audit_id"]: row for row in template}
    reviewed_by_id = {row["audit_id"]: row for row in reviewed}
    if len(template_by_id) != len(template) or len(reviewed_by_id) != len(reviewed):
        raise ValueError("audit_id 重复")
    if set(template_by_id) != set(reviewed_by_id):
        raise ValueError("audit_id 集合与模板不一致")

    scene_by_audit = {}
    for audit_id in sorted(template_by_id):
        evidence = json.loads(
            (audit_dir / "evidence" / f"{audit_id}.json").read_text(
                encoding="utf-8"
            )
        )
        scene_by_audit[audit_id] = evidence["scene_id"]
        source = template_by_id[audit_id]
        row = reviewed_by_id[audit_id]
        if set(row) != set(source):
            raise ValueError(f"字段集合被修改: {audit_id}")
        if _non_review(row) != _non_review(source):
            raise ValueError(f"不可变 review 引用被修改: {audit_id}")
        components = [str(row[field]).strip() for field in COMPONENT_FIELDS]
        if any(value not in COMPONENT_VALUES for value in components):
            raise ValueError(f"component verdict 非法或缺失: {audit_id}")
        overall = str(row["overall_verdict"]).strip()
        if overall not in OVERALL_VALUES:
            raise ValueError(f"overall_verdict 非法或缺失: {audit_id}")
        codes = row["failure_codes"]
        if (
            not isinstance(codes, list)
            or len(codes) != len(set(codes))
            or any(code not in FAILURE_CODES for code in codes)
        ):
            raise ValueError(f"failure_codes 非法或重复: {audit_id}")
        if not str(row["reviewer"]).strip() or not str(row["notes"]).strip():
            raise ValueError(f"reviewer/notes 未填写: {audit_id}")
        if overall == "TRUE_POSITIVE":
            if any(value != "VALID" for value in components) or codes:
                raise ValueError(
                    f"TRUE_POSITIVE 与 component/failure_codes 矛盾: {audit_id}"
                )
        elif overall == "FALSE_POSITIVE":
            if "INVALID" not in components or not codes:
                raise ValueError(
                    f"FALSE_POSITIVE 必须含 INVALID 与 failure code: {audit_id}"
                )
        elif "UNCERTAIN" not in components or "INVALID" in components:
            raise ValueError(f"UNCERTAIN 与 component verdict 矛盾: {audit_id}")

    overall_counts = Counter(row["overall_verdict"] for row in reviewed)
    tp = overall_counts["TRUE_POSITIVE"]
    fp = overall_counts["FALSE_POSITIVE"]
    uncertain = overall_counts["UNCERTAIN"]
    determinate = tp + fp
    precision = tp / determinate if determinate else None
    wilson = wilson_lower_bound(tp, determinate)
    uncertain_fraction = uncertain / len(reviewed) if reviewed else None
    tp_scenes = {
        scene_by_audit[row["audit_id"]]
        for row in reviewed
        if row["overall_verdict"] == "TRUE_POSITIVE"
    }
    gates = config["human_audit"]
    checks = {
        "machine_research_support": bool(summary["machine_gate_passed"]),
        "audit_ready": bool(summary["audit_ready"]),
        "reviewed_items": len(reviewed) >= int(gates["min_reviewed_items"]),
        "true_positive_count": tp >= int(gates["min_true_positive_count"]),
        "true_positive_scenes": len(tp_scenes)
        >= int(gates["min_true_positive_scenes"]),
        "precision": precision is not None
        and precision >= float(gates["min_precision"]),
        "wilson_95_lower_bound": wilson is not None
        and wilson >= float(gates["min_wilson_95_lower_bound"]),
        "uncertain_fraction": uncertain_fraction is not None
        and uncertain_fraction <= float(gates["max_uncertain_fraction"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": "n1-receiver-cutin-human-review-validation-v1",
        "run_dir": str(run_dir),
        "review_file": str(review_file),
        "review_file_sha256": file_fingerprint(str(review_file)),
        "reviewed_item_count": len(reviewed),
        "overall_counts": dict(sorted(overall_counts.items())),
        "determinate_count": determinate,
        "precision": precision,
        "wilson_95_lower_bound": wilson,
        "uncertain_fraction": uncertain_fraction,
        "true_positive_scene_count": len(tp_scenes),
        "true_positive_scenes": sorted(tp_scenes),
        "gate_checks": checks,
        "all_human_gates_passed": passed,
        "recommended_research_verdict": (
            "n1_receiver_cutin_human_audit_pass"
            if passed
            else "reject_n1_receiver_cutin_after_human_audit"
        ),
        "decision_note": (
            "仅校验并聚合人工填写；最终 verdict 仍由用户确认并写入独立 "
            "adjudication run。"
        ),
        "n2_authorized": False,
    }


def _validate_v2(run_dir: Path, review_file: Path, manifest: dict) -> dict:
    if not (run_dir / "AWAITING_HUMAN_REVIEW").is_file():
        raise RuntimeError("final 候选 run 不处于 AWAITING_HUMAN_REVIEW")
    config = yaml.safe_load((run_dir / "resolved.yaml").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != "receiver-centric-cutin-final-v1":
        raise RuntimeError("resolved.yaml 不是 final receiver-centric schema")
    if summary.get("task_id") != "N1-EVENT-CUTIN-FINAL-01":
        raise RuntimeError("summary task_id 不是 final N1")
    if summary.get("n2_authorized") is not False:
        raise RuntimeError("final run 的 N2 fail-closed 字段异常")
    if manifest.get("strict_event_pool_sha256") != summary.get("strict_event_pool_sha256"):
        raise RuntimeError("audit/strict event pool SHA256 链断裂")
    audit_dir = run_dir / "audit"
    hashes = manifest["immutable_file_hashes"]
    for relative, expected in hashes.items():
        actual = file_fingerprint(str(audit_dir / relative))
        if actual != expected:
            raise RuntimeError(f"审计证据 hash 不匹配: {relative}")
    if canonical_sha256(hashes) != manifest["immutable_artifact_set_sha256"]:
        raise RuntimeError("immutable_artifact_set_sha256 不匹配")

    template = _read_jsonl(audit_dir / "review_template.jsonl")
    reviewed = _read_jsonl(review_file)
    if len(template) != len(reviewed):
        raise ValueError("review 行数与模板不一致")
    template_ids = [row["audit_id"] for row in template]
    reviewed_ids = [row["audit_id"] for row in reviewed]
    if len(set(template_ids)) != len(template_ids) or len(set(reviewed_ids)) != len(reviewed_ids):
        raise ValueError("audit_id 重复")
    if reviewed_ids != template_ids:
        raise ValueError("audit_id 顺序与模板不一致")

    scene_by_audit = {}
    tiers = {}
    for source, row in zip(template, reviewed):
        audit_id = source["audit_id"]
        evidence = json.loads((audit_dir / "evidence" / f"{audit_id}.json").read_text(encoding="utf-8"))
        scene_by_audit[audit_id] = evidence["scene_id"]
        tiers[audit_id] = source["review_tier"]
        if set(row) != set(source):
            raise ValueError(f"字段集合被修改: {audit_id}")
        if _non_review(row) != _non_review(source):
            raise ValueError(f"不可变 review 引用被修改: {audit_id}")
        components = [str(row[field]).strip() for field in COMPONENT_FIELDS]
        if any(value not in COMPONENT_VALUES for value in components):
            raise ValueError(f"component verdict 非法或缺失: {audit_id}")
        overall = str(row["overall_verdict"]).strip()
        if overall not in OVERALL_VALUES:
            raise ValueError(f"overall_verdict 非法或缺失: {audit_id}")
        codes = row["failure_codes"]
        if (
            not isinstance(codes, list)
            or len(codes) != len(set(codes))
            or any(code not in FAILURE_CODES for code in codes)
        ):
            raise ValueError(f"failure_codes 非法或重复: {audit_id}")
        if not str(row["reviewer"]).strip() or not str(row["notes"]).strip():
            raise ValueError(f"reviewer/notes 未填写: {audit_id}")
        if overall == "TRUE_POSITIVE":
            if any(value != "VALID" for value in components) or codes:
                raise ValueError(f"TRUE_POSITIVE 与 component/failure_codes 矛盾: {audit_id}")
        elif overall == "FALSE_POSITIVE":
            if "INVALID" not in components or not codes:
                raise ValueError(f"FALSE_POSITIVE 必须含 INVALID 与 failure code: {audit_id}")
        elif "UNCERTAIN" not in components or "INVALID" in components:
            raise ValueError(f"UNCERTAIN 与 component verdict 矛盾: {audit_id}")

    primary = [row for row in reviewed if tiers[row["audit_id"]] == "primary_pass"]
    diagnostic = [row for row in reviewed if tiers[row["audit_id"]] == "diagnostic_abstain"]
    primary_counts = Counter(row["overall_verdict"] for row in primary)
    diagnostic_counts = Counter(row["overall_verdict"] for row in diagnostic)
    tp = primary_counts["TRUE_POSITIVE"]
    fp = primary_counts["FALSE_POSITIVE"]
    uncertain = primary_counts["UNCERTAIN"]
    determinate = tp + fp
    precision = tp / determinate if determinate else None
    wilson = wilson_lower_bound(tp, determinate)
    uncertain_fraction = uncertain / len(primary) if primary else None
    tp_scenes = {
        scene_by_audit[row["audit_id"]]
        for row in primary
        if row["overall_verdict"] == "TRUE_POSITIVE"
    }
    gates = config["human_gates"]
    pass_checks = {
        "reviewed_determinate": determinate >= int(gates["pass_min_reviewed_determinate"]),
        "true_positive_count": tp >= int(gates["pass_min_true_positive"]),
        "true_positive_scenes": len(tp_scenes) >= int(gates["pass_min_positive_scenes"]),
        "precision": precision is not None and precision >= float(gates["pass_min_precision"]),
        "wilson_95_lower_bound": wilson is not None and wilson >= float(gates["pass_min_wilson_lower_bound"]),
        "uncertain_fraction": uncertain_fraction is not None and uncertain_fraction <= float(gates["pass_max_uncertain_fraction"]),
    }
    sparse_checks = {
        "all_primary_reviewed": len(primary) == int(manifest["primary_audit_item_count"]),
        "true_positive_count": tp >= int(gates["sparse_min_true_positive"]),
        "true_positive_scenes": len(tp_scenes) >= int(gates["sparse_min_positive_scenes"]),
        "precision": precision is not None and precision >= float(gates["sparse_min_precision"]),
        "uncertain_fraction": uncertain_fraction is not None and uncertain_fraction <= float(gates["sparse_max_uncertain_fraction"]),
    }
    passed = all(pass_checks.values())
    sparse = not passed and all(sparse_checks.values())
    verdict = (
        "cutin_pool_pass"
        if passed
        else "usable_but_sparse"
        if sparse
        else "stop_nuscenes_cutin_mining"
    )
    return {
        "schema_version": "n1-receiver-cutin-human-review-validation-v2",
        "run_dir": str(run_dir),
        "review_file": str(review_file),
        "review_file_sha256": file_fingerprint(str(review_file)),
        "primary_reviewed_item_count": len(primary),
        "diagnostic_reviewed_item_count": len(diagnostic),
        "primary_overall_counts": dict(sorted(primary_counts.items())),
        "diagnostic_overall_counts": dict(sorted(diagnostic_counts.items())),
        "determinate_count": determinate,
        "precision": precision,
        "wilson_95_lower_bound": wilson,
        "uncertain_fraction": uncertain_fraction,
        "true_positive_scene_count": len(tp_scenes),
        "true_positive_scenes": sorted(tp_scenes),
        "pass_gate_checks": pass_checks,
        "sparse_gate_checks": sparse_checks,
        "all_human_gates_passed": passed,
        "usable_but_sparse": sparse,
        "recommended_research_verdict": verdict,
        "decision_note": "仅聚合人工 verdict；validator 不写终态且不启动 N2。",
        "n2_authorized": False,
    }


def validate(run_dir: Path, review_file: Path) -> dict:
    manifest_path = run_dir / "audit" / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema == "n1-receiver-cutin-human-audit-pack-v1":
        return _validate_v1(run_dir, review_file)
    if schema == "n1-receiver-cutin-human-audit-pack-v2":
        return _validate_v2(run_dir, review_file, manifest)
    raise RuntimeError(f"不支持的 audit manifest schema: {schema}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.run_dir, args.review_file)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
