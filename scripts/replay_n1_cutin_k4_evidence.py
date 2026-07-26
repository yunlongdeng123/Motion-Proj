#!/usr/bin/env python
"""重放冻结 K4 evidence 的 strict-v2 calibration regression。

这个入口只消费已经冻结且带哈希的第四轮 evidence，用来快速验证 schema、
taxonomy 和几个已知根因。它不是 nuScenes 原始场景重放，也绝不能被写成
prospective precision；完整的原始 annotation/map 重放由
``replay_n1_cutin_k4_scenes.py`` 负责。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.cutin_receiver import strict_v2_result
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return value


def _fixture_contract(fixture: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = fixture / "audit_manifest_minimal.json"
    expected_path = fixture / "expected_strict_status.yaml"
    manifest = _read_json(manifest_path)
    expected = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or not isinstance(expected.get("cases"), dict):
        raise ValueError("K4 expected_strict_status.yaml 缺少 cases")
    evidence_dir = fixture / "evidence"
    evidence_paths = sorted(evidence_dir.glob("K4-*.json"))
    expected_cases = expected["cases"]
    if {path.stem for path in evidence_paths} != set(expected_cases):
        raise ValueError("K4 evidence 文件与 expected cases 不一致")
    expected_hashes = dict(manifest.get("evidence_sha256", {}))
    if len(evidence_paths) != int(manifest.get("evidence_count", -1)):
        raise ValueError("K4 evidence_count 与 manifest 不一致")
    for path in evidence_paths:
        expected_hash = expected_hashes.get(path.name)
        if not expected_hash or file_fingerprint(str(path)) != expected_hash:
            raise ValueError(f"K4 evidence hash 不匹配: {path.name}")
    return manifest, expected


def _legacy_checks(evidence: Mapping[str, Any]) -> dict[str, bool]:
    cutin = dict(evidence.get("cutin", {}))
    topology = dict(evidence.get("topology", {}))
    subject = dict(cutin.get("subject_checks", {}))
    receiver = dict(cutin.get("receiver_checks", {}))
    post = list(cutin.get("post_keyframes", []))
    heading_errors = [
        abs(float(row["heading_error_deg"]))
        for row in post
        if isinstance(row, Mapping) and row.get("heading_error_deg") is not None
    ]
    return {
        "source_target_parallel": float(topology.get("parallel_heading_error_deg") or 0.0)
        <= 10.0,
        "raw_pre_outside": int(cutin.get("pre_center_outside_count") or 0) >= 2,
        "raw_post_inside": int(cutin.get("post_box_inside_count") or 0) >= 2,
        "lateral_convergence": bool(subject.get("lateral_convergence"))
        and bool(subject.get("lateral_convergence_consistency")),
        "post_heading_stable": bool(heading_errors)
        and max(heading_errors) <= 10.0,
        "subject_dynamic": bool(subject.get("minimum_motion_speed")),
        "receiver_dynamic": bool(cutin.get("receiver_longitudinal_speed_mps", 0.0) >= 1.0),
        "receiver_same_direction": True,
        "receiver_identity_persistent": True,
        "receiver_nearest_rear_persistent": True,
        "path_clear": bool(receiver.get("receiver_branch_excludes_subject_source")),
        "corridor_unambiguous": True,
    }


def _raw_receiver(evidence: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cutin = dict(evidence.get("cutin", {}))
    rows = [row for row in cutin.get("per_frame", []) if isinstance(row, Mapping)]
    actor_by_frame: list[int | None] = []
    switch_frames: list[int] = []
    gaps: list[float | None] = []
    for row in rows:
        receiver = row.get("receiver")
        actor = receiver.get("actor_id") if isinstance(receiver, Mapping) else None
        actor = int(actor) if actor is not None else None
        actor_by_frame.append(actor)
        if actor is not None and actor_by_frame[:-1] and any(
            prior is not None and prior != actor for prior in actor_by_frame[:-1]
        ):
            switch_frames.append(int(row.get("frame", -1)))
        gap = receiver.get("bumper_gap_m") if isinstance(receiver, Mapping) else None
        gaps.append(float(gap) if gap is not None else None)
    non_null = [actor for actor in actor_by_frame if actor is not None]
    unique = sorted(set(non_null))
    selected = unique[0] if len(unique) == 1 else None
    receiver = {
        "selected_actor_id": selected,
        "actor_id_by_frame": actor_by_frame,
        "gap_m_by_frame": gaps,
        "identity_switch_frames": switch_frames,
        "identity_persistent": len(unique) == 1,
        "nearest_rear_persistent": len(unique) == 1,
        "path_clear": True,
    }
    return receiver, [str(actor) for actor in unique]


def strict_from_frozen_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """按 raw 字段复核 legacy evidence，绝不把 legacy PASS 直接当作 v2 PASS。"""
    mode = str(evidence.get("maneuver_mode", "parallel_lane_change"))
    cutin = dict(evidence.get("cutin", {}))
    checks = _legacy_checks(evidence)
    receiver, unique_receiver_ids = _raw_receiver(evidence)
    reasons: list[str] = []
    if mode == "receiver_branch_merge":
        status = "ABSTAIN"
        reasons = ["UNSUPPORTED_BRANCH_MERGE_MODE"]
    elif len(unique_receiver_ids) > 1:
        status = "FAIL"
        reasons = ["RECEIVER_IDENTITY_SWITCH"]
        checks["receiver_identity_persistent"] = False
        checks["receiver_nearest_rear_persistent"] = False
    else:
        ordered_failures = (
            ("source_target_parallel", "SOURCE_TARGET_NOT_PARALLEL"),
            ("raw_pre_outside", "NO_RAW_LATERAL_ENTRY"),
            ("raw_post_inside", "NO_RAW_LATERAL_ENTRY"),
            ("post_heading_stable", "POST_HEADING_UNSTABLE"),
            ("subject_dynamic", "SUBJECT_NOT_DYNAMIC"),
        )
        reasons = [reason for key, reason in ordered_failures if not checks[key]]
        if reasons:
            status = "FAIL"
        elif str(cutin.get("status", "")).upper() != "PASS":
            status = "ABSTAIN"
            reasons = ["INSUFFICIENT_RAW_SUPPORT"]
        else:
            status = "PASS"
    subject = {
        "actor_id": None,
        "instance_token": dict(evidence.get("roles", {})).get("SUBJECT"),
        "source_token": dict(evidence.get("source_run", {})).get("token"),
        "target_token": dict(evidence.get("target_run", {})).get("token"),
        "pre_frames": [int(row["frame"]) for row in cutin.get("pre_keyframes", [])],
        "post_frames": [int(row["frame"]) for row in cutin.get("post_keyframes", [])],
    }
    strict = strict_v2_result(
        status=status,
        maneuver_mode=mode,
        reasons=reasons,
        checks=checks,
        subject=subject,
        receiver=receiver,
        provenance={
            "source_event_record_sha256": evidence.get("event_record_sha256"),
            "map_version": evidence.get("map_name"),
            "fixture_evidence_sha256": canonical_sha256(evidence),
        },
        extra={
            "fixture_audit_id": evidence.get("audit_id"),
            "fixture_event_id": evidence.get("event_id"),
            "fixture_crossing_frame": evidence.get("crossing_frame"),
            "fixture_receiver_ids": unique_receiver_ids,
        },
    )
    return strict


def _expected_match(case: Mapping[str, Any], strict: Mapping[str, Any]) -> tuple[bool, str]:
    allowed_statuses = set(case.get("allowed_statuses", []))
    if strict["status"] != case["expected_status"] and strict["status"] not in allowed_statuses:
        return False, "status_mismatch"
    expected_reason = case.get("expected_primary_reason")
    allowed_reasons = set(case.get("allowed_primary_reasons", []))
    actual_reason = strict.get("primary_reason")
    if (
        expected_reason is not None
        and actual_reason != expected_reason
        and actual_reason not in allowed_reasons
    ):
        return False, "primary_reason_mismatch"
    expected_exact_pass = (
        strict["status"] == case["expected_status"]
        and expected_reason is None
        and actual_reason is None
    )
    if allowed_reasons and actual_reason not in allowed_reasons and not expected_exact_pass:
        return False, "primary_reason_not_allowed"
    required_all_reasons = set(case.get("required_all_reasons", []))
    if not required_all_reasons.issubset(set(strict.get("all_reasons", []))):
        return False, "required_reason_missing"
    return True, "ok"


def replay(fixture: Path, output_root: Path, report_path: Path | None = None) -> dict[str, Any]:
    fixture = fixture.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output_root 已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    manifest, expected = _fixture_contract(fixture)
    cases: list[dict[str, Any]] = []
    records_path = output_root / "strict_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for evidence_path in sorted((fixture / "evidence").glob("K4-*.json")):
            audit_id = evidence_path.stem
            evidence = _read_json(evidence_path)
            strict = strict_from_frozen_evidence(evidence)
            case_expected = expected["cases"][audit_id]
            matched, match_reason = _expected_match(case_expected, strict)
            row = {
                "audit_id": audit_id,
                "event_id": evidence["event_id"],
                "scene_id": evidence["scene_id"],
                "human_verdict": case_expected["human_verdict"],
                "release_blocking": bool(case_expected.get("release_blocking", False)),
                "expected_status": case_expected["expected_status"],
                "allowed_statuses": case_expected.get("allowed_statuses", []),
                "expected_primary_reason": case_expected.get("expected_primary_reason"),
                "allowed_primary_reasons": case_expected.get("allowed_primary_reasons", []),
                "required_all_reasons": case_expected.get("required_all_reasons", []),
                "strict": strict,
                "matched_expectation": matched,
                "match_reason": match_reason,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            cases.append(row)
    human_fp = [row for row in cases if row["human_verdict"] == "FALSE_POSITIVE"]
    blocking_tp = [row for row in cases if row["release_blocking"]]
    report = {
        "schema_version": "n1-cutin-k4-evidence-regression-v2",
        "purpose": "calibration_regression_only_not_prospective_precision",
        "fixture": str(fixture),
        "fixture_manifest_sha256": file_fingerprint(str(fixture / "audit_manifest_minimal.json")),
        "fixture_review_sha256": manifest.get("completed_review_sha256"),
        "case_count": len(cases),
        "human_false_positive_pass_count": sum(
            row["strict"]["status"] == "PASS" for row in human_fp
        ),
        "blocking_tp_passed": all(
            row["strict"]["status"] == "PASS" and row["matched_expectation"]
            for row in blocking_tp
        ),
        "all_expectations_matched": all(row["matched_expectation"] for row in cases),
        "uses_interpolated_physics": False,
        "n2_authorized": False,
        "cases": [
            {
                "audit_id": row["audit_id"],
                "event_id": row["event_id"],
                "human_verdict": row["human_verdict"],
                "release_blocking": row["release_blocking"],
                "status": row["strict"]["status"],
                "primary_reason": row["strict"]["primary_reason"],
                "receiver_ids": row["strict"]["fixture_receiver_ids"],
                "matched_expectation": row["matched_expectation"],
                "match_reason": row["match_reason"],
            }
            for row in cases
        ],
    }
    report["passed"] = bool(
        report["all_expectations_matched"]
        and report["blocking_tp_passed"]
        and report["human_false_positive_pass_count"] == 0
    )
    destination = report_path.resolve() if report_path is not None else output_root / "K4_REGRESSION.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(destination), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = replay(args.fixture, args.output_root, args.report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
