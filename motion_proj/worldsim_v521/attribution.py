"""V5.2.1 人工复核归因层与回测合同生成器。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "worldsim-v5.2.1-human-attribution-v1"
ALLOWED_GATES = {"BASE_FAILURE", "M123_ELIGIBLE", "ATTRIBUTION_UNRESOLVED"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _index_unique(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"{label} 的 {key} 重复: {value}")
        result[value] = row
    return result


def _storage_snapshot(target_path: str) -> str:
    prefix = "/root/autodl-tmp/data/"
    if not target_path.startswith(prefix) or "/images/" not in target_path:
        raise ValueError(f"无法解析冻结数据快照: {target_path}")
    return target_path[len(prefix) :].split("/images/", 1)[0]


def _module_ids(records: list[dict[str, Any]], module: str, split: str) -> list[str]:
    return [
        row["case_id"]
        for row in records
        if row["research_gate"] == "M123_ELIGIBLE"
        and row["split_role"] == split
        and module in row["module_relevance"]
    ]


def build_attribution_records(
    *,
    review_rows: list[dict[str, Any]],
    badcase_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    annotation_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """将不可变 census 与用户人工 verdict 合并成可回测归因层。"""

    review_by_id = _index_unique(review_rows, "case_id", "review cases")
    badcase_by_id = _index_unique(badcase_rows, "case_id", "badcase registry")
    matched_by_key = _index_unique(
        matched_rows,
        "_join_key",
        "matched frame registry",
    ) if matched_rows and "_join_key" in matched_rows[0] else {
        (row["scene"], row["canonical_sample_index"], row["camera"]): row
        for row in matched_rows
    }
    annotations = list(annotation_config["annotations"])
    profiles = annotation_config["metric_profiles"]

    if len(annotations) != 18:
        raise ValueError(f"人工审核分母必须为 18，实际为 {len(annotations)}")
    if {row["case_id"] for row in annotations} != set(review_by_id):
        raise ValueError("人工审核 case 集与 REVIEW_CASES.jsonl 不一致")

    records: list[dict[str, Any]] = []
    for annotation in sorted(annotations, key=lambda row: row["review_order"]):
        case_id = annotation["case_id"]
        gate = annotation["research_gate"]
        if gate not in ALLOWED_GATES:
            raise ValueError(f"未知 research_gate: {gate}")

        review = review_by_id[case_id]
        badcase = badcase_by_id[case_id]
        key = (badcase["scene"], badcase["canonical_sample_index"], badcase["camera"])
        matched = matched_by_key.get(key)
        if matched is None:
            raise ValueError(f"缺少 matched frame: {key}")
        if review["base"] != badcase["base"] or review["split_role"] != badcase["split_role"]:
            raise ValueError(f"review/badcase 身份不一致: {case_id}")
        if matched["partition"] != badcase["split_role"]:
            raise ValueError(f"split 身份不一致: {case_id}")

        review_axis = review["review_axis"]
        event_id = badcase["event_ids"].get(review_axis)
        if event_id != review["event_id"]:
            raise ValueError(f"event ID 不一致: {case_id}/{review_axis}")

        profile_name = annotation["metric_profile"]
        profile = profiles[profile_name]
        is_eligible = gate == "M123_ELIGIBLE"
        if is_eligible and not annotation["primary_modules"]:
            raise ValueError(f"eligible case 缺 primary_modules: {case_id}")
        if not is_eligible and annotation["primary_modules"]:
            raise ValueError(f"非 eligible case 不得进入 primary_modules: {case_id}")

        if is_eligible:
            use_role = "design" if badcase["split_role"] == "discovery" else "confirmation_only"
        elif gate == "BASE_FAILURE":
            use_role = "base_sentinel"
        else:
            use_role = "diagnostic_only"

        module_relevance = list(dict.fromkeys(annotation["primary_modules"] + annotation["secondary_modules"]))
        metrics = badcase["metrics"]
        record = {
            "schema_version": SCHEMA_VERSION,
            "review_order": annotation["review_order"],
            "case_id": case_id,
            "event_id": event_id,
            "base": badcase["base"],
            "dataset": badcase["dataset"],
            "dataset_split": "trainval",
            "dataset_storage_snapshot": _storage_snapshot(matched["target_path"]),
            "scene": badcase["scene"],
            "scene_index": matched["scene_index"],
            "canonical_sample_index": badcase["canonical_sample_index"],
            "frame": badcase["frame"],
            "camera": badcase["camera"],
            "unit_key": matched["unit_key"],
            "split_role": badcase["split_role"],
            "split_hash": matched["split_hash"],
            "evidence_tier": badcase["evidence_tier"],
            "review_axis": review_axis,
            "failure_axes": badcase["failure_axes"],
            "failure_class": badcase["failure_class"],
            "cross_base_status": review["cross_base_status"],
            "research_gate": gate,
            "eligible_for_primary_eval": is_eligible,
            "use_role": use_role,
            "issue_code": annotation["issue_code"],
            "visual_observation": annotation["visual_observation"],
            "suspected_causes": annotation["suspected_causes"],
            "causal_claim_status": annotation_config["causal_claim_status"],
            "primary_modules": annotation["primary_modules"],
            "secondary_modules": annotation["secondary_modules"],
            "module_relevance": module_relevance,
            "metric_profile": profile_name,
            "primary_metrics": profile["primary_metrics"],
            "non_regression_metrics": profile["non_regression_metrics"],
            "required_bridges": profile["required_bridges"],
            "success_hypothesis": annotation["success_hypothesis"],
            "baseline_metrics": {
                "global": metrics["global"],
                "actor": metrics["actor"],
                "boundary": metrics["boundary"],
            },
            "actor_context": badcase["actor_context"],
            "source": {
                "target_path": matched["target_path"],
                "target_sha256": matched["target_sha256"],
                "prediction_sha256": badcase["asset_provenance"]["prediction_sha256"],
                "dynamic_mask_sha256": badcase["asset_provenance"]["dynamic_mask_sha256"],
                "canonical_panel_path": badcase["panel_path"],
                "canonical_panel_sha256": review["panel_sha256"],
                "review_image_path": review["review_path"],
                "metric_row_sha256": review["metric_row_sha256"],
            },
        }
        records.append(record)

    orders = [row["review_order"] for row in records]
    if orders != list(range(1, 19)):
        raise ValueError(f"review_order 必须连续为 1..18: {orders}")

    gate_counts = Counter(row["research_gate"] for row in records)
    expected_counts = {"BASE_FAILURE": 9, "M123_ELIGIBLE": 8, "ATTRIBUTION_UNRESOLVED": 1}
    if dict(gate_counts) != expected_counts:
        raise ValueError(f"research gate 分母漂移: {dict(gate_counts)}")

    eligible = [row for row in records if row["eligible_for_primary_eval"]]
    if {row["base"] for row in eligible} != {"streetgs"}:
        raise ValueError("当前 M123 eligible seed 必须全部来自 StreetGS")
    if Counter(row["split_role"] for row in eligible) != {"discovery": 5, "confirmation": 3}:
        raise ValueError("M123 eligible split 必须冻结为 5 Discovery + 3 Confirmation")

    contract = {
        "schema_version": "worldsim-v5.2-backtest-contract-v1",
        "task_id": "WS-V52-R0-ATTRIBUTION-BRIDGE-01",
        "design_case_ids": [row["case_id"] for row in eligible if row["split_role"] == "discovery"],
        "confirmation_case_ids": [row["case_id"] for row in eligible if row["split_role"] == "confirmation"],
        "base_sentinel_case_ids": [row["case_id"] for row in records if row["research_gate"] == "BASE_FAILURE"],
        "diagnostic_only_case_ids": [row["case_id"] for row in records if row["research_gate"] == "ATTRIBUTION_UNRESOLVED"],
        "module_cohorts": {
            module: {
                split: _module_ids(records, module, split)
                for split in ("discovery", "confirmation")
            }
            for module in ("M1", "M3", "M2_SAFETY")
        },
        "locks": {
            "confirmation_used_for_design": False,
            "base_failure_used_for_m123_primary_eval": False,
            "manual_visual_diagnosis_is_causal_proof": False,
            "fresh_validation_test_kitti_read": False,
            "threshold_refit_on_confirmation": False,
        },
    }
    return records, contract


def build_attribution_package(
    *,
    run_id: str,
    source_commit: str,
    review_cases_path: Path,
    badcase_registry_path: Path,
    matched_frame_registry_path: Path,
    annotation_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    annotation_config = json.loads(annotation_config_path.read_text(encoding="utf-8"))
    records, contract = build_attribution_records(
        review_rows=_read_jsonl(review_cases_path),
        badcase_rows=_read_jsonl(badcase_registry_path),
        matched_rows=_read_jsonl(matched_frame_registry_path),
        annotation_config=annotation_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    contract_path = output_dir / "backtest_contract.json"
    summary_path = output_dir / "summary.json"
    status_path = output_dir / "status.json"
    _write_jsonl(cases_path, records)
    _write_json(contract_path, contract)

    summary = {
        "run_id": run_id,
        "task_id": annotation_config["task_id"],
        "outcome": "human_attribution_and_backtest_denominator_frozen",
        "total_cases": len(records),
        "gate_counts": dict(Counter(row["research_gate"] for row in records)),
        "design_case_ids": contract["design_case_ids"],
        "confirmation_case_ids": contract["confirmation_case_ids"],
        "primary_base": "streetgs",
        "dataset": "nuscenes",
        "causal_claim_status": annotation_config["causal_claim_status"],
    }
    status = {
        "run_id": run_id,
        "task_id": annotation_config["task_id"],
        "status": "done",
        "outcome": summary["outcome"],
    }
    _write_json(summary_path, summary)
    _write_json(status_path, status)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": annotation_config["task_id"],
        "source_commit": source_commit,
        "status": "done",
        "review_date": annotation_config["review_date"],
        "reviewer_role": annotation_config["reviewer_role"],
        "causal_claim_status": annotation_config["causal_claim_status"],
        "review_source_sha256": annotation_config["review_source_sha256"],
        "research_draft_sha256": annotation_config["research_draft_sha256"],
        "source_files": {
            "review_cases": {
                "canonical_path": annotation_config["canonical_inputs"]["review_cases"],
                "sha256": _sha256(review_cases_path),
            },
            "badcase_registry": {
                "canonical_path": annotation_config["canonical_inputs"]["badcase_registry"],
                "sha256": _sha256(badcase_registry_path),
            },
            "matched_frame_registry": {
                "canonical_path": annotation_config["canonical_inputs"]["matched_frame_registry"],
                "sha256": _sha256(matched_frame_registry_path),
            },
            "annotation_config": {
                "canonical_path": "configs/worldsim_v52/human_review_annotations_v1.json",
                "sha256": _sha256(annotation_config_path),
            },
        },
        "counts": {
            "total": len(records),
            "base_failure": sum(row["research_gate"] == "BASE_FAILURE" for row in records),
            "m123_eligible": sum(row["research_gate"] == "M123_ELIGIBLE" for row in records),
            "attribution_unresolved": sum(row["research_gate"] == "ATTRIBUTION_UNRESOLVED" for row in records),
            "m123_discovery": sum(row["use_role"] == "design" for row in records),
            "m123_confirmation": sum(row["use_role"] == "confirmation_only" for row in records),
        },
        "outputs": {
            "cases": {"path": "cases.jsonl", "sha256": _sha256(cases_path)},
            "backtest_contract": {"path": "backtest_contract.json", "sha256": _sha256(contract_path)},
            "summary": {"path": "summary.json", "sha256": _sha256(summary_path)},
            "status": {"path": "status.json", "sha256": _sha256(status_path)},
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
