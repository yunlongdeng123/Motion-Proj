"""PA2 两阶段人工偏好审查的抽样与 fail-closed 聚合。"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pair_scoring import DECISIVE_LABELS, wilson_lower_bound


STAGE_VERDICTS = frozenset({"a_better", "b_better", "tie", "both_invalid", "uncertain"})
QUALITY_VALUES = frozenset({"pass", "fail", "uncertain"})
MOTION_AMOUNT_VALUES = frozenset({"a_more", "b_more", "similar", "neither_moves", "uncertain"})
FAILURE_REASONS = frozenset({
    "physics_implausible", "temporal_jitter", "geometry_deformation", "identity_switch",
    "camera_motion_inconsistent", "low_motion_or_frozen", "blur_or_artifact", "other",
})


class PreferenceReviewError(RuntimeError):
    """PA2 review schema、blind mapping 或聚合门禁无效。"""


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _margin_bucket(row: Mapping[str, Any]) -> str:
    confidence = _finite(row.get("pair_confidence"))
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.25:
        return "medium"
    return "low"


def select_review_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    required: int,
    seed: int,
) -> list[dict[str, Any]]:
    """按 machine label × margin bucket 轮询抽样，组内固定随机。"""
    if required < 0:
        raise PreferenceReviewError("review required 不得为负数")
    unique: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        pair_id = str(row.get("pair_id", ""))
        if not pair_id or pair_id in unique:
            raise PreferenceReviewError(f"review pair_id 缺失或重复: {pair_id!r}")
        unique[pair_id] = row
    if len(unique) < required:
        raise PreferenceReviewError(f"review pair 不足: required={required}, available={len(unique)}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        groups[(str(row.get("global_label", "invalid")), _margin_bucket(row))].append(row)
    for index, key in enumerate(sorted(groups)):
        random.Random(int(seed) + index).shuffle(groups[key])
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < required:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < required:
                row = groups[key].pop()
                selected.append({**row, "review_margin_bucket": key[1], "review_machine_stratum": key[0]})
                progressed = True
        if not progressed:
            break
    if len(selected) != required:
        raise PreferenceReviewError("分层抽样未达到 required 数量")
    return selected


def _read_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreferenceReviewError(f"缺少 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise PreferenceReviewError(f"{label} 必须是 object list")
    return value


def _side_map(value: Any, *, field: str, allowed: frozenset[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"a", "b"}:
        raise PreferenceReviewError(f"{field} 必须严格包含 a/b")
    result = {"a": value["a"], "b": value["b"]}
    if allowed is not None and any(str(item) not in allowed for item in result.values()):
        raise PreferenceReviewError(f"{field} 含非法值")
    return result


def _validate_review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(row.get("case_id", ""))
    stage_a = str(row.get("stage_a_verdict", ""))
    stage_b = str(row.get("stage_b_verdict", ""))
    if stage_a not in STAGE_VERDICTS or stage_b not in STAGE_VERDICTS:
        raise PreferenceReviewError(f"{case_id}: stage verdict 非法")
    reviewer = str(row.get("reviewer", "")).strip()
    if not reviewer or reviewer.lower() in {"pending", "codex", "ai"}:
        raise PreferenceReviewError(f"{case_id}: reviewer 必须是真实人工标识")
    if not isinstance(row.get("notes", ""), str) or not isinstance(row.get("stage_b_change_reason", ""), str):
        raise PreferenceReviewError(f"{case_id}: notes/change_reason 必须是字符串")
    _side_map(row.get("stage_a_motion_plausibility"), field="stage_a_motion_plausibility", allowed=QUALITY_VALUES)
    _side_map(row.get("stage_a_visual_quality"), field="stage_a_visual_quality", allowed=QUALITY_VALUES)
    _side_map(row.get("stage_a_identity_consistency"), field="stage_a_identity_consistency", allowed=QUALITY_VALUES)
    if str(row.get("stage_a_motion_amount", "")) not in MOTION_AMOUNT_VALUES:
        raise PreferenceReviewError(f"{case_id}: stage_a_motion_amount 非法")
    reasons = _side_map(row.get("stage_a_failure_reasons"), field="stage_a_failure_reasons")
    for side, values in reasons.items():
        if not isinstance(values, list) or len(values) != len(set(values)) or any(str(value) not in FAILURE_REASONS for value in values):
            raise PreferenceReviewError(f"{case_id}: {side} failure reasons 非法")
    collapse = _side_map(row.get("low_motion_collapse"), field="low_motion_collapse")
    catastrophic = _side_map(row.get("catastrophic_quality_failure"), field="catastrophic_quality_failure")
    if any(not isinstance(value, bool) for value in [*collapse.values(), *catastrophic.values()]):
        raise PreferenceReviewError(f"{case_id}: collapse/quality flag 必须是 boolean")
    if stage_a != stage_b and not str(row.get("stage_b_change_reason", "")).strip():
        raise PreferenceReviewError(f"{case_id}: 阶段 B 改判必须填写 stage_b_change_reason")
    return dict(row)


def review_summary(run_dir: Path, review_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """读取完整 48-case review；不推断、不补写人工 verdict。"""
    public = _read_json_list(run_dir / "review_cases.json", "review_cases.json")
    private = _read_json_list(run_dir / "review_cases.private.json", "review_cases.private.json")
    public_ids = {str(row.get("case_id", "")) for row in public}
    private_by_id = {str(row.get("case_id", "")): row for row in private}
    if "" in public_ids or len(public_ids) != len(public) or set(private_by_id) != public_ids:
        raise PreferenceReviewError("public/private review case 映射不一致")
    reviews_path = run_dir / "reviews.jsonl"
    reviews: dict[str, dict[str, Any]] = {}
    if reviews_path.is_file():
        for raw in reviews_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            loaded = json.loads(raw)
            if not isinstance(loaded, Mapping):
                raise PreferenceReviewError("reviews.jsonl row 必须是 object")
            case_id = str(loaded.get("case_id", ""))
            if case_id not in public_ids or case_id in reviews:
                raise PreferenceReviewError(f"reviews.jsonl case_id 未知或重复: {case_id!r}")
            reviews[case_id] = _validate_review_row(loaded)
    completed = [reviews[case_id] for case_id in sorted(public_ids) if case_id in reviews]
    agreements = 0
    eligible = 0
    scorer_chosen_collapse = 0
    scorer_chosen_catastrophic = 0
    transitions: Counter[str] = Counter()
    changed = 0
    for row in completed:
        case_id = str(row["case_id"])
        private_row = private_by_id[case_id]
        mapping = private_row.get("blind_mapping")
        if not isinstance(mapping, Mapping) or set(mapping) != {"A", "B"}:
            raise PreferenceReviewError(f"{case_id}: blind_mapping 非法")
        machine_label = str(private_row.get("machine_global_label", ""))
        winner_id = private_row.get("machine_winner_id")
        winner_side = next((side.lower() for side, candidate in mapping.items() if candidate == winner_id), None)
        stage_b = str(row["stage_b_verdict"])
        if machine_label in DECISIVE_LABELS and winner_side in {"a", "b"}:
            collapse = _side_map(row["low_motion_collapse"], field="low_motion_collapse")
            catastrophic = _side_map(row["catastrophic_quality_failure"], field="catastrophic_quality_failure")
            scorer_chosen_collapse += int(bool(collapse[winner_side]))
            scorer_chosen_catastrophic += int(bool(catastrophic[winner_side]))
            if stage_b in {"a_better", "b_better"}:
                eligible += 1
                agreements += int(stage_b == f"{winner_side}_better")
        stage_a = str(row["stage_a_verdict"])
        transitions[f"{stage_a}->{stage_b}"] += 1
        changed += int(stage_a != stage_b)
    agreement_rate = agreements / eligible if eligible else None
    wilson = wilson_lower_bound(agreements, eligible)
    required = int(review_cfg["required_cases"])
    complete_enough = len(completed) >= required
    decisive_enough = eligible >= int(review_cfg["minimum_decisive_agreement_cases"])
    checks = {
        "required_cases_complete": complete_enough,
        "minimum_decisive_agreement_cases": decisive_enough,
        "minimum_agreement_rate": agreement_rate is not None and agreement_rate >= float(review_cfg["minimum_agreement_rate"]),
        "minimum_wilson_lower_bound": wilson is not None and wilson > float(review_cfg["minimum_wilson_lower_bound"]),
        "maximum_low_motion_collapse": scorer_chosen_collapse <= int(review_cfg["maximum_low_motion_collapse"]),
        "maximum_catastrophic_quality_failures": scorer_chosen_catastrophic <= int(review_cfg["maximum_catastrophic_quality_failures"]),
        "stage_changes_explained": True,
    }
    passed = all(checks.values())
    if not complete_enough:
        status = "awaiting_reviews"
    elif not decisive_enough:
        status = "needs_more_reviews"
    else:
        status = "pass" if passed else "rejected"
    return {
        "required_cases": required,
        "completed_cases": len(completed),
        "decisive_agreement_cases": eligible,
        "agreements": agreements,
        "agreement_rate": agreement_rate,
        "wilson_lower_bound_95": wilson,
        "stage_verdict_changes": changed,
        "stage_transition_counts": dict(sorted(transitions.items())),
        "scorer_chosen_low_motion_collapse": scorer_chosen_collapse,
        "scorer_chosen_catastrophic_quality_failures": scorer_chosen_catastrophic,
        "checks": checks,
        "pass": passed,
        "status": status,
    }
