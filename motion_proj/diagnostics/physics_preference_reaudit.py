"""PA2-UPO-03B：在冻结 sibling RGB 上重建 common-support selective oracle。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..config import config_fingerprint, load_config, save_resolved_config
from ..preference.calibration import (
    PRIMARY_COMPONENTS,
    holm_adjust,
    measurement_ropes,
    paired_cluster_block_bootstrap,
    scene_hash_split,
    split_conformal_threshold,
)
from ..preference.common_support import CommonSupportWindow, build_common_support
from ..preference.paired_tracks import (
    PairModeRAFTTracker,
    PairedQuerySet,
    RawTrackObservation,
)
from ..preference.residual_motion import MotionComponentEvidence, compute_motion_component_evidence
from ..preference.selective_order import (
    build_condition_partial_order,
    decide_selective_relation,
    quality_comparability,
    video_quality_metrics,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything


class PreferenceReauditError(RuntimeError):
    """PA2-UPO provenance、协议或门禁失败。"""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreferenceReauditError(f"缺少 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreferenceReauditError(f"{label} 必须是 object")
    return value


def _read_json_list(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreferenceReauditError(f"缺少 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise PreferenceReauditError(f"{label} 必须是 object list")
    return [dict(row) for row in value]


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreferenceReauditError(f"缺少 {label}: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PreferenceReauditError(f"{label} JSONL row 必须是 object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(
        str(path),
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def _assert_no_future_gt(value: Any, *, label: str) -> None:
    """递归拒绝显式 future-GT 标志；普通 future-video RGB 不属于 GT 监督。"""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key == "uses_future_gt" and bool(child):
                raise PreferenceReauditError(f"{label} 检测到 uses_future_gt=true")
            if key in {"future_gt", "future_ground_truth", "future_ego_pose"}:
                raise PreferenceReauditError(f"{label} 检测到禁止字段 {raw_key}")
            if key == "generated_geometry_mode" and str(child) == "gt_ego_debug":
                raise PreferenceReauditError(f"{label} 检测到 gt_ego_debug")
            _assert_no_future_gt(child, label=label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_future_gt(child, label=label)


def _source_fingerprints(source: Path) -> dict[str, str]:
    names = (
        "manifest.json", "summary.json", "conditions.jsonl", "candidates.jsonl",
        "source_index.jsonl", "preferences.jsonl", "segments.jsonl",
        "candidate_diagnostics.jsonl", "reviews.jsonl", "review_cases.private.json",
    )
    result = {}
    for name in names:
        path = source / name
        if path.is_file():
            result[name] = file_fingerprint(str(path))
    return result


def _ensure_source_immutable(source: Path, expected: Mapping[str, str]) -> None:
    actual = _source_fingerprints(source)
    if dict(expected) != actual:
        changed = sorted(set(expected) | set(actual))
        changed = [name for name in changed if expected.get(name) != actual.get(name)]
        raise PreferenceReauditError(f"历史 PA2 artifact 被修改: {changed}")


def _source_inputs(cfg: Any) -> dict[str, Any]:
    source = Path(str(cfg.upo.source_run))
    summary = _read_json(source / "summary.json", label="source summary")
    manifest = _read_json(source / "manifest.json", label="source manifest")
    conditions = _read_jsonl(source / "conditions.jsonl", label="conditions")
    candidates = _read_jsonl(source / "candidates.jsonl", label="candidates")
    source_index = _read_jsonl(source / "source_index.jsonl", label="source index")
    private_reviews = _read_json_list(source / "review_cases.private.json", label="private review cases")
    reviews = _read_jsonl(source / "reviews.jsonl", label="human reviews")
    if str(summary.get("run_id")) != str(cfg.upo.expected_source_run_id):
        raise PreferenceReauditError("source run_id 与冻结配置不一致")
    if int(summary.get("condition_count", -1)) != int(cfg.upo.expected_condition_count):
        raise PreferenceReauditError("source condition 数与冻结配置不一致")
    if str(summary.get("status")) != "needs_more_reviews" or len(reviews) != 48:
        raise PreferenceReauditError("source 必须保留完成 48 review 后的 rejected recipe 状态")
    if len(conditions) != int(cfg.upo.expected_condition_count):
        raise PreferenceReauditError("conditions 数量不完整")
    if len(candidates) != len(conditions) * 5:
        raise PreferenceReauditError("core candidates 必须是每 condition 1 Base + 4 sibling")
    _assert_no_future_gt(summary, label="source summary")
    _assert_no_future_gt(manifest, label="source manifest")
    _assert_no_future_gt(conditions, label="conditions")
    _assert_no_future_gt(candidates, label="candidates")
    return {
        "source": source,
        "summary": summary,
        "manifest": manifest,
        "conditions": conditions,
        "candidates": candidates,
        "source_index": source_index,
        "private_reviews": private_reviews,
        "reviews": reviews,
        "fingerprints": _source_fingerprints(source),
    }


def _core_candidates(
    conditions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in candidates:
        row = dict(raw)
        grouped.setdefault(str(row["condition_id"]), []).append(row)
    result = {}
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        rows = grouped.get(condition_id, [])
        base = [row for row in rows if str(row.get("candidate_role")) == "base_guard"]
        siblings = [row for row in rows if str(row.get("branch_family")) == "common_prefix"]
        if len(base) != 1 or len(siblings) != 4 or len(rows) != 5:
            raise PreferenceReauditError(f"{condition_id} 必须恰含 1 Base + 4 common-prefix siblings")
        prefix_hashes = {str(row.get("prefix_trace_hash")) for row in siblings}
        if len(prefix_hashes) != 1:
            raise PreferenceReauditError(f"{condition_id} sibling prefix 不一致")
        result[condition_id] = {
            "condition": dict(condition),
            "base": base[0],
            "siblings": sorted(siblings, key=lambda row: str(row["candidate_id"])),
        }
    return result


def _review_partitions(inputs: Mapping[str, Any], grouped: Mapping[str, Mapping[str, Any]], cfg: Any) -> dict[str, Any]:
    review_by_id = {str(row["case_id"]): dict(row) for row in inputs["reviews"]}
    conditions = {str(row["condition_id"]): dict(row) for row in inputs["conditions"]}
    p1_rows = []
    for private in inputs["private_reviews"]:
        if str(private.get("constructor")) != "P1-common-prefix":
            continue
        case_id = str(private["case_id"])
        review = review_by_id.get(case_id)
        if review is None:
            raise PreferenceReauditError(f"P1 review 缺失: {case_id}")
        condition_id = str(private["condition_id"])
        if condition_id not in grouped or condition_id not in conditions:
            raise PreferenceReauditError(f"P1 review condition 不在 source: {condition_id}")
        mapping = dict(private["blind_mapping"])
        candidate_a, candidate_b = str(mapping["A"]), str(mapping["B"])
        sibling_ids = {str(row["candidate_id"]) for row in grouped[condition_id]["siblings"]}
        if candidate_a not in sibling_ids or candidate_b not in sibling_ids:
            raise PreferenceReauditError(f"P1 review mapping 不是冻结 sibling: {case_id}")
        p1_rows.append({
            "case_id": case_id,
            "condition_id": condition_id,
            "scene_id": str(conditions[condition_id]["scene_id"]),
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "human_verdict": str(review["stage_b_verdict"]),
        })
    if len(p1_rows) != 24:
        raise PreferenceReauditError("冻结 P1 review 必须是 24 条")
    ties = [row for row in p1_rows if row["human_verdict"] == "tie"]
    uncertain = [row for row in p1_rows if row["human_verdict"] == "uncertain"]
    invalid = [row for row in p1_rows if row["human_verdict"] == "both_invalid"]
    if (len(ties), len(uncertain), len(invalid)) != (22, 1, 1):
        raise PreferenceReauditError("P1 review verdict 分布不等于冻结的 22/1/1")
    split = scene_hash_split(
        ties,
        calibration_count=int(cfg.upo.calibration.calibration_ties),
        salt=str(cfg.upo.calibration.scene_hash_salt),
    )
    by_case = {row["case_id"]: row for row in ties}
    split["calibration_cases"] = [by_case[case_id] for case_id in split["calibration_case_ids"]]
    split["holdout_cases"] = [by_case[case_id] for case_id in split["holdout_case_ids"]]
    split["uncertain_cases"] = uncertain
    split["both_invalid_cases"] = invalid
    split["reviewed_condition_ids"] = sorted({row["condition_id"] for row in p1_rows})
    return split


def _decode_video(path: Path, *, expected_frames: int) -> torch.Tensor:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise PreferenceReauditError("reaudit 需要 OpenCV 与 NumPy") from exc
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != expected_frames:
        raise PreferenceReauditError(f"video 不是 {expected_frames} 帧: {path}")
    array = np.stack(frames)
    return torch.from_numpy(array).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def _stable_seed(base: int, *parts: Any) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int(base) + int(hashlib.sha256(payload).hexdigest()[:8], 16) % 1_000_000_000


def _edge_id(condition_id: str, candidate_a: str, candidate_b: str) -> str:
    ordered = sorted((candidate_a, candidate_b))
    digest = hashlib.sha256(f"{condition_id}\0{ordered[0]}\0{ordered[1]}".encode("utf-8")).hexdigest()
    return f"upo-edge-{digest}"


def _lite_evidence(evidence: MotionComponentEvidence) -> SimpleNamespace:
    return SimpleNamespace(
        valid=evidence.valid,
        reason=evidence.reason,
        camera_distance_px=evidence.camera_distance_px,
        activity_a=dict(evidence.activity_a),
        activity_b=dict(evidence.activity_b),
    )


def _bootstrap_context(
    *,
    condition_id: str,
    edge_id: str,
    support: CommonSupportWindow,
    evidence: MotionComponentEvidence,
    cfg: Any,
) -> dict[str, Any]:
    intervals = {}
    dynamic_clusters = support.spatial_cluster_ids[support.dynamic_mask]
    for component in PRIMARY_COMPONENTS:
        primary_seed = _stable_seed(int(cfg.seed), condition_id, edge_id, support.start_frame, component)
        primary = paired_cluster_block_bootstrap(
            evidence.differences[component],
            dynamic_clusters,
            component=component,
            samples=int(cfg.upo.bootstrap.samples),
            seed=primary_seed,
            confidence=float(cfg.upo.bootstrap.confidence),
            temporal_block=int(cfg.upo.bootstrap.temporal_block),
        )
        secondary = paired_cluster_block_bootstrap(
            evidence.differences[component],
            dynamic_clusters,
            component=component,
            samples=int(cfg.upo.bootstrap.secondary_samples),
            seed=primary_seed + int(cfg.upo.bootstrap.secondary_seed_offset),
            confidence=float(cfg.upo.bootstrap.confidence),
            temporal_block=int(cfg.upo.bootstrap.temporal_block),
        )
        intervals[component] = primary.to_record() | {"secondary": secondary.to_record()}
    return {
        "condition_id": condition_id,
        "edge_id": edge_id,
        "support": support,
        "evidence": _lite_evidence(evidence),
        "intervals": intervals,
    }


def _apply_holm(contexts: list[dict[str, Any]], cfg: Any) -> None:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for context in contexts:
        for component in PRIMARY_COMPONENTS:
            groups.setdefault((context["support"].start_frame, component), []).append(context)
    alpha = float(cfg.upo.bootstrap.familywise_alpha)
    for (_, component), rows in groups.items():
        primary = holm_adjust(
            {row["edge_id"]: row["intervals"][component]["p_value_two_sided"] for row in rows}, alpha=alpha
        )
        secondary = holm_adjust(
            {row["edge_id"]: row["intervals"][component]["secondary"]["p_value_two_sided"] for row in rows},
            alpha=alpha,
        )
        for row in rows:
            edge = row["edge_id"]
            first = row["intervals"][component]
            second = first["secondary"]
            primary_direction = (
                "positive" if first["lower"] is not None and float(first["lower"]) > 0
                else "negative" if first["upper"] is not None and float(first["upper"]) < 0
                else "none"
            )
            secondary_direction = (
                "positive" if second["lower"] is not None and float(second["lower"]) > 0
                else "negative" if second["upper"] is not None and float(second["upper"]) < 0
                else "none"
            )
            stable = primary_direction != "none" and primary_direction == secondary_direction
            first["holm"] = primary[edge]
            first["secondary_holm"] = secondary[edge]
            first["bootstrap_direction_stable"] = stable
            first["holm_significant"] = bool(
                primary[edge]["significant"] and secondary[edge]["significant"] and stable
            )


def _context_quality(
    quality_by_candidate: Mapping[str, Mapping[str, Any]],
    support: CommonSupportWindow,
    cfg: Any,
) -> dict[str, Any]:
    return quality_comparability(
        quality_by_candidate[support.candidate_a],
        quality_by_candidate[support.candidate_b],
        cfg.upo.quality,
    )


def _interval_view(context: Mapping[str, Any], *, secondary: bool = False) -> dict[str, dict[str, Any]]:
    result = {}
    for component in PRIMARY_COMPONENTS:
        row = context["intervals"][component]
        selected = row["secondary"] if secondary else row
        result[component] = dict(selected) | {
            "holm_significant": (
                bool(row["secondary_holm"]["significant"] and row["bootstrap_direction_stable"])
                if secondary else bool(row["holm_significant"])
            )
        }
    return result


def _decide_context(
    context: Mapping[str, Any],
    *,
    ropes: Mapping[str, float],
    threshold: float,
    cfg: Any,
    secondary: bool = False,
) -> dict[str, Any]:
    return decide_selective_relation(
        condition_id=str(context["condition_id"]),
        edge_id=str(context["edge_id"]),
        support=context["support"],
        evidence=context["evidence"],
        intervals=_interval_view(context, secondary=secondary),
        ropes=ropes,
        strict_threshold=threshold,
        quality=context["quality"],
        settings=cfg.upo.relation,
    )


def _case_contexts(case: Mapping[str, Any], contexts_by_condition: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    pair = {str(case["candidate_a"]), str(case["candidate_b"])}
    return [
        context for context in contexts_by_condition.get(str(case["condition_id"]), [])
        if {context["support"].candidate_a, context["support"].candidate_b} == pair
    ]


def _calibration_case_score(
    case: Mapping[str, Any],
    contexts_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ropes: Mapping[str, float],
    cfg: Any,
    secondary: bool = False,
) -> dict[str, Any]:
    contexts = _case_contexts(case, contexts_by_condition)
    comparable = 0
    scores = []
    for context in contexts:
        support = context["support"]
        evidence = context["evidence"]
        quality = context["quality"]
        gate = (
            support.valid
            and evidence.valid
            and evidence.camera_distance_px is not None
            and float(evidence.camera_distance_px) <= float(cfg.upo.relation.maximum_camera_distance_px)
            and bool(quality.get("comparable"))
        )
        comparable += int(gate)
        relation = _decide_context(
            context, ropes=ropes, threshold=0.0, cfg=cfg, secondary=secondary
        )
        if relation["relation"] == "strict" and relation["strict_score"] is not None:
            scores.append(float(relation["strict_score"]))
    return {
        "case_id": str(case["case_id"]),
        "condition_id": str(case["condition_id"]),
        "comparable_windows": comparable,
        "score": max(scores) if scores else 0.0,
    }


def _measurement_perturbations(frames: torch.Tensor) -> dict[str, torch.Tensor]:
    unit = frames.add(1.0).mul(0.5).clamp(0, 1)
    # 保留 0.01/0.99 saturation 分类，只施加不会跨越 quality gate 的轻微亮度压缩。
    photometric = unit.mul(0.99).add(0.005).clamp(0, 1).mul(2.0).sub(1.0)
    height, width = frames.shape[-2:]
    smaller = F.interpolate(frames, size=(max(8, int(height * 0.875)), max(8, int(width * 0.875))), mode="bilinear", align_corners=False)
    resized = F.interpolate(smaller, size=(height, width), mode="bilinear", align_corners=False)
    return {"identical_rerun": frames.clone(), "photometric": photometric, "resize_roundtrip": resized}


def _observation_exactness(
    reference: RawTrackObservation,
    repeated: RawTrackObservation,
) -> dict[str, Any]:
    """Identical rerun 必须逐位复现；该检查独立于 measurement ROPE。"""
    visibility_exact = torch.equal(reference.raw_visibility, repeated.raw_visibility)
    points_exact = torch.allclose(
        reference.raw_points, repeated.raw_points, rtol=0.0, atol=0.0, equal_nan=True
    )
    confidence_exact = torch.allclose(
        reference.raw_confidence, repeated.raw_confidence, rtol=0.0, atol=0.0, equal_nan=True
    )
    fb_exact = torch.allclose(
        reference.forward_backward_error,
        repeated.forward_backward_error,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    def maximum_delta(left: torch.Tensor, right: torch.Tensor) -> float | None:
        finite = torch.isfinite(left) & torch.isfinite(right)
        return float((left[finite] - right[finite]).abs().max()) if bool(finite.any()) else None

    return {
        "exact": bool(visibility_exact and points_exact and confidence_exact and fb_exact),
        "visibility_exact": visibility_exact,
        "points_exact": points_exact,
        "confidence_exact": confidence_exact,
        "forward_backward_error_exact": fb_exact,
        "maximum_point_delta": maximum_delta(reference.raw_points, repeated.raw_points),
        "maximum_confidence_delta": maximum_delta(reference.raw_confidence, repeated.raw_confidence),
        "maximum_forward_backward_error_delta": maximum_delta(
            reference.forward_backward_error, repeated.forward_backward_error
        ),
    }


def _measurement_window_eligible(
    support: CommonSupportWindow,
    evidence: MotionComponentEvidence,
    quality: Mapping[str, Any],
    cfg: Any,
) -> tuple[bool, list[str]]:
    """ROPE 只描述在完整 comparability gate 内观测到的 benign noise。"""
    reasons = []
    if not support.valid:
        reasons.append("support_invalid")
    if not evidence.valid:
        reasons.append(evidence.reason or "evidence_invalid")
    if evidence.camera_distance_px is None:
        reasons.append("camera_distance_invalid")
    elif evidence.camera_distance_px > float(cfg.upo.relation.maximum_camera_distance_px):
        reasons.append("camera_mismatch")
    if not bool(quality.get("comparable")):
        reasons.append("quality_mismatch")
    return not reasons, list(dict.fromkeys(reasons))


def _jitter_query_set(query_set: PairedQuerySet, *, amplitude: float) -> PairedQuerySet:
    offsets = []
    for index in range(len(query_set.query_ids)):
        sx = -1.0 if index % 2 else 1.0
        sy = -1.0 if (index // 2) % 2 else 1.0
        offsets.append([sx * amplitude, sy * amplitude])
    return PairedQuerySet(
        points=query_set.points + torch.tensor(offsets, dtype=query_set.points.dtype),
        query_ids=query_set.query_ids,
        strata=query_set.strata,
        selection_scores=query_set.selection_scores,
        valid=query_set.valid,
        diagnostics=dict(query_set.diagnostics) | {"measurement_query_jitter_px": amplitude},
    )


def _remap_observation_query_hash(observation: RawTrackObservation, query_set: PairedQuerySet) -> RawTrackObservation:
    return RawTrackObservation(
        candidate_id=observation.candidate_id,
        query_set_hash=query_set.query_set_hash,
        raw_points=observation.raw_points,
        raw_visibility=observation.raw_visibility,
        raw_confidence=observation.raw_confidence,
        forward_backward_error=observation.forward_backward_error,
        optional_smoothed_points=observation.optional_smoothed_points,
        diagnostics=dict(observation.diagnostics) | {"measurement_query_hash_remapped": True},
    )


def _collect_measurement_differences(
    *,
    condition_id: str,
    query_set: PairedQuerySet,
    base_observation: RawTrackObservation,
    base_frames: torch.Tensor,
    tracker: PairModeRAFTTracker,
    cfg: Any,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    values = {component: [] for component in PRIMARY_COMPONENTS}
    records = []
    variants: dict[str, tuple[RawTrackObservation, torch.Tensor]] = {}
    for name, frames in _measurement_perturbations(base_frames).items():
        variants[name] = tracker.track_fixed_queries(
            candidate_id=f"measurement-{condition_id}-{name}", frames=frames, query_set=query_set
        ), frames
    jittered = _jitter_query_set(query_set, amplitude=float(cfg.upo.measurement.query_jitter_px))
    jittered_observation = tracker.track_fixed_queries(
        candidate_id=f"measurement-{condition_id}-query-jitter", frames=base_frames, query_set=jittered
    )
    variants["query_jitter"] = _remap_observation_query_hash(jittered_observation, query_set), base_frames

    base_quality = video_quality_metrics(base_frames)
    for name, (variant, variant_frames) in variants.items():
        quality = quality_comparability(
            base_quality, video_quality_metrics(variant_frames), cfg.upo.quality
        )
        exactness = (
            _observation_exactness(base_observation, variant)
            if name == "identical_rerun" else None
        )
        windows = build_common_support(
            query_set, base_observation, variant, cfg.upo.support,
            window_starts=tuple(int(value) for value in cfg.upo.windows.starts),
            window_length=int(cfg.upo.windows.length),
        )
        valid_windows = 0
        invalid_reasons: dict[str, int] = {}
        component_counts = {component: 0 for component in PRIMARY_COMPONENTS}
        for window in windows:
            evidence = compute_motion_component_evidence(
                query_set, base_observation, variant, window, cfg.upo.motion,
                image_hw=tuple(int(value) for value in base_frames.shape[-2:]),
            )
            eligible, reasons = _measurement_window_eligible(window, evidence, quality, cfg)
            valid_windows += int(eligible)
            for reason in reasons:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            if not eligible:
                continue
            for component in PRIMARY_COMPONENTS:
                finite = evidence.differences[component][torch.isfinite(evidence.differences[component])]
                values[component].extend(abs(float(value)) for value in finite.tolist())
                component_counts[component] += int(finite.numel())
        records.append({
            "condition_id": condition_id,
            "perturbation": name,
            "valid_windows": valid_windows,
            "component_counts": component_counts,
            "invalid_window_reasons": invalid_reasons,
            "quality_comparable": bool(quality["comparable"]),
            "quality_reasons": list(quality["reasons"]),
            "identical_rerun_exactness": exactness,
            "uses_future_gt": False,
        })
    return values, records


def _merge_measurement_values(target: dict[str, list[float]], source: Mapping[str, Sequence[float]]) -> None:
    for component in PRIMARY_COMPONENTS:
        target[component].extend(float(value) for value in source.get(component, []))


def _clone_attack(
    observation: RawTrackObservation,
    *,
    candidate_id: str,
    points: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
) -> RawTrackObservation:
    visible = observation.raw_visibility.clone() if visibility is None else visibility.clone()
    raw_points = observation.raw_points.clone() if points is None else points.clone()
    raw_points[~visible] = float("nan")
    confidence = observation.raw_confidence.clone()
    confidence[~visible] = 0.0
    fb = observation.forward_backward_error.clone()
    fb[~visible] = float("nan")
    return RawTrackObservation(
        candidate_id=candidate_id,
        query_set_hash=observation.query_set_hash,
        raw_points=raw_points,
        raw_visibility=visible,
        raw_confidence=confidence,
        forward_backward_error=fb,
    )


def _stress_relation(
    query_set: PairedQuerySet,
    attack: RawTrackObservation,
    reference: RawTrackObservation,
    quality_attack: Mapping[str, Any],
    quality_reference: Mapping[str, Any],
    ropes: Mapping[str, float],
    threshold: float,
    cfg: Any,
    *,
    image_hw: tuple[int, int],
) -> list[dict[str, Any]]:
    rows = []
    for support in build_common_support(
        query_set, attack, reference, cfg.upo.support,
        window_starts=(0,), window_length=int(cfg.upo.windows.length),
    ):
        evidence = compute_motion_component_evidence(
            query_set, attack, reference, support, cfg.upo.motion, image_hw=image_hw
        )
        context = _bootstrap_context(
            condition_id="stress", edge_id=f"stress-{attack.candidate_id}", support=support,
            evidence=evidence, cfg=cfg,
        )
        for component in PRIMARY_COMPONENTS:
            row = context["intervals"][component]
            first_direction = (
                "positive" if row["lower"] is not None and float(row["lower"]) > 0
                else "negative" if row["upper"] is not None and float(row["upper"]) < 0
                else "none"
            )
            second = row["secondary"]
            second_direction = (
                "positive" if second["lower"] is not None and float(second["lower"]) > 0
                else "negative" if second["upper"] is not None and float(second["upper"]) < 0
                else "none"
            )
            stable = first_direction != "none" and first_direction == second_direction
            row["holm_significant"] = stable and row["p_value_two_sided"] is not None and float(row["p_value_two_sided"]) <= float(cfg.upo.bootstrap.familywise_alpha)
            row["bootstrap_direction_stable"] = stable
            row["holm"] = {"significant": row["holm_significant"]}
            row["secondary_holm"] = {"significant": stable}
        context["quality"] = quality_comparability(quality_attack, quality_reference, cfg.upo.quality)
        rows.append(_decide_context(context, ropes=ropes, threshold=threshold, cfg=cfg))
    return rows


def _run_stress_suite(
    fixture: Mapping[str, Any] | None,
    *,
    ropes: Mapping[str, float],
    threshold: float,
    cfg: Any,
) -> dict[str, Any]:
    if fixture is None:
        return {"pass": False, "reason": "no_valid_real_fixture", "attacks": []}
    query_set: PairedQuerySet = fixture["query_set"]
    reference: RawTrackObservation = fixture["observation_a"]
    sibling: RawTrackObservation = fixture["observation_b"]
    frames_a: torch.Tensor = fixture["frames_a"]
    frames_b: torch.Tensor = fixture["frames_b"]
    image_hw = tuple(int(value) for value in frames_a.shape[-2:])
    quality_reference = video_quality_metrics(frames_a)
    attacks = []

    freeze_points = reference.raw_points.clone()
    freeze_points[:, 2:] = freeze_points[:, 1:2]
    freeze = _clone_attack(reference, candidate_id="stress-freeze", points=freeze_points)
    freeze_relations = _stress_relation(
        query_set, freeze, reference, quality_reference, quality_reference,
        ropes, threshold, cfg, image_hw=image_hw,
    )
    freeze_pass = all(
        not (row["relation"] == "strict" and row["winner_candidate_id"] == freeze.candidate_id)
        for row in freeze_relations
    )
    attacks.append({"attack": "freeze", "pass": freeze_pass, "relations": freeze_relations})

    slow_points = reference.raw_points.clone()
    slow_points[:, 1] = 0.5 * (reference.raw_points[:, 0] + reference.raw_points[:, 1])
    slow_points[:, 2] = reference.raw_points[:, 1]
    slow_points[:, 3] = 0.5 * (reference.raw_points[:, 1] + reference.raw_points[:, 2])
    slow = _clone_attack(reference, candidate_id="stress-time-slow", points=slow_points)
    slow_relations = _stress_relation(
        query_set, slow, reference, quality_reference, quality_reference,
        ropes, threshold, cfg, image_hw=image_hw,
    )
    slow_pass = all(
        not (row["relation"] == "strict" and row["winner_candidate_id"] == slow.candidate_id)
        for row in slow_relations
    )
    attacks.append({"attack": "time_slow", "pass": slow_pass, "relations": slow_relations})

    dropout_visibility = reference.raw_visibility.clone()
    dynamic = query_set.mask("dynamic")
    dropout_visibility[dynamic, 2:] = False
    dropout = _clone_attack(reference, candidate_id="stress-dropout", visibility=dropout_visibility)
    dropout_support = build_common_support(
        query_set, dropout, reference, cfg.upo.support, window_starts=(0,),
        window_length=int(cfg.upo.windows.length),
    )
    dropout_pass = bool(dropout_support) and all(not window.valid for window in dropout_support)
    attacks.append({
        "attack": "track_dropout", "pass": dropout_pass,
        "support_reasons": [window.reason for window in dropout_support],
    })

    camera_points = reference.raw_points.clone()
    time_index = torch.arange(camera_points.shape[1]).float()[None, :, None]
    camera_points = camera_points + time_index * torch.tensor([3.0, 0.0])[None, None]
    camera = _clone_attack(reference, candidate_id="stress-camera", points=camera_points)
    camera_relations = _stress_relation(
        query_set, camera, reference, quality_reference, quality_reference,
        ropes, threshold, cfg, image_hw=image_hw,
    )
    camera_pass = all(row["relation"] != "strict" for row in camera_relations)
    attacks.append({"attack": "camera", "pass": camera_pass, "relations": camera_relations})

    quality_frames = frames_a.clone()
    quality_frames[1::2] = quality_frames[1::2].mul(-1.0)
    quality_gate = quality_comparability(
        video_quality_metrics(quality_frames), quality_reference, cfg.upo.quality
    )
    quality_pass = not bool(quality_gate["comparable"])
    attacks.append({"attack": "quality", "pass": quality_pass, "quality_gate": quality_gate})

    base_support = build_common_support(
        query_set, reference, sibling, cfg.upo.support, window_starts=(0,),
        window_length=int(cfg.upo.windows.length),
    )[0]
    base_evidence = compute_motion_component_evidence(
        query_set, reference, sibling, base_support, cfg.upo.motion, image_hw=image_hw
    )
    transform = torch.arange(reference.raw_points.shape[1]).float()[None, :, None] * torch.tensor([1.0, 0.5])[None, None]
    transformed_a = _clone_attack(reference, candidate_id=reference.candidate_id, points=reference.raw_points + transform)
    transformed_b = _clone_attack(sibling, candidate_id=sibling.candidate_id, points=sibling.raw_points + transform)
    transformed_support = build_common_support(
        query_set, transformed_a, transformed_b, cfg.upo.support, window_starts=(0,),
        window_length=int(cfg.upo.windows.length),
    )[0]
    transformed_evidence = compute_motion_component_evidence(
        query_set, transformed_a, transformed_b, transformed_support, cfg.upo.motion, image_hw=image_hw
    )
    deltas = {}
    for component in PRIMARY_COMPONENTS:
        first = base_evidence.differences[component]
        second = transformed_evidence.differences[component]
        finite = torch.isfinite(first) & torch.isfinite(second)
        deltas[component] = (
            float((first[finite] - second[finite]).abs().max()) if bool(finite.any()) else None
        )
    invariance_pass = base_evidence.valid and transformed_evidence.valid and all(
        value is None or value <= float(cfg.upo.stress.maximum_common_transform_component_delta)
        for value in deltas.values()
    )
    attacks.append({"attack": "common_transform", "pass": invariance_pass, "component_max_delta": deltas})
    return {
        "pass": all(bool(row["pass"]) for row in attacks),
        "fixture_condition_id": fixture["condition_id"],
        "attacks": attacks,
        "uses_future_gt": False,
    }


def preflight_physics_preference_reaudit(cfg: Any) -> dict[str, Any]:
    result = {"task_id": str(cfg.upo.task_id), "status": "ready", "blockers": [], "uses_gpu": True}
    try:
        inputs = _source_inputs(cfg)
        grouped = _core_candidates(inputs["conditions"], inputs["candidates"])
        partitions = _review_partitions(inputs, grouped, cfg)
        path_by_id = {str(row["candidate_id"]): Path(str(row["video_path"])) for row in inputs["source_index"]}
        core_ids = {str(row["candidate_id"]) for row in inputs["candidates"]}
        missing = sorted(candidate_id for candidate_id in core_ids if candidate_id not in path_by_id or not path_by_id[candidate_id].is_file())
        if missing:
            raise PreferenceReauditError(f"缺少 core sibling RGB: {missing[:3]}")
        result.update({
            "condition_count": len(grouped),
            "core_candidate_count": len(core_ids),
            "calibration_ties": len(partitions["calibration_case_ids"]),
            "holdout_ties": len(partitions["holdout_case_ids"]),
            "source_fingerprints": inputs["fingerprints"],
            "training": False,
            "candidate_generation": False,
            "uses_future_gt": False,
        })
    except Exception as exc:
        result["status"] = "blocked"
        result["blockers"].append(repr(exc))
    return result


def run_physics_preference_reaudit(cfg: Any) -> dict[str, Any]:
    git = git_state(".")
    if git.get("dirty"):
        raise PreferenceReauditError("PA2-UPO 正式 run 拒绝 dirty worktree")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"PA2-UPO run 已存在: {work_dir}")
    inputs = _source_inputs(cfg)
    grouped = _core_candidates(inputs["conditions"], inputs["candidates"])
    partitions = _review_partitions(inputs, grouped, cfg)
    path_by_id = {str(row["candidate_id"]): Path(str(row["video_path"])) for row in inputs["source_index"]}
    core_ids = {str(row["candidate_id"]) for row in inputs["candidates"]}
    if any(candidate_id not in path_by_id or not path_by_id[candidate_id].is_file() for candidate_id in core_ids):
        raise PreferenceReauditError("core sibling RGB 不完整")
    work_dir.mkdir(parents=True, exist_ok=False)
    cfg_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=cfg_fp,
        cache_fingerprint="not-applicable:existing-sibling-rgb-only", seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split="preference_train",
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.upo.task_id), "status": "running", "training": False,
        "candidate_generation": False, "uses_future_gt": False,
        "source_run": str(inputs["source"]), "source_fingerprints": inputs["fingerprints"],
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    atomic_write_json(str(work_dir / "calibration_split.json"), partitions)
    seed_everything(int(cfg.seed), deterministic=True)

    query_records: list[dict[str, Any]] = []
    track_records: list[dict[str, Any]] = []
    support_records: list[dict[str, Any]] = []
    background_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    interval_records: list[dict[str, Any]] = []
    contexts_by_condition: dict[str, list[dict[str, Any]]] = {}
    measurement_values = {component: [] for component in PRIMARY_COMPONENTS}
    measurement_records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    stress_fixture = None
    tracker = PairModeRAFTTracker(cfg.upo.tracker, device=str(cfg.device))
    sorted_conditions = sorted(inputs["conditions"], key=lambda row: str(row["condition_id"]))
    measurement_ranked = sorted(
        sorted_conditions,
        key=lambda row: hashlib.sha256(
            f"{cfg.upo.measurement.scene_hash_salt}\0{row['scene_id']}".encode("utf-8")
        ).hexdigest(),
    )
    measurement_ids = {
        str(row["condition_id"]) for row in measurement_ranked[: int(cfg.upo.measurement.condition_count)]
    }
    started = time.perf_counter()
    try:
        for condition_index, condition in enumerate(sorted_conditions):
            condition_started = time.perf_counter()
            condition_id = str(condition["condition_id"])
            group = grouped[condition_id]
            base_id = str(group["base"]["candidate_id"])
            sibling_ids = [str(row["candidate_id"]) for row in group["siblings"]]
            base_frames = _decode_video(path_by_id[base_id], expected_frames=int(cfg.upo.expected_num_frames))
            sibling_frames = {
                candidate_id: _decode_video(path_by_id[candidate_id], expected_frames=int(cfg.upo.expected_num_frames))
                for candidate_id in sibling_ids
            }
            query_set, observations = tracker.track_condition(
                base_candidate_id=base_id, base_frames=base_frames, sibling_frames=sibling_frames
            )
            query_records.append(query_set.to_record(condition_id=condition_id))
            track_records.extend(
                observation.to_record(condition_id=condition_id) for observation in observations.values()
            )
            quality_by_candidate = {
                candidate_id: video_quality_metrics(frames) for candidate_id, frames in sibling_frames.items()
            }
            contexts: list[dict[str, Any]] = []
            for candidate_a, candidate_b in itertools.combinations(sibling_ids, 2):
                edge = _edge_id(condition_id, candidate_a, candidate_b)
                windows = build_common_support(
                    query_set, observations[candidate_a], observations[candidate_b], cfg.upo.support,
                    window_starts=tuple(int(value) for value in cfg.upo.windows.starts),
                    window_length=int(cfg.upo.windows.length),
                )
                for support in windows:
                    evidence = compute_motion_component_evidence(
                        query_set, observations[candidate_a], observations[candidate_b], support,
                        cfg.upo.motion, image_hw=tuple(int(value) for value in base_frames.shape[-2:]),
                    )
                    context = _bootstrap_context(
                        condition_id=condition_id, edge_id=edge, support=support, evidence=evidence, cfg=cfg
                    )
                    context["quality"] = _context_quality(quality_by_candidate, support, cfg)
                    contexts.append(context)
                    support_records.append(support.to_record(
                        condition_id=condition_id, edge_id=edge, query_set=query_set
                    ))
                    background_records.append(evidence.background_record(
                        condition_id=condition_id, edge_id=edge, support=support
                    ))
                    component_records.append(evidence.component_record(
                        condition_id=condition_id, edge_id=edge, support=support
                    ))
                    if stress_fixture is None and query_set.valid and evidence.valid and support.valid:
                        stress_fixture = {
                            "condition_id": condition_id,
                            "query_set": query_set,
                            "observation_a": observations[candidate_a],
                            "observation_b": observations[candidate_b],
                            "frames_a": sibling_frames[candidate_a],
                            "frames_b": sibling_frames[candidate_b],
                        }
            _apply_holm(contexts, cfg)
            for context in contexts:
                for component in PRIMARY_COMPONENTS:
                    interval_records.append({
                        "condition_id": condition_id,
                        "edge_id": context["edge_id"],
                        "candidate_a": context["support"].candidate_a,
                        "candidate_b": context["support"].candidate_b,
                        "start_frame": context["support"].start_frame,
                        "end_frame": context["support"].end_frame,
                        **context["intervals"][component],
                        "uses_future_gt": False,
                    })
            contexts_by_condition[condition_id] = contexts
            if condition_id in measurement_ids and query_set.valid:
                values, records = _collect_measurement_differences(
                    condition_id=condition_id,
                    query_set=query_set,
                    base_observation=observations[base_id],
                    base_frames=base_frames,
                    tracker=tracker,
                    cfg=cfg,
                )
                _merge_measurement_values(measurement_values, values)
                measurement_records.extend(records)
            metrics.append({
                "condition_index": condition_index,
                "condition_id": condition_id,
                "query_valid": query_set.valid,
                "background_queries": query_set.diagnostics["background_query_count"],
                "dynamic_queries": query_set.diagnostics["dynamic_query_count"],
                "seconds": time.perf_counter() - condition_started,
                "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _write_jsonl(work_dir / "query_sets.jsonl", query_records)
        _write_jsonl(work_dir / "paired_tracks.jsonl", track_records)
        _write_jsonl(work_dir / "common_support.jsonl", support_records)
        _write_jsonl(work_dir / "background_fields.jsonl", background_records)
        _write_jsonl(work_dir / "component_differences.jsonl", component_records)
        _write_jsonl(work_dir / "bootstrap_intervals.jsonl", interval_records)
        _write_jsonl(work_dir / "metrics.jsonl", metrics)

        rope_rows = measurement_ropes(
            measurement_values,
            quantile=float(cfg.upo.measurement.rope_quantile),
            minimums={component: float(cfg.upo.measurement.minimum_rope[component]) for component in PRIMARY_COMPONENTS},
        )
        ropes = {component: float(row["rope"]) for component, row in rope_rows.items()}
        calibration_cases = partitions["calibration_cases"]
        calibration_scores = [
            _calibration_case_score(case, contexts_by_condition, ropes=ropes, cfg=cfg)
            for case in calibration_cases
        ]
        comparable_calibration = sum(row["comparable_windows"] > 0 for row in calibration_scores)
        threshold_row = split_conformal_threshold(
            [row["score"] for row in calibration_scores if row["comparable_windows"] > 0],
            alpha=float(cfg.upo.calibration.false_strict_alpha),
            minimum=float(cfg.upo.calibration.minimum_strict_threshold),
        )
        secondary_scores = [
            _calibration_case_score(case, contexts_by_condition, ropes=ropes, cfg=cfg, secondary=True)
            for case in calibration_cases
        ]
        secondary_threshold = split_conformal_threshold(
            [row["score"] for row in secondary_scores if row["comparable_windows"] > 0],
            alpha=float(cfg.upo.calibration.false_strict_alpha),
            minimum=float(cfg.upo.calibration.minimum_strict_threshold),
        )
        threshold = float(threshold_row["threshold"])
        threshold_relative_delta = abs(threshold - float(secondary_threshold["threshold"])) / max(
            threshold, float(secondary_threshold["threshold"]), 1.0e-8
        )
        exactness_rows = [
            row["identical_rerun_exactness"] for row in measurement_records
            if row["perturbation"] == "identical_rerun"
        ]
        measurement_valid_condition_count = len({
            row["condition_id"] for row in measurement_records if int(row["valid_windows"]) > 0
        })
        calibration_summary = {
            "measurement_ropes": rope_rows,
            "measurement_audit": measurement_records,
            "measurement_condition_count": len({row["condition_id"] for row in measurement_records}),
            "measurement_valid_condition_count": measurement_valid_condition_count,
            "identical_rerun_exact": bool(exactness_rows) and all(bool(row["exact"]) for row in exactness_rows),
            "calibration_cases": calibration_scores,
            "comparable_calibration_ties": comparable_calibration,
            "threshold": threshold_row,
            "secondary_seed_threshold": secondary_threshold,
            "threshold_relative_delta": threshold_relative_delta,
            "threshold_seed_stable": threshold_relative_delta <= float(cfg.upo.calibration.maximum_threshold_relative_delta),
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "calibration_summary.json"), calibration_summary)

        all_relations: dict[str, list[dict[str, Any]]] = {}
        secondary_relations: dict[str, list[dict[str, Any]]] = {}
        graph_rows = []
        graph_by_condition = {}
        for condition_id, contexts in contexts_by_condition.items():
            relations = [
                _decide_context(context, ropes=ropes, threshold=threshold, cfg=cfg)
                for context in contexts
            ]
            second = [
                _decide_context(
                    context, ropes=ropes, threshold=float(secondary_threshold["threshold"]),
                    cfg=cfg, secondary=True,
                )
                for context in contexts
            ]
            graph = build_condition_partial_order(
                condition_id, relations,
                minimum_tie_fraction=float(cfg.upo.relation.minimum_condition_tie_fraction),
            )
            graph["relations"] = relations
            graph_rows.append(graph)
            graph_by_condition[condition_id] = graph
            all_relations[condition_id] = relations
            secondary_relations[condition_id] = second
        _write_jsonl(work_dir / "graphs.jsonl", graph_rows)

        def audit_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            rows = []
            for case in cases:
                contexts = _case_contexts(case, contexts_by_condition)
                relations = [
                    _decide_context(context, ropes=ropes, threshold=threshold, cfg=cfg)
                    for context in contexts
                ]
                strict = [row for row in relations if row["relation"] == "strict"]
                rows.append({
                    "case_id": case["case_id"],
                    "condition_id": case["condition_id"],
                    "predicted_strict": bool(strict),
                    "high_confidence_strict": any(bool(row["high_confidence"]) for row in strict),
                    "strict_windows": len(strict),
                    "relations": relations,
                })
            return rows

        holdout_rows = audit_cases(partitions["holdout_cases"])
        invalidity_rows = audit_cases([*partitions["uncertain_cases"], *partitions["both_invalid_cases"]])
        false_strict = sum(bool(row["predicted_strict"]) for row in holdout_rows)
        high_false_strict = sum(bool(row["high_confidence_strict"]) for row in holdout_rows)
        invalid_high_strict = sum(bool(row["high_confidence_strict"]) for row in invalidity_rows)
        holdout_summary = {
            "holdout_cases": holdout_rows,
            "false_strict": false_strict,
            "high_confidence_false_strict": high_false_strict,
            "invalidity_audit": invalidity_rows,
            "invalidity_high_confidence_strict": invalid_high_strict,
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "holdout_summary.json"), holdout_summary)

        stress_summary = _run_stress_suite(
            stress_fixture, ropes=ropes, threshold=threshold, cfg=cfg
        )
        atomic_write_json(str(work_dir / "stress_summary.json"), stress_summary)
        cycle_conditions = [row for row in graph_rows if row["status"] == "invalid_cycle"]
        cycle_rate = len(cycle_conditions) / max(len(graph_rows), 1)
        reviewed_conditions = set(partitions["reviewed_condition_ids"])
        prospective = [row for row in graph_rows if row["condition_id"] not in reviewed_conditions]
        prospective_counts = {
            status: sum(row["status"] == status for row in prospective)
            for status in ("strict", "tie", "incomparable", "invalid_cycle", "invalid_component_conflict")
        }
        strict_primary = {
            condition_id for condition_id, rows in all_relations.items()
            if any(row["relation"] == "strict" for row in rows)
        }
        strict_secondary = {
            condition_id for condition_id, rows in secondary_relations.items()
            if any(row["relation"] == "strict" for row in rows)
        }
        strict_seed_jaccard = len(strict_primary & strict_secondary) / max(len(strict_primary | strict_secondary), 1)
        checks = {
            "measurement_minimum_valid_conditions": measurement_valid_condition_count
                >= int(cfg.upo.measurement.minimum_valid_conditions),
            "measurement_identical_rerun_exact": bool(calibration_summary["identical_rerun_exact"]),
            "calibration_holdout_scene_disjoint": bool(partitions["scene_disjoint"]),
            "minimum_comparable_calibration_ties": comparable_calibration >= int(cfg.upo.calibration.minimum_comparable_ties),
            "holdout_false_strict": false_strict <= int(cfg.upo.calibration.maximum_holdout_false_strict),
            "holdout_high_confidence_false_strict": high_false_strict == 0,
            "invalidity_high_confidence_false_strict": invalid_high_strict == 0,
            "shortcut_stress": bool(stress_summary.get("pass")),
            "cycle_rate": cycle_rate <= float(cfg.upo.relation.maximum_cycle_rate),
            "bootstrap_threshold_stability": bool(calibration_summary["threshold_seed_stable"])
                and strict_seed_jaccard >= float(cfg.upo.calibration.minimum_strict_seed_jaccard),
        }
        oracle_gates_pass = all(checks.values())
        strict_yield = prospective_counts["strict"]
        review_mix_pass = (
            strict_yield >= int(cfg.upo.prospective.minimum_strict_conditions)
            and prospective_counts["tie"] >= int(cfg.upo.prospective.minimum_tie_conditions)
            and prospective_counts["incomparable"] >= int(cfg.upo.prospective.minimum_incomparable_conditions)
        )
        if not oracle_gates_pass:
            status = "rejected"
            next_gate = "stop SVD sibling preference oracle"
        elif strict_yield < int(cfg.upo.prospective.minimum_strict_conditions):
            status = "blocked_candidate_yield"
            next_gate = "PA2-CAND-03D single earlier-fork fallback"
        elif not review_mix_pass:
            status = "blocked_review_mix"
            next_gate = "AC decision required; thresholds remain frozen"
        else:
            status = "awaiting_reviews"
            next_gate = "PA2-PROSPECT-03C"
        summary = {
            "status": status,
            "task_id": str(cfg.upo.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": cfg_fp,
            "condition_count": len(sorted_conditions),
            "query_valid_conditions": sum(bool(row["valid"]) for row in query_records),
            "paired_track_records": len(track_records),
            "common_support_windows": len(support_records),
            "calibrated_strict_threshold": threshold,
            "measurement_ropes": ropes,
            "holdout_false_strict": false_strict,
            "holdout_high_confidence_false_strict": high_false_strict,
            "invalidity_high_confidence_strict": invalid_high_strict,
            "cycle_condition_count": len(cycle_conditions),
            "cycle_rate": cycle_rate,
            "strict_seed_jaccard": strict_seed_jaccard,
            "prospective_condition_count": len(prospective),
            "prospective_counts": prospective_counts,
            "prospective_review_mix_pass": review_mix_pass,
            "checks": checks,
            "oracle_gates_pass": oracle_gates_pass,
            "next_gate": next_gate,
            "training": False,
            "candidate_generation": False,
            "uses_future_gt": False,
            "seconds_total": time.perf_counter() - started,
            "peak_vram_bytes": max((int(row["peak_vram_bytes"]) for row in metrics), default=0),
            "source_fingerprints_before": inputs["fingerprints"],
        }
        _ensure_source_immutable(inputs["source"], inputs["fingerprints"])
        summary["source_fingerprints_after"] = _source_fingerprints(inputs["source"])
        atomic_write_json(str(work_dir / "summary.json"), summary)
        marker = "REJECTED" if status == "rejected" else "COMPLETE"
        atomic_write_text(str(work_dir / marker), sha256_json(summary) + "\n")
        manifest_data.update({
            "status": status, "ended_at": utc_now(), "exit_reason": next_gate,
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {
            "status": "failed", "task_id": str(cfg.upo.task_id), "run_id": str(cfg.run_id),
            "config_fingerprint": cfg_fp, "error": repr(exc), "training": False,
            "candidate_generation": False, "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "summary.json"), failure)
        atomic_write_text(str(work_dir / "REJECTED"), sha256_json(failure) + "\n")
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PA2-UPO common-support preference reaudit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = preflight_physics_preference_reaudit(cfg) if args.preflight else run_physics_preference_reaudit(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
