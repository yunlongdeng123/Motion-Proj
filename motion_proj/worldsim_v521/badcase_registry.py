"""V5.2.1 分轴 ranking、taxonomy 与 deterministic badcase ID。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .census import CensusError


TAXONOMY_VERSION = "worldsim_v521_badcase_taxonomy_v1"
METRIC_PROTOCOL_VERSION = "worldsim_v521_census_protocol_v1"
AXIS_CLASS = {
    "GLOBAL_RGB": "B-RGB-GLOBAL",
    "ACTOR_RGB": "B-ACTOR",
    "BOUNDARY": "B-BOUNDARY",
}


def _hash12(parts: Sequence[Any]) -> str:
    value = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def case_id(row: Mapping[str, Any], *, entity_kind: str = "view") -> str:
    sample = row.get("window_id") or row.get("sample_token") or row.get("canonical_sample_index") or row.get("frame")
    return "BC-{}-{}".format(
        str(row["base"]).upper(),
        _hash12(
            (
                row["base"], "nuscenes", row["scene"], sample, row.get("camera"),
                entity_kind, row.get("actor_token"), METRIC_PROTOCOL_VERSION,
            )
        ),
    )


def event_id(case: str, axis: str) -> str:
    base = case.split("-")[1]
    return f"BCE-{base}-{_hash12((case, axis, TAXONOMY_VERSION))}"


def _valid_region(row: Mapping[str, Any], region: str, minimum: int) -> bool:
    result = row["metrics"][region]
    return (
        result.get("status") == "done"
        and int(result.get("pixel_count") or 0) >= minimum
        and result.get("psnr") is not None
        and np.isfinite(float(result["psnr"]))
    )


def scene_balanced_tail_threshold(
    rows: Sequence[Mapping[str, Any]], *, region: str, minimum: int, q: float = 0.10
) -> dict[str, Any]:
    """每场先算同 q（numpy linear），再以跨场中位数等权聚合。"""
    per_scene: dict[str, float] = {}
    for scene in sorted({str(row["scene"]) for row in rows}):
        values = [
            float(row["metrics"][region]["psnr"])
            for row in rows
            if row["scene"] == scene and _valid_region(row, region, minimum)
        ]
        if values:
            per_scene[scene] = float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))
    if not per_scene:
        return {
            "status": "undefined_insufficient_denominator", "value": None,
            "per_scene": {}, "valid_scenes": 0,
        }
    return {
        "status": "done",
        "value": float(np.quantile(np.asarray(list(per_scene.values())), 0.5, method="linear")),
        "per_scene": per_scene,
        "valid_scenes": len(per_scene),
        "operator": "per_scene_numpy_q10_linear_then_equal_scene_median",
    }


def freeze_thresholds(base_rows: Sequence[Mapping[str, Any]], minimums: Mapping[str, int]) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    for base in sorted({str(row["base"]) for row in base_rows}):
        selected = [row for row in base_rows if row["base"] == base]
        thresholds[base] = {
            region: scene_balanced_tail_threshold(
                selected, region=region, minimum=int(minimums[region]), q=0.10
            )
            for region in ("global", "actor", "boundary")
        }
    return thresholds


def failure_axes(row: Mapping[str, Any], thresholds: Mapping[str, Any], minimums: Mapping[str, int]) -> list[str]:
    base = str(row["base"])
    axes: list[str] = []
    global_result = row["metrics"]["global"]
    actor = row["metrics"]["actor"]
    static = row["metrics"]["static"]
    boundary = row["metrics"]["boundary"]
    if _valid_region(row, "global", int(minimums["global"])) and float(global_result["psnr"]) <= float(thresholds[base]["global"]["value"]):
        axes.append("GLOBAL_RGB")
    if (
        _valid_region(row, "actor", int(minimums["actor"]))
        and static.get("psnr") is not None
        and float(actor["psnr"]) <= float(thresholds[base]["actor"]["value"])
        and float(actor["psnr"]) < float(static["psnr"])
    ):
        axes.append("ACTOR_RGB")
    if (
        _valid_region(row, "boundary", int(minimums["boundary"]))
        and actor.get("psnr") is not None
        and float(boundary["psnr"]) <= float(thresholds[base]["boundary"]["value"])
        and float(boundary["psnr"]) < float(actor["psnr"])
    ):
        axes.append("BOUNDARY")
    return axes


def _view_sort_key(row: Mapping[str, Any], region: str) -> tuple[Any, ...]:
    result = row["metrics"][region]
    lpips = result.get("lpips_alex")
    return (
        float(result["psnr"]),
        -float(lpips) if lpips is not None else 0.0,
        str(row["scene"]), int(row["canonical_sample_index"]), int(row["camera"]), str(row["base"]),
    )


def _select_tables(rows: Sequence[Mapping[str, Any]], *, region: str, minimum: int, k: int = 12) -> dict[str, Any]:
    ordered = sorted([row for row in rows if _valid_region(row, region, minimum)], key=lambda row: _view_sort_key(row, region))
    limit = min(k, len(ordered))
    severity = ordered[:limit]
    coverage, scene_counts = [], Counter()
    for row in ordered:
        if scene_counts[str(row["scene"])] >= 2:
            continue
        coverage.append(row)
        scene_counts[str(row["scene"])] += 1
        if len(coverage) == limit:
            break

    def compact(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "case_id": case_id(row), "base": row["base"], "scene": row["scene"],
            "frame": row["frame"], "canonical_sample_index": row["canonical_sample_index"],
            "camera": row["camera"], "region": region, "metric": row["metrics"][region],
        }

    return {
        "valid_rows": len(ordered), "k": limit,
        "severity_topk": [compact(row) for row in severity],
        "scene_coverage_topk": [compact(row) for row in coverage],
    }


def _temporal_tables(rows: Sequence[Mapping[str, Any]], k: int = 12) -> dict[str, Any]:
    valid = [row for row in rows if row["metrics"].get("global_residual_change_l1") is not None]
    ordered = sorted(
        valid,
        key=lambda row: (
            -float(row["metrics"]["global_residual_change_l1"]), str(row["scene"]),
            int(row["member_canonical_sample_indices"][0]), int(row["camera"]), str(row["base"]),
        ),
    )
    limit = min(k, len(ordered))
    coverage, counts = [], Counter()
    for row in ordered:
        if counts[str(row["scene"])] >= 2:
            continue
        coverage.append(row)
        counts[str(row["scene"])] += 1
        if len(coverage) == limit:
            break

    def compact(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "case_id": case_id(row, entity_kind="temporal_window"), "base": row["base"],
            "scene": row["scene"], "camera": row["camera"], "window_id": row["window_id"],
            "member_canonical_sample_indices": row["member_canonical_sample_indices"],
            "metric": row["metrics"], "classification_status": "unresolved_proxy_only",
        }

    return {
        "valid_rows": len(ordered), "k": limit,
        "severity_topk": [compact(row) for row in ordered[:limit]],
        "scene_coverage_topk": [compact(row) for row in coverage],
        "failure_label_enabled": False,
    }


def build_leaderboards(
    base_rows: Sequence[Mapping[str, Any]], temporal_rows: Sequence[Mapping[str, Any]], minimums: Mapping[str, int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for base in sorted({str(row["base"]) for row in base_rows}):
        rows = [row for row in base_rows if row["base"] == base]
        windows = [row for row in temporal_rows if row["base"] == base]
        result[base] = {
            "GLOBAL_RGB": _select_tables(rows, region="global", minimum=int(minimums["global"])),
            "ACTOR_RGB": _select_tables(rows, region="actor", minimum=int(minimums["actor"])),
            "BOUNDARY": _select_tables(rows, region="boundary", minimum=int(minimums["boundary"])),
            "TEMPORAL_PROXY": _temporal_tables(windows),
            "GEOMETRY": {"status": "undefined_no_comparable_base_depth", "valid_rows": 0},
            "OCCLUSION_TRANSITION": {"status": "undefined_no_audited_visibility_transition", "valid_rows": 0},
            "OBSERVABILITY": {"status": "undefined_no_audited_observability_denominator", "valid_rows": 0},
        }
    return result


def panel_union(leaderboards: Mapping[str, Any], limit: int = 120) -> set[str]:
    selected: set[str] = set()
    for base in sorted(leaderboards):
        for axis in ("GLOBAL_RGB", "ACTOR_RGB", "BOUNDARY", "TEMPORAL_PROXY"):
            for table in ("severity_topk", "scene_coverage_topk"):
                selected.update(row["case_id"] for row in leaderboards[base][axis][table])
    if len(selected) > limit:
        raise CensusError(f"panel union {len(selected)} > frozen limit {limit}")
    return selected


def build_registry(
    base_rows: Sequence[Mapping[str, Any]], temporal_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any], minimums: Mapping[str, int], selected: set[str],
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    for row in base_rows:
        axes = failure_axes(row, thresholds, minimums)
        if not axes:
            continue
        identifier = case_id(row)
        classes = [AXIS_CLASS[axis] for axis in axes]
        if len(classes) >= 2:
            classes.append("B-MIXED")
        registry.append(
            {
                "case_id": identifier,
                "event_ids": {axis: event_id(identifier, axis) for axis in axes},
                "base": row["base"], "dataset": "nuscenes", "scene": row["scene"],
                "frame": row["frame"], "sample_token": row.get("sample_token"),
                "canonical_sample_index": row["canonical_sample_index"], "camera": row["camera"],
                "entity_kind": "view", "actor_token": None, "temporal_window_id": None,
                "evidence_tier": "D", "split_role": "discovery",
                "failure_axes": axes, "failure_class": classes,
                "metrics": row["metrics"], "actor_context": row["actor_context"],
                "m1_context": {"status": "not_exactly_mapped"},
                "asset_provenance": {
                    "prediction_sha256": row["prediction_sha256"], "target_sha256": row["target_sha256"],
                    "dynamic_mask_sha256": row["dynamic_mask_sha256"],
                },
                "panel_path": None, "classification_status": "labeled",
                "confirmation_verdict": "not_applicable", "selected_for_panel": identifier in selected,
                "blocker_reason": None,
            }
        )
    temporal_lookup = {case_id(row, entity_kind="temporal_window"): row for row in temporal_rows}
    for identifier in sorted(selected & set(temporal_lookup)):
        row = temporal_lookup[identifier]
        registry.append(
            {
                "case_id": identifier, "event_ids": {}, "base": row["base"], "dataset": "nuscenes",
                "scene": row["scene"], "frame": None, "sample_token": None,
                "canonical_sample_index": None, "camera": row["camera"], "entity_kind": "temporal_window",
                "actor_token": None, "temporal_window_id": row["window_id"], "evidence_tier": "D",
                "split_role": "discovery", "failure_axes": ["TEMPORAL_PROXY"],
                "failure_class": ["B-UNRESOLVED"], "metrics": row["metrics"], "actor_context": {},
                "m1_context": {"status": "not_exactly_mapped"}, "asset_provenance": {}, "panel_path": None,
                "classification_status": "unresolved", "confirmation_verdict": "not_applicable",
                "selected_for_panel": True,
                "blocker_reason": "unwarped_temporal_proxy_cannot_trigger_B-TEMPORAL",
            }
        )
    registry.sort(key=lambda row: row["case_id"])
    if len({row["case_id"] for row in registry}) != len(registry):
        raise CensusError("BADCASE_REGISTRY case_id collision")
    return registry


def registry_summary(registry: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_base = {}
    for base in sorted({str(row["base"]) for row in base_rows}):
        denominator = sum(row["base"] == base for row in base_rows)
        cases = [row for row in registry if row["base"] == base and row["entity_kind"] == "view"]
        classes = Counter(label for row in cases for label in row["failure_class"] if label != "B-MIXED")
        by_base[base] = {
            "view_denominator": denominator,
            "labeled_view_cases": len(cases),
            "labeled_view_prevalence": len(cases) / denominator if denominator else None,
            "class_case_counts": dict(sorted(classes.items())),
            "per_scene_case_counts": dict(sorted(Counter(row["scene"] for row in cases).items())),
        }
    return {
        "schema": "worldsim_v521_badcase_summary_v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "by_base": by_base,
        "registry_rows": len(registry),
        "scalar_composite_score": False,
    }
