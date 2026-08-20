"""Deterministic representative subset selection for V5.2.1 manual review."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


AXIS_REGION = {
    "GLOBAL_RGB": "global",
    "ACTOR_RGB": "actor",
    "BOUNDARY": "boundary",
}

RESEARCH_DIRECTION = {
    "GLOBAL_RGB": "base_reconstruction_and_appearance_robustness",
    "ACTOR_RGB": "dynamic_actor_representation_before_ownership_method_design",
    "BOUNDARY": "dynamic_boundary_compositing_conditioned_on_exact_m1_m2_overlap",
}


def _severity_key(row: Mapping[str, Any], axis: str) -> tuple[Any, ...]:
    metric = row["metrics"][AXIS_REGION[axis]]
    lpips = metric.get("lpips_alex")
    return (
        float(metric["psnr"]),
        -float(lpips) if lpips is not None else 0.0,
        str(row["scene"]),
        int(row["canonical_sample_index"]),
        int(row["camera"]),
        str(row["case_id"]),
    )


def _view_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["split_role"]),
        str(row["scene"]),
        int(row["canonical_sample_index"]),
        int(row["camera"]),
    )


def select_representative_cases(
    registry: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select severity, cross-scene, and confirmation anchors without refitting.

    The selector consumes only already-labeled rows that already have frozen panels.
    It never changes thresholds, predicates, rankings, or confirmation verdicts.
    """

    panel_by_case = {str(row["case_id"]): dict(row) for row in panel_rows}
    eligible = [
        dict(row)
        for row in registry
        if row.get("entity_kind") == "view"
        and row.get("classification_status") == "labeled"
        and row.get("selected_for_panel") is True
        and row.get("panel_path")
        and str(row["case_id"]) in panel_by_case
    ]
    by_identity = {
        (str(row["base"]), *_view_key(row)): row
        for row in registry
        if row.get("entity_kind") == "view"
    }
    used_cases: set[str] = set()
    used_views: set[tuple[Any, ...]] = set()
    selected: list[dict[str, Any]] = []
    slot_counts: defaultdict[str, int] = defaultdict(int)

    def choose(candidates: list[dict[str, Any]], *, different_scene: str | None = None) -> dict[str, Any]:
        pools = [
            [row for row in candidates if row["case_id"] not in used_cases and _view_key(row) not in used_views],
            [row for row in candidates if row["case_id"] not in used_cases],
            candidates,
        ]
        for pool in pools:
            if different_scene is not None:
                diverse = [row for row in pool if row["scene"] != different_scene]
                if diverse:
                    return diverse[0]
            if pool:
                return pool[0]
        raise ValueError("representative slot has no eligible panel")

    for axis in ("GLOBAL_RGB", "ACTOR_RGB", "BOUNDARY"):
        for base in ("adgs", "streetgs"):
            discovery = sorted(
                [
                    row for row in eligible
                    if row["base"] == base and row["split_role"] == "discovery" and axis in row["failure_axes"]
                ],
                key=lambda row: _severity_key(row, axis),
            )
            confirmation = sorted(
                [
                    row for row in eligible
                    if row["base"] == base
                    and row["split_role"] == "confirmation"
                    and row.get("confirmation_verdict") == "direction_confirmed"
                    and axis in row["failure_axes"]
                ],
                key=lambda row: _severity_key(row, axis),
            )
            if not discovery or not confirmation:
                raise ValueError(f"representative coverage missing: {base}/{axis}")
            anchors = [
                (choose(discovery), "discovery_severity_anchor"),
            ]
            anchors.append((choose(discovery, different_scene=anchors[0][0]["scene"]), "discovery_cross_scene_anchor"))
            anchors.append((choose(confirmation), "confirmation_direction_anchor"))
            for row, reason in anchors:
                case = str(row["case_id"])
                view = _view_key(row)
                panel = panel_by_case[case]
                counterpart_base = "streetgs" if base == "adgs" else "adgs"
                counterpart = by_identity.get((counterpart_base, *view))
                counterpart_axes = [] if counterpart is None else list(counterpart.get("failure_axes", []))
                metric = row["metrics"][AXIS_REGION[axis]]
                selected.append(
                    {
                        "review_order": len(selected) + 1,
                        "case_id": case,
                        "event_id": row["event_ids"][axis],
                        "selection_slot": f"{row['split_role']}|{base}|{axis}|{reason}",
                        "selection_reason": reason,
                        "base": base,
                        "split_role": row["split_role"],
                        "evidence_tier": row["evidence_tier"],
                        "scene": row["scene"],
                        "canonical_sample_index": int(row["canonical_sample_index"]),
                        "camera": int(row["camera"]),
                        "review_axis": axis,
                        "failure_axes": list(row["failure_axes"]),
                        "failure_class": list(row["failure_class"]),
                        "confirmation_verdict": row["confirmation_verdict"],
                        "severity_metric": {
                            "region": AXIS_REGION[axis],
                            "psnr": metric.get("psnr"),
                            "ssim": metric.get("ssim"),
                            "lpips_alex": metric.get("lpips_alex"),
                            "pixel_count": metric.get("pixel_count"),
                        },
                        "cross_base_status": "shared_failure" if axis in counterpart_axes else "base_specific_failure",
                        "counterpart_case_id": None if counterpart is None else counterpart.get("case_id"),
                        "counterpart_failure_axes": counterpart_axes,
                        "research_direction": RESEARCH_DIRECTION[axis],
                        "panel_path": panel["panel_path"],
                        "panel_sha256": panel["panel_sha256"],
                        "metric_row_sha256": panel["metric_row_sha256"],
                    }
                )
                used_cases.add(case)
                used_views.add(view)
                slot_counts[f"{row['split_role']}|{base}|{axis}"] += 1

    expected = {
        f"{split}|{base}|{axis}": 2 if split == "discovery" else 1
        for split in ("discovery", "confirmation")
        for base in ("adgs", "streetgs")
        for axis in ("GLOBAL_RGB", "ACTOR_RGB", "BOUNDARY")
    }
    observed = dict(sorted(slot_counts.items()))
    if observed != expected:
        raise ValueError(f"representative coverage mismatch: {observed}")
    summary = {
        "selection_contract": "per base/axis: Discovery severity + different-scene anchor; Confirmation severity anchor",
        "threshold_refit": False,
        "predicate_change": False,
        "manual_case_selection": False,
        "slots": len(selected),
        "unique_case_ids": len({row["case_id"] for row in selected}),
        "unique_view_keys": len({(row["split_role"], row["scene"], row["canonical_sample_index"], row["camera"]) for row in selected}),
        "coverage": observed,
    }
    return selected, summary
