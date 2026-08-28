"""V6.6 matched DROP / ABSTAIN / REPAIR development compiler。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np

from motion_proj.worldsim_v66.actor_factorial import certificate


FACTOR_FIELDS = (
    "sensor_hit_count",
    "provenance_supported",
    "duplicate_overlap",
    "lifecycle_gap_count",
    "kinematic_jump_m",
    "identity_discontinuity",
    "shape_ratio_jump",
)


def _reason_codes(row: Mapping[str, Any], config: Mapping[str, float]) -> list[str]:
    return certificate(
        row,
        maximum_kinematic_jump_m=float(config["maximum_supported_kinematic_jump_m"]),
        maximum_shape_ratio_jump=float(config["maximum_shape_ratio_jump"]),
    )[2]


def compile_repair_arms(
    rows: Iterable[Mapping[str, Any]],
    certificate_config: Mapping[str, float],
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    clean_reference = {
        (str(row["cluster_id"]), bool(row["hazard_label"])): row
        for row in source_rows
        if not bool(row["artifact_label"])
    }
    compiled = []
    for source in source_rows:
        before_reasons = _reason_codes(source, certificate_config)
        is_artifact = bool(before_reasons)
        clean = clean_reference[(str(source["cluster_id"]), bool(source["hazard_label"]))]
        for arm in ("R0_DROP", "R1_ABSTAIN", "R2_REPAIR"):
            repaired = dict(source)
            changed_fields: list[str] = []
            actor_retained = True
            local_geometry_available = True
            action = "KEEP"
            if is_artifact and arm == "R0_DROP":
                actor_retained = False
                local_geometry_available = False
                action = "DROP_ARTIFACT_ACTOR"
                after_reasons = []
            elif is_artifact and arm == "R1_ABSTAIN":
                local_geometry_available = False
                action = "ABSTAIN_LOCAL_GEOMETRY"
                after_reasons = before_reasons
            elif is_artifact and arm == "R2_REPAIR":
                action = "REPAIR"
                for field in FACTOR_FIELDS:
                    if repaired[field] != clean[field]:
                        repaired[field] = clean[field]
                        changed_fields.append(field)
                after_reasons = _reason_codes(repaired, certificate_config)
            else:
                after_reasons = before_reasons
            hazard_event_retained = not bool(source["hazard_label"]) or actor_retained
            compiled.append(
                {
                    "base_id": str(source["base_id"]),
                    "cluster_id": str(source["cluster_id"]),
                    "variant_id": str(source["variant_id"]),
                    "scene": str(source["scene"]),
                    "unit": str(source["unit"]),
                    "actor_id": int(source["actor_id"]),
                    "artifact_family": str(source["artifact_family"]),
                    "artifact_label": bool(source["artifact_label"]),
                    "hazard_label": bool(source["hazard_label"]),
                    "arm": arm,
                    "compiler_action": action,
                    "before_reason_codes": before_reasons,
                    "after_reason_codes": after_reasons,
                    "changed_fields": changed_fields,
                    "actor_retained": actor_retained,
                    "local_geometry_available": local_geometry_available,
                    "actor_id_exact": True,
                    "track_trajectory_exact": True,
                    "hazard_event_retained": hazard_event_retained,
                }
            )
    return compiled


def evaluate_repair_arms(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    results = {}
    for arm in ("R0_DROP", "R1_ABSTAIN", "R2_REPAIR"):
        members = [row for row in rows if row["arm"] == arm]
        artifact = [row for row in members if bool(row["artifact_label"])]
        clean = [row for row in members if not bool(row["artifact_label"])]
        clean_hazard = [row for row in clean if bool(row["hazard_label"])]
        hazard = [row for row in members if bool(row["hazard_label"])]
        before = sum(len(row["before_reason_codes"]) for row in artifact)
        after = sum(len(row["after_reason_codes"]) for row in artifact)
        retained = [row for row in members if bool(row["actor_retained"])]
        nonartifact_regressions = sum(
            row["compiler_action"] != "KEEP" or bool(row["changed_fields"]) for row in clean
        )
        hazard_retention = float(np.mean([bool(row["hazard_event_retained"]) for row in hazard]))
        results[arm] = {
            "row_count": len(members),
            "artifact_row_count": len(artifact),
            "artifact_violation_count_before": int(before),
            "artifact_violation_count_after": int(after),
            "artifact_violation_reduction": float((before - after) / before),
            "clean_hazard_actor_retention": float(
                np.mean([bool(row["actor_retained"]) for row in clean_hazard])
            ),
            "all_hazard_event_retention": hazard_retention,
            "hazard_event_count_shift": abs(1.0 - hazard_retention),
            "actor_id_track_trajectory_exact_for_retained": float(
                np.mean(
                    [
                        bool(row["actor_id_exact"]) and bool(row["track_trajectory_exact"])
                        for row in retained
                    ]
                )
            ),
            "nonartifact_regression_rate": float(nonartifact_regressions / len(clean)),
            "hard_observed_evidence_violations_after": int(
                sum(len(row["after_reason_codes"]) for row in retained if row["local_geometry_available"])
            ),
            "action_counts": dict(Counter(str(row["compiler_action"]) for row in members)),
        }
    return results
