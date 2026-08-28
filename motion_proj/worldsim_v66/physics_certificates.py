"""V6.6 reason-coded Actor 物理合法性证书。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v66.actor_factorial import certificate


def _compiler_action(row: Mapping[str, Any], reasons: list[str]) -> str:
    if not reasons:
        return "KEEP"
    if "duplicate_overlap" in reasons:
        return "DROP_ARTIFACT_PRIMITIVE"
    if "sensor_and_provenance_missing" in reasons:
        return "ABSTAIN_LOCAL_GEOMETRY"
    return "REPAIR"


def compile_certificate_rows(
    rows: Iterable[Mapping[str, Any]],
    certificate_config: Mapping[str, float],
) -> list[dict[str, Any]]:
    compiled = []
    for source in rows:
        row = dict(source)
        score, state, reasons = certificate(
            row,
            maximum_kinematic_jump_m=float(
                certificate_config["maximum_supported_kinematic_jump_m"]
            ),
            maximum_shape_ratio_jump=float(
                certificate_config["maximum_shape_ratio_jump"]
            ),
        )
        compiled.append(
            {
                "base_id": str(row["base_id"]),
                "cluster_id": str(row["cluster_id"]),
                "variant_id": str(row["variant_id"]),
                "scene": str(row["scene"]),
                "unit": str(row["unit"]),
                "actor_id": int(row["actor_id"]),
                "artifact_family": str(row["artifact_family"]),
                "artifact_label": bool(row["artifact_label"]),
                "hazard_label": bool(row["hazard_label"]),
                "certificate_score": float(score),
                "certificate_state": state,
                "reason_codes": reasons,
                "compiler_action": _compiler_action(row, reasons),
                "actor_existence_retained": True,
                "actor_id_retained": True,
                "lifecycle_retained": True,
            }
        )
    return compiled


def evaluate_certificates(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([bool(row["artifact_label"]) for row in rows])
    hazard = np.asarray([bool(row["hazard_label"]) for row in rows])
    scores = np.asarray([float(row["certificate_score"]) for row in rows])
    predicted = scores >= 0.5
    clean = ~labels
    clean_hazard = clean & hazard
    clean_benign = clean & ~hazard
    family_recall = {}
    for family in sorted({str(row["artifact_family"]) for row in rows}):
        members = np.asarray(
            [str(row["artifact_family"]) == family and bool(row["artifact_label"]) for row in rows]
        )
        family_recall[family] = float(np.mean(predicted[members]))

    by_key: dict[tuple[str, bool], dict[bool, float]] = {}
    for row in rows:
        by_key.setdefault(
            (str(row["cluster_id"]), bool(row["artifact_label"])), {}
        )[bool(row["hazard_label"])] = float(row["certificate_score"])
    pair_deltas = [
        abs(pair[True] - pair[False])
        for pair in by_key.values()
        if set(pair) == {False, True}
    ]
    hard_observed_violations = sum(
        bool(row["artifact_label"])
        and "sensor_and_provenance_missing" in row["reason_codes"]
        and row["certificate_state"] != "ARTIFACT"
        for row in rows
    )
    return {
        "row_count": len(rows),
        "base_actor_unit_count": len({str(row["base_id"]) for row in rows}),
        "artifact_recall": float(np.mean(predicted[labels])),
        "artifact_auroc": float(roc_auc_score(labels, scores)),
        "artifact_auprc": float(average_precision_score(labels, scores)),
        "artifact_family_recall": family_recall,
        "clean_hazard_false_artifact_rate": float(np.mean(predicted[clean_hazard])),
        "clean_benign_false_artifact_rate": float(np.mean(predicted[clean_benign])),
        "legitimate_hazardous_actor_retention": float(np.mean(~predicted[clean_hazard])),
        "legitimate_benign_actor_retention": float(np.mean(~predicted[clean_benign])),
        "actor_existence_retention": float(
            np.mean([bool(row["actor_existence_retained"]) for row in rows])
        ),
        "actor_id_retention": float(np.mean([bool(row["actor_id_retained"]) for row in rows])),
        "lifecycle_retention": float(
            np.mean([bool(row["lifecycle_retained"]) for row in rows])
        ),
        "mean_absolute_hazard_pair_score_delta": float(np.mean(pair_deltas)),
        "hard_observed_evidence_violations": int(hard_observed_violations),
        "decision_counts": dict(Counter(str(row["compiler_action"]) for row in rows)),
    }
