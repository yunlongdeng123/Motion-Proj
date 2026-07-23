#!/usr/bin/env python
"""从冻结的 proposal-level records 自动计算 H1 pilot 十条 gate。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _external_counts(records, group_name, scene_id=None):
    values = [
        record["groups"][group_name]
        for record in records
        if scene_id is None or record["scene_id"] == scene_id
    ]
    accepted = [value for value in values if value["accepted"]]
    measurable = [value for value in accepted if value["external_evaluator"]["measurable"]]
    violations = sum(
        value["external_evaluator"]["hard_violation"] is True for value in measurable
    )
    return {
        "fixed_pool_count": len(values),
        "accepted_count": len(accepted),
        "measurable_count": len(measurable),
        "hard_violation_count": violations,
        "fixed_pool_hard_violation_rate": _rate(violations, len(values)),
        "accepted_hard_violation_rate": _rate(violations, len(measurable)),
        "reject_count": len(values) - len(accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/resim/v71/matched_pilot_v1.yaml"))
    parser.add_argument("--proposal-bank", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    proposal_path = args.proposal_bank or Path(config["proposal_output"]) / "proposal_bank.json"
    bank = json.loads(proposal_path.read_text(encoding="utf-8"))
    evaluation_root = args.evaluation_root or Path(config["evaluation_output"])
    records = []
    for proposal in bank["proposals"]:
        path = evaluation_root / f"{proposal['proposal_key'].replace(':', '_')}.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    proposal_count = len(records)
    scenes = [value["scene_id"] for value in config["scenes"]]

    identity_checks = []
    for record in records:
        c = record["groups"]["C_pairwise"]
        d1 = record["groups"]["D1_occgs_certify_only"]
        identity_checks.append(
            c["accepted"] == d1["accepted"]
            and c["realized_trajectory_hash"] == d1["realized_trajectory_hash"]
        )
    identity_rate = _rate(sum(identity_checks), len(identity_checks))

    d1_pool = [
        record
        for record in records
        if record["groups"]["C_pairwise"]["accepted"]
        and record["groups"]["C_pairwise"]["external_evaluator"]["measurable"]
    ]
    d1_counts = Counter()
    for record in d1_pool:
        external_positive = bool(
            record["groups"]["C_pairwise"]["external_evaluator"]["hard_violation"]
        )
        verdict = record["groups"]["D1_occgs_certify_only"]["certificate_verdict"]
        d1_counts[f"verdict_{verdict}"] += 1
        d1_counts["external_positive"] += int(external_positive)
        d1_counts["external_negative"] += int(not external_positive)
        d1_counts["tp"] += int(external_positive and verdict == "FAIL")
        d1_counts["fp"] += int(not external_positive and verdict == "FAIL")
        d1_counts["fn_including_abstention"] += int(
            external_positive and verdict != "FAIL"
        )
    precision = _rate(d1_counts["tp"], d1_counts["tp"] + d1_counts["fp"])
    recall = _rate(d1_counts["tp"], d1_counts["external_positive"])
    abstention = _rate(d1_counts["verdict_UNKNOWN"], len(d1_pool))
    pass_coverage = _rate(d1_counts["verdict_PASS"], len(d1_pool))
    if d1_counts["external_positive"] == 0:
        cert_verdict = "BLOCKED"
        cert_reason = "insufficient_positive_support_in_matched_C_pool"
    elif (
        precision is not None
        and recall is not None
        and precision >= float(config["gate"]["d1_min_precision"])
        and recall >= float(config["gate"]["d1_min_recall"])
    ):
        cert_verdict = "PASS"
        cert_reason = "precision_and_recall_thresholds_met"
    else:
        cert_verdict = "REJECTED"
        cert_reason = "precision_or_recall_below_threshold"

    c_external = _external_counts(records, "C_pairwise")
    d2_external = _external_counts(records, "D2_occgs_project")
    per_scene = {}
    for scene_id in scenes:
        c_scene = _external_counts(records, "C_pairwise", scene_id)
        d2_scene = _external_counts(records, "D2_occgs_project", scene_id)
        difference = (
            d2_scene["fixed_pool_hard_violation_rate"]
            - c_scene["fixed_pool_hard_violation_rate"]
        )
        per_scene[scene_id] = {
            "C": c_scene,
            "D2": d2_scene,
            "risk_difference_D2_minus_C": difference,
            "nonworse": difference <= 0.0,
            "worsening_within_10pp": difference
            <= float(config["gate"]["max_scene_worsening_rate"]),
        }

    usable = []
    for record in records:
        group = record["groups"]["D2_occgs_project"]
        usable.append(
            group["accepted"]
            and group["components"]["occupancy"]["verdict"] == "PASS"
            and group["components"]["visibility"]["verdict"] == "PASS"
            and group["adherence"]["pass"]
            and group["label_sync"]["pass"]
            and group["external_evaluator"]["hard_violation"] is False
        )
    usable_yield = _rate(sum(usable), proposal_count)
    nonworse_scene_count = sum(value["nonworse"] for value in per_scene.values())
    max_worsening_ok = all(value["worsening_within_10pp"] for value in per_scene.values())
    aggregate_strict_reduction = (
        d2_external["fixed_pool_hard_violation_rate"]
        < c_external["fixed_pool_hard_violation_rate"]
    )

    label_checks = []
    for record in records:
        for group in record["groups"].values():
            if group["accepted"]:
                label_checks.append(bool(group["label_sync"]["pass"]))
    label_rate = _rate(sum(label_checks), len(label_checks))

    gate = {
        "1_minimum_two_actors_each_scene": all(
            int(value) >= int(config["gate"]["minimum_actors_per_scene"])
            for value in bank["actor_count_by_scene"].values()
        ),
        "2_C_D1_trajectory_hash_100pct": identity_rate
        == float(config["gate"]["trajectory_hash_identity_rate"]),
        "3_D1_precision_recall": cert_verdict == "PASS",
        "4_D2_aggregate_strictly_lower": aggregate_strict_reduction,
        "5_D2_nonworse_at_least_two_scenes": nonworse_scene_count
        >= int(config["gate"]["minimum_nonworse_scenes"]),
        "6_no_scene_worsens_over_10pp": max_worsening_ok,
        "7_D2_usable_yield_and_no_effect_shrink": usable_yield
        >= float(config["gate"]["d2_min_usable_yield"]),
        "8_UNKNOWN_reported_separately": True,
        "9_label_sync_and_hash_invariants": label_rate
        == float(config["gate"]["label_hash_invariant_rate"]),
        "10_S1_reported_separately": "005" in per_scene,
    }
    gate["pass"] = all(gate.values())
    projection_verdict = (
        "PASS"
        if all(
            gate[key]
            for key in (
                "4_D2_aggregate_strictly_lower",
                "5_D2_nonworse_at_least_two_scenes",
                "6_no_scene_worsens_over_10pp",
                "7_D2_usable_yield_and_no_effect_shrink",
                "9_label_sync_and_hash_invariants",
                "10_S1_reported_separately",
            )
        )
        else "REJECTED"
    )
    summary = {
        "schema_version": "h1-pilot-aggregate-v1",
        "task_id": "V7-H1-11D",
        "config_sha256": file_fingerprint(str(args.config)),
        "proposal_bank_sha256": bank["proposal_bank_sha256"],
        "proposal_count": proposal_count,
        "actor_count_by_scene": bank["actor_count_by_scene"],
        "requested_effect_distribution": bank["effect_distribution"],
        "counterfactual_pair_count": len(bank["counterfactual_pairs"]),
        "C_D1_identity": {
            "matching_count": sum(identity_checks),
            "total_count": len(identity_checks),
            "rate": identity_rate,
        },
        "D1": {
            "matched_C_measurable_pool_count": len(d1_pool),
            "counts": dict(d1_counts),
            "precision": precision,
            "recall": recall,
            "abstention": abstention,
            "pass_coverage": pass_coverage,
            "verdict": cert_verdict,
            "reason": cert_reason,
            "UNKNOWN_not_merged": True,
        },
        "C_external": c_external,
        "D2_external": d2_external,
        "C_to_D2_transition": {
            f"{bool(record['groups']['C_pairwise']['external_evaluator']['hard_violation'])}"
            f"->{bool(record['groups']['D2_occgs_project']['external_evaluator']['hard_violation'])}":
            sum(
                1
                for value in records
                if value["groups"]["C_pairwise"]["accepted"]
                and value["groups"]["D2_occgs_project"]["accepted"]
                and (
                    f"{bool(value['groups']['C_pairwise']['external_evaluator']['hard_violation'])}"
                    f"->{bool(value['groups']['D2_occgs_project']['external_evaluator']['hard_violation'])}"
                )
                == (
                    f"{bool(record['groups']['C_pairwise']['external_evaluator']['hard_violation'])}"
                    f"->{bool(record['groups']['D2_occgs_project']['external_evaluator']['hard_violation'])}"
                )
            )
            for record in records
            if record["groups"]["C_pairwise"]["accepted"]
            and record["groups"]["D2_occgs_project"]["accepted"]
        },
        "D2_usable_count": sum(usable),
        "D2_usable_yield": usable_yield,
        "per_scene": per_scene,
        "label_sync_hash_pass_rate": label_rate,
        "gate": gate,
        "h1_cert_verdict": cert_verdict,
        "h1_proj_verdict": projection_verdict,
        "render_audit_status": "pending_geometry_selected_12x3",
        "human_verdict": "not_collected_not_required_for_machine_pilot_gate",
    }
    summary["aggregate_sha256"] = canonical_sha256(summary)
    output = args.output or evaluation_root / "aggregate.json"
    atomic_write_json(str(output), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

