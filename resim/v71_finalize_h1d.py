#!/usr/bin/env python
"""封装 H1-11D 的正式 REJECTED/BLOCKED/COMPLETE run。"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import CANONICALIZATION_VERSION, canonical_sha256
from motion_proj.resim.schema import WORLD_STATE_SCHEMA_VERSION
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import (
    V71RunContract,
    compute_artifact_set_hash,
    generate_run_id,
    utc_now,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _review_prompt() -> str:
    return """# V7-H1-11D Blind Review Prompt Draft

This template is intentionally unfilled. A human reviewer, never the research agent,
must provide verdicts.

For each blinded sample compare the synchronized three-camera, 12-frame bundle and
record exactly one primary verdict:

- `physically_valid`
- `hard_collision_or_offroad`
- `depth_or_occlusion_contradiction`
- `label_or_identity_mismatch`
- `insufficient_evidence`

Also record `scenario_effect_retained` (`yes/no/unknown`), the first failing frame,
camera, concise reason, and reviewer confidence (`low/medium/high`). UNKNOWN must
remain separate from pass/fail. Do not infer group identity from edit magnitude and
do not use visual attractiveness as a legality proxy.

Machine context may be shown only after the human verdict is locked. The present
pilot did not instantiate the pack because D2 exported zero trajectories and the
pre-registered immediate-stop rule fired before high-cost rendering.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/v71/matched_pilot_v1.yaml"),
    )
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--proposal-bank", type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation_root = args.evaluation_root or Path(config["evaluation_output"])
    proposal_path = args.proposal_bank or Path(config["proposal_output"]) / "proposal_bank.json"
    aggregate = _load(evaluation_root / "aggregate_v2.json")
    pre_repair = _load(evaluation_root / "aggregate_pre_export_denominator_fix.json")
    eval_summary = _load(evaluation_root / "summary.json")
    bank = _load(proposal_path)
    if aggregate["proposal_count"] != 30 or eval_summary["proposal_count"] != 30:
        raise RuntimeError("H1-11D 必须完整聚合 30 proposals")
    if aggregate["C_D1_identity"]["rate"] != 1.0:
        raise RuntimeError("C/D1 byte identity gate failed")
    if aggregate["gate"]["pass"]:
        marker = "COMPLETE"
        exit_reason = "H1_pilot_gate_passed"
    elif (
        aggregate["h1_cert_verdict"] == "BLOCKED"
        and aggregate["h1_proj_verdict"] != "REJECTED"
    ):
        marker = "BLOCKED"
        exit_reason = "H1_CERT_blocked_insufficient_positive_support"
    else:
        marker = "REJECTED"
        exit_reason = "H1_CERT_and_or_H1_PROJ_pre_registered_gate_rejected"

    records = [
        _load(evaluation_root / f"{value['proposal_key'].replace(':', '_')}.json")
        for value in bank["proposals"]
    ]
    false_positives, false_negatives = [], []
    d2_reasons, c_occupancy = Counter(), Counter()
    world_hashes, audit_requests = {}, {}
    for record in records:
        c = record["groups"]["C_pairwise"]
        d1 = record["groups"]["D1_occgs_certify_only"]
        d2 = record["groups"]["D2_occgs_project"]
        external = c["external_evaluator"]
        if d1["certificate_verdict"] == "FAIL" and not external["hard_violation"]:
            false_positives.append(
                {
                    "proposal_key": record["proposal_key"],
                    "static_overlap_voxels": d1["components"]["occupancy"][
                        "static_overlap_voxels"
                    ],
                    "raw_lidar_non_source_points": external["components"]["raw_lidar"][
                        "non_source_points_inside"
                    ],
                }
            )
        if d1["certificate_verdict"] != "FAIL" and external["hard_violation"]:
            false_negatives.append(
                {
                    "proposal_key": record["proposal_key"],
                    "D1_verdict": d1["certificate_verdict"],
                    "known_fraction": d1["components"]["occupancy"]["known_fraction"],
                    "raw_lidar_non_source_points": external["components"]["raw_lidar"][
                        "non_source_points_inside"
                    ],
                }
            )
        d2_reasons[d2["reason"]] += 1
        c_occupancy[c["components"]["occupancy"]["verdict"]] += 1
        for name, group in record["groups"].items():
            if group["accepted"]:
                world_hashes[f"{record['proposal_key']}:{name}"] = group[
                    "world_state_hash"
                ]
                audit_requests[f"{record['proposal_key']}:{name}"] = {
                    "world_state_hash": group["world_state_hash"],
                    "frames": group["render_audit_frames"],
                    "cameras": config["render_audit"]["cameras"],
                    "status": "not_rendered_after_primary_reject",
                }
    diagnostic = {
        "schema_version": "h1-reject-diagnostic-v1",
        "D1_false_positives": false_positives,
        "D1_false_negatives_including_abstention": false_negatives,
        "D2_rejection_reasons": dict(d2_reasons),
        "C_occupancy_verdict_distribution": dict(c_occupancy),
        "source_effect_distribution": bank["effect_distribution"],
        "counterfactual_pair_count": len(bank["counterfactual_pairs"]),
        "allowed_repair_used": {
            "mechanism": "metric_aggregation_bug",
            "commit": "b82c5400fdf50eac0e04c1b55c7baa732dd6fa5b",
            "changed_method_outputs": False,
        },
        "additional_method_repair_allowed": False,
        "render_audit_status": (
            "not_run_due_immediate_stop_D2_rejected_all_proposals"
        ),
        "blind_review_pack_status": "not_instantiated_no_D2_exports",
    }
    diagnostic["diagnostic_sha256"] = canonical_sha256(diagnostic)
    diagnostic_path = args.run_root / "H1D_REJECT_DIAGNOSTIC.json"
    prompt_path = args.run_root / "H1D_BLIND_REVIEW_PROMPT_DRAFT.md"
    atomic_write_json(str(diagnostic_path), diagnostic)
    atomic_write_text(str(prompt_path), _review_prompt())

    full_rows = []
    for path in sorted(value for value in evaluation_root.rglob("*") if value.is_file()):
        full_rows.append(
            {
                "relative_path": path.relative_to(evaluation_root).as_posix(),
                "sha256": file_fingerprint(str(path)),
                "size_bytes": path.stat().st_size,
            }
        )
    full_index = {
        "schema_version": 1,
        "root": str(evaluation_root),
        "file_count": len(full_rows),
        "total_size_bytes": sum(value["size_bytes"] for value in full_rows),
        "files": full_rows,
        "index_sha256": canonical_sha256(full_rows),
    }
    index_path = args.run_root / "H1D_FULL_ARTIFACT_INDEX.json"
    atomic_write_json(str(index_path), full_index)

    world_state_hash = canonical_sha256(world_hashes)
    render_request_hash = canonical_sha256(audit_requests)
    sources = [
        (proposal_path, "artifacts/proposal_bank_full.json", "application/json"),
        (evaluation_root / "summary.json", "artifacts/matched_eval_summary.json", "application/json"),
        (evaluation_root / "aggregate_v2.json", "artifacts/aggregate_v2.json", "application/json"),
        (
            evaluation_root / "aggregate_pre_export_denominator_fix.json",
            "artifacts/aggregate_pre_export_denominator_fix.json",
            "application/json",
        ),
        (diagnostic_path, "artifacts/reject_diagnostic.json", "application/json"),
        (prompt_path, "artifacts/blind_review_prompt_draft.md", "text/markdown"),
        (index_path, "artifacts/full_artifact_index.json", "application/json"),
    ]
    artifact_rows = [
        {
            "relative_path": relative,
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
            "artifact_hash": file_fingerprint(str(source)),
            "size_bytes": source.stat().st_size,
            "media_type": media,
        }
        for source, relative, media in sources
    ]
    artifact_set_hash = compute_artifact_set_hash(artifact_rows)
    code = git_state(str(PROJECT_ROOT))
    if code["dirty"]:
        raise RuntimeError("正式 H1-11D finalizer 要求 clean git state")
    config_fingerprint = file_fingerprint(str(args.config))
    run_id = generate_run_id("V7-H1-11D", "PILOT-3-matched", 0, config_fingerprint)
    registry_root = Path(config["registry_root"])
    checkpoint_hashes, actor_ids = {}, []
    for scene_id, count in bank["actor_count_by_scene"].items():
        registry = _load(registry_root / scene_id / "actor_registry.json")
        checkpoint_hashes[scene_id] = registry["checkpoint_sha256"]
        actor_ids.extend(
            f"{scene_id}:{actor_id}"
            for actor_id in bank["scenes"][scene_id]["selected_actor_ids"]
        )
    manifest = {
        "schema_version": 1,
        "task_id": "V7-H1-11D",
        "run_id": run_id,
        "parent_run_id": (
            "v71_v7-h1-11c__pilot-3-interface__s0__"
            "20260723T152729207694Z__8429e9a5"
        ),
        "command": list(sys.argv),
        "plan_version": "V7.1",
        "code_commit": code["commit"],
        "code_dirty": False,
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_fingerprint,
        "data_fingerprint": canonical_sha256(
            {
                "proposal_bank": bank["proposal_bank_sha256"],
                "matched_eval": eval_summary["matched_eval_sha256"],
                "aggregate": aggregate["aggregate_sha256"],
                "full_index": full_index["index_sha256"],
            }
        ),
        "split_fingerprint": canonical_sha256(
            {"split": config["split"], "scenes": list(bank["scenes"])}
        ),
        "proposal_fingerprint": bank["proposal_bank_sha256"],
        "third_party_commit": _git_commit(
            Path("/root/autodl-tmp/third_party/drivestudio")
        ),
        "checkpoint_hashes": checkpoint_hashes,
        "scene_ids": list(bank["scenes"]),
        "actor_ids": actor_ids,
        "camera_ids": config["render_audit"]["cameras"],
        "time_range": {
            "frame_range": config["frame_range"],
            "hz": int(round(1.0 / float(config["frame_period_s"]))),
            "render_audit": "not_run_after_primary_reject",
        },
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cuda": "not_used_for_primary_geometry_gate",
        "gpu": "RTX 4090",
        "world_state_schema_version": WORLD_STATE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
        "safety_geometry_version": "continuous-obb-adaptive-v1",
        "observation_evidence_version": "observation-evidence-v2",
        "render_support_version": "source-checkpoint-count-plus-projected-visibility-v1",
        "certificate_version": "D1-occgs-certificate-v1",
        "scenario_effect_version": "scenario-effect-v1",
        "provenance_version": "h1-matched-pilot-v1",
        "recovery_policy": "unknown_never_coerced_to_pass",
        "started_at": utc_now(),
        "ended_at": None,
        "exit_reason": None,
        "terminal_status": "running",
        "status": "running",
    }
    contract = V71RunContract.initialize(args.run_root, run_id, manifest)
    run_dir = contract.run_dir
    atomic_write_text(
        str(run_dir / "resolved.yaml"), args.config.read_text(encoding="utf-8")
    )
    atomic_write_json(str(run_dir / "fingerprints" / "code.json"), code)
    atomic_write_json(
        str(run_dir / "fingerprints" / "environment.json"), manifest["environment"]
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "data.json"),
        {
            "proposal_bank_sha256": bank["proposal_bank_sha256"],
            "matched_eval_sha256": eval_summary["matched_eval_sha256"],
            "aggregate_sha256": aggregate["aggregate_sha256"],
            "full_index_sha256": full_index["index_sha256"],
        },
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "third_party.json"),
        {"drivestudio_commit": manifest["third_party_commit"]},
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "checkpoints.json"), checkpoint_hashes
    )
    shutil.copy2(proposal_path, run_dir / "proposal_bank.json")
    atomic_write_json(
        str(run_dir / "actor_registry.json"),
        {
            "selected_actor_ids": actor_ids,
            "actor_count_by_scene": bank["actor_count_by_scene"],
            "checkpoint_hashes": checkpoint_hashes,
        },
    )
    atomic_write_json(
        str(run_dir / "world_state_manifest.json"),
        {
            "world_state_hash": world_state_hash,
            "trajectory_world_state_hashes": world_hashes,
        },
    )
    atomic_write_json(
        str(run_dir / "render_request_manifest.json"),
        {
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
            "requests": audit_requests,
            "status": "not_run_after_primary_reject",
        },
    )
    for source, relative, _ in sources:
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_write_json(
        str(run_dir / "artifact_manifest.json"),
        {
            "schema_version": 1,
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
            "artifact_set_hash": artifact_set_hash,
            "artifacts": artifact_rows,
        },
    )
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(
            {
                "time": utc_now(),
                "step": 0,
                "proposal_count": aggregate["proposal_count"],
                "D1_precision": aggregate["D1"]["precision"],
                "D1_recall": aggregate["D1"]["recall"],
                "D1_abstention": aggregate["D1"]["abstention"],
                "D2_usable_yield": aggregate["D2_usable_yield"],
                "C_D1_identity_rate": aggregate["C_D1_identity"]["rate"],
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    summary = {
        "task_id": "V7-H1-11D",
        "run_id": run_id,
        "engineering_gate": "PASS",
        "hypothesis_verdict": "REJECTED" if marker == "REJECTED" else marker,
        "h1_cert_verdict": aggregate["h1_cert_verdict"],
        "h1_proj_verdict": aggregate["h1_proj_verdict"],
        "pilot_gate": aggregate["gate"],
        "D1": aggregate["D1"],
        "C_external": aggregate["C_external"],
        "D2_external": aggregate["D2_external"],
        "D2_usable_yield": aggregate["D2_usable_yield"],
        "C_D1_identity": aggregate["C_D1_identity"],
        "requested_effect_distribution": aggregate["requested_effect_distribution"],
        "counterfactual_pair_count": aggregate["counterfactual_pair_count"],
        "per_scene": aggregate["per_scene"],
        "aggregation_repair": aggregate["aggregation_repair"],
        "render_audit_status": diagnostic["render_audit_status"],
        "blind_review_pack_status": diagnostic["blind_review_pack_status"],
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
        "full_artifact_index_sha256": full_index["index_sha256"],
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(
        str(run_dir / "logs" / "finalize.log"),
        (
            "H1-11D stopped at the pre-registered primary gate. "
            "D1 precision was below threshold and D2 rejected every proposal; "
            "high-cost render and blind review pack were not instantiated.\n"
        ),
    )
    contract.finalize(marker, exit_reason=exit_reason)
    print(json.dumps({"run_dir": str(run_dir), "terminal_marker": marker, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()

