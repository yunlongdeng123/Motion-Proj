"""WorldSim V6 R125: R109-digest recovery for the binary perception surrogate corpus."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R125-BINARY-PERCEPTION-SURROGATE-CORPUS-RECOVERY-01"


class R125ExperimentError(RuntimeError):
    """The preregistered R125 recovery contract was violated."""


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, ~target).sum())
    fn = int(np.logical_and(~predicted, target).sum())
    tn = int(np.logical_and(~predicted, ~target).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "skip_fraction": float((~predicted).mean()),
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R125ExperimentError("formal R125 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R125ExperimentError("R125 task_id drift")
    runtime = config["runtime"]
    thresholds = config["thresholds"]
    resources = config["resources"]

    policy_spec = config["policy_source"]
    policy_run = _resolve_runs_uri(policy_spec["run"])
    policy_path = policy_run / policy_spec["policy_path"]
    _verify(policy_run / "MANIFEST.json", policy_spec["manifest_sha256"])
    _verify(policy_run / policy_spec["gate_name"], policy_spec["gate_sha256"])
    _verify(policy_path, policy_spec["policy_sha256"])
    policy_gate = json.loads(
        (policy_run / policy_spec["gate_name"]).read_text(encoding="utf-8")
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    frozen_threshold = int(policy["threshold_pixels"])

    condition_rows: list[dict[str, Any]] = []
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    frozen_files: dict[Path, str] = {
        policy_run / "MANIFEST.json": policy_spec["manifest_sha256"],
        policy_run / policy_spec["gate_name"]: policy_spec["gate_sha256"],
        policy_path: policy_spec["policy_sha256"],
    }
    authority_gates = []
    expected_frames = int(runtime["expected_frame_count_per_condition"])
    condition_specs = config["conditions"]
    for condition_name in runtime["condition_order"]:
        spec = condition_specs[condition_name]
        transfer_run = _resolve_runs_uri(spec["transfer_run"])
        authority_run = _resolve_runs_uri(spec.get("authority_run", spec["transfer_run"]))
        transfer_files = {
            transfer_run / "MANIFEST.json": spec["transfer_manifest_sha256"],
            transfer_run / "SUMMARY.json": spec["transfer_summary_sha256"],
            transfer_run / "SELECTOR_TRANSFER.json": spec["selector_transfer_sha256"],
            authority_run / spec["authority_gate_name"]: spec["authority_gate_sha256"],
        }
        if "authority_manifest_sha256" in spec:
            transfer_files[authority_run / "MANIFEST.json"] = spec["authority_manifest_sha256"]
        for path, expected in transfer_files.items():
            _verify(path, expected)
            frozen_files[path] = expected
        gate = json.loads(
            (authority_run / spec["authority_gate_name"]).read_text(encoding="utf-8")
        )
        authority_gates.append(gate)
        transfer = json.loads(
            (transfer_run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8")
        )
        if int(transfer["frame_count"]) != expected_frames:
            raise R125ExperimentError(f"{condition_name} frame denominator drift")
        features = np.asarray(
            [
                int(transfer["sensor_changed_pixels_by_frame"][str(frame)])
                for frame in range(expected_frames)
            ],
            dtype=np.int64,
        )
        label_counts = np.asarray(
            [
                int(transfer["changed_label_pixels_by_frame"][str(frame)])
                for frame in range(expected_frames)
            ],
            dtype=np.int64,
        )
        target = label_counts >= int(runtime["minimum_changed_label_pixels"])
        predicted = features >= frozen_threshold
        metrics = _metrics(predicted, target)
        negative_values = features[~target]
        positive_values = features[target]
        condition_rows.append(
            {
                "condition": condition_name,
                "scene": spec["scene"],
                "actor_count": int(spec["actor_count"]),
                "direction": spec["direction"],
                "magnitude_m": float(spec["magnitude_m"]),
                "frame_count": expected_frames,
                "positive_frames": int(target.sum()),
                "negative_frames": int((~target).sum()),
                "minimum_positive_feature": int(positive_values.min())
                if positive_values.size
                else None,
                "maximum_negative_feature": int(negative_values.max())
                if negative_values.size
                else None,
                "threshold45_metrics": metrics,
                "calibration_frames_in_condition": int(
                    transfer.get("calibration_frames_in_target_scene", 0)
                ),
            }
        )
        all_features.append(features)
        all_targets.append(target)

    features = np.concatenate(all_features)
    target = np.concatenate(all_targets)
    predicted = features >= frozen_threshold
    corpus_metrics = _metrics(predicted, target)
    max_negative = int(features[~target].max())
    min_positive = int(features[target].min())
    exact_threshold_interval = [max_negative + 1, min_positive]
    certificate = {
        "schema_version": "worldsim_v6.r125_binary_perception_surrogate_corpus_recovery.v1",
        "policy_id": policy["policy_id"],
        "feature": policy["feature"],
        "comparator": policy["comparator"],
        "frozen_threshold_pixels": frozen_threshold,
        "condition_count": len(condition_rows),
        "total_frame_count": int(features.size),
        "scene_count": len({row["scene"] for row in condition_rows}),
        "condition_rows": condition_rows,
        "corpus_metrics": corpus_metrics,
        "corpus_positive_frames": int(target.sum()),
        "corpus_negative_frames": int((~target).sum()),
        "maximum_negative_feature": max_negative,
        "minimum_positive_feature": min_positive,
        "exact_integer_threshold_interval": exact_threshold_interval,
        "binary_frozen_deeplab_impact_equivalent_on_bound_corpus": bool(
            np.array_equal(predicted, target)
        ),
        "prospective_unseen_condition_generalization": "ABSTAIN",
        "semantic_correctness_label_identity_local_causality_physics_planning_safety": "ABSTAIN",
    }
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R125ExperimentError("R125 disk resource insufficient")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__binary-perception-surrogate-corpus-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "SURROGATE_CORPUS_CERTIFICATE.json", certificate)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "r90_frozen_policy_authority_accepted": bool(policy_gate["checks"]["passed"])
        and policy["feature"] == "edited_vs_logged_rgb_changed_pixels"
        and policy["comparator"] == "greater_than_or_equal"
        and frozen_threshold == int(runtime["expected_frozen_threshold_pixels"]),
        "all_condition_authorities_accepted": all(
            gate["checks"]["passed"] for gate in authority_gates
        ),
        "source_files_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "condition_and_frame_denominators_exact": len(condition_rows)
        == int(runtime["expected_condition_count"])
        and int(features.size)
        == expected_frames * int(runtime["expected_condition_count"]),
        "coverage_axes_nontrivial": len({row["scene"] for row in condition_rows})
        >= int(thresholds["minimum_scene_count"])
        and len({row["actor_count"] for row in condition_rows})
        >= int(thresholds["minimum_actor_count_levels"])
        and len({row["direction"] for row in condition_rows})
        >= int(thresholds["minimum_direction_count"])
        and len({row["magnitude_m"] for row in condition_rows})
        >= int(thresholds["minimum_magnitude_count"]),
        "positive_and_negative_corpus_support_nontrivial": int(target.sum())
        >= int(thresholds["minimum_positive_frames"])
        and int((~target).sum()) >= int(thresholds["minimum_negative_frames"]),
        "frozen_threshold45_exact_every_condition": all(
            row["threshold45_metrics"]["false_positive"] == 0
            and row["threshold45_metrics"]["false_negative"] == 0
            and row["threshold45_metrics"]["f1"] == 1.0
            for row in condition_rows
        ),
        "frozen_threshold45_exact_on_full_corpus": corpus_metrics["false_positive"] == 0
        and corpus_metrics["false_negative"] == 0
        and corpus_metrics["f1"] == 1.0,
        "exact_threshold_interval_contains45_with_margin": exact_threshold_interval[0]
        <= frozen_threshold
        <= exact_threshold_interval[1]
        and exact_threshold_interval[1] - frozen_threshold
        >= int(thresholds["minimum_positive_side_margin_pixels"]),
        "zero_target_condition_calibration": all(
            row["calibration_frames_in_condition"] == 0 for row in condition_rows
        ),
        "binary_equivalence_is_corpus_bound_and_unseen_semantics_abstain": True,
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R125_GATE.json",
        {
            "schema_version": "worldsim_v6.r125_gate.v1",
            "checks": checks,
            "decision": "accept_corpus_bound_binary_perception_surrogate"
            if checks["passed"]
            else "reject_binary_perception_surrogate_corpus",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r125_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r125_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_corpus_bound_binary_perception_surrogate"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "condition_count": len(condition_rows),
        "total_frame_count": int(features.size),
        "corpus_positive_frames": int(target.sum()),
        "corpus_negative_frames": int((~target).sum()),
        "precision": corpus_metrics["precision"],
        "recall": corpus_metrics["recall"],
        "f1": corpus_metrics["f1"],
        "exact_integer_threshold_interval": exact_threshold_interval,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R125_GATE.json",
        "SUMMARY.json",
        "RESOURCE_AUDIT.json",
        "SURROGATE_CORPUS_CERTIFICATE.json",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r125_manifest.v1",
            "files": {
                name: {
                    "bytes": (run_dir / name).stat().st_size,
                    "sha256": _sha256(run_dir / name),
                }
                for name in tracked
            },
        },
    )
    _write_json(
        run_dir / "TERMINAL.json",
        {
            "schema_version": "worldsim_v6.terminal.v1",
            "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        },
    )
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r125_binary_perception_surrogate_corpus_recovery_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
