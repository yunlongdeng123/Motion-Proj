"""WorldSim V6 R115: immutable 1-to-3 actor selective-runtime scaling envelope."""

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


TASK_ID = "WS-V6-R115-SCENE0255-COMPOSITION-SCALING-ENVELOPE-01"


class R115ExperimentError(RuntimeError):
    """The preregistered R115 contract was violated."""


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
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
        raise R115ExperimentError("formal R115 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R115ExperimentError("R115 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    stage_specs = [
        ("actor34_only", "r102", "R102_GATE.json"),
        ("actor34_actor24", "r109", "R109_GATE.json"),
        ("actor34_actor24_actor9", "r114", "R114_GATE.json"),
    ]
    frozen_files: dict[Path, str] = {}
    runs: dict[str, Path] = {}
    for name, prefix, gate_name in stage_specs:
        run = _resolve_runs_uri(sources[f"{prefix}_run"])
        runs[name] = run
        frozen_files.update(
            {
                run / "MANIFEST.json": sources[f"{prefix}_manifest_sha256"],
                run / gate_name: sources[f"{prefix}_gate_sha256"],
                run / "SUMMARY.json": sources[f"{prefix}_summary_sha256"],
                run / "SELECTOR_TRANSFER.json": sources[f"{prefix}_selector_transfer_sha256"],
            }
        )
    for path, expected in frozen_files.items():
        _verify(path, expected)
    gates = {
        name: json.loads((runs[name] / gate_name).read_text(encoding="utf-8"))
        for name, _, gate_name in stage_specs
    }
    transfers = {
        name: json.loads((runs[name] / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8"))
        for name, _, _ in stage_specs
    }
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R115ExperimentError("R115 disk resource insufficient")
    frame_indices = list(range(int(runtime["expected_frame_count"])))
    stage_rows: list[dict[str, Any]] = []
    target_masks: dict[str, np.ndarray] = {}
    selector_masks: dict[str, np.ndarray] = {}
    threshold_pixels = int(runtime["frozen_policy_threshold_pixels"])
    for name, _, _ in stage_specs:
        transfer = transfers[name]
        sensor = np.asarray(
            [int(transfer["sensor_changed_pixels_by_frame"][str(frame)]) for frame in frame_indices],
            dtype=np.int64,
        )
        labels = np.asarray(
            [int(transfer["changed_label_pixels_by_frame"][str(frame)]) for frame in frame_indices],
            dtype=np.int64,
        )
        target = labels >= int(runtime["minimum_changed_label_pixels"])
        selector = sensor >= threshold_pixels
        target_masks[name] = target
        selector_masks[name] = selector
        metrics = _metrics(selector, target)
        stage_rows.append(
            {
                "stage": name,
                "edited_actor_count": int(runtime["edited_actor_count_by_stage"][name]),
                "positive_frames": int(target.sum()),
                "negative_frames": int((~target).sum()),
                "selector_metrics": metrics,
                "total_changed_label_pixels": int(labels.sum()),
                "total_sensor_changed_pixels": int(sensor.sum()),
            }
        )
    marginal_rows = []
    for previous, current in zip(stage_rows, stage_rows[1:]):
        previous_name = previous["stage"]
        current_name = current["stage"]
        added = target_masks[current_name] & ~target_masks[previous_name]
        removed = target_masks[previous_name] & ~target_masks[current_name]
        marginal_rows.append(
            {
                "transition": f"{previous_name}_to_{current_name}",
                "added_positive_frames": int(added.sum()),
                "removed_positive_frames": int(removed.sum()),
                "remaining_negative_frames": int((~target_masks[current_name]).sum()),
                "skip_fraction_drop": float(
                    previous["selector_metrics"]["skip_fraction"]
                    - current["selector_metrics"]["skip_fraction"]
                ),
            }
        )
    certificate = {
        "schema_version": "worldsim_v6.r115_composition_scaling_envelope.v1",
        "scene": runtime["scene"],
        "frame_count": len(frame_indices),
        "frozen_policy_threshold_pixels": threshold_pixels,
        "stages": stage_rows,
        "marginal_transitions": marginal_rows,
        "observed_skip_floor_at_three_actors": stage_rows[-1]["selector_metrics"]["skip_fraction"],
        "observed_remaining_negative_frames_at_three_actors": stage_rows[-1]["negative_frames"],
        "extrapolation_to_four_actors": "ABSTAIN",
        "semantic_correctness_local_causality_contact_dynamics_physics_planning_safety": "ABSTAIN",
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__composition-scaling-envelope-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "SCALING_ENVELOPE.json", certificate)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    expected_counts = runtime["expected_positive_frames_by_stage"]
    checks = {
        "r102_r109_r114_authorities_accepted": all(gate["checks"]["passed"] for gate in gates.values()),
        "source_artifacts_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "three_stage_denominators_exact": all(
            int(transfers[row["stage"]]["frame_count"]) == len(frame_indices) for row in stage_rows
        ),
        "positive_frame_counts_exact": all(
            row["positive_frames"] == int(expected_counts[row["stage"]]) for row in stage_rows
        ),
        "threshold45_zero_error_all_stages": all(
            row["selector_metrics"]["f1"] == 1.0
            and row["selector_metrics"]["false_positive"] == 0
            and row["selector_metrics"]["false_negative"] == 0
            for row in stage_rows
        ),
        "target_support_nested_under_actor_addition": all(
            row["removed_positive_frames"] == 0 for row in marginal_rows
        ),
        "each_actor_addition_has_nontrivial_marginal_support": all(
            row["added_positive_frames"] >= int(thresholds["minimum_added_positive_frames"])
            for row in marginal_rows
        ),
        "skip_fraction_strictly_decreases": all(
            row["skip_fraction_drop"] > 0.0 for row in marginal_rows
        ),
        "three_actor_skip_floor_still_nonzero": stage_rows[-1]["negative_frames"]
        >= int(thresholds["minimum_remaining_negative_frames"])
        and stage_rows[-1]["selector_metrics"]["skip_fraction"]
        >= float(thresholds["minimum_three_actor_skip_fraction"]),
        "four_actor_extrapolation_semantics_and_physics_abstain": True,
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R115_GATE.json",
        {
            "schema_version": "worldsim_v6.r115_gate.v1",
            "checks": checks,
            "decision": "accept_observed_one_to_three_actor_scaling_envelope"
            if checks["passed"]
            else "reject_composition_scaling_envelope",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r115_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r115_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_observed_one_to_three_actor_scaling_envelope"
        if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "positive_frames_by_stage": {row["stage"]: row["positive_frames"] for row in stage_rows},
        "skip_fraction_by_stage": {
            row["stage"]: row["selector_metrics"]["skip_fraction"] for row in stage_rows
        },
        "added_positive_frames_by_transition": {
            row["transition"]: row["added_positive_frames"] for row in marginal_rows
        },
        "remaining_negative_frames": stage_rows[-1]["negative_frames"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R115_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "SCALING_ENVELOPE.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r115_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
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
        "--config", type=Path,
        default=Path("configs/worldsim_v6/r115_scene0255_composition_scaling_envelope_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
