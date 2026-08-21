"""WorldSim V6 R88: calibrate a cheap selective gate for full-episode perception impact."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R88-SELECTIVE-PERCEPTION-RUNTIME-GATE-01"


class R88ExperimentError(RuntimeError):
    """The preregistered R88 experiment contract was violated."""


def _metrics(targets: list[bool], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(target and prediction for target, prediction in zip(targets, predictions))
    fp = sum((not target) and prediction for target, prediction in zip(targets, predictions))
    fn = sum(target and (not prediction) for target, prediction in zip(targets, predictions))
    tn = len(targets) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_count": tp + fp,
        "target_positive_count": tp + fn,
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R88ExperimentError("formal R88 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R88ExperimentError("R88 task_id drift")
    sources = config["sources"]
    protocol = config["protocol"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    r86_run = _resolve_runs_uri(sources["r86_run"])
    r87_run = _resolve_runs_uri(sources["r87_run"])
    frozen_files = {
        r86_run / "MANIFEST.json": sources["r86_manifest_sha256"],
        r86_run / "R86_GATE.json": sources["r86_gate_sha256"],
        r86_run / "FULL_EPISODE_SENSOR_EFFECT.json": sources[
            "r86_full_episode_sensor_effect_sha256"
        ],
        r87_run / "MANIFEST.json": sources["r87_manifest_sha256"],
        r87_run / "R87_GATE.json": sources["r87_gate_sha256"],
        r87_run / "SUMMARY.json": sources["r87_summary_sha256"],
        r87_run / "FULL_EPISODE_PERCEPTION_IMPACT.json": sources[
            "r87_full_episode_perception_impact_sha256"
        ],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R88ExperimentError("R88 disk resource insufficient")
    r86_gate = json.loads((r86_run / "R86_GATE.json").read_text(encoding="utf-8"))
    r87_gate = json.loads((r87_run / "R87_GATE.json").read_text(encoding="utf-8"))
    sensor = json.loads(
        (r86_run / "FULL_EPISODE_SENSOR_EFFECT.json").read_text(encoding="utf-8")
    )
    perception = json.loads(
        (r87_run / "FULL_EPISODE_PERCEPTION_IMPACT.json").read_text(encoding="utf-8")
    )
    frames = list(range(int(protocol["frame_start"]), int(protocol["frame_stop_exclusive"])))
    calibration_frames = [
        frame
        for frame in frames
        if frame % int(protocol["split_modulus"]) not in protocol["holdout_remainders"]
    ]
    holdout_frames = [frame for frame in frames if frame not in set(calibration_frames)]
    if len(calibration_frames) != int(protocol["expected_calibration_count"]) or len(
        holdout_frames
    ) != int(protocol["expected_holdout_count"]):
        raise R88ExperimentError("R88 split denominator drift")
    features = {
        frame: int(sensor["edited_vs_logged_changed_rgb_pixels_by_frame"][str(frame)])
        for frame in frames
    }
    targets = {
        frame: int(perception["changed_label_pixels_by_frame"][str(frame)])
        >= int(protocol["minimum_changed_label_pixels_for_target"])
        for frame in frames
    }
    validity = sensor["validity_sequences"]

    candidate_thresholds = sorted({features[frame] for frame in calibration_frames})
    candidate_thresholds.append(max(candidate_thresholds) + 1)
    calibration_rows = []
    for candidate in candidate_thresholds:
        metrics = _metrics(
            [targets[frame] for frame in calibration_frames],
            [features[frame] >= candidate for frame in calibration_frames],
        )
        calibration_rows.append({"threshold_pixels": candidate, **metrics})
    selected = max(
        calibration_rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            -int(row["threshold_pixels"]),
        ),
    )
    selected_threshold = int(selected["threshold_pixels"])

    def evaluate(frame_subset: list[int], predicate) -> dict[str, float | int]:
        return _metrics(
            [targets[frame] for frame in frame_subset],
            [bool(predicate(frame)) for frame in frame_subset],
        )

    holdout_metrics = {
        "selected_sensor_change_gate": evaluate(
            holdout_frames, lambda frame: features[frame] >= selected_threshold
        ),
        "fixed_r86_visible_256px": evaluate(
            holdout_frames,
            lambda frame: features[frame] >= int(protocol["fixed_visible_threshold_pixels"]),
        ),
        "any_edited_actor_lifecycle_active": evaluate(
            holdout_frames,
            lambda frame: any(bool(validity[actor_id][frame]) for actor_id in validity),
        ),
        "all_three_actor_lifecycles_active": evaluate(
            holdout_frames,
            lambda frame: all(bool(validity[actor_id][frame]) for actor_id in validity),
        ),
    }
    selected_holdout = holdout_metrics["selected_sensor_change_gate"]
    fixed_holdout = holdout_metrics["fixed_r86_visible_256px"]
    any_active_holdout = holdout_metrics["any_edited_actor_lifecycle_active"]
    skip_fraction_vs_any_active = 1.0 - float(
        selected_holdout["predicted_positive_count"]
    ) / max(1, int(any_active_holdout["predicted_positive_count"]))
    wall_seconds = time.monotonic() - started
    checks = {
        "r86_and_r87_authorities_accepted": bool(
            r86_gate["checks"]["passed"] and r87_gate["checks"]["passed"]
        ),
        "calibration_holdout_denominators_exact_and_disjoint": bool(
            len(set(calibration_frames) & set(holdout_frames)) == 0
            and sorted(calibration_frames + holdout_frames) == frames
        ),
        "threshold_selected_from_calibration_only": True,
        "holdout_precision_sufficient": float(selected_holdout["precision"])
        >= float(thresholds["minimum_holdout_precision"]),
        "holdout_recall_sufficient": float(selected_holdout["recall"])
        >= float(thresholds["minimum_holdout_recall"]),
        "holdout_f1_sufficient": float(selected_holdout["f1"])
        >= float(thresholds["minimum_holdout_f1"]),
        "no_worse_than_fixed_r86_visible_gate": float(selected_holdout["f1"])
        + float(thresholds["metric_tolerance"])
        >= float(fixed_holdout["f1"]),
        "expensive_perception_invocation_reduced_vs_any_active_lifecycle": skip_fraction_vs_any_active
        >= float(thresholds["minimum_skip_fraction_vs_any_active"]),
        "semantic_correctness_local_causality_physics_planning_safety_abstain": True,
        "frozen_sources_immutable": all(
            _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
        ),
        "cpu_only_within_wall_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__selective-perception-gate-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "SELECTIVE_GATE.json",
        {
            "schema_version": "worldsim_v6.r88_selective_gate.v1",
            "feature": "edited_vs_logged_rgb_changed_pixels",
            "target": "frozen_deeplab_changed_label_pixels_at_least_one",
            "split": {
                "modulus": int(protocol["split_modulus"]),
                "holdout_remainders": protocol["holdout_remainders"],
                "calibration_frame_count": len(calibration_frames),
                "holdout_frame_count": len(holdout_frames),
            },
            "selected_threshold_pixels": selected_threshold,
            "selected_calibration_metrics": selected,
            "holdout_metrics": holdout_metrics,
            "skip_fraction_vs_any_active_lifecycle": skip_fraction_vs_any_active,
        },
    )
    _write_json(
        run_dir / "CALIBRATION_SWEEP.json",
        {
            "schema_version": "worldsim_v6.r88_calibration_sweep.v1",
            "candidate_count": len(calibration_rows),
            "rows": calibration_rows,
        },
    )
    _write_json(
        run_dir / "R88_GATE.json",
        {
            "schema_version": "worldsim_v6.r88_gate.v1",
            "checks": checks,
            "decision": "accept_selective_perception_runtime_gate"
            if checks["passed"]
            else "reject_or_pivot_selective_perception_runtime_gate",
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r88_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_selective_perception_runtime_gate"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "selected_threshold_pixels": selected_threshold,
        "holdout_metrics": holdout_metrics,
        "skip_fraction_vs_any_active_lifecycle": skip_fraction_vs_any_active,
        "semantic_correctness": "ABSTAIN",
        "local_causality": "ABSTAIN",
        "physical_planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R88_GATE.json",
        "SUMMARY.json",
        "SELECTIVE_GATE.json",
        "CALIBRATION_SWEEP.json",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r88_manifest.v1",
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
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r88_selective_perception_runtime_gate_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
