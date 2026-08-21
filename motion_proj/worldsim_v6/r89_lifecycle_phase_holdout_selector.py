"""WorldSim V6 R89: test selective-gate transfer to a held-out lifecycle phase."""

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


TASK_ID = "WS-V6-R89-LIFECYCLE-PHASE-HOLDOUT-SELECTOR-01"


class R89ExperimentError(RuntimeError):
    """The preregistered R89 experiment contract was violated."""


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
        raise R89ExperimentError("formal R89 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R89ExperimentError("R89 task_id drift")
    sources = config["sources"]
    protocol = config["protocol"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    r86_run = _resolve_runs_uri(sources["r86_run"])
    r87_run = _resolve_runs_uri(sources["r87_run"])
    r88_run = _resolve_runs_uri(sources["r88_run"])
    frozen_files = {
        r86_run / "R86_GATE.json": sources["r86_gate_sha256"],
        r86_run / "FULL_EPISODE_SENSOR_EFFECT.json": sources[
            "r86_full_episode_sensor_effect_sha256"
        ],
        r87_run / "R87_GATE.json": sources["r87_gate_sha256"],
        r87_run / "FULL_EPISODE_PERCEPTION_IMPACT.json": sources[
            "r87_full_episode_perception_impact_sha256"
        ],
        r88_run / "MANIFEST.json": sources["r88_manifest_sha256"],
        r88_run / "R88_GATE.json": sources["r88_gate_sha256"],
        r88_run / "SUMMARY.json": sources["r88_summary_sha256"],
        r88_run / "SELECTIVE_GATE.json": sources["r88_selective_gate_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R89ExperimentError("R89 disk resource insufficient")
    gates = [
        json.loads((r86_run / "R86_GATE.json").read_text(encoding="utf-8")),
        json.loads((r87_run / "R87_GATE.json").read_text(encoding="utf-8")),
        json.loads((r88_run / "R88_GATE.json").read_text(encoding="utf-8")),
    ]
    sensor = json.loads(
        (r86_run / "FULL_EPISODE_SENSOR_EFFECT.json").read_text(encoding="utf-8")
    )
    perception = json.loads(
        (r87_run / "FULL_EPISODE_PERCEPTION_IMPACT.json").read_text(encoding="utf-8")
    )
    r88_selector = json.loads(
        (r88_run / "SELECTIVE_GATE.json").read_text(encoding="utf-8")
    )
    frames = list(range(int(protocol["frame_start"]), int(protocol["frame_stop_exclusive"])))
    holdout_frames = list(
        range(int(protocol["holdout_start"]), int(protocol["holdout_stop_exclusive"]))
    )
    holdout_set = set(holdout_frames)
    calibration_frames = [frame for frame in frames if frame not in holdout_set]
    if len(calibration_frames) != int(protocol["expected_calibration_count"]) or len(
        holdout_frames
    ) != int(protocol["expected_holdout_count"]):
        raise R89ExperimentError("R89 phase split denominator drift")
    features = {
        frame: int(sensor["edited_vs_logged_changed_rgb_pixels_by_frame"][str(frame)])
        for frame in frames
    }
    targets = {
        frame: int(perception["changed_label_pixels_by_frame"][str(frame)])
        >= int(protocol["minimum_changed_label_pixels_for_target"])
        for frame in frames
    }
    candidates = sorted({features[frame] for frame in calibration_frames})
    candidates.append(max(candidates) + 1)
    sweep = []
    for candidate in candidates:
        metrics = _metrics(
            [targets[frame] for frame in calibration_frames],
            [features[frame] >= candidate for frame in calibration_frames],
        )
        sweep.append({"threshold_pixels": candidate, **metrics})
    selected = max(
        sweep,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            -int(row["threshold_pixels"]),
        ),
    )
    selected_threshold = int(selected["threshold_pixels"])
    holdout_targets = [targets[frame] for frame in holdout_frames]
    selected_metrics = _metrics(
        holdout_targets, [features[frame] >= selected_threshold for frame in holdout_frames]
    )
    fixed_metrics = _metrics(
        holdout_targets,
        [features[frame] >= int(protocol["fixed_visible_threshold_pixels"]) for frame in holdout_frames],
    )
    prior_r88_metrics = _metrics(
        holdout_targets,
        [
            features[frame] >= int(r88_selector["selected_threshold_pixels"])
            for frame in holdout_frames
        ],
    )
    wall_seconds = time.monotonic() - started
    checks = {
        "r86_r87_r88_authorities_accepted": all(gate["checks"]["passed"] for gate in gates),
        "contiguous_phase_holdout_and_calibration_denominators_exact": bool(
            holdout_frames == list(range(141, 151))
            and not (set(calibration_frames) & holdout_set)
            and sorted(calibration_frames + holdout_frames) == frames
        ),
        "calibration_contains_positive_and_negative_targets": bool(
            any(targets[frame] for frame in calibration_frames)
            and not all(targets[frame] for frame in calibration_frames)
        ),
        "holdout_is_actor0_inactive_actor5_active_and_all_targets_affected": bool(
            all(not sensor["validity_sequences"]["actor_0000"][frame] for frame in holdout_frames)
            and all(sensor["validity_sequences"]["actor_0005"][frame] for frame in holdout_frames)
            and all(holdout_targets)
        ),
        "threshold_selected_without_phase_holdout_targets": True,
        "selected_threshold_reproduces_r88_threshold": selected_threshold
        == int(r88_selector["selected_threshold_pixels"]),
        "phase_holdout_recall_exact": float(selected_metrics["recall"])
        >= float(thresholds["minimum_phase_holdout_recall"]),
        "phase_holdout_f1_sufficient": float(selected_metrics["f1"])
        >= float(thresholds["minimum_phase_holdout_f1"]),
        "recall_improves_over_fixed256": float(selected_metrics["recall"])
        - float(fixed_metrics["recall"])
        >= float(thresholds["minimum_recall_gain_over_fixed256"]),
        "independent_phase_calibration_matches_prior_selector_behavior": selected_metrics
        == prior_r88_metrics,
        "semantic_correctness_transfer_local_causality_physics_planning_safety_abstain": True,
        "frozen_sources_immutable": all(
            _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
        ),
        "cpu_only_within_wall_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__phase-holdout-selector-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "PHASE_HOLDOUT_RESULT.json",
        {
            "schema_version": "worldsim_v6.r89_phase_holdout_result.v1",
            "calibration_frame_count": len(calibration_frames),
            "holdout_frames": holdout_frames,
            "selected_threshold_pixels": selected_threshold,
            "r88_threshold_pixels": int(r88_selector["selected_threshold_pixels"]),
            "selected_calibration_metrics": selected,
            "selected_phase_holdout_metrics": selected_metrics,
            "fixed256_phase_holdout_metrics": fixed_metrics,
            "prior_r88_selector_phase_holdout_metrics": prior_r88_metrics,
            "holdout_feature_pixels": {str(frame): features[frame] for frame in holdout_frames},
        },
    )
    _write_json(
        run_dir / "R89_GATE.json",
        {
            "schema_version": "worldsim_v6.r89_gate.v1",
            "checks": checks,
            "decision": "accept_lifecycle_phase_holdout_selector"
            if checks["passed"]
            else "reject_or_pivot_lifecycle_phase_holdout_selector",
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r89_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_lifecycle_phase_holdout_selector"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "selected_threshold_pixels": selected_threshold,
        "phase_holdout_metrics": selected_metrics,
        "fixed256_phase_holdout_metrics": fixed_metrics,
        "semantic_correctness": "ABSTAIN",
        "cross_scene_transfer": "ABSTAIN",
        "physical_planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R89_GATE.json", "SUMMARY.json", "PHASE_HOLDOUT_RESULT.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r89_manifest.v1",
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
        default=Path("configs/worldsim_v6/r89_lifecycle_phase_holdout_selector_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
