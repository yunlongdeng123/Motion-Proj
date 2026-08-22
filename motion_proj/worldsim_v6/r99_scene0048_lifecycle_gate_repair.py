"""WorldSim V6 R99: repair the scene0048 lifecycle-aware sensor gate."""

from __future__ import annotations

import argparse
import json
import sys
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
from motion_proj.worldsim_v6.r98_second_independent_scene_selector_transfer import (
    _load_rows,
    _metrics,
    _sensor_pass,
)


TASK_ID = "WS-V6-R99-SCENE0048-LIFECYCLE-GATE-REPAIR-01"


class R99ExperimentError(RuntimeError):
    """The preregistered R99 governance-repair contract was violated."""


def _same_metric(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left[key] == right[key] for key in left) and set(left) == set(right)


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R99ExperimentError("formal R99 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R99ExperimentError("R99 task_id drift")
    sources = config["sources"]
    expected = config["expected"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    r98_run = _resolve_runs_uri(sources["r98_run"])
    r97_run = _resolve_runs_uri(sources["r97_run"])
    r97_package = r97_run / "package"
    frozen_files = {
        r98_run / "MANIFEST.json": sources["r98_manifest_sha256"],
        r98_run / "R98_GATE.json": sources["r98_gate_sha256"],
        r98_run / "SUMMARY.json": sources["r98_summary_sha256"],
        r98_run / "SELECTOR_TRANSFER.json": sources["r98_selector_transfer_sha256"],
        r97_run / "MANIFEST.json": sources["r97_manifest_sha256"],
        r97_run / "R97_GATE.json": sources["r97_gate_sha256"],
        r97_package / "PACKAGE_MANIFEST.json": sources["r97_package_manifest_sha256"],
    }
    for path, digest in frozen_files.items():
        _verify(path, digest)

    r98_manifest = json.loads((r98_run / "MANIFEST.json").read_text(encoding="utf-8"))
    r98_artifact_files: dict[Path, str] = {}
    for relative, row in r98_manifest["files"].items():
        path = r98_run / relative
        _verify(path, row["sha256"])
        r98_artifact_files[path] = row["sha256"]
    r97_package_manifest = json.loads(
        (r97_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    r97_package_files: dict[Path, str] = {}
    for relative, row in r97_package_manifest["files"].items():
        path = r97_package / relative
        _verify(path, row["sha256"])
        r97_package_files[path] = row["sha256"]

    r98_gate = json.loads((r98_run / "R98_GATE.json").read_text(encoding="utf-8"))
    transfer = json.loads((r98_run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8"))
    sensor_rows = _load_rows(r98_run / "sensor_worker/FRAME_METRICS.jsonl")
    perception_rows = _load_rows(r98_run / "perception/PERCEPTION_OUTPUTS.jsonl")
    geometry = json.loads((r97_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
    lifecycle_ref = geometry["arrays"]["actor_frame_validity"]
    lifecycle_path = r97_package / lifecycle_ref["path"]
    _verify(lifecycle_path, sources["lifecycle_sha256"])
    lifecycle = np.load(lifecycle_path, allow_pickle=False).astype(bool)

    frame_count = int(expected["frame_count"])
    frame_indices = list(range(frame_count))
    if [int(row["frame_index"]) for row in sensor_rows] != frame_indices:
        raise R99ExperimentError("R98 sensor denominator/order drift")
    actor_id = str(expected["actor_id"])
    sensor_conformant = [
        _sensor_pass(row, actor_id, thresholds)
        and bool(row["compiled_repeat_exact"])
        and bool(row["native_translation_state_restored_exact"])
        for row in sensor_rows
    ]
    package_validity = np.asarray(
        [bool(row["actors"][actor_id]["package_actor_frame_valid"]) for row in sensor_rows],
        dtype=bool,
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in perception_rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    repeat_exact = True
    changed_labels: list[int] = []
    for frame in frame_indices:
        labels: dict[str, np.ndarray] = {}
        for variant in ("logged", "edited"):
            case_id = f"frame{frame:03d}_{variant}"
            items = sorted(grouped.get(case_id, []), key=lambda row: int(row["repeat_index"]))
            repeat_exact = repeat_exact and len(items) == int(expected["repeat_count"])
            repeat_exact = repeat_exact and len({row["label_array_sha256"] for row in items}) == 1
            labels[variant] = np.load(
                r98_run / "perception" / items[0]["label_path"], allow_pickle=False
            )
        changed_labels.append(int((labels["edited"] != labels["logged"]).sum()))

    sensor_changed = np.asarray(
        [int(row["edited_vs_logged_rgb_changed_pixels"]) for row in sensor_rows], dtype=np.int64
    )
    changed_labels_array = np.asarray(changed_labels, dtype=np.int64)
    target = changed_labels_array >= int(expected["target_minimum_changed_label_pixels"])
    frozen = sensor_changed >= int(expected["frozen_threshold_pixels"])
    fixed = sensor_changed >= int(expected["fixed_threshold_pixels"])
    frozen_metrics = _metrics(frozen, target)
    fixed_metrics = _metrics(fixed, target)
    lifecycle_metrics = _metrics(lifecycle, target)
    r98_failed_checks = sorted(name for name, value in r98_gate["checks"].items() if not value)
    transfer_exact = (
        sensor_changed.astype(int).tolist()
        == [int(transfer["sensor_changed_pixels_by_frame"][str(frame)]) for frame in frame_indices]
        and changed_labels == [
            int(transfer["changed_label_pixels_by_frame"][str(frame)]) for frame in frame_indices
        ]
        and _same_metric(frozen_metrics, transfer["frozen_policy_metrics"])
        and _same_metric(fixed_metrics, transfer["fixed256_metrics"])
        and _same_metric(lifecycle_metrics, transfer["native_lifecycle_metrics"])
    )

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__scene0048-lifecycle-gate-repair-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    repair = {
        "schema_version": "worldsim_v6.r99_lifecycle_gate_repair.v1",
        "source_r98_status": "rejected",
        "source_r98_failed_checks": r98_failed_checks,
        "source_r98_retroactively_accepted": False,
        "rejected_contract": "require_package_actor_frame_valid_true_on_all196_frames",
        "repaired_contract": "require_package_actor_frame_valid_equal_frozen_native_lifecycle_per_frame",
        "frame_count": frame_count,
        "native_active_frame_count": int(lifecycle.sum()),
        "native_inactive_frame_count": int((~lifecycle).sum()),
        "sensor_conformant_frame_count": int(sum(sensor_conformant)),
        "package_lifecycle_match_frame_count": int((package_validity == lifecycle).sum()),
        "frozen_policy_metrics": frozen_metrics,
        "fixed256_metrics": fixed_metrics,
        "native_lifecycle_metrics": lifecycle_metrics,
    }
    _write_json(run_dir / "LIFECYCLE_GATE_REPAIR.json", repair)
    checks = {
        "r98_rejection_retained": r98_gate["checks"]["passed"] is False
        and r98_failed_checks
        == ["all196_compiled_native_sensor_conformant", "passed"],
        "r97_lifecycle_exact": lifecycle.shape == (frame_count,)
        and int(lifecycle.sum()) == int(expected["active_frame_count"]),
        "all196_sensor_numerics_conformant": all(sensor_conformant),
        "all196_package_validity_matches_native_lifecycle": np.array_equal(
            package_validity, lifecycle
        ),
        "perception_denominator_and_repeat_exact": len(perception_rows)
        == frame_count * 2 * int(expected["repeat_count"])
        and repeat_exact,
        "r98_selector_outputs_recomputed_exact": transfer_exact,
        "zero_calibration_selector_metrics_exact": frozen_metrics["true_positive"]
        == int(expected["true_positive"])
        and frozen_metrics["true_negative"] == int(expected["true_negative"])
        and frozen_metrics["false_positive"] == 0
        and frozen_metrics["false_negative"] == 0,
        "threshold45_strictly_improves_fixed256_and_native_lifecycle_f1": frozen_metrics["f1"]
        > fixed_metrics["f1"]
        and frozen_metrics["f1"] > lifecycle_metrics["f1"],
        "all_frozen_artifacts_immutable": all(
            _sha256(path) == digest for path, digest in frozen_files.items()
        )
        and all(_sha256(path) == digest for path, digest in r98_artifact_files.items())
        and all(_sha256(path) == digest for path, digest in r97_package_files.items()),
        "no_training_model_inference_or_confirmation_read": "torch" not in sys.modules,
        "wall_within_budget": time.monotonic() - started
        <= float(resources["maximum_wall_seconds"]),
        "semantic_correctness_local_causality_physics_planning_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R99_GATE.json",
        {
            "schema_version": "worldsim_v6.r99_gate.v1",
            "checks": checks,
            "decision": "accept_scene0048_lifecycle_gate_repair"
            if checks["passed"]
            else "reject_scene0048_lifecycle_gate_repair",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r99_resource_audit.v1",
            "wall_seconds": time.monotonic() - started,
            "gpu_used": False,
            "model_inference_started": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    _write_json(
        run_dir / "SUMMARY.json",
        {
            "schema_version": "worldsim_v6.r99_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_lifecycle_gate_repair"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "source_r98_status_retained": "rejected",
            "frame_count": frame_count,
            "active_frame_count": int(lifecycle.sum()),
            "sensor_conformant_frame_count": int(sum(sensor_conformant)),
            "package_lifecycle_match_frame_count": int((package_validity == lifecycle).sum()),
            "precision": frozen_metrics["precision"],
            "recall": frozen_metrics["recall"],
            "f1": frozen_metrics["f1"],
            "skip_fraction": frozen_metrics["skip_fraction"],
            "fixed256_f1": fixed_metrics["f1"],
            "native_lifecycle_f1": lifecycle_metrics["f1"],
            "claim_boundary": config["claim_boundary"],
        },
    )
    tracked = ["R99_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "LIFECYCLE_GATE_REPAIR.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r99_manifest.v1",
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
            "status": "done" if checks["passed"] else "rejected",
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        },
    )
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r99_scene0048_lifecycle_gate_repair_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
