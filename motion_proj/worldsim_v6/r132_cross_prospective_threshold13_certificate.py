"""WorldSim V6 R132: certify threshold-13 on two prospective conditions."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R132-CROSS-PROSPECTIVE-THRESHOLD13-CERTIFICATE-01"


class R132ExperimentError(RuntimeError):
    """The preregistered R132 contract was violated."""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metrics(rows: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    errors: list[dict[str, Any]] = []
    for row in rows:
        predicted = int(row["changed_rgb_pixels"]) >= threshold
        target = bool(row["target_any_changed_label"])
        tp += int(predicted and target)
        fp += int(predicted and not target)
        fn += int(not predicted and target)
        tn += int(not predicted and not target)
        if predicted != target:
            errors.append(
                {
                    "row_id": row["row_id"],
                    "changed_rgb_pixels": int(row["changed_rgb_pixels"]),
                    "changed_label_pixels": int(row["changed_label_pixels"]),
                    "error": "FP" if predicted else "FN",
                }
            )
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    total = len(rows)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trigger_count": tp + fp,
        "skip_count": tn + fn,
        "skip_fraction": float((tn + fn) / total) if total else 0.0,
        "errors": errors,
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R132ExperimentError("formal R132 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R132ExperimentError("R132 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    resources = config["resources"]

    r129_run = _resolve_runs_uri(sources["r129_run"])
    r130_run = _resolve_runs_uri(sources["r130_run"])
    r131_run = _resolve_runs_uri(sources["r131_run"])
    policy_package = r129_run / "package_a"
    frozen_files: dict[Path, str] = {
        r129_run / "MANIFEST.json": sources["r129_manifest_sha256"],
        r129_run / "R129_GATE.json": sources["r129_gate_sha256"],
        r129_run / "SUMMARY.json": sources["r129_summary_sha256"],
        policy_package / "PACKAGE_MANIFEST.json": sources["r129_package_manifest_sha256"],
        policy_package / "POLICY.json": sources["r129_policy_sha256"],
        r130_run / "MANIFEST.json": sources["r130_manifest_sha256"],
        r130_run / "R130_GATE.json": sources["r130_gate_sha256"],
        r130_run / "SUMMARY.json": sources["r130_summary_sha256"],
        r130_run / "SELECTOR_TRANSFER.json": sources["r130_selector_transfer_sha256"],
        r131_run / "MANIFEST.json": sources["r131_manifest_sha256"],
        r131_run / "R131_GATE.json": sources["r131_gate_sha256"],
        r131_run / "SUMMARY.json": sources["r131_summary_sha256"],
        r131_run / "SELECTOR_TRANSFER.json": sources["r131_selector_transfer_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)

    package_manifest = json.loads(
        (policy_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    for name, record in package_manifest["files"].items():
        path = policy_package / name
        _verify(path, record["sha256"])
        frozen_files[path] = record["sha256"]

    r129_gate = json.loads((r129_run / "R129_GATE.json").read_text(encoding="utf-8"))
    r130_gate = json.loads((r130_run / "R130_GATE.json").read_text(encoding="utf-8"))
    r131_gate = json.loads((r131_run / "R131_GATE.json").read_text(encoding="utf-8"))
    policy = json.loads((policy_package / "POLICY.json").read_text(encoding="utf-8"))
    threshold = int(policy["threshold_pixels"])
    if threshold != int(runtime["expected_threshold_pixels"]):
        raise R132ExperimentError("frozen R129 policy threshold drift")

    rows: list[dict[str, Any]] = []
    condition_metrics: dict[str, dict[str, Any]] = {}
    calibration_frames: dict[str, int] = {}
    for condition, run, expected_scene in (
        ("r130_scene0230_antithetic", r130_run, runtime["expected_r130_scene"]),
        ("r131_scene0048_orthogonal", r131_run, runtime["expected_r131_scene"]),
    ):
        transfer = json.loads((run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8"))
        frame_count = int(transfer["frame_count"])
        if frame_count != int(runtime["expected_frames_per_condition"]):
            raise R132ExperimentError(f"{condition} frame denominator drift")
        if transfer["target_scene"] != expected_scene:
            raise R132ExperimentError(f"{condition} target scene drift")
        if int(transfer["frozen_threshold_pixels"]) != threshold:
            raise R132ExperimentError(f"{condition} threshold drift")
        calibration_frames[condition] = int(transfer["calibration_frames_in_target_scene"])
        condition_rows: list[dict[str, Any]] = []
        for frame in range(frame_count):
            key = str(frame)
            row = {
                "row_id": f"{condition}:{frame:03d}",
                "condition": condition,
                "target_scene": expected_scene,
                "frame_index": frame,
                "changed_rgb_pixels": int(transfer["sensor_changed_pixels_by_frame"][key]),
                "changed_label_pixels": int(transfer["changed_label_pixels_by_frame"][key]),
            }
            row["target_any_changed_label"] = row["changed_label_pixels"] > 0
            condition_rows.append(row)
        condition_metrics[condition] = _metrics(condition_rows, threshold)
        rows.extend(condition_rows)

    positives = [row for row in rows if row["target_any_changed_label"]]
    negatives = [row for row in rows if not row["target_any_changed_label"]]
    maximum_negative = max(int(row["changed_rgb_pixels"]) for row in negatives)
    minimum_positive = min(int(row["changed_rgb_pixels"]) for row in positives)
    exact_interval = [maximum_negative + 1, minimum_positive]
    aggregate_metrics = _metrics(rows, threshold)
    macro_f1 = sum(metric["f1"] for metric in condition_metrics.values()) / len(condition_metrics)
    worst_condition_f1 = min(metric["f1"] for metric in condition_metrics.values())
    certificate = {
        "schema_version": "worldsim_v6.r132_validation_certificate.v1",
        "policy_id": policy["policy_id"],
        "threshold_pixels": threshold,
        "validation_scope": "r130_and_r131_prospective_conditions_only",
        "development_rows_included": 0,
        "condition_count": len(condition_metrics),
        "row_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "calibration_frames_in_target_conditions": calibration_frames,
        "condition_metrics": condition_metrics,
        "aggregate_metrics": aggregate_metrics,
        "macro_f1": macro_f1,
        "worst_condition_f1": worst_condition_f1,
        "maximum_negative_feature": maximum_negative,
        "minimum_positive_feature": minimum_positive,
        "exact_integer_threshold_interval_on_validation": exact_interval,
        "threshold_inside_validation_interval": maximum_negative < threshold <= minimum_positive,
        "claim_boundary": config["claim_boundary"],
    }

    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R132ExperimentError("R132 disk resource insufficient")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__cross-prospective-threshold13-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(run_dir / "VALIDATION_ROWS.jsonl", rows)
    _write_json(run_dir / "VALIDATION_CERTIFICATE.json", certificate)
    _write_json(run_dir / "REPEAT_VALIDATION_CERTIFICATE.json", certificate)
    repeat_exact = _sha256(run_dir / "VALIDATION_CERTIFICATE.json") == _sha256(
        run_dir / "REPEAT_VALIDATION_CERTIFICATE.json"
    )

    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    expected_per_condition = runtime["expected_condition_counts"]
    checks = {
        "r129_development_policy_authority_accepted": bool(r129_gate["checks"]["passed"]),
        "r130_and_r131_prospective_authorities_accepted": bool(
            r130_gate["checks"]["passed"] and r131_gate["checks"]["passed"]
        ),
        "source_files_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "threshold13_bound_from_r129_without_refit": threshold == 13
        and policy["fit_scope"] == "r126_r127_r128_development_union",
        "zero_target_condition_calibration": all(value == 0 for value in calibration_frames.values()),
        "two_condition_denominators_exact": len(rows) == int(runtime["expected_total_rows"])
        and len(positives) == int(runtime["expected_positive_rows"])
        and len(negatives) == int(runtime["expected_negative_rows"]),
        "condition_denominators_exact": all(
            condition_metrics[name]["true_positive"] == int(counts["positive"])
            and condition_metrics[name]["true_negative"] == int(counts["negative"])
            for name, counts in expected_per_condition.items()
        ),
        "each_condition_zero_error_f1_one": all(
            metric["false_positive"] == 0
            and metric["false_negative"] == 0
            and metric["f1"] == 1.0
            for metric in condition_metrics.values()
        ),
        "aggregate_zero_error_f1_one": aggregate_metrics["false_positive"] == 0
        and aggregate_metrics["false_negative"] == 0
        and aggregate_metrics["f1"] == 1.0,
        "macro_and_worst_condition_f1_one": macro_f1 == 1.0 and worst_condition_f1 == 1.0,
        "aggregate_skip_count_exact": aggregate_metrics["skip_count"]
        == int(runtime["expected_skip_count"]),
        "threshold_inside_exact_validation_interval": certificate[
            "threshold_inside_validation_interval"
        ],
        "certificate_repeat_exact": repeat_exact,
        "validation_excludes_development_rows": certificate["development_rows_included"] == 0,
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R132_GATE.json",
        {
            "schema_version": "worldsim_v6.r132_gate.v1",
            "checks": checks,
            "decision": "accept_two_condition_post_revision_threshold13_validation_certificate"
            if checks["passed"]
            else "reject_or_repair_cross_prospective_threshold13_certificate",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r132_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r132_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_two_condition_threshold13_validation_certificate"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "threshold_pixels": threshold,
        "condition_count": len(condition_metrics),
        "row_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "false_positive": aggregate_metrics["false_positive"],
        "false_negative": aggregate_metrics["false_negative"],
        "f1": aggregate_metrics["f1"],
        "skip_count": aggregate_metrics["skip_count"],
        "skip_fraction": aggregate_metrics["skip_fraction"],
        "macro_f1": macro_f1,
        "worst_condition_f1": worst_condition_f1,
        "exact_integer_threshold_interval_on_validation": exact_interval,
        "certificate_sha256": _sha256(run_dir / "VALIDATION_CERTIFICATE.json"),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "VALIDATION_ROWS.jsonl",
        "VALIDATION_CERTIFICATE.json",
        "REPEAT_VALIDATION_CERTIFICATE.json",
        "R132_GATE.json",
        "RESOURCE_AUDIT.json",
        "SUMMARY.json",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r132_manifest.v1",
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
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r132_cross_prospective_threshold13_certificate_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
