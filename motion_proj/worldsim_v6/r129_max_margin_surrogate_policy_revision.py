"""WorldSim V6 R129: revise the binary surrogate by a frozen max-margin rule."""

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


TASK_ID = "WS-V6-R129-MAX-MARGIN-SURROGATE-POLICY-REVISION-01"


class R129ExperimentError(RuntimeError):
    """The preregistered R129 contract was violated."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metrics(rows: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    errors = []
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
                    "source": row["source"],
                    "frame_index": row.get("frame_index"),
                    "changed_rgb_pixels": int(row["changed_rgb_pixels"]),
                    "changed_label_pixels": row.get("changed_label_pixels"),
                    "error": "FP" if predicted else "FN",
                }
            )
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
        "errors": errors,
    }


def _build_package(package: Path, policy: dict[str, Any], certificate: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    package.mkdir(parents=True, exist_ok=False)
    _write_json(package / "POLICY.json", policy)
    _write_json(package / "DEVELOPMENT_CERTIFICATE.json", certificate)
    decisions = [
        {
            "row_id": row["row_id"],
            "predict_any_changed_frozen_deeplab_label": int(row["changed_rgb_pixels"])
            >= int(policy["threshold_pixels"]),
        }
        for row in rows
    ]
    _write_jsonl(package / "DEVELOPMENT_DECISIONS.jsonl", decisions)
    names = ["POLICY.json", "DEVELOPMENT_CERTIFICATE.json", "DEVELOPMENT_DECISIONS.jsonl"]
    _write_json(
        package / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r129_package_manifest.v1",
            "files": {
                name: {"bytes": (package / name).stat().st_size, "sha256": _sha256(package / name)}
                for name in names
            },
        },
    )


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R129ExperimentError("formal R129 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R129ExperimentError("R129 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    resources = config["resources"]

    r126_run = _resolve_runs_uri(sources["r126_run"])
    r127_run = _resolve_runs_uri(sources["r127_run"])
    r128_run = _resolve_runs_uri(sources["r128_run"])
    package126 = r126_run / "package_a"
    frozen_files = {
        r126_run / "MANIFEST.json": sources["r126_manifest_sha256"],
        r126_run / "R126_GATE.json": sources["r126_gate_sha256"],
        r126_run / "SUMMARY.json": sources["r126_summary_sha256"],
        package126 / "PACKAGE_MANIFEST.json": sources["r126_package_manifest_sha256"],
        r127_run / "MANIFEST.json": sources["r127_manifest_sha256"],
        r127_run / "R127_GATE.json": sources["r127_gate_sha256"],
        r127_run / "SUMMARY.json": sources["r127_summary_sha256"],
        r127_run / "SELECTOR_TRANSFER.json": sources["r127_selector_transfer_sha256"],
        r128_run / "MANIFEST.json": sources["r128_manifest_sha256"],
        r128_run / "R128_GATE.json": sources["r128_gate_sha256"],
        r128_run / "SUMMARY.json": sources["r128_summary_sha256"],
        r128_run / "SELECTOR_TRANSFER.json": sources["r128_selector_transfer_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    package126_manifest = json.loads(
        (package126 / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    for name, record in package126_manifest["files"].items():
        _verify(package126 / name, record["sha256"])
        frozen_files[package126 / name] = record["sha256"]

    r126_gate = json.loads((r126_run / "R126_GATE.json").read_text(encoding="utf-8"))
    r127_gate = json.loads((r127_run / "R127_GATE.json").read_text(encoding="utf-8"))
    r128_gate = json.loads((r128_run / "R128_GATE.json").read_text(encoding="utf-8"))
    r128_summary = json.loads((r128_run / "SUMMARY.json").read_text(encoding="utf-8"))

    features126 = _read_jsonl(package126 / "FEATURE_ROWS.jsonl")
    targets126 = _read_jsonl(package126 / "TARGET_ROWS.jsonl")
    if len(features126) != len(targets126):
        raise R129ExperimentError("R126 package feature/target denominator drift")
    rows: list[dict[str, Any]] = []
    for feature, target in zip(features126, targets126):
        if feature["row_id"] != target["row_id"]:
            raise R129ExperimentError("R126 package row order drift")
        rows.append(
            {
                "row_id": feature["row_id"],
                "source": "r126_bound_corpus",
                "changed_rgb_pixels": int(feature["changed_rgb_pixels"]),
                "target_any_changed_label": bool(target["any_changed_frozen_deeplab_label"]),
            }
        )

    for source_name, run, expected_frames in (
        ("r127_prospective_diagonal", r127_run, int(runtime["expected_r127_frames"])),
        ("r128_rejected_orthogonal_development", r128_run, int(runtime["expected_r128_frames"])),
    ):
        transfer = json.loads((run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8"))
        if int(transfer["frame_count"]) != expected_frames:
            raise R129ExperimentError(f"{source_name} frame denominator drift")
        for frame in range(expected_frames):
            key = str(frame)
            rows.append(
                {
                    "row_id": f"{source_name}:{frame:03d}",
                    "source": source_name,
                    "frame_index": frame,
                    "changed_rgb_pixels": int(transfer["sensor_changed_pixels_by_frame"][key]),
                    "changed_label_pixels": int(transfer["changed_label_pixels_by_frame"][key]),
                    "target_any_changed_label": int(transfer["changed_label_pixels_by_frame"][key]) > 0,
                }
            )

    positives = [int(row["changed_rgb_pixels"]) for row in rows if row["target_any_changed_label"]]
    negatives = [int(row["changed_rgb_pixels"]) for row in rows if not row["target_any_changed_label"]]
    max_negative = max(negatives)
    min_positive = min(positives)
    interval = [max_negative + 1, min_positive]
    candidates = list(range(interval[0], interval[1] + 1))
    selected_threshold = max(
        candidates,
        key=lambda value: (min(value - max_negative, min_positive - value), -value),
    )
    selected_margin = min(selected_threshold - max_negative, min_positive - selected_threshold)
    threshold45_metrics = _metrics(rows, int(runtime["old_threshold_pixels"]))
    revised_metrics = _metrics(rows, selected_threshold)
    expected_error = {
        "row_id": f"r128_rejected_orthogonal_development:{int(runtime['expected_false_negative_frame']):03d}",
        "source": "r128_rejected_orthogonal_development",
        "frame_index": int(runtime["expected_false_negative_frame"]),
        "changed_rgb_pixels": int(runtime["expected_false_negative_rgb_pixels"]),
        "changed_label_pixels": int(runtime["expected_false_negative_label_pixels"]),
        "error": "FN",
    }
    certificate = {
        "schema_version": "worldsim_v6.r129_development_certificate.v1",
        "row_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "old_threshold_pixels": int(runtime["old_threshold_pixels"]),
        "old_threshold_metrics": threshold45_metrics,
        "maximum_negative_feature": max_negative,
        "minimum_positive_feature": min_positive,
        "exact_integer_threshold_interval": interval,
        "selection_rule": "maximize_min_threshold_distance_to_max_negative_and_min_positive_then_lower_value",
        "selected_threshold_pixels": selected_threshold,
        "selected_minimum_margin_pixels": selected_margin,
        "revised_threshold_metrics": revised_metrics,
        "r128_is_development_not_validation": True,
        "new_prospective_holdout_required": True,
    }
    policy = {
        "schema_version": "worldsim_v6.r129_binary_surrogate_policy.v1",
        "policy_id": "worldsim-v6-r129-max-margin-binary-impact-threshold13",
        "feature": "edited_vs_logged_rgb_changed_pixels",
        "comparator": "greater_than_or_equal",
        "threshold_pixels": selected_threshold,
        "target": "any_changed_frozen_deeplab_label_pixel",
        "fit_scope": "r126_r127_r128_development_union",
        "prospective_validation": "PENDING_NEW_CONDITION",
    }

    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R129ExperimentError("R129 disk resource insufficient")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__max-margin-policy-revision-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    package_a = run_dir / "package_a"
    package_b = run_dir / "package_b"
    _build_package(package_a, policy, certificate, rows)
    _build_package(package_b, policy, certificate, rows)
    package_names = [
        "POLICY.json",
        "DEVELOPMENT_CERTIFICATE.json",
        "DEVELOPMENT_DECISIONS.jsonl",
        "PACKAGE_MANIFEST.json",
    ]
    repeat_exact = all(_sha256(package_a / name) == _sha256(package_b / name) for name in package_names)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    r128_false_checks = sorted(name for name, value in r128_gate["checks"].items() if not value)
    expected_r128_false_checks = sorted(runtime["expected_r128_false_checks"])
    checks = {
        "r126_and_r127_authorities_accepted": bool(
            r126_gate["checks"]["passed"] and r127_gate["checks"]["passed"]
        ),
        "r128_formally_rejected_only_by_preregistered_selector_checks": not r128_gate["checks"]["passed"]
        and r128_summary["status"] == "rejected"
        and r128_false_checks == expected_r128_false_checks,
        "source_files_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "development_union_denominators_exact": len(rows) == int(runtime["expected_total_rows"])
        and len(positives) == int(runtime["expected_positive_rows"])
        and len(negatives) == int(runtime["expected_negative_rows"]),
        "threshold45_unique_error_identity_exact": threshold45_metrics["false_positive"] == 0
        and threshold45_metrics["false_negative"] == 1
        and threshold45_metrics["errors"] == [expected_error],
        "exact_interval_is_one_to26": interval == runtime["expected_exact_interval"],
        "max_margin_rule_selects_threshold13": selected_threshold
        == int(runtime["expected_selected_threshold_pixels"])
        and selected_margin == int(runtime["expected_selected_margin_pixels"]),
        "threshold13_exact_on_development_union": revised_metrics["false_positive"] == 0
        and revised_metrics["false_negative"] == 0
        and revised_metrics["f1"] == 1.0,
        "two_policy_package_bakes_repeat_exact": repeat_exact,
        "r128_explicitly_development_and_new_holdout_pending": certificate[
            "r128_is_development_not_validation"
        ]
        and certificate["new_prospective_holdout_required"],
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R129_GATE.json",
        {
            "schema_version": "worldsim_v6.r129_gate.v1",
            "checks": checks,
            "decision": "accept_development_threshold13_policy_revision"
            if checks["passed"]
            else "reject_or_repair_threshold_policy_revision",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r129_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r129_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_max_margin_threshold13_revision"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "row_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "old_threshold_false_negative": threshold45_metrics["false_negative"],
        "exact_integer_threshold_interval": interval,
        "selected_threshold_pixels": selected_threshold,
        "selected_minimum_margin_pixels": selected_margin,
        "revised_false_positive": revised_metrics["false_positive"],
        "revised_false_negative": revised_metrics["false_negative"],
        "package_manifest_sha256": _sha256(package_a / "PACKAGE_MANIFEST.json"),
        "prospective_validation": "PENDING_NEW_CONDITION",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R129_GATE.json", "RESOURCE_AUDIT.json", "SUMMARY.json"]
    for package_name in ("package_a", "package_b"):
        tracked.extend(f"{package_name}/{name}" for name in package_names)
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r129_manifest.v1",
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
        default=Path("configs/worldsim_v6/r129_max_margin_surrogate_policy_revision_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
