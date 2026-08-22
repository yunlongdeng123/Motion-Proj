"""WorldSim V6 R135: bake an explicit StreetGS/AD-GS conditioned surrogate policy."""

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


TASK_ID = "WS-V6-R135-FRONTEND-CONDITIONED-SURROGATE-POLICY-01"


class R135ExperimentError(RuntimeError):
    """The preregistered R135 contract was violated."""


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
                    "frame_index": int(row["frame_index"]),
                    "changed_rgb_pixels": int(row["changed_rgb_pixels"]),
                    "changed_label_pixels": int(row["changed_label_pixels"]),
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
        "trigger_count": tp + fp,
        "skip_count": tn + fn,
        "skip_fraction": float((tn + fn) / len(rows)),
        "errors": errors,
    }


def _build_package(
    package: Path,
    policy: dict[str, Any],
    certificate: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> None:
    package.mkdir(parents=True, exist_ok=False)
    _write_json(package / "POLICY.json", policy)
    _write_json(package / "DEVELOPMENT_CERTIFICATE.json", certificate)
    _write_jsonl(package / "ADGS_DEVELOPMENT_DECISIONS.jsonl", decisions)
    names = ["POLICY.json", "DEVELOPMENT_CERTIFICATE.json", "ADGS_DEVELOPMENT_DECISIONS.jsonl"]
    _write_json(
        package / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r135_package_manifest.v1",
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
        raise R135ExperimentError("formal R135 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R135ExperimentError("R135 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    resources = config["resources"]
    r132_run = _resolve_runs_uri(sources["r132_run"])
    r133_run = _resolve_runs_uri(sources["r133_run"])
    r134_run = _resolve_runs_uri(sources["r134_run"])
    frozen_files = {
        r132_run / "MANIFEST.json": sources["r132_manifest_sha256"],
        r132_run / "R132_GATE.json": sources["r132_gate_sha256"],
        r132_run / "VALIDATION_CERTIFICATE.json": sources["r132_certificate_sha256"],
        r133_run / "MANIFEST.json": sources["r133_manifest_sha256"],
        r133_run / "R133_GATE.json": sources["r133_gate_sha256"],
        r133_run / "SELECTIVE_EXECUTION_RESULT.json": sources[
            "r133_selective_execution_result_sha256"
        ],
        r134_run / "MANIFEST.json": sources["r134_manifest_sha256"],
        r134_run / "R134_GATE.json": sources["r134_gate_sha256"],
        r134_run / "SUMMARY.json": sources["r134_summary_sha256"],
        r134_run / "CROSS_FRONTEND_TRANSFER.json": sources[
            "r134_cross_frontend_transfer_sha256"
        ],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    r132_gate = json.loads((r132_run / "R132_GATE.json").read_text(encoding="utf-8"))
    r133_gate = json.loads((r133_run / "R133_GATE.json").read_text(encoding="utf-8"))
    r134_gate = json.loads((r134_run / "R134_GATE.json").read_text(encoding="utf-8"))
    streetgs_certificate = json.loads(
        (r132_run / "VALIDATION_CERTIFICATE.json").read_text(encoding="utf-8")
    )
    transfer = json.loads(
        (r134_run / "CROSS_FRONTEND_TRANSFER.json").read_text(encoding="utf-8")
    )
    rows = [
        {
            "frame_index": frame,
            "changed_rgb_pixels": int(transfer["sensor_changed_pixels_by_frame"][str(frame)]),
            "changed_label_pixels": int(transfer["changed_label_pixels_by_frame"][str(frame)]),
            "target_any_changed_label": int(
                transfer["changed_label_pixels_by_frame"][str(frame)]
            )
            > 0,
        }
        for frame in sorted(map(int, transfer["sensor_changed_pixels_by_frame"]))
    ]
    positives = [int(row["changed_rgb_pixels"]) for row in rows if row["target_any_changed_label"]]
    negatives = [int(row["changed_rgb_pixels"]) for row in rows if not row["target_any_changed_label"]]
    maximum_negative = max(negatives)
    minimum_positive = min(positives)
    interval = [maximum_negative + 1, minimum_positive]
    candidates = list(range(interval[0], interval[1] + 1))
    selected = max(
        candidates,
        key=lambda value: (min(value - maximum_negative, minimum_positive - value), -value),
    )
    old_metrics = _metrics(rows, int(runtime["streetgs_threshold_pixels"]))
    revised_metrics = _metrics(rows, selected)
    expected_error = {
        "frame_index": int(runtime["expected_false_negative_frame"]),
        "changed_rgb_pixels": int(runtime["expected_false_negative_rgb_pixels"]),
        "changed_label_pixels": int(runtime["expected_false_negative_label_pixels"]),
        "error": "FN",
    }
    certificate = {
        "schema_version": "worldsim_v6.r135_development_certificate.v1",
        "fit_scope": "r134_adgs_train_plus_development_only",
        "heldout_frames_read": 0,
        "row_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "streetgs_threshold_on_adgs_development_metrics": old_metrics,
        "maximum_negative_feature": maximum_negative,
        "minimum_positive_feature": minimum_positive,
        "exact_integer_threshold_interval": interval,
        "selection_rule": "maximize_min_threshold_distance_then_lower_value",
        "selected_adgs_threshold_pixels": selected,
        "selected_positive_side_margin_pixels": minimum_positive - selected,
        "selected_negative_side_margin_pixels": selected - maximum_negative,
        "adgs_development_metrics": revised_metrics,
        "heldout_confirmation_required": True,
    }
    policy = {
        "schema_version": "worldsim_v6.r135_frontend_conditioned_policy.v1",
        "policy_id": "worldsim-v6-r135-frontend-conditioned-binary-impact",
        "routing_key": "declared_compiler_frontend",
        "feature": "edited_vs_logged_quantized_rgb_changed_pixels",
        "comparator": "greater_than_or_equal",
        "routes": {
            "streetgs": {
                "threshold_pixels": int(runtime["streetgs_threshold_pixels"]),
                "authority": "r132_validation_plus_r133_real_execution",
            },
            "ad_gs": {
                "threshold_pixels": selected,
                "authority": "r134_rejected_development_fit_only",
                "heldout_status": "PENDING_EXACT_ONCE",
            },
        },
        "unknown_frontend_action": "ABSTAIN_RUN_EXPENSIVE_PERCEPTION",
        "target": "any_changed_frozen_deeplab_label_pixel",
    }
    decisions = [
        {
            "frame_index": int(row["frame_index"]),
            "changed_rgb_pixels": int(row["changed_rgb_pixels"]),
            "target_any_changed_label": bool(row["target_any_changed_label"]),
            "trigger_expensive_perception": int(row["changed_rgb_pixels"]) >= selected,
        }
        for row in rows
    ]
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R135ExperimentError("R135 disk resource insufficient")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__frontend-conditioned-policy-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    package_a = run_dir / "package_a"
    package_b = run_dir / "package_b"
    _build_package(package_a, policy, certificate, decisions)
    _build_package(package_b, policy, certificate, decisions)
    names = [
        "POLICY.json",
        "DEVELOPMENT_CERTIFICATE.json",
        "ADGS_DEVELOPMENT_DECISIONS.jsonl",
        "PACKAGE_MANIFEST.json",
    ]
    repeat_exact = all(_sha256(package_a / name) == _sha256(package_b / name) for name in names)
    false_checks = sorted(name for name, value in r134_gate["checks"].items() if not value)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "streetgs_authorities_accepted": bool(
            r132_gate["checks"]["passed"] and r133_gate["checks"]["passed"]
        ),
        "r134_remains_rejected_only_by_threshold13_transfer_checks": not r134_gate["checks"][
            "passed"
        ]
        and false_checks == sorted(runtime["expected_r134_false_checks"]),
        "source_files_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "adgs_development_denominators_exact": len(rows) == int(runtime["expected_rows"])
        and len(positives) == int(runtime["expected_positive_rows"])
        and len(negatives) == int(runtime["expected_negative_rows"]),
        "threshold13_unique_false_negative_identity_exact": old_metrics["false_positive"] == 0
        and old_metrics["false_negative"] == 1
        and old_metrics["errors"] == [expected_error],
        "adgs_exact_interval_is_singleton_one": interval == runtime["expected_interval"],
        "frontend_router_selects_streetgs13_adgs1": policy["routes"]["streetgs"][
            "threshold_pixels"
        ]
        == 13
        and policy["routes"]["ad_gs"]["threshold_pixels"] == 1,
        "adgs_threshold1_exact_on_development": revised_metrics["false_positive"] == 0
        and revised_metrics["false_negative"] == 0
        and revised_metrics["f1"] == 1.0
        and revised_metrics["skip_count"] == int(runtime["expected_negative_rows"]),
        "unknown_frontend_fail_closed": policy["unknown_frontend_action"]
        == "ABSTAIN_RUN_EXPENSIVE_PERCEPTION",
        "two_packages_repeat_exact": repeat_exact,
        "heldout_unread_and_confirmation_required": certificate["heldout_frames_read"] == 0
        and certificate["heldout_confirmation_required"],
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R135_GATE.json",
        {
            "schema_version": "worldsim_v6.r135_gate.v1",
            "checks": checks,
            "decision": "accept_frontend_conditioned_policy_development_freeze"
            if checks["passed"]
            else "reject_frontend_conditioned_policy_revision",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r135_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r135_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_frontend_conditioned_policy_development_freeze"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "streetgs_threshold_pixels": int(runtime["streetgs_threshold_pixels"]),
        "adgs_threshold_pixels": selected,
        "adgs_exact_integer_threshold_interval": interval,
        "adgs_development_false_positive": revised_metrics["false_positive"],
        "adgs_development_false_negative": revised_metrics["false_negative"],
        "adgs_development_f1": revised_metrics["f1"],
        "adgs_development_skip_fraction": revised_metrics["skip_fraction"],
        "package_manifest_sha256": _sha256(package_a / "PACKAGE_MANIFEST.json"),
        "adgs_heldout_status": "PENDING_EXACT_ONCE",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R135_GATE.json", "RESOURCE_AUDIT.json", "SUMMARY.json"]
    for package_name in ("package_a", "package_b"):
        tracked.extend(f"{package_name}/{name}" for name in names)
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r135_manifest.v1",
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
        default=Path("configs/worldsim_v6/r135_frontend_conditioned_surrogate_policy_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
