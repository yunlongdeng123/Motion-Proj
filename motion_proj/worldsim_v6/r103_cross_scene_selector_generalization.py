"""WorldSim V6 R103: aggregate frozen-selector evidence across independent scenes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "WS-V6-R103-CROSS-SCENE-SELECTOR-GENERALIZATION-01"


class R103ExperimentError(RuntimeError):
    """The preregistered R103 contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R103ExperimentError(f"frozen input drift: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://worldsim_v6/"
    if not uri.startswith(prefix):
        raise R103ExperimentError(f"unsupported run URI: {uri}")
    return Path("/root/autodl-tmp/runs/worldsim_v6") / uri[len(prefix):]


def _aggregate(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in metric_rows)
    tn = sum(int(row["true_negative"]) for row in metric_rows)
    fp = sum(int(row["false_positive"]) for row in metric_rows)
    fn = sum(int(row["false_negative"]) for row in metric_rows)
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trigger_count": tp + fp,
        "skip_count": tn + fn,
        "skip_fraction": (tn + fn) / total if total else 0.0,
        "frame_count": total,
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R103ExperimentError("formal R103 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R103ExperimentError("R103 task_id drift")
    resources = config["resources"]
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R103ExperimentError("R103 disk resource insufficient")

    policy_run = _resolve_runs_uri(config["policy_source"]["run"])
    policy_path = policy_run / config["policy_source"]["relative_path"]
    frozen_files: dict[Path, str] = {
        policy_run / "MANIFEST.json": config["policy_source"]["manifest_sha256"],
        policy_run / "R90_GATE.json": config["policy_source"]["gate_sha256"],
        policy_path: config["policy_source"]["policy_sha256"],
    }
    target_sources = []
    for target in config["targets"]:
        run_dir = _resolve_runs_uri(target["run"])
        files = {
            run_dir / "MANIFEST.json": target["manifest_sha256"],
            run_dir / target["gate_file"]: target["gate_sha256"],
            run_dir / "SUMMARY.json": target["summary_sha256"],
            run_dir / target["evidence_file"]: target["evidence_sha256"],
        }
        frozen_files.update(files)
        target_sources.append((target, run_dir))
    for path, expected in frozen_files.items():
        _verify(path, expected)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    expected_threshold = int(config["evaluation"]["frozen_policy_threshold_pixels"])
    scene_rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    for target, run_dir in target_sources:
        gate = json.loads((run_dir / target["gate_file"]).read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
        evidence = json.loads((run_dir / target["evidence_file"]).read_text(encoding="utf-8"))
        frozen = evidence["frozen_policy_metrics"]
        fixed = evidence["fixed256_metrics"]
        lifecycle = evidence[target["lifecycle_metrics_key"]]
        row = {
            "scene": target["scene"],
            "authority_task_id": summary["task_id"],
            "authority_passed": bool(gate["checks"]["passed"]),
            "frame_count": int(evidence["frame_count"]),
            "calibration_frames": int(evidence.get("calibration_frames_in_target_scene", 0)),
            "positive_prevalence": (int(frozen["true_positive"]) + int(frozen["false_negative"]))
            / int(evidence["frame_count"]),
            "frozen_policy_metrics": frozen,
            "fixed256_metrics": fixed,
            "lifecycle_metrics": lifecycle,
        }
        scene_rows.append(row)
        frozen_rows.append(frozen)
        fixed_rows.append(fixed)
        lifecycle_rows.append(lifecycle)

    frozen_micro = _aggregate(frozen_rows)
    fixed_micro = _aggregate(fixed_rows)
    lifecycle_micro = _aggregate(lifecycle_rows)
    macro_f1 = sum(float(row["frozen_policy_metrics"]["f1"]) for row in scene_rows) / len(scene_rows)
    worst_scene_f1 = min(float(row["frozen_policy_metrics"]["f1"]) for row in scene_rows)
    prevalences = [float(row["positive_prevalence"]) for row in scene_rows]
    aggregate = {
        "schema_version": "worldsim_v6.r103_cross_scene_generalization.v1",
        "source_policy_id": policy["policy_id"],
        "frozen_policy_threshold_pixels": expected_threshold,
        "target_scene_count": len(scene_rows),
        "target_frame_count": sum(int(row["frame_count"]) for row in scene_rows),
        "target_calibration_frame_count": sum(int(row["calibration_frames"]) for row in scene_rows),
        "scene_rows": scene_rows,
        "frozen_policy_micro": frozen_micro,
        "fixed256_micro": fixed_micro,
        "lifecycle_micro": lifecycle_micro,
        "frozen_policy_macro_f1": macro_f1,
        "frozen_policy_worst_scene_f1": worst_scene_f1,
        "positive_prevalence_minimum": min(prevalences),
        "positive_prevalence_maximum": max(prevalences),
        "positive_prevalence_range": max(prevalences) - min(prevalences),
        "semantic_correctness": "ABSTAIN",
        "local_causality": "ABSTAIN",
        "confirmation_quality_claim": "ABSTAIN",
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__cross-scene-selector-generalization-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "CROSS_SCENE_GENERALIZATION.json", aggregate)

    minimum_f1 = float(config["evaluation"]["minimum_f1"])
    minimum_skip = float(config["evaluation"]["minimum_skip_fraction"])
    checks = {
        "r90_policy_authority_accepted_and_threshold45_exact": bool(
            json.loads((policy_run / "R90_GATE.json").read_text(encoding="utf-8"))["checks"]["passed"]
        ) and int(policy["threshold_pixels"]) == expected_threshold == 45,
        "three_distinct_independent_target_scenes_exact": [row["scene"] for row in scene_rows]
        == list(config["evaluation"]["expected_target_scenes"]),
        "all_target_authorities_accepted": all(row["authority_passed"] for row in scene_rows),
        "all_three_full196_denominators_and_total588_exact": all(row["frame_count"] == 196 for row in scene_rows)
        and aggregate["target_frame_count"] == 588,
        "zero_target_scene_calibration_across_all_targets": aggregate["target_calibration_frame_count"] == 0,
        "per_scene_precision_recall_f1_and_skip_gates": all(
            float(row["frozen_policy_metrics"][metric]) >= minimum_f1
            for row in scene_rows for metric in ("precision", "recall", "f1")
        ) and all(float(row["frozen_policy_metrics"]["skip_fraction"]) >= minimum_skip for row in scene_rows),
        "micro_precision_recall_f1_and_skip_gates": all(
            float(frozen_micro[metric]) >= minimum_f1 for metric in ("precision", "recall", "f1")
        ) and float(frozen_micro["skip_fraction"]) >= minimum_skip,
        "macro_and_worst_scene_f1_gates": macro_f1 >= minimum_f1 and worst_scene_f1 >= minimum_f1,
        "frozen_policy_no_more_errors_than_fixed256_or_lifecycle": (
            frozen_micro["false_positive"] + frozen_micro["false_negative"]
            <= fixed_micro["false_positive"] + fixed_micro["false_negative"]
            and frozen_micro["false_positive"] + frozen_micro["false_negative"]
            <= lifecycle_micro["false_positive"] + lifecycle_micro["false_negative"]
        ),
        "target_positive_prevalence_spans_distinct_regimes": aggregate["positive_prevalence_range"]
        >= float(config["evaluation"]["minimum_positive_prevalence_range"]),
        "frozen_sources_immutable_after_aggregation": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "semantic_correctness_local_causality_confirmation_quality_abstain": True,
        "cpu_only_no_training_no_confirmation_read": True,
        "wall_and_output_within_budget": time.monotonic() - started <= float(resources["maximum_wall_seconds"])
        and (run_dir / "CROSS_SCENE_GENERALIZATION.json").stat().st_size
        <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R103_GATE.json",
        {
            "schema_version": "worldsim_v6.r103_gate.v1",
            "checks": checks,
            "decision": "accept_cross_scene_zero_calibration_selector_generalization_evidence"
            if checks["passed"] else "reject_cross_scene_selector_generalization_evidence",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r103_resource_audit.v1",
            "wall_seconds": time.monotonic() - started,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r103_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_cross_scene_selector_generalization"
        if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "source_scene": "scene-0242",
        "target_scenes": [row["scene"] for row in scene_rows],
        "target_frame_count": aggregate["target_frame_count"],
        "frozen_policy_threshold_pixels": expected_threshold,
        "micro_precision": frozen_micro["precision"],
        "micro_recall": frozen_micro["recall"],
        "micro_f1": frozen_micro["f1"],
        "macro_f1": macro_f1,
        "worst_scene_f1": worst_scene_f1,
        "skip_fraction": frozen_micro["skip_fraction"],
        "fixed256_micro_f1": fixed_micro["f1"],
        "lifecycle_micro_f1": lifecycle_micro["f1"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["CROSS_SCENE_GENERALIZATION.json", "R103_GATE.json", "RESOURCE_AUDIT.json", "SUMMARY.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r103_manifest.v1",
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
        default=Path("configs/worldsim_v6/r103_cross_scene_selector_generalization_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
