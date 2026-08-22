"""WorldSim V6 R105: certify a shared selector threshold interval across scenes and directions."""

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


TASK_ID = "WS-V6-R105-CROSS-SCENE-INTERVENTION-THRESHOLD-CERTIFICATE-01"


class R105ExperimentError(RuntimeError):
    """The preregistered R105 contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R105ExperimentError(f"frozen input drift: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://worldsim_v6/"
    if not uri.startswith(prefix):
        raise R105ExperimentError(f"unsupported run URI: {uri}")
    return Path("/root/autodl-tmp/runs/worldsim_v6") / uri[len(prefix):]


def _metrics(prediction: list[bool], target: list[bool]) -> dict[str, Any]:
    tp = sum(p and t for p, t in zip(prediction, target))
    tn = sum((not p) and (not t) for p, t in zip(prediction, target))
    fp = sum(p and (not t) for p, t in zip(prediction, target))
    fn = sum((not p) and t for p, t in zip(prediction, target))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R105ExperimentError("formal R105 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R105ExperimentError("R105 task_id drift")
    evaluation = config["evaluation"]
    resources = config["resources"]
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R105ExperimentError("R105 disk resource insufficient")

    frozen_files: dict[Path, str] = {}
    policy_run = _resolve_runs_uri(config["policy_source"]["run"])
    policy_path = policy_run / config["policy_source"]["relative_path"]
    frozen_files.update({
        policy_run / "MANIFEST.json": config["policy_source"]["manifest_sha256"],
        policy_run / "R90_GATE.json": config["policy_source"]["gate_sha256"],
        policy_path: config["policy_source"]["policy_sha256"],
    })
    episode_sources = []
    for episode in config["episodes"]:
        vector_run = _resolve_runs_uri(episode["vector_run"])
        authority_run = _resolve_runs_uri(episode["authority_run"])
        files = {
            vector_run / "MANIFEST.json": episode["vector_manifest_sha256"],
            vector_run / episode["vector_gate_file"]: episode["vector_gate_sha256"],
            vector_run / "SELECTOR_TRANSFER.json": episode["selector_transfer_sha256"],
            authority_run / "MANIFEST.json": episode["authority_manifest_sha256"],
            authority_run / episode["authority_gate_file"]: episode["authority_gate_sha256"],
        }
        if episode.get("authority_evidence_file"):
            files[authority_run / episode["authority_evidence_file"]] = episode["authority_evidence_sha256"]
        frozen_files.update(files)
        episode_sources.append((episode, vector_run, authority_run))
    for path, expected in frozen_files.items():
        _verify(path, expected)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    threshold = int(evaluation["frozen_policy_threshold_pixels"])
    episode_rows = []
    all_features: list[int] = []
    all_targets: list[bool] = []
    authority_checks = []
    vector_status_checks = []
    repaired_rejection_retained = True
    for episode, vector_run, authority_run in episode_sources:
        vector_gate = json.loads((vector_run / episode["vector_gate_file"]).read_text(encoding="utf-8"))
        authority_gate = json.loads((authority_run / episode["authority_gate_file"]).read_text(encoding="utf-8"))
        evidence = json.loads((vector_run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8"))
        frame_count = int(evidence["frame_count"])
        features = [int(evidence["sensor_changed_pixels_by_frame"][str(i)]) for i in range(frame_count)]
        labels = [int(evidence["changed_label_pixels_by_frame"][str(i)]) for i in range(frame_count)]
        targets = [value >= int(evaluation["perception_target_minimum_changed_label_pixels"]) for value in labels]
        positives = [value for value, target in zip(features, targets) if target]
        negatives = [value for value, target in zip(features, targets) if not target]
        if not positives or not negatives:
            raise R105ExperimentError(f"R105 empty class support: {episode['episode_id']}")
        maximum_negative = max(negatives)
        minimum_positive = min(positives)
        lower = maximum_negative + 1
        upper = minimum_positive
        metrics = _metrics([value >= threshold for value in features], targets)
        row = {
            "episode_id": episode["episode_id"],
            "scene": episode["scene"],
            "intervention_direction": episode["intervention_direction"],
            "frame_count": frame_count,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "maximum_negative_feature_pixels": maximum_negative,
            "minimum_positive_feature_pixels": minimum_positive,
            "perfect_integer_threshold_lower_inclusive": lower,
            "perfect_integer_threshold_upper_inclusive": upper,
            "perfect_integer_threshold_count": max(0, upper - lower + 1),
            "threshold45_lower_margin_pixels": threshold - lower,
            "threshold45_upper_margin_pixels": upper - threshold,
            "threshold45_metrics_recomputed": metrics,
        }
        episode_rows.append(row)
        all_features.extend(features)
        all_targets.extend(targets)
        authority_checks.append(bool(authority_gate["checks"]["passed"]))
        vector_status_checks.append(bool(vector_gate["checks"]["passed"]) == bool(episode["vector_gate_passed"]))
        if episode.get("authority_evidence_file"):
            repair = json.loads((authority_run / episode["authority_evidence_file"]).read_text(encoding="utf-8"))
            repaired_rejection_retained = repaired_rejection_retained and (
                repair["source_r98_status"] == "rejected" and not repair["source_r98_retroactively_accepted"]
            )

    aggregate_positive = [value for value, target in zip(all_features, all_targets) if target]
    aggregate_negative = [value for value, target in zip(all_features, all_targets) if not target]
    aggregate_lower = max(aggregate_negative) + 1
    aggregate_upper = min(aggregate_positive)
    aggregate_metrics = _metrics([value >= threshold for value in all_features], all_targets)
    certificate = {
        "schema_version": "worldsim_v6.r105_threshold_certificate.v1",
        "source_policy_id": policy["policy_id"],
        "frozen_policy_threshold_pixels": threshold,
        "episode_count": len(episode_rows),
        "distinct_scene_count": len({row["scene"] for row in episode_rows}),
        "frame_count": len(all_features),
        "positive_count": sum(all_targets),
        "negative_count": len(all_targets) - sum(all_targets),
        "episode_rows": episode_rows,
        "aggregate_maximum_negative_feature_pixels": max(aggregate_negative),
        "aggregate_minimum_positive_feature_pixels": min(aggregate_positive),
        "perfect_integer_threshold_lower_inclusive": aggregate_lower,
        "perfect_integer_threshold_upper_inclusive": aggregate_upper,
        "perfect_integer_threshold_count": max(0, aggregate_upper - aggregate_lower + 1),
        "threshold45_lower_margin_pixels": threshold - aggregate_lower,
        "threshold45_upper_margin_pixels": aggregate_upper - threshold,
        "threshold45_two_sided_margin_pixels": min(threshold - aggregate_lower, aggregate_upper - threshold),
        "threshold45_metrics_recomputed": aggregate_metrics,
        "semantic_correctness": "ABSTAIN",
        "unseen_scene_or_intervention_guarantee": "ABSTAIN",
        "confirmation_quality_claim": "ABSTAIN",
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__threshold-certificate-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "THRESHOLD_CERTIFICATE.json", certificate)
    checks = {
        "r90_policy_authority_and_threshold45_exact": bool(
            json.loads((policy_run / "R90_GATE.json").read_text(encoding="utf-8"))["checks"]["passed"]
        ) and int(policy["threshold_pixels"]) == threshold == 45,
        "four_episode_three_scene_two_direction_scope_exact": len(episode_rows) == 4
        and len({row["scene"] for row in episode_rows}) == 3
        and {row["intervention_direction"] for row in episode_rows if row["scene"] == "scene-0255"}
        == {"x_minus", "z_plus"},
        "all_authority_gates_accepted": all(authority_checks),
        "vector_gate_statuses_exact_and_r98_rejection_retained": all(vector_status_checks)
        and repaired_rejection_retained,
        "four_full196_denominators_and_total784_exact": all(row["frame_count"] == 196 for row in episode_rows)
        and len(all_features) == 784,
        "class_support_exact": certificate["positive_count"] == int(evaluation["expected_positive_count"])
        and certificate["negative_count"] == int(evaluation["expected_negative_count"]),
        "per_episode_perfect_intervals_contain_threshold45": all(
            row["perfect_integer_threshold_lower_inclusive"] <= threshold
            <= row["perfect_integer_threshold_upper_inclusive"] for row in episode_rows
        ),
        "aggregate_perfect_interval_nonempty_and_contains_threshold45": aggregate_lower <= threshold <= aggregate_upper,
        "aggregate_interval_width_gate": certificate["perfect_integer_threshold_count"]
        >= int(evaluation["minimum_perfect_integer_threshold_count"]),
        "threshold45_two_sided_margin_gate": certificate["threshold45_two_sided_margin_pixels"]
        >= int(evaluation["minimum_two_sided_margin_pixels"]),
        "threshold45_recomputed_zero_errors": aggregate_metrics["false_positive"] == 0
        and aggregate_metrics["false_negative"] == 0,
        "frozen_sources_immutable_after_certificate": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "semantic_correctness_unseen_generalization_confirmation_quality_abstain": True,
        "cpu_only_no_training_no_inference_no_confirmation_read": True,
        "wall_and_output_within_budget": time.monotonic() - started <= float(resources["maximum_wall_seconds"])
        and (run_dir / "THRESHOLD_CERTIFICATE.json").stat().st_size <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R105_GATE.json",
        {
            "schema_version": "worldsim_v6.r105_gate.v1",
            "checks": checks,
            "decision": "accept_cross_scene_intervention_threshold_certificate"
            if checks["passed"] else "reject_cross_scene_intervention_threshold_certificate",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r105_resource_audit.v1",
            "wall_seconds": time.monotonic() - started,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "model_inference_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r105_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_cross_scene_intervention_threshold_certificate"
        if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "episode_count": certificate["episode_count"],
        "distinct_scene_count": certificate["distinct_scene_count"],
        "frame_count": certificate["frame_count"],
        "positive_count": certificate["positive_count"],
        "negative_count": certificate["negative_count"],
        "perfect_integer_threshold_lower_inclusive": aggregate_lower,
        "perfect_integer_threshold_upper_inclusive": aggregate_upper,
        "perfect_integer_threshold_count": certificate["perfect_integer_threshold_count"],
        "threshold45_two_sided_margin_pixels": certificate["threshold45_two_sided_margin_pixels"],
        "precision": aggregate_metrics["precision"],
        "recall": aggregate_metrics["recall"],
        "f1": aggregate_metrics["f1"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["THRESHOLD_CERTIFICATE.json", "R105_GATE.json", "RESOURCE_AUDIT.json", "SUMMARY.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r105_manifest.v1",
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
        default=Path("configs/worldsim_v6/r105_cross_scene_intervention_threshold_certificate_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
