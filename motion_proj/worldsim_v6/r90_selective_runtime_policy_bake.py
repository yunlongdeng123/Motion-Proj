"""WorldSim V6 R90: bake the accepted selective gate as a deterministic runtime policy."""

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


TASK_ID = "WS-V6-R90-SELECTIVE-RUNTIME-POLICY-BAKE-01"


class R90ExperimentError(RuntimeError):
    """The preregistered R90 experiment contract was violated."""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _bake_package(
    package_dir: Path,
    config: dict,
    source_commit: str,
    features: dict[int, int],
    targets: dict[int, bool],
) -> dict:
    package_dir.mkdir(parents=True, exist_ok=False)
    threshold = int(config["policy"]["threshold_pixels"])
    decisions = [
        {
            "frame_index": frame,
            "feature_value_pixels": features[frame],
            "trigger_expensive_perception": features[frame] >= threshold,
        }
        for frame in sorted(features)
    ]
    _write_json(
        package_dir / "POLICY.json",
        {
            "schema_version": "worldsim_v6.r90_policy.v1",
            "policy_id": config["policy"]["policy_id"],
            "feature": config["policy"]["feature"],
            "comparator": "greater_than_or_equal",
            "threshold_pixels": threshold,
            "action_true": "run_frozen_perception_check",
            "action_false": "skip_frozen_perception_check",
            "source_commit": source_commit,
        },
    )
    _write_json(
        package_dir / "FEATURE_SCHEMA.json",
        {
            "schema_version": "worldsim_v6.r90_feature_schema.v1",
            "name": "edited_vs_logged_rgb_changed_pixels",
            "dtype": "nonnegative_integer",
            "rgb_change_epsilon": 1.0e-6,
            "ownership": "compiler_sensor_runtime",
            "required_inputs": ["logged_rgb", "compiled_edited_rgb"],
        },
    )
    _write_json(
        package_dir / "PROVENANCE.json",
        {
            "schema_version": "worldsim_v6.r90_provenance.v1",
            "authorities": config["sources"],
            "calibration_protocol": {
                "r88_interleaved_calibration_frames": 147,
                "r88_holdout_frames": 49,
                "r89_contiguous_phase_calibration_frames": 186,
                "r89_contiguous_phase_holdout_frames": list(range(141, 151)),
            },
        },
    )
    _write_json(
        package_dir / "VALIDITY.json",
        {
            "schema_version": "worldsim_v6.r90_validity.v1",
            "q_same_episode_frozen_model_perception_impact_trigger": "ACCEPT_DEVELOPMENT",
            "q_cross_scene_transfer": "ABSTAIN",
            "q_cross_model_transfer": "ABSTAIN",
            "q_semantic_correctness": "ABSTAIN",
            "q_local_causality": "ABSTAIN",
            "q_physics_planning_safety": "ABSTAIN",
        },
    )
    _write_jsonl(package_dir / "DECISIONS.jsonl", decisions)
    files = [
        "POLICY.json",
        "FEATURE_SCHEMA.json",
        "PROVENANCE.json",
        "VALIDITY.json",
        "DECISIONS.jsonl",
    ]
    _write_json(
        package_dir / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r90_package_manifest.v1",
            "files": {
                name: {
                    "bytes": (package_dir / name).stat().st_size,
                    "sha256": _sha256(package_dir / name),
                }
                for name in files
            },
        },
    )
    return {
        "decisions": decisions,
        "targets": targets,
        "package_manifest_sha256": _sha256(package_dir / "PACKAGE_MANIFEST.json"),
        "files": files + ["PACKAGE_MANIFEST.json"],
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R90ExperimentError("formal R90 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R90ExperimentError("R90 task_id drift")
    sources = config["sources"]
    resources = config["resources"]
    r86_run = _resolve_runs_uri(sources["r86_run"])
    r87_run = _resolve_runs_uri(sources["r87_run"])
    r88_run = _resolve_runs_uri(sources["r88_run"])
    r89_run = _resolve_runs_uri(sources["r89_run"])
    frozen_files = {
        r86_run / "FULL_EPISODE_SENSOR_EFFECT.json": sources[
            "r86_full_episode_sensor_effect_sha256"
        ],
        r87_run / "FULL_EPISODE_PERCEPTION_IMPACT.json": sources[
            "r87_full_episode_perception_impact_sha256"
        ],
        r88_run / "R88_GATE.json": sources["r88_gate_sha256"],
        r88_run / "SELECTIVE_GATE.json": sources["r88_selective_gate_sha256"],
        r89_run / "MANIFEST.json": sources["r89_manifest_sha256"],
        r89_run / "R89_GATE.json": sources["r89_gate_sha256"],
        r89_run / "SUMMARY.json": sources["r89_summary_sha256"],
        r89_run / "PHASE_HOLDOUT_RESULT.json": sources["r89_phase_holdout_result_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R90ExperimentError("R90 disk resource insufficient")
    r88_gate = json.loads((r88_run / "R88_GATE.json").read_text(encoding="utf-8"))
    r89_gate = json.loads((r89_run / "R89_GATE.json").read_text(encoding="utf-8"))
    r88_selector = json.loads(
        (r88_run / "SELECTIVE_GATE.json").read_text(encoding="utf-8")
    )
    r89_result = json.loads(
        (r89_run / "PHASE_HOLDOUT_RESULT.json").read_text(encoding="utf-8")
    )
    sensor = json.loads(
        (r86_run / "FULL_EPISODE_SENSOR_EFFECT.json").read_text(encoding="utf-8")
    )
    perception = json.loads(
        (r87_run / "FULL_EPISODE_PERCEPTION_IMPACT.json").read_text(encoding="utf-8")
    )
    frames = list(range(int(config["policy"]["frame_count"])))
    features = {
        frame: int(sensor["edited_vs_logged_changed_rgb_pixels_by_frame"][str(frame)])
        for frame in frames
    }
    targets = {
        frame: int(perception["changed_label_pixels_by_frame"][str(frame)]) >= 1
        for frame in frames
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__selective-policy-bake-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    package_a = _bake_package(run_dir / "package_a", config, source_commit, features, targets)
    package_b = _bake_package(run_dir / "package_b", config, source_commit, features, targets)
    byte_exact = all(
        _sha256(run_dir / "package_a" / name) == _sha256(run_dir / "package_b" / name)
        for name in package_a["files"]
    )
    decisions = package_a["decisions"]
    predictions = {
        int(row["frame_index"]): bool(row["trigger_expensive_perception"]) for row in decisions
    }
    trigger_count = sum(predictions.values())
    exact_target_match = all(predictions[frame] == targets[frame] for frame in frames)
    transition_frames = [
        frame for frame in frames[1:] if predictions[frame] != predictions[frame - 1]
    ]
    wall_seconds = time.monotonic() - started
    checks = {
        "r88_and_r89_authorities_accepted": bool(
            r88_gate["checks"]["passed"] and r89_gate["checks"]["passed"]
        ),
        "threshold_authorities_agree_at45_pixels": bool(
            int(r88_selector["selected_threshold_pixels"])
            == int(r89_result["selected_threshold_pixels"])
            == int(config["policy"]["threshold_pixels"])
            == 45
        ),
        "five_payload_files_and_manifest_exact": len(package_a["files"]) == 6,
        "full_196_decision_denominator_exact": len(decisions) == 196,
        "trigger_and_skip_counts_exact": bool(
            trigger_count == int(config["policy"]["expected_trigger_count"])
            and len(decisions) - trigger_count == int(config["policy"]["expected_skip_count"])
        ),
        "single_trigger_to_skip_transition_at151": transition_frames == [151],
        "all_decisions_match_frozen_perception_impact_target": exact_target_match,
        "double_bake_byte_exact": byte_exact,
        "package_manifest_content_address_stable": package_a["package_manifest_sha256"]
        == package_b["package_manifest_sha256"],
        "validity_abstains_cross_scene_model_semantics_and_safety": True,
        "frozen_sources_immutable": all(
            _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
        ),
        "cpu_only_within_wall_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R90_GATE.json",
        {
            "schema_version": "worldsim_v6.r90_gate.v1",
            "checks": checks,
            "decision": "accept_selective_runtime_policy_package"
            if checks["passed"]
            else "reject_or_repair_selective_runtime_policy_package",
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r90_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_selective_runtime_policy_package"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "policy_id": config["policy"]["policy_id"],
        "threshold_pixels": int(config["policy"]["threshold_pixels"]),
        "trigger_count": trigger_count,
        "skip_count": len(decisions) - trigger_count,
        "transition_frames": transition_frames,
        "package_manifest_sha256": package_a["package_manifest_sha256"],
        "double_bake_byte_exact": byte_exact,
        "cross_scene_transfer": "ABSTAIN",
        "cross_model_transfer": "ABSTAIN",
        "semantic_correctness": "ABSTAIN",
        "physical_planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R90_GATE.json", "SUMMARY.json"]
    tracked.extend(f"package_a/{name}" for name in package_a["files"])
    tracked.extend(f"package_b/{name}" for name in package_b["files"])
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r90_manifest.v1",
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
        default=Path("configs/worldsim_v6/r90_selective_runtime_policy_bake_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
