"""WorldSim V6 R83: execute a three-actor scene-edit package in one shared sensor render."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)
from motion_proj.worldsim_v6.r70_two_actor_scene_package_bake import _verify_scene_package


TASK_ID = "WS-V6-R83-THREE-ACTOR-SCENE-SENSOR-RUNTIME-01"


class R83ExperimentError(RuntimeError):
    """The preregistered R83 experiment contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R83ExperimentError("formal R83 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R83ExperimentError("R83 task_id drift")
    sources = config["sources"]
    contract = config["runtime_contract"]
    thresholds = config["thresholds"]
    resources = config["resources"]

    r82_run = _resolve_runs_uri(sources["r82_run"])
    scene_package = r82_run / "package"
    baseline_run = _resolve_runs_uri(sources["baseline_run"])
    baseline_sensor = baseline_run / sources["baseline_sensor"]
    selection_r71_run = _resolve_runs_uri(sources["selection_r71_run"])
    selection_r73_run = _resolve_runs_uri(sources["selection_r73_run"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r82_run / "MANIFEST.json": sources["r82_manifest_sha256"],
        r82_run / "R82_GATE.json": sources["r82_gate_sha256"],
        r82_run / "SUMMARY.json": sources["r82_summary_sha256"],
        scene_package / "SCENE_PACKAGE_MANIFEST.json": sources["r82_scene_package_manifest_sha256"],
        scene_package / "SCENE_COMPOSITION.json": sources["r82_scene_composition_sha256"],
        scene_package / "RUNTIME_CONTRACT.json": sources["r82_runtime_contract_sha256"],
        scene_package / "VALIDITY.json": sources["r82_validity_sha256"],
        baseline_run / "MANIFEST.json": sources["baseline_manifest_sha256"],
        baseline_run / sources["baseline_gate_filename"]: sources["baseline_gate_sha256"],
        baseline_run / "SUMMARY.json": sources["baseline_summary_sha256"],
        baseline_sensor: sources["baseline_sensor_sha256"],
        selection_r71_run / "R71_GATE.json": sources["selection_r71_gate_sha256"],
        selection_r71_run / "SUMMARY.json": sources["selection_r71_summary_sha256"],
        selection_r71_run / "JOINT_SENSOR_EFFECT.json": sources["selection_r71_joint_sensor_effect_sha256"],
        selection_r73_run / "R73_GATE.json": sources["selection_r73_gate_sha256"],
        selection_r73_run / "SUMMARY.json": sources["selection_r73_summary_sha256"],
        selection_r73_run / "THIRD_ACTOR_SELECTION.json": sources["selection_r73_selection_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    package_manifest = _verify_scene_package(scene_package)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R83ExperimentError("StreetGS upstream commit drift")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R83ExperimentError("R83 disk resource insufficient")

    r82_gate = json.loads((r82_run / "R82_GATE.json").read_text(encoding="utf-8"))
    baseline_gate = json.loads(
        (baseline_run / sources["baseline_gate_filename"]).read_text(encoding="utf-8")
    )
    composition = json.loads(
        (scene_package / "SCENE_COMPOSITION.json").read_text(encoding="utf-8")
    )
    package_runtime = json.loads(
        (scene_package / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8")
    )
    validity = json.loads((scene_package / "VALIDITY.json").read_text(encoding="utf-8"))
    actors = composition["actors"]
    package_contract_exact = bool(
        package_runtime["runtime_mode"] == contract["runtime_mode"]
        and package_runtime["actor_order"] == contract["actor_ids"]
        and [row["actor_id"] for row in actors] == contract["actor_ids"]
        and [int(row["actor_model_index"]) for row in actors]
        == contract["actor_model_indices"]
        and [row["proposal_id"] for row in actors] == contract["proposal_ids"]
        and all(
            row["translation_delta_m"] == contract["translations_m"][row["actor_id"]]
            and int(row["primitive_count"]) == int(contract["expected_primitives"][row["actor_id"]])
            for row in actors
        )
    )

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__three-actor-sensor-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "worker"
        command = [
            sources["drivestudio_python"],
            str(repo_root / "scripts/worldsim_v6/r71_two_actor_scene_sensor_worker.py"),
            "--repo-root",
            str(repo_root),
            "--checkpoint",
            str(checkpoint),
            "--upstream-root",
            str(upstream),
            "--scene-package",
            str(scene_package),
            "--frames",
            str(contract["frame_index"]),
            "--output",
            str(worker_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=float(resources["maximum_worker_seconds"]),
        )
        (run_dir / "worker.log").write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise R83ExperimentError(f"three-actor sensor worker failed: rc={completed.returncode}")
        rows = [
            json.loads(line)
            for line in (worker_dir / "FRAME_METRICS.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != 1:
            raise R83ExperimentError("R83 requires exactly one sensor frame")
        row = rows[0]
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        sensor_path = worker_dir / row["sensor_path"]
        _verify(sensor_path, row["sensor_sha256"])
        with np.load(baseline_sensor, allow_pickle=False) as baseline, np.load(
            sensor_path, allow_pickle=False
        ) as actual:
            baseline_rgb = baseline["native_rgb"].astype(np.float32)
            joint_rgb = actual["native_rgb"].astype(np.float32)
            sensor_array_names = sorted(actual.files)
        changed_pixels = int(
            np.count_nonzero(
                np.mean(np.abs(joint_rgb - baseline_rgb), axis=-1)
                > float(thresholds["rgb_change_epsilon"])
            )
        )
        actor_rows = row["actors"]
        actor_runtime_exact = bool(
            list(actor_rows) == contract["actor_ids"]
            and all(
                int(actor_rows[actor_id]["actor_model_index"])
                == int(contract["actor_model_indices"][index])
                and actor_rows[actor_id]["translation_delta_m"]
                == contract["translations_m"][actor_id]
                and actor_rows[actor_id]["package_actor_frame_valid"]
                == contract["expected_frame_validity"][actor_id]
                and int(actor_rows[actor_id]["primitive_count"])
                == int(contract["expected_primitives"][actor_id])
                for index, actor_id in enumerate(contract["actor_ids"])
            )
        )
        actor_fields_pass = all(
            actor_rows[actor_id]["native_actor_field_max_error"]["means_m"]
            <= float(thresholds["maximum_means_error_m"])
            and actor_rows[actor_id]["native_actor_field_max_error"]["quaternions_wxyz"]
            <= float(thresholds["maximum_quaternion_error"])
            and all(
                actor_rows[actor_id]["native_actor_field_max_error"][key]
                <= float(thresholds["maximum_static_field_error"])
                for key in ("scales_m", "opacities", "view_dependent_rgb")
            )
            for actor_id in contract["actor_ids"]
        )
        sensors_pass = bool(
            row["full_sensor_rgb_mae"] <= float(thresholds["maximum_rgb_mae"])
            and row["full_sensor_rgb_p99_absolute_error"]
            <= float(thresholds["maximum_rgb_p99_absolute_error"])
            and row["full_sensor_depth_mae_m"] <= float(thresholds["maximum_depth_mae_m"])
            and row["full_sensor_opacity_mae"] <= float(thresholds["maximum_opacity_mae"])
        )
        wall_seconds = time.monotonic() - started
        peak_mib = int(audit["peak_torch_reserved_bytes"]) / (1024**2)
        expected_nested_manifests = {
            actor["actor_id"]: actor["actor_package_manifest_sha256"] for actor in actors
        }
        checks = {
            "r82_and_logged_baseline_authorities_accepted": bool(
                r82_gate["checks"]["passed"] and baseline_gate["checks"]["passed"]
            ),
            "scene_package_runtime_contract_exact": package_contract_exact,
            "worker_runtime_ownership_exact": bool(
                audit["runtime_mode"] == contract["runtime_mode"]
                and audit["actor_order"] == contract["actor_ids"]
                and audit["actor_model_indices"] == contract["actor_model_indices"]
                and audit["translation_source"] == contract["required_translation_source"]
                and audit["lifecycle_source"] == contract["required_lifecycle_source"]
            ),
            "frame_actor_translation_lifecycle_and_primitive_bindings_exact": bool(
                int(row["frame_index"]) == int(contract["frame_index"])
                and actor_runtime_exact
            ),
            "all_three_actors_have_visible_effect": all(
                int(actor_rows[actor_id]["actor_effect_pixels"])
                >= int(thresholds["minimum_per_actor_effect_pixels"])
                for actor_id in contract["actor_ids"]
            ),
            "joint_actor_effect_nontrivial": int(row["joint_actor_effect_pixels"])
            >= int(thresholds["minimum_joint_actor_effect_pixels"]),
            "joint_edit_nontrivial_vs_logged": changed_pixels
            >= int(thresholds["minimum_joint_changed_rgb_pixels_vs_logged"]),
            "both_compiled_actor_fields_match_native": actor_fields_pass,
            "compiled_shared_sensor_matches_native": sensors_pass,
            "compiled_repeat_exact": bool(row["compiled_repeat_exact"]),
            "native_multi_actor_state_restored_exact": bool(
                row["native_translation_state_restored_exact"]
            ),
            "scene_and_nested_package_manifests_immutable": bool(
                audit["scene_package_manifest_sha256_before"]
                == audit["scene_package_manifest_sha256_after"]
                == sources["r82_scene_package_manifest_sha256"]
                and audit["nested_actor_package_manifest_sha256_before"]
                == audit["nested_actor_package_manifest_sha256_after"]
                == expected_nested_manifests
            ),
            "checkpoint_immutable": bool(
                audit["checkpoint_sha256_before"]
                == audit["checkpoint_sha256_after"]
                == sources["streetgs_checkpoint_sha256"]
            ),
            "joint_conformance_preserved_and_unsupported_claims_abstain": bool(
                validity["q_joint_aabb_interaction"] == "ACCEPT"
                and validity["q_collision_physics"] == "ABSTAIN"
                and validity["semantic_road"] == "ABSTAIN"
                and validity["physical_dynamics"] == "ABSTAIN"
                and validity["planning_safety"] == "ABSTAIN"
            ),
            "frozen_sources_immutable": bool(
                all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items())
                and package_manifest == _verify_scene_package(scene_package)
            ),
            "gpu_within_budget": peak_mib <= float(resources["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
            "training_not_started": not bool(audit["training_started"]),
            "confirmation_not_read": not bool(audit["confirmation_content_read"]),
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R83_GATE.json",
            {
                "schema_version": "worldsim_v6.r83_gate.v1",
                "checks": checks,
                "decision": "accept_three_actor_scene_sensor_runtime"
                if checks["passed"]
                else "reject_or_repair_three_actor_scene_sensor_runtime",
            },
        )
        _write_json(
            run_dir / "JOINT_SENSOR_EFFECT.json",
            {
                "schema_version": "worldsim_v6.r83_joint_sensor_effect.v1",
                "frame_index": int(row["frame_index"]),
                "logged_sensor_sha256": sources["baseline_sensor_sha256"],
                "joint_sensor_sha256": row["sensor_sha256"],
                "sensor_array_names": sensor_array_names,
                "changed_rgb_pixels_vs_logged": changed_pixels,
                "joint_actor_effect_pixels": int(row["joint_actor_effect_pixels"]),
                "per_actor_effect_pixels": {
                    actor_id: int(actor_rows[actor_id]["actor_effect_pixels"])
                    for actor_id in contract["actor_ids"]
                },
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r83_resource_audit.v1",
                "gpu_used": True,
                "peak_torch_reserved_mib": peak_mib,
                "worker_wall_seconds": float(audit["wall_seconds"]),
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r83_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_three_actor_scene_sensor_runtime"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "frame_index": int(row["frame_index"]),
            "actor_ids": contract["actor_ids"],
            "actor_model_indices": contract["actor_model_indices"],
            "runtime_mode": audit["runtime_mode"],
            "translation_source": audit["translation_source"],
            "lifecycle_source": audit["lifecycle_source"],
            "per_actor_effect_pixels": {
                actor_id: int(actor_rows[actor_id]["actor_effect_pixels"])
                for actor_id in contract["actor_ids"]
            },
            "joint_actor_effect_pixels": int(row["joint_actor_effect_pixels"]),
            "changed_rgb_pixels_vs_logged": changed_pixels,
            "compiled_native_rgb_mae": float(row["full_sensor_rgb_mae"]),
            "compiled_native_depth_mae_m": float(row["full_sensor_depth_mae_m"]),
            "q_joint_aabb_interaction": "ACCEPT",
            "q_collision_physics": "ABSTAIN",
            "semantic_road": "ABSTAIN",
            "physical_dynamics": "ABSTAIN",
            "planning_safety": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R83_GATE.json",
            "SUMMARY.json",
            "JOINT_SENSOR_EFFECT.json",
            "RESOURCE_AUDIT.json",
            "worker.log",
            "worker/FRAME_METRICS.jsonl",
            "worker/WORKER_AUDIT.json",
            f"worker/{row['sensor_path']}",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r83_manifest.v1",
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
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r83_three_actor_scene_sensor_runtime_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
