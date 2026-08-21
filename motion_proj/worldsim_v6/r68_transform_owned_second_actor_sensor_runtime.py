"""WorldSim V6 R68: execute actor2 directly from its transform-owned package."""

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
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R68-TRANSFORM-OWNED-SECOND-ACTOR-SENSOR-RUNTIME-01"


class R68ExperimentError(RuntimeError):
    """The preregistered R68 experiment contract was violated."""


def _all_arrays_exact(left_path: Path, right_path: Path) -> tuple[bool, list[str]]:
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        names = sorted(set(left.files) | set(right.files))
        mismatches = [
            name
            for name in names
            if name not in left.files
            or name not in right.files
            or not np.array_equal(left[name], right[name])
        ]
    return not mismatches, mismatches


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R68ExperimentError("formal R68 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R68ExperimentError("R68 task_id drift")

    sources = config["sources"]
    contract = config["runtime_contract"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    r67_run = _resolve_runs_uri(sources["r67_run"])
    package = r67_run / "package"
    r61_run = _resolve_runs_uri(sources["r61_run"])
    r61_sensor = r61_run / sources["r61_sensor"]
    r57_run = _resolve_runs_uri(sources["r57_run"])
    baseline_sensor = r57_run / sources["r57_sensor"]
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r67_run / "MANIFEST.json": sources["r67_manifest_sha256"],
        r67_run / "R67_GATE.json": sources["r67_gate_sha256"],
        r67_run / "SUMMARY.json": sources["r67_summary_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r67_package_manifest_sha256"],
        package / "TRAJECTORY_GEOMETRY.json": sources["r67_trajectory_geometry_sha256"],
        package / "RUNTIME_CONTRACT.json": sources["r67_runtime_contract_sha256"],
        package / "VALIDITY.json": sources["r67_validity_sha256"],
        r61_run / "MANIFEST.json": sources["r61_manifest_sha256"],
        r61_run / "R61_GATE.json": sources["r61_gate_sha256"],
        r61_run / "SUMMARY.json": sources["r61_summary_sha256"],
        r61_sensor: sources["r61_sensor_sha256"],
        r57_run / "MANIFEST.json": sources["r57_manifest_sha256"],
        r57_run / "R57_GATE.json": sources["r57_gate_sha256"],
        r57_run / "SUMMARY.json": sources["r57_summary_sha256"],
        baseline_sensor: sources["r57_sensor_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R68ExperimentError("StreetGS upstream commit drift")

    r67_gate = json.loads((r67_run / "R67_GATE.json").read_text(encoding="utf-8"))
    r61_gate = json.loads((r61_run / "R61_GATE.json").read_text(encoding="utf-8"))
    r57_gate = json.loads((r57_run / "R57_GATE.json").read_text(encoding="utf-8"))
    geometry = json.loads((package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
    runtime = json.loads((package / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8"))
    validity = json.loads((package / "VALIDITY.json").read_text(encoding="utf-8"))
    package_manifest = _verify_package(package)
    package_binding_exact = bool(
        geometry["asset_id"] == contract["asset_id"]
        and int(geometry["actor_model_index"]) == int(contract["actor_model_index"])
        and geometry["proposal_id"] == contract["proposal_id"]
        and geometry["translation_delta_m"] == contract["translation_delta_m"]
        and runtime["proposal_id"] == contract["proposal_id"]
        and validity["proposal_id"] == contract["proposal_id"]
        and validity["joint_admissibility"] == "ACCEPT_CONFORMANCE"
        and all(value == "ACCEPT" for value in validity["factor_decisions"].values())
    )
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R68ExperimentError("R68 disk resource insufficient")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor2-transform-runtime-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "worker"
        command = [
            sources["drivestudio_python"],
            str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
            "--repo-root",
            str(repo_root),
            "--checkpoint",
            str(checkpoint),
            "--upstream-root",
            str(upstream),
            "--package",
            str(package),
            "--frames",
            str(contract["frame_index"]),
            "--actor-model-index",
            str(contract["actor_model_index"]),
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
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise R68ExperimentError(f"transform-owned sensor worker failed: rc={completed.returncode}")

        rows = [
            json.loads(line)
            for line in (worker_dir / "FRAME_METRICS.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != 1:
            raise R68ExperimentError("R68 requires exactly one sensor frame")
        row = rows[0]
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        sensor_path = worker_dir / row["sensor_path"]
        _verify(sensor_path, row["sensor_sha256"])
        arrays_exact, array_mismatches = _all_arrays_exact(sensor_path, r61_sensor)
        with np.load(baseline_sensor, allow_pickle=False) as baseline, np.load(sensor_path, allow_pickle=False) as actual:
            baseline_rgb = baseline["native_rgb"].astype(np.float32)
            runtime_rgb = actual["native_rgb"].astype(np.float32)
        changed_pixels = int(
            np.count_nonzero(
                np.mean(np.abs(runtime_rgb - baseline_rgb), axis=-1)
                > float(thresholds["rgb_change_epsilon"])
            )
        )
        actor_fields_pass = bool(
            row["native_actor_field_max_error"]["means_m"] <= float(thresholds["maximum_means_error_m"])
            and row["native_actor_field_max_error"]["quaternions_wxyz"] <= float(thresholds["maximum_quaternion_error"])
            and all(
                row["native_actor_field_max_error"][key] <= float(thresholds["maximum_static_field_error"])
                for key in ("scales_m", "opacities", "view_dependent_rgb")
            )
        )
        sensors_pass = bool(
            row["full_sensor_rgb_mae"] <= float(thresholds["maximum_rgb_mae"])
            and row["full_sensor_rgb_p99_absolute_error"] <= float(thresholds["maximum_rgb_p99_absolute_error"])
            and row["full_sensor_depth_mae_m"] <= float(thresholds["maximum_depth_mae_m"])
            and row["full_sensor_opacity_mae"] <= float(thresholds["maximum_opacity_mae"])
        )
        wall_seconds = time.monotonic() - started
        peak_mib = int(audit["peak_torch_reserved_bytes"]) / (1024**2)
        checks = {
            "r67_r61_r57_authorities_accepted": bool(
                r67_gate["checks"]["passed"]
                and r61_gate["checks"]["passed"]
                and r57_gate["checks"]["passed"]
            ),
            "package_proposal_factor_binding_exact": package_binding_exact,
            "frame_actor_translation_and_lifecycle_exact": bool(
                int(row["frame_index"]) == int(contract["frame_index"])
                and int(audit["actor_model_index"]) == int(contract["actor_model_index"])
                and row["translation_delta_m"] == contract["translation_delta_m"]
                and row["package_actor_frame_valid"] == contract["expected_frame_validity"]
            ),
            "direct_transform_and_lifecycle_runtime_modes_exact": bool(
                audit["runtime_mode"] == contract["required_worker_runtime_mode"]
                and audit["translation_source"] == contract["required_translation_source"]
                and audit["lifecycle_source"] == contract["required_lifecycle_source"]
            ),
            "r68_sensor_arrays_match_r61_cli_reference_exact": arrays_exact,
            "edited_sensor_nontrivial_vs_r57_logged": changed_pixels
            >= int(thresholds["minimum_counterfactual_changed_pixels"]),
            "actor_visible_support_nontrivial": row["actor_effect_pixels"]
            >= int(thresholds["minimum_actor_effect_pixels"]),
            "compiled_actor_fields_match_native": actor_fields_pass,
            "compiled_full_sensor_matches_native": sensors_pass,
            "compiled_repeat_exact": bool(row["compiled_repeat_exact"]),
            "native_state_restored_exact": bool(row["native_translation_state_restored_exact"]),
            "package_manifest_immutable": bool(
                audit["package_manifest_sha256_before"]
                == audit["package_manifest_sha256_after"]
                == sources["r67_package_manifest_sha256"]
            ),
            "checkpoint_immutable": bool(
                audit["checkpoint_sha256_before"]
                == audit["checkpoint_sha256_after"]
                == sources["streetgs_checkpoint_sha256"]
            ),
            "four_factor_validity_preserved_and_unsupported_claims_abstain": bool(
                validity["joint_admissibility"] == "ACCEPT_CONFORMANCE"
                and validity["semantic_road"] == "ABSTAIN"
                and validity["physical_dynamics"] == "ABSTAIN"
                and validity["planning_safety"] == "ABSTAIN"
            ),
            "source_immutable": bool(
                all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items())
                and package_manifest == _verify_package(package)
            ),
            "gpu_within_budget": peak_mib <= float(resources["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
            "training_not_started": not bool(audit["training_started"]),
            "confirmation_not_read": not bool(audit["confirmation_content_read"]),
        }
        checks["passed"] = all(checks.values())
        with np.load(sensor_path, allow_pickle=False) as sensor_arrays:
            sensor_array_names = sorted(sensor_arrays.files)
        _write_json(
            run_dir / "R68_GATE.json",
            {
                "schema_version": "worldsim_v6.r68_gate.v1",
                "checks": checks,
                "decision": "accept_transform_owned_second_actor_sensor_runtime"
                if checks["passed"]
                else "reject_or_repair_transform_owned_second_actor_sensor_runtime",
            },
        )
        _write_json(
            run_dir / "R61_RUNTIME_EQUIVALENCE.json",
            {
                "schema_version": "worldsim_v6.r68_r61_runtime_equivalence.v1",
                "reference_sensor_sha256": sources["r61_sensor_sha256"],
                "actual_sensor_sha256": row["sensor_sha256"],
                "sensor_array_names": sensor_array_names,
                "sensor_arrays_exact": arrays_exact,
                "array_mismatches": array_mismatches,
                "changed_rgb_pixels_vs_r57_logged": changed_pixels,
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r68_resource_audit.v1",
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
            "schema_version": "worldsim_v6.r68_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_transform_owned_second_actor_sensor_runtime"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "proposal_id": contract["proposal_id"],
            "translation_delta_m": contract["translation_delta_m"],
            "frame_index": int(contract["frame_index"]),
            "actor_model_index": int(contract["actor_model_index"]),
            "runtime_mode": audit["runtime_mode"],
            "translation_source": audit["translation_source"],
            "lifecycle_source": audit["lifecycle_source"],
            "sensor_arrays_match_r61_exact": arrays_exact,
            "changed_rgb_pixels_vs_r57_logged": changed_pixels,
            "actor_effect_pixels": int(row["actor_effect_pixels"]),
            "compiled_native_rgb_mae": float(row["full_sensor_rgb_mae"]),
            "compiled_native_depth_mae_m": float(row["full_sensor_depth_mae_m"]),
            "joint_admissibility": validity["joint_admissibility"],
            "semantic_road": validity["semantic_road"],
            "physical_dynamics": validity["physical_dynamics"],
            "planning_safety": validity["planning_safety"],
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R68_GATE.json",
            "SUMMARY.json",
            "R61_RUNTIME_EQUIVALENCE.json",
            "RESOURCE_AUDIT.json",
            "worker.log",
            "worker/FRAME_METRICS.jsonl",
            "worker/WORKER_AUDIT.json",
            f"worker/{row['sensor_path']}",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r68_manifest.v1",
                "files": {
                    name: {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
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
        default=Path("configs/worldsim_v6/r68_transform_owned_second_actor_sensor_runtime_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
