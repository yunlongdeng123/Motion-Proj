"""WorldSim V6 R86: execute a three-actor package over the complete 196-frame episode."""

from __future__ import annotations

import json
import shutil
import subprocess
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
from motion_proj.worldsim_v6.r70_two_actor_scene_package_bake import _verify_scene_package


TASK_ID = "WS-V6-R86-THREE-ACTOR-FULL-EPISODE-SENSOR-RUNTIME-01"


class R86ExperimentError(RuntimeError):
    """The preregistered R86 experiment contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R86ExperimentError("formal R86 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R86ExperimentError("R86 task_id drift")
    sources = config["sources"]
    contract = config["runtime_contract"]
    thresholds = config["thresholds"]
    resources = config["resources"]

    r82_run = _resolve_runs_uri(sources["r82_run"])
    scene_package = r82_run / "package"
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
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    package_manifest = _verify_scene_package(scene_package)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R86ExperimentError("StreetGS upstream commit drift")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R86ExperimentError("R86 disk resource insufficient")

    r82_gate = json.loads((r82_run / "R82_GATE.json").read_text(encoding="utf-8"))
    composition = json.loads(
        (scene_package / "SCENE_COMPOSITION.json").read_text(encoding="utf-8")
    )
    package_runtime = json.loads(
        (scene_package / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8")
    )
    validity = json.loads((scene_package / "VALIDITY.json").read_text(encoding="utf-8"))
    actors = composition["actors"]
    frame_indices = list(
        range(
            int(contract["frame_start"]),
            int(contract["frame_stop_exclusive"]),
            int(contract["frame_stride"]),
        )
    )
    if len(frame_indices) != int(contract["expected_frame_count"]):
        raise R86ExperimentError("R86 frame denominator drift")
    package_contract_exact = bool(
        package_runtime["runtime_mode"] == contract["runtime_mode"]
        and package_runtime["actor_order"] == contract["actor_ids"]
        and [row["actor_id"] for row in actors] == contract["actor_ids"]
        and [int(row["actor_model_index"]) for row in actors]
        == contract["actor_model_indices"]
        and [row["proposal_id"] for row in actors] == contract["proposal_ids"]
        and all(
            row["translation_delta_m"] == contract["translations_m"][row["actor_id"]]
            and int(row["primitive_count"])
            == int(contract["expected_primitives"][row["actor_id"]])
            for row in actors
        )
    )

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__three-actor-full-episode-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "worker"
        command = [
            sources["drivestudio_python"],
            str(repo_root / "scripts/worldsim_v6/r86_three_actor_full_episode_sensor_worker.py"),
            "--repo-root",
            str(repo_root),
            "--checkpoint",
            str(checkpoint),
            "--upstream-root",
            str(upstream),
            "--scene-package",
            str(scene_package),
            "--frames",
            ",".join(str(value) for value in frame_indices),
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
            raise R86ExperimentError(
                f"three-actor full-episode sensor worker failed: rc={completed.returncode}"
            )
        rows = [
            json.loads(line)
            for line in (worker_dir / "FRAME_METRICS.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if [int(row["frame_index"]) for row in rows] != frame_indices:
            raise R86ExperimentError("R86 output frame denominator drift")
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        expected_nested_manifests = {
            actor["actor_id"]: actor["actor_package_manifest_sha256"] for actor in actors
        }
        actor_field_pass = True
        sensor_pass = True
        runtime_binding_pass = True
        repeat_exact = True
        restored_exact = True
        inactive_zero_opacity = True
        sensor_sha256 = {}
        for row in rows:
            actor_rows = row["actors"]
            runtime_binding_pass = runtime_binding_pass and bool(
                list(actor_rows) == contract["actor_ids"]
                and all(
                    int(actor_rows[actor_id]["actor_model_index"])
                    == int(contract["actor_model_indices"][index])
                    and actor_rows[actor_id]["translation_delta_m"]
                    == contract["translations_m"][actor_id]
                    and bool(actor_rows[actor_id]["package_actor_frame_valid"])
                    == bool(
                        int(row["frame_index"])
                        < int(contract["active_stop_exclusive"][actor_id])
                    )
                    and int(actor_rows[actor_id]["primitive_count"])
                    == int(contract["expected_primitives"][actor_id])
                    for index, actor_id in enumerate(contract["actor_ids"])
                )
            )
            actor_field_pass = actor_field_pass and all(
                actor_rows[actor_id]["native_actor_field_max_error"]["means_m"]
                <= float(thresholds["maximum_means_error_m"])
                and actor_rows[actor_id]["native_actor_field_max_error"][
                    "quaternions_wxyz"
                ]
                <= float(thresholds["maximum_quaternion_error"])
                and all(
                    actor_rows[actor_id]["native_actor_field_max_error"][key]
                    <= float(thresholds["maximum_static_field_error"])
                    for key in ("scales_m", "opacities", "view_dependent_rgb")
                )
                for actor_id in contract["actor_ids"]
            )
            sensor_pass = sensor_pass and bool(
                row["full_sensor_rgb_mae"] <= float(thresholds["maximum_rgb_mae"])
                and row["full_sensor_rgb_p99_absolute_error"]
                <= float(thresholds["maximum_rgb_p99_absolute_error"])
                and row["full_sensor_depth_mae_m"]
                <= float(thresholds["maximum_depth_mae_m"])
                and row["full_sensor_opacity_mae"]
                <= float(thresholds["maximum_opacity_mae"])
            )
            repeat_exact = repeat_exact and bool(row["compiled_repeat_exact"])
            restored_exact = restored_exact and bool(
                row["native_translation_state_restored_exact"]
            )
            inactive_zero_opacity = inactive_zero_opacity and all(
                bool(actor_rows[actor_id]["package_actor_frame_valid"])
                or int(actor_rows[actor_id]["nonzero_opacity_primitives"]) == 0
                for actor_id in contract["actor_ids"]
            )
            sensor_path = worker_dir / row["sensor_path"]
            _verify(sensor_path, row["sensor_sha256"])
            sensor_sha256[str(row["frame_index"])] = row["sensor_sha256"]

        max_effect_by_actor = {
            actor_id: max(int(row["actors"][actor_id]["actor_effect_pixels"]) for row in rows)
            for actor_id in contract["actor_ids"]
        }
        validity_sequences = {
            actor_id: [
                bool(row["actors"][actor_id]["package_actor_frame_valid"]) for row in rows
            ]
            for actor_id in contract["actor_ids"]
        }
        maximum_errors = {
            "rgb_mae": max(float(row["full_sensor_rgb_mae"]) for row in rows),
            "rgb_p99_absolute_error": max(
                float(row["full_sensor_rgb_p99_absolute_error"]) for row in rows
            ),
            "depth_mae_m": max(float(row["full_sensor_depth_mae_m"]) for row in rows),
            "opacity_mae": max(float(row["full_sensor_opacity_mae"]) for row in rows),
        }
        wall_seconds = time.monotonic() - started
        peak_mib = int(audit["peak_torch_reserved_bytes"]) / (1024**2)
        output_bytes = sum(path.stat().st_size for path in worker_dir.rglob("*") if path.is_file())
        checks = {
            "r82_authority_accepted": bool(r82_gate["checks"]["passed"]),
            "full_196_frame_denominator_exact": len(rows)
            == int(contract["expected_frame_count"]),
            "scene_package_runtime_contract_exact": package_contract_exact,
            "worker_runtime_ownership_exact": bool(
                audit["runtime_mode"] == contract["runtime_mode"]
                and audit["actor_order"] == contract["actor_ids"]
                and audit["actor_model_indices"] == contract["actor_model_indices"]
                and audit["translation_source"] == contract["required_translation_source"]
                and audit["lifecycle_source"] == contract["required_lifecycle_source"]
            ),
            "all_frame_actor_translation_lifecycle_and_primitive_bindings_exact": runtime_binding_pass,
            "all_preregistered_lifecycle_active_counts_exact": all(
                sum(validity_sequences[actor_id])
                == int(contract["expected_active_frame_counts"][actor_id])
                for actor_id in contract["actor_ids"]
            ),
            "inactive_actor_states_have_zero_opacity": inactive_zero_opacity,
            "each_actor_has_nontrivial_visible_support": all(
                pixels >= int(thresholds["minimum_per_actor_max_effect_pixels"])
                for pixels in max_effect_by_actor.values()
            ),
            "joint_effect_nontrivial_in_at_least_one_frame": max(
                int(row["joint_actor_effect_pixels"]) for row in rows
            )
            >= int(thresholds["minimum_joint_max_effect_pixels"]),
            "edited_episode_has_nontrivial_visible_coverage": sum(
                int(row["edited_vs_logged_rgb_changed_pixels"])
                >= int(thresholds["minimum_changed_pixels_for_visible_frame"])
                for row in rows
            )
            >= int(thresholds["minimum_visible_edited_frame_count"]),
            "edited_episode_total_sensor_change_nontrivial": sum(
                int(row["edited_vs_logged_rgb_changed_pixels"]) for row in rows
            )
            >= int(thresholds["minimum_total_changed_rgb_pixels"]),
            "all_compiled_actor_fields_match_native": actor_field_pass,
            "all_compiled_shared_sensors_match_native": sensor_pass,
            "all_compiled_repeats_exact": repeat_exact,
            "native_multi_actor_state_restored_exact_every_frame": restored_exact,
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
            "gpu_within_budget": peak_mib
            <= float(resources["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
            "output_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
            "training_not_started": not bool(audit["training_started"]),
            "confirmation_not_read": not bool(audit["confirmation_content_read"]),
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R86_GATE.json",
            {
                "schema_version": "worldsim_v6.r86_gate.v1",
                "checks": checks,
                "decision": "accept_three_actor_full_episode_sensor_runtime"
                if checks["passed"]
                else "reject_or_repair_three_actor_full_episode_sensor_runtime",
            },
        )
        _write_json(
            run_dir / "FULL_EPISODE_SENSOR_EFFECT.json",
            {
                "schema_version": "worldsim_v6.r86_full_episode_sensor_effect.v1",
                "frame_indices": frame_indices,
                "validity_sequences": validity_sequences,
                "max_effect_pixels_by_actor": max_effect_by_actor,
                "joint_effect_pixels_by_frame": {
                    str(row["frame_index"]): int(row["joint_actor_effect_pixels"])
                    for row in rows
                },
                "sensor_sha256_by_frame": sensor_sha256,
                "edited_vs_logged_changed_rgb_pixels_by_frame": {
                    str(row["frame_index"]): int(row["edited_vs_logged_rgb_changed_pixels"])
                    for row in rows
                },
                "edited_vs_logged_rgb_mae_by_frame": {
                    str(row["frame_index"]): float(row["edited_vs_logged_rgb_mae"])
                    for row in rows
                },
                "maximum_compiled_native_errors": maximum_errors,
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r86_resource_audit.v1",
                "gpu_used": True,
                "peak_torch_reserved_mib": peak_mib,
                "worker_wall_seconds": float(audit["wall_seconds"]),
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "worker_output_bytes": output_bytes,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r86_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_three_actor_full_episode_sensor_runtime"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "frame_indices": frame_indices,
            "actor_ids": contract["actor_ids"],
            "validity_sequences": validity_sequences,
            "max_effect_pixels_by_actor": max_effect_by_actor,
            "maximum_compiled_native_errors": maximum_errors,
            "visible_edited_frame_count": sum(
                int(row["edited_vs_logged_rgb_changed_pixels"])
                >= int(thresholds["minimum_changed_pixels_for_visible_frame"])
                for row in rows
            ),
            "total_changed_rgb_pixels_vs_logged": sum(
                int(row["edited_vs_logged_rgb_changed_pixels"]) for row in rows
            ),
            "worker_output_bytes": output_bytes,
            "q_joint_aabb_interaction": "ACCEPT",
            "q_collision_physics": "ABSTAIN",
            "semantic_road": "ABSTAIN",
            "physical_dynamics": "ABSTAIN",
            "planning_safety": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R86_GATE.json",
            "SUMMARY.json",
            "FULL_EPISODE_SENSOR_EFFECT.json",
            "RESOURCE_AUDIT.json",
            "worker.log",
            "worker/FRAME_METRICS.jsonl",
            "worker/WORKER_AUDIT.json",
        ]
        tracked.extend(f"worker/{row['sensor_path']}" for row in rows)
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r86_manifest.v1",
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
        default=Path("configs/worldsim_v6/r86_three_actor_full_episode_sensor_runtime_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
