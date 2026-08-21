"""WorldSim V6 R92: independent-scene actor inventory plus visibility selection."""

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
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R92-INDEPENDENT-SCENE-ACTOR-VISIBILITY-INVENTORY-01"


class R92ExperimentError(RuntimeError):
    """The preregistered R92 contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R92ExperimentError("formal R92 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R92ExperimentError("R92 task_id drift")
    sources = config["sources"]
    contract = config["inventory_contract"]
    selection = config["visibility_selection"]
    resources = config["resources"]
    matched_run = Path(sources["matched_formal_run"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        matched_run / "manifest.json": sources["matched_manifest_sha256"],
        matched_run / "summary.json": sources["matched_summary_sha256"],
        matched_run / "status.json": sources["matched_status_sha256"],
        matched_run / "fingerprint.json": sources["matched_fingerprint_sha256"],
        matched_run / "resolved.yaml": sources["matched_resolved_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R92ExperimentError("StreetGS upstream commit drift")
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R92ExperimentError("R92 disk resource insufficient")
    matched_status = json.loads((matched_run / "status.json").read_text())
    matched_summary = json.loads((matched_run / "summary.json").read_text())
    probe_frames = [int(value) for value in selection["probe_frames"]]

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__scene0230-actor-visibility-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    worker_dir = run_dir / "worker"
    command = [
        sources["drivestudio_python"],
        str(repo_root / "scripts/worldsim_v6/r92_independent_scene_actor_visibility_worker.py"),
        "--repo-root", str(repo_root),
        "--checkpoint", str(checkpoint),
        "--upstream-root", str(upstream),
        "--probe-frames", ",".join(str(value) for value in probe_frames),
        "--minimum-primitives", str(selection["minimum_primitive_count"]),
        "--opacity-threshold", str(selection["opacity_threshold"]),
        "--output", str(worker_dir),
    ]
    with (run_dir / "worker.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=float(resources["maximum_worker_seconds"]),
        )
    inventory = json.loads((worker_dir / "ACTOR_INVENTORY.json").read_text())
    visibility = json.loads((worker_dir / "VISIBILITY.json").read_text())
    audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text())
    actors = inventory["actors"]
    lifecycle = np.load(worker_dir / inventory["lifecycle_matrix_path"], allow_pickle=False)
    derived_candidates = [
        actor["actor_model_index"]
        for actor in actors
        if actor["primitive_count"] >= int(selection["minimum_primitive_count"])
        and any(lifecycle[frame, actor["actor_model_index"]] for frame in probe_frames)
    ]
    rows = visibility["rows"]
    selected = sorted(
        (row for row in rows if row["native_frame_valid"]),
        key=lambda row: (-row["actor_effect_pixels"], row["actor_model_index"], row["frame_index"]),
    )[0]
    decision = {
        "schema_version": "worldsim_v6.r92_independent_scene_actor_selection.v1",
        "scene": sources["scene"],
        "scene_index": int(sources["scene_index"]),
        "probe_frames": probe_frames,
        "selection_rule": selection["selection_rule"],
        "candidate_actor_model_indices": derived_candidates,
        "selected_actor_model_index": selected["actor_model_index"],
        "selected_frame_index": selected["frame_index"],
        "selected_actor_effect_pixels": selected["actor_effect_pixels"],
        "selected_primitive_count": selected["primitive_count"],
        "semantic_identity": "ABSTAIN",
    }
    _write_json(run_dir / "ACTOR_VISIBILITY_SELECTION.json", decision)
    expected_pairs = {
        (frame, actor_index) for frame in probe_frames for actor_index in derived_candidates
    }
    observed_pairs = {(row["frame_index"], row["actor_model_index"]) for row in rows}
    checks = {
        "matched_formal_source_done_and_scene_exact": matched_status["status"] == "done"
        and matched_status["scene"] == sources["scene"]
        and matched_summary["scene"] == sources["scene"],
        "matched_checkpoint_authority_exact": matched_summary["checkpoint"]["sha256"]
        == sources["streetgs_checkpoint_sha256"],
        "frame_denominator_exact": inventory["frame_count"] == int(contract["expected_frame_count"]),
        "actor_model_denominator_nonempty": inventory["actor_model_count"]
        >= int(contract["minimum_actor_model_count"]),
        "all_point_ids_in_actor_range": inventory["out_of_range_point_id_count"] == 0,
        "primitive_partition_exact": inventory["assigned_primitive_count"]
        == inventory["primitive_count"]
        == sum(actor["primitive_count"] for actor in actors),
        "rigid_primitive_count_matches_formal_summary": inventory["primitive_count"]
        == int(contract["expected_rigid_primitive_count"])
        == int(matched_summary["checkpoint"]["gaussian_counts"]["RigidNodes"]),
        "all_actor_rows_reported": len(actors) == inventory["actor_model_count"]
        and [actor["actor_model_index"] for actor in actors]
        == list(range(inventory["actor_model_count"])),
        "each_lifecycle_content_addressed": all(
            _sha256(worker_dir / actor["lifecycle_path"]) == actor["lifecycle_sha256"]
            for actor in actors
        ),
        "fixed_probe_frames_exact": visibility["probe_frames"] == probe_frames,
        "candidate_derivation_exact_and_nonempty": visibility["candidate_actor_model_indices"]
        == derived_candidates and bool(derived_candidates),
        "visibility_pair_denominator_exact": observed_pairs == expected_pairs
        and len(rows) == len(expected_pairs),
        "row_primitive_counts_match_inventory": all(
            row["primitive_count"] == actors[row["actor_model_index"]]["primitive_count"]
            for row in rows
        ),
        "inactive_rows_have_zero_effect": all(
            row["native_frame_valid"] or row["actor_effect_pixels"] == 0 for row in rows
        ),
        "selected_by_preregistered_rule": decision["selected_actor_model_index"]
        == selected["actor_model_index"] and decision["selected_frame_index"] == selected["frame_index"],
        "selected_visible_support_nontrivial": selected["actor_effect_pixels"]
        >= int(selection["minimum_selected_effect_pixels"]),
        "checkpoint_immutable": audit["checkpoint_sha256_before"]
        == audit["checkpoint_sha256_after"]
        == sources["streetgs_checkpoint_sha256"],
        "upstream_commit_exact": audit["upstream_commit"] == sources["streetgs_upstream_commit"],
        "frozen_sources_immutable": all(_sha256(path) == expected for path, expected in frozen_files.items()),
        "semantic_identity_edit_validity_physics_planning_safety_abstain": True,
        "gpu_within_budget": audit["peak_torch_reserved_bytes"] / (1024**2)
        <= float(resources["maximum_peak_gpu_memory_mib"]),
        "worker_within_budget": audit["wall_seconds"] <= float(resources["maximum_worker_seconds"]),
        "wall_within_budget": time.monotonic() - started <= float(resources["maximum_wall_seconds"]),
        "training_not_started": not audit["training_started"],
        "confirmation_not_read": not audit["confirmation_content_read"],
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R92_GATE.json",
        {
            "schema_version": "worldsim_v6.r92_gate.v1",
            "checks": checks,
            "decision": "accept_independent_scene_actor_for_compiler_extension"
            if checks["passed"] else "reject_or_repair_independent_scene_actor_selection",
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r92_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_independent_scene_actor_visibility_selection"
        if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "scene": sources["scene"],
        "frame_count": inventory["frame_count"],
        "actor_model_count": inventory["actor_model_count"],
        "rigid_primitive_count": inventory["primitive_count"],
        "candidate_actor_model_count": len(derived_candidates),
        "selected_actor_model_index": selected["actor_model_index"],
        "selected_frame_index": selected["frame_index"],
        "selected_actor_effect_pixels": selected["actor_effect_pixels"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R92_GATE.json", "SUMMARY.json", "ACTOR_VISIBILITY_SELECTION.json", "worker.log",
        "worker/ACTOR_INVENTORY.json", "worker/ACTOR_LIFECYCLE_MATRIX.npy",
        "worker/VISIBILITY.json", "worker/WORKER_AUDIT.json",
    ]
    tracked.extend(f"worker/{actor['lifecycle_path']}" for actor in actors)
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r92_manifest.v1",
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
        default=Path("configs/worldsim_v6/r92_independent_scene_actor_visibility_inventory_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
