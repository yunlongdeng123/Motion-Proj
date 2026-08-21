"""WorldSim V6 R96: exhaustive front-camera recovery after R95 rejection."""

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
    _git, _resolve_runs_uri, _sha256, _verify, _write_json,
)


TASK_ID = "WS-V6-R96-EXHAUSTIVE-FRONT-CAMERA-VISIBILITY-01"


class R96ExperimentError(RuntimeError):
    """The preregistered R96 contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R96ExperimentError("formal R96 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R96ExperimentError("R96 task_id drift")
    sources = config["sources"]
    scan = config["scan"]
    resources = config["resources"]
    r95_run = _resolve_runs_uri(sources["r95_run"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    inventory_path = r95_run / "worker/ACTOR_INVENTORY.json"
    visibility_path = r95_run / "worker/VISIBILITY.json"
    frozen_files = {
        r95_run / "MANIFEST.json": sources["r95_manifest_sha256"],
        r95_run / "R95_GATE.json": sources["r95_gate_sha256"],
        r95_run / "SUMMARY.json": sources["r95_summary_sha256"],
        r95_run / "ACTOR_VISIBILITY_SELECTION.json": sources["r95_selection_sha256"],
        inventory_path: sources["r95_inventory_sha256"],
        visibility_path: sources["r95_visibility_sha256"],
        Path(sources["failure_ledger"]): sources["failure_ledger_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R96ExperimentError("StreetGS upstream commit drift")
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R96ExperimentError("R96 disk resource insufficient")
    r95_gate = json.loads((r95_run / "R95_GATE.json").read_text())
    inventory = json.loads(inventory_path.read_text())
    lifecycle = np.load(r95_run / "worker" / inventory["lifecycle_matrix_path"], allow_pickle=False)
    candidates = [int(value) for value in scan["candidate_actor_model_indices"]]
    frames = list(range(int(scan["frame_start"]), int(scan["frame_stop_exclusive"]), int(scan["frame_stride"])))
    if len(frames) != int(scan["expected_frame_count"]):
        raise R96ExperimentError("R96 frame denominator drift")
    expected_pairs = {
        (frame, actor_index) for frame in frames for actor_index in candidates
        if bool(lifecycle[frame, actor_index])
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__scene0048-exhaustive-front-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    worker_dir = run_dir / "worker"
    command = [
        sources["drivestudio_python"],
        str(repo_root / "scripts/worldsim_v6/r96_exhaustive_front_camera_visibility_worker.py"),
        "--repo-root", str(repo_root), "--checkpoint", str(checkpoint),
        "--upstream-root", str(upstream), "--frames", ",".join(map(str, frames)),
        "--candidates", ",".join(map(str, candidates)), "--camera-offset", str(scan["camera_offset"]),
        "--opacity-threshold", str(scan["opacity_threshold"]), "--output", str(worker_dir),
    ]
    with (run_dir / "worker.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command, cwd=repo_root, stdout=log_stream, stderr=subprocess.STDOUT,
            check=True, timeout=float(resources["maximum_worker_seconds"]),
        )
    exhaustive = json.loads((worker_dir / "EXHAUSTIVE_VISIBILITY.json").read_text())
    audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text())
    rows = exhaustive["rows"]
    if not rows:
        raise R96ExperimentError("no lifecycle-active actor/frame rows")
    selected = sorted(
        rows,
        key=lambda row: (-row["actor_effect_pixels"], row["actor_model_index"], row["frame_index"]),
    )[0]
    decision = {
        "schema_version": "worldsim_v6.r96_exhaustive_front_selection.v1",
        "scene": "scene-0048", "camera_offset": int(scan["camera_offset"]),
        "frame_denominator": len(frames), "active_pair_denominator": len(expected_pairs),
        "selection_rule": scan["selection_rule"],
        "selected_actor_model_index": selected["actor_model_index"],
        "selected_frame_index": selected["frame_index"],
        "selected_actor_effect_pixels": selected["actor_effect_pixels"],
        "selected_primitive_count": selected["primitive_count"], "semantic_identity": "ABSTAIN",
    }
    _write_json(run_dir / "EXHAUSTIVE_FRONT_SELECTION.json", decision)
    failed_checks = sorted(
        key for key, value in r95_gate["checks"].items() if key != "passed" and not value
    )
    observed_pairs = {(row["frame_index"], row["actor_model_index"]) for row in rows}
    checks = {
        "r95_rejection_retained_and_failure_is_visibility_only": not r95_gate["checks"]["passed"]
        and failed_checks == ["selected_visible_support_nontrivial"],
        "v6_f85_failure_ledger_bound": "V6-F85" in Path(sources["failure_ledger"]).read_text(encoding="utf-8"),
        "full196_front_frame_denominator_exact": exhaustive["frames"] == frames
        and len(frames) == int(scan["expected_frame_count"]),
        "candidate_denominator_exact": exhaustive["candidates"] == candidates
        and candidates == list(range(inventory["actor_model_count"])),
        "all_and_only_lifecycle_active_pairs_rendered": observed_pairs == expected_pairs
        and len(rows) == len(expected_pairs) == audit["row_count"],
        "row_primitive_counts_match_r95_inventory": all(
            row["primitive_count"] == inventory["actors"][row["actor_model_index"]]["primitive_count"]
            for row in rows
        ),
        "selected_by_preregistered_rule": selected == sorted(
            rows,
            key=lambda row: (-row["actor_effect_pixels"], row["actor_model_index"], row["frame_index"]),
        )[0],
        "selected_visible_support_nontrivial": selected["actor_effect_pixels"]
        >= int(scan["minimum_selected_effect_pixels"]),
        "checkpoint_immutable": audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"]
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
        run_dir / "R96_GATE.json",
        {"schema_version": "worldsim_v6.r96_gate.v1", "checks": checks,
         "decision": "accept_exhaustive_front_visible_actor_for_scene0048"
         if checks["passed"] else "reject_front_camera_and_expand_to_three_camera_coverage"},
    )
    summary = {
        "schema_version": "worldsim_v6.r96_summary.v1", "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_exhaustive_front_camera_actor_selection"
        if checks["passed"] else "rejected", "source_commit": source_commit,
        "scene": "scene-0048", "frame_count": len(frames), "active_pair_count": len(rows),
        "selected_actor_model_index": selected["actor_model_index"],
        "selected_frame_index": selected["frame_index"],
        "selected_actor_effect_pixels": selected["actor_effect_pixels"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R96_GATE.json", "SUMMARY.json", "EXHAUSTIVE_FRONT_SELECTION.json", "worker.log",
        "worker/EXHAUSTIVE_VISIBILITY.json", "worker/WORKER_AUDIT.json",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {"schema_version": "worldsim_v6.r96_manifest.v1", "files": {
            name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked
        }},
    )
    _write_json(
        run_dir / "TERMINAL.json",
        {"schema_version": "worldsim_v6.terminal.v1", "status": summary["status"],
         "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
         "summary_sha256": _sha256(run_dir / "SUMMARY.json")},
    )
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r96_exhaustive_front_camera_visibility_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
