"""WorldSim V6 R73: select a third compiler actor by frozen native visibility."""

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


TASK_ID = "WS-V6-R73-THIRD-ACTOR-VISIBILITY-SELECTION-01"


class R73ExperimentError(RuntimeError):
    """The preregistered R73 experiment contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R73ExperimentError("formal R73 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R73ExperimentError("R73 task_id drift")
    sources = config["sources"]
    selection = config["selection"]
    resources = config["resources"]
    r54_run = _resolve_runs_uri(sources["r54_run"])
    r72_run = _resolve_runs_uri(sources["r72_run"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    inventory_path = r54_run / "worker/ACTOR_INVENTORY.json"
    frozen_files = {
        r54_run / "MANIFEST.json": sources["r54_manifest_sha256"],
        r54_run / "R54_GATE.json": sources["r54_gate_sha256"],
        r54_run / "SUMMARY.json": sources["r54_summary_sha256"],
        inventory_path: sources["r54_inventory_sha256"],
        r72_run / "MANIFEST.json": sources["r72_manifest_sha256"],
        r72_run / "R72_GATE.json": sources["r72_gate_sha256"],
        r72_run / "SUMMARY.json": sources["r72_summary_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R73ExperimentError("StreetGS upstream commit drift")
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R73ExperimentError("R73 disk resource insufficient")
    r54_gate = json.loads((r54_run / "R54_GATE.json").read_text())
    r72_gate = json.loads((r72_run / "R72_GATE.json").read_text())
    inventory = json.loads(inventory_path.read_text())
    frame = int(selection["frame_index"])
    excluded = {int(value) for value in selection["excluded_actor_model_indices"]}
    minimum_primitives = int(selection["minimum_primitive_count"])
    derived_candidates = [
        int(actor["actor_model_index"])
        for actor in inventory["actors"]
        if actor["actor_model_index"] not in excluded
        and actor["primitive_count"] >= minimum_primitives
        and actor["first_active_frame"] is not None
        and actor["first_active_frame"] <= frame <= actor["last_active_frame"]
    ]
    expected_candidates = [int(value) for value in selection["expected_candidate_indices"]]
    if derived_candidates != expected_candidates:
        raise R73ExperimentError(
            f"candidate denominator drift: {derived_candidates} != {expected_candidates}"
        )
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__third-actor-visibility-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    worker_dir = run_dir / "worker"
    command = [
        sources["drivestudio_python"],
        str(repo_root / "scripts/worldsim_v6/r73_third_actor_visibility_worker.py"),
        "--repo-root", str(repo_root),
        "--checkpoint", str(checkpoint),
        "--upstream-root", str(upstream),
        "--frame", str(frame),
        "--candidates", ",".join(str(value) for value in expected_candidates),
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
    visibility = json.loads((worker_dir / "VISIBILITY.json").read_text())
    audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text())
    rows = visibility["rows"]
    selected = sorted(rows, key=lambda row: (-row["actor_effect_pixels"], row["actor_model_index"]))[0]
    decision = {
        "schema_version": "worldsim_v6.r73_third_actor_selection.v1",
        "frame_index": frame,
        "selection_rule": "maximum_actor_effect_pixels_then_minimum_actor_model_index",
        "candidate_actor_model_indices": expected_candidates,
        "selected_actor_model_index": selected["actor_model_index"],
        "selected_actor_effect_pixels": selected["actor_effect_pixels"],
        "selected_primitive_count": selected["primitive_count"],
        "semantic_identity": "ABSTAIN",
    }
    _write_json(run_dir / "THIRD_ACTOR_SELECTION.json", decision)
    checks = {
        "r54_and_r72_authorities_accepted": bool(r54_gate["checks"]["passed"] and r72_gate["checks"]["passed"]),
        "candidate_denominator_exact": visibility["candidates"] == expected_candidates and len(rows) == len(expected_candidates),
        "candidate_order_exact": [row["actor_model_index"] for row in rows] == expected_candidates,
        "all_candidates_native_active": all(row["native_frame_valid"] for row in rows),
        "all_candidate_primitive_counts_match_inventory": all(
            row["primitive_count"] == inventory["actors"][row["actor_model_index"]]["primitive_count"] for row in rows
        ),
        "selected_by_preregistered_rule": selected["actor_model_index"] == decision["selected_actor_model_index"],
        "selected_visible_support_nontrivial": selected["actor_effect_pixels"] >= int(selection["minimum_selected_effect_pixels"]),
        "checkpoint_immutable": audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"],
        "upstream_commit_exact": audit["upstream_commit"] == sources["streetgs_upstream_commit"],
        "frozen_sources_immutable": all(_sha256(path) == expected for path, expected in frozen_files.items()),
        "semantic_identity_physics_planning_safety_abstain": True,
        "gpu_within_budget": audit["peak_torch_reserved_bytes"] / (1024**2) <= float(resources["maximum_peak_gpu_memory_mib"]),
        "worker_within_budget": audit["wall_seconds"] <= float(resources["maximum_worker_seconds"]),
        "wall_within_budget": time.monotonic() - started <= float(resources["maximum_wall_seconds"]),
        "training_not_started": not audit["training_started"],
        "confirmation_not_read": not audit["confirmation_content_read"],
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R73_GATE.json",
        {
            "schema_version": "worldsim_v6.r73_gate.v1",
            "checks": checks,
            "decision": "accept_third_actor_for_compiler_extension" if checks["passed"] else "reject_or_repair_third_actor_selection",
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r73_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_third_actor_visibility_selection" if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "frame_index": frame,
        "candidate_visibility_rows": rows,
        "selected_actor_model_index": selected["actor_model_index"],
        "selected_actor_effect_pixels": selected["actor_effect_pixels"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R73_GATE.json", "SUMMARY.json", "THIRD_ACTOR_SELECTION.json", "worker.log", "worker/VISIBILITY.json", "worker/WORKER_AUDIT.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r73_manifest.v1",
            "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r73_third_actor_visibility_selection_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
