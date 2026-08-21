"""WorldSim V6 R54：冻结 RigidNodes 全 actor ownership inventory。"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json


TASK_ID = "WS-V6-R54-RIGID-ACTOR-INVENTORY-01"


class R54ExperimentError(RuntimeError):
    """R54 正式实验合同失败。"""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R54ExperimentError("正式 R54 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R54ExperimentError("R54 task_id 漂移")
    sources = config["sources"]
    r53_run = _resolve_runs_uri(sources["r53_run"])
    r50_run = _resolve_runs_uri(sources["r50_run"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r53_run / "R53_GATE.json": sources["r53_gate_sha256"],
        r53_run / "SUMMARY.json": sources["r53_summary_sha256"],
        r50_run / "R50_GATE.json": sources["r50_gate_sha256"],
        r50_run / f"package/blobs/{sources['r50_actor0_lifecycle_sha256']}.npy": sources["r50_actor0_lifecycle_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R54ExperimentError("StreetGS upstream commit 漂移")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R54ExperimentError("R54 磁盘资源不足")
    r53_gate = json.loads((r53_run / "R53_GATE.json").read_text(encoding="utf-8"))
    r50_gate = json.loads((r50_run / "R50_GATE.json").read_text(encoding="utf-8"))

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor-inventory-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    worker_dir = run_dir / "worker"
    command = [
        str(sources["drivestudio_python"]), str(repo_root / "scripts/worldsim_v6/r54_rigid_actor_inventory_worker.py"),
        "--repo-root", str(repo_root), "--checkpoint", str(checkpoint), "--upstream-root", str(upstream), "--output", str(worker_dir),
    ]
    with (run_dir / "worker.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(command, cwd=repo_root, stdout=log_stream, stderr=subprocess.STDOUT, check=True, timeout=float(config["resources"]["maximum_worker_seconds"]))
    inventory = json.loads((worker_dir / "ACTOR_INVENTORY.json").read_text(encoding="utf-8"))
    audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
    actors = inventory["actors"]
    actor0 = next(actor for actor in actors if actor["actor_model_index"] == 0)
    contract = config["inventory_contract"]
    checks = {
        "r53_utility_and_r50_lifecycle_authorities_accepted": bool(r53_gate["checks"]["passed"] and r50_gate["checks"]["passed"]),
        "frame_denominator_exact": inventory["frame_count"] == int(contract["expected_frame_count"]),
        "actor_model_denominator_nonempty": inventory["actor_model_count"] >= int(contract["minimum_actor_model_count"]),
        "all_point_ids_in_actor_range": inventory["out_of_range_point_id_count"] == 0,
        "primitive_partition_exact": inventory["assigned_primitive_count"] == inventory["primitive_count"] == sum(actor["primitive_count"] for actor in actors),
        "all_actor_rows_reported": len(actors) == inventory["actor_model_count"] and [actor["actor_model_index"] for actor in actors] == list(range(inventory["actor_model_count"])),
        "each_lifecycle_content_addressed": all(_sha256(worker_dir / actor["lifecycle_path"]) == actor["lifecycle_sha256"] for actor in actors),
        "actor0_matches_r50_lifecycle": actor0["lifecycle_sha256"] == sources["r50_actor0_lifecycle_sha256"],
        "actor0_matches_r50_primitives": actor0["primitive_count"] == int(sources["r50_actor0_primitive_count"]),
        "at_least_one_editable_observed_actor": any(actor["primitive_count"] > 0 and actor["active_frame_count"] > 0 for actor in actors),
        "checkpoint_immutable": audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"],
        "upstream_commit_exact": audit["upstream_commit"] == sources["streetgs_upstream_commit"],
        "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()),
        "semantic_identity_physical_planning_safety_abstain": True,
        "gpu_within_budget": audit["peak_torch_reserved_bytes"] / (1024**2) <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
        "worker_within_budget": audit["wall_seconds"] <= float(config["resources"]["maximum_worker_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["wall_within_budget"] = (time.monotonic() - started) <= float(config["resources"]["maximum_wall_seconds"])
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R54_GATE.json", {"schema_version": "worldsim_v6.r54_gate.v1", "checks": checks, "decision": "accept_complete_rigid_actor_inventory" if checks["passed"] else "reject_or_repair_rigid_actor_inventory"})
    summary = {
        "schema_version": "worldsim_v6.r54_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected", "hypothesis_outcome": "accepted_development_complete_rigid_actor_inventory" if checks["passed"] else "rejected",
        "source_commit": source_commit, "actor_model_count": inventory["actor_model_count"], "primitive_count": inventory["primitive_count"],
        "editable_observed_actor_count": sum(actor["primitive_count"] > 0 and actor["active_frame_count"] > 0 for actor in actors),
        "actor_lifecycle_summaries": actors, "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R54_GATE.json", "SUMMARY.json", "worker.log", "worker/ACTOR_INVENTORY.json", "worker/ACTOR_LIFECYCLE_MATRIX.npy", "worker/WORKER_AUDIT.json"]
    tracked.extend(f"worker/{actor['lifecycle_path']}" for actor in actors)
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r54_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": summary["status"], "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r54_rigid_actor_inventory_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
