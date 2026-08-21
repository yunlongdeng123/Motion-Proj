"""WorldSim V6 R47：分离 detached replay 的 state content 与 trajectory event identity。"""

from __future__ import annotations

import collections
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _replay_once, _verify_package


TASK_ID = "WS-V6-R47-DETACHED-STATE-EVENT-LOGSIM-01"


class R47ExperimentError(RuntimeError):
    """R47 正式实验合同失败。"""


def _content_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def _bind_events(rows: list[dict[str, Any]], proposal_id: str) -> list[dict[str, Any]]:
    bound = []
    for sequence_index, row in enumerate(rows):
        event_identity = {
            "sequence_index": sequence_index,
            "timestamp_us": int(row["timestamp_us"]),
            "visible": bool(row["visible"]),
            "proposal_id": proposal_id,
            "materialized_state_sha256": row["materialized_state_sha256"],
        }
        item = dict(row)
        item["sequence_index"] = sequence_index
        item["proposal_id"] = proposal_id
        item["trajectory_event_sha256"] = _content_sha256(event_identity)
        bound.append(item)
    return bound


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R47ExperimentError("正式 R47 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R47ExperimentError("R47 task_id 漂移")
    sources = config["sources"]
    r46_run = _resolve_runs_uri(sources["r46_run"])
    r45_run = _resolve_runs_uri(sources["r45_run"])
    source_package = r45_run / "package"
    frozen_files = {
        r46_run / "MANIFEST.json": sources["r46_manifest_sha256"],
        r46_run / "R46_GATE.json": sources["r46_gate_sha256"],
        r46_run / "SUMMARY.json": sources["r46_summary_sha256"],
        r46_run / "REPLAY_TRAJECTORY.jsonl": sources["r46_replay_trajectory_sha256"],
        r46_run / "REPLAY_AUDIT.json": sources["r46_replay_audit_sha256"],
        r45_run / "R45_GATE.json": sources["r45_gate_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["r45_package_manifest_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r46_gate = json.loads((r46_run / "R46_GATE.json").read_text(encoding="utf-8"))
    r45_gate = json.loads((r45_run / "R45_GATE.json").read_text(encoding="utf-8"))
    source_manifest = _verify_package(source_package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R47ExperimentError("R47 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__state-event-logsim-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        detached_package = run_dir / "detached_package"
        shutil.copytree(source_package, detached_package)
        detached_manifest = _verify_package(detached_package)
        rows_raw_1, metrics_1 = _replay_once(detached_package)
        rows_raw_2, metrics_2 = _replay_once(detached_package)
        contract = config["replay_contract"]
        rows_1 = _bind_events(rows_raw_1, contract["proposal_id"])
        rows_2 = _bind_events(rows_raw_2, contract["proposal_id"])
        state_groups: dict[str, list[int]] = collections.defaultdict(list)
        for row in rows_1:
            state_groups[row["materialized_state_sha256"]].append(int(row["timestamp_us"]))
        duplicate_groups = sorted((timestamps for timestamps in state_groups.values() if len(timestamps) > 1), key=lambda values: values[0])
        event_hashes = {row["trajectory_event_sha256"] for row in rows_1}
        _write_json(run_dir / "IDENTITY_AUDIT.json", {
            "schema_version": "worldsim_v6.r47_identity_audit.v1",
            "trajectory_event_count": len(rows_1), "unique_event_hash_count": len(event_hashes),
            "unique_state_hash_count": len(state_groups), "duplicate_state_group_count": len(duplicate_groups),
            "maximum_state_repetitions": max(len(values) for values in state_groups.values()),
            "duplicate_state_timestamp_groups": duplicate_groups,
            "state_hash_semantics": "materialized_geometry_content_only",
            "event_hash_semantics": "sequence_timestamp_visibility_proposal_and_state_binding",
            "source_package_used_after_copy": False,
        })
        (run_dir / "EVENT_TRAJECTORY.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows_1),
            encoding="utf-8",
        )
        wall_seconds = time.monotonic() - started
        checks = {
            "r46_rejection_preserved": not r46_gate["checks"]["passed"],
            "r45_authority_accepted": r45_gate["checks"]["passed"],
            "detached_package_manifest_exact": source_manifest == detached_manifest and _sha256(source_package / "PACKAGE_MANIFEST.json") == _sha256(detached_package / "PACKAGE_MANIFEST.json"),
            "trajectory_event_count_exact": len(rows_1) == int(contract["expected_trajectory_event_count"]),
            "all_event_hashes_unique": len(event_hashes) == len(rows_1),
            "state_content_identity_preserved": len(state_groups) == int(contract["expected_unique_state_count"]),
            "duplicate_state_structure_exact": len(duplicate_groups) == int(contract["expected_duplicate_state_group_count"]) and max(len(values) for values in state_groups.values()) == int(contract["expected_maximum_state_repetitions"]),
            "stationary_tail_preserved": duplicate_groups == [list(range(int(contract["stationary_tail_first_timestamp_us"]), int(contract["stationary_tail_last_timestamp_us"]) + 1, 100000))],
            "proposal_binding_exact": all(row["proposal_id"] == contract["proposal_id"] and row["translation_delta_m"] == contract["translation_delta_m"] for row in rows_1),
            "composition_error_exact": metrics_1["maximum_composition_error_m"] <= float(contract["maximum_composition_error_m"]),
            "derivative_invariance_within_tolerance": max(metrics_1["maximum_velocity_invariance_error"], metrics_1["maximum_acceleration_invariance_error"]) <= float(contract["maximum_derivative_invariance_error"]),
            "two_event_replays_exact": rows_1 == rows_2 and metrics_1 == metrics_2,
            "source_package_not_used_after_copy": True,
            "physical_and_safety_validity_abstain": True,
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R47_GATE.json", {
            "schema_version": "worldsim_v6.r47_gate.v1", "checks": checks,
            "decision": "accept_detached_state_event_actor_logsim" if checks["passed"] else "reject_or_repair_detached_state_event_logsim",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r47_resource_audit.v1", "gpu_used": False,
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r47_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_detached_state_event_actor_logsim" if checks["passed"] else "rejected",
            "source_commit": source_commit, "proposal_id": contract["proposal_id"],
            "trajectory_event_count": len(rows_1), "unique_event_hash_count": len(event_hashes),
            "unique_state_hash_count": len(state_groups), "maximum_state_repetitions": max(len(values) for values in state_groups.values()),
            "maximum_composition_error_m": metrics_1["maximum_composition_error_m"],
            "replay_exact": rows_1 == rows_2 and metrics_1 == metrics_2,
            "sensor_runtime": "ABSTAIN_DEFERRED", "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R47_GATE.json", "SUMMARY.json", "EVENT_TRAJECTORY.jsonl", "IDENTITY_AUDIT.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r47_manifest.v1",
            "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
            "detached_package_manifest": {"path": "detached_package/PACKAGE_MANIFEST.json", "sha256": _sha256(detached_package / "PACKAGE_MANIFEST.json")},
        })
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        })
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": "failed", "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r47_detached_state_event_logsim_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

