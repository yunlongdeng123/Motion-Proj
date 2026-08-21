"""WorldSim V6 R91: replay a baked selective policy in two fresh CPU processes."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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


TASK_ID = "WS-V6-R91-SELECTIVE-POLICY-FRESH-PROCESS-REPLAY-01"


class R91ExperimentError(RuntimeError):
    """The preregistered R91 experiment contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R91ExperimentError("formal R91 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R91ExperimentError("R91 task_id drift")
    sources = config["sources"]
    resources = config["resources"]
    r86_run = _resolve_runs_uri(sources["r86_run"])
    r90_run = _resolve_runs_uri(sources["r90_run"])
    package = r90_run / "package_a"
    feature_map = r86_run / "FULL_EPISODE_SENSOR_EFFECT.json"
    frozen_files = {
        feature_map: sources["r86_full_episode_sensor_effect_sha256"],
        r90_run / "MANIFEST.json": sources["r90_manifest_sha256"],
        r90_run / "R90_GATE.json": sources["r90_gate_sha256"],
        r90_run / "SUMMARY.json": sources["r90_summary_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r90_package_manifest_sha256"],
        package / "POLICY.json": sources["r90_policy_sha256"],
        package / "DECISIONS.jsonl": sources["r90_decisions_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R91ExperimentError("R91 disk resource insufficient")
    r90_gate = json.loads((r90_run / "R90_GATE.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__fresh-policy-replay-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    worker = repo_root / "scripts/worldsim_v6/r91_selective_policy_replay_worker.py"
    process_records = []
    for repeat in range(int(config["runtime"]["fresh_process_count"])):
        output = run_dir / f"process_{repeat}"
        completed = subprocess.run(
            [
                sys.executable,
                str(worker),
                "--package",
                str(package),
                "--features",
                str(feature_map),
                "--output",
                str(output),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=float(resources["maximum_worker_seconds"]),
        )
        (run_dir / f"process_{repeat}.log").write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise R91ExperimentError(f"fresh policy worker failed: process={repeat}")
        audit = json.loads((output / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        process_records.append(
            {
                "repeat_index": repeat,
                "decisions_sha256": _sha256(output / "DECISIONS.jsonl"),
                "audit_sha256": _sha256(output / "WORKER_AUDIT.json"),
                "audit": audit,
            }
        )
    policy_decisions_sha = sources["r90_decisions_sha256"]
    wall_seconds = time.monotonic() - started
    checks = {
        "r90_authority_accepted": bool(r90_gate["checks"]["passed"]),
        "two_fresh_processes_completed": len(process_records) == 2,
        "fresh_process_decisions_repeat_exact": len(
            {row["decisions_sha256"] for row in process_records}
        )
        == 1,
        "fresh_process_audits_repeat_exact": len({row["audit_sha256"] for row in process_records})
        == 1,
        "fresh_decisions_equal_baked_decisions": all(
            row["decisions_sha256"] == policy_decisions_sha for row in process_records
        ),
        "full_frame_trigger_skip_denominators_exact": all(
            row["audit"]["frame_count"] == 196
            and row["audit"]["trigger_count"] == 151
            and row["audit"]["skip_count"] == 45
            for row in process_records
        ),
        "policy_and_feature_content_addresses_bound": all(
            row["audit"]["policy_manifest_sha256"]
            == sources["r90_package_manifest_sha256"]
            and row["audit"]["feature_map_sha256"]
            == sources["r86_full_episode_sensor_effect_sha256"]
            for row in process_records
        ),
        "torch_and_perception_model_not_loaded": all(
            not row["audit"]["torch_imported"]
            and not row["audit"]["perception_model_loaded"]
            for row in process_records
        ),
        "training_and_confirmation_not_used": all(
            not row["audit"]["training_started"]
            and not row["audit"]["confirmation_content_read"]
            for row in process_records
        ),
        "frozen_sources_immutable": all(
            _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
        ),
        "cpu_only_within_wall_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "cross_scene_model_semantics_and_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R91_GATE.json",
        {
            "schema_version": "worldsim_v6.r91_gate.v1",
            "checks": checks,
            "decision": "accept_selective_policy_fresh_process_runtime"
            if checks["passed"]
            else "reject_or_repair_selective_policy_fresh_process_runtime",
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r91_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_selective_policy_fresh_process_runtime"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "fresh_process_count": len(process_records),
        "decisions_sha256": process_records[0]["decisions_sha256"],
        "trigger_count": process_records[0]["audit"]["trigger_count"],
        "skip_count": process_records[0]["audit"]["skip_count"],
        "torch_imported": False,
        "perception_model_loaded": False,
        "cross_scene_transfer": "ABSTAIN",
        "cross_model_transfer": "ABSTAIN",
        "semantic_correctness": "ABSTAIN",
        "physical_planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R91_GATE.json", "SUMMARY.json"]
    for repeat in range(2):
        tracked.extend(
            [
                f"process_{repeat}.log",
                f"process_{repeat}/DECISIONS.jsonl",
                f"process_{repeat}/WORKER_AUDIT.json",
            ]
        )
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r91_manifest.v1",
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
        default=Path("configs/worldsim_v6/r91_selective_policy_fresh_process_replay_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
