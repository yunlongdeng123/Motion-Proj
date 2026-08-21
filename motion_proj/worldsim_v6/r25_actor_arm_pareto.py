"""WorldSim V6 R25 actor verifier arms 离散 Pareto 评估。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V6-R25-ACTOR-ARM-PARETO-EVALUATION-01"


class R25ExperimentError(RuntimeError):
    """R25 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R25ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R25ExperimentError("正式 R25 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R25ExperimentError("R25 task_id 漂移")
    source_run = _resolve_runs_uri(config["sources"]["r24_run"])
    source_files = {
        source_run / "MANIFEST.json": config["sources"]["r24_manifest_sha256"],
        source_run / "R24_GATE.json": config["sources"]["r24_gate_sha256"],
        source_run / "ACTOR_ARM_SUMMARIES.jsonl": config["sources"][
            "r24_actor_arm_summaries_sha256"
        ],
        source_run / "verifier_worker/PER_CASE_ARMS.jsonl": config["sources"][
            "r24_per_case_arms_sha256"
        ],
    }
    for path, expected in source_files.items():
        if _sha256(path) != expected:
            raise R25ExperimentError(f"冻结 R24 输入漂移：{path.name}")
    source_gate = json.loads((source_run / "R24_GATE.json").read_text(encoding="utf-8"))
    if source_gate["decision"] != "reject_or_pivot_verifier_arms":
        raise R25ExperimentError("R24 rejected decision 漂移")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R25ExperimentError("R25 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__actor-pareto-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        all_rows = _read_jsonl(source_run / "verifier_worker/PER_CASE_ARMS.jsonl")
        actor_rows = [row for row in all_rows if row["hole_type"] == "actor_removal_hole"]
        expected = int(config["cohort"]["expected_actor_case_count"])
        if len(actor_rows) != expected:
            raise R25ExperimentError("R25 actor denominator 漂移")
        gate_cfg = config["gate"]
        arm_results: list[dict[str, Any]] = []
        for arm_id in config["selected_arms"]:
            accepted = [row for row in actor_rows if row[arm_id]["decision"] == "ACCEPT"]
            false_safe_count = sum(bool(row[arm_id]["false_safe"]) for row in accepted)
            false_safe_rate = (
                0.0 if not accepted else float(false_safe_count / len(accepted))
            )
            p0_false_safe_count = sum(
                not bool(row[arm_id]["truth_safe"]) for row in actor_rows
            )
            reduction_count = int(p0_false_safe_count - false_safe_count)
            checks = {
                "minimum_accept_coverage": len(accepted) / expected
                >= float(gate_cfg["minimum_accept_coverage"]),
                "maximum_false_safe_rate": false_safe_rate
                <= float(gate_cfg["maximum_false_safe_rate"]),
                "minimum_false_safe_reduction_count": reduction_count
                >= int(gate_cfg["minimum_false_safe_reduction_count"]),
            }
            checks["passed"] = all(checks.values())
            arm_results.append(
                {
                    "schema_version": "worldsim_v6.r25_actor_arm.v1",
                    "arm": arm_id,
                    "denominator": expected,
                    "accept_count": len(accepted),
                    "accept_coverage": float(len(accepted) / expected),
                    "false_safe_count": false_safe_count,
                    "false_safe_rate": false_safe_rate,
                    "p0_false_safe_count": p0_false_safe_count,
                    "false_safe_reduction_count": reduction_count,
                    "checks": checks,
                    "eligible_for_actor_factorization": bool(checks["passed"]),
                }
            )
        _write_jsonl(run_dir / "ACTOR_ARM_PARETO.jsonl", arm_results)
        checks = {
            "source_r24_remains_rejected": not source_gate["checks"]["passed"],
            "actor_denominator_exact": len(actor_rows) == expected,
            "p1_p2_both_pareto_eligible": [
                row["arm"]
                for row in arm_results
                if row["eligible_for_actor_factorization"]
            ]
            == ["P1", "P2"],
            "proposal_and_verifier_outputs_immutable": all(
                _sha256(path) == expected_sha
                for path, expected_sha in source_files.items()
            ),
            "no_new_inference": True,
            "no_threshold_or_decision_change": True,
            "bake_not_started": True,
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R25_GATE.json",
            {
                "schema_version": "worldsim_v6.r25_gate.v1",
                "checks": checks,
                "eligible_actor_arms": [
                    row["arm"]
                    for row in arm_results
                    if row["eligible_for_actor_factorization"]
                ],
                "decision": "proceed_to_fresh_typed_semantic_evaluation"
                if checks["passed"]
                else "reject_actor_pareto_protocol",
            },
        )
        wall_seconds = time.monotonic() - started
        summary = {
            "schema_version": "worldsim_v6.r25_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_evaluation_protocol"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "source_r24_status": "rejected_retained",
            "new_inference_started": False,
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r25_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
            },
        )
        tracked = ["ACTOR_ARM_PARETO.jsonl", "R25_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json"]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r25_manifest.v1",
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
        default=Path("configs/worldsim_v6/r25_actor_arm_pareto_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0
