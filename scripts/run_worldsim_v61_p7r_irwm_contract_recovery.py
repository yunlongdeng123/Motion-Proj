#!/usr/bin/env python3
"""WorldSim V6.1 P7R：复用不可变 H001 forward 的窄形式合同恢复。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V61-P7R-IRWM-CONTRACT-RECOVERY-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class P7RContractError(RuntimeError):
    """H001 artifact 或窄恢复合同不满足。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def resolve_runs_uri(uri: str) -> Path:
    if not uri.startswith("runs://"):
        raise P7RContractError("只接受 runs URI")
    relative = Path(uri[len("runs://") :])
    if ".." in relative.parts:
        raise P7RContractError("runs URI 不得包含上级路径")
    return (RUNS_ROOT / relative).resolve()


def frozen_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_h001_files(root: Path, expected: Mapping[str, str]) -> list[dict[str, Any]]:
    records = []
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise P7RContractError(f"H001 authority 漂移: {path}")
        records.append(frozen_record(path))
    return records


def recovery_checks(
    config: Mapping[str, Any], gate: Mapping[str, Any], report: Mapping[str, Any],
    summary: Mapping[str, Any], terminal: Mapping[str, Any], source_text: str,
) -> dict[str, bool]:
    contract = config["contract"]
    false_checks = sorted(name for name, passed in gate["checks"].items() if not passed)
    unchanged_checks = [
        passed for name, passed in gate["checks"].items()
        if name not in set(contract["required_h001_false_checks"])
    ]
    expected_environment = contract["expected_environment"]
    actual_environment = {
        "python": report["python"],
        "torch": report["torch"],
        **report["environment_versions"],
    }
    return {
        "h001_rejected_terminal_exact": bool(
            not gate["passed"]
            and summary["status"] == "rejected"
            and terminal["status"] == "rejected"
            and terminal["canonical"]
        ),
        "h001_only_two_known_checks_failed": bool(
            false_checks == sorted(contract["required_h001_false_checks"])
            and all(unchanged_checks)
        ),
        "cuda_wheel_build_version_exact": actual_environment == expected_environment,
        "official_deleted_missing_keys_exact": bool(
            sorted(report["model_load_missing_keys"])
            == sorted(contract["allowed_missing_keys"])
            and report["model_load_unexpected_keys"] == contract["required_unexpected_keys"]
            and contract["required_source_statement"] in source_text
        ),
        "h001_source_commit_exact": summary["source_commit"]
        == config["sources"]["h001_source_commit"],
        "output_contract_unchanged": bool(
            report["finite_logits"]
            and report["raw_logits_shape"] == contract["raw_logits_shape"]
            and report["class_label_shape"] == contract["class_grid_shape"]
            and report["occupied_voxel_count"] >= contract["minimum_occupied_voxels"]
            and report["free_voxel_count"] >= contract["minimum_free_voxels"]
        ),
        "truth_free_current_state_contract_unchanged": bool(
            report["current_state_extraction"]
            == "official_scene_encoder_plus_official_final_decoder_occupancy_head"
            and not report["future_decoder_started"]
            and not report["planning_head_started"]
            and not report["occupancy_ground_truth_read"]
            and not report["o_method_or_o_eval_read"]
            and not report["confirmation_content_read"]
            and not report["training_started"]
        ),
    }


def run(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if git(repo_root, "status", "--porcelain"):
        raise P7RContractError("正式 P7R 要求 motion_proj 工作树干净")
    source_commit = git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise P7RContractError("P7R task_id 漂移")

    h001_root = resolve_runs_uri(config["sources"]["h001_run"])
    frozen_files = verify_h001_files(h001_root, config["sources"]["h001_files"])
    source_cfg = config["sources"]["official_deleted_parameter_source"]
    official_source = Path(source_cfg["path"])
    source_record = frozen_record(official_source)
    if source_record["bytes"] != source_cfg["bytes"] or source_record["sha256"] != source_cfg["sha256"]:
        raise P7RContractError("官方 deleted-parameter source 漂移")
    frozen_files.append(source_record)

    gate = json.loads((h001_root / "P7_GATE.json").read_text(encoding="utf-8"))
    report = json.loads((h001_root / "WORKER_REPORT.json").read_text(encoding="utf-8"))
    summary = json.loads((h001_root / "SUMMARY.json").read_text(encoding="utf-8"))
    terminal = json.loads((h001_root / "TERMINAL.json").read_text(encoding="utf-8"))
    checks = recovery_checks(
        config, gate, report, summary, terminal, official_source.read_text(encoding="utf-8")
    )
    disk_free_gib = shutil.disk_usage(run_root).free / 1024**3
    checks["disk_free_within_budget"] = disk_free_gib >= config["resources"]["minimum_disk_free_gib"]
    passed = all(checks.values())

    run_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__irwm-contract-recovery-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    recovery = {
        "schema_version": "worldsim_v61.p7r_irwm_capability_recovery.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "source_commit": source_commit,
        "h001_run": config["sources"]["h001_run"],
        "h001_output_sha256": report["output_sha256"],
        "allowed_missing_keys": config["contract"]["allowed_missing_keys"],
        "environment_versions": {"python": report["python"], "torch": report["torch"], **report["environment_versions"]},
        "raw_logits_shape": report["raw_logits_shape"],
        "class_label_shape": report["class_label_shape"],
        "occupied_voxel_count": report["occupied_voxel_count"],
        "free_voxel_count": report["free_voxel_count"],
        "inference_seconds": report["inference_seconds"],
        "peak_gpu_memory_gib": report["peak_gpu_memory_gib"],
        "new_gpu_forward_started": False,
        "training_started": False,
        "confirmation_content_read": False,
    }
    gate_out = {
        "schema_version": "worldsim_v61.p7r_irwm_gate.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "checks": checks,
        "passed": passed,
    }
    summary_out = {
        "schema_version": "worldsim_v61.p7r_irwm_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "source_commit": source_commit,
        "status": "accepted" if passed else "rejected",
        "hypothesis_outcome": "accepted_irwm_current_occupancy_3090_capability" if passed else "rejected_irwm_contract_recovery",
        "next": "pre_register_single_me3_irwm_scientific_recovery" if passed else "stop_learned_occupancy_and_close_v61_negative",
        "claim_boundary": config["claim_boundary"],
        "failure_ledger_delta": "none" if passed else "required",
    }
    resource = {
        "schema_version": "worldsim_v61.p7r_irwm_resource_audit.v1",
        "disk_free_gib_at_start": disk_free_gib,
        "wall_seconds": time.monotonic() - started,
        "gpu_used": False,
        "model_loaded": False,
        "training_started": False,
        "confirmation_content_read": False,
    }
    terminal_out = {
        "schema_version": "worldsim_v61.p7r_irwm_terminal.v1",
        "task_id": TASK_ID,
        "status": summary_out["status"],
        "canonical": True,
        "run_uri": "run://" + str(run_dir.relative_to(RUNS_ROOT)),
    }
    for name, value in (
        ("CAPABILITY_RECOVERY.json", recovery),
        ("P7R_GATE.json", gate_out),
        ("SUMMARY.json", summary_out),
        ("RESOURCE_AUDIT.json", resource),
        ("TERMINAL.json", terminal_out),
    ):
        write_json(run_dir / name, value)
    generated = [frozen_record(run_dir / name) for name in (
        "CAPABILITY_RECOVERY.json", "P7R_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "TERMINAL.json"
    )]
    write_json(run_dir / "MANIFEST.json", {
        "schema_version": "worldsim_v61.p7r_irwm_manifest.v1",
        "task_id": TASK_ID,
        "source_commit": source_commit,
        "frozen_authorities": frozen_files,
        "generated_artifacts": generated,
    })
    print(run_dir)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=RUNS_ROOT / "worldsim_v61")
    args = parser.parse_args()
    run(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

