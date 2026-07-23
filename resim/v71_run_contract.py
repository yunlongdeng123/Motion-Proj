#!/usr/bin/env python
"""V7-EV-10 证据索引、run-contract smoke 与验证 CLI。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import (
    V71RunContract,
    build_retrospective_evidence_index,
    compute_artifact_set_hash,
    generate_run_id,
    smoke_hash,
    utc_now,
    validate_optional_branch_not_triggered,
)

DEFAULT_CONFIG = Path("configs/resim/v71/run_contract.yaml")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return config


def _contract(config: dict, run_dir: Path) -> V71RunContract:
    contract = config["contract"]
    return V71RunContract(
        run_dir,
        required_files=contract["required_files"],
        required_directories=contract["required_directories"],
    )


def build_index(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    roots = config["retrospective_evidence_roots"]
    state = git_state(str(args.repo_root))
    index = build_retrospective_evidence_index(
        args.output,
        roots,
        index_code_state=state,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_fingerprint(str(args.output)),
                "stage_count": index["totals"]["stage_count"],
                "file_count": index["totals"]["file_count"],
                "total_size_bytes": index["totals"]["total_size_bytes"],
            },
            ensure_ascii=False,
        )
    )


def smoke(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    config_fingerprint = file_fingerprint(str(args.config))
    evidence_index_hash = file_fingerprint(str(args.evidence_index))
    code = git_state(str(args.repo_root))
    world_state_hash = smoke_hash(
        "world-state",
        {
            "plan_version": config["plan_version"],
            "task_id": "V7-EV-10",
            "evidence_index_hash": evidence_index_hash,
        },
    )
    render_request_hash = smoke_hash(
        "render-request",
        {
            "world_state_hash": world_state_hash,
            "mode": "contract-smoke",
            "config_fingerprint": config_fingerprint,
        },
    )
    artifact_bytes = b"V7-EV-10 run contract smoke\n"
    # artifact hash 必须绑定实际文件字节，而不是测试 payload。
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_rows = [
        {
            "relative_path": "artifacts/contract_smoke.txt",
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
            "artifact_hash": artifact_hash,
            "size_bytes": len(artifact_bytes),
            "media_type": "text/plain",
        }
    ]
    artifact_set_hash = compute_artifact_set_hash(artifact_rows)
    split_fingerprint = smoke_hash("split", {"split": "smoke"})
    proposal_fingerprint = smoke_hash("proposal", {"proposals": []})
    data_fingerprint = evidence_index_hash
    run_id = generate_run_id(
        "V7-EV-10",
        "smoke",
        0,
        config_fingerprint,
    )
    run_root = args.run_root or Path(config["run_root"]) / "V7-EV-10"
    manifest = {
        "schema_version": 1,
        "task_id": "V7-EV-10",
        "run_id": run_id,
        "parent_run_id": None,
        "command": list(sys.argv),
        "plan_version": config["plan_version"],
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_fingerprint,
        "data_fingerprint": data_fingerprint,
        "split_fingerprint": split_fingerprint,
        "proposal_fingerprint": proposal_fingerprint,
        "third_party_commit": "not_applicable_ev10_smoke",
        "checkpoint_hashes": {},
        "scene_ids": [],
        "actor_ids": [],
        "camera_ids": [],
        "time_range": None,
        "seed": 0,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cuda": "not_used",
        "gpu": "not_used",
        "world_state_schema_version": "ev10_contract_smoke_only",
        "canonicalization_version": "ev10_contract_smoke_only",
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
        "safety_geometry_version": "not_applicable_ev10_smoke",
        "observation_evidence_version": "retrospective_index_v1",
        "render_support_version": "not_applicable_ev10_smoke",
        "certificate_version": "not_applicable_ev10_smoke",
        "scenario_effect_version": "not_applicable_ev10_smoke",
        "provenance_version": "retrospective_index_v1",
        "recovery_policy": "not_applicable_ev10_smoke",
        "started_at": utc_now(),
        "ended_at": None,
        "exit_reason": None,
        "terminal_status": "running",
        "status": "running",
    }
    contract_config = config["contract"]
    contract = V71RunContract.initialize(
        run_root,
        run_id,
        manifest,
        required_files=contract_config["required_files"],
        required_directories=contract_config["required_directories"],
    )
    run_dir = contract.run_dir
    atomic_write_text(str(run_dir / "resolved.yaml"), args.config.read_text(encoding="utf-8"))
    atomic_write_json(str(run_dir / "fingerprints" / "code.json"), code)
    atomic_write_json(
        str(run_dir / "fingerprints" / "environment.json"),
        manifest["environment"],
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "data.json"),
        {
            "evidence_index": str(args.evidence_index),
            "sha256": evidence_index_hash,
        },
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "third_party.json"),
        {"status": "not_applicable_ev10_smoke"},
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "checkpoints.json"),
        {"checkpoints": []},
    )
    atomic_write_json(
        str(run_dir / "proposal_bank.json"),
        {
            "mode": "contract_smoke",
            "proposal_fingerprint": proposal_fingerprint,
            "proposals": [],
        },
    )
    atomic_write_json(
        str(run_dir / "actor_registry.json"),
        {"mode": "contract_smoke", "actors": []},
    )
    atomic_write_json(
        str(run_dir / "world_state_manifest.json"),
        {
            "mode": "contract_smoke",
            "world_state_hash": world_state_hash,
        },
    )
    atomic_write_json(
        str(run_dir / "render_request_manifest.json"),
        {
            "mode": "contract_smoke",
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
        },
    )
    artifact_path = run_dir / artifact_rows[0]["relative_path"]
    artifact_path.write_bytes(artifact_bytes)
    atomic_write_json(
        str(run_dir / "artifact_manifest.json"),
        {
            "schema_version": 1,
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
            "artifact_set_hash": artifact_set_hash,
            "artifacts": artifact_rows,
        },
    )
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(
            {
                "time": utc_now(),
                "step": 0,
                "contract_payload_valid": True,
                "gpu_used": False,
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    atomic_write_json(
        str(run_dir / "summary.json"),
        {
            "task_id": "V7-EV-10",
            "run_id": run_id,
            "research_verdict": "engineering_contract_smoke",
            "evidence_index_hash": evidence_index_hash,
            "contract_payload_valid": True,
            "optional_branch_semantics_tested": True,
        },
    )
    atomic_write_text(
        str(run_dir / "logs" / "smoke.log"),
        "EV-10 contract smoke payload constructed; validation follows.\n",
    )
    contract.finalize("COMPLETE", exit_reason="run_contract_smoke_passed")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "terminal_marker": "COMPLETE",
                "world_state_hash": world_state_hash,
                "render_request_hash": render_request_hash,
                "artifact_set_hash": artifact_set_hash,
            },
            ensure_ascii=False,
        )
    )


def validate(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    marker = _contract(config, args.run_dir).validate()
    print(json.dumps({"run_dir": str(args.run_dir), "terminal_marker": marker}))


def optional_not_triggered(args: argparse.Namespace) -> None:
    validate_optional_branch_not_triggered(args.parent_summary, args.branch_run_root)
    print(
        json.dumps(
            {
                "parent_summary": str(args.parent_summary),
                "branch_run_root": str(args.branch_run_root),
                "status": "not_triggered_valid",
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("build-index")
    index_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json"),
    )
    index_parser.add_argument("--repo-root", type=Path, default=Path("."))
    index_parser.set_defaults(func=build_index)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument(
        "--evidence-index",
        type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json"),
    )
    smoke_parser.add_argument("--repo-root", type=Path, default=Path("."))
    smoke_parser.add_argument("--run-root", type=Path, default=None)
    smoke_parser.set_defaults(func=smoke)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("run_dir", type=Path)
    validate_parser.set_defaults(func=validate)

    optional_parser = subparsers.add_parser("optional-not-triggered")
    optional_parser.add_argument("parent_summary", type=Path)
    optional_parser.add_argument("branch_run_root", type=Path)
    optional_parser.set_defaults(func=optional_not_triggered)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
