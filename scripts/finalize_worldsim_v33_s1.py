#!/usr/bin/env python
"""对 S1 run 做 fail-closed 验收、源码快照与 manifest 固化。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file
from motion_proj.worldsim_v33.instance_field import load_instance_field


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "done" or summary.get("phase") != args.phase:
        raise RuntimeError("S1 summary terminal/phase 不合法")
    if summary["config_sha256"] != sha256_file(args.config):
        raise RuntimeError("S1 summary config SHA 漂移")
    expected_checkpoint = config["inputs"]["checkpoint_sha256"]
    if not (
        summary["checkpoint_sha256_before"]
        == summary["checkpoint_sha256_after"]
        == expected_checkpoint
        == sha256_file(config["inputs"]["checkpoint"])
    ):
        raise RuntimeError("S1 RGB checkpoint bitwise 合同失败")
    if not summary.get("rgb_checkpoint_bitwise_exact"):
        raise RuntimeError("S1 未声明 RGB bitwise exact")
    if summary.get("base_requires_grad") is not False:
        raise RuntimeError("S1 base_requires_grad 必须为 false")
    if not summary.get("instance_field_independent") or summary.get("heldout_leaks") != 0:
        raise RuntimeError("S1 sidecar/split 合同失败")
    expected_eval = set(
        int(value)
        for value in config["split"][
            "development_frames" if args.phase == "smoke" else "heldout_frames"
        ]
    )
    if not set(int(value) for value in summary["evaluation_frames"]) <= expected_eval:
        raise RuntimeError("S1 evaluation frame 越权")
    if set(summary["optimization_frames"]) & set(summary["evaluation_frames"]):
        raise RuntimeError("S1 optimization/evaluation frame 泄漏")
    evaluation_source = summary["evaluation_source"]
    if sha256_file(evaluation_source["manifest"]) != evaluation_source["manifest_sha256"]:
        raise RuntimeError("S1 evaluation source manifest SHA 漂移")
    if args.phase == "formal":
        target_manifest = json.loads(
            Path(evaluation_source["manifest"]).read_text(encoding="utf-8")
        )
        if not evaluation_source.get("optimization_forbidden") or not target_manifest.get(
            "optimization_forbidden"
        ):
            raise RuntimeError("formal evaluation source 未声明 optimization_forbidden")
        if target_manifest.get("config_sha256") != summary["config_sha256"]:
            raise RuntimeError("formal evaluation target/config SHA 漂移")
    for arm_name, arm in summary["arms"].items():
        field_path = Path(arm["instance_field"])
        if sha256_file(field_path) != arm["instance_field_sha256"]:
            raise RuntimeError(f"{arm_name} instance field SHA 漂移")
        if field_path.stat().st_size != int(arm["instance_field_bytes"]):
            raise RuntimeError(f"{arm_name} instance field bytes 漂移")
        load_instance_field(field_path)
        metric = arm["evaluation"]["aggregate"]
        required = {
            "iou",
            "boundary_f1",
            "normalized_boundary_distance",
            "false_positive_semantic_mass",
            "false_negative_semantic_mass",
            "identity_present",
        }
        if required - set(metric):
            raise RuntimeError(f"{arm_name} metric 缺字段")
        if arm["evaluation"]["identity_parameter_stability"] != 1.0:
            raise RuntimeError(f"{arm_name} global identity parameter 不稳定")
    if args.phase == "formal":
        frozen = str(config["optimization"]["formal_selected_arm"])
        expected_arms = {"O0_heuristic"} | (
            set() if frozen == "O0_heuristic" else {frozen}
        )
        if set(summary["arms"]) != expected_arms:
            raise RuntimeError("formal arm 集不等于 O0 + frozen O*")

    selected = str(summary["recommended_arm"])
    acceptance = {
        "schema_version": "worldsim_v33_s1_acceptance_v1",
        "task_id": config["task_id"],
        "status": "done",
        "phase": args.phase,
        "selected_arm": selected,
        "fallback_to_v32_heuristic": selected == "O0_heuristic",
        "gates": {
            "rgb_checkpoint_bitwise_exact": True,
            "base_requires_grad_false": True,
            "instance_field_independent": True,
            "identity_contract_fail_closed": True,
            "optimization_evaluation_disjoint": True,
            "metrics_complete": True,
            "source_hashes_exact": True,
        },
    }
    acceptance_path = args.run_dir / "acceptance.json"
    atomic_json(acceptance_path, acceptance)

    snapshot_dir = args.run_dir / "artifacts/source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    snapshot_sources = [
        args.config,
        PROJECT / "motion_proj/worldsim_v33/instance_field.py",
        PROJECT / "motion_proj/worldsim_v33/instance_renderer.py",
        PROJECT / "motion_proj/worldsim_v33/semantic_reassignment.py",
        PROJECT / "scripts/run_worldsim_v33_s1_instance_field.py",
        PROJECT / "scripts/finalize_worldsim_v33_s1.py",
        PROJECT / "tests/test_worldsim_v33_instance_field.py",
        PROJECT / "tests/test_worldsim_v33_instance_render.py",
        PROJECT / "tests/test_worldsim_v33_no_rgb_mutation.py",
    ]
    for source in snapshot_sources:
        shutil.copy2(source, snapshot_dir / source.name)
    entries = []
    for path in sorted(args.run_dir.rglob("*")):
        if not path.is_file() or path.name in {"run_manifest.json", "status.json"}:
            continue
        entries.append(
            {
                "path": str(path.relative_to(args.run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "worldsim_v33_s1_run_manifest_v1",
        "task_id": config["task_id"],
        "phase": args.phase,
        "file_count": len(entries),
        "files": entries,
    }
    atomic_json(args.run_dir / "run_manifest.json", manifest)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
