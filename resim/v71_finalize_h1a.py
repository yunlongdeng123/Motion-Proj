#!/usr/bin/env python
"""验证并封装正式 V7-H1-11A run。"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.actor_registry import validate_registry_hash
from motion_proj.resim.canonical_hash import (
    CANONICALIZATION_VERSION,
    canonical_sha256,
)
from motion_proj.resim.schema import (
    RENDER_REQUEST_SCHEMA_VERSION,
    WORLD_STATE_SCHEMA_VERSION,
)
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import (
    V71RunContract,
    compute_artifact_set_hash,
    generate_run_id,
    utc_now,
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是 object: {path}")
    return value


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/v71/world_state_v1.yaml"),
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11A/registry_build"
        ),
    )
    parser.add_argument(
        "--reload-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11A/registry_reload_build"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11A"),
    )
    args = parser.parse_args()

    first_summary = _load(args.registry_root / "summary.json")
    reload_summary = _load(args.reload_root / "summary.json")
    if (
        first_summary["registry_hash_by_scene"]
        != reload_summary["registry_hash_by_scene"]
    ):
        raise RuntimeError("跨进程 registry reload hash 不稳定")
    if not first_summary["minimum_two_actors_per_scene"]:
        raise RuntimeError("PILOT-3 存在少于 2 个 RigidNodes actor 的场景")
    coordinate_audit = _load(args.registry_root / "coordinate_audit.json")
    if not coordinate_audit["coordinate_roundtrip_gate"]:
        raise RuntimeError("coordinate round-trip gate 未通过")

    scene_ids = sorted(first_summary["registry_hash_by_scene"])
    registries = {}
    actor_ids = []
    checkpoint_hashes = {}
    for scene_id in scene_ids:
        registry = _load(args.registry_root / scene_id / "actor_registry.json")
        validate_registry_hash(registry)
        if (
            registry["actor_registry_sha256"]
            != first_summary["registry_hash_by_scene"][scene_id]
        ):
            raise RuntimeError(f"scene {scene_id} registry summary hash 不匹配")
        registries[scene_id] = registry
        actor_ids.extend(
            f"{scene_id}:{actor['true_instance_id']}" for actor in registry["actors"]
        )
        checkpoint_hashes[scene_id] = registry["checkpoint_sha256"]

    world_state_contract = {
        "schema_version": WORLD_STATE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "split": "PILOT-3",
        "scene_ids": scene_ids,
        "actor_registry_hashes": first_summary["registry_hash_by_scene"],
        "coordinate_frames": {
            "annotation": "world",
            "model": "start_CAM_FRONT_sensor_frame",
            "occupancy_grid": "per_frame_LiDAR_sensor_frame",
            "transform_naming": "T_dst_src",
        },
        "checkpoint_hashes": checkpoint_hashes,
    }
    world_state_hash = canonical_sha256(world_state_contract)
    render_contract = {
        "schema_version": RENDER_REQUEST_SCHEMA_VERSION,
        "world_state_hash": world_state_hash,
        "rendering": "not_applicable_H1-11A",
    }
    render_request_hash = canonical_sha256(render_contract)

    sources = []
    for scene_id in scene_ids:
        sources.append(
            (
                args.registry_root / scene_id / "actor_registry.json",
                f"artifacts/registries/{scene_id}.json",
                "application/json",
            )
        )
    sources.extend(
        [
            (
                args.registry_root / "summary.json",
                "artifacts/registry_build_summary.json",
                "application/json",
            ),
            (
                args.reload_root / "summary.json",
                "artifacts/registry_reload_summary.json",
                "application/json",
            ),
            (
                args.registry_root / "coordinate_audit.json",
                "artifacts/coordinate_audit.json",
                "application/json",
            ),
        ]
    )
    artifact_rows = [
        {
            "relative_path": relative_path,
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
            "artifact_hash": file_fingerprint(str(source)),
            "size_bytes": source.stat().st_size,
            "media_type": media_type,
        }
        for source, relative_path, media_type in sources
    ]
    artifact_set_hash = compute_artifact_set_hash(artifact_rows)
    config_fingerprint = file_fingerprint(str(args.config))
    run_id = generate_run_id("V7-H1-11A", "PILOT-3", 0, config_fingerprint)
    code = git_state(str(PROJECT_ROOT))
    manifest = {
        "schema_version": 1,
        "task_id": "V7-H1-11A",
        "run_id": run_id,
        "parent_run_id": None,
        "command": list(sys.argv),
        "plan_version": "V7.1",
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_fingerprint,
        "data_fingerprint": first_summary["combined_registry_sha256"],
        "split_fingerprint": canonical_sha256({"split": "PILOT-3", "scenes": scene_ids}),
        "proposal_fingerprint": canonical_sha256({"proposals": []}),
        "third_party_commit": _git_commit(
            Path("/root/autodl-tmp/third_party/drivestudio")
        ),
        "checkpoint_hashes": checkpoint_hashes,
        "scene_ids": scene_ids,
        "actor_ids": actor_ids,
        "camera_ids": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        "time_range": {"start": 0, "end": 79, "hz": 10},
        "seed": 0,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cuda": "not_used",
        "gpu": "not_used",
        "world_state_schema_version": WORLD_STATE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
        "safety_geometry_version": "not_implemented_H1-11A",
        "observation_evidence_version": "O0_retrospective_audited",
        "render_support_version": "not_implemented_H1-11A",
        "certificate_version": "not_implemented_H1-11A",
        "scenario_effect_version": "not_implemented_H1-11A",
        "provenance_version": "actor_registry_v1",
        "recovery_policy": "not_applicable_H1-11A",
        "started_at": utc_now(),
        "ended_at": None,
        "exit_reason": None,
        "terminal_status": "running",
        "status": "running",
    }
    contract = V71RunContract.initialize(args.run_root, run_id, manifest)
    run_dir = contract.run_dir
    atomic_write_text(str(run_dir / "resolved.yaml"), args.config.read_text(encoding="utf-8"))
    atomic_write_json(str(run_dir / "fingerprints" / "code.json"), code)
    atomic_write_json(
        str(run_dir / "fingerprints" / "environment.json"), manifest["environment"]
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "data.json"),
        {
            "combined_registry_sha256": first_summary["combined_registry_sha256"],
            "registry_hash_by_scene": first_summary["registry_hash_by_scene"],
        },
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "third_party.json"),
        {"drivestudio_commit": manifest["third_party_commit"]},
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "checkpoints.json"),
        checkpoint_hashes,
    )
    atomic_write_json(
        str(run_dir / "proposal_bank.json"),
        {"status": "not_applicable_H1-11A", "proposals": []},
    )
    atomic_write_json(
        str(run_dir / "actor_registry.json"),
        {
            "schema_version": 1,
            "registry_hash_by_scene": first_summary["registry_hash_by_scene"],
            "actor_count_by_scene": first_summary["actor_count_by_scene"],
            "combined_registry_sha256": first_summary["combined_registry_sha256"],
        },
    )
    atomic_write_json(str(run_dir / "world_state_manifest.json"), {
        **world_state_contract,
        "world_state_hash": world_state_hash,
    })
    atomic_write_json(str(run_dir / "render_request_manifest.json"), {
        **render_contract,
        "render_request_hash": render_request_hash,
    })
    for source, relative_path, _ in sources:
        target = run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
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
    metrics = {
        "time": utc_now(),
        "step": 0,
        "scene_count": len(scene_ids),
        "actor_count": len(actor_ids),
        "minimum_two_actors_per_scene": True,
        "registry_reload_stable": True,
        "coordinate_roundtrip_gate": True,
    }
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(metrics, ensure_ascii=False) + "\n",
    )
    summary = {
        "task_id": "V7-H1-11A",
        "run_id": run_id,
        "engineering_gate": "PASS",
        "hypothesis_verdict": "not_evaluated",
        "actor_count_by_scene": first_summary["actor_count_by_scene"],
        "registry_reload_stable": True,
        "coordinate_roundtrip_gate": True,
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(
        str(run_dir / "logs" / "finalize.log"),
        "H1-11A registry, canonical hash and coordinate gates passed.\n",
    )
    contract.finalize("COMPLETE", exit_reason="H1-11A_engineering_gate_passed")
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), **summary}))


if __name__ == "__main__":
    main()
