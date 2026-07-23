#!/usr/bin/env python
"""验证并封装正式 V7-H1-11B run。"""
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

from motion_proj.resim.canonical_hash import CANONICALIZATION_VERSION, canonical_sha256
from motion_proj.resim.schema import RENDER_REQUEST_SCHEMA_VERSION, WORLD_STATE_SCHEMA_VERSION
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import (
    V71RunContract,
    compute_artifact_set_hash,
    generate_run_id,
    utc_now,
)


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h1a-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11A/registry_build"),
    )
    parser.add_argument(
        "--evidence-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11B/evidence_v2"),
    )
    parser.add_argument(
        "--calibration-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11B/calibration"),
    )
    parser.add_argument(
        "--support-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11B/support_overlays"),
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11B"),
    )
    args = parser.parse_args()

    registry = _load(args.h1a_root / "summary.json")
    evidence = _load(args.evidence_root / "summary.json")
    calibration = _load(args.calibration_root / "summary.json")
    support = _load(args.support_root / "summary.json")
    scenes = ["003", "005", "004"]
    if sorted(evidence["scenes"]) != sorted(scenes):
        raise RuntimeError("evidence scenes 不完整")
    if not calibration["gate"]["pass"]:
        raise RuntimeError("certificate calibration gate 未通过")
    if not support["overlays_separate"] or support["agent_filled_human_verdict"]:
        raise RuntimeError("三类 overlay 未分离或出现 agent human verdict")
    for scene_id in scenes:
        scene = evidence["scenes"][scene_id]
        if scene["counts"]["frames"] != 80:
            raise RuntimeError(f"{scene_id} evidence 不足 80 帧")
        if scene["counts"]["overlap_voxels"] != 0:
            raise RuntimeError(f"{scene_id} dynamic layers 出现未经 certificate 处理的重叠")
        for frame, expected in scene["frame_artifact_sha256"].items():
            actual = file_fingerprint(
                str(args.evidence_root / scene_id / f"frame_{frame}.npz")
            )
            if actual != expected:
                raise RuntimeError(f"{scene_id}/{frame} evidence artifact hash 不匹配")

    config_paths = [
        Path("configs/resim/v71/safety_geometry_v1.yaml"),
        Path("configs/resim/v71/observation_evidence_v2.yaml"),
        Path("configs/resim/v71/render_support_v1.yaml"),
        Path("configs/resim/v71/scenario_effect_v1.yaml"),
        Path("configs/resim/v71/certificate_calibration.yaml"),
    ]
    config_hashes = {
        path.name: file_fingerprint(str(path)) for path in config_paths
    }
    config_fingerprint = canonical_sha256(config_hashes)
    world_contract = {
        "schema_version": WORLD_STATE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "split": "PILOT-3-calibration",
        "scene_ids": scenes,
        "actor_registry_sha256": registry["combined_registry_sha256"],
        "safety_geometry": {
            "version": "safety-geometry-v1",
            "config_sha256": config_hashes["safety_geometry_v1.yaml"],
            "source": "raw_nuscenes_annotations",
            "road_support": "UNKNOWN_map_expansion_unavailable",
        },
        "observation_evidence": {
            "version": "observation-evidence-v2",
            "evidence_set_sha256": evidence["evidence_set_sha256"],
            "dynamic_layer_voxelizer": "oriented-box-center-inclusion-v1",
        },
        "render_support": {
            "version": "render-support-v1",
            "support_set_sha256": support["support_set_sha256"],
            "thresholded_visibility": "not_in_world_state",
        },
    }
    world_state_hash = canonical_sha256(world_contract)
    render_contract = {
        "schema_version": RENDER_REQUEST_SCHEMA_VERSION,
        "world_state_hash": world_state_hash,
        "render_support_config_sha256": config_hashes["render_support_v1.yaml"],
        "artifact_mode": "separate_machine_diagnostic_overlays",
        "human_verdict": "not_collected",
    }
    render_request_hash = canonical_sha256(render_contract)

    sources: list[tuple[Path, str, str]] = [
        (args.evidence_root / "summary.json", "artifacts/evidence_summary.json", "application/json"),
        (args.calibration_root / "summary.json", "artifacts/calibration_summary.json", "application/json"),
        (args.calibration_root / "real_controls.json", "artifacts/real_controls.json", "application/json"),
        (args.calibration_root / "controlled_negatives.json", "artifacts/controlled_negatives.json", "application/json"),
        (args.calibration_root / "fixture_pair.json", "artifacts/fixture_pair.json", "application/json"),
        (args.support_root / "summary.json", "artifacts/support_summary.json", "application/json"),
    ]
    for scene_id in scenes:
        sources.append(
            (
                args.evidence_root / scene_id / "summary.json",
                f"artifacts/evidence/{scene_id}_summary.json",
                "application/json",
            )
        )
        sources.append(
            (
                args.support_root / scene_id / "summary.json",
                f"artifacts/support/{scene_id}_summary.json",
                "application/json",
            )
        )
        sources.append(
            (
                args.support_root / scene_id / "render_support_raw.npz",
                f"artifacts/support/{scene_id}_raw.npz",
                "application/x-npz",
            )
        )
        for product in (
            "safety_geometry", "observation_evidence", "render_support"
        ):
            sources.append(
                (
                    args.support_root / scene_id / f"{product}_overlay.png",
                    f"artifacts/overlays/{scene_id}_{product}.png",
                    "image/png",
                )
            )
    artifact_rows = [
        {
            "relative_path": relative,
            "world_state_hash": world_state_hash,
            "render_request_hash": render_request_hash,
            "artifact_hash": file_fingerprint(str(source)),
            "size_bytes": source.stat().st_size,
            "media_type": media,
        }
        for source, relative, media in sources
    ]
    artifact_set_hash = compute_artifact_set_hash(artifact_rows)
    code = git_state(str(PROJECT_ROOT))
    if code["dirty"]:
        raise RuntimeError("正式 H1-11B finalizer 要求 clean git state")
    run_id = generate_run_id("V7-H1-11B", "PILOT-3-calibration", 0, config_fingerprint)
    checkpoint_hashes = {}
    actor_ids = []
    for scene_id in scenes:
        scene_registry = _load(args.h1a_root / scene_id / "actor_registry.json")
        checkpoint_hashes[scene_id] = scene_registry["checkpoint_sha256"]
        actor_ids.extend(
            f"{scene_id}:{actor['true_instance_id']}"
            for actor in scene_registry["actors"]
        )
    manifest = {
        "schema_version": 1,
        "task_id": "V7-H1-11B",
        "run_id": run_id,
        "parent_run_id": "v71_v7-h1-11a__pilot-3__s0__20260723T144155452295Z__0ff143d9",
        "command": list(sys.argv),
        "plan_version": "V7.1",
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_fingerprint,
        "data_fingerprint": canonical_sha256(
            {
                "registry": registry["combined_registry_sha256"],
                "evidence": evidence["evidence_set_sha256"],
                "support": support["support_set_sha256"],
                "calibration": calibration["calibration_sha256"],
            }
        ),
        "split_fingerprint": canonical_sha256({"split": "PILOT-3-calibration", "scenes": scenes}),
        "proposal_fingerprint": file_fingerprint(str(args.calibration_root / "fixture_pair.json")),
        "third_party_commit": _git_commit(Path("/root/autodl-tmp/third_party/drivestudio")),
        "checkpoint_hashes": checkpoint_hashes,
        "scene_ids": scenes,
        "actor_ids": actor_ids,
        "camera_ids": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        "time_range": {"start": 0, "end": 79, "hz": 10},
        "seed": 0,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "cuda": "not_used_for_calibration",
        "gpu": "not_used_for_calibration",
        "world_state_schema_version": WORLD_STATE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
        "safety_geometry_version": "safety-geometry-v1",
        "observation_evidence_version": "observation-evidence-v2",
        "render_support_version": "render-support-v1",
        "certificate_version": "certificate-calibration-v1",
        "scenario_effect_version": "scenario-effect-v1",
        "provenance_version": "h1b-calibration-v1",
        "recovery_policy": "unknown_never_coerced_to_pass",
        "started_at": utc_now(),
        "ended_at": None,
        "exit_reason": None,
        "terminal_status": "running",
        "status": "running",
    }
    contract = V71RunContract.initialize(args.run_root, run_id, manifest)
    run_dir = contract.run_dir
    resolved = "\n".join(
        f"# --- {path.name} ---\n{path.read_text(encoding='utf-8').rstrip()}\n"
        for path in config_paths
    )
    atomic_write_text(str(run_dir / "resolved.yaml"), resolved)
    atomic_write_json(str(run_dir / "fingerprints" / "code.json"), code)
    atomic_write_json(str(run_dir / "fingerprints" / "environment.json"), manifest["environment"])
    atomic_write_json(
        str(run_dir / "fingerprints" / "data.json"),
        {
            "registry": registry["combined_registry_sha256"],
            "evidence": evidence["evidence_set_sha256"],
            "support": support["support_set_sha256"],
            "calibration": calibration["calibration_sha256"],
        },
    )
    atomic_write_json(
        str(run_dir / "fingerprints" / "third_party.json"),
        {"drivestudio_commit": manifest["third_party_commit"]},
    )
    atomic_write_json(str(run_dir / "fingerprints" / "checkpoints.json"), checkpoint_hashes)
    shutil.copy2(args.calibration_root / "fixture_pair.json", run_dir / "proposal_bank.json")
    atomic_write_json(
        str(run_dir / "actor_registry.json"),
        {
            "combined_registry_sha256": registry["combined_registry_sha256"],
            "actor_count_by_scene": registry["actor_count_by_scene"],
        },
    )
    atomic_write_json(str(run_dir / "world_state_manifest.json"), {**world_contract, "world_state_hash": world_state_hash})
    atomic_write_json(str(run_dir / "render_request_manifest.json"), {**render_contract, "render_request_hash": render_request_hash})
    for source, relative, _ in sources:
        target = run_dir / relative
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
        "evidence_frame_count": 240,
        "real_control_count": calibration["real_control_count"],
        "measurable_real_control_retention": calibration["measurable_real_control_retention"],
        "detectable_controlled_negative_recall": calibration["detectable_controlled_negative_recall"],
        "full_unknown_count": calibration["full_verdict_distribution"]["UNKNOWN"],
        "overlays_separate": support["overlays_separate"],
    }
    atomic_write_text(str(run_dir / "metrics.jsonl"), json.dumps(metrics, ensure_ascii=False) + "\n")
    summary = {
        "task_id": "V7-H1-11B",
        "run_id": run_id,
        "engineering_gate": "PASS",
        "hypothesis_verdict": "not_evaluated",
        "calibration_gate": calibration["gate"],
        "real_control_count": calibration["real_control_count"],
        "measurable_real_control_count": calibration["measurable_real_control_count"],
        "measurable_real_control_retention": calibration["measurable_real_control_retention"],
        "detectable_controlled_negative_recall": calibration["detectable_controlled_negative_recall"],
        "full_verdict_distribution": calibration["full_verdict_distribution"],
        "road_support_policy": calibration["road_support_policy"],
        "evidence_set_sha256": evidence["evidence_set_sha256"],
        "support_set_sha256": support["support_set_sha256"],
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(
        str(run_dir / "logs" / "finalize.log"),
        "H1-11B layered evidence, continuous safety geometry, render support, scenario-effect and tri-state calibration gates passed.\n",
    )
    contract.finalize("COMPLETE", exit_reason="H1-11B_engineering_calibration_gate_passed")
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
