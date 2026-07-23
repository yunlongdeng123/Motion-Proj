#!/usr/bin/env python
"""验证并封装正式 V7-H1-11C run。"""
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


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/resim/v71/render_and_label_v1.yaml"),
    )
    parser.add_argument(
        "--world-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/world_states"),
    )
    parser.add_argument(
        "--render-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/render_labels"),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/label_sync_audit.json"),
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C"),
    )
    args = parser.parse_args()
    world = _load(args.world_root / "summary.json")
    render = _load(args.render_root / "summary.json")
    audit = _load(args.audit)
    if not audit["checks"]["pass"] or not render["label_sync_all_pass"]:
        raise RuntimeError("H1-11C label-sync gate 未通过")
    if render["sample_count"] != 18:
        raise RuntimeError("PILOT-3 × V0/V1 × 3 cameras 必须为 18 samples")
    config_fingerprint = file_fingerprint(str(args.config))
    code = git_state(str(PROJECT_ROOT))
    if code["dirty"]:
        raise RuntimeError("正式 H1-11C finalizer 要求 clean git state")

    full_index = []
    for path in sorted(value for value in args.render_root.rglob("*") if value.is_file()):
        full_index.append(
            {
                "relative_path": path.relative_to(args.render_root).as_posix(),
                "sha256": file_fingerprint(str(path)),
                "size_bytes": path.stat().st_size,
            }
        )
    full_index_payload = {
        "schema_version": 1,
        "root": str(args.render_root),
        "file_count": len(full_index),
        "total_size_bytes": sum(value["size_bytes"] for value in full_index),
        "files": full_index,
    }
    full_index_payload["index_sha256"] = canonical_sha256(full_index)
    index_path = args.run_root / "H1C_FULL_ARTIFACT_INDEX.json"
    atomic_write_json(str(index_path), full_index_payload)

    request_hashes = {
        key: value["render_request_hash"] for key, value in render["samples"].items()
    }
    world_state_hash = world["world_state_set_sha256"]
    render_request_hash = canonical_sha256(request_hashes)
    sources = [
        (args.world_root / "summary.json", "artifacts/world_state_summary.json", "application/json"),
        (args.render_root / "summary.json", "artifacts/render_label_summary.json", "application/json"),
        (args.audit, "artifacts/label_sync_audit.json", "application/json"),
        (index_path, "artifacts/full_artifact_index.json", "application/json"),
    ]
    for scene_id, variant, camera in (
        ("003", "V1", "CAM_FRONT"),
        ("005", "V1", "CAM_FRONT"),
        ("004", "V1", "CAM_FRONT"),
    ):
        base = args.render_root / scene_id / variant / camera
        sources.extend(
            [
                (base / "rgb.png", f"artifacts/samples/{scene_id}_rgb.png", "image/png"),
                (base / "boxes.json", f"artifacts/samples/{scene_id}_boxes.json", "application/json"),
            ]
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
    run_id = generate_run_id("V7-H1-11C", "PILOT-3-interface", 0, config_fingerprint)
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    registry_root = Path(config["registry_root"])
    checkpoint_hashes, actor_ids = {}, []
    for scene_id in ("003", "005", "004"):
        registry = _load(registry_root / scene_id / "actor_registry.json")
        checkpoint_hashes[scene_id] = registry["checkpoint_sha256"]
        actor_ids.extend(
            f"{scene_id}:{value['true_instance_id']}" for value in registry["actors"]
        )
    manifest = {
        "schema_version": 1,
        "task_id": "V7-H1-11C",
        "run_id": run_id,
        "parent_run_id": "v71_v7-h1-11b__pilot-3-calibration__s0__20260723T145956893820Z__b8349bc0",
        "command": list(sys.argv),
        "plan_version": "V7.1",
        "code_commit": code["commit"],
        "code_dirty": False,
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_fingerprint,
        "data_fingerprint": canonical_sha256(
            {
                "world": world_state_hash,
                "render": render["render_label_set_sha256"],
                "audit": audit["audit_sha256"],
                "full_artifacts": full_index_payload["index_sha256"],
            }
        ),
        "split_fingerprint": canonical_sha256({"split": "PILOT-3-interface", "scenes": ["003", "005", "004"]}),
        "proposal_fingerprint": canonical_sha256({"variants": ["V0", "V1"]}),
        "third_party_commit": _git_commit(Path("/root/autodl-tmp/third_party/drivestudio")),
        "checkpoint_hashes": checkpoint_hashes,
        "scene_ids": ["003", "005", "004"],
        "actor_ids": actor_ids,
        "camera_ids": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        "time_range": {"selected_frames": {"003": 45, "005": 52, "004": 42}, "hz": 10},
        "seed": 0,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "cuda": "used",
        "gpu": "RTX 4090",
        "peak_cuda_bytes": render["peak_cuda_bytes"],
        "world_state_schema_version": WORLD_STATE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
        "safety_geometry_version": "safety-geometry-v1",
        "observation_evidence_version": "observation-evidence-v2-state-specific",
        "render_support_version": "render-support-v1",
        "certificate_version": "not_evaluated_H1-11C",
        "scenario_effect_version": "scenario-effect-v1",
        "provenance_version": "typed-label-sidecar-v1",
        "recovery_policy": "unknown_never_coerced_to_pass",
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
    atomic_write_json(str(run_dir / "fingerprints" / "environment.json"), manifest["environment"])
    atomic_write_json(str(run_dir / "fingerprints" / "data.json"), {"full_artifact_index_sha256": full_index_payload["index_sha256"], "world_state_set_sha256": world_state_hash, "render_label_set_sha256": render["render_label_set_sha256"], "audit_sha256": audit["audit_sha256"]})
    atomic_write_json(str(run_dir / "fingerprints" / "third_party.json"), {"drivestudio_commit": manifest["third_party_commit"]})
    atomic_write_json(str(run_dir / "fingerprints" / "checkpoints.json"), checkpoint_hashes)
    atomic_write_json(str(run_dir / "proposal_bank.json"), {"variants": ["V0", "V1"], "status": "interface_gate_only"})
    atomic_write_json(str(run_dir / "actor_registry.json"), {"actor_ids": actor_ids, "checkpoint_hashes": checkpoint_hashes})
    atomic_write_json(str(run_dir / "world_state_manifest.json"), {"world_state_set_sha256": world_state_hash, "sequences": world["sequences"], "world_state_hash": world_state_hash})
    atomic_write_json(str(run_dir / "render_request_manifest.json"), {"request_hashes": request_hashes, "render_request_hash": render_request_hash, "world_state_hash": world_state_hash})
    for source, relative, _ in sources:
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_write_json(str(run_dir / "artifact_manifest.json"), {"schema_version": 1, "world_state_hash": world_state_hash, "render_request_hash": render_request_hash, "artifact_set_hash": artifact_set_hash, "artifacts": artifact_rows})
    metrics = {"time": utc_now(), "step": 0, "sample_count": 18, "sidecar_count": audit["sidecar_count"], "label_sync_all_pass": True, "peak_cuda_bytes": render["peak_cuda_bytes"]}
    atomic_write_text(str(run_dir / "metrics.jsonl"), json.dumps(metrics, ensure_ascii=False) + "\n")
    summary = {
        "task_id": "V7-H1-11C",
        "run_id": run_id,
        "engineering_gate": "PASS",
        "hypothesis_verdict": "not_evaluated",
        "sample_count": 18,
        "sidecar_count": audit["sidecar_count"],
        "label_sync_checks": audit["checks"],
        "peak_cuda_bytes": render["peak_cuda_bytes"],
        "world_state_hash": world_state_hash,
        "render_request_hash": render_request_hash,
        "artifact_set_hash": artifact_set_hash,
        "full_artifact_index_sha256": full_index_payload["index_sha256"],
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(str(run_dir / "logs" / "finalize.log"), "H1-11C synchronized render and label implementation gate passed.\n")
    contract.finalize("COMPLETE", exit_reason="H1-11C_engineering_gate_passed")
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
