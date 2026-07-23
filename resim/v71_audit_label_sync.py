#!/usr/bin/env python
"""独立复验 H1-11C WorldState/render/label/sidecar 同步合同。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.label_regeneration import bytes_sha256
from motion_proj.resim.schema import render_request_hash, world_state_hash
from motion_proj.runtime.atomic import atomic_write_json


REQUIRED = {
    "rgb.png",
    "alpha.npy",
    "depth_render_expected.npy",
    "depth_surface_first_hit.npy",
    "depth_surface_first_hit_valid.npy",
    "depth_lidar_measured.npy",
    "depth_lidar_measured_valid.npy",
    "rgb_background.npy",
    "alpha_background.npy",
    "depth_surface_first_hit_background.npy",
    "depth_surface_first_hit_background_valid.npy",
    "rgb_actor_only.npy",
    "alpha_actor_only.npy",
    "depth_surface_first_hit_actor_only.npy",
    "depth_surface_first_hit_actor_only_valid.npy",
    "vehicle_instance_mask.npy",
    "limited_semantic_mask.npy",
    "safety_geometry_map.npy",
    "observation_evidence_map.npy",
    "render_support_map.npy",
    "boxes.json",
    "evidence_refs.json",
    "render_request.json",
    "label_sync_audit.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/world_states"),
    )
    parser.add_argument(
        "--render-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/render_labels"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/label_sync_audit.json"),
    )
    args = parser.parse_args()
    render_summary = json.loads((args.render_root / "summary.json").read_text())
    checks = {
        "sample_count_18": render_summary["sample_count"] == 18,
        "writer_audits_all_pass": render_summary["label_sync_all_pass"],
    }
    world_hashes = {}
    identity_rows = []
    for world_path in sorted(args.world_root.glob("*/*/world_state.json")):
        world = json.loads(world_path.read_text())
        expected = world.pop("world_state_hash")
        actual = world_state_hash(world)
        world_hashes[world_path.parent.relative_to(args.world_root).as_posix()] = expected
        identity_rows.append(
            (
                world["scene_id"],
                world["edit_spec"]["variant"],
                {int(frame["actor_nodes"][0]["true_instance_id"]) for frame in world["frames"]},
                {
                    camera["camera_id"]
                    for frame in world["frames"]
                    for camera in frame["camera_models"]
                },
            )
        )
        if actual != expected:
            raise RuntimeError(f"WorldState hash mismatch: {world_path}")
    checks["world_state_hashes_recompute"] = len(world_hashes) == 6
    checks["temporal_identity_stable"] = all(len(row[2]) == 1 for row in identity_rows)
    checks["three_cameras_every_sequence"] = all(
        row[3] == {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"}
        for row in identity_rows
    )

    sample_results = {}
    sidecar_count = 0
    tier_counts = Counter()
    for audit_path in sorted(args.render_root.glob("*/*/*/label_sync_audit.json")):
        sample_dir = audit_path.parent
        relative = sample_dir.relative_to(args.render_root).as_posix()
        names = {path.name for path in sample_dir.iterdir() if not path.name.endswith(".sidecar.json")}
        missing = sorted(REQUIRED - names)
        artifact_ok = not missing
        request = json.loads((sample_dir / "render_request.json").read_text())
        request_world_hash = request["world_state_hash"]
        expected_request_hash = request.pop("render_request_hash")
        request_ok = render_request_hash(request) == expected_request_hash
        sidecars_ok = True
        for artifact_name in REQUIRED:
            artifact = sample_dir / artifact_name
            sidecar_path = artifact.with_suffix(artifact.suffix + ".sidecar.json")
            if not artifact.exists() or not sidecar_path.exists():
                sidecars_ok = False
                continue
            sidecar = json.loads(sidecar_path.read_text())
            sidecar_count += 1
            sidecars_ok &= (
                sidecar["artifact_hash"] == bytes_sha256(artifact)
                and sidecar["world_state_hash"] == request_world_hash
                and sidecar["render_request_hash"] == expected_request_hash
            )
            if sidecar.get("artifact_type") == "typed_depth":
                tier_counts[(artifact_name, sidecar.get("truth_tier"))] += 1
        instance = np.load(sample_dir / "vehicle_instance_mask.npy")
        semantic = np.load(sample_dir / "limited_semantic_mask.npy")
        safety = np.load(sample_dir / "safety_geometry_map.npy").astype(bool)
        observation = np.load(sample_dir / "observation_evidence_map.npy")
        render_support = np.load(sample_dir / "render_support_map.npy")
        bg_depth = np.load(sample_dir / "depth_surface_first_hit_background.npy")
        bg_valid = np.load(sample_dir / "depth_surface_first_hit_background_valid.npy").astype(bool)
        actor_depth = np.load(sample_dir / "depth_surface_first_hit_actor_only.npy")
        actor_valid = np.load(sample_dir / "depth_surface_first_hit_actor_only_valid.npy").astype(bool)
        vehicle_pixels = instance > 0
        depth_order = np.all(
            (~vehicle_pixels)
            | (
                actor_valid
                & ((~bg_valid) | (actor_depth <= bg_depth + 1e-4))
            )
        )
        product_separation = (
            safety.shape != render_support.shape
            and observation.shape != render_support.shape
            and safety.shape == observation.shape
        )
        evidence_alignment = bool(
            safety.any() and np.all(observation[safety] == 3)
        )
        writer_audit = json.loads(audit_path.read_text())
        sample_results[relative] = {
            "required_artifacts": artifact_ok,
            "request_hash_recomputes": request_ok,
            "sidecars_integral": bool(sidecars_ok),
            "limited_semantic_scope": set(np.unique(semantic)).issubset({0, 1, 2, 255}),
            "instance_depth_order": bool(depth_order),
            "state_evidence_alignment": evidence_alignment,
            "three_products_separate": product_separation,
            "writer_audit_pass": writer_audit["pass"],
        }
        sample_results[relative]["pass"] = all(sample_results[relative].values())
    checks["all_18_sample_audits_pass"] = (
        len(sample_results) == 18 and all(value["pass"] for value in sample_results.values())
    )
    checks["all_artifacts_have_sidecars"] = sidecar_count == 18 * len(REQUIRED)
    checks["typed_depth_tiers_exact"] = (
        tier_counts[("depth_render_expected.npy", "diagnostic")] == 18
        and tier_counts[("depth_surface_first_hit.npy", "T1")] == 18
        and tier_counts[("depth_lidar_measured.npy", "T0")] == 18
        and sum(tier_counts.values()) == 54
    )
    checks["S1_included"] = any(key.startswith("005/") for key in sample_results)
    checks["pass"] = all(checks.values())
    result = {
        "schema_version": "label-sync-audit-v1",
        "task_id": "V7-H1-11C",
        "checks": checks,
        "sample_results": sample_results,
        "sidecar_count": sidecar_count,
        "typed_depth_tier_counts": {
            f"{name}:{tier}": count for (name, tier), count in tier_counts.items()
        },
        "world_state_set_sha256": canonical_sha256(world_hashes),
        "render_label_set_sha256": render_summary["render_label_set_sha256"],
        "peak_cuda_bytes": render_summary["peak_cuda_bytes"],
    }
    result["audit_sha256"] = canonical_sha256(result)
    atomic_write_json(str(args.output), result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
