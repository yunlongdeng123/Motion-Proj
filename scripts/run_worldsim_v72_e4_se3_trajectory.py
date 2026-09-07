"""Evaluate learned canonical surfaces under real actor trajectories and SE(3) interventions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.eas_vggt.actor_visual_cache import load_actor_visual_cache
from motion_proj.worldsim_v72.eas_vggt.models import (
    CanonicalLateFusionEvidenceAdapter,
    lift_actor_surface_to_world,
)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _random_se3(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = rng.uniform(-25.0, 25.0, size=3)
    return transform


def _load_tracks(cache_run: Path, backbone: str, maximum_views: int):
    grouped = defaultdict(list)
    for path in sorted((cache_run / "actor_cache" / backbone).glob("*/*.npz")):
        cache = load_actor_visual_cache(path)
        grouped[(cache.scene_id, cache.track_id)].append(cache)
    tracks = []
    for (scene_id, track_id), caches in sorted(grouped.items()):
        first = caches[0]
        if len(caches) > maximum_views:
            raise ValueError(f"{track_id} exceeds maximum_views")
        for cache in caches[1:]:
            if not np.array_equal(cache.candidates_actor_m, first.candidates_actor_m):
                raise ValueError(f"candidate ordering changed for {track_id}")
        count = len(first.candidates_actor_m)
        feature_dim = first.visual_features.shape[-1]
        visual = np.zeros((count, maximum_views, feature_dim), dtype=np.float32)
        confidence = np.zeros((count, maximum_views), dtype=np.float32)
        observed = np.zeros((count, maximum_views), dtype=bool)
        for index, cache in enumerate(caches):
            visual[:, index] = cache.visual_features
            confidence[:, index] = cache.confidence_sum
            observed[:, index] = cache.confidence_sum > 0.0
        tracks.append(
            {
                "scene_id": scene_id,
                "track_id": track_id,
                "caches": caches,
                "candidates": first.candidates_actor_m,
                "base_features": first.base_features,
                "input_evidence_fou": first.input_evidence_fou,
                "opportunity_count": first.opportunity_count,
                "visual": visual,
                "confidence": confidence,
                "observed": observed,
            }
        )
    return tracks


def _model_inputs(track: dict[str, Any], device: torch.device):
    return {
        "base_features": torch.from_numpy(track["base_features"]).float().to(device),
        "canonical_xyz": torch.from_numpy(track["candidates"]).float().to(device),
        "evidence_fou": torch.from_numpy(track["input_evidence_fou"]).float().to(device),
        "opportunity_count": torch.from_numpy(track["opportunity_count"]).to(device),
        "view_visual_features": torch.from_numpy(track["visual"]).float().to(device),
        "view_observed": torch.from_numpy(track["observed"]).to(device),
        "view_confidence": torch.from_numpy(track["confidence"]).float().to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["task_id"] != "WS-V72-E4-SE3-RIGID-TRAJECTORY-01":
        raise ValueError("E4 task id mismatch")
    if config["data"]["source_test_read"] or config["data"]["external_test_read"]:
        raise ValueError("E4 role contract violated")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device(config["device"])
    rng = np.random.default_rng(int(config["seed"]))
    checkpoint = torch.load(config["model"]["checkpoint"], map_location=device, weights_only=True)
    if checkpoint["variant"] != "late_eas":
        raise ValueError("E4 requires a learned late_eas checkpoint")
    model = CanonicalLateFusionEvidenceAdapter(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tracks = _load_tracks(
        Path(config["data"]["cache_run"]),
        str(config["data"]["backbone"]),
        int(config["data"]["maximum_views"]),
    )
    manifest = {
        "schema_version": "worldsim_v72.e4_se3_trajectory.v1",
        "task_id": config["task_id"],
        "run_id": args.run_id,
        "status": "running",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "seed": int(config["seed"]),
        "source_test_read": False,
        "external_test_read": False,
        "checkpoint": str(config["model"]["checkpoint"]),
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "evaluate"})

    commute_errors = []
    recovery_errors = []
    distance_errors = []
    permutation_errors = []
    appearance_surface_errors = []
    pose_translations = []
    pose_rotations = []
    rows = []
    with torch.inference_mode():
        for track in tracks:
            inputs = _model_inputs(track, device)
            output = model(**inputs)
            candidates = inputs["canonical_xyz"]
            delta = output.surface_delta_actor_m
            canonical = candidates + delta
            order = torch.as_tensor(
                rng.permutation(inputs["view_visual_features"].shape[1]), device=device
            )
            permuted = dict(inputs)
            for name in ("view_visual_features", "view_observed", "view_confidence"):
                permuted[name] = inputs[name][:, order]
            permuted_output = model(**permuted)
            permutation_errors.append(
                float(torch.max(torch.abs(output.evidence_fou - permuted_output.evidence_fou)).cpu())
            )
            changed = dict(inputs)
            changed["view_visual_features"] = torch.randn_like(inputs["view_visual_features"])
            changed_output = model(**changed)
            appearance_surface_errors.append(
                float(torch.max(torch.abs(delta - changed_output.surface_delta_actor_m)).cpu())
            )
            poses = [torch.from_numpy(cache.world_from_actor).float().to(device) for cache in track["caches"]]
            base_pose = poses[0]
            base_rotation = base_pose[:3, :3]
            centers = []
            angles = []
            track_commute = []
            for pose in poses:
                world = lift_actor_surface_to_world(candidates, delta, pose)
                recovered = (world - pose[:3, 3]) @ pose[:3, :3]
                recovery_errors.append(float(torch.max(torch.abs(recovered - canonical)).cpu()))
                subset = min(int(config["evaluation"]["maximum_pairwise_points"]), len(world))
                reference_distance = torch.cdist(canonical[:subset], canonical[:subset])
                world_distance = torch.cdist(world[:subset], world[:subset])
                distance_errors.append(
                    float(torch.max(torch.abs(reference_distance - world_distance)).cpu())
                )
                for _ in range(int(config["evaluation"]["global_transform_count"])):
                    global_transform = torch.from_numpy(_random_se3(rng)).to(device)
                    transformed_pose = global_transform @ pose
                    transformed_world = lift_actor_surface_to_world(candidates, delta, transformed_pose)
                    expected = world @ global_transform[:3, :3].T + global_transform[:3, 3]
                    error = float(torch.max(torch.abs(transformed_world - expected)).cpu())
                    commute_errors.append(error)
                    track_commute.append(error)
                centers.append(pose[:3, 3].cpu().numpy())
                relative = base_rotation.T @ pose[:3, :3]
                cosine = torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
                angles.append(float(torch.arccos(cosine).cpu()))
            centers = np.asarray(centers)
            translation_span = float(np.max(np.linalg.norm(centers - centers[0], axis=1)))
            rotation_span = float(max(angles))
            pose_translations.append(translation_span)
            pose_rotations.append(rotation_span)
            rows.append(
                {
                    "scene_id": track["scene_id"],
                    "track_id": track["track_id"],
                    "candidate_count": len(track["candidates"]),
                    "pose_count": len(poses),
                    "translation_span_m": translation_span,
                    "rotation_span_rad": rotation_span,
                    "maximum_commutation_error_m": max(track_commute),
                }
            )

    metrics = {
        "actor_count": len(tracks),
        "multi_pose_actor_count": sum(len(track["caches"]) >= 2 for track in tracks),
        "pose_count": sum(len(track["caches"]) for track in tracks),
        "maximum_se3_commutation_error_m": max(commute_errors),
        "mean_se3_commutation_error_m": float(np.mean(commute_errors)),
        "maximum_actor_frame_recovery_error_m": max(recovery_errors),
        "maximum_pairwise_distance_error_m": max(distance_errors),
        "maximum_view_permutation_fou_error": max(permutation_errors),
        "maximum_appearance_to_surface_error_m": max(appearance_surface_errors),
        "maximum_observed_translation_span_m": max(pose_translations),
        "maximum_observed_rotation_span_rad": max(pose_rotations),
    }
    summary = {
        **manifest,
        "status": "done",
        "failure_ledger_delta": "none",
        "metrics": metrics,
        "rows": rows,
        "contracts": {
            "canonical_surface": "one actor-local surface prediction per tracked actor",
            "trajectory_action": "world_surface(t) = world_from_actor(t) * canonical_surface",
            "evidence_fusion": "commutative sum of non-negative per-view concentrations",
            "appearance_ownership": "visual tokens cannot change physical surface coordinates",
        },
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "summary": "summary.json"})
    _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
