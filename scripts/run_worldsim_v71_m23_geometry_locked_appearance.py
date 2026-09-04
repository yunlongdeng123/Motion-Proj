"""Attach frozen image-trained attributes to frozen M8 physical Gaussians."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v71.evaluate_surface import (
    evaluate_actor_surface,
    summarize_surface_rows,
)


def _nearest_appearance(
    physical_centers: torch.Tensor,
    appearance_centers: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference = torch.as_tensor(
        appearance_centers,
        dtype=physical_centers.dtype,
        device=physical_centers.device,
    )
    indices: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    for start in range(0, len(physical_centers), chunk_size):
        pairwise = torch.cdist(
            physical_centers[start : start + chunk_size], reference
        )
        values, nearest = pairwise.min(dim=1)
        indices.append(nearest)
        distances.append(values)
    return (
        torch.cat(indices).cpu().numpy().astype(np.int64),
        torch.cat(distances).cpu().numpy().astype(np.float32),
    )


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    m22_runner._write_json(
        run_dir / "status.json", {"status": "running", "phase": "loading"}
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M23 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        model, base, standardizer, model_config, base_config = m22_runner._load_m8(
            config, device
        )
        rigid = torch.load(
            Path(config["streetgs_checkpoint"]), map_location="cpu", weights_only=False
        )["models"]["RigidNodes"]
        appearance_means = rigid["_means"].detach().cpu().numpy()
        appearance_actor_ids = rigid["points_ids"].detach().cpu().numpy().reshape(-1)
        features_dc = rigid["_features_dc"].detach().cpu().numpy()
        features_rest = rigid["_features_rest"].detach().cpu().numpy()
        opacity_logits = rigid["_opacities"].detach().cpu().numpy()
        registry = json.loads(Path(config["actor_registry"]).read_text(encoding="utf-8"))
        registry_by_token = {
            str(actor["instance_token"]): actor
            for actor in registry["actors"]
            if actor.get("availability") == "available"
        }

        rows: list[dict[str, Any]] = []
        center_parts: list[np.ndarray] = []
        scale_parts: list[np.ndarray] = []
        dc_parts: list[np.ndarray] = []
        rest_parts: list[np.ndarray] = []
        opacity_parts: list[np.ndarray] = []
        source_index_parts: list[np.ndarray] = []
        offsets = [0]
        actor_tokens: list[str] = []
        rigid_indices: list[int] = []
        hazards: list[bool] = []
        anchor_scale_m = float(config["appearance"]["anchor_scale_m"])
        chunk_size = int(config["appearance"]["association_chunk_size"])
        m22_runner._write_json(
            run_dir / "status.json",
            {"status": "running", "phase": "geometry_locked_association"},
        )
        with torch.inference_mode():
            for path in sorted(Path(config["actor_cache_dir"]).glob("*.npz")):
                actor_token = path.stem
                registry_actor = registry_by_token.get(actor_token)
                if registry_actor is None:
                    continue
                actor = m0_runner._prepare_actor(path, standardizer, device)
                if actor is None:
                    continue
                rigid_index = int(registry_actor["rigid_model_index"])
                owned_indices = np.flatnonzero(appearance_actor_ids == rigid_index)
                if len(owned_indices) == 0:
                    continue

                _, base_centers = m5_runner._move(base, actor, base_config)
                actor["m5_centers_t"] = base_centers
                children, _, child_scales = m7_runner._predict(
                    model, actor, model_config
                )
                physical_centers = torch.cat(
                    [actor["anchors_t"], children], dim=0
                )
                physical_scales = torch.cat(
                    [
                        torch.full(
                            (len(actor["anchors_t"]),),
                            anchor_scale_m,
                            dtype=physical_centers.dtype,
                            device=device,
                        ),
                        child_scales,
                    ]
                )
                nearest_local, distances = _nearest_appearance(
                    physical_centers,
                    appearance_means[owned_indices],
                    chunk_size,
                )
                nearest_global = owned_indices[nearest_local]

                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                output = _voxel_unique(
                    physical_centers.cpu().numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                row = evaluate_actor_surface(
                    baseline,
                    output,
                    actor["target"],
                    actor["target_sensor_origins"],
                    hazardous=bool(actor["hazardous"]),
                    device=device,
                    lateral_tolerance_m=float(
                        config["evaluation"]["literal_lateral_tolerance_m"]
                    ),
                    depth_tolerance_m=float(
                        config["evaluation"]["literal_depth_tolerance_m"]
                    ),
                    distance_chunk_size=int(
                        config["evaluation"]["distance_chunk_size"]
                    ),
                )
                row.update(
                    {
                        "scene_name": config["scene_name"],
                        "actor_token": actor_token,
                        "rigid_model_index": rigid_index,
                        "physical_gaussian_count": int(len(physical_centers)),
                        "appearance_gaussian_count": int(len(owned_indices)),
                        "assigned_attribute_count": int(len(nearest_global)),
                        "assignment_distance_mean_m": float(distances.mean()),
                        "assignment_distance_median_m": float(np.median(distances)),
                        "assignment_distance_q90_m": float(
                            np.quantile(distances, 0.90)
                        ),
                        "assignment_distance_max_m": float(distances.max()),
                        "physical_center_source": "frozen_m8",
                        "physical_scale_source": "frozen_m8_or_fixed_anchor",
                        "copied_visual_attributes": [
                            "features_dc",
                            "features_rest",
                            "opacity_logit",
                        ],
                    }
                )
                rows.append(row)
                centers_np = physical_centers.cpu().numpy().astype(np.float32)
                center_parts.append(centers_np)
                scale_parts.append(physical_scales.cpu().numpy().astype(np.float32))
                dc_parts.append(features_dc[nearest_global].astype(np.float32))
                rest_parts.append(features_rest[nearest_global].astype(np.float32))
                opacity_parts.append(opacity_logits[nearest_global].astype(np.float32))
                source_index_parts.append(nearest_global.astype(np.int64))
                offsets.append(offsets[-1] + len(centers_np))
                actor_tokens.append(actor_token)
                rigid_indices.append(rigid_index)
                hazards.append(bool(actor["hazardous"]))
        if not rows:
            raise RuntimeError("no identity-matched M8/StreetGS Actors")

        metrics = summarize_surface_rows(rows)
        physical_count = sum(int(row["physical_gaussian_count"]) for row in rows)
        assigned_count = sum(int(row["assigned_attribute_count"]) for row in rows)
        all_distances = np.concatenate(
            [
                np.asarray(
                    [row["assignment_distance_mean_m"]], dtype=np.float64
                )
                for row in rows
            ]
        )
        decisions = {
            "identity_cohort_retained": len(rows)
            == int(config["decision"]["required_identity_matches"]),
            "all_physical_gaussians_attributed": assigned_count == physical_count,
            "actor_and_hazard_state_retained": float(
                metrics["minimum_actor_state_retention"]
            )
            >= float(config["decision"]["required_actor_state_retention"])
            and float(metrics["minimum_hazard_state_retention"])
            >= float(config["decision"]["required_hazard_state_retention"]),
        }
        passed = all(decisions.values())
        m22_runner._write_jsonl(run_dir / "ACTOR_APPEARANCE_ROWS.jsonl", rows)
        np.savez_compressed(
            run_dir / "GEOMETRY_LOCKED_APPEARANCE_SIDECAR.npz",
            centers=np.concatenate(center_parts, axis=0),
            scales=np.concatenate(scale_parts, axis=0),
            features_dc=np.concatenate(dc_parts, axis=0),
            features_rest=np.concatenate(rest_parts, axis=0),
            opacity_logits=np.concatenate(opacity_parts, axis=0),
            source_appearance_indices=np.concatenate(source_index_parts, axis=0),
            offsets=np.asarray(offsets, dtype=np.int64),
            actor_tokens=np.asarray(actor_tokens),
            rigid_model_indices=np.asarray(rigid_indices, dtype=np.int64),
            hazardous=np.asarray(hazards, dtype=np.bool_),
        )
        shutil.copy2(Path(config["rendered_asset"]), run_dir / "STREETGS_APPEARANCE.png")
        summary = {
            "schema_version": "worldsim_v71.m23_geometry_locked_appearance.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "geometry_locked_appearance_carrier_supported"
                if passed
                else "geometry_locked_appearance_carrier_rejected"
            ),
            "scene_name": config["scene_name"],
            "matched_actor_count": len(rows),
            "matched_hazard_actor_count": int(metrics["hazard"]["actor_count"]),
            "physical_gaussian_count": physical_count,
            "assigned_attribute_count": assigned_count,
            "appearance_gaussian_count": sum(
                int(row["appearance_gaussian_count"]) for row in rows
            ),
            "actor_mean_assignment_distance_m": float(all_distances.mean()),
            "assignment_distance_max_m": max(
                float(row["assignment_distance_max_m"]) for row in rows
            ),
            "physical": metrics,
            "decisions": decisions,
            "geometry_locked": True,
            "appearance_to_geometry_gradient": False,
            "appearance_geometry_copied": [],
            "appearance_attributes_copied": [
                "features_dc",
                "features_rest",
                "opacity_logit",
            ],
            "background_used": False,
            "trajectory_mutated": False,
            "checkpoint_written": False,
            "training": False,
            "render_filter": False,
            "external_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        m22_runner._write_json(run_dir / "summary.json", summary)
        m22_runner._write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "geometry_locked_association",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        m22_runner._write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "geometry_locked_association",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

