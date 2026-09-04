"""Audit rigid deployment of frozen Actor-canonical M8 Gaussian physics."""

from __future__ import annotations

import argparse
import json
import resource
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
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    value = value / np.linalg.norm(value, axis=-1, keepdims=True).clip(min=1.0e-12)
    w, x, y, z = np.moveaxis(value, -1, 0)
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(value.shape[:-1] + (3, 3))


def _select_frames(valid_frames: np.ndarray, maximum: int) -> np.ndarray:
    if len(valid_frames) <= maximum:
        return valid_frames
    positions = np.linspace(0, len(valid_frames) - 1, num=maximum, dtype=np.int64)
    return valid_frames[np.unique(positions)]


def _select_rows(values: torch.Tensor, maximum: int) -> torch.Tensor:
    if len(values) <= maximum:
        return values
    positions = torch.linspace(
        0, len(values) - 1, steps=maximum, device=values.device
    ).round().long()
    return values.index_select(0, positions)


def _gaussian_energy(
    queries: torch.Tensor, centers: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    normalized_distance = torch.cdist(queries, centers) / scales.reshape(1, -1)
    return torch.logsumexp(-0.5 * normalized_distance.square(), dim=1)


def _load_m8(
    config: Mapping[str, Any], device: torch.device
) -> tuple[
    GaussianSeedExpansionMLP,
    RaySurfaceRelocationMLP,
    FeatureStandardizer,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    m8_run = Path(config["m8_run"])
    checkpoint = torch.load(m8_run / "MODEL.pt", map_location=device, weights_only=False)
    model_config = yaml.safe_load(
        (m8_run / "resolved.yaml").read_text(encoding="utf-8")
    )["model"]
    standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
    model = GaussianSeedExpansionMLP(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["branch_factor"]),
        int(checkpoint["slot_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)

    m5_run = Path(checkpoint["m5_run"])
    m5_checkpoint = torch.load(
        m5_run / "MODEL.pt", map_location=device, weights_only=False
    )
    base_config = yaml.safe_load(
        (m5_run / "resolved.yaml").read_text(encoding="utf-8")
    )["model"]
    base = RaySurfaceRelocationMLP(
        int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
    ).to(device)
    base.load_state_dict(m5_checkpoint["state_dict"])
    base.eval().requires_grad_(False)
    return model, base, standardizer, model_config, base_config


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M22 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        model, base, standardizer, model_config, base_config = _load_m8(config, device)
        scene_models = torch.load(
            Path(config["streetgs_checkpoint"]), map_location="cpu", weights_only=False
        )["models"]
        rigid = scene_models["RigidNodes"]
        background_gaussian_count = int(len(scene_models["Background"]["_means"]))
        appearance_means = rigid["_means"].detach().cpu().numpy()
        appearance_actor_ids = rigid["points_ids"].detach().cpu().numpy().reshape(-1)
        translations = rigid["instances_trans"].detach().cpu().numpy()
        quaternions = rigid["instances_quats"].detach().cpu().numpy()
        frame_validity = rigid["instances_fv"].detach().cpu().numpy()

        registry = json.loads(Path(config["actor_registry"]).read_text(encoding="utf-8"))
        registry_by_token = {
            str(actor["instance_token"]): actor
            for actor in registry["actors"]
            if actor.get("availability") == "available"
        }
        audit = config["audit"]
        anchor_scale_m = float(config["energy"]["anchor_scale_m"])
        rows: list[dict[str, Any]] = []
        _write_json(
            run_dir / "status.json",
            {"status": "running", "phase": "se3_composition_audit"},
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
                owned = appearance_means[appearance_actor_ids == rigid_index]
                valid_frames = np.flatnonzero(frame_validity[:, rigid_index])
                if len(owned) == 0 or len(valid_frames) == 0:
                    continue

                _, base_centers = m5_runner._move(base, actor, base_config)
                actor["m5_centers_t"] = base_centers
                children, _, child_scales = m7_runner._predict(
                    model, actor, model_config
                )
                centers = torch.cat([actor["anchors_t"], children], dim=0)
                scales = torch.cat(
                    [
                        torch.full(
                            (len(actor["anchors_t"]),),
                            anchor_scale_m,
                            dtype=centers.dtype,
                            device=device,
                        ),
                        child_scales,
                    ],
                    dim=0,
                )
                queries = _select_rows(
                    centers, int(audit["maximum_query_count"])
                )
                pairwise_points = _select_rows(
                    centers, int(audit["maximum_pairwise_points"])
                )
                canonical_energy = _gaussian_energy(queries, centers, scales)
                canonical_distances = torch.cdist(pairwise_points, pairwise_points)

                selected_frames = _select_frames(
                    valid_frames, int(audit["frames_per_actor"])
                )
                energy_residuals: list[float] = []
                distance_residuals: list[float] = []
                for frame in selected_frames:
                    rotation = torch.as_tensor(
                        _quaternion_wxyz_to_matrix(quaternions[frame, rigid_index]),
                        dtype=centers.dtype,
                        device=device,
                    )
                    translation = torch.as_tensor(
                        translations[frame, rigid_index],
                        dtype=centers.dtype,
                        device=device,
                    )
                    world_centers = centers @ rotation.T + translation
                    world_queries = queries @ rotation.T + translation
                    world_energy = _gaussian_energy(world_queries, world_centers, scales)
                    world_pairwise = pairwise_points @ rotation.T + translation
                    energy_residuals.append(
                        float(torch.max(torch.abs(world_energy - canonical_energy)))
                    )
                    distance_residuals.append(
                        float(
                            torch.max(
                                torch.abs(
                                    torch.cdist(world_pairwise, world_pairwise)
                                    - canonical_distances
                                )
                            )
                        )
                    )

                valid_translations = translations[valid_frames, rigid_index]
                translation_displacement = np.linalg.norm(
                    valid_translations - valid_translations[0], axis=1
                )
                canonical_centroid = centers.mean(dim=0).cpu().numpy()
                rotations = _quaternion_wxyz_to_matrix(
                    quaternions[valid_frames, rigid_index]
                )
                world_centroids = (
                    np.einsum("tij,j->ti", rotations, canonical_centroid)
                    + valid_translations
                )
                centroid_displacement = np.linalg.norm(
                    world_centroids - world_centroids[0], axis=1
                )
                moving = float(translation_displacement.max(initial=0.0)) > float(
                    audit["moving_translation_m"]
                )
                rows.append(
                    {
                        "scene_name": config["scene_name"],
                        "actor_token": actor_token,
                        "rigid_model_index": rigid_index,
                        "hazardous": bool(actor["hazardous"]),
                        "moving": moving,
                        "valid_trajectory_frame_count": int(len(valid_frames)),
                        "audited_frames": selected_frames.tolist(),
                        "trajectory_max_translation_m": float(
                            translation_displacement.max(initial=0.0)
                        ),
                        "world_centroid_max_displacement_m": float(
                            centroid_displacement.max(initial=0.0)
                        ),
                        "canonical_physical_gaussian_count": int(len(centers)),
                        "appearance_gaussian_count": int(len(owned)),
                        "canonical_extent_min_m": centers.min(dim=0).values.cpu().tolist(),
                        "canonical_extent_max_m": centers.max(dim=0).values.cpu().tolist(),
                        "mean_physical_scale_m": float(scales.mean()),
                        "maximum_energy_absolute_residual": max(
                            energy_residuals, default=0.0
                        ),
                        "maximum_pairwise_distance_residual_m": max(
                            distance_residuals, default=0.0
                        ),
                    }
                )
        if not rows:
            raise RuntimeError("no identity-matched M8/StreetGS Actors")

        maximum_energy_residual = max(
            float(row["maximum_energy_absolute_residual"]) for row in rows
        )
        maximum_distance_residual = max(
            float(row["maximum_pairwise_distance_residual_m"]) for row in rows
        )
        moving_actor_count = sum(bool(row["moving"]) for row in rows)
        decisions = {
            "identity_cohort_retained": len(rows)
            == int(config["decision"]["required_identity_matches"]),
            "moving_actor_present": moving_actor_count
            >= int(config["decision"]["minimum_moving_actor_count"]),
            "gaussian_energy_se3_equivariant": maximum_energy_residual
            <= float(config["decision"]["maximum_energy_absolute_residual"]),
            "pairwise_distance_rigidly_preserved": maximum_distance_residual
            <= float(config["decision"]["maximum_pairwise_distance_residual_m"]),
        }
        passed = all(decisions.values())
        _write_jsonl(run_dir / "DYNAMIC_COMPOSITION_ROWS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m22_se3_dynamic_static_composition.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "se3_dynamic_static_factorisation_supported"
                if passed
                else "se3_composition_implementation_rejected"
            ),
            "scene_name": config["scene_name"],
            "matched_actor_count": len(rows),
            "matched_hazard_actor_count": sum(bool(row["hazardous"]) for row in rows),
            "moving_actor_count": moving_actor_count,
            "quasi_static_actor_count": len(rows) - moving_actor_count,
            "valid_trajectory_frame_count": sum(
                int(row["valid_trajectory_frame_count"]) for row in rows
            ),
            "audited_actor_frame_count": sum(len(row["audited_frames"]) for row in rows),
            "canonical_physical_gaussian_count": sum(
                int(row["canonical_physical_gaussian_count"]) for row in rows
            ),
            "appearance_gaussian_count": sum(
                int(row["appearance_gaussian_count"]) for row in rows
            ),
            "static_background_gaussian_count": background_gaussian_count,
            "maximum_energy_absolute_residual": maximum_energy_residual,
            "maximum_pairwise_distance_residual_m": maximum_distance_residual,
            "maximum_actor_translation_m": max(
                float(row["trajectory_max_translation_m"]) for row in rows
            ),
            "maximum_world_centroid_displacement_m": max(
                float(row["world_centroid_max_displacement_m"]) for row in rows
            ),
            "decisions": decisions,
            "geometry_authority": "frozen_m8_actor_canonical_gt_supervised",
            "physics_authority": "frozen_m21_decoder_free_gaussian_energy",
            "motion_authority": "read_only_streetgs_rigid_se3",
            "static_authority": "streetgs_background_untouched",
            "appearance_channels_used_for_physics": [],
            "trajectory_inputs_to_geometry": [],
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
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "se3_composition_audit",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "se3_composition_audit",
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

