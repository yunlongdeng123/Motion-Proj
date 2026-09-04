"""Diagnose smooth-depth versus hard earliest-return disagreement for M50."""

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
import run_worldsim_v71_m6_gt_supervised_gaussian_relocation as m6_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m8_temporal_frame_coverage as m8_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.first_return_renderer import (
    differentiable_first_return_depth,
    literal_first_return_partition,
)
from motion_proj.worldsim_v71.gaussian_anchor_relocation import GaussianSeedExpansionMLP
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _model(checkpoint: Mapping[str, Any], device: torch.device) -> GaussianSeedExpansionMLP:
    model = GaussianSeedExpansionMLP(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["branch_factor"]),
        int(checkpoint["slot_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model


def _hard(
    surface: np.ndarray | torch.Tensor,
    targets: np.ndarray,
    origins: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    return literal_first_return_partition(
        surface,
        targets,
        origins,
        lateral_tolerance_m=float(config["lateral_tolerance_m"]),
        depth_tolerance_m=float(config["depth_tolerance_m"]),
        device=device,
        ray_chunk_size=int(config["ray_chunk_size"]),
    )


def _paired_row(
    actor: Mapping[str, Any],
    reference_children: torch.Tensor,
    current_children: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    targets = np.asarray(actor["target"], dtype=np.float32)
    origins = np.asarray(actor["target_sensor_origins"], dtype=np.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    origin_tensor = torch.as_tensor(origins, dtype=torch.float32, device=device)
    limited_anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    reference_train = torch.cat([limited_anchors, reference_children], dim=0)
    current_train = torch.cat([limited_anchors, current_children], dim=0)
    reference_deploy = _voxel_unique(
        torch.cat([actor["anchors_t"], reference_children], dim=0).cpu().numpy(),
        float(config["output_voxel_size_m"]),
    )
    current_deploy = _voxel_unique(
        torch.cat([actor["anchors_t"], current_children], dim=0).cpu().numpy(),
        float(config["output_voxel_size_m"]),
    )

    smooth_reference = differentiable_first_return_depth(
        reference_train, origin_tensor, target_tensor, **config["renderer"]
    ).cpu().numpy()
    smooth_current = differentiable_first_return_depth(
        current_train, origin_tensor, target_tensor, **config["renderer"]
    ).cpu().numpy()
    target_depth = np.linalg.norm(targets - origins, axis=1)
    smooth_reference_error = np.abs(smooth_reference - target_depth)
    smooth_current_error = np.abs(smooth_current - target_depth)
    soft_improved = smooth_current_error < smooth_reference_error
    soft_worsened = smooth_current_error > smooth_reference_error

    hard_train_reference = _hard(
        reference_train, targets, origins, config, device
    )
    hard_train_current = _hard(current_train, targets, origins, config, device)
    hard_deploy_reference = _hard(
        reference_deploy, targets, origins, config, device
    )
    hard_deploy_current = _hard(current_deploy, targets, origins, config, device)

    train_added = (~hard_train_reference["early"]) & hard_train_current["early"]
    train_removed = hard_train_reference["early"] & (~hard_train_current["early"])
    deploy_added = (~hard_deploy_reference["early"]) & hard_deploy_current["early"]
    deploy_removed = hard_deploy_reference["early"] & (~hard_deploy_current["early"])
    support_flip = hard_train_current["early"] != hard_deploy_current["early"]

    displacement = m8_runner._motion_displacement(actor)
    ray_count = len(targets)
    row = {
        "scene_name": str(actor["scene_name"]),
        "track_id": str(actor["track_id"]),
        "hazardous": bool(actor["hazardous"]),
        "moving": displacement > float(config["moving_max_displacement_m"]),
        "trajectory_max_displacement_m": displacement,
        "ray_count": ray_count,
        "smooth_reference_abs_error_sum": float(smooth_reference_error.sum()),
        "smooth_current_abs_error_sum": float(smooth_current_error.sum()),
        "soft_improved_count": int(soft_improved.sum()),
        "soft_worsened_count": int(soft_worsened.sum()),
        "train_reference_early_count": int(hard_train_reference["early"].sum()),
        "train_current_early_count": int(hard_train_current["early"].sum()),
        "train_added_early_count": int(train_added.sum()),
        "train_removed_early_count": int(train_removed.sum()),
        "train_added_soft_improved_count": int((train_added & soft_improved).sum()),
        "train_removed_soft_worsened_count": int((train_removed & soft_worsened).sum()),
        "deploy_reference_early_count": int(hard_deploy_reference["early"].sum()),
        "deploy_current_early_count": int(hard_deploy_current["early"].sum()),
        "deploy_added_early_count": int(deploy_added.sum()),
        "deploy_removed_early_count": int(deploy_removed.sum()),
        "deploy_added_soft_improved_count": int((deploy_added & soft_improved).sum()),
        "deploy_removed_soft_worsened_count": int((deploy_removed & soft_worsened).sum()),
        "current_support_realization_flip_count": int(support_flip.sum()),
    }
    arrays = {
        "smooth_depth_delta": smooth_current - smooth_reference,
        "smooth_abs_error_delta": smooth_current_error - smooth_reference_error,
        "hard_train_depth_delta": np.where(
            np.isfinite(hard_train_current["first_depth"]),
            hard_train_current["first_depth"],
            target_depth + float(config["fallback_margin_m"]),
        )
        - np.where(
            np.isfinite(hard_train_reference["first_depth"]),
            hard_train_reference["first_depth"],
            target_depth + float(config["fallback_margin_m"]),
        ),
        "hard_deploy_depth_delta": np.where(
            np.isfinite(hard_deploy_current["first_depth"]),
            hard_deploy_current["first_depth"],
            target_depth + float(config["fallback_margin_m"]),
        )
        - np.where(
            np.isfinite(hard_deploy_reference["first_depth"]),
            hard_deploy_reference["first_depth"],
            target_depth + float(config["fallback_margin_m"]),
        ),
    }
    return row, arrays


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 2 or np.std(left[finite]) == 0 or np.std(right[finite]) == 0:
        return None
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _summary(
    rows: list[dict[str, Any]],
    arrays: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    def stratum(indices: list[int]) -> dict[str, Any]:
        selected = [rows[index] for index in indices]
        rays = sum(int(row["ray_count"]) for row in selected)
        sums = lambda key: sum(int(row[key]) for row in selected)
        float_sums = lambda key: sum(float(row[key]) for row in selected)
        added_train = sums("train_added_early_count")
        added_deploy = sums("deploy_added_early_count")
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "smooth_reference_abs_error_m": float_sums("smooth_reference_abs_error_sum") / max(rays, 1),
            "smooth_current_abs_error_m": float_sums("smooth_current_abs_error_sum") / max(rays, 1),
            "soft_improved_fraction": sums("soft_improved_count") / max(rays, 1),
            "soft_worsened_fraction": sums("soft_worsened_count") / max(rays, 1),
            "train_reference_early_rate": sums("train_reference_early_count") / max(rays, 1),
            "train_current_early_rate": sums("train_current_early_count") / max(rays, 1),
            "train_added_early_fraction": added_train / max(rays, 1),
            "train_removed_early_fraction": sums("train_removed_early_count") / max(rays, 1),
            "train_added_early_with_soft_improvement_fraction": sums("train_added_soft_improved_count") / max(added_train, 1),
            "deploy_reference_early_rate": sums("deploy_reference_early_count") / max(rays, 1),
            "deploy_current_early_rate": sums("deploy_current_early_count") / max(rays, 1),
            "deploy_added_early_fraction": added_deploy / max(rays, 1),
            "deploy_removed_early_fraction": sums("deploy_removed_early_count") / max(rays, 1),
            "deploy_added_early_with_soft_improvement_fraction": sums("deploy_added_soft_improved_count") / max(added_deploy, 1),
            "current_support_realization_flip_fraction": sums("current_support_realization_flip_count") / max(rays, 1),
        }

    all_indices = list(range(len(rows)))
    pooled = {
        key: np.concatenate([item[key] for item in arrays], axis=0) for key in arrays[0]
    }
    return {
        "all": stratum(all_indices),
        "hazard": stratum([i for i, row in enumerate(rows) if bool(row["hazardous"])]),
        "clear": stratum([i for i, row in enumerate(rows) if not bool(row["hazardous"])]),
        "moving": stratum([i for i, row in enumerate(rows) if bool(row["moving"])]),
        "quasi_static": stratum([i for i, row in enumerate(rows) if not bool(row["moving"])]),
        "pooled_depth_delta_correlation_train": _correlation(
            pooled["smooth_depth_delta"], pooled["hard_train_depth_delta"]
        ),
        "pooled_depth_delta_correlation_deploy": _correlation(
            pooled["smooth_depth_delta"], pooled["hard_deploy_depth_delta"]
        ),
        "pooled_soft_error_vs_hard_train_depth_correlation": _correlation(
            pooled["smooth_abs_error_delta"], pooled["hard_train_depth_delta"]
        ),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    m6_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M51 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        m8_run = Path(config["m8_run"])
        m50_run = Path(config["m50_run"])
        m8_checkpoint = torch.load(m8_run / "MODEL.pt", map_location=device, weights_only=False)
        m50_checkpoint = torch.load(m50_run / "MODEL.pt", map_location=device, weights_only=False)
        m8_config = yaml.safe_load((m8_run / "resolved.yaml").read_text(encoding="utf-8"))
        m50_config = yaml.safe_load((m50_run / "resolved.yaml").read_text(encoding="utf-8"))
        standardizer = FeatureStandardizer.from_payload(m8_checkpoint["standardizer"])
        m5_run = Path(m8_checkpoint["m5_run"])
        m5_checkpoint = torch.load(m5_run / "MODEL.pt", map_location=device, weights_only=False)
        m5_config = yaml.safe_load((m5_run / "resolved.yaml").read_text(encoding="utf-8"))
        base = RaySurfaceRelocationMLP(
            int(m5_checkpoint["input_dim"]), int(m5_checkpoint["hidden_dim"])
        ).to(device)
        base.load_state_dict(m5_checkpoint["state_dict"])
        base.eval()
        base.requires_grad_(False)
        reference_model = _model(m8_checkpoint, device)
        current_model = _model(m50_checkpoint, device)
        actors = [
            actor
            for path in m0_runner._paths(
                Path(config["cache_root"]), int(config["maximum_training_actors"])
            )
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        stride = int(config["holdout_stride"])
        holdout = [actor for index, actor in enumerate(actors) if index % stride == 0]
        rows: list[dict[str, Any]] = []
        arrays: list[dict[str, np.ndarray]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout):
                _, centers = m5_runner._move(base, actor, m5_config["model"])
                actor["m5_centers_t"] = centers.detach()
                reference_children, _, _ = m7_runner._predict(
                    reference_model, actor, m8_config["model"]
                )
                current_children, _, _ = m7_runner._predict(
                    current_model, actor, m50_config["model"]
                )
                row, item_arrays = _paired_row(
                    actor,
                    reference_children,
                    current_children,
                    config["diagnosis"],
                    device,
                )
                rows.append(row)
                arrays.append(item_arrays)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout):
                    print(
                        json.dumps({"stage": "m51_diagnosis", "progress": f"{index + 1}/{len(holdout)}"}),
                        flush=True,
                    )
        diagnosis = _summary(rows, arrays)
        m6_runner._write_jsonl(run_dir / "ACTOR_ROWS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m51_soft_hard_first_return_diagnosis.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "smooth_hard_first_return_nonimplication_confirmed",
            "holdout_actor_count": len(holdout),
            "training": False,
            "model_selection": False,
            "diagnosis": diagnosis,
            "same_support_comparison": "limited_anchors_plus_children_no_voxelization",
            "deployment_comparison": "full_anchors_plus_children_voxelized_0p06m",
            "external_read": False,
            "m43_partial_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        m6_runner._write_json(run_dir / "summary.json", summary)
        m6_runner._write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "diagnosis", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        m6_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m51", "error": f"{type(error).__name__}: {error}"},
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
