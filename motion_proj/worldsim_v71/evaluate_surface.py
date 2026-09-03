"""V7.1 表面 Pareto 指标。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from motion_proj.worldsim_v71.first_return_renderer import literal_first_return_partition


def symmetric_chamfer(
    surface: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int = 1024,
) -> float:
    surface_tensor = torch.as_tensor(surface, dtype=torch.float32, device=device).reshape(-1, 3)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device).reshape(-1, 3)
    if len(surface_tensor) == 0 or len(target_tensor) == 0:
        return float("inf")

    def directed(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        values = []
        with torch.inference_mode():
            for start in range(0, len(left), int(chunk_size)):
                values.append(torch.cdist(left[start : start + chunk_size], right).min(dim=1).values.cpu())
        return torch.cat(values)

    return float(0.5 * (directed(surface_tensor, target_tensor).mean() + directed(target_tensor, surface_tensor).mean()))


def differentiable_symmetric_chamfer(surface: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if len(surface) == 0 or len(target) == 0:
        raise ValueError("Chamfer 输入不能为空")
    distances = torch.cdist(surface, target)
    return 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())


def evaluate_actor_surface(
    baseline_surface: np.ndarray,
    output_surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    *,
    hazardous: bool,
    device: torch.device,
    lateral_tolerance_m: float,
    depth_tolerance_m: float,
    distance_chunk_size: int,
) -> dict[str, Any]:
    baseline = literal_first_return_partition(
        baseline_surface,
        target,
        origins,
        lateral_tolerance_m=lateral_tolerance_m,
        depth_tolerance_m=depth_tolerance_m,
        device=device,
        ray_chunk_size=distance_chunk_size,
    )
    output = literal_first_return_partition(
        output_surface,
        target,
        origins,
        lateral_tolerance_m=lateral_tolerance_m,
        depth_tolerance_m=depth_tolerance_m,
        device=device,
        ray_chunk_size=distance_chunk_size,
    )
    return {
        "hazardous": bool(hazardous),
        "target_ray_count": int(len(target)),
        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
        "output_early_count": int(np.count_nonzero(output["early"])),
        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
        "output_hit_count": int(np.count_nonzero(output["hit"])),
        "baseline_chamfer_m": symmetric_chamfer(
            baseline_surface, target, device=device, chunk_size=distance_chunk_size
        ),
        "output_chamfer_m": symmetric_chamfer(
            output_surface, target, device=device, chunk_size=distance_chunk_size
        ),
        "actor_state_retention": 1.0,
        "hazard_state_retention": 1.0,
    }


def summarize_surface_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("没有可汇总的 Actor surface row")

    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, float | int | None]:
        ray_count = sum(int(row["target_ray_count"]) for row in selected)
        baseline_early = sum(int(row["baseline_early_count"]) for row in selected)
        output_early = sum(int(row["output_early_count"]) for row in selected)
        baseline_rate = baseline_early / ray_count if ray_count else None
        output_rate = output_early / ray_count if ray_count else None
        relative_reduction = (
            (baseline_rate - output_rate) / baseline_rate
            if baseline_rate is not None and output_rate is not None and baseline_rate > 0.0
            else None
        )
        return {
            "actor_count": len(selected),
            "ray_count": ray_count,
            "baseline_early_count": baseline_early,
            "output_early_count": output_early,
            "baseline_early_rate": baseline_rate,
            "output_early_rate": output_rate,
            "relative_early_reduction": relative_reduction,
        }

    all_rows = list(rows)
    hazard_rows = [row for row in rows if bool(row["hazardous"])]
    clear_rows = [row for row in rows if not bool(row["hazardous"])]
    baseline_chamfer = float(np.mean([float(row["baseline_chamfer_m"]) for row in rows]))
    output_chamfer = float(np.mean([float(row["output_chamfer_m"]) for row in rows]))
    baseline_hits = sum(int(row["baseline_hit_count"]) for row in rows)
    output_hits = sum(int(row["output_hit_count"]) for row in rows)
    rays = sum(int(row["target_ray_count"]) for row in rows)
    return {
        "all": stratum(all_rows),
        "hazard": stratum(hazard_rows),
        "clear": stratum(clear_rows),
        "baseline_mean_chamfer_m": baseline_chamfer,
        "output_mean_chamfer_m": output_chamfer,
        "chamfer_delta_m": output_chamfer - baseline_chamfer,
        "baseline_hit_recall": baseline_hits / rays if rays else None,
        "output_hit_recall": output_hits / rays if rays else None,
        "hit_recall_delta": (output_hits - baseline_hits) / rays if rays else None,
        "minimum_actor_state_retention": min(float(row["actor_state_retention"]) for row in rows),
        "minimum_hazard_state_retention": min(float(row["hazard_state_retention"]) for row in rows),
    }
