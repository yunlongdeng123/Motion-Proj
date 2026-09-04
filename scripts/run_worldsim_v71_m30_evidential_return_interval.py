"""Evaluate known/possible first-return intervals without collapsing UNKNOWN."""

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
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
from motion_proj.worldsim_v71.first_return_renderer import (
    literal_first_return_partition,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _motion_displacement(actor: Mapping[str, Any]) -> float:
    trajectory = np.asarray(actor["trajectory_xyz_m"], dtype=np.float64).reshape(-1, 3)
    if len(trajectory) == 0:
        return 0.0
    return float(np.linalg.norm(trajectory - trajectory[0], axis=1).max(initial=0.0))


def _actor_interval_row(
    actor: Mapping[str, Any],
    certain_surface: np.ndarray,
    possible_surface: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    kwargs = {
        "lateral_tolerance_m": float(config["lateral_tolerance_m"]),
        "depth_tolerance_m": float(config["depth_tolerance_m"]),
        "device": device,
        "ray_chunk_size": int(config["ray_chunk_size"]),
        "point_chunk_size": int(config["point_chunk_size"]),
    }
    possible = literal_first_return_partition(
        possible_surface,
        actor["target"],
        actor["target_sensor_origins"],
        **kwargs,
    )
    certain = literal_first_return_partition(
        certain_surface,
        actor["target"],
        actor["target_sensor_origins"],
        **kwargs,
    )
    target = possible["target_depth"].astype(np.float64)
    possible_depth = possible["first_depth"].astype(np.float64)
    certain_depth = certain["first_depth"].astype(np.float64)
    possible_finite = np.isfinite(possible_depth)
    certain_finite = np.isfinite(certain_depth)
    tolerance = float(config["depth_tolerance_m"])

    ordering = possible_finite & (
        (~certain_finite) | (possible_depth <= certain_depth + 1.0e-5)
    )
    ordering_violation = (~possible_finite & certain_finite) | (
        possible_finite & certain_finite & (possible_depth > certain_depth + 1.0e-5)
    )
    possible_lower_covers = possible_finite & (possible_depth <= target + tolerance)
    certain_upper_covers = (~certain_finite) | (target <= certain_depth + tolerance)
    bracketed = possible_lower_covers & certain_upper_covers
    finite_interval = possible_finite & certain_finite & ordering
    widths = certain_depth[finite_interval] - possible_depth[finite_interval]

    displacement = _motion_displacement(actor)
    moving = displacement > float(config["moving_max_displacement_m"])
    return {
        "scene_name": str(actor["scene_name"]),
        "track_id": str(actor["track_id"]),
        "hazardous": bool(actor["hazardous"]),
        "moving": bool(moving),
        "trajectory_max_displacement_m": displacement,
        "ray_count": int(len(target)),
        "certain_point_count": int(len(certain_surface)),
        "possible_point_count": int(len(possible_surface)),
        "ordering_violation_count": int(np.count_nonzero(ordering_violation)),
        "possible_lower_cover_count": int(np.count_nonzero(possible_lower_covers)),
        "certain_upper_cover_count": int(np.count_nonzero(certain_upper_covers)),
        "bracketed_target_count": int(np.count_nonzero(bracketed)),
        "finite_interval_count": int(np.count_nonzero(finite_interval)),
        "unbounded_upper_count": int(np.count_nonzero(~certain_finite)),
        "possible_early_count": int(np.count_nonzero(possible["early"])),
        "possible_hit_count": int(np.count_nonzero(possible["hit"])),
        "certain_early_count": int(np.count_nonzero(certain["early"])),
        "certain_hit_count": int(np.count_nonzero(certain["hit"])),
        "finite_width_median_m": float(np.median(widths)) if len(widths) else None,
        "finite_width_q90_m": float(np.quantile(widths, 0.90)) if len(widths) else None,
        "actor_state_retention": 1.0,
        "hazard_state_retention": 1.0,
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        total = lambda name: sum(int(row[name]) for row in selected)
        medians = [
            float(row["finite_width_median_m"])
            for row in selected
            if row["finite_width_median_m"] is not None
        ]
        q90s = [
            float(row["finite_width_q90_m"])
            for row in selected
            if row["finite_width_q90_m"] is not None
        ]
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "ordering_violation_count": total("ordering_violation_count"),
            "bracketed_target_rate": total("bracketed_target_count") / rays,
            "finite_interval_rate": total("finite_interval_count") / rays,
            "unbounded_upper_rate": total("unbounded_upper_count") / rays,
            "possible_early_rate": total("possible_early_count") / rays,
            "possible_hit_rate": total("possible_hit_count") / rays,
            "certain_early_rate": total("certain_early_count") / rays,
            "certain_hit_rate": total("certain_hit_count") / rays,
            "actor_mean_finite_width_median_m": float(np.mean(medians))
            if medians
            else None,
            "actor_mean_finite_width_q90_m": float(np.mean(q90s)) if q90s else None,
        }

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
        "moving": stratum([row for row in rows if bool(row["moving"])]),
        "quasi_static": stratum([row for row in rows if not bool(row["moving"])]),
    }


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
        raise RuntimeError("M30 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        model, base, standardizer, model_config, base_config = m22_runner._load_m8(
            config, device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_training_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        holdout = [
            actor
            for index, actor in enumerate(actors)
            if index % int(config["holdout_stride"]) == 0
        ]
        _write_json(
            run_dir / "status.json",
            {"status": "running", "phase": "return_interval"},
        )
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout):
                _, centers = m5_runner._move(base, actor, base_config)
                actor["m5_centers_t"] = centers.detach()
                children, _, _ = m7_runner._predict(model, actor, model_config)
                certain_surface = actor["anchors_t"].cpu().numpy()
                possible_surface = torch.cat(
                    [actor["anchors_t"], children], dim=0
                ).cpu().numpy()
                rows.append(
                    _actor_interval_row(
                        actor,
                        certain_surface,
                        possible_surface,
                        config["evaluation"],
                        device,
                    )
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout):
                    print(
                        json.dumps(
                            {
                                "stage": "m30_return_interval",
                                "progress": f"{index + 1}/{len(holdout)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = _summarize(rows)
        ordering_passed = all(
            int(group["ordering_violation_count"]) == 0
            for group in metrics.values()
        )
        summary = {
            "schema_version": "worldsim_v71.m30_evidential_return_interval.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "set_order_interval_supported"
            if ordering_passed
            else "set_order_interval_implementation_failed",
            "actor_count": len(rows),
            "pretrained_holdout_exposure": True,
            "certain_surface": "immutable_observed_anchors",
            "possible_surface": "immutable_anchors_plus_all_m8_generated_children",
            "interval_semantics": "d_possible_le_d_true_le_d_certain_conditional_on_completion_subset",
            "metrics": metrics,
            "ordering_passed": ordering_passed,
            "empirical_target_bracketing_is_not_a_coverage_guarantee": True,
            "unknown_collapsed_to_free_or_occupied": False,
            "training": False,
            "checkpoint_written": False,
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
        _write_jsonl(run_dir / "RETURN_INTERVAL_ROWS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "return_interval",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "m30",
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
