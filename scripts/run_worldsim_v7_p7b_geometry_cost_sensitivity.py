"""Audit a deterministic geometry-to-task-cost sensitivity bound on GPU."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q50, q90, q99 = np.quantile(values.astype(np.float64), (0.5, 0.9, 0.99))
    return {"q50": float(q50), "q90": float(q90), "q99": float(q99)}


def _summarize(
    mask: np.ndarray,
    baseline: np.ndarray,
    perturbed: np.ndarray,
    shift: np.ndarray,
    bound: np.ndarray,
    tightness: np.ndarray,
    clip_change: np.ndarray,
    sign_change: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"row_count": 0}
    local_shift = shift[mask]
    local_bound = bound[mask]
    excess = local_shift - local_bound
    return {
        "row_count": count,
        "baseline_cost_mean": float(np.mean(baseline[mask])),
        "baseline_cost_quantiles": _quantiles(baseline[mask]),
        "perturbed_cost_mean": float(np.mean(perturbed[mask])),
        "perturbed_cost_quantiles": _quantiles(perturbed[mask]),
        "absolute_cost_shift_mean": float(np.mean(local_shift)),
        "absolute_cost_shift_quantiles": _quantiles(local_shift),
        "local_bound_mean": float(np.mean(local_bound)),
        "local_bound_quantiles": _quantiles(local_bound),
        "bound_tightness_quantiles": _quantiles(tightness[mask]),
        "bound_violation_count": int(np.count_nonzero(excess > tolerance)),
        "maximum_shift_minus_bound": float(np.max(excess)),
        "denominator_clip_state_change_fraction": float(np.mean(clip_change[mask])),
        "signed_clearance_crossing_fraction": float(np.mean(sign_change[mask])),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    try:
        with np.load(str(config["source_rows"]), allow_pickle=False) as archive:
            error_vector = archive["actor_position_error_vector_ego_profile_m"].astype(np.float32)
            boundary_normal = archive["query_boundary_normal_ego_profile"].astype(np.float32)
            predicted_separation = archive["predicted_separation_profile_m"].astype(np.float32)
            interaction_radius = archive["occupancy_interaction_radius_m"].astype(np.float32)
            scene_index = archive["scene_index"].astype(np.int64)
            actor_id = archive["actor_id"].astype(np.int64)

        actor_rows = [
            row for row in _read_jsonl(Path(config["p5b_actor_rows"]))
            if row["p4_role"] == "test"
        ]
        all_keys = np.asarray(
            [(int(row["scene_index"]) << 32) | int(row["v67_actor_id"]) for row in actor_rows],
            dtype=np.int64,
        )
        selected_keys = np.asarray(
            [
                (int(row["scene_index"]) << 32) | int(row["v67_actor_id"])
                for row in actor_rows
                if bool(row["factorized_selected"])
            ],
            dtype=np.int64,
        )
        source_keys = (scene_index << 32) | actor_id
        group_masks = {
            "all_source": np.ones(len(source_keys), dtype=bool),
            "p5_test_all": np.isin(source_keys, all_keys),
            "p5_test_selected": np.isin(source_keys, selected_keys),
            "p5_test_abstained": np.isin(source_keys, all_keys) & ~np.isin(source_keys, selected_keys),
        }

        epsilon = float(config["clearance_floor_m"])
        tolerance = float(config["bound_tolerance"])
        chunk_size = int(config["chunk_size"])
        device = torch.device("cuda")
        dtype = torch.float64 if config["compute_dtype"] == "float64" else torch.float32
        torch.cuda.reset_peak_memory_stats(device)
        perturbation_results: dict[str, Any] = {}
        total_violations = 0
        for signed_delta in config["signed_clearance_perturbations_m"]:
            delta = float(signed_delta)
            baseline_parts: list[np.ndarray] = []
            perturbed_parts: list[np.ndarray] = []
            shift_parts: list[np.ndarray] = []
            bound_parts: list[np.ndarray] = []
            tightness_parts: list[np.ndarray] = []
            clip_change_parts: list[np.ndarray] = []
            sign_change_parts: list[np.ndarray] = []
            for start in range(0, len(scene_index), chunk_size):
                stop = min(start + chunk_size, len(scene_index))
                error = torch.from_numpy(error_vector[start:stop]).to(device=device, dtype=dtype)
                normal = torch.from_numpy(boundary_normal[start:stop]).to(device=device, dtype=dtype)
                separation = torch.from_numpy(predicted_separation[start:stop]).to(
                    device=device, dtype=dtype
                )
                radius = torch.from_numpy(interaction_radius[start:stop]).to(
                    device=device, dtype=dtype
                )[:, None]
                projection = torch.abs(torch.sum(error * normal, dim=2))
                clearance = separation - radius
                perturbed_clearance = clearance + delta
                denominator = torch.clamp(clearance, min=epsilon)
                perturbed_denominator = torch.clamp(perturbed_clearance, min=epsilon)
                baseline = torch.max(projection / denominator, dim=1).values
                perturbed = torch.max(projection / perturbed_denominator, dim=1).values
                shift = torch.abs(perturbed - baseline)
                bound = torch.max(
                    projection * abs(delta) / (denominator * perturbed_denominator), dim=1
                ).values
                tightness = torch.where(bound > 0.0, shift / bound, torch.zeros_like(bound))
                clip_change = torch.any(
                    (clearance <= epsilon) != (perturbed_clearance <= epsilon), dim=1
                )
                sign_change = torch.any(
                    (clearance <= 0.0) != (perturbed_clearance <= 0.0), dim=1
                )
                baseline_parts.append(baseline.cpu().numpy())
                perturbed_parts.append(perturbed.cpu().numpy())
                shift_parts.append(shift.cpu().numpy())
                bound_parts.append(bound.cpu().numpy())
                tightness_parts.append(tightness.cpu().numpy())
                clip_change_parts.append(clip_change.cpu().numpy())
                sign_change_parts.append(sign_change.cpu().numpy())

            baseline_np = np.concatenate(baseline_parts)
            perturbed_np = np.concatenate(perturbed_parts)
            shift_np = np.concatenate(shift_parts)
            bound_np = np.concatenate(bound_parts)
            tightness_np = np.concatenate(tightness_parts)
            clip_change_np = np.concatenate(clip_change_parts)
            sign_change_np = np.concatenate(sign_change_parts)
            groups = {
                name: _summarize(
                    group_masks[name], baseline_np, perturbed_np, shift_np, bound_np,
                    tightness_np, clip_change_np, sign_change_np, tolerance,
                )
                for name in config["fixed_groups"]
            }
            total_violations += sum(int(group.get("bound_violation_count", 0)) for group in groups.values())
            perturbation_results[f"{delta:+.2f}"] = {"signed_delta_m": delta, "groups": groups}

        summary = {
            "schema_version": "worldsim_v7.p7b_geometry_cost_sensitivity.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "supported_deterministic_geometry_cost_bound"
                if total_violations == 0
                else "violated_deterministic_geometry_cost_bound"
            ),
            "claim_boundary": config["claim_boundary"],
            "clearance_floor_m": epsilon,
            "compute_dtype": config["compute_dtype"],
            "source_row_count": int(len(scene_index)),
            "p5_test_actor_count": len(actor_rows),
            "p5_selected_actor_count": int(sum(bool(row["factorized_selected"]) for row in actor_rows)),
            "total_bound_violation_count_across_reported_groups": total_violations,
            "perturbations": perturbation_results,
            "resources": {
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), **summary}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
