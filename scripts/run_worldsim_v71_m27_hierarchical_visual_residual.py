"""Fuse frozen coarse and fine visual carriers without changing 3D geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_worldsim_v71_m25_geometry_locked_attribute_optimization as m25


def _half_optical_depth(logits: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(logits, dtype=torch.float32)
    alpha = torch.sigmoid(values)
    half_alpha = 1.0 - torch.sqrt(1.0 - alpha)
    return torch.logit(half_alpha.clamp(1.0e-6, 1.0 - 1.0e-6)).numpy()


def _hierarchical_carrier(
    carrier: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(
        Path(config["m25_reference_run"]) / "OPTIMIZED_APPEARANCE_SIDECAR.npz",
        allow_pickle=False,
    ) as arrays:
        parent_dc = arrays["features_dc"].copy()
        parent_rest = arrays["features_rest"].copy()
        parent_opacity = arrays["opacity_logits"].copy()
    with np.load(
        Path(config["m26_reference_run"]) / "OPTIMIZED_APPEARANCE_SIDECAR.npz",
        allow_pickle=False,
    ) as arrays:
        child_centers = arrays["centers"].copy()
        child_scales = arrays["scales_xyz"].copy()
        child_quaternions = arrays["quaternions"].copy()
        child_dc = arrays["features_dc"].copy()
        child_rest = arrays["features_rest"].copy()
        child_opacity = arrays["opacity_logits"].copy()

    parent_centers = carrier["centers"].astype(np.float32)
    parent_scales = np.repeat(
        carrier["scales"].astype(np.float32)[:, None], 3, axis=1
    )
    parent_quaternions = np.zeros((len(parent_centers), 4), dtype=np.float32)
    parent_quaternions[:, 0] = 1.0
    fused = {
        "centers": np.concatenate([parent_centers, child_centers], axis=0),
        "scales_xyz": np.concatenate([parent_scales, child_scales], axis=0),
        "quaternions": np.concatenate(
            [parent_quaternions, child_quaternions], axis=0
        ),
        "features_dc": np.concatenate([parent_dc, child_dc], axis=0),
        "features_rest": np.concatenate([parent_rest, child_rest], axis=0),
        "opacity_logits": np.concatenate(
            [
                _half_optical_depth(parent_opacity),
                _half_optical_depth(child_opacity),
            ],
            axis=0,
        ),
    }
    metadata = {
        "representation": "two_level_coarse_parent_fine_surfel_visual_residual",
        "physical_carrier_count": int(len(parent_centers)),
        "coarse_visual_count": int(len(parent_centers)),
        "fine_visual_count": int(len(child_centers)),
        "total_visual_count": int(len(fused["centers"])),
        "coarse_attribute_source": "frozen_m25_optimized_sidecar",
        "fine_attribute_source": "frozen_m26_optimized_sidecar",
        "geometry_source": "frozen_m8_and_deterministic_m26_surface_frames",
        "branch_optical_depth_initialization": "equal_half_power_transmittance",
        "visual_geometry_trainable": False,
        "image_to_visual_geometry_gradient": False,
        "visual_geometry_in_physical_query": False,
    }
    return fused, metadata


def _compare_with_m25(
    summary: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    reference = json.loads(
        (Path(config["m25_reference_run"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    reference_rows = {
        (int(row["frame"]), int(row["camera"])): row
        for row in reference["rows"]
        if row["split"] == "heldout"
    }
    deltas = []
    for row in summary["rows"]:
        if row["split"] != "heldout":
            continue
        key = (int(row["frame"]), int(row["camera"]))
        delta = float(row["final_actor_psnr_db"]) - float(
            reference_rows[key]["final_actor_psnr_db"]
        )
        deltas.append({"frame": key[0], "camera": key[1], "delta_db": delta})
    values = np.asarray([row["delta_db"] for row in deltas], dtype=np.float64)
    comparison = {
        "reference": "m25_frozen_final",
        "reference_pooled_actor_psnr_db": float(
            reference["aggregate"]["heldout_final_actor_psnr_db"]
        ),
        "current_pooled_actor_psnr_db": float(
            summary["aggregate"]["heldout_final_actor_psnr_db"]
        ),
        "pooled_delta_db": float(
            summary["aggregate"]["heldout_final_actor_psnr_db"]
            - reference["aggregate"]["heldout_final_actor_psnr_db"]
        ),
        "per_view": deltas,
        "positive_view_count": int((values > 0.0).sum()),
        "median_view_delta_db": float(np.median(values)),
        "minimum_view_delta_db": float(values.min()),
        "maximum_view_delta_db": float(values.max()),
        "development_views_previously_exposed": True,
    }
    summary["m25_comparison"] = comparison
    summary["decisions"]["median_heldout_delta_vs_m25_positive"] = bool(
        comparison["median_view_delta_db"] > 0.0
    )
    summary["verdict"] = (
        config["success_verdict"]
        if all(summary["decisions"].values())
        else config["failure_verdict"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = m25.run(
        config_path,
        args.run_id,
        carrier_transform=_hierarchical_carrier,
    )
    summary = _compare_with_m25(summary, config)
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / args.run_id
    m25._write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
