"""Adapt target-free V6.4 point states to semantic Gaussians and BEV splats."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from motion_proj.worldsim_v61.occupancy import OCCUPIED


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _gaussian_kernel(sigma_cells: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma_cells**2))
    return kernel[None, None]


def _count_grid(indices: np.ndarray, selected: np.ndarray, shape_xy: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape_xy, dtype=np.float32)
    xy = np.asarray(indices[selected, :2], dtype=np.int64)
    np.add.at(grid, (xy[:, 0], xy[:, 1]), 1.0)
    return grid


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()

    source_root = runs_root / config["inputs"]["state_bake_run"]
    source_paths = sorted((source_root / "units").glob("*/*/PHYSICAL_STATE.npz"))
    expected_count = int(config["gates"]["expected_package_count"])
    if len(source_paths) != expected_count:
        raise RuntimeError(f"expected {expected_count} state packages, found {len(source_paths)}")

    grid_shape_xy = tuple(int(value) for value in config["render"]["grid_shape_xy"])
    opacity = float(config["gaussian"]["opacity"])
    scale_m = float(config["gaussian"]["isotropic_scale_m"])
    voxel_size_m = float(config["render"]["voxel_size_m"])
    sigma_cells = scale_m / voxel_size_m
    support_threshold = float(config["render"]["support_threshold"])
    c0_grids = []
    m0_grids = []
    payloads = []
    for source_path in source_paths:
        with np.load(source_path, allow_pickle=False) as source:
            indices = np.asarray(source["native_indices"], dtype=np.uint16)
            centers = np.asarray(source["centers_m"], dtype=np.float32)
            c0_state = np.asarray(source["c0_state"], dtype=np.uint8)
            m0_state = np.asarray(source["m0_state"], dtype=np.uint8)
        c0_selected = c0_state == OCCUPIED
        m0_selected = m0_state == OCCUPIED
        c0_grids.append(_count_grid(indices, c0_selected, grid_shape_xy))
        m0_grids.append(_count_grid(indices, m0_selected, grid_shape_xy))
        payloads.append((source_path, centers, c0_selected, m0_selected))

    device = torch.device("cuda")
    kernel = _gaussian_kernel(sigma_cells, device)
    padding = kernel.shape[-1] // 2
    optical_thickness = -math.log1p(-opacity)
    with torch.inference_mode():
        counts = torch.from_numpy(
            np.stack((np.stack(c0_grids), np.stack(m0_grids)), axis=1)
        ).to(device=device, dtype=torch.float32)
        density = 1.0 - torch.exp(
            -functional.conv2d(
                counts.reshape(-1, 1, *grid_shape_xy) * optical_thickness,
                kernel,
                padding=padding,
            )
        )
        density = density.reshape(len(payloads), 2, *grid_shape_xy).cpu().numpy()

    rows = []
    rows_path = run_dir / "GAUSSIAN_RUNTIME_ROWS.jsonl"
    for ordinal, ((source_path, centers, c0_selected, m0_selected), rendered) in enumerate(
        zip(payloads, density, strict=True)
    ):
        relative = source_path.relative_to(source_root / "units")
        scene, unit = relative.parts[:2]
        means = centers[m0_selected]
        count = means.shape[0]
        scales = np.full((count, 3), scale_m, dtype=np.float32)
        rotations = np.zeros((count, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.full(count, opacity, dtype=np.float32)
        semantics = np.full(count, OCCUPIED, dtype=np.uint8)
        c0_membership = c0_selected[m0_selected]
        unit_dir = run_dir / "units" / scene / unit
        unit_dir.mkdir(parents=True, exist_ok=False)
        output_path = unit_dir / "GAUSSIAN_STATE.npz"
        np.savez(
            output_path,
            means_m=means,
            scales_m=scales,
            rotations_wxyz=rotations,
            opacity=opacities,
            semantic_state=semantics,
            c0_membership=c0_membership,
            c0_bev_density=rendered[0].astype(np.float16),
            m0_bev_density=rendered[1].astype(np.float16),
        )
        c0_support = int(np.count_nonzero(rendered[0] >= support_threshold))
        m0_support = int(np.count_nonzero(rendered[1] >= support_threshold))
        row = {
            "ordinal": ordinal,
            "scene": scene,
            "unit": unit,
            "gaussian_count": count,
            "c0_gaussian_count": int(np.count_nonzero(c0_membership)),
            "additional_gaussian_count": int(count - np.count_nonzero(c0_membership)),
            "c0_bev_support_cells": c0_support,
            "m0_bev_support_cells": m0_support,
            "additional_bev_support_cells": m0_support - c0_support,
            "output": str(output_path.relative_to(run_dir)),
        }
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    gaussian_count = sum(int(row["gaussian_count"]) for row in rows)
    c0_gaussian_count = sum(int(row["c0_gaussian_count"]) for row in rows)
    c0_support = sum(int(row["c0_bev_support_cells"]) for row in rows)
    m0_support = sum(int(row["m0_bev_support_cells"]) for row in rows)
    output_bytes = sum(
        path.stat().st_size for path in (run_dir / "units").rglob("GAUSSIAN_STATE.npz")
    )
    gates = {
        "all_packages_rendered": len(rows) == expected_count,
        "positive_conditional_bev_support_gain": m0_support > c0_support,
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "package_count": len(rows),
        "m0_gaussian_count": gaussian_count,
        "c0_gaussian_count": c0_gaussian_count,
        "additional_gaussian_count": gaussian_count - c0_gaussian_count,
        "c0_bev_support_cells": c0_support,
        "m0_bev_support_cells": m0_support,
        "additional_bev_support_cells": m0_support - c0_support,
        "gaussian": {
            "isotropic_scale_m": scale_m,
            "identity_rotation": True,
            "opacity": opacity,
            "semantic_state": "OCCUPIED",
        },
        "render": {
            "mode": "probabilistic_bev_gaussian_superposition",
            "grid_shape_xy": list(grid_shape_xy),
            "voxel_size_m": voxel_size_m,
            "support_threshold": support_threshold,
        },
        "target_evidence_read": False,
        "model_access": False,
        "streetgs_checkpoint_access": False,
        "gate_results": gates,
        "output_bytes": output_bytes,
        "resources": {
            "gpu_used": True,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "additional_gaussian_count": gaussian_count - c0_gaussian_count,
        "additional_bev_support_cells": m0_support - c0_support,
        "gate_results": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
