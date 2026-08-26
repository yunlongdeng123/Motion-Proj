"""Audit route-local hidden-FREE conflict of frozen C0/M0 emitted states."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED
from motion_proj.worldsim_v64.gaussian_route_consumer import _future_route_in_target_lidar
from motion_proj.worldsim_v64.native_voxel_uq import _evidence_on_native_grid


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(config_path: Path, runs_root: Path, processed_root: Path, run_id: str) -> dict[str, object]:
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

    inputs = config["inputs"]
    state_root = runs_root / inputs["state_bake_run"]
    evidence_root = runs_root / inputs["evidence_run"]
    source_paths = sorted((state_root / "units").glob("*/*/PHYSICAL_STATE.npz"))
    expected_count = int(config["gates"]["expected_case_count"])
    if len(source_paths) != expected_count:
        raise RuntimeError(f"expected {expected_count} state cases, found {len(source_paths)}")
    processed_indices = {
        str(row["name"]): int(row["processed_index"]) for row in config["scenes"]
    }
    route = config["route"]
    future_frame_count = int(route["future_frame_count"])
    corridor_radius_m = float(route["corridor_radius_m"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    shape = tuple(int(value) for value in config["native_grid"]["shape"])
    x = origin[0] + (np.arange(shape[0], dtype=np.float32) + 0.5) * voxel_size
    y = origin[1] + (np.arange(shape[1], dtype=np.float32) + 0.5) * voxel_size
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_xy = torch.from_numpy(np.stack((xx, yy), axis=-1)).to("cuda")

    rows = []
    rows_path = run_dir / "ROUTE_CONFLICT_ROWS.jsonl"
    with torch.inference_mode():
        for source_path in source_paths:
            relative = source_path.relative_to(state_root / "units")
            scene, unit = relative.parts[:2]
            target_frame = int(unit.removeprefix("f"))
            route_xy = _future_route_in_target_lidar(
                processed_root / f"{processed_indices[scene]:03d}",
                target_frame,
                future_frame_count,
            )
            route_tensor = torch.from_numpy(route_xy).to("cuda")
            corridor = (
                (grid_xy[None] - route_tensor[:, None, None]).square().sum(dim=-1).amin(dim=0)
                <= corridor_radius_m**2
            ).cpu().numpy()

            with np.load(source_path, allow_pickle=False) as source:
                indices = np.asarray(source["native_indices"], dtype=np.int64)
                c0_state = np.asarray(source["c0_state"], dtype=np.uint8)
                m0_state = np.asarray(source["m0_state"], dtype=np.uint8)
            target_path = evidence_root / "units" / scene / unit / "TARGET_EVIDENCE.npz"
            with np.load(target_path, allow_pickle=False) as source:
                target = {name: np.asarray(source[name]) for name in source.files}
            target_state, target_valid = _evidence_on_native_grid(
                target,
                native_shape=shape,
                native_origin_m=origin,
                native_voxel_size_m=voxel_size,
            )
            ix, iy, iz = indices.T
            in_route = corridor[ix, iy] & target_valid[ix, iy, iz]
            hidden_free = target_state[ix, iy, iz] == FREE
            c0_selected = in_route & (c0_state == OCCUPIED)
            m0_selected = in_route & (m0_state == OCCUPIED)
            c0_count = int(np.count_nonzero(c0_selected))
            m0_count = int(np.count_nonzero(m0_selected))
            c0_conflicts = int(np.count_nonzero(c0_selected & hidden_free))
            m0_conflicts = int(np.count_nonzero(m0_selected & hidden_free))
            row = {
                "scene": scene,
                "unit": unit,
                "target_frame": target_frame,
                "c0_route_emitted_voxel_count": c0_count,
                "m0_route_emitted_voxel_count": m0_count,
                "additional_route_emitted_voxel_count": m0_count - c0_count,
                "c0_route_hidden_free_conflict_count": c0_conflicts,
                "m0_route_hidden_free_conflict_count": m0_conflicts,
                "additional_route_hidden_free_conflict_count": m0_conflicts - c0_conflicts,
                "c0_route_hidden_free_conflict_rate": float(c0_conflicts / c0_count) if c0_count else None,
                "m0_route_hidden_free_conflict_rate": float(m0_conflicts / m0_count) if m0_count else None,
            }
            rows.append(row)
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    c0_count = sum(int(row["c0_route_emitted_voxel_count"]) for row in rows)
    m0_count = sum(int(row["m0_route_emitted_voxel_count"]) for row in rows)
    c0_conflicts = sum(int(row["c0_route_hidden_free_conflict_count"]) for row in rows)
    m0_conflicts = sum(int(row["m0_route_hidden_free_conflict_count"]) for row in rows)
    threshold = float(config["risk"]["maximum_m0_route_hidden_free_conflict_rate"])
    c0_rate = float(c0_conflicts / c0_count) if c0_count else 0.0
    m0_rate = float(m0_conflicts / m0_count) if m0_count else 0.0
    m0_case_failures = sum(
        row["m0_route_hidden_free_conflict_rate"] is not None
        and float(row["m0_route_hidden_free_conflict_rate"]) > threshold
        for row in rows
    )
    gates = {
        "positive_additional_route_emitted_state": m0_count > c0_count,
        "maximum_m0_route_hidden_free_conflict_rate": m0_rate <= threshold,
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "case_count": len(rows),
        "c0_route_emitted_voxel_count": c0_count,
        "m0_route_emitted_voxel_count": m0_count,
        "additional_route_emitted_voxel_count": m0_count - c0_count,
        "c0_route_hidden_free_conflict_count": c0_conflicts,
        "m0_route_hidden_free_conflict_count": m0_conflicts,
        "additional_route_hidden_free_conflict_count": m0_conflicts - c0_conflicts,
        "c0_route_hidden_free_conflict_rate": c0_rate,
        "m0_route_hidden_free_conflict_rate": m0_rate,
        "m0_case_failure_count_descriptive": m0_case_failures,
        "maximum_conflict_rate": threshold,
        "target_evidence_read": True,
        "model_refit": False,
        "policy_selection_during_run": False,
        "collision_ground_truth_read": False,
        "gate_results": gates,
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
        "additional_route_emitted_voxel_count": m0_count - c0_count,
        "m0_route_hidden_free_conflict_rate": m0_rate,
        "gate_results": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.runs_root.resolve(),
                args.processed_root.resolve(),
                args.run_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
