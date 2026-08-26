"""Bake frozen C0/M0 native selections into target-free physical-state packages."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage
import yaml

from motion_proj.worldsim_v61.occupancy import OCCUPIED, UNKNOWN
from motion_proj.worldsim_v64.native_voxel_uq import (
    _evidence_on_native_grid,
    _native_unit_dir,
    _unit_dirs,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _target_free_boundary(
    evidence_unit: Path,
    native_unit: Path,
    *,
    native_origin_m: np.ndarray,
    native_voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(evidence_unit / "METHOD_EVIDENCE.npz", allow_pickle=False) as source:
        method = {name: np.asarray(source[name]) for name in source.files}

    argmax = np.load(native_unit / "ARGMAX.npy", mmap_mode="r")
    native_shape = tuple(int(value) for value in argmax.shape)
    method_state, native_valid = _evidence_on_native_grid(
        method,
        native_shape=native_shape,
        native_origin_m=native_origin_m,
        native_voxel_size_m=native_voxel_size_m,
    )
    method_contradiction, contradiction_valid = _evidence_on_native_grid(
        {**method, "semantics": method["contradiction"]},
        native_shape=native_shape,
        native_origin_m=native_origin_m,
        native_voxel_size_m=native_voxel_size_m,
    )
    proposal_occupied = (np.asarray(argmax) != 0) | (method_state == OCCUPIED)
    structure = ndimage.generate_binary_structure(3, 1)
    boundary = proposal_occupied & ~ndimage.binary_erosion(
        proposal_occupied, structure=structure, border_value=0
    )
    eligible = (
        boundary
        & native_valid
        & contradiction_valid
        & (method_state == UNKNOWN)
        & ~method_contradiction.astype(bool)
    )
    indices = np.argwhere(eligible)
    if indices.shape[0] == 0:
        raise RuntimeError(f"target-free native boundary is empty: {evidence_unit}")

    logits_grid = np.load(native_unit / "NATIVE_LOGITS.npy", mmap_mode="r")
    bev_grid = np.load(native_unit / "BEV_LATENT.npy", mmap_mode="r")
    x, y, z = indices.T
    logits = np.asarray(logits_grid[x, y, z], dtype=np.float32)
    bev = np.asarray(bev_grid[x, y], dtype=np.float32)
    features = np.concatenate((logits, bev), axis=1)
    centers = native_origin_m + (indices.astype(np.float64) + 0.5) * native_voxel_size_m
    return indices, centers.astype(np.float32), features


def _compile_states(scores: np.ndarray, coverage: float) -> np.ndarray:
    order = np.argsort(np.asarray(scores), kind="stable")
    selected_count = max(1, int(np.floor(float(coverage) * order.size)))
    states = np.full(order.size, UNKNOWN, dtype=np.uint8)
    states[order[:selected_count]] = OCCUPIED
    return states


def _consume_package(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as payload:
        c0_state = np.asarray(payload["c0_state"], dtype=np.uint8)
        m0_state = np.asarray(payload["m0_state"], dtype=np.uint8)
        eligible_count = int(np.asarray(payload["centers_m"]).shape[0])
    c0_emitted = int(np.count_nonzero(c0_state == OCCUPIED))
    m0_emitted = int(np.count_nonzero(m0_state == OCCUPIED))
    return {
        "eligible_count": eligible_count,
        "c0_emitted_count": c0_emitted,
        "m0_emitted_count": m0_emitted,
        "additional_emitted_count": m0_emitted - c0_emitted,
        "m0_unknown_count": eligible_count - m0_emitted,
    }


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

    inputs = config["inputs"]
    evidence_root = runs_root / inputs["evidence_run"]
    native_root = runs_root / inputs["native_run"]
    model = joblib.load(runs_root / inputs["risk_run"] / inputs["risk_model_relative_path"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    scenes = [str(row["name"]) for row in config["scenes"]]
    strata = {str(row["name"]): str(row["stratum"]) for row in config["scenes"]}
    partition_by_scene = {scene: str(inputs["native_partition"]) for scene in scenes}
    global_coverage = float(config["policy"]["global_nominal_coverage"])
    conditional_coverages = {
        str(key): float(value)
        for key, value in config["policy"]["conditional_nominal_coverages"].items()
    }

    package_rows = []
    runtime_rows_path = run_dir / "RUNTIME_ROWS.jsonl"
    for scene in scenes:
        for evidence_unit in _unit_dirs(evidence_root, scene):
            native_unit = _native_unit_dir(
                native_root, scene, evidence_unit.name, partition_by_scene
            )
            indices, centers, features = _target_free_boundary(
                evidence_unit,
                native_unit,
                native_origin_m=origin,
                native_voxel_size_m=voxel_size,
            )
            logits = features[:, :17]
            scores = np.asarray(model.score(features, logits), dtype=np.float32)
            c0_state = _compile_states(scores, global_coverage)
            m0_coverage = conditional_coverages[strata[scene]]
            m0_state = _compile_states(scores, m0_coverage)

            package_dir = run_dir / "units" / scene / evidence_unit.name
            package_dir.mkdir(parents=True, exist_ok=False)
            package_path = package_dir / "PHYSICAL_STATE.npz"
            np.savez(
                package_path,
                native_indices=indices.astype(np.uint16),
                centers_m=centers,
                risk_score=scores,
                c0_state=c0_state,
                m0_state=m0_state,
                grid_origin_m=origin.astype(np.float32),
                voxel_size_m=np.asarray(voxel_size, dtype=np.float32),
            )
            consumed = _consume_package(package_path)
            row = {
                "scene": scene,
                "stratum": strata[scene],
                "unit": evidence_unit.name,
                "package": str(package_path.relative_to(run_dir)),
                "c0_nominal_coverage": global_coverage,
                "m0_nominal_coverage": m0_coverage,
                **consumed,
            }
            package_rows.append(row)
            _append_jsonl(runtime_rows_path, row)

    package_count = len(package_rows)
    eligible_total = sum(int(row["eligible_count"]) for row in package_rows)
    c0_total = sum(int(row["c0_emitted_count"]) for row in package_rows)
    m0_total = sum(int(row["m0_emitted_count"]) for row in package_rows)
    mean_c0_coverage = float(
        np.mean([row["c0_emitted_count"] / row["eligible_count"] for row in package_rows])
    )
    mean_m0_coverage = float(
        np.mean([row["m0_emitted_count"] / row["eligible_count"] for row in package_rows])
    )
    uplift = mean_m0_coverage - mean_c0_coverage
    additional_by_stratum = {
        stratum: sum(
            int(row["additional_emitted_count"])
            for row in package_rows
            if row["stratum"] == stratum
        )
        for stratum in sorted(set(strata.values()))
    }
    output_bytes = sum(
        path.stat().st_size for path in (run_dir / "units").rglob("PHYSICAL_STATE.npz")
    )
    gates = {
        "minimum_mean_coverage_uplift": uplift
        >= float(config["gates"]["minimum_mean_coverage_uplift"]),
        "runtime_packages_consumed_with_positive_additional_state": package_count
        == int(config["gates"]["expected_package_count"])
        and (m0_total - c0_total) > 0,
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "scene_count": len(scenes),
        "package_count": package_count,
        "eligible_voxel_count": eligible_total,
        "c0_emitted_voxel_count": c0_total,
        "m0_emitted_voxel_count": m0_total,
        "additional_emitted_voxel_count": m0_total - c0_total,
        "additional_emitted_by_stratum": additional_by_stratum,
        "c0_mean_realized_coverage": mean_c0_coverage,
        "m0_mean_realized_coverage": mean_m0_coverage,
        "conditional_coverage_uplift": uplift,
        "voxel_size_m": voxel_size,
        "additional_emitted_volume_m3": float((m0_total - c0_total) * voxel_size**3),
        "output_bytes": output_bytes,
        "target_evidence_read": False,
        "runtime_consumer_loaded_model_or_evidence": False,
        "model_refit": False,
        "policy_selection_during_run": False,
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
        "additional_emitted_voxel_count": m0_total - c0_total,
        "conditional_coverage_uplift": uplift,
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
