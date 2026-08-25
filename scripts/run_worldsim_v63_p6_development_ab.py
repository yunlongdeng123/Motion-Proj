#!/usr/bin/env python3
"""Evaluate one preregistered stage of the WorldSim V6.3 P6 matched AB."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v62.cpsc_lite import CPSCLite
from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
)
from motion_proj.worldsim_v63.surfncc import SurfNCC, load_surface_unit
from scripts.run_worldsim_v63_p5_train import (
    _forward_with_global_context,
    _global_patch_context,
    _prepare_unit_batches,
    _tensor_batch,
)


TASK_ID = "WS-V63-P6-DEVELOPMENT-AB-01"
BASELINE_ARMS = ("B0", "B1", "B2")
SURFACE_ARMS = ("B3", "B4", "B5", "M0")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _cvar(values: np.ndarray, alpha: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if not flat.size:
        return 0.0
    count = max(1, int(math.ceil((1.0 - float(alpha)) * flat.size)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def _native_probabilities(unit: Any) -> np.ndarray:
    semantic = _softmax(np.asarray(unit.native_logits, dtype=np.float32))
    tristate = np.stack(
        (
            semantic[:, 0],
            semantic[:, 1:].sum(axis=1),
            np.zeros(semantic.shape[0], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    tristate[~unit.native_source_valid] = np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float32
    )
    return tristate


def _hard_project_probabilities(unit: Any, probabilities: np.ndarray) -> np.ndarray:
    projected = np.asarray(probabilities, dtype=np.float32).copy()
    contradiction = np.asarray(unit.arrays["method_contradiction"], dtype=bool)
    method = np.asarray(unit.method_class, dtype=np.int64)
    projected[contradiction] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    observed_free = (method == FREE_INDEX) & ~contradiction
    observed_occ = (method == OCCUPIED_INDEX) & ~contradiction
    projected[observed_free] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    projected[observed_occ] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return projected


def _load_b2(contract: dict[str, Any], device: torch.device) -> CPSCLite:
    model_config = yaml.safe_load(
        Path(contract["inputs"]["native_b2_config"]).read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        Path(contract["inputs"]["native_b2_checkpoint"]),
        map_location=device,
        weights_only=True,
    )
    model = CPSCLite(
        int(checkpoint["prior_feature_dimension"]),
        int(checkpoint["query_feature_dimension"]),
        hidden_width=int(model_config["model"]["hidden_width"]),
        decoder_layers=int(model_config["model"]["query_decoder_layers"]),
        residual_blocks=int(model_config["model"]["residual_blocks"]),
        projection_iterations=int(model_config["model"]["projection_iterations"]),
        dropout=float(model_config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _b2_probabilities(
    unit: Any,
    model: CPSCLite,
    shape: tuple[int, int, int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    semantic = _softmax(np.asarray(unit.native_logits, dtype=np.float32))
    prior_tristate = np.stack(
        (
            semantic[:, 0],
            semantic[:, 1:].sum(axis=1),
            np.zeros(semantic.shape[0], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    prior_tristate[~unit.native_source_valid] = np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float32
    )
    prior_features = np.concatenate(
        (
            np.asarray(unit.native_logits, dtype=np.float32),
            np.asarray(unit.native_entropy, dtype=np.float32)[:, None],
            prior_tristate,
            np.asarray(unit.native_source_valid, dtype=np.float32)[:, None],
            np.asarray(unit.native_bev, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    indices = np.asarray(unit.arrays["grid_indices"], dtype=np.int32)
    normalized = 2.0 * (
        (indices.astype(np.float32) + 0.5) / np.asarray(shape, dtype=np.float32)[None]
    ) - 1.0
    method = np.asarray(unit.method_class, dtype=np.int64)
    method_one_hot = np.eye(3, dtype=np.float32)[method]
    contradiction = np.asarray(unit.arrays["method_contradiction"], dtype=bool)
    actor = np.stack(
        (
            np.asarray(unit.arrays["actor_id"]) >= 0,
            np.asarray(unit.arrays["actor_current_support"], dtype=bool),
            np.asarray(unit.arrays["actor_swept_support"], dtype=bool),
        ),
        axis=1,
    ).astype(np.float32)
    query_features = np.concatenate(
        (
            normalized,
            method_one_hot,
            contradiction[:, None].astype(np.float32),
            actor,
            prior_tristate - method_one_hot,
        ),
        axis=1,
    ).astype(np.float32)
    result = np.empty((indices.shape[0], 3), dtype=np.float32)
    for start in range(0, indices.shape[0], int(batch_size)):
        stop = min(indices.shape[0], start + int(batch_size))
        output = model(
            torch.from_numpy(prior_features[start:stop]).to(device),
            torch.from_numpy(query_features[start:stop]).to(device),
            observed_free=torch.from_numpy(
                (method[start:stop] == FREE_INDEX) & ~contradiction[start:stop]
            ).to(device),
            observed_occupied=torch.from_numpy(
                (method[start:stop] == OCCUPIED_INDEX) & ~contradiction[start:stop]
            ).to(device),
            contradiction=torch.from_numpy(contradiction[start:stop]).to(device),
        )
        result[start:stop] = output["probabilities"].float().cpu().numpy()
    return result


def _load_surface_model(
    contract: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> SurfNCC:
    model_config = contract["model"]
    model = SurfNCC(
        int(model_config["input_dimension"]),
        hidden_dimension=int(model_config["hidden_dimension"]),
        neighbor_blocks=int(model_config["neighbor_blocks"]),
        patch_transformer_layers=int(model_config["patch_transformer_layers"]),
        attention_heads=int(model_config["attention_heads"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def _surface_probabilities(
    unit: Any,
    model: SurfNCC,
    contract: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    point_limit = int(contract["training"]["point_microbatch"])
    alpha = float(contract["risk"]["cvar_alpha"])
    prepared = _prepare_unit_batches(unit, point_limit)
    cached_patch_tokens, global_patch_proposal = _global_patch_context(
        model, unit, prepared, device
    )
    probabilities = np.empty((unit.target_class.shape[0], 3), dtype=np.float32)
    authority = np.empty(unit.target_class.shape[0], dtype=np.float32)
    for raw_batch, features, method, contradiction, authority_target in prepared:
        tensor_batch = _tensor_batch(
            raw_batch,
            features,
            method,
            contradiction,
            device,
            authority_target,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = _forward_with_global_context(
                model,
                raw_batch,
                tensor_batch,
                cached_patch_tokens,
                global_patch_proposal,
                alpha,
            )
        selected = np.asarray(raw_batch["selected"], dtype=np.int64)
        probabilities[selected] = output["probabilities"].float().cpu().numpy()
        authority[selected] = output["authority"].float().cpu().numpy()
    del prepared, cached_patch_tokens, global_patch_proposal
    return probabilities, authority


def _measure_unit(
    unit: Any,
    arm: str,
    probabilities: np.ndarray,
    decision: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    method = np.asarray(unit.method_class, dtype=np.int64)
    target = np.asarray(unit.target_class, dtype=np.int64)
    contradiction = np.asarray(unit.arrays["method_contradiction"], dtype=bool)
    eligible = (method == UNKNOWN_INDEX) & ~contradiction
    hidden_free = eligible & (target == FREE_INDEX)
    safe_occ = eligible & (target == OCCUPIED_INDEX)
    expected = np.full(method.shape, UNKNOWN_INDEX, dtype=np.int64)
    constrained = contradiction | (method != UNKNOWN_INDEX)
    expected[(method == FREE_INDEX) & ~contradiction] = FREE_INDEX
    expected[(method == OCCUPIED_INDEX) & ~contradiction] = OCCUPIED_INDEX
    hard_violations = int(np.count_nonzero(constrained & (decision != expected)))
    occ_confidence = np.asarray(probabilities[:, OCCUPIED_INDEX], dtype=np.float32)
    final_occ_confidence = np.where(
        decision == OCCUPIED_INDEX, occ_confidence, 0.0
    ).astype(np.float32)

    surface_values: dict[int, float] = {}
    surface_to_proposal: dict[int, int] = {}
    surface_index = np.asarray(unit.arrays["surface_index"], dtype=np.int64)
    for surface in np.unique(surface_index[hidden_free]):
        selected = hidden_free & (surface_index == surface)
        surface_values[int(surface)] = _cvar(occ_confidence[selected], alpha)
        surface_to_proposal[int(surface)] = int(unit.proposal_index[np.flatnonzero(selected)[0]])
    proposal_surface_risk: dict[int, float] = {}
    for surface, value in surface_values.items():
        proposal = surface_to_proposal[surface]
        proposal_surface_risk[proposal] = max(
            proposal_surface_risk.get(proposal, 0.0), value
        )
    proposal_false_safe = []
    for proposal in np.unique(unit.proposal_index[hidden_free]):
        selected = hidden_free & (unit.proposal_index == proposal)
        proposal_false_safe.append(_cvar(final_occ_confidence[selected], alpha))

    actor_point = np.asarray(unit.arrays["actor_id"], dtype=np.int32) >= 0
    actor_total = actor_accepted = static_total = static_accepted = 0
    for proposal in range(len(unit.proposal_ids)):
        selected = unit.proposal_index == proposal
        is_actor = bool(actor_point[selected].any())
        accepted = bool(np.any(selected & (decision == OCCUPIED_INDEX)))
        if is_actor:
            actor_total += 1
            actor_accepted += int(accepted)
        else:
            static_total += 1
            static_accepted += int(accepted)

    source_valid = np.asarray(unit.native_source_valid, dtype=bool)
    return {
        "arm": arm,
        "scene": unit.scene,
        "target_frame": int(unit.target_frame),
        "point_count": int(decision.size),
        "hard_violations": hard_violations,
        "safe_occ_total": int(np.count_nonzero(safe_occ)),
        "safe_occ_retained": int(np.count_nonzero(safe_occ & (decision == OCCUPIED_INDEX))),
        "source_valid_count": int(np.count_nonzero(source_valid)),
        "source_valid_unknown": int(np.count_nonzero(source_valid & (decision == UNKNOWN_INDEX))),
        "unknown_count": int(np.count_nonzero(decision == UNKNOWN_INDEX)),
        "emitted_occ_count": int(np.count_nonzero(decision == OCCUPIED_INDEX)),
        "correct_count": int(np.count_nonzero(decision == target)),
        "accepted_case": bool(np.any(decision == OCCUPIED_INDEX)),
        "actor_proposal_total": actor_total,
        "actor_proposal_accepted": actor_accepted,
        "static_proposal_total": static_total,
        "static_proposal_accepted": static_accepted,
        "surface_risk_sum": float(sum(proposal_surface_risk.values())),
        "surface_risk_count": len(proposal_surface_risk),
        "proposal_false_safe_sum": float(sum(proposal_false_safe)),
        "proposal_false_safe_count": len(proposal_false_safe),
    }


def _reduce(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_keys = (
        "point_count",
        "hard_violations",
        "safe_occ_total",
        "safe_occ_retained",
        "source_valid_count",
        "source_valid_unknown",
        "unknown_count",
        "emitted_occ_count",
        "correct_count",
        "actor_proposal_total",
        "actor_proposal_accepted",
        "static_proposal_total",
        "static_proposal_accepted",
        "surface_risk_count",
        "proposal_false_safe_count",
    )
    sums = {key: sum(int(row[key]) for row in rows) for key in count_keys}
    surface_sum = sum(float(row["surface_risk_sum"]) for row in rows)
    false_safe_sum = sum(float(row["proposal_false_safe_sum"]) for row in rows)
    return {
        **sums,
        "unit_count": len(rows),
        "accepted_case_count": sum(bool(row["accepted_case"]) for row in rows),
        "safe_occ_retention": sums["safe_occ_retained"] / max(1, sums["safe_occ_total"]),
        "source_valid_unknown_fraction": sums["source_valid_unknown"]
        / max(1, sums["source_valid_count"]),
        "unknown_fraction": sums["unknown_count"] / max(1, sums["point_count"]),
        "emitted_occ_coverage": sums["emitted_occ_count"] / max(1, sums["point_count"]),
        "accepted_case_coverage": sum(bool(row["accepted_case"]) for row in rows)
        / max(1, len(rows)),
        "target_accuracy_secondary": sums["correct_count"] / max(1, sums["point_count"]),
        "actor_proposal_coverage": sums["actor_proposal_accepted"]
        / max(1, sums["actor_proposal_total"]),
        "static_proposal_coverage": sums["static_proposal_accepted"]
        / max(1, sums["static_proposal_total"]),
        "common_surface_free_conflict_cvar": surface_sum
        / max(1, sums["surface_risk_count"]),
        "proposal_false_safe_surrogate": false_safe_sum
        / max(1, sums["proposal_false_safe_count"]),
    }


def _metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    scenes = sorted({str(row["scene"]) for row in selected})
    return {
        "pooled": _reduce(selected),
        "scenes": {
            scene: _reduce([row for row in selected if row["scene"] == scene])
            for scene in scenes
        },
    }


def _load_reference_metrics(paths: list[Path]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for path in paths:
        summary = json.loads((path / "P6_EVAL_SUMMARY.json").read_text(encoding="utf-8"))
        output.update(summary["metrics"])
    return output


def _anti_trivial(
    current: dict[str, Any], b2: dict[str, Any], gates: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    area_ratio = current["emitted_occ_count"] / max(1, b2["emitted_occ_count"])
    checks = {
        "hard_violations": current["hard_violations"] <= int(gates["hard_violations_max"]),
        "safe_occ_retention": current["safe_occ_retention"]
        >= float(gates["safe_occ_retention_min"]),
        "source_valid_unknown": current["source_valid_unknown_fraction"]
        <= float(gates["source_valid_unknown_max"]),
        "accepted_case_coverage": current["accepted_case_coverage"]
        >= float(gates["accepted_case_coverage_min"]),
        "accepted_surface_area": area_ratio
        >= float(gates["accepted_surface_area_vs_native_b2_min_ratio"]),
        "actor_coverage": current["actor_proposal_accepted"]
        >= int(gates["actor_accepted_proposal_count_min"]),
        "static_coverage": current["static_proposal_accepted"]
        >= int(gates["static_accepted_proposal_count_min"]),
    }
    return all(checks.values()), {"checks": checks, "accepted_surface_area_ratio": area_ratio}


def _relative_improvement(baseline: float, current: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0 if abs(float(current)) <= 1e-12 else -1.0
    return (float(baseline) - float(current)) / abs(float(baseline))


def _stage_gate(
    arm: str,
    current: dict[str, Any],
    references: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any] | None:
    if arm in BASELINE_ARMS:
        return None
    if "B2" not in references:
        raise ValueError("surface-arm evaluation requires a B2 reference")
    scene_rows = {}
    for scene, metrics in current["scenes"].items():
        feasible, anti = _anti_trivial(metrics, references["B2"]["scenes"][scene], gates)
        row: dict[str, Any] = {"anti_trivial_feasible": feasible, **anti}
        if arm == "B3":
            improvement = _relative_improvement(
                references["B2"]["scenes"][scene]["common_surface_free_conflict_cvar"],
                metrics["common_surface_free_conflict_cvar"],
            )
            row.update(
                {
                    "relative_improvement": improvement,
                    "relative_gate_passed": improvement
                    >= float(gates["b3_common_tail_relative_improvement_vs_b2_min"]),
                }
            )
        elif arm == "B4":
            row.update({"relative_improvement": None, "relative_gate_passed": True})
        elif arm == "B5":
            candidates = []
            for comparator in ("B3", "B4"):
                if comparator not in references:
                    continue
                comparator_metrics = references[comparator]["scenes"][scene]
                comparator_feasible, _ = _anti_trivial(
                    comparator_metrics, references["B2"]["scenes"][scene], gates
                )
                if comparator_feasible:
                    candidates.append((comparator, comparator_metrics))
            if not candidates:
                raise ValueError(f"no feasible B3/B4 comparator for {scene}")
            comparator, best = min(
                candidates,
                key=lambda item: item[1]["common_surface_free_conflict_cvar"],
            )
            improvement = _relative_improvement(
                best["common_surface_free_conflict_cvar"],
                metrics["common_surface_free_conflict_cvar"],
            )
            row.update(
                {
                    "comparator": comparator,
                    "relative_improvement": improvement,
                    "relative_gate_passed": improvement
                    >= float(
                        gates[
                            "b5_common_tail_relative_improvement_vs_best_feasible_b3_b4_min"
                        ]
                    ),
                }
            )
        elif arm == "M0":
            if "B5" not in references:
                raise ValueError("M0 evaluation requires a B5 reference")
            improvement = _relative_improvement(
                references["B5"]["scenes"][scene]["proposal_false_safe_surrogate"],
                metrics["proposal_false_safe_surrogate"],
            )
            row.update(
                {
                    "relative_improvement": improvement,
                    "relative_gate_passed": improvement
                    >= float(gates["m0_false_safe_surrogate_relative_improvement_vs_b5_min"]),
                }
            )
        scene_rows[scene] = row
    supporting = sum(
        row["anti_trivial_feasible"] and row["relative_gate_passed"]
        for row in scene_rows.values()
    )
    return {
        "arm": arm,
        "scene_results": scene_rows,
        "supporting_scene_count": supporting,
        "required_scene_count": int(gates["independent_selection_scenes_min"]),
        "stage_passed": supporting >= int(gates["independent_selection_scenes_min"]),
    }


def run(
    config_path: Path,
    repo_root: Path,
    run_dir: Path,
    arms: tuple[str, ...],
    checkpoint_path: Path | None,
    reference_runs: list[Path],
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).strip():
        raise RuntimeError("P6 formal requires clean source")
    contract = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if contract["task_id"] != TASK_ID:
        raise ValueError("P6 task identity drift")
    if not arms or any(arm not in BASELINE_ARMS + SURFACE_ARMS for arm in arms):
        raise ValueError("unsupported P6 arm set")
    if any(arm in SURFACE_ARMS for arm in arms) and len(arms) != 1:
        raise ValueError("surface arms must be evaluated one stage at a time")
    disk_probe = run_dir.parent
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    if shutil.disk_usage(disk_probe).free / 1024**3 < float(
        contract["resources"]["minimum_disk_free_gib"]
    ):
        raise RuntimeError("insufficient disk before P6")
    if not torch.cuda.is_available():
        raise RuntimeError("P6 requires CUDA")
    run_dir.mkdir(parents=True)

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device(f"cuda:{int(contract['resources']['gpu'])}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    b2_model = _load_b2(contract, device) if "B2" in arms else None
    surface_model = None
    surface_arm = arms[0] if arms[0] in SURFACE_ARMS else None
    if surface_arm is not None:
        if checkpoint_path is None:
            if surface_arm != "M0":
                raise ValueError(f"{surface_arm} requires an explicit checkpoint")
            checkpoint_path = Path(contract["inputs"]["m0_checkpoint"])
        surface_model = _load_surface_model(contract, checkpoint_path, device)

    rows: list[dict[str, Any]] = []
    surface_run = Path(contract["inputs"]["p3_surface_run"])
    native_run = Path(contract["inputs"]["p2_native_run"])
    shape = tuple(int(value) for value in contract["target_grid"]["shape"])
    alpha = float(contract["risk"]["cvar_alpha"])
    for scene in contract["cohort"]["selection_scenes"]:
        for target_frame in contract["cohort"]["targets"]:
            unit = load_surface_unit(surface_run, native_run, scene, int(target_frame))
            native = None
            if "B0" in arms or "B1" in arms:
                native = _native_probabilities(unit)
            if "B0" in arms:
                rows.append(
                    _measure_unit(unit, "B0", native, native.argmax(axis=1), alpha)
                )
            if "B1" in arms:
                projected = _hard_project_probabilities(unit, native)
                rows.append(
                    _measure_unit(
                        unit, "B1", projected, projected.argmax(axis=1), alpha
                    )
                )
            if "B2" in arms:
                probabilities = _b2_probabilities(
                    unit,
                    b2_model,
                    shape,
                    int(contract["resources"]["b2_query_batch_size"]),
                    device,
                )
                rows.append(
                    _measure_unit(
                        unit, "B2", probabilities, probabilities.argmax(axis=1), alpha
                    )
                )
            if surface_arm is not None:
                probabilities, authority = _surface_probabilities(
                    unit, surface_model, contract, device
                )
                decision = probabilities.argmax(axis=1)
                if surface_arm == "M0":
                    contradiction = np.asarray(
                        unit.arrays["method_contradiction"], dtype=bool
                    )
                    eligible = (unit.method_class == UNKNOWN_INDEX) & ~contradiction
                    veto = (
                        eligible
                        & (decision == OCCUPIED_INDEX)
                        & (
                            authority
                            < float(contract["risk"]["authority_threshold"])
                        )
                    )
                    decision[veto] = UNKNOWN_INDEX
                rows.append(
                    _measure_unit(
                        unit, surface_arm, probabilities, decision, alpha
                    )
                )
            del unit
            gc.collect()

    metrics = {arm: _metrics(rows, arm) for arm in arms}
    references = _load_reference_metrics(reference_runs)
    stage_gate = (
        _stage_gate(
            surface_arm,
            metrics[surface_arm],
            references,
            contract["gates"],
        )
        if surface_arm is not None
        else None
    )
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    summary = {
        "schema_version": "worldsim_v63.p6_development_ab_summary.v1",
        "task_id": TASK_ID,
        "arms": list(arms),
        "metrics": metrics,
        "stage_gate": stage_gate,
        "passed_execution_capability": peak_gib
        <= float(contract["resources"]["gpu_peak_gib_max"]),
        "peak_gpu_gib": peak_gib,
        "wall_seconds": time.monotonic() - started,
        "selection_scene_count": len(contract["cohort"]["selection_scenes"]),
        "selection_unit_count": len(contract["cohort"]["selection_scenes"])
        * len(contract["cohort"]["targets"]),
        "p6_quality_read": True,
        "threshold_fitted": False,
        "legacy_quality_read": False,
        "calibration_quality_read": False,
        "confirmation_read": False,
        "exact_once_test_read": False,
    }
    _write_jsonl(run_dir / "P6_PER_UNIT.jsonl", rows)
    _write_json(run_dir / "P6_EVAL_SUMMARY.json", summary)
    _write_json(
        run_dir / "P6_EVAL_MANIFEST.json",
        {
            "task_id": TASK_ID,
            "arms": list(arms),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "reference_runs": [str(path) for path in reference_runs],
            "source_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo_root, text=True
            ).strip(),
            "identity_policy": "semantic_paths_task_run_git_history_no_artifact_hash",
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-run", action="append", type=Path, default=[])
    args = parser.parse_args()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    summary = run(
        args.config.resolve(),
        args.repo_root.resolve(),
        args.run_dir.resolve(),
        arms,
        args.checkpoint.resolve() if args.checkpoint else None,
        [path.resolve() for path in args.reference_run],
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
