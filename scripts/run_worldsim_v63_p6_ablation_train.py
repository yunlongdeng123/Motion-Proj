#!/usr/bin/env python3
"""Train one matched P6 surface-risk ablation with frozen rate constraints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v62.projection import FREE_INDEX, OCCUPIED_INDEX, UNKNOWN_INDEX
from motion_proj.worldsim_v63.surfncc import (
    POINT_FEATURE_DIMENSION,
    SurfNCC,
    compute_surfncc_losses,
    cvar_tail,
    load_surface_unit,
    plan_packed_structural_dropout,
    structural_dropout_records,
)
from scripts.run_worldsim_v63_p5_train import (
    _evaluate_selection,
    _forward_with_global_context,
    _global_patch_context,
    _global_rank_loss,
    _hard_violations,
    _prepare_unit_batches,
    _tensor_batch,
    _unit_rank_pairs,
)


TASK_ID = "WS-V63-P6-DEVELOPMENT-AB-01"
ARM_HYPOTHESES = {
    "B3": "WS-V63-H-P6-001",
    "B4": "WS-V63-H-P6-002",
    "B5": "WS-V63-H-P6-002",
}
CONSTRAINTS = (
    "safe_occ_retention",
    "emitted_occ_coverage",
    "non_unknown_coverage",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _arm_config(contract: dict[str, Any], arm_id: str) -> dict[str, Any]:
    if contract["task_id"] != TASK_ID:
        raise ValueError("P6 task identity drift")
    if arm_id not in ARM_HYPOTHESES:
        raise ValueError(f"unsupported P6 training arm: {arm_id}")
    arm = contract["arms"][arm_id]
    if arm["training"] != "independent_matched_ablation":
        raise ValueError(f"{arm_id} is not a trainable matched ablation")
    training = contract["training"]
    gates = contract["gates"]
    return {
        "task_id": TASK_ID,
        "hypothesis_id": ARM_HYPOTHESES[arm_id],
        "arm_id": arm_id,
        "seed": int(contract["seed"]),
        "inputs": {
            "p5_train_run": contract["inputs"]["p5_train_run"],
            "p3_surface_run": contract["inputs"]["p3_surface_run"],
            "p2_native_run": contract["inputs"]["p2_native_run"],
        },
        "cohort": contract["cohort"],
        "model": contract["model"],
        "risk": {
            "cvar_alpha": float(contract["risk"]["cvar_alpha"]),
            "authority_threshold": float(contract["risk"]["authority_threshold"]),
            "authority_enabled": False,
            "training_aggregator": str(arm["risk_training_aggregator"]),
        },
        "training": {
            key: training[key]
            for key in (
                "optimizer",
                "learning_rate",
                "weight_decay",
                "precision",
                "amp_initial_scale",
                "attention_backend",
                "cublas_workspace_config",
                "point_microbatch",
                "gradient_accumulation",
                "max_epochs",
                "min_epochs",
                "patience",
            )
        },
        "dropout": {
            "seed": int(training["structural_dropout_seed"]),
            "one_applicable_family_per_proposal": True,
            "support_fraction": float(training["structural_dropout_support_fraction"]),
            "families": list(training["structural_dropout_families"]),
        },
        "base_loss_weights": training["base_loss_weights"],
        "constraints": training["constraints"],
        "selection": {
            "hard_violations_max": int(gates["hard_violations_max"]),
            "safe_occ_retention_min": float(gates["safe_occ_retention_min"]),
            "emitted_occ_coverage_min": float(training["constraints"]["emitted_occ_coverage_min"]),
            "source_valid_unknown_max": float(gates["source_valid_unknown_max"]),
        },
        "resources": contract["resources"],
    }


def _authority_free_risk(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Remove q_AUTH from B3--B5 risk while retaining the frozen state decoder."""
    values = dict(outputs)
    point_risk = (
        outputs["probabilities"].float()[:, OCCUPIED_INDEX]
        * outputs["hidden_free"].float()
    )
    patch_index = batch["patch_index"]
    patch_count = int(patch_index.max().item()) + 1
    patch_cvar = torch.stack(
        [cvar_tail(point_risk[patch_index == patch], alpha) for patch in range(patch_count)]
    )
    patch_proposal = batch["patch_proposal_index"]
    proposal_count = int(patch_proposal.max().item()) + 1
    proposal_cvar = torch.stack(
        [
            patch_cvar[patch_proposal == proposal].max()
            for proposal in range(proposal_count)
        ]
    )
    values.update(
        {
            "point_risk": point_risk,
            "patch_cvar": patch_cvar,
            "proposal_cvar": proposal_cvar,
        }
    )
    return values


def _empty_exact_counts() -> dict[str, int]:
    return {
        "safe_occ_retained": 0,
        "safe_occ_total": 0,
        "emitted_occ": 0,
        "non_unknown": 0,
        "point_count": 0,
    }


def _add_exact_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def _exact_rates(counts: dict[str, int]) -> dict[str, float | None]:
    return {
        "safe_occ_retention": (
            counts["safe_occ_retained"] / counts["safe_occ_total"]
            if counts["safe_occ_total"]
            else None
        ),
        "emitted_occ_coverage": counts["emitted_occ"] / max(1, counts["point_count"]),
        "non_unknown_coverage": counts["non_unknown"] / max(1, counts["point_count"]),
    }


def _constraint_terms(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[
    dict[str, torch.Tensor | None],
    dict[str, torch.Tensor],
    dict[str, int],
]:
    probabilities = outputs["probabilities"].float()
    method = batch["method_class"]
    contradiction = batch["contradiction"]
    target = batch["target_class"]
    eligible = (method == UNKNOWN_INDEX) & ~contradiction
    safe_occ = eligible & (target == OCCUPIED_INDEX)
    authority = (
        outputs["authority"].float()
        if bool(config["risk"].get("authority_enabled", True))
        else torch.ones_like(outputs["authority"].float())
    )

    soft_emitted_occ = probabilities[:, OCCUPIED_INDEX]
    soft_emitted_occ = torch.where(
        eligible,
        probabilities[:, OCCUPIED_INDEX] * authority,
        soft_emitted_occ,
    )
    soft_non_unknown = 1.0 - probabilities[:, UNKNOWN_INDEX]
    soft_non_unknown = torch.where(
        eligible,
        probabilities[:, FREE_INDEX]
        + probabilities[:, OCCUPIED_INDEX] * authority,
        soft_non_unknown,
    )
    safe_rate = soft_emitted_occ[safe_occ].mean() if bool(safe_occ.any()) else None
    emitted_rate = soft_emitted_occ.mean()
    non_unknown_rate = soft_non_unknown.mean()
    zero = probabilities.sum() * 0.0
    thresholds = config["constraints"]
    violations = {
        "safe_occ_retention": (
            float(thresholds["safe_occ_retention_min"]) - safe_rate
            if safe_rate is not None
            else zero
        ),
        "emitted_occ_coverage": float(thresholds["emitted_occ_coverage_min"])
        - emitted_rate,
        "non_unknown_coverage": float(thresholds["non_unknown_coverage_min"])
        - non_unknown_rate,
    }

    projected = probabilities.argmax(dim=1)
    if bool(config["risk"].get("authority_enabled", True)):
        veto = (
            eligible
            & (projected == OCCUPIED_INDEX)
            & (authority < float(config["risk"]["authority_threshold"]))
        )
    else:
        veto = torch.zeros_like(eligible)
    decision = projected.clone()
    decision[veto] = UNKNOWN_INDEX
    exact_counts = {
        "safe_occ_retained": int((safe_occ & (decision == OCCUPIED_INDEX)).sum().item()),
        "safe_occ_total": int(safe_occ.sum().item()),
        "emitted_occ": int((decision == OCCUPIED_INDEX).sum().item()),
        "non_unknown": int((decision != UNKNOWN_INDEX).sum().item()),
        "point_count": int(decision.numel()),
    }
    return (
        {
            "safe_occ_retention": safe_rate,
            "emitted_occ_coverage": emitted_rate,
            "non_unknown_coverage": non_unknown_rate,
        },
        violations,
        exact_counts,
    )


def _update_multipliers(
    multipliers: dict[str, float],
    counts: dict[str, int],
    config: dict[str, Any],
) -> dict[str, float | None]:
    rates = _exact_rates(counts)
    thresholds = {
        "safe_occ_retention": float(config["constraints"]["safe_occ_retention_min"]),
        "emitted_occ_coverage": float(config["constraints"]["emitted_occ_coverage_min"]),
        "non_unknown_coverage": float(config["constraints"]["non_unknown_coverage_min"]),
    }
    step = float(config["constraints"]["dual_step_size"])
    for name in CONSTRAINTS:
        if rates[name] is None:
            continue
        multipliers[name] = max(
            0.0,
            float(multipliers[name]) + step * (thresholds[name] - float(rates[name])),
        )
    return rates


def _selection_contract(
    selection: dict[str, Any], config: dict[str, Any]
) -> tuple[bool, tuple[float, ...], tuple[float, ...]]:
    gates = config["selection"]
    hard = int(selection["hard_violations"])
    retention = float(selection["safe_occ_retention"])
    coverage = float(selection["emitted_occ_coverage"])
    unknown = float(selection["unknown_fraction"])
    primary = float(selection["hidden_free_tail"]) + float(
        selection["matched_rank_surrogate"]
    )
    retention_deficit = max(0.0, float(gates["safe_occ_retention_min"]) - retention)
    coverage_deficit = max(0.0, float(gates["emitted_occ_coverage_min"]) - coverage)
    unknown_deficit = max(0.0, unknown - float(gates["source_valid_unknown_max"]))
    feasible = (
        hard <= int(gates["hard_violations_max"])
        and retention_deficit == 0.0
        and coverage_deficit == 0.0
        and unknown_deficit == 0.0
    )
    progress_objective = (
        float(hard),
        retention_deficit,
        coverage_deficit,
        unknown_deficit,
        primary,
        -retention,
        -coverage,
        -float(selection["target_accuracy_secondary"]),
    )
    candidate_objective = (
        primary,
        -retention,
        -coverage,
        -float(selection["target_accuracy_secondary"]),
    )
    return feasible, progress_objective, candidate_objective


def _checkpoint_payload(
    model: SurfNCC,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    selection: dict[str, Any],
    multipliers: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "selection": selection,
        "multipliers": dict(multipliers),
        "input_dimension": POINT_FEATURE_DIMENSION,
    }


def run(
    config_path: Path, repo_root: Path, run_dir: Path, arm_id: str
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip():
        raise RuntimeError("P6 ablation formal requires clean source")
    contract = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _arm_config(contract, arm_id)
    checkpoint_stem = f"SURFNCC_{arm_id}"
    artifact_prefix = f"P6_{arm_id}"
    disk_probe = run_dir.parent
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    if shutil.disk_usage(disk_probe).free / 1024**3 < float(
        config["resources"]["minimum_disk_free_gib"]
    ):
        raise RuntimeError("insufficient disk before P6 ablation")
    if not torch.cuda.is_available():
        raise RuntimeError("P6 ablation requires CUDA")
    run_dir.mkdir(parents=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device(f"cuda:{int(config['resources']['gpu'])}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_config = config["model"]
    model = SurfNCC(
        int(model_config["input_dimension"]),
        hidden_dimension=int(model_config["hidden_dimension"]),
        neighbor_blocks=int(model_config["neighbor_blocks"]),
        patch_transformer_layers=int(model_config["patch_transformer_layers"]),
        attention_heads=int(model_config["attention_heads"]),
    ).to(device)
    p5_run = Path(config["inputs"]["p5_train_run"])
    p5_summary = json.loads((p5_run / "P5_TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
    warm_start = torch.load(p5_run / "SURFNCC_BEST.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(warm_start["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=float(config["training"]["amp_initial_scale"]),
    )
    multipliers = {
        name: float(config["constraints"]["initial_multipliers"][name])
        for name in CONSTRAINTS
    }

    alpha = float(config["risk"]["cvar_alpha"])
    accumulation = int(config["training"]["gradient_accumulation"])
    point_limit = int(config["training"]["point_microbatch"])
    max_epochs = int(config["training"]["max_epochs"])
    min_epochs = int(config["training"]["min_epochs"])
    patience = int(config["training"]["patience"])
    dropout_rng = np.random.default_rng(int(config["dropout"]["seed"]))
    order_rng = np.random.default_rng(seed)
    surface_run = Path(config["inputs"]["p3_surface_run"])
    native_run = Path(config["inputs"]["p2_native_run"])
    unit_specs = [
        (scene, int(target_frame))
        for scene in config["cohort"]["train_scenes"]
        for target_frame in config["cohort"]["targets"]
    ]
    evaluation_config = dict(config)
    evaluation_config["loss_weights"] = config["base_loss_weights"]

    best_progress_objective: tuple[float, ...] | None = None
    best_progress_epoch: int | None = None
    best_candidate_objective: tuple[float, ...] | None = None
    best_candidate_epoch: int | None = None
    epochs_since_improvement = 0
    epoch_rows = []
    optimizer_step = 0
    finite_training = True
    hard_violations = 0
    dropout_path = run_dir / "STRUCTURAL_DROPOUT.jsonl"
    with dropout_path.open("w", encoding="utf-8") as dropout_stream:
        for epoch in range(max_epochs):
            epoch_started = time.monotonic()
            model.train()
            order = np.arange(len(unit_specs))
            order_rng.shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            accumulation_step = 0
            dual_window = _empty_exact_counts()
            epoch_exact = _empty_exact_counts()
            proxy_values = {name: [] for name in CONSTRAINTS}
            base_total = 0.0
            primal_total = 0.0
            batch_count = 0
            point_count = 0
            rank_pair_count = 0
            for unit_ordinal in order:
                scene, target_frame = unit_specs[int(unit_ordinal)]
                unit = load_surface_unit(surface_run, native_run, scene, target_frame)
                dropout_plans = plan_packed_structural_dropout(
                    unit,
                    dropout_rng,
                    float(config["dropout"]["support_fraction"]),
                )
                for row in structural_dropout_records(dropout_plans):
                    row.update({"epoch": epoch, "scene": scene, "target_frame": target_frame})
                    dropout_stream.write(json.dumps(row, sort_keys=True) + "\n")
                prepared = _prepare_unit_batches(unit, point_limit, dropout_plans)
                unit_rank_pairs = _unit_rank_pairs(prepared)
                unit_rank_applied = False
                batch_order = np.arange(len(prepared))
                order_rng.shuffle(batch_order)
                global_context = None
                for batch_ordinal in batch_order:
                    if global_context is None:
                        global_context = _global_patch_context(model, unit, prepared, device)
                    cached_patch_tokens, global_patch_proposal = global_context
                    raw_batch, features, method, contradiction, authority_target = prepared[
                        int(batch_ordinal)
                    ]
                    tensor_batch = _tensor_batch(
                        raw_batch,
                        features,
                        method,
                        contradiction,
                        device,
                        authority_target,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = _forward_with_global_context(
                            model,
                            raw_batch,
                            tensor_batch,
                            cached_patch_tokens,
                            global_patch_proposal,
                            alpha,
                        )
                        outputs = _authority_free_risk(outputs, tensor_batch, alpha)
                        losses = compute_surfncc_losses(
                            outputs,
                            tensor_batch,
                            cvar_alpha=alpha,
                            weights=config["base_loss_weights"],
                            hidden_free_aggregator=str(
                                config["risk"]["training_aggregator"]
                            ),
                        )
                        if unit_rank_applied:
                            ranking = outputs["proposal_risk_head"].sum() * 0.0
                            pairs = 0
                        else:
                            ranking, pairs = _global_rank_loss(
                                model,
                                cached_patch_tokens,
                                global_patch_proposal,
                                unit_rank_pairs,
                                float(config["base_loss_weights"]["ranking_margin"]),
                            )
                            unit_rank_applied = True
                        base = losses["total"] + float(
                            config["base_loss_weights"]["proposal_rank"]
                        ) * ranking
                        proxy_rates, violations, exact_counts = _constraint_terms(
                            outputs, tensor_batch, config
                        )
                        primal = base + sum(
                            float(multipliers[name]) * violations[name]
                            for name in CONSTRAINTS
                        )
                        scaled = primal / float(accumulation)
                    finite_training = finite_training and bool(torch.isfinite(primal).item())
                    scaler.scale(scaled).backward()
                    hard_violations += _hard_violations(outputs, tensor_batch)
                    _add_exact_counts(dual_window, exact_counts)
                    _add_exact_counts(epoch_exact, exact_counts)
                    for name, rate in proxy_rates.items():
                        if rate is not None:
                            proxy_values[name].append(float(rate.detach()))
                    accumulation_step += 1
                    batch_count += 1
                    point_count += int(features.shape[0])
                    rank_pair_count += int(pairs)
                    base_total += float(base.detach())
                    primal_total += float(primal.detach())
                    if accumulation_step == accumulation:
                        scaler.unscale_(optimizer)
                        finite_training = finite_training and all(
                            bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                        _update_multipliers(multipliers, dual_window, config)
                        dual_window = _empty_exact_counts()
                        accumulation_step = 0
                        global_context = None
                del prepared, global_context, dropout_plans, unit_rank_pairs, unit
                gc.collect()
            if accumulation_step:
                scaler.unscale_(optimizer)
                finite_training = finite_training and all(
                    bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                _update_multipliers(multipliers, dual_window, config)

            selection = _evaluate_selection(model, evaluation_config, device)
            feasible, progress_objective, candidate_objective = _selection_contract(
                selection, config
            )
            progress_improved = (
                best_progress_objective is None
                or progress_objective < best_progress_objective
            )
            if progress_improved:
                best_progress_objective = progress_objective
                best_progress_epoch = epoch
                torch.save(
                    _checkpoint_payload(model, optimizer, epoch, selection, multipliers),
                    run_dir / f"{checkpoint_stem}_BEST_PROGRESS.pt",
                )
            candidate_improved = feasible and (
                best_candidate_objective is None
                or candidate_objective < best_candidate_objective
            )
            if candidate_improved:
                best_candidate_objective = candidate_objective
                best_candidate_epoch = epoch
                torch.save(
                    _checkpoint_payload(model, optimizer, epoch, selection, multipliers),
                    run_dir / f"{checkpoint_stem}_BEST_CANDIDATE.pt",
                )

            if best_candidate_epoch is None:
                monitor_improved = progress_improved
            else:
                monitor_improved = candidate_improved
            epochs_since_improvement = 0 if monitor_improved else epochs_since_improvement + 1
            selection_row = dict(selection)
            selection_row.update(
                {
                    "candidate_feasible": feasible,
                    "progress_objective": list(progress_objective),
                    "candidate_objective": list(candidate_objective) if feasible else None,
                }
            )
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "optimizer_steps_total": optimizer_step,
                    "train_batch_count": batch_count,
                    "train_point_count": point_count,
                    "train_rank_pair_count": rank_pair_count,
                    "mean_base_loss": base_total / max(1, batch_count),
                    "mean_primal_loss": primal_total / max(1, batch_count),
                    "mean_proxy_rates": {
                        name: (
                            float(np.mean(values)) if values else None
                        )
                        for name, values in proxy_values.items()
                    },
                    "train_exact_rates": _exact_rates(epoch_exact),
                    "multipliers": dict(multipliers),
                    "selection": selection_row,
                    "progress_improved": progress_improved,
                    "candidate_improved": candidate_improved,
                    "wall_seconds": time.monotonic() - epoch_started,
                }
            )
            _write_json(run_dir / f"{artifact_prefix}_EPOCHS.json", epoch_rows)
            if epoch + 1 >= min_epochs and epochs_since_improvement >= patience:
                break

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    candidate_promotable = (
        best_candidate_epoch is not None
        and (run_dir / f"{checkpoint_stem}_BEST_CANDIDATE.pt").is_file()
    )
    passed = (
        finite_training
        and hard_violations == 0
        and best_progress_epoch is not None
        and (run_dir / f"{checkpoint_stem}_BEST_PROGRESS.pt").is_file()
        and peak_gib <= float(config["resources"]["gpu_peak_gib_max"])
    )
    summary = {
        "schema_version": "worldsim_v63.p6_surface_ablation_train_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "arm_id": arm_id,
        "training_risk_aggregator": config["risk"]["training_aggregator"],
        "authority_enabled": False,
        "passed_training_capability": passed,
        "candidate_promotable": candidate_promotable,
        "epoch_count": len(epoch_rows),
        "optimizer_steps": optimizer_step,
        "warm_start_p5_best_epoch": int(p5_summary["best_epoch"]),
        "warm_start_model_only": True,
        "amp_initial_scale": float(config["training"]["amp_initial_scale"]),
        "amp_final_scale": float(scaler.get_scale()),
        "best_progress_epoch": best_progress_epoch,
        "best_progress_objective": (
            list(best_progress_objective) if best_progress_objective is not None else None
        ),
        "best_progress_selection": (
            epoch_rows[best_progress_epoch]["selection"]
            if best_progress_epoch is not None
            else None
        ),
        "best_candidate_epoch": best_candidate_epoch,
        "best_candidate_objective": (
            list(best_candidate_objective) if best_candidate_objective is not None else None
        ),
        "best_candidate_selection": (
            epoch_rows[best_candidate_epoch]["selection"]
            if best_candidate_epoch is not None
            else None
        ),
        "final_multipliers": dict(multipliers),
        "hard_violations": hard_violations,
        "finite_training": finite_training,
        "peak_gpu_gib": peak_gib,
        "wall_seconds": time.monotonic() - started,
        "train_scene_count": len(config["cohort"]["train_scenes"]),
        "selection_scene_count": len(config["cohort"]["selection_scenes"]),
        "calibration_quality_read": False,
        "confirmation_read": False,
        "exact_once_test_read": False,
    }
    _write_json(run_dir / f"{artifact_prefix}_TRAIN_SUMMARY.json", summary)
    _write_json(
        run_dir / f"{artifact_prefix}_TRAIN_MANIFEST.json",
        {
            "task_id": TASK_ID,
            "arm_id": arm_id,
            "training_risk_aggregator": config["risk"]["training_aggregator"],
            "authority_enabled": False,
            "source_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo_root, text=True
            ).strip(),
            "warm_start_checkpoint": str(p5_run / "SURFNCC_BEST.pt"),
            "train_scenes": config["cohort"]["train_scenes"],
            "selection_scenes": config["cohort"]["selection_scenes"],
            "identity_policy": "semantic_paths_task_run_git_history_no_artifact_hash",
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_HYPOTHESES), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.repo_root.resolve(),
                args.run_dir.resolve(),
                args.arm,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
