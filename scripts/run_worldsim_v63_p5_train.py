#!/usr/bin/env python3
"""Train the frozen SurfNCC model on Tier-D train scenes and select by Tier-D selection scenes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v62.projection import FREE_INDEX, OCCUPIED_INDEX, UNKNOWN_INDEX
from motion_proj.worldsim_v63.surfncc import (
    POINT_FEATURE_DIMENSION,
    SurfNCC,
    apply_packed_structural_dropout,
    compute_surfncc_losses,
    load_surface_unit,
    packed_unit_batches,
    plan_packed_structural_dropout,
    structural_dropout_records,
)


TASK_ID = "WS-V63-P5-SURFNCC-TRAIN-01"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tensor_batch(
    batch: dict[str, np.ndarray],
    features: np.ndarray,
    method: np.ndarray,
    contradiction: np.ndarray,
    device: torch.device,
    authority_target: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "point_features": torch.as_tensor(features, device=device),
        "edge_index": torch.as_tensor(batch["edge_index"], device=device, dtype=torch.long),
        "patch_index": torch.as_tensor(batch["patch_index"], device=device, dtype=torch.long),
        "patch_proposal_index": torch.as_tensor(
            batch["patch_proposal_index"], device=device, dtype=torch.long
        ),
        "point_proposal_index": torch.as_tensor(
            batch["point_proposal_index"], device=device, dtype=torch.long
        ),
        "proposal_actor": torch.as_tensor(
            batch["proposal_actor"], device=device, dtype=torch.bool
        ),
        "proposal_point_count": torch.as_tensor(
            batch["proposal_point_count"], device=device, dtype=torch.long
        ),
        "proposal_safe": torch.as_tensor(
            batch["proposal_safe"], device=device, dtype=torch.bool
        ),
        "proposal_unsafe": torch.as_tensor(
            batch["proposal_unsafe"], device=device, dtype=torch.bool
        ),
        "target_class": torch.as_tensor(batch["target_class"], device=device, dtype=torch.long),
        "method_class": torch.as_tensor(method, device=device, dtype=torch.long),
        "contradiction": torch.as_tensor(contradiction, device=device, dtype=torch.bool),
        "authority_target": torch.as_tensor(
            batch["authority_target"] if authority_target is None else authority_target,
            device=device,
            dtype=torch.bool,
        ),
    }


def _prepare_unit_batches(
    unit: Any,
    point_limit: int,
    dropout_plans: dict[int, dict[str, Any]] | None = None,
) -> list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    prepared = []
    for raw_batch in packed_unit_batches(unit, point_limit):
        if dropout_plans is None:
            prepared.append(
                (
                    raw_batch,
                    raw_batch["point_features"],
                    raw_batch["method_class"],
                    raw_batch["contradiction"],
                    raw_batch["authority_target"],
                )
            )
            continue
        features, method, contradiction, authority_target = (
            apply_packed_structural_dropout(unit, raw_batch, dropout_plans)
        )
        prepared.append(
            (raw_batch, features, method, contradiction, authority_target)
        )
    return prepared


def _global_patch_context(
    model: SurfNCC,
    unit: Any,
    prepared: list[
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize current full-unit patch tokens without retaining point graphs."""
    patch_ids = np.asarray(unit.arrays["patch_index"], dtype=np.int64)
    patch_count = int(patch_ids.max()) + 1
    proposal_by_patch = np.empty(patch_count, dtype=np.int64)
    proposal_by_patch[patch_ids] = unit.proposal_index
    global_tokens = None
    with torch.no_grad():
        for raw_batch, features, _, _, _ in prepared:
            point_features = torch.as_tensor(features, device=device)
            edge_index = torch.as_tensor(
                raw_batch["edge_index"], device=device, dtype=torch.long
            )
            patch_index = torch.as_tensor(
                raw_batch["patch_index"], device=device, dtype=torch.long
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, local_tokens = model.encode_surface_points(
                    point_features, edge_index, patch_index
                )
            if global_tokens is None:
                global_tokens = local_tokens.new_zeros(
                    (patch_count, local_tokens.shape[1])
                )
            global_index = torch.as_tensor(
                raw_batch["patch_global_index"], device=device, dtype=torch.long
            )
            global_tokens.index_copy_(0, global_index, local_tokens)
    assert global_tokens is not None
    return global_tokens, torch.as_tensor(
        proposal_by_patch, device=device, dtype=torch.long
    )


def _forward_with_global_context(
    model: SurfNCC,
    raw_batch: dict[str, np.ndarray],
    tensor_batch: dict[str, torch.Tensor],
    cached_patch_tokens: torch.Tensor,
    global_patch_proposal: torch.Tensor,
    alpha: float,
) -> dict[str, torch.Tensor]:
    hidden, local_tokens = model.encode_surface_points(
        tensor_batch["point_features"],
        tensor_batch["edge_index"],
        tensor_batch["patch_index"],
    )
    global_index = torch.as_tensor(
        raw_batch["patch_global_index"],
        device=tensor_batch["point_features"].device,
        dtype=torch.long,
    )
    complete_tokens = cached_patch_tokens.index_copy(0, global_index, local_tokens)
    proposal_index = torch.as_tensor(
        raw_batch["proposal_global_index"],
        device=tensor_batch["point_features"].device,
        dtype=torch.long,
    )
    relevant_patch = torch.isin(global_patch_proposal, proposal_index)
    relevant_global_index = torch.nonzero(
        relevant_patch, as_tuple=False
    ).squeeze(1)
    relevant_patch_proposal = torch.searchsorted(
        proposal_index, global_patch_proposal[relevant_patch]
    )
    encoded_relevant, proposal_tokens = model.encode_proposal_patches(
        complete_tokens[relevant_patch], relevant_patch_proposal
    )
    local_relevant_index = torch.searchsorted(relevant_global_index, global_index)
    outputs = model.decode_surface_points(
        hidden,
        tensor_batch["patch_index"],
        tensor_batch["patch_proposal_index"],
        encoded_relevant[local_relevant_index],
        proposal_tokens,
        tensor_batch["method_class"],
        tensor_batch["contradiction"],
        cvar_alpha=alpha,
    )
    complete_patch_risk = torch.sigmoid(
        model.patch_risk_head(encoded_relevant)
    ).squeeze(1)
    outputs["proposal_head_target"] = torch.stack(
        [
            complete_patch_risk[relevant_patch_proposal == proposal].max()
            for proposal in range(int(proposal_index.shape[0]))
        ]
    )
    return outputs


def _unit_rank_pairs(
    prepared: list[
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> list[tuple[int, int]]:
    """Match complete proposals once per unit, independent of point packing."""
    rows: dict[int, dict[str, Any]] = {}
    for raw_batch, _, _, _, _ in prepared:
        for local, proposal in enumerate(raw_batch["proposal_global_index"]):
            rows[int(proposal)] = {
                "actor": bool(raw_batch["proposal_actor"][local]),
                "safe": bool(raw_batch["proposal_safe"][local]),
                "unsafe": bool(raw_batch["proposal_unsafe"][local]),
                "point_count": int(raw_batch["proposal_point_count"][local]),
            }
    used_safe: set[int] = set()
    pairs = []
    for unsafe_index in sorted(
        index for index, row in rows.items() if row["unsafe"]
    ):
        candidates = [
            index for index, row in rows.items()
            if row["safe"]
            and index not in used_safe
            and row["actor"] == rows[unsafe_index]["actor"]
        ]
        if not candidates:
            continue
        safe_index = min(
            candidates,
            key=lambda index: (
                abs(rows[index]["point_count"] - rows[unsafe_index]["point_count"]),
                index,
            ),
        )
        used_safe.add(safe_index)
        pairs.append((safe_index, unsafe_index))
    return pairs


def _global_rank_loss(
    model: SurfNCC,
    cached_patch_tokens: torch.Tensor,
    global_patch_proposal: torch.Tensor,
    pairs: list[tuple[int, int]],
    margin: float,
) -> tuple[torch.Tensor, int]:
    """Apply the frozen ranking margin to the complete unit proposal set."""
    _, proposal_tokens = model.encode_proposal_patches(
        cached_patch_tokens, global_patch_proposal
    )
    risks = torch.sigmoid(model.proposal_risk_head(proposal_tokens)).squeeze(1)
    losses = [
        torch.relu(float(margin) + risks[safe] - risks[unsafe])
        for safe, unsafe in pairs
    ]
    if not losses:
        return risks.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def _hard_violations(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> int:
    predicted = outputs["probabilities"].argmax(dim=1)
    method = batch["method_class"]
    contradiction = batch["contradiction"]
    invalid = (
        ((method == FREE_INDEX) & ~contradiction & (predicted != FREE_INDEX))
        | ((method == OCCUPIED_INDEX) & ~contradiction & (predicted != OCCUPIED_INDEX))
        | (contradiction & (predicted != UNKNOWN_INDEX))
    )
    return int(invalid.sum().item())


def _match_rank_surrogate(rows: list[dict[str, Any]], margin: float) -> tuple[float, int]:
    used_safe: set[int] = set()
    values = []
    for unsafe_index, unsafe_row in enumerate(rows):
        if not unsafe_row["unsafe"]:
            continue
        candidates = [
            index
            for index, row in enumerate(rows)
            if row["safe"]
            and index not in used_safe
            and row["actor"] == unsafe_row["actor"]
        ]
        if not candidates:
            continue
        safe_index = min(
            candidates,
            key=lambda index: (
                abs(int(rows[index]["point_count"]) - int(unsafe_row["point_count"])),
                index,
            ),
        )
        used_safe.add(safe_index)
        values.append(
            max(
                0.0,
                float(margin)
                + float(rows[safe_index]["risk"])
                - float(unsafe_row["risk"]),
            )
        )
    return (float(np.mean(values)) if values else 0.0), len(values)


def _unit_grouped_rank_surrogate(
    rows: list[dict[str, Any]], margin: float
) -> tuple[float, int]:
    """Match within complete scene/frame units and average the unit losses."""
    values = []
    pair_count = 0
    unit_keys = sorted(
        {(str(row["scene"]), int(row["target_frame"])) for row in rows}
    )
    for scene, target_frame in unit_keys:
        value, count = _match_rank_surrogate(
            [
                row
                for row in rows
                if row["scene"] == scene
                and int(row["target_frame"]) == target_frame
            ],
            margin,
        )
        if count:
            values.append(value)
            pair_count += count
    return (float(np.mean(values)) if values else 0.0), pair_count


def _numpy_cvar_tail(values: list[float], alpha: float) -> float:
    """Exact empirical upper-tail CVaR after all chunks of one proposal are joined."""
    array = np.asarray(values, dtype=np.float32)
    if not array.size:
        return 0.0
    count = max(1, int(math.ceil((1.0 - float(alpha)) * array.size)))
    return float(np.partition(array, array.size - count)[-count:].mean())


def _evaluate_selection(
    model: SurfNCC,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    alpha = float(config["risk"]["cvar_alpha"])
    authority_threshold = float(config["risk"]["authority_threshold"])
    point_limit = int(config["training"]["point_microbatch"])
    proposal_map: dict[tuple[str, int, int], dict[str, Any]] = {}
    hard_violations = 0
    safe_occ_total = 0
    safe_occ_retained = 0
    emitted_occ = 0
    point_count = 0
    correct = 0
    unknown = 0
    surface_run = Path(config["inputs"]["p3_surface_run"])
    native_run = Path(config["inputs"]["p2_native_run"])
    with torch.no_grad():
        for scene in config["cohort"]["selection_scenes"]:
            for target_frame in config["cohort"]["targets"]:
                unit = load_surface_unit(surface_run, native_run, scene, int(target_frame))
                prepared = _prepare_unit_batches(unit, point_limit)
                cached_patch_tokens, global_patch_proposal = _global_patch_context(
                    model, unit, prepared, device
                )
                for (
                    raw_batch,
                    features,
                    method_values,
                    contradiction_values,
                    authority_target,
                ) in prepared:
                    tensor_batch = _tensor_batch(
                        raw_batch,
                        features,
                        method_values,
                        contradiction_values,
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
                    probabilities = outputs["probabilities"].float()
                    predicted = probabilities.argmax(dim=1)
                    target = tensor_batch["target_class"]
                    method = tensor_batch["method_class"]
                    contradiction = tensor_batch["contradiction"]
                    learned_low_authority = (
                        (method == UNKNOWN_INDEX)
                        & ~contradiction
                        & (predicted == OCCUPIED_INDEX)
                        & (outputs["authority"] < authority_threshold)
                    )
                    decision = predicted.clone()
                    decision[learned_low_authority] = UNKNOWN_INDEX
                    hidden_free = (
                        (target == FREE_INDEX)
                        & (method == UNKNOWN_INDEX)
                        & ~contradiction
                    )
                    safe_occ = (
                        (target == OCCUPIED_INDEX)
                        & (method == UNKNOWN_INDEX)
                        & ~contradiction
                    )
                    emitted = decision == OCCUPIED_INDEX
                    safe_occ_total += int(safe_occ.sum())
                    safe_occ_retained += int((safe_occ & emitted).sum())
                    emitted_occ += int(emitted.sum())
                    point_count += int(target.numel())
                    correct += int((decision == target).sum())
                    unknown += int((decision == UNKNOWN_INDEX).sum())
                    hard_violations += _hard_violations(outputs, tensor_batch)
                    actor = raw_batch["proposal_actor"]
                    counts = raw_batch["proposal_chunk_point_count"]
                    point_proposal = tensor_batch["point_proposal_index"]
                    for proposal in range(int(actor.shape[0])):
                        selected = point_proposal == proposal
                        proposal_hidden_free = (
                            selected
                            & (target == FREE_INDEX)
                            & (method == UNKNOWN_INDEX)
                            & ~contradiction
                        )
                        proposal_heldout_occ = (
                            selected
                            & (target == OCCUPIED_INDEX)
                            & (method == UNKNOWN_INDEX)
                            & ~contradiction
                        )
                        global_index = int(raw_batch["proposal_global_index"][proposal])
                        key = (scene, int(target_frame), global_index)
                        row = proposal_map.setdefault(
                            key,
                            {
                                "scene": scene,
                                "target_frame": int(target_frame),
                                "proposal_global_index": global_index,
                                "actor": bool(actor[proposal]),
                                "safe": bool(raw_batch["proposal_safe"][proposal]),
                                "heldout_occ": False,
                                "unsafe": bool(raw_batch["proposal_unsafe"][proposal]),
                                "point_count": 0,
                                "risk": 0.0,
                                "hidden_free_values": [],
                            },
                        )
                        row["heldout_occ"] = bool(row["heldout_occ"]) or bool(
                            proposal_heldout_occ.any()
                        )
                        row["actor"] = bool(row["actor"]) or bool(actor[proposal])
                        row["safe"] = bool(row["safe"]) or bool(
                            raw_batch["proposal_safe"][proposal]
                        )
                        row["unsafe"] = bool(row["unsafe"]) or bool(
                            raw_batch["proposal_unsafe"][proposal]
                        )
                        row["point_count"] = int(row["point_count"]) + int(counts[proposal])
                        row["risk"] = max(
                            float(row["risk"]),
                            float(outputs["proposal_cvar"][proposal]),
                        )
                        if bool(proposal_hidden_free.any()):
                            row["hidden_free_values"].extend(
                                probabilities[
                                    proposal_hidden_free, OCCUPIED_INDEX
                                ].detach().cpu().tolist()
                            )
                del prepared, cached_patch_tokens, global_patch_proposal, unit
                gc.collect()
    proposal_rows = list(proposal_map.values())
    hidden_tail_values = [
        _numpy_cvar_tail(row["hidden_free_values"], alpha)
        for row in proposal_rows
        if row["hidden_free_values"]
    ]
    ranking, matched_pairs = _unit_grouped_rank_surrogate(
        proposal_rows, float(config["loss_weights"]["ranking_margin"])
    )
    retention = safe_occ_retained / max(1, safe_occ_total)
    coverage = emitted_occ / max(1, point_count)
    accuracy = correct / max(1, point_count)
    hidden_tail = float(np.mean(hidden_tail_values)) if hidden_tail_values else 0.0
    primary = hidden_tail + ranking
    selection = config["selection"]
    objective = [
        int(hard_violations),
        primary,
        max(0.0, float(selection["safe_occ_retention_min"]) - retention),
        max(0.0, float(selection["emitted_occ_coverage_min"]) - coverage),
        -retention,
        -coverage,
        -accuracy,
    ]
    return {
        "objective": objective,
        "hard_violations": hard_violations,
        "hidden_free_tail": hidden_tail,
        "matched_rank_surrogate": ranking,
        "matched_pair_count": matched_pairs,
        "safe_occ_retention": retention,
        "emitted_occ_coverage": coverage,
        "target_accuracy_secondary": accuracy,
        "unknown_fraction": unknown / max(1, point_count),
        "point_count": point_count,
        "proposal_count": len(proposal_rows),
    }


def run(config_path: Path, repo_root: Path, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip():
        raise RuntimeError("P5 formal requires clean source")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise ValueError("P5 task identity drift")
    free_gib = shutil.disk_usage(run_dir.parent).free / 1024**3
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise RuntimeError(f"insufficient disk before P5: {free_gib:.3f} GiB")
    if not torch.cuda.is_available():
        raise RuntimeError("P5 requires CUDA")
    run_dir.mkdir(parents=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
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
    best_objective = None
    best_epoch = None
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
            loss_total = 0.0
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
                    row.update(
                        {
                            "epoch": epoch,
                            "scene": scene,
                            "target_frame": target_frame,
                        }
                    )
                    dropout_stream.write(json.dumps(row, sort_keys=True) + "\n")
                prepared = _prepare_unit_batches(unit, point_limit, dropout_plans)
                unit_rank_pairs = _unit_rank_pairs(prepared)
                unit_rank_applied = False
                batch_order = np.arange(len(prepared))
                order_rng.shuffle(batch_order)
                global_context = None
                for batch_ordinal in batch_order:
                    if global_context is None:
                        global_context = _global_patch_context(
                            model, unit, prepared, device
                        )
                    cached_patch_tokens, global_patch_proposal = global_context
                    (
                        raw_batch,
                        features,
                        method,
                        contradiction,
                        authority_target,
                    ) = prepared[int(batch_ordinal)]
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
                        losses = compute_surfncc_losses(
                            outputs,
                            tensor_batch,
                            cvar_alpha=alpha,
                            weights=config["loss_weights"],
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
                                float(config["loss_weights"]["ranking_margin"]),
                            )
                            unit_rank_applied = True
                        total = losses["total"] + float(
                            config["loss_weights"]["proposal_rank"]
                        ) * ranking
                        scaled = total / float(accumulation)
                    finite_training = finite_training and bool(torch.isfinite(total).item())
                    scaler.scale(scaled).backward()
                    hard_violations += _hard_violations(outputs, tensor_batch)
                    accumulation_step += 1
                    batch_count += 1
                    point_count += int(features.shape[0])
                    rank_pair_count += int(pairs)
                    loss_total += float(total.detach())
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

            selection = _evaluate_selection(model, config, device)
            objective = tuple(float(value) for value in selection["objective"])
            improved = best_objective is None or objective < best_objective
            if improved:
                best_objective = objective
                best_epoch = epoch
                epochs_since_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "selection": selection,
                        "input_dimension": POINT_FEATURE_DIMENSION,
                    },
                    run_dir / "SURFNCC_BEST.pt",
                )
            else:
                epochs_since_improvement += 1
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "optimizer_steps_total": optimizer_step,
                    "train_batch_count": batch_count,
                    "train_point_count": point_count,
                    "train_rank_pair_count": rank_pair_count,
                    "mean_train_loss": loss_total / max(1, batch_count),
                    "selection": selection,
                    "improved": improved,
                    "wall_seconds": time.monotonic() - epoch_started,
                }
            )
            _write_json(run_dir / "EPOCHS.json", epoch_rows)
            if epoch + 1 >= min_epochs and epochs_since_improvement >= patience:
                break

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    passed = (
        finite_training
        and hard_violations == 0
        and best_epoch is not None
        and (run_dir / "SURFNCC_BEST.pt").is_file()
        and peak_gib <= float(config["resources"]["gpu_peak_gib_max"])
    )
    summary = {
        "schema_version": "worldsim_v63.p5_train_summary.v1",
        "task_id": TASK_ID,
        "passed": passed,
        "epoch_count": len(epoch_rows),
        "optimizer_steps": optimizer_step,
        "best_epoch": best_epoch,
        "best_objective": list(best_objective) if best_objective is not None else None,
        "best_selection": epoch_rows[best_epoch]["selection"] if best_epoch is not None else None,
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
    _write_json(run_dir / "P5_TRAIN_SUMMARY.json", summary)
    _write_json(
        run_dir / "P5_TRAIN_MANIFEST.json",
        {
            "task_id": TASK_ID,
            "source_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo_root, text=True
            ).strip(),
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
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_dir.resolve()),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
