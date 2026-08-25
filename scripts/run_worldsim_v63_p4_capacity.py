#!/usr/bin/env python3
"""Run the frozen single-GPU SurfNCC capacity probe without quality claims."""

from __future__ import annotations

import argparse
import json
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
    capacity_proposal_indices,
    compute_surfncc_losses,
    load_surface_unit,
    plan_packed_structural_dropout,
    proposal_batches,
    structural_dropout_records,
)


TASK_ID = "WS-V63-P4-CAPACITY-01"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tensor_batch(
    batch: dict[str, np.ndarray],
    point_features: np.ndarray,
    method_class: np.ndarray,
    contradiction: np.ndarray,
    device: torch.device,
    authority_target: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "point_features": torch.as_tensor(point_features, device=device),
        "edge_index": torch.as_tensor(batch["edge_index"], device=device, dtype=torch.long),
        "patch_index": torch.as_tensor(batch["patch_index"], device=device, dtype=torch.long),
        "patch_proposal_index": torch.as_tensor(
            batch["patch_proposal_index"], device=device, dtype=torch.long
        ),
        "point_proposal_index": torch.as_tensor(
            batch["point_proposal_index"], device=device, dtype=torch.long
        ),
        "target_class": torch.as_tensor(batch["target_class"], device=device, dtype=torch.long),
        "method_class": torch.as_tensor(method_class, device=device, dtype=torch.long),
        "contradiction": torch.as_tensor(contradiction, device=device, dtype=torch.bool),
        "authority_target": torch.as_tensor(
            batch["authority_target"] if authority_target is None else authority_target,
            device=device,
            dtype=torch.bool,
        ),
    }


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


def _prepare_proposal_group(
    unit: Any,
    batches: list[dict[str, np.ndarray]],
    dropout_plans: dict[int, dict[str, Any]] | None = None,
) -> list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    prepared = []
    for raw_batch in batches:
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
        else:
            features, method, contradiction, authority_target = (
                apply_packed_structural_dropout(unit, raw_batch, dropout_plans)
            )
            prepared.append(
                (raw_batch, features, method, contradiction, authority_target)
            )
    return prepared


def _proposal_context(
    model: SurfNCC,
    prepared: list[
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    tokens = []
    spans = []
    start = 0
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
            stop = start + int(local_tokens.shape[0])
            tokens.append(local_tokens)
            spans.append((start, stop))
            start = stop
    return torch.cat(tokens, dim=0), spans


def _forward_proposal(
    model: SurfNCC,
    raw_batch: dict[str, np.ndarray],
    batch: dict[str, torch.Tensor],
    cached_patch_tokens: torch.Tensor,
    patch_span: tuple[int, int],
    alpha: float,
) -> dict[str, torch.Tensor]:
    hidden, local_tokens = model.encode_surface_points(
        batch["point_features"],
        batch["edge_index"],
        batch["patch_index"],
    )
    start, stop = patch_span
    positions = torch.arange(
        start, stop, device=batch["point_features"].device, dtype=torch.long
    )
    complete_tokens = cached_patch_tokens.index_copy(0, positions, local_tokens)
    patch_proposal = torch.zeros(
        complete_tokens.shape[0],
        device=batch["point_features"].device,
        dtype=torch.long,
    )
    encoded, proposal_tokens = model.encode_proposal_patches(
        complete_tokens, patch_proposal
    )
    outputs = model.decode_surface_points(
        hidden,
        batch["patch_index"],
        batch["patch_proposal_index"],
        encoded[positions],
        proposal_tokens,
        batch["method_class"],
        batch["contradiction"],
        cvar_alpha=alpha,
    )
    outputs["proposal_head_target"] = torch.sigmoid(
        model.patch_risk_head(encoded)
    ).squeeze(1).max()[None]
    return outputs


def run(config_path: Path, repo_root: Path, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip():
        raise RuntimeError("P4 capacity requires clean source")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise ValueError("P4 task identity drift")
    free_gib = shutil.disk_usage(run_dir.parent).free / 1024**3
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise RuntimeError(f"insufficient disk before P4: {free_gib:.3f} GiB")
    if not torch.cuda.is_available():
        raise RuntimeError("P4 requires CUDA")
    run_dir.mkdir(parents=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{int(config['resources']['gpu'])}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    surface_run = Path(config["inputs"]["p3_surface_run"])
    native_run = Path(config["inputs"]["p2_native_run"])
    train_spec = config["units"]["train"]
    selection_spec = config["units"]["selection"]
    train_unit = load_surface_unit(
        surface_run, native_run, train_spec["scene"], int(train_spec["target_frame"])
    )
    selection_unit = load_surface_unit(
        surface_run,
        native_run,
        selection_spec["scene"],
        int(selection_spec["target_frame"]),
    )
    point_limit = int(config["training"]["point_microbatch"])
    train_proposal_indices = capacity_proposal_indices(train_unit)
    selection_proposal_indices = capacity_proposal_indices(selection_unit)
    train_groups = [
        proposal_batches(train_unit, proposal, point_limit)
        for proposal in train_proposal_indices
    ]
    selection_groups = [
        proposal_batches(selection_unit, proposal, point_limit)
        for proposal in selection_proposal_indices
    ]
    if any(
        batch["point_features"].shape[1] != POINT_FEATURE_DIMENSION
        for group in train_groups
        for batch in group
    ):
        raise RuntimeError("native surface feature dimension mismatch")

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
    optimizer_steps = int(config["training"]["optimizer_steps"])
    dropout_rng = np.random.default_rng(int(config["dropout"]["seed"]))
    records = []
    hard_violations = 0
    finite_loss = True
    finite_gradient = True
    cvar_gradient_nonzero = False
    cvar_gradient_heads: set[str] = set()
    proposal_token_gradient_nonzero = False
    last_tensor_batch = None
    last_group_index = None
    last_batch_index = None

    dropout_plans = plan_packed_structural_dropout(
        train_unit,
        dropout_rng,
        float(config["dropout"]["support_fraction"]),
    )
    all_dropout_records = {
        int(row["proposal_global_index"]): row
        for row in structural_dropout_records(dropout_plans)
    }
    dropout_records = [
        all_dropout_records[int(proposal)] for proposal in train_proposal_indices
    ]
    prepared_train_groups = [
        _prepare_proposal_group(train_unit, group, dropout_plans)
        for group in train_groups
    ]
    prepared_selection_groups = [
        _prepare_proposal_group(selection_unit, group)
        for group in selection_groups
    ]
    group_cursors = [0 for _ in prepared_train_groups]

    model.train()
    for optimizer_step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        for accumulation_step in range(accumulation):
            group_index = (
                optimizer_step * accumulation + accumulation_step
            ) % len(prepared_train_groups)
            prepared = prepared_train_groups[group_index]
            batch_index = group_cursors[group_index] % len(prepared)
            group_cursors[group_index] += 1
            cached_patch_tokens, patch_spans = _proposal_context(
                model, prepared, device
            )
            (
                raw_batch,
                features,
                effective_method,
                effective_contradiction,
                authority_target,
            ) = prepared[batch_index]
            tensor_batch = _tensor_batch(
                raw_batch,
                features,
                effective_method,
                effective_contradiction,
                device,
                authority_target,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = _forward_proposal(
                    model,
                    raw_batch,
                    tensor_batch,
                    cached_patch_tokens,
                    patch_spans[batch_index],
                    alpha,
                )
                losses = compute_surfncc_losses(
                    outputs,
                    tensor_batch,
                    cvar_alpha=alpha,
                    weights=config["loss_weights"],
                )
                scaled_loss = losses["total"] / float(accumulation)
            direct_cvar_gradients = torch.autograd.grad(
                outputs["proposal_cvar"].float().mean(),
                (
                    model.state_head.weight,
                    model.hidden_free_head.weight,
                    model.authority_head.weight,
                ),
                retain_graph=True,
                allow_unused=True,
            )
            for name, gradient in zip(
                ("state_head", "hidden_free_head", "authority_head"),
                direct_cvar_gradients,
            ):
                if (
                    gradient is not None
                    and bool(torch.isfinite(gradient).all())
                    and bool(torch.count_nonzero(gradient).item())
                ):
                    cvar_gradient_heads.add(name)
            cvar_gradient_nonzero = bool(cvar_gradient_heads)
            finite_loss = finite_loss and bool(torch.isfinite(losses["total"]).item())
            scaler.scale(scaled_loss).backward()
            hard_violations += _hard_violations(outputs, tensor_batch)
            records.append(
                {
                    "optimizer_step": optimizer_step,
                    "accumulation_step": accumulation_step,
                    "proposal_group_index": group_index,
                    "proposal_global_index": int(
                        train_proposal_indices[group_index]
                    ),
                    "proposal_id": train_unit.proposal_ids[
                        train_proposal_indices[group_index]
                    ],
                    "proposal_full_point_count": int(
                        sum(item[1].shape[0] for item in prepared)
                    ),
                    "proposal_chunk_count": len(prepared),
                    "proposal_full_patch_count": int(cached_patch_tokens.shape[0]),
                    "chunk_point_count": int(features.shape[0]),
                    "chunk_patch_count": int(np.max(raw_batch["patch_index"])) + 1,
                    "edge_count": int(raw_batch["edge_index"].shape[1]),
                    "finite_loss": bool(torch.isfinite(losses["total"]).item()),
                }
            )
            last_tensor_batch = tensor_batch
            last_group_index = group_index
            last_batch_index = batch_index
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite_gradient = finite_gradient and all(bool(torch.isfinite(value).all()) for value in gradients)
        proposal_token_gradient_nonzero = proposal_token_gradient_nonzero or bool(
            model.proposal_token.grad is not None
            and torch.count_nonzero(model.proposal_token.grad).item()
        )
        scaler.step(optimizer)
        scaler.update()

    checkpoint = run_dir / "SURFNCC_CAPACITY.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "input_dimension": POINT_FEATURE_DIMENSION},
        checkpoint,
    )
    reloaded = SurfNCC(
        int(model_config["input_dimension"]),
        hidden_dimension=int(model_config["hidden_dimension"]),
        neighbor_blocks=int(model_config["neighbor_blocks"]),
        patch_transformer_layers=int(model_config["patch_transformer_layers"]),
        attention_heads=int(model_config["attention_heads"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded.load_state_dict(payload["model"])
    checkpoint_reload = True

    model.eval()
    reloaded.eval()
    assert last_tensor_batch is not None
    assert last_group_index is not None and last_batch_index is not None
    last_prepared = prepared_train_groups[last_group_index]
    last_raw_batch = last_prepared[last_batch_index][0]
    model_context, model_spans = _proposal_context(model, last_prepared, device)
    reload_context, reload_spans = _proposal_context(reloaded, last_prepared, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        first = _forward_proposal(
            model,
            last_raw_batch,
            last_tensor_batch,
            model_context,
            model_spans[last_batch_index],
            alpha,
        )["probabilities"]
        second = _forward_proposal(
            model,
            last_raw_batch,
            last_tensor_batch,
            model_context,
            model_spans[last_batch_index],
            alpha,
        )["probabilities"]
        restored = _forward_proposal(
            reloaded,
            last_raw_batch,
            last_tensor_batch,
            reload_context,
            reload_spans[last_batch_index],
            alpha,
        )["probabilities"]
    repeated_difference = float(torch.max(torch.abs(first - second)).item())
    reload_difference = float(torch.max(torch.abs(first - restored)).item())

    selection_rows = []
    with torch.no_grad():
        for group_index, prepared in enumerate(prepared_selection_groups):
            cached_patch_tokens, patch_spans = _proposal_context(
                model, prepared, device
            )
            finite_outputs = True
            for batch_index, (
                raw_batch,
                features,
                method,
                contradiction,
                authority_target,
            ) in enumerate(prepared):
                tensor_batch = _tensor_batch(
                    raw_batch,
                    features,
                    method,
                    contradiction,
                    device,
                    authority_target,
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = _forward_proposal(
                        model,
                        raw_batch,
                        tensor_batch,
                        cached_patch_tokens,
                        patch_spans[batch_index],
                        alpha,
                    )
                hard_violations += _hard_violations(outputs, tensor_batch)
                finite_outputs = finite_outputs and all(
                    bool(torch.isfinite(value).all()) for value in outputs.values()
                )
            selection_rows.append(
                {
                    "proposal_group_index": group_index,
                    "proposal_global_index": int(
                        selection_proposal_indices[group_index]
                    ),
                    "proposal_id": selection_unit.proposal_ids[
                        selection_proposal_indices[group_index]
                    ],
                    "point_count": int(sum(item[1].shape[0] for item in prepared)),
                    "patch_count": int(cached_patch_tokens.shape[0]),
                    "chunk_count": len(prepared),
                    "finite_outputs": finite_outputs,
                }
            )

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    expected = config["pass_contract"]
    passed = (
        hard_violations <= int(expected["hard_violations_max"])
        and finite_loss
        and finite_gradient
        and cvar_gradient_nonzero
        and proposal_token_gradient_nonzero
        and checkpoint_reload
        and repeated_difference <= float(expected["repeated_forward_max_abs_difference"])
        and reload_difference <= float(expected["repeated_forward_max_abs_difference"])
        and peak_gib <= float(expected["gpu_peak_gib_max"])
        and all(row["finite_outputs"] for row in selection_rows)
    )
    summary = {
        "schema_version": "worldsim_v63.p4_capacity_summary.v2",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "passed": passed,
        "train_scene_count": 1,
        "selection_scene_count": 1,
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation": accumulation,
        "train_proposal_count": len(prepared_train_groups),
        "selection_proposal_count": len(prepared_selection_groups),
        "train_chunk_count": sum(len(group) for group in prepared_train_groups),
        "selection_chunk_count": sum(
            len(group) for group in prepared_selection_groups
        ),
        "maximum_full_proposal_point_count": max(
            sum(item[1].shape[0] for item in group)
            for group in prepared_train_groups + prepared_selection_groups
        ),
        "maximum_full_proposal_patch_count": max(
            sum(int(item[0]["patch_global_index"].shape[0]) for item in group)
            for group in prepared_train_groups + prepared_selection_groups
        ),
        "input_dimension": POINT_FEATURE_DIMENSION,
        "hard_violations": hard_violations,
        "finite_loss": finite_loss,
        "finite_gradient": finite_gradient,
        "cvar_gradient_nonzero": cvar_gradient_nonzero,
        "cvar_gradient_heads": sorted(cvar_gradient_heads),
        "proposal_token_gradient_nonzero": proposal_token_gradient_nonzero,
        "checkpoint_reload": checkpoint_reload,
        "repeated_forward_max_abs_difference": repeated_difference,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "peak_gpu_gib": peak_gib,
        "wall_seconds": time.monotonic() - started,
        "selection_rows": selection_rows,
        "quality_conclusion": False,
        "calibration_quality_read": False,
        "confirmation_read": False,
        "exact_once_test_read": False,
    }
    _write_json(run_dir / "P4_CAPACITY_SUMMARY.json", summary)
    (run_dir / "STRUCTURAL_DROPOUT.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in dropout_records
        ),
        encoding="utf-8",
    )
    (run_dir / "CAPACITY_STEPS.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "P4_CAPACITY_MANIFEST.json",
        {
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "source_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo_root, text=True
            ).strip(),
            "train_unit": train_spec,
            "selection_unit": selection_spec,
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
