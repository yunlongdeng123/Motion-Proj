#!/usr/bin/env python3
"""Train and select the first bounded CPSC-Lite model on P2/P4 development data."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v62.cpsc_lite import (
    CPSCLite,
    compute_cpsc_losses,
    iter_unit_batches,
    load_unit_arrays,
    projection_only_probabilities,
)
from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _unit_specs(scenes: list[str], frames: list[int]) -> list[tuple[str, int]]:
    return [(scene, int(frame)) for scene in scenes for frame in frames]


def _to_device(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(np.asarray(value))
        result[key] = tensor.to(device=device, non_blocking=False)
    return result


def _model_forward(
    model: CPSCLite, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    method = batch["method_class"].long()
    contradiction = batch["contradiction"].bool()
    return model(
        batch["prior_features"].float(),
        batch["query_features"].float(),
        observed_free=(method == FREE_INDEX) & ~contradiction,
        observed_occupied=(method == OCCUPIED_INDEX) & ~contradiction,
        contradiction=contradiction,
    )


def _empty_metric_counts() -> dict[str, float]:
    return {
        "query_count": 0.0,
        "correct": 0.0,
        "hidden_free_count": 0.0,
        "hidden_free_false_occ": 0.0,
        "safe_occ_count": 0.0,
        "safe_occ_retained": 0.0,
        "unconstrained_count": 0.0,
        "unconstrained_unknown": 0.0,
        "predicted_free": 0.0,
        "predicted_occupied": 0.0,
        "predicted_unknown": 0.0,
        "hard_constraint_count": 0.0,
        "hard_constraint_violation": 0.0,
    }


def _update_metric_counts(
    counts: dict[str, float],
    probabilities: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> None:
    prediction = probabilities.argmax(dim=-1)
    target = batch["target_class"].long()
    method = batch["method_class"].long()
    dropout = batch["dropout_class"].long()
    contradiction = batch["contradiction"].bool()
    method_unknown = method == UNKNOWN_INDEX
    hidden_free = method_unknown & (dropout == FREE_INDEX) & ~contradiction
    safe_occ = (
        method_unknown
        & (dropout == OCCUPIED_INDEX)
        & (target == OCCUPIED_INDEX)
        & ~contradiction
    )
    hard_expected = torch.full_like(method, UNKNOWN_INDEX)
    observed_free = (method == FREE_INDEX) & ~contradiction
    observed_occupied = (method == OCCUPIED_INDEX) & ~contradiction
    hard_expected[observed_free] = FREE_INDEX
    hard_expected[observed_occupied] = OCCUPIED_INDEX
    constrained = observed_free | observed_occupied | contradiction

    counts["query_count"] += float(target.numel())
    counts["correct"] += float((prediction == target).sum())
    counts["hidden_free_count"] += float(hidden_free.sum())
    counts["hidden_free_false_occ"] += float(
        ((prediction == OCCUPIED_INDEX) & hidden_free).sum()
    )
    counts["safe_occ_count"] += float(safe_occ.sum())
    counts["safe_occ_retained"] += float(
        ((prediction == OCCUPIED_INDEX) & safe_occ).sum()
    )
    counts["unconstrained_count"] += float(method_unknown.sum())
    counts["unconstrained_unknown"] += float(
        ((prediction == UNKNOWN_INDEX) & method_unknown).sum()
    )
    counts["predicted_free"] += float((prediction == FREE_INDEX).sum())
    counts["predicted_occupied"] += float((prediction == OCCUPIED_INDEX).sum())
    counts["predicted_unknown"] += float((prediction == UNKNOWN_INDEX).sum())
    counts["hard_constraint_count"] += float(constrained.sum())
    counts["hard_constraint_violation"] += float(
        ((prediction != hard_expected) & constrained).sum()
    )


def _finalize_metric_counts(counts: dict[str, float]) -> dict[str, float]:
    query_count = max(1.0, counts["query_count"])
    hidden_count = max(1.0, counts["hidden_free_count"])
    safe_count = max(1.0, counts["safe_occ_count"])
    unconstrained_count = max(1.0, counts["unconstrained_count"])
    hard_count = max(1.0, counts["hard_constraint_count"])
    return {
        "query_count": int(counts["query_count"]),
        "target_accuracy": counts["correct"] / query_count,
        "hidden_free_count": int(counts["hidden_free_count"]),
        "hidden_free_false_occ_rate": counts["hidden_free_false_occ"] / hidden_count,
        "safe_occ_count": int(counts["safe_occ_count"]),
        "safe_occ_retention_rate": counts["safe_occ_retained"] / safe_count,
        "unconstrained_unknown_fraction": counts["unconstrained_unknown"]
        / unconstrained_count,
        "predicted_free_fraction": counts["predicted_free"] / query_count,
        "predicted_occupied_fraction": counts["predicted_occupied"] / query_count,
        "predicted_unknown_fraction": counts["predicted_unknown"] / query_count,
        "hard_constraint_count": int(counts["hard_constraint_count"]),
        "hard_constraint_violation_rate": counts["hard_constraint_violation"] / hard_count,
    }


@torch.no_grad()
def evaluate(
    model: CPSCLite,
    unit_specs: list[tuple[str, int]],
    *,
    p2_run: Path,
    p4_run: Path,
    batch_size: int,
    device: torch.device,
    loss_config: dict[str, Any],
    epoch: int,
    include_projection_baseline: bool,
) -> dict[str, Any]:
    model.eval()
    learned_counts = _empty_metric_counts()
    baseline_counts = _empty_metric_counts()
    loss_sums = {
        name: 0.0
        for name in (
            "total",
            "query",
            "evidential",
            "hidden_free",
            "safe_occ",
            "actor_temporal",
            "prior_preserve",
        )
    }
    loss_weight = 0
    loss_denominators = {
        "supervised_count": 0,
        "hidden_free_count": 0,
        "safe_occ_count": 0,
        "actor_temporal_count": 0,
        "hard_conflict_count": 0,
    }
    for scene, target_frame in unit_specs:
        unit = load_unit_arrays(p2_run, p4_run, scene, target_frame)
        for numpy_batch in iter_unit_batches(unit, batch_size, rng=None):
            batch = _to_device(numpy_batch, device)
            outputs = _model_forward(model, batch)
            losses = compute_cpsc_losses(
                outputs, batch, loss_config=loss_config, epoch=epoch
            )
            size = int(batch["target_class"].numel())
            for name in loss_sums:
                loss_sums[name] += float(losses[name]) * size
            loss_weight += size
            for name in loss_denominators:
                loss_denominators[name] += int(losses[name])
            _update_metric_counts(learned_counts, outputs["probabilities"], batch)
            if include_projection_baseline:
                baseline = projection_only_probabilities(
                    batch["prior_tristate"].float(),
                    batch["method_class"].long(),
                    batch["contradiction"].bool(),
                )
                _update_metric_counts(baseline_counts, baseline, batch)
        del unit
    result = {
        "losses": {
            name: value / max(1, loss_weight) for name, value in loss_sums.items()
        },
        "loss_denominators": loss_denominators,
        "learned": _finalize_metric_counts(learned_counts),
    }
    if include_projection_baseline:
        result["projection_only"] = _finalize_metric_counts(baseline_counts)
    return result


def run(
    config_path: Path,
    repo_root: Path,
    run_dir: Path,
    limit_train_units: int | None,
    limit_selection_units: int | None,
    maximum_optimizer_steps: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{int(config['resources']['gpu'])}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    frames = [int(value) for value in config["inputs"]["target_frames"]]
    train_specs = _unit_specs(config["split"]["train_scenes"], frames)
    selection_specs = _unit_specs(config["split"]["selection_scenes"], frames)
    if limit_train_units is not None:
        train_specs = train_specs[: int(limit_train_units)]
    if limit_selection_units is not None:
        selection_specs = selection_specs[: int(limit_selection_units)]
    mode = (
        "probe"
        if limit_train_units is not None
        or limit_selection_units is not None
        or maximum_optimizer_steps is not None
        else "formal"
    )
    p2_run = Path(config["inputs"]["p2_run"])
    p4_run = Path(config["inputs"]["p4_run"])
    first_unit = load_unit_arrays(p2_run, p4_run, *train_specs[0])
    prior_dimension = int(first_unit.prior_features.shape[1])
    query_dimension = int(first_unit.query_features.shape[1])
    del first_unit

    model = CPSCLite(
        prior_dimension,
        query_dimension,
        hidden_width=int(config["model"]["hidden_width"]),
        decoder_layers=int(config["model"]["query_decoder_layers"]),
        residual_blocks=int(config["model"]["residual_blocks"]),
        projection_iterations=int(config["model"]["projection_iterations"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    use_amp = config["training"]["precision"] == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    batch_size = int(config["training"]["query_batch_size"])
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    maximum_epochs = 1 if mode == "probe" else int(config["training"]["maximum_epochs"])
    rng = np.random.default_rng(seed)
    log_rows = []
    best_objective = float("inf")
    best_epoch = -1
    stale_epochs = 0
    optimizer_steps = 0
    best_path = run_dir / "BEST_MODEL.pt"
    final_path = run_dir / "FINAL_MODEL.pt"

    for epoch in range(maximum_epochs):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        unit_order = list(train_specs)
        rng.shuffle(unit_order)
        train_loss_sum = 0.0
        train_query_count = 0
        pending_batches = 0
        stop_training = False
        for scene, target_frame in unit_order:
            unit = load_unit_arrays(p2_run, p4_run, scene, target_frame)
            for numpy_batch in iter_unit_batches(unit, batch_size, rng=rng):
                batch = _to_device(numpy_batch, device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = _model_forward(model, batch)
                    losses = compute_cpsc_losses(
                        outputs, batch, loss_config=config["loss"], epoch=epoch
                    )
                    scaled_loss = losses["total"] / accumulation_steps
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError("non-finite P5 training loss")
                scaler.scale(scaled_loss).backward()
                size = int(batch["target_class"].numel())
                train_loss_sum += float(losses["total"]) * size
                train_query_count += size
                pending_batches += 1
                if pending_batches == accumulation_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["training"]["gradient_clip_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending_batches = 0
                    optimizer_steps += 1
                    if (
                        maximum_optimizer_steps is not None
                        and optimizer_steps >= int(maximum_optimizer_steps)
                    ):
                        stop_training = True
                        break
            del unit
            if stop_training:
                break
        if pending_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        selection = evaluate(
            model,
            selection_specs,
            p2_run=p2_run,
            p4_run=p4_run,
            batch_size=batch_size,
            device=device,
            loss_config=config["loss"],
            epoch=epoch,
            include_projection_baseline=epoch == 0,
        )
        objective = float(selection["losses"]["total"])
        row = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train_query_count": train_query_count,
            "train_total_loss": train_loss_sum / max(1, train_query_count),
            "selection": selection,
            "selection_objective": objective,
            "epoch_wall_seconds": time.monotonic() - epoch_started,
        }
        log_rows.append(row)
        _write_jsonl(run_dir / "TRAIN_LOG.jsonl", log_rows)
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "prior_feature_dimension": prior_dimension,
                    "query_feature_dimension": query_dimension,
                    "epoch": epoch,
                    "selection_objective": objective,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if mode == "formal" and (
            epoch + 1 >= int(config["training"]["minimum_epochs"])
            and stale_epochs >= int(config["training"]["early_stop_patience"])
        ):
            break
        if stop_training:
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prior_feature_dimension": prior_dimension,
            "query_feature_dimension": query_dimension,
            "epoch": log_rows[-1]["epoch"],
            "selection_objective": log_rows[-1]["selection_objective"],
        },
        final_path,
    )
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_selection = evaluate(
        model,
        selection_specs,
        p2_run=p2_run,
        p4_run=p4_run,
        batch_size=batch_size,
        device=device,
        loss_config=config["loss"],
        epoch=best_epoch,
        include_projection_baseline=False,
    )
    peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**3
    source_git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    summary = {
        "schema_version": "worldsim_v62.p5_cpsc_lite_summary.v1",
        "task_id": config["task_id"],
        "mode": mode,
        "train_unit_count": len(train_specs),
        "selection_unit_count": len(selection_specs),
        "epochs_completed": len(log_rows),
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "best_selection_objective": best_objective,
        "best_selection": best_selection,
        "final_epoch_selection": log_rows[-1]["selection"],
        "projection_only_selection": log_rows[0]["selection"]["projection_only"],
        "prior_feature_dimension": prior_dimension,
        "query_feature_dimension": query_dimension,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_gib": peak_gpu,
        "wall_seconds": time.monotonic() - started,
        "target_evidence_method_input": False,
        "legacy_o_eval_read": False,
        "confirmation_content_read": False,
        "exact_once_test_read": False,
        "irwm_process_resident": False,
        "passed": bool(
            np.isfinite(best_objective)
            and best_path.exists()
            and final_path.exists()
            and peak_gpu <= float(config["resources"]["maximum_gpu_memory_gib"])
        ),
    }
    _write_json(run_dir / "P5_SUMMARY.json", summary)
    manifest = {
        "schema_version": "worldsim_v62.p5_cpsc_lite_manifest.v1",
        "task_id": config["task_id"],
        "mode": mode,
        "source_git_commit": source_git_commit,
        "p2_run": str(p2_run),
        "p4_run": str(p4_run),
        "train_scenes": config["split"]["train_scenes"],
        "selection_scenes": config["split"]["selection_scenes"],
        "train_units": [f"{scene}/f{frame:03d}" for scene, frame in train_specs],
        "selection_units": [
            f"{scene}/f{frame:03d}" for scene, frame in selection_specs
        ],
        "model_inputs": ["P2 method-visible evidence/query coordinates", "P4 frozen sidecar"],
        "supervision_only": ["P2 dropout evidence", "P2 target evidence"],
        "identity_policy": config["identity_policy"],
        "hash_checksum_fingerprint": False,
    }
    _write_json(run_dir / "P5_MANIFEST.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit-train-units", type=int)
    parser.add_argument("--limit-selection-units", type=int)
    parser.add_argument("--maximum-optimizer-steps", type=int)
    args = parser.parse_args()
    summary = run(
        args.config,
        args.repo_root.resolve(),
        args.run_dir,
        args.limit_train_units,
        args.limit_selection_units,
        args.maximum_optimizer_steps,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
