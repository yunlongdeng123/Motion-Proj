#!/usr/bin/env python3
"""Train the sole prototype-feature-dropout recovery from the frozen P5 best model."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from motion_proj.worldsim_v62.cpsc_lite import (
    CPSCLite,
    UnitArrays,
    compute_cpsc_losses,
    load_unit_arrays,
)
from motion_proj.worldsim_v62.legacy_bridge import (
    bridge_unit_features,
    fit_query_weighted_class_prototypes,
)
from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
)


TASK_ID = "WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01"


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


def _specs(scenes: list[str], frames: list[int]) -> list[tuple[str, int]]:
    return [(scene, int(frame)) for scene in scenes for frame in frames]


def _make_model(checkpoint: dict[str, Any], config: dict[str, Any]) -> CPSCLite:
    model = CPSCLite(
        int(checkpoint["prior_feature_dimension"]),
        int(checkpoint["query_feature_dimension"]),
        hidden_width=int(config["hidden_width"]),
        decoder_layers=int(config["query_decoder_layers"]),
        residual_blocks=int(config["residual_blocks"]),
        projection_iterations=int(config["projection_iterations"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _metadata(unit: UnitArrays, selected: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "target_class": unit.target_class[selected],
        "method_class": unit.method_class[selected],
        "dropout_class": unit.dropout_class[selected],
        "query_type": unit.query_type[selected],
        "contradiction": unit.contradiction[selected],
        "actor_bound": unit.actor_bound[selected],
        "prior_valid": unit.prior_valid[selected],
    }


def _to_device(
    values: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(value)).to(device)
        for key, value in values.items()
    }


def _forward(
    model: CPSCLite,
    prior: torch.Tensor,
    query: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    method = batch["method_class"].long()
    contradiction = batch["contradiction"].bool()
    return model(
        prior.float(),
        query.float(),
        observed_free=(method == FREE_INDEX) & ~contradiction,
        observed_occupied=(method == OCCUPIED_INDEX) & ~contradiction,
        contradiction=contradiction,
    )


def _counts() -> dict[str, int]:
    return {
        "query": 0,
        "correct": 0,
        "hidden_free": 0,
        "hidden_false_occ": 0,
        "safe_occ": 0,
        "safe_occ_retained": 0,
        "unknown": 0,
        "hard": 0,
        "hard_violation": 0,
    }


def _update(
    counts: dict[str, int], probabilities: torch.Tensor, batch: dict[str, torch.Tensor]
) -> None:
    prediction = probabilities.argmax(dim=1)
    target = batch["target_class"].long()
    method = batch["method_class"].long()
    dropout = batch["dropout_class"].long()
    contradiction = batch["contradiction"].bool()
    hidden_free = (
        (method == UNKNOWN_INDEX) & (dropout == FREE_INDEX) & ~contradiction
    )
    safe_occ = (
        (method == UNKNOWN_INDEX)
        & (dropout == OCCUPIED_INDEX)
        & (target == OCCUPIED_INDEX)
        & ~contradiction
    )
    observed_free = (method == FREE_INDEX) & ~contradiction
    observed_occupied = (method == OCCUPIED_INDEX) & ~contradiction
    expected = torch.full_like(method, UNKNOWN_INDEX)
    expected[observed_free] = FREE_INDEX
    expected[observed_occupied] = OCCUPIED_INDEX
    constrained = observed_free | observed_occupied | contradiction
    counts["query"] += int(target.numel())
    counts["correct"] += int((prediction == target).sum())
    counts["hidden_free"] += int(hidden_free.sum())
    counts["hidden_false_occ"] += int(
        ((prediction == OCCUPIED_INDEX) & hidden_free).sum()
    )
    counts["safe_occ"] += int(safe_occ.sum())
    counts["safe_occ_retained"] += int(
        ((prediction == OCCUPIED_INDEX) & safe_occ).sum()
    )
    counts["unknown"] += int((prediction == UNKNOWN_INDEX).sum())
    counts["hard"] += int(constrained.sum())
    counts["hard_violation"] += int(((prediction != expected) & constrained).sum())


def _finalize(counts: dict[str, int]) -> dict[str, Any]:
    return {
        "query_count": counts["query"],
        "target_accuracy": counts["correct"] / max(1, counts["query"]),
        "hidden_free_count": counts["hidden_free"],
        "hidden_free_false_occ_rate": counts["hidden_false_occ"]
        / max(1, counts["hidden_free"]),
        "safe_occ_count": counts["safe_occ"],
        "safe_occ_retention_rate": counts["safe_occ_retained"]
        / max(1, counts["safe_occ"]),
        "predicted_unknown_fraction": counts["unknown"] / max(1, counts["query"]),
        "hard_constraint_count": counts["hard"],
        "hard_violation_rate": counts["hard_violation"] / max(1, counts["hard"]),
    }


@torch.no_grad()
def evaluate(
    student: CPSCLite,
    teacher: CPSCLite,
    unit_specs: list[tuple[str, int]],
    *,
    p2_run: Path,
    p4_run: Path,
    prototypes: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    loss_config: dict[str, Any],
    epoch: int,
    consistency_weight: float,
) -> dict[str, Any]:
    student.eval()
    teacher.eval()
    bridge_counts = _counts()
    full_counts = _counts()
    task_sum = 0.0
    kl_sum = 0.0
    total_queries = 0
    for scene, frame in unit_specs:
        unit = load_unit_arrays(p2_run, p4_run, scene, frame)
        bridge_prior, bridge_query = bridge_unit_features(unit, prototypes)
        for start in range(0, unit.query_count, batch_size):
            stop = min(unit.query_count, start + batch_size)
            selected = np.arange(start, stop, dtype=np.int64)
            batch = _to_device(_metadata(unit, selected), device)
            full_prior = torch.from_numpy(unit.prior_features[selected]).to(device)
            full_query = torch.from_numpy(unit.query_features[selected]).to(device)
            bridge_output = _forward(
                student,
                torch.from_numpy(bridge_prior[selected]).to(device),
                torch.from_numpy(bridge_query[selected]).to(device),
                batch,
            )
            teacher_output = _forward(teacher, full_prior, full_query, batch)
            full_output = _forward(student, full_prior, full_query, batch)
            losses = compute_cpsc_losses(
                bridge_output, batch, loss_config=loss_config, epoch=epoch
            )
            kl = F.kl_div(
                torch.log(
                    bridge_output["base_probabilities"].float().clamp_min(1e-8)
                ),
                teacher_output["base_probabilities"].float(),
                reduction="batchmean",
            )
            size = stop - start
            task_sum += float(losses["total"]) * size
            kl_sum += float(kl) * size
            total_queries += size
            _update(bridge_counts, bridge_output["probabilities"], batch)
            _update(full_counts, full_output["probabilities"], batch)
    task_loss = task_sum / max(1, total_queries)
    consistency_kl = kl_sum / max(1, total_queries)
    return {
        "objective": task_loss + consistency_weight * consistency_kl,
        "task_loss": task_loss,
        "consistency_kl": consistency_kl,
        "bridge": _finalize(bridge_counts),
        "full": _finalize(full_counts),
    }


def run(config_path: Path, repo_root: Path, run_root: Path) -> Path:
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise ValueError("P6R task identity drift")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__feature-dropout-train-s0-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        p5_config = yaml.safe_load(Path(config["inputs"]["p5_config"]).read_text())
        p2_run = Path(config["inputs"]["p2_run"])
        p4_run = Path(config["inputs"]["p4_run"])
        frames = [int(value) for value in config["inputs"]["target_frames"]]
        train_specs = _specs(config["split"]["train_scenes"], frames)
        selection_specs = _specs(config["split"]["selection_scenes"], frames)
        prototypes = fit_query_weighted_class_prototypes(
            p2_run, p4_run, config["split"]["train_scenes"], frames
        )
        np.savez_compressed(
            run_dir / "CLASS_PROTOTYPES.npz",
            class_query_count=prototypes["counts"],
            mean_prior_logits=prototypes["logits"],
            mean_bev_features=prototypes["bev"],
        )
        device = torch.device(f"cuda:{int(config['resources']['gpu'])}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        checkpoint = torch.load(
            Path(config["inputs"]["p5_best_model"]),
            map_location=device,
            weights_only=True,
        )
        continued_epoch_offset = int(checkpoint["epoch"]) + 1
        teacher = _make_model(checkpoint, p5_config["model"]).to(device).eval()
        teacher.requires_grad_(False)
        student = _make_model(checkpoint, p5_config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        use_amp = config["training"]["precision"] == "fp16"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        batch_size = int(config["training"]["query_batch_size"])
        accumulation = int(config["training"]["gradient_accumulation_steps"])
        consistency_weight = float(config["recovery"]["consistency_weight"])
        corruption_probability = float(
            config["recovery"]["prototype_feature_probability"]
        )
        rng = np.random.default_rng(seed)
        baseline = evaluate(
            student,
            teacher,
            selection_specs,
            p2_run=p2_run,
            p4_run=p4_run,
            prototypes=prototypes,
            batch_size=batch_size,
            device=device,
            loss_config=p5_config["loss"],
            epoch=int(checkpoint["epoch"]),
            consistency_weight=consistency_weight,
        )
        best_objective = float("inf")
        best_epoch = -1
        stale_epochs = 0
        optimizer_steps = 0
        log_rows: list[dict[str, Any]] = []
        best_path = run_dir / "BEST_MODEL.pt"
        for epoch in range(int(config["training"]["maximum_epochs"])):
            epoch_started = time.monotonic()
            student.train()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            train_task_sum = 0.0
            train_kl_sum = 0.0
            train_queries = 0
            unit_order = list(train_specs)
            rng.shuffle(unit_order)
            for scene, frame in unit_order:
                unit = load_unit_arrays(p2_run, p4_run, scene, frame)
                bridge_prior, bridge_query = bridge_unit_features(unit, prototypes)
                order = np.arange(unit.query_count, dtype=np.int64)
                rng.shuffle(order)
                for start in range(0, unit.query_count, batch_size):
                    selected = order[start : start + batch_size]
                    batch = _to_device(_metadata(unit, selected), device)
                    full_prior = unit.prior_features[selected]
                    full_query = unit.query_features[selected]
                    use_bridge = rng.random(selected.size) < corruption_probability
                    corrupt_prior = full_prior.copy()
                    corrupt_query = full_query.copy()
                    corrupt_prior[use_bridge] = bridge_prior[selected][use_bridge]
                    corrupt_query[use_bridge] = bridge_query[selected][use_bridge]
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                        teacher_output = _forward(
                            teacher,
                            torch.from_numpy(full_prior).to(device),
                            torch.from_numpy(full_query).to(device),
                            batch,
                        )
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output = _forward(
                            student,
                            torch.from_numpy(corrupt_prior).to(device),
                            torch.from_numpy(corrupt_query).to(device),
                            batch,
                        )
                        losses = compute_cpsc_losses(
                            output,
                            batch,
                            loss_config=p5_config["loss"],
                            epoch=continued_epoch_offset + epoch,
                        )
                        kl = F.kl_div(
                            torch.log(
                                output["base_probabilities"].float().clamp_min(1e-8)
                            ),
                            teacher_output["base_probabilities"].float(),
                            reduction="batchmean",
                        )
                        total_loss = losses["total"] + consistency_weight * kl
                    if not bool(torch.isfinite(total_loss)):
                        raise RuntimeError("non-finite P6R training loss")
                    scaler.scale(total_loss / accumulation).backward()
                    size = selected.size
                    train_task_sum += float(losses["total"]) * size
                    train_kl_sum += float(kl) * size
                    train_queries += size
                    pending += 1
                    if pending == accumulation:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            student.parameters(),
                            float(config["training"]["gradient_clip_norm"]),
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        pending = 0
                        optimizer_steps += 1
            if pending:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    float(config["training"]["gradient_clip_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            selection = evaluate(
                student,
                teacher,
                selection_specs,
                p2_run=p2_run,
                p4_run=p4_run,
                prototypes=prototypes,
                batch_size=batch_size,
                device=device,
                loss_config=p5_config["loss"],
                epoch=continued_epoch_offset + epoch,
                consistency_weight=consistency_weight,
            )
            objective = float(selection["objective"])
            log_rows.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "train_query_count": train_queries,
                    "train_task_loss": train_task_sum / max(1, train_queries),
                    "train_consistency_kl": train_kl_sum / max(1, train_queries),
                    "selection": selection,
                    "epoch_wall_seconds": time.monotonic() - epoch_started,
                }
            )
            _write_jsonl(run_dir / "TRAIN_LOG.jsonl", log_rows)
            if objective < best_objective:
                best_objective = objective
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "prior_feature_dimension": checkpoint["prior_feature_dimension"],
                        "query_feature_dimension": checkpoint["query_feature_dimension"],
                        "epoch": epoch,
                        "selection_objective": objective,
                        "recovery": "prototype_feature_dropout_teacher_consistency",
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            if (
                epoch + 1 >= int(config["training"]["minimum_epochs"])
                and stale_epochs >= int(config["training"]["early_stop_patience"])
            ):
                break
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        student.load_state_dict(best_checkpoint["model_state_dict"])
        best_selection = evaluate(
            student,
            teacher,
            selection_specs,
            p2_run=p2_run,
            p4_run=p4_run,
            prototypes=prototypes,
            batch_size=batch_size,
            device=device,
            loss_config=p5_config["loss"],
            epoch=continued_epoch_offset + best_epoch,
            consistency_weight=consistency_weight,
        )
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**3
        wall_seconds = time.monotonic() - started
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        summary = {
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "epochs_completed": len(log_rows),
            "optimizer_steps": optimizer_steps,
            "best_epoch": best_epoch,
            "best_selection_objective": best_objective,
            "baseline_selection": baseline,
            "best_selection": best_selection,
            "peak_gpu_memory_gib": peak_gpu,
            "wall_seconds": wall_seconds,
            "legacy_o_eval_read": False,
            "confirmation_read": False,
            "exact_once_test_read": False,
            "irwm_process_resident": False,
            "passed": bool(
                np.isfinite(best_objective)
                and peak_gpu <= float(config["resources"]["maximum_gpu_memory_gib"])
                and wall_seconds <= float(config["resources"]["maximum_wall_seconds"])
            ),
        }
        _write_json(run_dir / "P6R_TRAIN_SUMMARY.json", summary)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "wall_seconds": wall_seconds,
                "peak_gpu_memory_gib": peak_gpu,
                "output_bytes": sum(
                    path.stat().st_size
                    for path in run_dir.rglob("*")
                    if path.is_file()
                ),
                "disk_free_gib": shutil.disk_usage(run_root).free / 1024**3,
                "irwm_inference_started": False,
                "legacy_o_eval_read": False,
                "training_started": True,
            },
        )
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "task_id": TASK_ID,
                "source_commit": source_commit,
                "config_path": str(config_path),
                "p2_run": str(p2_run),
                "p4_run": str(p4_run),
                "p5_best_model": config["inputs"]["p5_best_model"],
                "identity_policy": config["identity_policy"],
                "hash_checksum_fingerprint": False,
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "task_id": TASK_ID,
                "status": "done" if summary["passed"] else "rejected",
                "canonical": True,
                "run_uri": f"run://worldsim_v62/{TASK_ID}/{run_dir.name}",
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "task_id": TASK_ID,
                "status": "failed",
                "canonical": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v62")
    )
    args = parser.parse_args()
    print(run(args.config, args.repo_root.resolve(), args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
