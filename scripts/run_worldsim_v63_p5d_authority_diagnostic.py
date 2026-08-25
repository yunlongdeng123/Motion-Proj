#!/usr/bin/env python3
"""Diagnose SurfNCC positive-authority collapse on Tier-D training units only."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.nn import functional as F

from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
)
from motion_proj.worldsim_v63.surfncc import SurfNCC, cvar_tail, load_surface_unit
from scripts.run_worldsim_v63_p5_train import (
    _forward_with_global_context,
    _global_patch_context,
    _hard_violations,
    _prepare_unit_batches,
    _tensor_batch,
)


TASK_ID = "WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01"
GROUPS = ("safe_occ", "hidden_free", "unknown")
METRICS = (
    "q_authority",
    "base_p_occ",
    "projected_p_occ",
    "point_risk",
    "patch_cvar",
    "proposal_cvar",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class Histogram:
    """Bounded streaming histogram with deterministic approximate quantiles."""

    def __init__(self, bins: int) -> None:
        self.bins = int(bins)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: torch.Tensor) -> None:
        array = values.detach().float().cpu().numpy().reshape(-1)
        if not array.size:
            return
        if not bool(np.isfinite(array).all()):
            raise RuntimeError("non-finite diagnostic distribution")
        array = np.clip(array, 0.0, 1.0)
        indices = np.minimum((array * self.bins).astype(np.int64), self.bins - 1)
        self.counts += np.bincount(indices, minlength=self.bins)
        self.count += int(array.size)
        self.total += float(array.astype(np.float64).sum())
        self.total_square += float(np.square(array.astype(np.float64)).sum())
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def quantile(self, probability: float) -> float | None:
        if not self.count:
            return None
        rank = max(0, min(self.count - 1, int(math.ceil(probability * self.count) - 1)))
        index = int(np.searchsorted(np.cumsum(self.counts), rank + 1, side="left"))
        return (index + 0.5) / self.bins

    def summary(self) -> dict[str, Any]:
        mean = self.total / self.count if self.count else None
        variance = (
            max(0.0, self.total_square / self.count - float(mean) ** 2)
            if self.count
            else None
        )
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "min": self.minimum if self.count else None,
            "p01": self.quantile(0.01),
            "p05": self.quantile(0.05),
            "p25": self.quantile(0.25),
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "max": self.maximum if self.count else None,
            "histogram_counts": self.counts.tolist(),
        }


def _binned_auc(positive: Histogram, negative: Histogram) -> float | None:
    if not positive.count or not negative.count:
        return None
    negative_below = np.cumsum(negative.counts) - negative.counts
    favorable = np.sum(
        positive.counts.astype(np.float64)
        * (negative_below.astype(np.float64) + 0.5 * negative.counts)
    )
    return float(favorable / (positive.count * negative.count))


def _group_masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = batch["target_class"]
    method = batch["method_class"]
    eligible = (method == UNKNOWN_INDEX) & ~batch["contradiction"]
    return {
        "safe_occ": eligible & (target == OCCUPIED_INDEX),
        "hidden_free": eligible & (target == FREE_INDEX),
        "unknown": eligible & (target == UNKNOWN_INDEX),
    }


def _direct_tail(values: torch.Tensor, mask: torch.Tensor, proposal: torch.Tensor, alpha: float) -> torch.Tensor:
    terms = []
    proposal_count = int(proposal.max().item()) + 1
    for index in range(proposal_count):
        selected = mask & (proposal == index)
        if bool(selected.any()):
            terms.append(cvar_tail(values[selected], alpha))
    return torch.stack(terms).mean() if terms else values.sum() * 0.0


def _loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    masks = _group_masks(batch)
    probabilities = outputs["probabilities"].float()
    proposal = batch["point_proposal_index"]
    tail_direct = _direct_tail(
        probabilities[:, OCCUPIED_INDEX], masks["hidden_free"], proposal, alpha
    )
    hidden_free_aux = F.binary_cross_entropy_with_logits(
        outputs["hidden_free_logits"].float(), masks["hidden_free"].float()
    )
    retention = _direct_tail(
        1.0 - probabilities[:, OCCUPIED_INDEX], masks["safe_occ"], proposal, alpha
    )
    authority = F.binary_cross_entropy_with_logits(
        outputs["authority_logits"].float(), batch["authority_target"].float()
    )
    return (
        {
            "tail_training_term": tail_direct + hidden_free_aux,
            "tail_direct": tail_direct,
            "hidden_free_aux": hidden_free_aux,
            "safe_occ_retention": retention,
            "authority": authority,
        },
        {
            "hidden_free": int(masks["hidden_free"].sum().item()),
            "safe_occ": int(masks["safe_occ"].sum().item()),
            "authority_positive": int(batch["authority_target"].sum().item()),
            "points": int(batch["target_class"].numel()),
        },
    )


def _parameter_group(name: str) -> str:
    if name.startswith("state_head"):
        return "state_head"
    if name.startswith("authority_head"):
        return "authority_head"
    if name.startswith("hidden_free_head"):
        return "hidden_free_head"
    if name.startswith("patch_risk_head") or name.startswith("proposal_risk_head"):
        return "risk_heads"
    return "trunk"


def _gradient_norms(
    names: list[str], gradients: tuple[torch.Tensor | None, ...]
) -> dict[str, float]:
    squares = {group: 0.0 for group in (
        "all", "trunk", "state_head", "authority_head", "hidden_free_head", "risk_heads"
    )}
    for name, gradient in zip(names, gradients):
        if gradient is None:
            continue
        value = float(torch.sum(gradient.detach().float() ** 2).item())
        squares["all"] += value
        squares[_parameter_group(name)] += value
    return {group: math.sqrt(value) for group, value in squares.items()}


def _cosine(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
) -> float | None:
    dot = 0.0
    first_square = 0.0
    second_square = 0.0
    for left, right in zip(first, second):
        if left is None or right is None:
            continue
        left_float = left.detach().float()
        right_float = right.detach().float()
        dot += float(torch.sum(left_float * right_float).item())
        first_square += float(torch.sum(left_float**2).item())
        second_square += float(torch.sum(right_float**2).item())
    if first_square == 0.0 or second_square == 0.0:
        return None
    return dot / math.sqrt(first_square * second_square)


def _aggregate(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def _plot_distributions(
    path: Path,
    histograms: dict[str, dict[str, Histogram]],
) -> None:
    colors = {"safe_occ": "#16803c", "hidden_free": "#c73832", "unknown": "#4466aa"}
    labels = {"safe_occ": "safe OCC", "hidden_free": "hidden FREE", "unknown": "UNKNOWN"}
    bins = histograms[GROUPS[0]][METRICS[0]].bins
    x = (np.arange(bins) + 0.5) / bins
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, metric in zip(axes.reshape(-1), METRICS):
        for group in GROUPS:
            counts = histograms[group][metric].counts.astype(np.float64)
            density = counts / max(1.0, counts.sum())
            axis.plot(x, density, color=colors[group], label=labels[group], linewidth=1.3)
        axis.set_title(metric)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylabel("point mass / bin")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("value")
    axes[-1, 1].set_xlabel("value")
    axes[-1, 2].set_xlabel("value")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("SurfNCC train-only authority-collapse mechanism distributions")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path, repo_root: Path, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip():
        raise RuntimeError("P5D formal requires clean source")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise ValueError("P5D task identity drift")
    if shutil.disk_usage(run_dir.parent).free / 1024**3 < float(
        config["resources"]["minimum_disk_free_gib"]
    ):
        raise RuntimeError("insufficient disk before P5D")
    if not torch.cuda.is_available():
        raise RuntimeError("P5D requires CUDA")

    p5_run = Path(config["inputs"]["p5_train_run"])
    p5_summary = json.loads((p5_run / "P5_TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
    if float(p5_summary["best_selection"]["safe_occ_retention"]) != 0.0:
        raise RuntimeError("P5D is authorized only for terminal zero safe-OCC retention")
    p5_config_path = repo_root / str(config["inputs"]["p5_config"])
    p5_config = yaml.safe_load(p5_config_path.read_text(encoding="utf-8"))
    if config["cohort"]["train_scenes"] != p5_config["cohort"]["train_scenes"]:
        raise RuntimeError("P5D train-scene cohort drift")
    if config["cohort"]["targets"] != p5_config["cohort"]["targets"]:
        raise RuntimeError("P5D train-target cohort drift")

    run_dir.mkdir(parents=True)
    seed = int(config["seed"])
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

    model_config = p5_config["model"]
    model = SurfNCC(
        int(model_config["input_dimension"]),
        hidden_dimension=int(model_config["hidden_dimension"]),
        neighbor_blocks=int(model_config["neighbor_blocks"]),
        patch_transformer_layers=int(model_config["patch_transformer_layers"]),
        attention_heads=int(model_config["attention_heads"]),
    ).to(device)
    checkpoint = torch.load(p5_run / "SURFNCC_BEST.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    bins = int(config["diagnostic"]["histogram_bins"])
    alpha = float(config["diagnostic"]["cvar_alpha"])
    threshold = float(config["diagnostic"]["authority_threshold"])
    point_limit = int(config["diagnostic"]["point_microbatch"])
    histograms = {
        group: {metric: Histogram(bins) for metric in METRICS}
        for group in GROUPS
    }
    decisions = {
        group: {
            "point_count": 0,
            "authority_target_true": 0,
            "base_prediction_counts": [0, 0, 0],
            "projected_prediction_counts": [0, 0, 0],
            "post_authority_prediction_counts": [0, 0, 0],
            "authority_veto_count": 0,
        }
        for group in GROUPS
    }
    surface_run = Path(config["inputs"]["p3_surface_run"])
    native_run = Path(config["inputs"]["p2_native_run"])
    hard_violations = 0
    unit_count = 0
    with torch.no_grad():
        for scene in config["cohort"]["train_scenes"]:
            for target_frame in config["cohort"]["targets"]:
                unit = load_surface_unit(surface_run, native_run, scene, int(target_frame))
                prepared = _prepare_unit_batches(unit, point_limit)
                cached_patch_tokens, global_patch_proposal = _global_patch_context(
                    model, unit, prepared, device
                )
                for raw_batch, features, method, contradiction, authority_target in prepared:
                    batch = _tensor_batch(
                        raw_batch, features, method, contradiction, device, authority_target
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = _forward_with_global_context(
                            model,
                            raw_batch,
                            batch,
                            cached_patch_tokens,
                            global_patch_proposal,
                            alpha,
                        )
                    hard_violations += _hard_violations(outputs, batch)
                    base_prediction = outputs["base_probabilities"].argmax(dim=1)
                    projected_prediction = outputs["probabilities"].argmax(dim=1)
                    veto = (
                        (batch["method_class"] == UNKNOWN_INDEX)
                        & ~batch["contradiction"]
                        & (projected_prediction == OCCUPIED_INDEX)
                        & (outputs["authority"] < threshold)
                    )
                    final_prediction = projected_prediction.clone()
                    final_prediction[veto] = UNKNOWN_INDEX
                    metric_values = {
                        "q_authority": outputs["authority"],
                        "base_p_occ": outputs["base_probabilities"][:, OCCUPIED_INDEX],
                        "projected_p_occ": outputs["probabilities"][:, OCCUPIED_INDEX],
                        "point_risk": outputs["point_risk"],
                        "patch_cvar": outputs["patch_cvar"][batch["patch_index"]],
                        "proposal_cvar": outputs["proposal_cvar"][batch["point_proposal_index"]],
                    }
                    for group, mask in _group_masks(batch).items():
                        count = int(mask.sum().item())
                        if not count:
                            continue
                        decisions[group]["point_count"] += count
                        decisions[group]["authority_target_true"] += int(
                            batch["authority_target"][mask].sum().item()
                        )
                        decisions[group]["authority_veto_count"] += int((veto & mask).sum().item())
                        for key, prediction in (
                            ("base_prediction_counts", base_prediction),
                            ("projected_prediction_counts", projected_prediction),
                            ("post_authority_prediction_counts", final_prediction),
                        ):
                            counts = torch.bincount(prediction[mask], minlength=3).cpu().tolist()
                            decisions[group][key] = [
                                int(left) + int(right)
                                for left, right in zip(decisions[group][key], counts)
                            ]
                        for metric, values in metric_values.items():
                            histograms[group][metric].update(values[mask])
                unit_count += 1
                del prepared, cached_patch_tokens, global_patch_proposal, unit
                gc.collect()

    distribution_document = {
        "schema_version": "worldsim_v63.p5d_group_distributions.v1",
        "groups": {
            group: {
                metric: histograms[group][metric].summary()
                for metric in METRICS
            }
            for group in GROUPS
        },
        "safe_occ_greater_than_hidden_free_binned_auc": {
            metric: _binned_auc(
                histograms["safe_occ"][metric], histograms["hidden_free"][metric]
            )
            for metric in METRICS
        },
        "histogram_bins": bins,
        "unit_count": unit_count,
    }
    _write_json(run_dir / "GROUP_DISTRIBUTIONS.json", distribution_document)
    _write_json(
        run_dir / "DECISION_STAGE_COUNTS.json",
        {
            "schema_version": "worldsim_v63.p5d_decision_stage_counts.v1",
            "class_order": ["UNKNOWN", "FREE", "OCCUPIED"],
            "groups": decisions,
        },
    )
    _plot_distributions(run_dir / "SURFACE_AUTHORITY_DISTRIBUTIONS.png", histograms)

    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameter_names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    component_names = (
        "tail_training_term",
        "tail_direct",
        "hidden_free_aux",
        "safe_occ_retention",
        "authority",
    )
    norm_rows: dict[str, dict[str, list[float]]] = {
        component: {
            group: []
            for group in ("all", "trunk", "state_head", "authority_head", "hidden_free_head", "risk_heads")
        }
        for component in component_names
    }
    loss_values = {component: [] for component in component_names}
    support_totals = {"hidden_free": 0, "safe_occ": 0, "authority_positive": 0, "points": 0}
    tail_retention_cosines: list[float] = []
    gradient_batch_count = 0
    for scene in config["cohort"]["train_scenes"]:
        for target_frame in config["diagnostic"]["gradient_probe_targets"]:
            unit = load_surface_unit(surface_run, native_run, scene, int(target_frame))
            prepared = _prepare_unit_batches(unit, point_limit)
            cached_patch_tokens, global_patch_proposal = _global_patch_context(
                model, unit, prepared, device
            )
            for raw_batch, features, method, contradiction, authority_target in prepared:
                batch = _tensor_batch(
                    raw_batch, features, method, contradiction, device, authority_target
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = _forward_with_global_context(
                        model,
                        raw_batch,
                        batch,
                        cached_patch_tokens,
                        global_patch_proposal,
                        alpha,
                    )
                    components, support = _loss_components(outputs, batch, alpha)
                gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
                for ordinal, component in enumerate(component_names):
                    loss_values[component].append(float(components[component].detach()))
                    gradients[component] = torch.autograd.grad(
                        components[component],
                        parameters,
                        retain_graph=ordinal + 1 < len(component_names),
                        allow_unused=True,
                    )
                    norms = _gradient_norms(parameter_names, gradients[component])
                    for group, value in norms.items():
                        norm_rows[component][group].append(value)
                cosine = _cosine(
                    gradients["tail_training_term"], gradients["safe_occ_retention"]
                )
                if cosine is not None:
                    tail_retention_cosines.append(cosine)
                for key, value in support.items():
                    support_totals[key] += int(value)
                gradient_batch_count += 1
                del outputs, components, gradients
            del prepared, cached_patch_tokens, global_patch_proposal, unit
            gc.collect()

    weights = {
        "tail_training_term": float(p5_config["loss_weights"]["hidden_free_tail"]),
        "tail_direct": float(p5_config["loss_weights"]["hidden_free_tail"]),
        "hidden_free_aux": float(p5_config["loss_weights"]["hidden_free_tail"]),
        "safe_occ_retention": float(p5_config["loss_weights"]["safe_occ_retention"]),
        "authority": float(p5_config["loss_weights"]["authority"]),
    }
    gradient_document = {
        "schema_version": "worldsim_v63.p5d_gradient_magnitudes.v1",
        "probe_scenes": config["cohort"]["train_scenes"],
        "probe_targets": config["diagnostic"]["gradient_probe_targets"],
        "batch_count": gradient_batch_count,
        "support_totals": support_totals,
        "components": {
            component: {
                "loss": _aggregate(loss_values[component]),
                "raw_gradient_norm": {
                    group: _aggregate(values)
                    for group, values in norm_rows[component].items()
                },
                "frozen_weight": weights[component],
                "weighted_gradient_norm": {
                    group: _aggregate([weights[component] * value for value in values])
                    for group, values in norm_rows[component].items()
                },
            }
            for component in component_names
        },
        "tail_retention_gradient_cosine": _aggregate(tail_retention_cosines),
    }
    _write_json(run_dir / "GRADIENT_MAGNITUDES.json", gradient_document)

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    passed = (
        hard_violations == 0
        and unit_count == len(config["cohort"]["train_scenes"]) * len(config["cohort"]["targets"])
        and all(decisions[group]["point_count"] > 0 for group in GROUPS)
        and peak_gib <= float(config["resources"]["gpu_peak_gib_max"])
    )
    summary = {
        "schema_version": "worldsim_v63.p5d_authority_diagnostic_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "passed": passed,
        "p5_best_epoch": int(p5_summary["best_epoch"]),
        "p5_best_training_objective": p5_summary["best_objective"],
        "p5_candidate_promotable": False,
        "p5_safe_occ_retention": float(p5_summary["best_selection"]["safe_occ_retention"]),
        "train_unit_count": unit_count,
        "hard_violations": hard_violations,
        "group_point_counts": {
            group: int(decisions[group]["point_count"]) for group in GROUPS
        },
        "gradient_probe_batch_count": gradient_batch_count,
        "peak_gpu_gib": peak_gib,
        "wall_seconds": time.monotonic() - started,
        "training_started": False,
        "optimizer_steps": 0,
        "selection_scene_read": False,
        "calibration_quality_read": False,
        "confirmation_read": False,
        "exact_once_test_read": False,
        "mechanism_interpretation": "deferred_until_frozen_distributions_and_gradients_are_read",
    }
    _write_json(run_dir / "P5D_DIAGNOSTIC_SUMMARY.json", summary)
    _write_json(
        run_dir / "P5D_DIAGNOSTIC_MANIFEST.json",
        {
            "task_id": TASK_ID,
            "source_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo_root, text=True
            ).strip(),
            "input_checkpoint": str(p5_run / "SURFNCC_BEST.pt"),
            "train_scenes": config["cohort"]["train_scenes"],
            "train_targets": config["cohort"]["targets"],
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
