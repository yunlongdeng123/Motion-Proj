"""F1: read-only SVD feature discriminability and projector-resolution audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..auditor.generated_tracks import RAFTChainGeneratedTrackProvider
from ..backbones import build_backbone
from ..cache.dataset import ProjectionCacheDataset
from ..config import config_fingerprint, get_paths, load_config, save_resolved_config
from ..projector.smoothing import smooth_tracks
from ..projector.support import classify_support
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.pilot import _to_batch
from ..train.trainer import seed_everything


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    values = values[torch.isfinite(values)]
    if not int(values.numel()):
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def _sample_features(
    features: torch.Tensor,
    points: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Bilinearly sample ``[T,C,Hf,Wf]`` at image-pixel ``[N,T,2]`` points."""
    if features.dim() != 4 or points.dim() != 3 or points.shape[1] != features.shape[0]:
        raise ValueError("feature/point shapes must be [T,C,Hf,Wf] and [N,T,2]")
    image_h, image_w = image_hw
    output = []
    for time in range(features.shape[0]):
        coordinates = points[:, time].to(features)
        grid = torch.stack([
            2.0 * (coordinates[:, 0] + 0.5) / float(image_w) - 1.0,
            2.0 * (coordinates[:, 1] + 0.5) / float(image_h) - 1.0,
        ], dim=-1).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(
            features[time:time + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[0, :, 0].transpose(0, 1)
        output.append(sampled)
    return torch.stack(output, dim=1)


def _feature_grid_pixels(
    feature_hw: tuple[int, int], image_hw: tuple[int, int], device: torch.device
) -> torch.Tensor:
    feature_h, feature_w = feature_hw
    image_h, image_w = image_hw
    y = (torch.arange(feature_h, device=device, dtype=torch.float32) + 0.5) * image_h / feature_h - 0.5
    x = (torch.arange(feature_w, device=device, dtype=torch.float32) + 0.5) * image_w / feature_w - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _cell_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    stride_yx: tuple[float, float],
) -> torch.Tensor:
    stride_y, stride_x = stride_yx
    delta = left.float() - right.float()
    return torch.sqrt((delta[..., 0] / stride_x).square() + (delta[..., 1] / stride_y).square())


def _gaussian_target_tv(
    observed: torch.Tensor,
    projected: torch.Tensor,
    grid_pixels: torch.Tensor,
    stride_yx: tuple[float, float],
) -> torch.Tensor:
    stride_y, stride_x = stride_yx

    def distribution(points: torch.Tensor) -> torch.Tensor:
        delta = grid_pixels.unsqueeze(0) - points.float().unsqueeze(1)
        distance_sq = (delta[..., 0] / stride_x).square() + (delta[..., 1] / stride_y).square()
        return torch.softmax(-0.5 * distance_sq, dim=-1)

    return 0.5 * (distribution(observed) - distribution(projected)).abs().sum(dim=-1)


def relation_metrics(
    features: torch.Tensor,
    observed: torch.Tensor,
    projected: torch.Tensor,
    valid: torch.Tensor,
    categories: list[str],
    *,
    image_hw: tuple[int, int],
    temperature: float,
    pck_radius_cells: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure whether frozen SVD features track and resolve projected corrections."""
    if temperature <= 0 or pck_radius_cells <= 0:
        raise ValueError("temperature and pck radius must be positive")
    time_count, _, feature_h, feature_w = features.shape
    if observed.shape != projected.shape or observed.shape[:2] != valid.shape:
        raise ValueError("track tensor shapes do not agree")
    image_h, image_w = image_hw
    stride = (image_h / feature_h, image_w / feature_w)
    normalized = F.normalize(features.float(), dim=1, eps=1.0e-8)
    query_points = observed[:, :1].expand(-1, time_count, -1)
    queries = F.normalize(_sample_features(normalized, query_points, image_hw)[:, 0], dim=-1, eps=1.0e-8)
    scores = torch.einsum("nc,tchw->nthw", queries, normalized)
    flattened = scores.flatten(2)
    probabilities = torch.softmax(flattened / temperature, dim=-1)
    grid_pixels = _feature_grid_pixels((feature_h, feature_w), image_hw, features.device)
    softargmax = torch.einsum("ntk,kd->ntd", probabilities, grid_pixels)
    argmax = grid_pixels[flattened.argmax(dim=-1)]
    observed_features = F.normalize(_sample_features(normalized, observed, image_hw), dim=-1, eps=1.0e-8)
    projected_features = F.normalize(_sample_features(normalized, projected, image_hw), dim=-1, eps=1.0e-8)
    query_expanded = queries[:, None]
    observed_correlation = (observed_features * query_expanded).sum(dim=-1)
    projected_correlation = (projected_features * query_expanded).sum(dim=-1)

    future = torch.arange(time_count, device=valid.device).reshape(1, -1) > 0
    selected = valid.bool() & future
    if not bool(selected.any()):
        raise RuntimeError("feature audit has no supported future track points")
    observed_selected = observed[selected]
    projected_selected = projected[selected]
    correction_cells = _cell_distance(observed_selected, projected_selected, stride)
    arg_observed = _cell_distance(argmax[selected], observed_selected, stride)
    arg_projected = _cell_distance(argmax[selected], projected_selected, stride)
    soft_observed = _cell_distance(softargmax[selected], observed_selected, stride)
    soft_projected = _cell_distance(softargmax[selected], projected_selected, stride)
    heatmap_tv = _gaussian_target_tv(
        observed_selected, projected_selected, grid_pixels, stride
    )

    track_scores: list[float] = []
    track_labels: list[int] = []
    stratum: dict[str, dict[str, float]] = {}
    for category in sorted(set(categories)):
        track_mask = torch.tensor([value == category for value in categories], device=valid.device)
        point_mask = selected & track_mask[:, None]
        if bool(point_mask.any()):
            stratum[category] = {
                "count": int(point_mask.sum()),
                "observed_correlation": float(observed_correlation[point_mask].mean()),
                "projected_correlation": float(projected_correlation[point_mask].mean()),
                "correction_cells_median": float(_cell_distance(
                    observed[point_mask], projected[point_mask], stride
                ).median()),
                "argmax_observed_pck_1cell": float(
                    (_cell_distance(argmax[point_mask], observed[point_mask], stride) <= pck_radius_cells)
                    .float().mean()
                ),
            }
    for index, category in enumerate(categories):
        point_mask = selected[index]
        if bool(point_mask.any()):
            track_scores.append(float((1.0 - observed_correlation[index, point_mask]).mean()))
            track_labels.append(0 if category == "background" else 1)

    visible = {
        "valid_point_count": int(selected.sum()),
        "feature_height": feature_h,
        "feature_width": feature_w,
        "stride_y": float(stride[0]),
        "stride_x": float(stride[1]),
        "observed_correlation": float(observed_correlation[selected].mean()),
        "projected_correlation": float(projected_correlation[selected].mean()),
        "observed_projected_correlation_abs_delta": float(
            (observed_correlation[selected] - projected_correlation[selected]).abs().mean()
        ),
        "observed_projected_heatmap_tv": float(heatmap_tv.mean()),
        "peak_correlation": float(flattened.max(dim=-1).values[selected].mean()),
        "correction_cells_mean": float(correction_cells.mean()),
        "correction_cells_median": float(correction_cells.median()),
        "correction_cells_p90": float(torch.quantile(correction_cells, 0.9)),
        "correction_below_half_cell_fraction": float((correction_cells < 0.5).float().mean()),
        "argmax_observed_pck_1cell": float((arg_observed <= pck_radius_cells).float().mean()),
        "argmax_projected_pck_1cell": float((arg_projected <= pck_radius_cells).float().mean()),
        "softargmax_observed_pck_1cell": float((soft_observed <= pck_radius_cells).float().mean()),
        "softargmax_projected_pck_1cell": float((soft_projected <= pck_radius_cells).float().mean()),
        "argmax_observed_error_cells": float(arg_observed.mean()),
        "argmax_projected_error_cells": float(arg_projected.mean()),
        "softargmax_observed_error_cells": float(soft_observed.mean()),
        "softargmax_projected_error_cells": float(soft_projected.mean()),
        "dynamic_background_feature_change_auc": _binary_auc(track_scores, track_labels),
        "stratum": stratum,
    }
    raw = {
        "correction_cells": correction_cells.detach().cpu(),
        "track_scores": track_scores,
        "track_labels": track_labels,
    }
    return visible, raw


def feature_route_decision(
    layer_rows: list[dict[str, Any]],
    *,
    max_fraction_below_half_cell: float,
    min_observed_pck_1cell: float,
) -> dict[str, Any]:
    resolution = [
        row for row in layer_rows
        if float(row["correction_below_half_cell_fraction"]) <= max_fraction_below_half_cell
    ]
    eligible = [
        row for row in resolution
        if float(row["argmax_observed_pck_1cell"]) >= min_observed_pck_1cell
    ]
    best_tracking = max(layer_rows, key=lambda row: float(row["argmax_observed_pck_1cell"]))
    if not resolution:
        classification = "feature_resolution_failure"
    elif not eligible:
        classification = "feature_discriminability_failure"
    else:
        classification = "feature_layer_eligible"
    recommended = None
    if eligible:
        recommended = max(
            eligible,
            key=lambda row: (
                float(row["argmax_projected_pck_1cell"]),
                -float(row["softargmax_projected_error_cells"]),
            ),
        )["layer"]
    return {
        "classification": classification,
        "passed": bool(eligible),
        "recommended_layer": recommended,
        "best_tracking_layer_diagnostic_only": best_tracking["layer"],
        "resolution_eligible_layers": [row["layer"] for row in resolution],
        "fully_eligible_layers": [row["layer"] for row in eligible],
    }


def _first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return _first_tensor(value)
            except TypeError:
                pass
    raise TypeError(f"hook output contains no tensor: {type(output)!r}")


class _FeatureCapture:
    def __init__(self, module: torch.nn.Module, layer_paths: dict[str, str]):
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []
        for alias, path in layer_paths.items():
            target = module.get_submodule(path)
            self.handles.append(target.register_forward_hook(self._hook(alias)))

    def _hook(self, alias: str):
        def save(_module, _inputs, output):
            self.outputs[alias] = _first_tensor(output).detach()
        return save

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _track_payload(index: int, item: dict[str, Any], state: Any, smooth_lambda: float) -> dict[str, Any]:
    metadata = item["metadata"]
    cached = metadata["projector_diagnostics"]["generated_tracks"]
    actual = state.diagnostics
    exact_keys = ("provider", "uses_future_gt", "query_count", "stratum_query_count", "valid_track_count")
    mismatches = {key: {"cached": cached.get(key), "actual": actual.get(key)} for key in exact_keys if cached.get(key) != actual.get(key)}
    if mismatches:
        raise RuntimeError(f"generated-track reconstruction mismatch for index {index}: {mismatches}")
    observed_tracks = state.tracks
    projected_tracks = smooth_tracks(observed_tracks, lam=smooth_lambda)
    height, width = item["base_rgb"].shape[-2:]
    support = classify_support(observed_tracks, (height, width))
    observed = torch.stack([track.center.detach().cpu() for track in observed_tracks])
    projected = torch.stack([track.center.detach().cpu() for track in projected_tracks])
    present = torch.stack([track.present.detach().cpu() for track in observed_tracks])
    projected_present = torch.stack([track.present.detach().cpu() for track in projected_tracks])
    supported = torch.stack([support[track.instance_token].detach().cpu() for track in observed_tracks])
    categories = [track.category.rsplit("/", 1)[-1] for track in observed_tracks]
    return {
        "dataset_index": index,
        "sample_id": metadata["sample_id"],
        "observed": observed,
        "projected": projected,
        "present": present,
        "projected_present": projected_present,
        "supported": supported,
        "categories": categories,
        "provider_diagnostics": actual,
        "cached_provider_diagnostics": cached,
        "image_hw": (height, width),
    }


def _consecutive_dynamics(points: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = points[1:] - points[:-1]
    velocity_valid = present[1:] & present[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    acceleration_valid = velocity_valid[1:] & velocity_valid[:-1]
    jerk = acceleration[1:] - acceleration[:-1]
    jerk_valid = acceleration_valid[1:] & acceleration_valid[:-1]
    return (
        torch.linalg.vector_norm(velocity[velocity_valid], dim=-1),
        torch.linalg.vector_norm(acceleration[acceleration_valid], dim=-1),
        torch.linalg.vector_norm(jerk[jerk_valid], dim=-1),
    )


def projector_track_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, category in enumerate(payload["categories"]):
        present = payload["present"][index]
        observed = payload["observed"][index]
        projected = payload["projected"][index]
        valid_indices = torch.nonzero(present, as_tuple=False).flatten()
        first, last = int(valid_indices[0]), int(valid_indices[-1])
        observed_net = observed[last] - observed[first]
        projected_net = projected[last] - projected[first]
        observed_norm = torch.linalg.vector_norm(observed_net)
        projected_norm = torch.linalg.vector_norm(projected_net)
        direction = float(torch.dot(observed_net, projected_net) / (observed_norm * projected_norm).clamp_min(1.0e-8))
        observed_v, observed_a, observed_j = _consecutive_dynamics(observed, present)
        projected_v, projected_a, projected_j = _consecutive_dynamics(projected, present)
        correction = torch.linalg.vector_norm(projected[present] - observed[present], dim=-1)

        def rms(value: torch.Tensor) -> float:
            return float(value.square().mean().sqrt()) if int(value.numel()) else float("nan")

        def turn_sign(points: torch.Tensor) -> float:
            velocity = points[1:] - points[:-1]
            if velocity.shape[0] < 2:
                return 0.0
            cross = velocity[:-1, 0] * velocity[1:, 1] - velocity[:-1, 1] * velocity[1:, 0]
            return float(torch.sign(cross.sum()))

        obs_turn, proj_turn = turn_sign(observed[present]), turn_sign(projected[present])
        rows.append({
            "dataset_index": payload["dataset_index"],
            "sample_id": payload["sample_id"],
            "track_index": index,
            "category": category,
            "present_count": int(present.sum()),
            "projected_present_count": int(payload["projected_present"][index].sum()),
            "supported_count": int((present & payload["supported"][index]).sum()),
            "correction_px_mean": float(correction.mean()),
            "correction_px_max": float(correction.max()),
            "frame0_correction_px": float(torch.linalg.vector_norm(projected[0] - observed[0])),
            "net_displacement_observed_px": float(observed_norm),
            "net_displacement_projected_px": float(projected_norm),
            "net_displacement_ratio": float(projected_norm / observed_norm.clamp_min(1.0e-8)),
            "average_velocity_observed_px": float(observed_norm / max(last - first, 1)),
            "average_velocity_projected_px": float(projected_norm / max(last - first, 1)),
            "direction_cosine": direction,
            "turn_direction_observed": obs_turn,
            "turn_direction_projected": proj_turn,
            "turn_direction_preserved": bool(obs_turn == proj_turn),
            "velocity_rms_observed_px": rms(observed_v),
            "velocity_rms_projected_px": rms(projected_v),
            "acceleration_rms_observed_px": rms(observed_a),
            "acceleration_rms_projected_px": rms(projected_a),
            "jerk_rms_observed_px": rms(observed_j),
            "jerk_rms_projected_px": rms(projected_j),
        })
    return rows


def _aggregate_projector(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in ["all", "background", "dynamic_residual", "foreground_candidate"]:
        selected = rows if category == "all" else [row for row in rows if row["category"] == category]
        if not selected:
            continue
        result[category] = {
            "track_count": len(selected),
            "correction_px": _quantiles(torch.tensor([row["correction_px_mean"] for row in selected])),
            "frame0_correction_px": _quantiles(torch.tensor([row["frame0_correction_px"] for row in selected])),
            "net_displacement_ratio": _quantiles(torch.tensor([row["net_displacement_ratio"] for row in selected])),
            "direction_cosine": _quantiles(torch.tensor([row["direction_cosine"] for row in selected])),
            "turn_direction_preserved_fraction": sum(bool(row["turn_direction_preserved"]) for row in selected) / len(selected),
            "visibility_expansion_fraction": sum(row["projected_present_count"] - row["present_count"] for row in selected) / max(sum(row["present_count"] for row in selected), 1),
            "support_fraction": sum(row["supported_count"] for row in selected) / max(sum(row["present_count"] for row in selected), 1),
            "acceleration_rms_ratio_median": float(torch.tensor([
                row["acceleration_rms_projected_px"] / max(row["acceleration_rms_observed_px"], 1.0e-8)
                for row in selected if math.isfinite(row["acceleration_rms_observed_px"])
            ]).median()),
            "jerk_rms_ratio_median": float(torch.tensor([
                row["jerk_rms_projected_px"] / max(row["jerk_rms_observed_px"], 1.0e-8)
                for row in selected if math.isfinite(row["jerk_rms_observed_px"])
            ]).median()),
        }
    return result


def _weighted_layer_summary(rows: list[dict[str, Any]], raw: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(row["valid_point_count"]) for row in rows)
    weighted_keys = (
        "observed_correlation", "projected_correlation",
        "observed_projected_correlation_abs_delta", "observed_projected_heatmap_tv",
        "peak_correlation", "argmax_observed_pck_1cell", "argmax_projected_pck_1cell",
        "softargmax_observed_pck_1cell", "softargmax_projected_pck_1cell",
        "argmax_observed_error_cells", "argmax_projected_error_cells",
        "softargmax_observed_error_cells", "softargmax_projected_error_cells",
    )
    output = {
        key: sum(float(row[key]) * int(row["valid_point_count"]) for row in rows) / max(count, 1)
        for key in weighted_keys
    }
    corrections = torch.cat([value["correction_cells"] for value in raw])
    scores = [score for value in raw for score in value["track_scores"]]
    labels = [label for value in raw for label in value["track_labels"]]
    output.update({
        "valid_point_count": count,
        "feature_height": rows[0]["feature_height"],
        "feature_width": rows[0]["feature_width"],
        "stride_y": rows[0]["stride_y"],
        "stride_x": rows[0]["stride_x"],
        "correction_cells_mean": float(corrections.mean()),
        "correction_cells_median": float(corrections.median()),
        "correction_cells_p90": float(torch.quantile(corrections, 0.9)),
        "correction_below_half_cell_fraction": float((corrections < 0.5).float().mean()),
        "dynamic_background_feature_change_auc": _binary_auc(scores, labels),
    })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def _write_figures(layer_rows: list[dict[str, Any]], projector: dict[str, Any], directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aliases = [row["layer"] for row in layer_rows]
    x = range(len(aliases))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, [row["correction_below_half_cell_fraction"] for row in layer_rows])
    axes[0].axhline(0.5, color="red", linestyle="--")
    axes[0].set_ylabel("fraction correction < 0.5 feature cell")
    axes[1].bar([value - 0.2 for value in x], [row["argmax_observed_pck_1cell"] for row in layer_rows], width=0.4, label="observed")
    axes[1].bar([value + 0.2 for value in x], [row["argmax_projected_pck_1cell"] for row in layer_rows], width=0.4, label="projected")
    axes[1].set_ylabel("argmax PCK @ 1 cell")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(list(x), aliases, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "feature_resolution_and_pck.png", dpi=160)
    plt.close(fig)

    categories = [key for key in ("background", "dynamic_residual", "foreground_candidate") if key in projector]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(categories, [projector[key]["net_displacement_ratio"]["median"] for key in categories])
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set_ylabel("median projected / observed net displacement")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(directory / "projector_displacement_preservation.png", dpi=160)
    plt.close(fig)


def run_feature_discriminability(cfg: Any) -> dict[str, Any]:
    f1 = cfg.f1
    indices = [int(value) for value in f1.dataset_indices]
    if not 1 <= len(indices) <= 24 or len(indices) != len(set(indices)):
        raise ValueError("F1 requires 1-24 unique replay indices")
    sigmas = [float(value) for value in f1.sigmas]
    if not sigmas or any(value <= 0 for value in sigmas):
        raise ValueError("F1 sigmas must be positive")
    layer_paths = {str(key): str(value) for key, value in f1.feature_layers.items()}
    if not layer_paths:
        raise ValueError("F1 requires feature layers")
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("formal F1 refuses to run in a dirty worktree")
    work_dir = Path(cfg.work_dir)
    if work_dir.exists():
        raise RuntimeError(f"F1 run directory already exists: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)
    figures = work_dir / "figures"
    figures.mkdir()
    paths = get_paths(cfg)
    dataset = ProjectionCacheDataset(
        str(paths.cache_dir), expected_fingerprint=str(f1.cache_fingerprint)
    )
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("F1 replay index is outside cache")

    config_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=config_fp,
        cache_fingerprint=str(f1.cache_fingerprint), seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split=str(cfg.data.split),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(f1.task_id),
        "dataset_indices": indices,
        "sigmas": sigmas,
        "feature_layers": layer_paths,
        "preregistration": {
            "max_fraction_below_half_cell": float(f1.max_fraction_below_half_cell),
            "min_observed_pck_1cell": float(f1.min_observed_pck_1cell),
            "pck_radius_cells": float(f1.pck_radius_cells),
            "stop_rule": "stop before F2 if no layer resolves at least half of projected corrections",
        },
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))

    try:
        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        options = OmegaConf.to_container(cfg.auditor.generated_tracks, resolve=True)
        options = dict(options)
        options.pop("provider", None)
        provider = RAFTChainGeneratedTrackProvider(device=str(cfg.device), **options)
        payloads = []
        projector_rows: list[dict[str, Any]] = []
        reconstruction_rows = []
        for index in indices:
            item = dataset[index]
            metadata = item["metadata"]
            if (
                metadata.get("source") != "replay_v2"
                or metadata.get("parent_kind") != "base"
                or bool(metadata.get("adapter_loaded"))
                or bool(metadata.get("uses_future_gt_ego"))
                or bool(metadata.get("uses_future_gt_track"))
            ):
                raise RuntimeError(f"F1 index {index} is not leakage-free frozen-Base replay")
            state = provider.track(item["base_rgb"])
            if state.uses_future_gt:
                raise RuntimeError("F1 generated track provider reported future-GT use")
            payload = _track_payload(index, item, state, float(f1.smooth_lambda))
            payloads.append(payload)
            rows = projector_track_rows(payload)
            projector_rows.extend(rows)
            reconstruction = {
                "dataset_index": index,
                "sample_id": payload["sample_id"],
                "track_count": len(payload["categories"]),
                "category_counts": {
                    category: payload["categories"].count(category)
                    for category in sorted(set(payload["categories"]))
                },
                "provider_diagnostics": payload["provider_diagnostics"],
                "cache_match": True,
            }
            reconstruction_rows.append(reconstruction)
            metrics.append(index, {"phase": "track_reconstruction", **reconstruction})
        del provider
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        noise_rows = []
        for payload in payloads:
            item = dataset[payload["dataset_index"]]
            base = item["base_latent"].float()
            generator = torch.Generator(device="cpu").manual_seed(
                int(f1.noise_seed) + payload["dataset_index"]
            )
            noise_rows.append({
                "dataset_index": payload["dataset_index"],
                "sample_id": payload["sample_id"],
                "noise": torch.randn(base.shape, generator=generator, dtype=base.dtype),
            })
        noise_path = work_dir / "noise_bank.pt"
        torch.save({"noise_seed": int(f1.noise_seed), "sigmas": sigmas, "rows": noise_rows}, noise_path)
        noise_fingerprint = file_fingerprint(str(noise_path))
        manifest_data["noise_bank_fingerprint"] = noise_fingerprint
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.lora.enable = False
        model_cfg.gradient_checkpointing = False
        backbone = build_backbone(model_cfg, load=True, device=str(cfg.device))
        backbone.unet.eval()
        for parameter in backbone.unet.parameters():
            parameter.requires_grad_(False)
        capture = _FeatureCapture(backbone.unet, layer_paths)
        layer_metric_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        layer_raw_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        device = torch.device(str(cfg.device))
        try:
            for payload, noise_row in zip(payloads, noise_rows):
                item = dataset[payload["dataset_index"]]
                for sigma_value in sigmas:
                    base = item["base_latent"].float()
                    bank = {
                        "sigma": torch.tensor([sigma_value], dtype=base.dtype),
                        "noise": noise_row["noise"],
                        "z_sigma": base + sigma_value * noise_row["noise"],
                    }
                    batch = _to_batch(item, bank, device)
                    capture.clear()
                    with torch.no_grad():
                        backbone.predict_model_output(batch["z"], batch["sigma"], batch["condition"])
                    if set(capture.outputs) != set(layer_paths):
                        raise RuntimeError(
                            f"feature hooks missing: expected={sorted(layer_paths)}, actual={sorted(capture.outputs)}"
                        )
                    for alias, tensor in capture.outputs.items():
                        if tensor.dim() != 4 or tensor.shape[0] != int(cfg.model.num_frames):
                            raise RuntimeError(f"unexpected feature shape for {alias}: {tuple(tensor.shape)}")
                        result, raw = relation_metrics(
                            tensor,
                            payload["observed"].to(device),
                            payload["projected"].to(device),
                            (payload["present"] & payload["supported"]).to(device),
                            payload["categories"],
                            image_hw=payload["image_hw"],
                            temperature=float(f1.temperature),
                            pck_radius_cells=float(f1.pck_radius_cells),
                        )
                        row = {
                            "phase": "feature_relation",
                            "dataset_index": payload["dataset_index"],
                            "sample_id": payload["sample_id"],
                            "sigma": sigma_value,
                            "layer": alias,
                            "module_path": layer_paths[alias],
                            **result,
                        }
                        layer_metric_rows[alias].append(row)
                        layer_raw_rows[alias].append(raw)
                        metrics.append(payload["dataset_index"], row)
                    del batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            capture.close()

        aggregate_layers = []
        for alias in layer_paths:
            aggregate = _weighted_layer_summary(layer_metric_rows[alias], layer_raw_rows[alias])
            aggregate_layers.append({
                "layer": alias,
                "module_path": layer_paths[alias],
                **aggregate,
            })
        decision = feature_route_decision(
            aggregate_layers,
            max_fraction_below_half_cell=float(f1.max_fraction_below_half_cell),
            min_observed_pck_1cell=float(f1.min_observed_pck_1cell),
        )
        projector_summary = _aggregate_projector(projector_rows)

        atomic_write_text(
            str(work_dir / "track_reconstruction.jsonl"),
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reconstruction_rows),
        )
        projector_fields = list(projector_rows[0])
        _write_csv(work_dir / "projector_track_audit.csv", projector_rows, projector_fields)
        layer_fields = [
            "layer", "module_path", "feature_height", "feature_width", "stride_y", "stride_x",
            "valid_point_count", "correction_cells_mean", "correction_cells_median",
            "correction_cells_p90", "correction_below_half_cell_fraction",
            "observed_correlation", "projected_correlation",
            "observed_projected_correlation_abs_delta", "observed_projected_heatmap_tv",
            "argmax_observed_pck_1cell", "argmax_projected_pck_1cell",
            "softargmax_observed_pck_1cell", "softargmax_projected_pck_1cell",
            "argmax_observed_error_cells", "argmax_projected_error_cells",
            "softargmax_observed_error_cells", "softargmax_projected_error_cells",
            "dynamic_background_feature_change_auc",
        ]
        _write_csv(work_dir / "layer_summary.csv", aggregate_layers, layer_fields)
        _write_figures(aggregate_layers, projector_summary, figures)
        summary = {
            "status": "completed",
            "task_id": str(f1.task_id),
            "dataset_indices": indices,
            "sample_count": len(indices),
            "sigmas": sigmas,
            "noise_bank_fingerprint": noise_fingerprint,
            "track_reconstruction": reconstruction_rows,
            "projector_audit": projector_summary,
            "layers": aggregate_layers,
            "decision": decision,
            "experiment_fingerprint": sha256_json({
                "config": config_fp, "noise": noise_fingerprint,
                "layers": aggregate_layers, "projector": projector_summary,
                "decision": decision,
            }),
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({
            "status": "completed", "ended_at": utc_now(),
            "exit_reason": decision["classification"],
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        atomic_write_json(str(work_dir / "summary.json"), {"status": "failed", "error": repr(exc)})
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    result = run_feature_discriminability(load_config(args.config, list(args.overrides)))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
