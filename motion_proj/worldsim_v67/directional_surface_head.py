"""Learned residual rescue on top of motion-compensated inward-ray support."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v64.native_voxel_uq import _native_unit_dir, _unit_dirs
from motion_proj.worldsim_v66.sensor_surface_repair import _load_repair_unit


GEOMETRY_FEATURE_NAMES = (
    "nearest_hit_distance_voxels",
    "ray_longitudinal_voxels",
    "ray_transverse_voxels",
    "delta_x_voxels",
    "delta_y_voxels",
    "delta_z_voxels",
    "ray_direction_x",
    "ray_direction_y",
    "ray_direction_z",
    "exact_same_actor_hit",
    "source_behind_hit",
)


class DirectionalSurfaceHead(torch.nn.Module):
    """Small implicit field that only rescues points rejected by the analytic core."""

    def __init__(self, input_dimension: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        dimensions = [int(input_dimension), *(int(v) for v in hidden_dimensions), 1]
        layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(torch.nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.extend((torch.nn.LayerNorm(right), torch.nn.GELU()))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).reshape(-1)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _geometry_features(loaded: Mapping[str, Any], radius: float) -> np.ndarray:
    centers = np.asarray(loaded["centers"], dtype=np.float64)
    actor_ids = np.asarray(loaded["actor_ids"], dtype=np.int64)
    hit_indices = np.asarray(loaded["actor_hit_indices"], dtype=np.int64)
    hit_ids = np.asarray(loaded["actor_hit_ids"], dtype=np.int64)
    origin = np.asarray(loaded["evidence_origin_m"], dtype=np.float64)
    voxel = float(loaded["evidence_voxel_size_m"])
    scale = max(float(radius), 1e-6)
    geometry = np.zeros((len(centers), len(GEOMETRY_FEATURE_NAMES)), dtype=np.float32)
    geometry[:, 0] = 4.0
    for actor_id in np.unique(actor_ids[actor_ids >= 0]):
        members = actor_ids == int(actor_id)
        actor_hits = hit_ids == int(actor_id)
        if not np.any(actor_hits):
            continue
        hit_centers = origin[None, :] + (hit_indices[actor_hits] + 0.5) * voxel
        query_centers = centers[members]
        distances, nearest = cKDTree(hit_centers).query(query_centers, k=1)
        nearest_hits = hit_centers[nearest]
        delta = query_centers - nearest_hits
        ray = nearest_hits / np.maximum(np.linalg.norm(nearest_hits, axis=1, keepdims=True), 1e-8)
        longitudinal = np.einsum("ij,ij->i", delta, ray)
        transverse = np.sqrt(np.maximum(np.sum(delta * delta, axis=1) - longitudinal**2, 0.0))
        positions = np.flatnonzero(members)
        geometry[positions, 0] = np.clip(distances / scale, 0.0, 4.0)
        geometry[positions, 1] = np.clip(longitudinal / scale, -4.0, 4.0)
        geometry[positions, 2] = np.clip(transverse / scale, 0.0, 4.0)
        geometry[positions, 3:6] = np.clip(delta / scale, -4.0, 4.0)
        geometry[positions, 6:9] = ray
    geometry[:, 9] = np.asarray(loaded["exact_same_actor_hit"], dtype=np.float32)
    geometry[:, 10] = np.asarray(loaded["behind_hit"], dtype=np.float32)
    return geometry


def materialize_points(
    cohort: Mapping[str, Any],
    *,
    seed: int,
    native_grid: Mapping[str, Any],
    sampling: Mapping[str, Any],
    support_radius_m: float,
    runs_root: Path,
) -> dict[str, Any]:
    evidence_root = runs_root / str(cohort["evidence_run"])
    native_root = runs_root / str(cohort["native_run"])
    partition = str(cohort["native_partition"])
    origin = np.asarray(native_grid["origin_m"], dtype=np.float64)
    voxel_size = float(native_grid["voxel_size_m"])
    descriptors = []
    for scene_index, scene in enumerate(cohort["scenes"]):
        for evidence_unit in _unit_dirs(evidence_root, str(scene)):
            descriptors.append(
                (
                    scene_index,
                    str(scene),
                    evidence_unit,
                    _native_unit_dir(native_root, str(scene), evidence_unit.name, {str(scene): partition}),
                )
            )

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(sampling["io_prefetch_workers"])
    )

    def submit(descriptor: tuple[int, str, Path, Path]):
        return executor.submit(
            _load_repair_unit,
            descriptor,
            origin=origin,
            voxel_size=voxel_size,
            point_limit=int(sampling["maximum_boundary_points_per_unit"]),
            seed=int(seed),
            support_radius_m=float(support_radius_m),
            support_expansion_requires_behind_hit=False,
            support_expansion_motion_compensated_inward_ray=True,
        )

    values, conflicts, analytic, base_ids, scenes = [], [], [], [], []
    future = submit(descriptors[0])
    try:
        for position, descriptor in enumerate(descriptors):
            loaded = future.result()
            if position + 1 < len(descriptors):
                future = submit(descriptors[position + 1])
            actor_ids = np.asarray(loaded["actor_ids"], dtype=np.int64)
            owned = actor_ids >= 0
            native = np.asarray(loaded["native_features"], dtype=np.float32)
            geometry = _geometry_features(loaded, float(support_radius_m))
            values.append(np.concatenate((native[owned], geometry[owned]), axis=1))
            conflicts.append(np.asarray(loaded["labels"], dtype=bool)[owned])
            analytic.append(np.asarray(loaded["same_actor_hit"], dtype=bool)[owned])
            owned_ids = actor_ids[owned]
            base_ids.extend(
                f"{descriptor[1]}/{descriptor[2].name}/actor-{int(actor_id)}"
                for actor_id in owned_ids
            )
            scenes.extend([descriptor[1]] * int(np.count_nonzero(owned)))
            print(
                f"DSH data {position + 1}/{len(descriptors)} scene={descriptor[1]} "
                f"unit={descriptor[2].name} points={int(np.count_nonzero(owned))}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)
    return {
        "values": np.concatenate(values).astype(np.float32),
        "conflicts": np.concatenate(conflicts),
        "analytic_support": np.concatenate(analytic),
        "base_ids": np.asarray(base_ids),
        "scenes": np.asarray(scenes),
        "source_unit_count": len(descriptors),
    }


def train_residual_head(
    values: np.ndarray,
    conflicts: np.ndarray,
    analytic_support: np.ndarray,
    model_config: Mapping[str, Any],
    seed: int,
) -> tuple[DirectionalSurfaceHead, np.ndarray, np.ndarray, float, dict[str, Any]]:
    residual = ~np.asarray(analytic_support, dtype=bool)
    x_np = np.asarray(values[residual], dtype=np.float32)
    conflict_np = np.asarray(conflicts[residual], dtype=bool)
    clean_np = ~conflict_np
    mean = x_np.mean(axis=0)
    scale = np.maximum(x_np.std(axis=0), 1e-5)
    x = torch.from_numpy((x_np - mean) / scale).cuda()
    y = torch.from_numpy(clean_np.astype(np.float32)).cuda()
    conflict = torch.from_numpy(conflict_np).cuda()
    torch.manual_seed(int(seed))
    model = DirectionalSurfaceHead(x.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    batch_size = int(model_config["batch_size"])
    conflict_weight = float(model_config["conflict_example_weight"])
    penalty_weight = float(model_config["conflict_rescue_penalty_weight"])
    maximum_soft_rescue = float(model_config["maximum_soft_conflict_rescue"])
    final_loss = float("nan")
    model.train()
    for _ in range(int(model_config["epochs"])):
        order = torch.randperm(len(x), device="cuda")
        for offset in range(0, len(x), batch_size):
            members = order[offset : offset + batch_size]
            logits = model(x[members])
            target = y[members]
            is_conflict = conflict[members]
            weights = torch.where(is_conflict, conflict_weight, 1.0)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, weight=weights
            )
            probabilities = torch.sigmoid(logits)
            if torch.any(is_conflict):
                conflict_rescue = probabilities[is_conflict].mean()
                penalty = torch.relu(conflict_rescue - maximum_soft_rescue).square()
            else:
                penalty = logits.new_zeros(())
            loss = bce + penalty_weight * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    model.eval()
    with torch.inference_mode():
        train_scores = torch.sigmoid(model(x)).float().cpu().numpy()
    conflict_scores = train_scores[conflict_np]
    threshold = max(
        float(model_config["minimum_probability_threshold"]),
        float(
            np.quantile(
                conflict_scores,
                1.0 - float(model_config["maximum_training_conflict_rescue_fraction"]),
            )
        ),
    )
    train_prediction = train_scores >= threshold
    return model, mean, scale, threshold, {
        "residual_point_count": int(len(x_np)),
        "residual_clean_count": int(np.count_nonzero(clean_np)),
        "residual_conflict_count": int(np.count_nonzero(conflict_np)),
        "final_loss": final_loss,
        "clean_auroc": float(roc_auc_score(clean_np, train_scores)),
        "clean_auprc": float(average_precision_score(clean_np, train_scores)),
        "threshold": threshold,
        "training_conflict_rescue_fraction": float(np.mean(train_prediction[conflict_np])),
        "training_clean_rescue_fraction": float(np.mean(train_prediction[clean_np])),
    }


def score_residual_head(
    model: DirectionalSurfaceHead,
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    scores = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(values), 65536):
            x = torch.from_numpy((values[offset : offset + 65536] - mean) / scale).cuda()
            scores.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(scores)


def train_residual_head_crossfit(
    values: np.ndarray,
    conflicts: np.ndarray,
    analytic_support: np.ndarray,
    scenes: np.ndarray,
    model_config: Mapping[str, Any],
    seed: int,
) -> tuple[DirectionalSurfaceHead, np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Calibrate rescue risk on held-out scenes, then fit the final model on all scenes."""
    residual = ~np.asarray(analytic_support, dtype=bool)
    scene_values = np.asarray(scenes)
    fold_scores, fold_conflicts = [], []
    fold_rows = []
    fold_config = dict(model_config)
    fold_config["epochs"] = int(model_config["crossfit_epochs"])
    for fold_index, held_scene in enumerate(sorted(set(scene_values.tolist()))):
        train_members = scene_values != held_scene
        held_members = (scene_values == held_scene) & residual
        model, mean, scale, _, metrics = train_residual_head(
            values[train_members],
            conflicts[train_members],
            analytic_support[train_members],
            fold_config,
            int(seed) + fold_index + 1,
        )
        scores = score_residual_head(model, values[held_members], mean, scale)
        held_conflicts = np.asarray(conflicts[held_members], dtype=bool)
        fold_scores.append(scores)
        fold_conflicts.append(held_conflicts)
        fold_rows.append(
            {
                "held_scene": str(held_scene),
                "point_count": int(len(scores)),
                "conflict_count": int(np.count_nonzero(held_conflicts)),
                "clean_count": int(np.count_nonzero(~held_conflicts)),
                "clean_auroc": float(roc_auc_score(~held_conflicts, scores))
                if np.unique(held_conflicts).size == 2
                else None,
                "fit_final_loss": metrics["final_loss"],
            }
        )
        del model
    crossfit_scores = np.concatenate(fold_scores)
    crossfit_conflicts = np.concatenate(fold_conflicts)
    conflict_scores = crossfit_scores[crossfit_conflicts]
    threshold = max(
        float(model_config["minimum_probability_threshold"]),
        float(
            np.quantile(
                conflict_scores,
                1.0 - float(model_config["maximum_training_conflict_rescue_fraction"]),
            )
        ),
    )
    final_model, mean, scale, _, training = train_residual_head(
        values, conflicts, analytic_support, model_config, int(seed)
    )
    crossfit_prediction = crossfit_scores >= threshold
    training.update(
        {
            "threshold": threshold,
            "threshold_source": "leave_one_scene_out_conflict_scores",
            "crossfit_point_count": int(len(crossfit_scores)),
            "crossfit_conflict_count": int(np.count_nonzero(crossfit_conflicts)),
            "crossfit_clean_count": int(np.count_nonzero(~crossfit_conflicts)),
            "crossfit_conflict_rescue_fraction": float(
                np.mean(crossfit_prediction[crossfit_conflicts])
            ),
            "crossfit_clean_rescue_fraction": float(
                np.mean(crossfit_prediction[~crossfit_conflicts])
            ),
            "crossfit_clean_auroc": float(
                roc_auc_score(~crossfit_conflicts, crossfit_scores)
            ),
            "crossfit_clean_auprc": float(
                average_precision_score(~crossfit_conflicts, crossfit_scores)
            ),
            "crossfit_folds": fold_rows,
        }
    )
    return final_model, mean, scale, threshold, training


def evaluate_support(
    points: Mapping[str, Any],
    scores: np.ndarray,
    threshold: float,
    action_rows_path: Path,
    action_arm: str,
) -> dict[str, Any]:
    action_rows = _jsonl(action_rows_path)
    actions = {
        str(row["base_id"]): str(row["local_action"])
        for row in action_rows
        if str(row["arm"]) == str(action_arm)
    }
    eligible = np.asarray(
        [str(base_id) in actions for base_id in points["base_ids"]], dtype=bool
    )
    excluded_point_count = int(np.count_nonzero(~eligible))
    base_ids = np.asarray(points["base_ids"])[eligible]
    acted = np.asarray(
        [actions[str(base_id)] == "RANK_REPAIR_OR_ABSTAIN" for base_id in base_ids],
        dtype=bool,
    )
    conflict = np.asarray(points["conflicts"], dtype=bool)[eligible]
    analytic_local = np.asarray(points["analytic_support"], dtype=bool)[eligible]
    scores = np.asarray(scores)[eligible]
    analytic_keep = (~acted) | analytic_local
    rescue = acted & (~analytic_local) & (scores >= float(threshold))
    learned_keep = analytic_keep | rescue

    def metrics(keep: np.ndarray) -> dict[str, float | int]:
        return {
            "retained_point_count": int(np.count_nonzero(keep)),
            "overall_boundary_retention": float(np.mean(keep)),
            "retained_conflict_point_count": int(np.count_nonzero(keep & conflict)),
            "conflict_point_reduction": float(1.0 - np.mean(keep[conflict])),
            "retained_clean_point_count": int(np.count_nonzero(keep & ~conflict)),
            "clean_boundary_retention": float(np.mean(keep[~conflict])),
        }

    analytic_metrics = metrics(analytic_keep)
    learned_metrics = metrics(learned_keep)
    rescued_count = int(np.count_nonzero(rescue))
    rescued_clean = int(np.count_nonzero(rescue & ~conflict))
    rescued_conflict = int(np.count_nonzero(rescue & conflict))
    return {
        "source_unit_count": int(points["source_unit_count"]),
        "excluded_ineligible_point_count": excluded_point_count,
        "actor_boundary_point_count": int(len(conflict)),
        "conflict_point_count": int(np.count_nonzero(conflict)),
        "clean_point_count": int(np.count_nonzero(~conflict)),
        "acted_point_count": int(np.count_nonzero(acted)),
        "analytic": analytic_metrics,
        "learned_residual": learned_metrics,
        "clean_retention_improvement_over_analytic": float(
            learned_metrics["clean_boundary_retention"]
            - analytic_metrics["clean_boundary_retention"]
        ),
        "conflict_reduction_delta_over_analytic": float(
            learned_metrics["conflict_point_reduction"]
            - analytic_metrics["conflict_point_reduction"]
        ),
        "rescued_point_count": rescued_count,
        "rescued_clean_point_count": rescued_clean,
        "rescued_conflict_point_count": rescued_conflict,
        "rescue_precision_clean": float(rescued_clean / rescued_count) if rescued_count else 0.0,
        "actor_existence_mutated": False,
        "analytic_core_removed": False,
    }
