"""Pure evaluation operators for the frozen V5.1 LUDVIG uplift gate."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from motion_proj.worldsim_v51.feature_uplift import sample_patch_grid_bilinear


def row_cosine(
    left: np.ndarray, right: np.ndarray, *, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-wise cosine and the finite, non-zero common denominator."""

    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.ndim != 2 or rhs.shape != lhs.shape:
        raise ValueError("cosine inputs must share a two-dimensional shape")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("cosine inputs must be finite")
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("cosine epsilon must be positive and finite")
    lhs_norm = np.linalg.norm(lhs, axis=1)
    rhs_norm = np.linalg.norm(rhs, axis=1)
    valid = (lhs_norm > float(epsilon)) & (rhs_norm > float(epsilon))
    values = np.full(lhs.shape[0], np.nan, dtype=np.float64)
    values[valid] = np.einsum(
        "ij,ij->i", lhs[valid], rhs[valid], dtype=np.float64
    ) / (lhs_norm[valid] * rhs_norm[valid])
    values[valid] = np.clip(values[valid], -1.0, 1.0)
    return values, valid


def deterministic_actor_pairs(
    gaussian_indices: np.ndarray,
    *,
    seed: int,
    maximum_pairs: int,
) -> np.ndarray:
    """Select stable unordered Gaussian pairs without materializing all large combinations."""

    indices = np.asarray(gaussian_indices, dtype=np.int64)
    if indices.ndim != 1 or np.unique(indices).size != indices.size:
        raise ValueError("actor Gaussian indices must be one-dimensional and unique")
    if indices.size < 2 or int(maximum_pairs) <= 0:
        return np.empty((0, 2), dtype=np.int64)
    indices = np.sort(indices)
    total = indices.size * (indices.size - 1) // 2
    target = min(int(maximum_pairs), int(total))
    if total <= int(maximum_pairs):
        return np.asarray(list(combinations(indices.tolist(), 2)), dtype=np.int64)

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    selected: set[tuple[int, int]] = set()
    while len(selected) < target:
        batch = max(64, 2 * (target - len(selected)))
        draws = rng.integers(0, indices.size, size=(batch, 2), endpoint=False)
        for first_pos, second_pos in draws:
            if first_pos == second_pos:
                continue
            first = int(indices[min(first_pos, second_pos)])
            second = int(indices[max(first_pos, second_pos)])
            selected.add((first, second))
            if len(selected) == target:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def derived_actor_seed(*, base_seed: int, scene: str, actor_id: int) -> int:
    payload = f"{int(base_seed)}|{scene}|{int(actor_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def single_view_gaussian_feature(
    *,
    gaussian_id: np.ndarray,
    pixel_id: np.ndarray,
    contribution_weight: np.ndarray,
    patch_grid: np.ndarray,
    gaussian_count: int,
    image_height: int,
    image_width: int,
    minimum_intersection_contribution: float,
    minimum_gaussian_view_mass: float,
    epsilon: float,
) -> dict[str, Any]:
    """Apply one frozen renderer-transpose view and return only covered rows."""

    gids = np.asarray(gaussian_id, dtype=np.int64)
    pixels = np.asarray(pixel_id, dtype=np.int64)
    weights = np.asarray(contribution_weight, dtype=np.float64)
    grid = np.asarray(patch_grid)
    if gids.ndim != 1 or pixels.shape != gids.shape or weights.shape != gids.shape:
        raise ValueError("intersection arrays must align")
    if grid.ndim != 3 or not np.issubdtype(grid.dtype, np.floating):
        raise ValueError("patch grid must be floating [D,H,W]")
    if np.any((gids < 0) | (gids >= int(gaussian_count))):
        raise ValueError("Gaussian id is out of range")
    pixel_count = int(image_height) * int(image_width)
    if np.any((pixels < 0) | (pixels >= pixel_count)):
        raise ValueError("pixel id is out of range")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("contribution weights must be finite and non-negative")

    selected = weights >= float(minimum_intersection_contribution)
    operator = sparse.coo_matrix(
        (weights[selected], (gids[selected], pixels[selected])),
        shape=(int(gaussian_count), pixel_count),
        dtype=np.float64,
    ).tocsr()
    mass = np.asarray(operator.sum(axis=1)).reshape(-1)
    covered_ids = np.flatnonzero(mass >= float(minimum_gaussian_view_mass))
    if covered_ids.size:
        dense_pixel_feature = sample_patch_grid_bilinear(
            grid,
            np.arange(pixel_count, dtype=np.int64),
            image_height=int(image_height),
            image_width=int(image_width),
        ).astype(np.float64)
        numerator = np.asarray(operator[covered_ids] @ dense_pixel_feature, dtype=np.float64)
        feature = numerator / (mass[covered_ids, None] + float(epsilon))
        feature = np.ascontiguousarray(feature, dtype=np.float32)
    else:
        feature = np.empty((0, grid.shape[0]), dtype=np.float32)
    return {
        "gaussian_id": covered_ids.astype(np.int64, copy=False),
        "feature": feature,
        "mass": mass[covered_ids],
        "input_intersection_count": int(gids.size),
        "supported_intersection_count": int(selected.sum()),
        "covered_gaussian_count": int(covered_ids.size),
    }


def repeatability_against_aggregate(
    *,
    aggregate_feature: np.ndarray,
    aggregate_covered: np.ndarray,
    view_gaussian_id: np.ndarray,
    view_feature: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    ids = np.asarray(view_gaussian_id, dtype=np.int64)
    covered = np.asarray(aggregate_covered, dtype=bool)
    selected = covered[ids]
    values, valid = row_cosine(
        np.asarray(aggregate_feature)[ids[selected]],
        np.asarray(view_feature)[selected],
        epsilon=float(epsilon),
    )
    valid_values = values[valid]
    return {
        "common_covered_count": int(selected.sum()),
        "valid_cosine_count": int(valid.sum()),
        "mean_cosine": float(valid_values.mean()) if valid_values.size else None,
    }


def actor_feature_metrics(
    *,
    feature: np.ndarray,
    covered: np.ndarray,
    background_count: int,
    rigid_actor_id: np.ndarray,
    active_actor: np.ndarray,
    background_world_position: np.ndarray,
    rigid_world_position: np.ndarray,
    scene: str,
    seed: int,
    minimum_actor_gaussians: int,
    maximum_pairs_per_actor: int,
    cosine_epsilon: float,
) -> dict[str, Any]:
    """Evaluate model-membership actor compactness against nearest frozen Background."""

    values = np.asarray(feature)
    support = np.asarray(covered, dtype=bool)
    actor_ids = np.asarray(rigid_actor_id, dtype=np.int64)
    active = np.asarray(active_actor, dtype=bool)
    background_position = np.asarray(background_world_position, dtype=np.float64)
    rigid_position = np.asarray(rigid_world_position, dtype=np.float64)
    rigid_count = actor_ids.size
    if values.ndim != 2 or support.shape != (values.shape[0],):
        raise ValueError("feature and coverage shape mismatch")
    if values.shape[0] != int(background_count) + rigid_count:
        raise ValueError("Background/Rigid global layout mismatch")
    if background_position.shape != (int(background_count), 3):
        raise ValueError("Background position shape mismatch")
    if rigid_position.shape != (rigid_count, 3):
        raise ValueError("Rigid position shape mismatch")

    background_global = np.arange(int(background_count), dtype=np.int64)
    background_valid = support[: int(background_count)].copy()
    background_norm = np.linalg.norm(
        np.asarray(values[: int(background_count)], dtype=np.float64), axis=1
    )
    background_valid &= background_norm > float(cosine_epsilon)
    background_global = background_global[background_valid]
    if background_global.size == 0:
        return {"eligible_actor_count": 0, "actor_reports": [], "scene_margin": None}
    tree = cKDTree(background_position[background_valid])

    reports: list[dict[str, Any]] = []
    for actor_id in np.unique(actor_ids):
        if actor_id < 0 or actor_id >= active.size or not bool(active[actor_id]):
            continue
        local = np.flatnonzero(actor_ids == actor_id)
        global_ids = int(background_count) + local
        actor_norm = np.linalg.norm(np.asarray(values[global_ids], dtype=np.float64), axis=1)
        selected = support[global_ids] & (actor_norm > float(cosine_epsilon))
        local = local[selected]
        global_ids = global_ids[selected]
        if global_ids.size < int(minimum_actor_gaussians):
            continue
        pair_seed = derived_actor_seed(base_seed=int(seed), scene=scene, actor_id=int(actor_id))
        pairs = deterministic_actor_pairs(
            global_ids, seed=pair_seed, maximum_pairs=int(maximum_pairs_per_actor)
        )
        same_values, same_valid = row_cosine(
            values[pairs[:, 0]], values[pairs[:, 1]], epsilon=float(cosine_epsilon)
        )
        _, nearest = tree.query(rigid_position[local], k=1, workers=1)
        nearest_global = background_global[np.asarray(nearest, dtype=np.int64)]
        background_values, background_pair_valid = row_cosine(
            values[global_ids], values[nearest_global], epsilon=float(cosine_epsilon)
        )
        if not same_valid.any() or not background_pair_valid.any():
            continue
        same_mean = float(same_values[same_valid].mean())
        background_mean = float(background_values[background_pair_valid].mean())
        reports.append(
            {
                "actor_id": int(actor_id),
                "covered_gaussian_count": int(global_ids.size),
                "pair_count": int(same_valid.sum()),
                "nearest_background_count": int(background_pair_valid.sum()),
                "pair_seed": int(pair_seed),
                "same_actor_cosine": same_mean,
                "actor_background_cosine": background_mean,
                "margin": same_mean - background_mean,
            }
        )
    margins = np.asarray([row["margin"] for row in reports], dtype=np.float64)
    return {
        "eligible_actor_count": len(reports),
        "actor_reports": reports,
        "scene_margin": float(margins.mean()) if margins.size else None,
    }


def reproject_feature_arms(
    *,
    features_by_arm: Mapping[str, np.ndarray],
    common_covered: np.ndarray,
    gaussian_id: np.ndarray,
    pixel_id: np.ndarray,
    contribution_weight: np.ndarray,
    patch_grid: np.ndarray,
    image_height: int,
    image_width: int,
    minimum_intersection_contribution: float,
    minimum_pixel_mass: float,
    cosine_epsilon: float,
) -> dict[str, Any]:
    """Render B0/B1 to an exact common heldout pixel denominator and compare to DINO."""

    if set(features_by_arm) != {"B0", "B1"}:
        raise ValueError("reprojection requires exact B0/B1 arms")
    b0 = np.asarray(features_by_arm["B0"])
    b1 = np.asarray(features_by_arm["B1"])
    if b0.ndim != 2 or b1.shape != b0.shape:
        raise ValueError("B0/B1 feature shape mismatch")
    support = np.asarray(common_covered, dtype=bool)
    if support.shape != (b0.shape[0],):
        raise ValueError("common coverage shape mismatch")
    gids = np.asarray(gaussian_id, dtype=np.int64)
    pixels = np.asarray(pixel_id, dtype=np.int64)
    weights = np.asarray(contribution_weight, dtype=np.float64)
    if gids.ndim != 1 or pixels.shape != gids.shape or weights.shape != gids.shape:
        raise ValueError("reprojection intersections must align")
    pixel_count = int(image_height) * int(image_width)
    selected = (
        (weights >= float(minimum_intersection_contribution))
        & support[gids]
    )
    projection = sparse.coo_matrix(
        (weights[selected], (pixels[selected], gids[selected])),
        shape=(pixel_count, b0.shape[0]),
        dtype=np.float64,
    ).tocsr()
    mass = np.asarray(projection.sum(axis=1)).reshape(-1)
    supported_pixels = np.flatnonzero(mass >= float(minimum_pixel_mass))
    if supported_pixels.size == 0:
        return {
            "supported_pixel_count": 0,
            "valid_cosine_count": 0,
            "pixel_coverage": 0.0,
            "B0_mean_cosine": None,
            "B1_mean_cosine": None,
            "B1_minus_B0": None,
        }
    target = sample_patch_grid_bilinear(
        np.asarray(patch_grid),
        supported_pixels,
        image_height=int(image_height),
        image_width=int(image_width),
    )
    predictions: dict[str, np.ndarray] = {}
    for arm, feature in (("B0", b0), ("B1", b1)):
        numerator = np.asarray(projection[supported_pixels] @ feature, dtype=np.float64)
        predictions[arm] = numerator / (mass[supported_pixels, None] + float(cosine_epsilon))
    b0_cosine, b0_valid = row_cosine(
        predictions["B0"], target, epsilon=float(cosine_epsilon)
    )
    b1_cosine, b1_valid = row_cosine(
        predictions["B1"], target, epsilon=float(cosine_epsilon)
    )
    common_valid = b0_valid & b1_valid
    if not common_valid.any():
        b0_mean = None
        b1_mean = None
        difference = None
    else:
        b0_mean = float(b0_cosine[common_valid].mean())
        b1_mean = float(b1_cosine[common_valid].mean())
        difference = b1_mean - b0_mean
    return {
        "supported_pixel_count": int(supported_pixels.size),
        "valid_cosine_count": int(common_valid.sum()),
        "pixel_coverage": float(supported_pixels.size / pixel_count),
        "B0_mean_cosine": b0_mean,
        "B1_mean_cosine": b1_mean,
        "B1_minus_B0": difference,
    }


def evaluate_h_gate(scene_reports: list[dict[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    if len(scene_reports) != int(gate["scene_count"]):
        raise ValueError("H gate scene denominator mismatch")
    evaluable = [row for row in scene_reports if row["evaluable"]]
    margins = np.asarray(
        [row["actor_metrics"]["B1"]["scene_margin"] for row in evaluable],
        dtype=np.float64,
    )
    rigid_coverage = np.asarray(
        [row["coverage"]["rigid"] for row in scene_reports], dtype=np.float64
    )
    heldout_delta = np.asarray(
        [row["heldout_reprojection"]["scene_B1_minus_B0"] for row in scene_reports],
        dtype=np.float64,
    )
    checks = {
        "minimum_evaluable_scenes": len(evaluable) >= int(gate["minimum_evaluable_scenes"]),
        "minimum_positive_b1_margin_scenes": int(
            (margins > float(gate["minimum_scene_balanced_b1_margin_exclusive"])).sum()
        )
        >= int(gate["minimum_positive_b1_margin_scenes"]),
        "scene_balanced_b1_margin_positive": bool(
            margins.size
            and margins.mean()
            > float(gate["minimum_scene_balanced_b1_margin_exclusive"])
        ),
        "scene_balanced_rigid_coverage": bool(
            rigid_coverage.mean() >= float(gate["minimum_scene_balanced_rigid_coverage"])
        ),
        "scene_balanced_heldout_non_degradation": bool(
            np.isfinite(heldout_delta).all()
            and heldout_delta.mean() >= float(gate["minimum_scene_balanced_heldout_b1_minus_b0"])
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "evaluable_scene_count": len(evaluable),
        "positive_b1_margin_scene_count": int(
            (margins > float(gate["minimum_scene_balanced_b1_margin_exclusive"])).sum()
        ),
        "scene_balanced_b1_margin": float(margins.mean()) if margins.size else None,
        "scene_balanced_rigid_coverage": float(rigid_coverage.mean()),
        "scene_balanced_heldout_b1_minus_b0": float(heldout_delta.mean()),
    }
