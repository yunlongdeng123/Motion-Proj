from __future__ import annotations

import hashlib

import numpy as np
import pytest

from motion_proj.worldsim_v33.roadpatch import (
    EXCLUDE_ACTOR_SEMANTIC,
    EXCLUDE_GENERATED,
    FEATURE_INDEX,
    HoleAnchor,
    atomic_save_patch_delta,
    atomic_save_patch_index,
    build_hole_anchor,
    build_patch_index,
    conservative_delete_mask,
    load_patch_delta,
    load_patch_index,
    materialize_patch_delta,
    search_donors,
)


def _synthetic_background() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(7)
    groups = []
    for center_x in (5.25, 7.25, 9.25, 11.25, 13.25, 15.25, 17.25):
        x = center_x + rng.uniform(-0.2, 0.2, 24)
        z = 10.25 + rng.uniform(-0.2, 0.2, 24)
        y = 2.0 + 0.01 * x - 0.005 * z + rng.normal(0.0, 0.002, 24)
        groups.append(np.column_stack([x, y, z]))
    means = np.concatenate(groups).astype(np.float32)
    count = means.shape[0]
    state = {
        "_means": means,
        "_scales": np.log(np.full((count, 3), 0.04, dtype=np.float32)),
        "_quats": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (count, 1)),
        "_features_dc": np.tile(np.array([[0.0, 0.02, -0.02]], np.float32), (count, 1)),
        "_features_rest": np.zeros((count, 15, 3), dtype=np.float32),
        "_opacities": np.zeros((count, 1), dtype=np.float32),
    }
    return state, np.arange(1000, 1000 + count, dtype=np.int64)


def _build_index(*, actor_row: int | None = None, generated_row: int | None = None):
    state, _ = _synthetic_background()
    count = state["_means"].shape[0]
    actor = np.zeros(count, dtype=np.float32)
    native = np.ones(count, dtype=bool)
    if actor_row is not None:
        actor[actor_row] = 1.0
    if generated_row is not None:
        native[generated_row] = False
    index = build_patch_index(
        means=state["_means"],
        raw_scales=state["_scales"],
        raw_opacities=state["_opacities"],
        features_dc=state["_features_dc"],
        actor_semantic_score=actor,
        train_view_observation_count=np.full(count, 20, dtype=np.int32),
        visibility_mass=np.full(count, 10.0, dtype=np.float32),
        multi_camera_count=np.full(count, 3, dtype=np.uint8),
        native_donor_mask=native,
        patch_sizes_m=(1.0, 2.0, 4.0),
        thresholds={
            "minimum_rows": 8,
            "maximum_actor_semantic": 0.8,
            "minimum_visibility_mass": 1.0,
            "minimum_train_view_observations": 2,
            "minimum_multi_camera_count": 2,
            "minimum_multi_camera_rows": 2,
            "maximum_plane_residual_m": 0.05,
            "minimum_abs_plane_normal_vertical": 0.9,
            "maximum_vertical_range_m": 0.2,
            "maximum_scale_m": 0.2,
        },
    )
    return state, index


def _anchor() -> HoleAnchor:
    return HoleAnchor(
        center_xyz=np.array([0.25, 2.0, 10.25], dtype=np.float32),
        bounds_bev=np.array([-0.2, 9.8, 0.7, 10.7], dtype=np.float32),
        patch_size_m=1.0,
        tangent_yaw=0.0,
        tangent_confidence=0.9,
        context_rgb_mean=np.array([0.5, 0.505, 0.495], dtype=np.float32),
        context_rgb_std=np.array([0.02, 0.02, 0.02], dtype=np.float32),
        valid_point_count=100,
        cross_view_observed_pixels=50,
    )


def test_patch_index_excludes_actor_and_generated_rows() -> None:
    _, index = _build_index(actor_row=0, generated_row=24)
    assert 0 not in index.flat_indices
    assert 24 not in index.flat_indices
    assert not np.any(index.exclusion_flags & int(EXCLUDE_ACTOR_SEMANTIC))
    assert not np.any(index.exclusion_flags & int(EXCLUDE_GENERATED))
    assert tuple(sorted(np.unique(index.patch_sizes_m).tolist())) == (1.0, 2.0, 4.0)


def test_patch_index_uses_densest_vertical_layer() -> None:
    state, _ = _synthetic_background()
    road_count = state["_means"].shape[0]
    facade_count = 12
    facade_means = state["_means"][:facade_count].copy()
    facade_means[:, 1] += 3.0
    state["_means"] = np.concatenate([state["_means"], facade_means], axis=0)
    for name in ("_scales", "_quats", "_features_dc", "_features_rest", "_opacities"):
        state[name] = np.concatenate([state[name], state[name][:facade_count]], axis=0)
    count = state["_means"].shape[0]
    index = build_patch_index(
        means=state["_means"],
        raw_scales=state["_scales"],
        raw_opacities=state["_opacities"],
        features_dc=state["_features_dc"],
        actor_semantic_score=np.zeros(count, dtype=np.float32),
        train_view_observation_count=np.full(count, 20, dtype=np.int32),
        visibility_mass=np.full(count, 10.0, dtype=np.float32),
        multi_camera_count=np.full(count, 3, dtype=np.uint8),
        native_donor_mask=np.ones(count, dtype=bool),
        patch_sizes_m=(1.0, 2.0, 4.0),
        thresholds={
            "minimum_rows": 8,
            "maximum_actor_semantic": 0.8,
            "minimum_visibility_mass": 1.0,
            "minimum_train_view_observations": 2,
            "minimum_multi_camera_count": 2,
            "minimum_multi_camera_rows": 2,
            "maximum_plane_residual_m": 0.05,
            "minimum_abs_plane_normal_vertical": 0.9,
            "maximum_vertical_range_m": 0.2,
            "maximum_scale_m": 0.2,
        },
    )
    assert index.patch_ids.size > 0
    assert not np.any(index.flat_indices >= road_count)
    assert float(np.max(index.features[:, FEATURE_INDEX["vertical_range"]])) <= 0.2 + 1e-6


def test_patch_index_serialization_is_byte_exact(tmp_path) -> None:
    state, index = _build_index()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    atomic_save_patch_index(first, index)
    atomic_save_patch_index(second, index)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    restored = load_patch_index(first)
    restored.validate(background_count=state["_means"].shape[0])
    np.testing.assert_array_equal(restored.flat_indices, index.flat_indices)


def test_conservative_mask_and_bottom_first_hit_anchor() -> None:
    instance = np.zeros((20, 30), dtype=bool)
    semantic = np.zeros_like(instance)
    instance[5:18, 10:20] = True
    semantic[6:18, 11:19] = True
    mask = conservative_delete_mask(instance, semantic)
    depth = np.full(mask.shape, 10.0, dtype=np.float32)
    rgb = np.full((*mask.shape, 3), 128, dtype=np.uint8)
    intrinsics = np.array([[100.0, 0.0, 15.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    anchor = build_hole_anchor(
        delete_mask=mask,
        first_hit_depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        camera_to_world=np.eye(4),
        cross_view_observed_pixels=40,
        patch_sizes_m=(1.0, 2.0, 4.0),
        bottom_quantile=0.6,
        robust_quantiles=(0.05, 0.95),
        minimum_anchor_pixels=8,
        minimum_cross_view_observed_pixels=32,
        ring_pixels=2,
    )
    assert anchor.patch_size_m in {1.0, 2.0, 4.0}
    assert anchor.valid_point_count >= 8
    assert anchor.center_xyz[2] == pytest.approx(10.0)


def test_anchor_abstains_without_cross_view_support() -> None:
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="ABSTAIN"):
        build_hole_anchor(
            delete_mask=mask,
            first_hit_depth=np.ones((4, 4), dtype=np.float32),
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            intrinsics=np.eye(3),
            camera_to_world=np.eye(4),
            cross_view_observed_pixels=0,
            patch_sizes_m=(1.0, 2.0, 4.0),
            bottom_quantile=0.6,
            robust_quantiles=(0.05, 0.95),
            minimum_anchor_pixels=1,
            minimum_cross_view_observed_pixels=1,
            ring_pixels=1,
        )


def test_search_top5_and_materialize_delta_without_source_mutation(tmp_path) -> None:
    state, index = _build_index()
    original = {name: value.copy() for name, value in state.items()}
    candidates = search_donors(
        index=index,
        anchor=_anchor(),
        top_k=5,
        weights={"geometry": 1.0, "appearance": 1.0, "semantic": 1.0, "visibility": 1.0},
        minimum_spatial_separation_m=2.0,
        minimum_tangent_confidence=0.1,
        maximum_abs_yaw_radians=0.6,
        maximum_abs_vertical_offset_m=0.5,
    )
    assert len(candidates) == 5
    assert [item.distance for item in candidates] == sorted(
        item.distance for item in candidates
    )
    source_ids = np.arange(1000, 1000 + state["_means"].shape[0], dtype=np.int64)
    delta, manifest = materialize_patch_delta(
        index=index,
        candidate=candidates[0],
        anchor=_anchor(),
        background_state=state,
        source_gaussian_ids=source_ids,
        target_role="high_support",
        opacity_feather_width_m=0.2,
        maximum_rgb_affine=0.05,
        minimum_scale_m=0.01,
        maximum_scale_m=0.2,
        duplicate_radius_m=1e-5,
    )
    assert delta["means"].shape[0] > 0
    assert set(delta["source_flat_indices"]) <= set(index.rows(candidates[0].patch_index))
    np.testing.assert_allclose(np.linalg.norm(delta["quats"], axis=1), 1.0, atol=1e-6)
    assert manifest["source_checkpoint_mutated"] is False
    assert manifest["provenance"] == "GENERATED_BY_PATCH_REUSE"
    for name in state:
        np.testing.assert_array_equal(state[name], original[name])
    first = tmp_path / "delta-a.npz"
    second = tmp_path / "delta-b.npz"
    atomic_save_patch_delta(first, delta)
    atomic_save_patch_delta(second, delta)
    assert first.read_bytes() == second.read_bytes()
    restored = load_patch_delta(first)
    np.testing.assert_array_equal(restored["source_gaussian_ids"], delta["source_gaussian_ids"])
