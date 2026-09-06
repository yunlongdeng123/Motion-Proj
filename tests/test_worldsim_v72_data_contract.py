from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from motion_proj.worldsim_v72.data.actor_dataset import (
    load_actor_bundle_v2,
    save_actor_bundle_v2,
)
from motion_proj.worldsim_v72.data.schema import ActorBundleV2, QueryRayBatch, RayTargets
from motion_proj.worldsim_v72.render.ray_queries import (
    RayOutcome,
    classify_ray_outcomes,
    known_free_intervals,
    predict_point_surface_returns,
)
from motion_proj.worldsim_v72.render.scene_compositor import compose_nearest_returns


def _queries(count: int = 4) -> QueryRayBatch:
    return QueryRayBatch(
        ray_id=np.asarray([f"ray-{index}" for index in range(count)]),
        log_id="log-1",
        frame_id=np.asarray(["frame-1"] * count),
        time_ns=np.arange(count, dtype=np.int64),
        sensor_id=np.asarray(["lidar"] * count),
        origin_m=np.zeros((count, 3), dtype=np.float32),
        unit_direction=np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        range_min_m=np.full(count, 0.1, dtype=np.float32),
        range_max_m=np.full(count, 10.0, dtype=np.float32),
        query_pose=np.tile(np.eye(4, dtype=np.float64)[None, :, :], (count, 1, 1)),
        beam_parameters=np.zeros((count, 1), dtype=np.float32),
        valid_emission_mask=np.asarray([False, True, True, True][:count]),
    )


def test_query_contract_excludes_target_depth_and_supports_inference() -> None:
    assert "target" not in {field.name for field in fields(QueryRayBatch)}
    prediction = predict_point_surface_returns(
        np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        _queries(2),
        lateral_tolerance_m=0.2,
    )
    assert prediction.return_valid.tolist() == [False, True]
    assert np.isnan(prediction.range_m[0])
    assert prediction.range_m[1] == 2.0


def test_invalid_missing_no_return_and_occlusion_remain_distinct() -> None:
    queries = _queries()
    targets = RayTargets(
        return_valid=np.asarray([False, False, False, True]),
        return_index=np.asarray([-1, -1, -1, 0]),
        range_m=np.asarray([np.nan, np.nan, np.nan, 3.0], dtype=np.float32),
        intensity=np.full(4, np.nan, dtype=np.float32),
        intensity_valid=np.zeros(4, dtype=bool),
        censoring_or_missing_mask=np.asarray([False, True, False, False]),
        object_id_for_evaluation_only=np.asarray(["", "", "", "background"]),
    )
    outcomes = classify_ray_outcomes(queries, targets, expected_actor_id="actor-1")
    assert outcomes.tolist() == [
        RayOutcome.INVALID_EMISSION.value,
        RayOutcome.CENSORED_OR_MISSING.value,
        RayOutcome.NO_RETURN.value,
        RayOutcome.BACKGROUND_OCCLUSION.value,
    ]
    free = known_free_intervals(queries, targets, endpoint_tolerance_m=0.1)
    assert np.isnan(free[:3]).all()
    assert np.allclose(free[3], [0.1, 2.9])


def test_scene_compositor_uses_world_depth_order() -> None:
    composed = compose_nearest_returns(
        np.asarray([2.0, np.nan, 8.0], dtype=np.float32),
        {
            "actor-a": np.asarray([4.0, 5.0, np.nan], dtype=np.float32),
            "actor-b": np.asarray([3.0, 6.0, 7.0], dtype=np.float32),
        },
    )
    assert np.allclose(composed.range_m, [2.0, 5.0, 7.0])
    assert composed.source.tolist() == ["background", "actor-a", "actor-b"]


def test_actor_bundle_v2_round_trip_does_not_touch_v1(tmp_path: Path) -> None:
    identity = np.eye(4, dtype=np.float64)[None, :, :]
    bundle = ActorBundleV2(
        dataset="nuScenes",
        log_id="log-1",
        scene_id="scene-1",
        actor_id="actor-1",
        category="car",
        size_lwh_m=np.asarray([4.0, 2.0, 1.5], dtype=np.float32),
        build_frame_ids=np.asarray(["frame-1"]),
        build_time_ns=np.asarray([1], dtype=np.int64),
        world_from_sensor=identity,
        world_from_actor=identity,
        build_points_actor_m=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        point_frame_id=np.asarray(["frame-1"]),
        point_sensor_id=np.asarray(["lidar"]),
        build_ray_origin_actor_m=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        build_ray_direction_actor=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        build_range_m=np.asarray([2.0], dtype=np.float32),
        build_intensity=np.asarray([0.5], dtype=np.float32),
        build_intensity_valid=np.asarray([True]),
        sensor_model_ref="nuscenes-lidar-v1",
        beam_or_ring_id=np.asarray(["unknown"]),
        per_point_time_offset_ns=np.asarray([0], dtype=np.int64),
        evidence_fou=np.asarray([[0.1, 0.8, 0.1]], dtype=np.float32),
        conflict_count=np.asarray([0], dtype=np.int32),
        opportunity_count=np.asarray([1], dtype=np.int32),
        build_only_surface_m=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        provenance={"source": "unit-test", "target_access": False},
    )
    output = tmp_path / "actor-v2.npz"
    save_actor_bundle_v2(bundle, output)
    restored = load_actor_bundle_v2(output)

    assert restored.actor_id == "actor-1"
    assert restored.point_count == 1
    assert np.array_equal(restored.build_time_ns, bundle.build_time_ns)
    assert restored.provenance["target_access"] is False
    assert not (tmp_path / "actor-v1.npz").exists()
