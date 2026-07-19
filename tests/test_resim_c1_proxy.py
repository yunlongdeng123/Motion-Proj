import math

import numpy as np
import torch

from motion_proj.diagnostics.resim_c1_proxy import (
    FEATURE_NAMES,
    action_class,
    affine_proxy_features,
    calibration_checks,
    flow_with_confidence_chunked,
    predict_proxy,
    select_scene_sets,
    trajectory_at_horizon,
)


def test_trajectory_at_horizon_interpolates_two_hz_waypoints():
    trajectory = [[float(index), float(index) * -0.5, 0.0] for index in range(1, 9)]
    value = trajectory_at_horizon(trajectory, horizon_seconds=2.4, waypoint_hz=2.0)
    assert np.allclose(value, [4.8, -2.4, 0.0])


def test_action_class_separates_stationary_from_command():
    assert action_class("Moving_Forward", 0.1, stationary_max=0.5, moving_min=3.0) == "stationary"
    assert action_class("Moving_Forward", 2.0, stationary_max=0.5, moving_min=3.0) is None
    assert action_class("Turning_Left", 5.0, stationary_max=0.5, moving_min=3.0) == "left"


def _record(class_name, scene_index, clip_index):
    command = {"stationary": "Moving_Forward", "forward": "Moving_Forward", "left": "Turning_Left", "right": "Turning_Right"}[class_name]
    displacement = 0.1 if class_name == "stationary" else 10.0 + scene_index
    return {
        "clip_index": clip_index, "clip_token": str(clip_index), "lidar_pc_token": str(clip_index),
        "sample_data_token": f"sd-{clip_index}", "sample_token": f"sample-{clip_index}",
        "scene_token": f"scene-token-{class_name}-{scene_index}", "scene_name": f"scene-{class_name}-{scene_index}",
        "scene_description": "day", "command": command, "action_class": class_name,
        "trajectory": [[displacement, 0.0, 0.0]] * 8,
        "trajectory_at_horizon": [displacement, 0.0, 0.0],
        "target_displacement_m": displacement, "target_lateral_m": 0.0,
        "annotation_count": 20 + scene_index, "source_frames": [f"frame-{clip_index}-{i}.jpg" for i in range(49)],
    }


def test_scene_selection_freezes_counts_and_disjoint_scenes():
    records = []
    for class_index, class_name in enumerate(("stationary", "forward", "left", "right")):
        records.extend(_record(class_name, scene, class_index * 100 + scene) for scene in range(20))
    selection = {
        "selection_seed": 7, "screen_seed_base": 100,
        "screen_counts": {"stationary": 2, "forward": 4, "left": 2, "right": 2},
        "calibration_per_class": 12, "calibration_fit_per_class": 6,
    }
    result = select_scene_sets(records, selection)
    assert len(result["screen"]) == 10
    assert len(result["calibration"]) == 48
    screen = {row["scene_token"] for row in result["screen"]}
    calibration = {row["scene_token"] for row in result["calibration"]}
    assert screen.isdisjoint(calibration)
    assert len(screen) == 10 and len(calibration) == 48


def test_affine_proxy_features_have_frozen_schema():
    rows = []
    for index in range(4):
        rows.append({
            "valid": True,
            "coefficients": [[4.0 + index, 2.0], [1.0, 0.5], [0.25, 0.75]],
            "residual_median_px": 0.2 + index * 0.01,
            "confidence_mean": 0.8,
        })
    result = affine_proxy_features({"pairs": rows}, height=256, width=448)
    assert result["valid"] is True
    assert result["feature_names"] == list(FEATURE_NAMES)
    assert len(result["features"]) == len(FEATURE_NAMES)
    assert all(math.isfinite(value) for value in result["features"])


def test_calibration_checks_require_proxy_to_beat_displacement_baselines():
    metrics = {
        "moving_balanced_accuracy": 0.8, "turn_sign_accuracy": 0.75,
        "displacement_spearman": 0.6, "displacement_mae_m": 1.0,
        "constant_displacement_mae_m": 2.0, "command_only_displacement_mae_m": 1.5,
    }
    thresholds = {
        "minimum_balanced_accuracy": 0.7, "minimum_turn_sign_accuracy": 0.75,
        "minimum_displacement_spearman": 0.5,
    }
    assert all(calibration_checks(metrics, thresholds).values())


def test_predict_proxy_rejects_feature_schema_drift():
    model = {
        "feature_names": list(FEATURE_NAMES), "classes": ["stationary", "forward", "left", "right"],
        "feature_mean": [0.0] * len(FEATURE_NAMES), "feature_scale": [1.0] * len(FEATURE_NAMES),
        "class_weights": np.zeros((len(FEATURE_NAMES) + 1, 4)).tolist(),
        "displacement_weights": np.zeros(len(FEATURE_NAMES) + 1).tolist(),
    }
    result = predict_proxy(model, [0.0] * len(FEATURE_NAMES))
    assert result["predicted_class"] == "stationary"
    assert result["predicted_displacement_m"] == 0.0


def test_chunked_flow_preserves_pair_order_and_shape():
    class FakeFlow:
        def flow(self, source, target):
            value = (target[:, 0, 0, 0] - source[:, 0, 0, 0]).view(-1, 1, 1, 1)
            return value.expand(-1, source.shape[-2], source.shape[-1], 2).clone()

        @staticmethod
        def _fb_consistency(forward, backward):
            return torch.ones(forward.shape[:-1])

    frames = torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1).expand(-1, 3, 8, 8)
    flow, confidence = flow_with_confidence_chunked(FakeFlow(), frames, pair_batch_size=4)
    assert flow.shape == (9, 8, 8, 2)
    assert confidence.shape == (9, 8, 8)
    assert torch.equal(flow[:, 0, 0, 0], torch.ones(9))
