from motion_proj.diagnostics.real_motion_target_audit import aggregate_a0_metrics, decide_a0_gate


THRESHOLDS = {
    "minimum_labeled_actor_pairs_each": 8,
    "minimum_moving_stationary_auc": 0.75,
    "minimum_projection_in_box_fraction": 0.98,
    "minimum_valid_actor_tracks": 32,
    "minimum_finite_target_fraction": 0.95,
    "minimum_background_direction_points": 256,
    "minimum_background_angular_agreement": 0.70,
    "minimum_velocity_direction_pairs": 16,
    "minimum_velocity_direction_positive_fraction": 0.60,
    "maximum_abs_residual_ego_spearman": 0.50,
    "maximum_calibration_drift": 1.0e-6,
    "minimum_lidar_points_per_frame": 64,
}


def _passing_metrics():
    return {
        "moving_pair_count": 20,
        "stationary_pair_count": 20,
        "moving_vs_stationary_residual_auc": 0.82,
        "center_projection_in_box_fraction": 0.99,
        "valid_paired_actor_track_count": 40,
        "finite_target_fraction": 0.98,
        "background_direction_point_count": 1000,
        "background_ego_vs_raft_angular_agreement": 0.75,
        "velocity_direction_pair_count": 30,
        "velocity_direction_positive_fraction": 0.8,
        "residual_vs_ego_speed_spearman": 0.2,
        "missing_schema_count": 0,
        "visibility_violation_count": 0,
        "maximum_cam2ego_drift": 0.0,
        "maximum_intrinsics_drift": 0.0,
        "minimum_lidar_points_per_frame": 100,
    }


def test_a0_gate_passes_only_when_all_legality_checks_pass():
    decision = decide_a0_gate(_passing_metrics(), THRESHOLDS)
    assert decision["machine_pass"]
    assert decision["next_gate"] == "RP-A1-SCAN-04A"


def test_a0_gate_fails_closed_for_ego_entanglement_or_missing_support():
    metrics = _passing_metrics()
    metrics["residual_vs_ego_speed_spearman"] = 0.7
    metrics["stationary_pair_count"] = 3
    decision = decide_a0_gate(metrics, THRESHOLDS)
    assert not decision["machine_pass"]
    assert {"ego_disentanglement", "moving_stationary_support"} <= set(decision["failed_checks"])
    assert decision["next_gate"] == "RP-B0-05"


def test_projection_denominator_excludes_only_offscreen_centers():
    base = {
        "finite": True,
        "motion_label": "moving",
        "residual_speed_px_per_s": 2.0,
        "velocity_direction_cosine": 1.0,
        "ego_translation_speed_mps": 1.0,
        "sample_id": "s",
        "instance_token": "i",
    }
    rows = [
        {
            **base,
            "center_projection_eligible_t": True,
            "center_projection_in_box_t": True,
            "center_projection_eligible_tp1": False,
            "center_projection_in_box_tp1": False,
        }
    ]
    metrics = aggregate_a0_metrics(rows, [], [])
    assert metrics["center_projection_eligible_count"] == 1
    assert metrics["offscreen_visible_center_count"] == 1
    assert metrics["center_projection_in_box_fraction"] == 1.0
