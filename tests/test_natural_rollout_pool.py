from motion_proj.eval.natural_rollout_ranking import candidate_eligibility, condition_diversity, pair_key


def _thresholds():
    return {
        "minimum_projection_points": 4,
        "minimum_primary_tracks": 1,
        "minimum_track_coverage": 0.4,
        "minimum_median_track_length": 3,
        "minimum_scorer_confidence": 0.4,
        "minimum_first_frame_psnr_db": 18.0,
        "maximum_first_frame_psnr_drop_db": 1.5,
        "minimum_sharpness_ratio_to_base": 0.8,
        "maximum_flicker_ratio_to_base": 1.5,
        "maximum_flicker_absolute_increase": 0.02,
        "minimum_motion_ratio_to_base": 0.8,
        "maximum_motion_ratio_to_base": 2.0,
        "maximum_survival_drop_to_base": 0.2,
        "maximum_saturation_fraction": 0.99,
        "maximum_punc_frame0_correction_px": 1.0e-6,
        "maximum_punc_visibility_changed_count": 0,
        "minimum_eligible_selection_candidates": 2,
        "minimum_pairwise_rgb_rms": 0.01,
        "minimum_candidate_metric_span": 0.02,
    }


def _row(candidate_id, index, *, dynamic=1.0, punc=1.0):
    return {
        "candidate_id": candidate_id,
        "condition_id": "c0",
        "candidate_index": index,
        "candidate_role": "base_fixed" if index == 0 else "natural_candidate",
        "uses_future_gt": False,
        "training_score": {
            "uses_future_gt": False,
            "valid": True,
            "projection_energy": punc,
            "projection_points": 20,
            "primary_track_count": 3,
            "track_coverage": 0.8,
            "median_track_length_frames": 7,
            "scorer_confidence": 0.8,
            "survival_rate": 0.9,
            "punc_invariants": {"frame0_correction_max_px": 0.0, "visibility_changed_count": 0},
        },
        "generic_smoothness_energy": 1.0,
        "pixel_metrics": {
            "finite_fraction": 1.0,
            "first_frame_psnr_db": 30.0,
            "spatial_laplacian_energy": 1.0,
            "global_luma_flicker_p95": 0.02,
            "dynamic_degree_mean_abs_rgb": dynamic,
        },
        "saturation_fraction": 0.0,
    }


def test_candidate_pool_requires_quality_motion_and_nonduplicate_diversity():
    thresholds = _thresholds()
    base = _row("base", 0)
    rows = [base, _row("a", 1, dynamic=1.0, punc=1.0), _row("b", 2, dynamic=1.1, punc=1.2)]
    for row in rows:
        row.update(candidate_eligibility(row, base, thresholds))
    diverse = condition_diversity(rows, {pair_key("a", "b"): 0.05}, thresholds)
    assert diverse["diverse"] is True
    duplicate = condition_diversity(rows, {pair_key("a", "b"): 0.001}, thresholds)
    assert duplicate["diverse"] is False


def test_candidate_eligibility_rejects_low_motion_shortcut():
    thresholds = _thresholds()
    base = _row("base", 0)
    frozen = _row("frozen", 1, dynamic=0.1)
    result = candidate_eligibility(frozen, base, thresholds)
    assert result["eligible"] is False
    assert result["checks"]["motion_floor_and_ceiling"] is False
