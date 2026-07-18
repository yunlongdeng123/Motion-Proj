from motion_proj.eval.natural_rollout_ranking import (
    aggregate_b0_gate,
    pair_key,
    pairwise_cotracker_preference,
)


def _thresholds():
    return {
        "minimum_eligible_selection_candidates": 2,
        "minimum_pairwise_rgb_rms": 0.01,
        "minimum_candidate_metric_span": 0.01,
        "minimum_pair_dynamic_ratio": 0.9,
        "maximum_pair_survival_drop": 0.05,
        "maximum_pair_coverage_drop": 0.05,
        "minimum_pair_sharpness_ratio": 0.9,
        "maximum_pair_flicker_ratio": 1.25,
        "maximum_pair_flicker_absolute_increase": 0.02,
        "maximum_pair_first_frame_psnr_drop_db": 1.0,
        "cotracker_relative_tie_margin": 0.02,
        "minimum_diverse_conditions": 2,
        "minimum_valid_comparison_conditions": 2,
        "minimum_punc_random_win_rate": 0.6,
        "minimum_punc_base_win_rate": 0.55,
        "minimum_systematic_dynamic_ratio": 0.95,
        "maximum_systematic_survival_drop": 0.05,
        "minimum_rank_disagreement_conditions": 1,
        "minimum_positive_improvement_conditions": 2,
        "maximum_single_condition_positive_share": 0.75,
        "maximum_winner_seed_fraction": 0.5,
    }


def _row(condition, index, punc, generic, score, *, dynamic=1.0):
    return {
        "candidate_id": f"{condition}-{index}",
        "condition_id": condition,
        "candidate_index": index,
        "candidate_role": "base_fixed" if index == 0 else "natural_candidate",
        "eligible": True,
        "training_score": {"projection_energy": punc},
        "generic_smoothness_energy": generic,
        "pixel_metrics": {
            "dynamic_degree_mean_abs_rgb": dynamic,
            "spatial_laplacian_energy": 1.0,
            "global_luma_flicker_p95": 0.02,
            "first_frame_psnr_db": 30.0,
        },
        "cotracker": {
            "valid": True,
            "plausibility_score": score,
            "survival_rate": 0.95,
            "track_coverage": 0.95,
        },
    }


def test_pairwise_preference_rejects_smoother_but_frozen_candidate():
    thresholds = _thresholds()
    candidate = _row("c", 1, 1.0, 1.0, 0.5, dynamic=0.5)
    reference = _row("c", 0, 2.0, 2.0, 1.0, dynamic=1.0)
    result = pairwise_cotracker_preference(candidate, reference, thresholds)
    assert result["outcome"] == "loss"
    assert result["selected_low_motion"] is True
    assert result["selected_catastrophic"] is True


def test_gate_rejects_candidate_index_monopoly_even_with_metric_wins():
    thresholds = _thresholds()
    rows = []
    rms = {}
    for condition in ("c0", "c1"):
        group = [
            _row(condition, 0, 5.0, 5.0, 2.0),
            _row(condition, 1, 1.0, 3.0, 1.0),
            _row(condition, 2, 3.0, 1.0, 1.5),
        ]
        rows.extend(group)
        rms[condition] = {pair_key(group[1]["candidate_id"], group[2]["candidate_id"]): 0.05}
    result = aggregate_b0_gate(rows, rms, thresholds, selection_seed=3)
    assert result["checks"]["no_seed_monopoly"] is False
    assert result["machine_pass"] is False
