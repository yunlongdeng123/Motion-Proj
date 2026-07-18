from motion_proj.eval.natural_rollout_ranking import eligibility_sensitivity, pair_key


def test_sensitivity_removes_only_named_check_and_keeps_diversity_requirements():
    rows = []
    for index, absolute in ((0, True), (1, False), (2, False)):
        rows.append(
            {
                "candidate_id": f"c-{index}",
                "condition_id": "c",
                "candidate_index": index,
                "candidate_role": "base_fixed" if index == 0 else "natural_candidate",
                "eligibility": {"checks": {"first_frame_absolute": absolute, "motion": True}},
                "pixel_metrics": {"dynamic_degree_mean_abs_rgb": 1.0 + index * 0.1},
                "training_score": {"projection_energy": 1.0 + index * 0.2},
            }
        )
    thresholds = {
        "minimum_eligible_selection_candidates": 2,
        "minimum_pairwise_rgb_rms": 0.01,
        "minimum_candidate_metric_span": 0.02,
    }
    pairwise = {"c": {pair_key("c-1", "c-2"): 0.05}}
    strict = eligibility_sensitivity(rows, pairwise, thresholds, ignored_checks=[])
    assert strict["eligible_candidate_count"] == 0
    relaxed = eligibility_sensitivity(
        rows, pairwise, thresholds, ignored_checks=["first_frame_absolute"],
    )
    assert relaxed["eligible_candidate_count"] == 2
    assert relaxed["condition_with_at_least_two_eligible_upper_bound"] == 1
    assert relaxed["diverse_condition_count"] == 1
