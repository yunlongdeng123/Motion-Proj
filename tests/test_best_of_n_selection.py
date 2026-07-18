from motion_proj.eval.natural_rollout_ranking import select_condition_candidates


def _row(index, punc, generic, oracle):
    return {
        "candidate_id": f"x{index}",
        "condition_id": "condition-a",
        "candidate_index": index,
        "candidate_role": "base_fixed" if index == 0 else "natural_candidate",
        "eligible": True,
        "training_score": {"projection_energy": punc},
        "generic_smoothness_energy": generic,
        "cotracker": {"valid": True, "plausibility_score": oracle},
    }


def test_best_of_n_keeps_fixed_base_out_of_selection_pool():
    rows = [
        _row(0, 0.0, 0.0, 0.0),
        _row(1, 3.0, 1.0, 2.0),
        _row(2, 1.0, 3.0, 3.0),
        _row(3, 2.0, 2.0, 1.0),
    ]
    selected = select_condition_candidates(rows, selection_seed=11)
    assert selected["base_id"] == "x0"
    assert selected["punc_best_id"] == "x2"
    assert selected["generic_best_id"] == "x1"
    assert selected["cotracker_oracle_id"] == "x3"
    assert selected["random_id"] in {"x1", "x2", "x3"}
    assert select_condition_candidates(rows, selection_seed=11) == selected
