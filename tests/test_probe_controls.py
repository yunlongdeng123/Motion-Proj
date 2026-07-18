import torch

from motion_proj.data.motion_feature_records import (
    fit_ridge,
    permute_instance_targets,
    permute_learned_features,
    permute_targets,
    vector_epe,
)


def test_feature_and_target_shuffle_controls_destroy_association():
    generator = torch.Generator().manual_seed(11)
    features = torch.randn((300, 8), generator=generator)
    targets = torch.stack([features[:, 0] + 0.5 * features[:, 1], features[:, 2] - features[:, 3]], dim=1)
    train_x, dev_x = features[:220], features[220:]
    train_y, dev_y = targets[:220], targets[220:]
    model = fit_ridge(train_x, train_y, regularization=1.0e-3)
    normal = vector_epe(model.predict(dev_x), dev_y)
    shuffled_features = permute_learned_features(dev_x, learned_width=6, seed=19)
    shuffled_feature_error = vector_epe(model.predict(shuffled_features), dev_y)
    shuffled_head = fit_ridge(train_x, permute_targets(train_y, seed=23), regularization=1.0e-3)
    shuffled_target_error = vector_epe(shuffled_head.predict(dev_x), dev_y)
    assert shuffled_feature_error > normal * 20
    assert shuffled_target_error > normal * 20


def test_instance_target_shuffle_preserves_each_donor_sequence_order():
    targets = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [8.0, 80.0], [9.0, 90.0], [4.0, 40.0]],
    )
    instance_ids = ["a", "a", "a", "b", "b", "c"]
    shuffled = permute_instance_targets(targets, instance_ids, seed=5)
    assert not torch.equal(shuffled, targets)
    # 每个 receiver 的 rows 都来自同一个 donor，且 donor 内部顺序未被逐行打散。
    for receiver in ("a", "b", "c"):
        rows = shuffled[[value == receiver for value in instance_ids]]
        assert bool((rows[1:, 0] >= rows[:-1, 0]).all()) if len(rows) > 1 else True


def test_scan_ranking_requires_two_sigma_stability_and_fail_closed_values():
    from motion_proj.diagnostics.motion_feature_probe import primary_scan_checks, rank_primary_configs

    thresholds = {
        "minimum_train_actor_queries": 8,
        "minimum_dev_actor_queries": 4,
        "minimum_train_ego_queries": 8,
        "minimum_dev_ego_queries": 4,
        "scan_minimum_ego_improvement": 0.10,
        "scan_minimum_actor_improvement": 0.075,
        "scan_minimum_res_vs_abs_improvement": 0.05,
        "minimum_control_degradation": 0.10,
        "minimum_ego_control_degradation": 0.05,
        "maximum_stationary_to_moving_ratio": 0.75,
        "minimum_stable_sigmas_per_layer": 2,
    }
    base = {
        "train_actor_count": 8,
        "dev_actor_count": 4,
        "train_ego_count": 8,
        "dev_ego_count": 4,
        "ego_vs_best_baseline_improvement": 0.20,
        "actor_res_vs_zero_improvement": 0.20,
        "actor_res_vs_abs_improvement": 0.20,
        "actor_time_shuffle_degradation": 0.20,
        "ego_time_shuffle_degradation": 0.20,
        "actor_target_shuffle_degradation": 0.20,
        "stationary_prediction_to_moving_target_ratio": 0.20,
        "same_actor_head_capacity": True,
    }
    rows = [
        {**base, "config_id": "layer_a-sigma0.05", "layer": "layer_a", "sigma": 0.05},
        {**base, "config_id": "layer_a-sigma0.2", "layer": "layer_a", "sigma": 0.2},
        {**base, "config_id": "layer_b-sigma0.05", "layer": "layer_b", "sigma": 0.05},
    ]
    ranked = rank_primary_configs(rows, thresholds, top_k=2)
    assert ranked["stable_layers"] == ["layer_a"]
    assert ranked["primary_selected_configs"] == ["layer_a-sigma0.05", "layer_a-sigma0.2"]
    assert not primary_scan_checks({**base, "ego_vs_best_baseline_improvement": None}, thresholds)["ego_signal"]
