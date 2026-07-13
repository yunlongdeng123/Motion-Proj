import torch

from motion_proj.train.pilot import _to_batch, capacity_decision, select_pair_indices


def test_select_pair_indices_is_stable_and_disjoint():
    first = select_pair_indices(122, pair_count=8, train_pair_count=4, seed=20260713)
    second = select_pair_indices(122, pair_count=8, train_pair_count=4, seed=20260713)
    assert first == second
    assert len(set(first["all"])) == 8
    assert set(first["train"]).isdisjoint(first["held_out"])


def test_capacity_decision_requires_every_preregistered_gate():
    metrics = {
        "initial_target_error": 1.0,
        "final_target_error": 0.1,
        "outside_teacher_drift_ratio": 0.01,
        "frame0_teacher_drift": 0.0,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "target_roundtrip_max_error": 0.0,
        "correction_direction_cosine": 0.2,
    }
    accepted = capacity_decision(metrics, required_error_reduction=0.8, max_outside_teacher_drift_ratio=0.02)
    assert accepted["passed"]
    metrics["correction_direction_cosine"] = -0.01
    assert not capacity_decision(metrics, required_error_reduction=0.8, max_outside_teacher_drift_ratio=0.02)["passed"]


def test_noise_bank_restores_batch_dimension_for_svd():
    latent = torch.zeros(8, 4, 2, 3)
    item = {
        "base_latent": latent,
        "projected_latent": latent,
        "latent_residual": latent,
        "static_mask": torch.zeros(8, 1, 2, 3),
        "object_mask": torch.ones(8, 1, 2, 3),
        "context": {"image_embeds": torch.zeros(1, 4)},
        "metadata": {"source": "replay_v2", "parent_kind": "base", "adapter_loaded": False,
                     "uses_future_gt_ego": False, "uses_future_gt_track": False, "sample_id": "pair"},
    }
    item["object_mask"][0] = 0
    bank = {"sigma": torch.tensor([0.1]), "noise": latent, "z_sigma": latent}
    batch = _to_batch(item, bank, torch.device("cpu"))
    assert batch["z"].shape == (1, 8, 4, 2, 3)
    assert batch["noise"].shape == (1, 8, 4, 2, 3)
