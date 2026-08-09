from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from motion_proj.worldsim_v3.actor_quota import (
    ActorQuotaController,
    ActorQuotaPolicy,
    D2_RANKING,
    validate_a2_d1_contract,
)


def make_policy(**overrides: object) -> ActorQuotaPolicy:
    values = {
        "densify_grad_threshold": 0.25,
        "minimum_initial_multiplier": 0.5,
        "minimum_absolute_floor": 1,
        "maximum_initial_multiplier": 2.0,
        "maximum_absolute_cap": 20,
    }
    values.update(overrides)
    return ActorQuotaPolicy(**values)


def test_a2_d1_contract_is_frozen() -> None:
    payload = yaml.safe_load(
        Path("configs/worldsim_v3/a2_d1_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_a2_d1_contract(payload)
    payload["actor_densification"]["background"]["quota"] = "enabled"
    with pytest.raises(ValueError, match="background"):
        validate_a2_d1_contract(payload)


def test_scene_0230_initial_quota_totals_are_exact() -> None:
    payload = yaml.safe_load(
        Path("configs/worldsim_v3/a2_d1_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    reference = payload["frozen_scene_0230_reference"]
    counts = torch.tensor(reference["initial_actor_counts"])
    actor_ids = torch.repeat_interleave(
        torch.arange(counts.numel()), counts
    )
    rigid = payload["actor_densification"]["rigid_nodes"]
    policy = ActorQuotaPolicy.from_mapping(
        {
            "densify_grad_threshold": rigid["densify_grad_threshold"],
            **rigid["quota"],
        }
    )
    controller = ActorQuotaController.initialize(
        actor_ids=actor_ids, policy=policy
    )

    assert controller.initial_counts.tolist() == counts.tolist()
    assert controller.initial_counts.sum().item() == 75002
    assert controller.minimum_counts.sum().item() == 37504
    assert controller.maximum_counts.sum().item() == 180013


def test_gradient_ranked_prefix_enforces_child_capacity() -> None:
    actor_ids = torch.tensor([0, 0, 0, 1, 1])
    controller = ActorQuotaController(
        policy=make_policy(),
        initial_counts=torch.tensor([3, 2]),
        minimum_counts=torch.tensor([1, 1]),
        maximum_counts=torch.tensor([6, 4]),
    )
    splits, clones, decision = controller.select_densification(
        actor_ids=actor_ids,
        average_gradients=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
        visibility_counts=torch.ones(5),
        split_geometry=torch.tensor([True, False, False, True, False]),
        clone_geometry=torch.tensor([False, True, True, True, True]),
        split_children=2,
    )

    assert splits.tolist() == [True, False, False, False, False]
    assert clones.tolist() == [False, True, False, False, False]
    assert decision["accepted_children"] == 3
    assert decision["rejected_by_maximum_parents"] == 3


def test_stable_tie_break_uses_gaussian_index() -> None:
    controller = ActorQuotaController(
        policy=make_policy(),
        initial_counts=torch.tensor([3]),
        minimum_counts=torch.tensor([1]),
        maximum_counts=torch.tensor([5]),
    )
    _, clones, _ = controller.select_densification(
        actor_ids=torch.tensor([0, 0, 0]),
        average_gradients=torch.tensor([0.5, 0.5, 0.5]),
        visibility_counts=torch.ones(3),
        split_geometry=torch.zeros(3, dtype=torch.bool),
        clone_geometry=torch.ones(3, dtype=torch.bool),
        split_children=2,
    )

    assert clones.tolist() == [True, True, False]


def test_d2_ranking_changes_order_without_changing_eligibility_or_quota() -> None:
    controller = ActorQuotaController(
        policy=make_policy(ranking=D2_RANKING),
        initial_counts=torch.tensor([3]),
        minimum_counts=torch.tensor([1]),
        maximum_counts=torch.tensor([5]),
    )
    _, clones, decision = controller.select_densification(
        actor_ids=torch.tensor([0, 0, 0]),
        average_gradients=torch.tensor([0.9, 0.8, 0.7]),
        visibility_counts=torch.ones(3),
        split_geometry=torch.zeros(3, dtype=torch.bool),
        clone_geometry=torch.ones(3, dtype=torch.bool),
        split_children=2,
        boundary_mean=torch.tensor([0.0, 1.0, 1.0]),
        boundary_count=torch.ones(3, dtype=torch.long),
        photometric_residual_mean=torch.tensor([10.0, 0.1, 0.2]),
        photometric_residual_count=torch.ones(3, dtype=torch.long),
    )

    assert clones.tolist() == [False, True, True]
    assert decision["accepted_children"] == 2


def test_d2_ranking_requires_all_diagnostic_vectors() -> None:
    controller = ActorQuotaController(
        policy=make_policy(ranking=D2_RANKING),
        initial_counts=torch.tensor([1]),
        minimum_counts=torch.tensor([1]),
        maximum_counts=torch.tensor([2]),
    )
    with pytest.raises(ValueError, match="requires all diagnostic vectors"):
        controller.select_densification(
            actor_ids=torch.tensor([0]),
            average_gradients=torch.tensor([0.9]),
            visibility_counts=torch.ones(1),
            split_geometry=torch.zeros(1, dtype=torch.bool),
            clone_geometry=torch.ones(1, dtype=torch.bool),
            split_children=2,
        )


def test_visible_below_threshold_candidates_only_recover_minimum() -> None:
    controller = ActorQuotaController(
        policy=make_policy(densify_grad_threshold=1.0),
        initial_counts=torch.tensor([4]),
        minimum_counts=torch.tensor([4]),
        maximum_counts=torch.tensor([8]),
    )
    splits, clones, decision = controller.select_densification(
        actor_ids=torch.tensor([0, 0]),
        average_gradients=torch.tensor([0.0, 0.2]),
        visibility_counts=torch.ones(2),
        split_geometry=torch.tensor([False, True]),
        clone_geometry=torch.tensor([True, False]),
        split_children=2,
    )

    assert splits.tolist() == [False, True]
    assert clones.tolist() == [False, False]
    assert decision["accepted_children"] == 2
    assert decision["admitted_below_threshold_parents"] == 1


def test_state_roundtrip_preserves_frozen_quotas_and_counters() -> None:
    actor_ids = torch.tensor([0, 0, 1, 1])
    controller = ActorQuotaController.initialize(
        actor_ids=actor_ids, policy=make_policy()
    )
    controller.select_densification(
        actor_ids=actor_ids,
        average_gradients=torch.ones(4),
        visibility_counts=torch.ones(4),
        split_geometry=torch.zeros(4, dtype=torch.bool),
        clone_geometry=torch.ones(4, dtype=torch.bool),
        split_children=2,
    )
    restored = ActorQuotaController.from_state_dict(
        controller.state_dict(), device="cpu"
    )

    assert restored.policy == controller.policy
    assert restored.state_dict()["counters"] == controller.state_dict()[
        "counters"
    ]
    assert restored.summary(actor_ids=actor_ids)["maximum_total"] == 8
    torch.testing.assert_close(
        restored.maximum_counts, controller.maximum_counts
    )


def test_invalid_policy_and_actor_drift_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_policy(densify_grad_threshold=-1).validate()
    controller = ActorQuotaController.initialize(
        actor_ids=torch.tensor([0, 0, 1]), policy=make_policy()
    )
    with pytest.raises(ValueError, match="frozen actor set"):
        controller.summary(actor_ids=torch.tensor([0, 2]))
