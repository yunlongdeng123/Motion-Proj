from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from motion_proj.worldsim_v3.gaussian_ancestry import (
    GaussianAncestryLedger,
    InitSource,
    validate_a2_instrumentation_contract,
)


def make_ledger() -> GaussianAncestryLedger:
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    return GaussianAncestryLedger.initialize(
        means=means,
        actor_ids=torch.tensor([4, 4, 7]),
        init_sources=int(InitSource.LIDAR),
    )


def test_a2_instrumentation_contract_is_frozen() -> None:
    payload = yaml.safe_load(
        Path("configs/worldsim_v3/a2_instrumentation_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_a2_instrumentation_contract(payload)
    payload["module_off_equivalence"][
        "require_native_tensor_bitwise_equality"
    ] = False
    with pytest.raises(ValueError, match="bitwise"):
        validate_a2_instrumentation_contract(payload)


def test_split_and_clone_keep_parent_and_lineage_order() -> None:
    ledger = make_ledger()
    split_ids = ledger.append_children(
        parent_indices=torch.tensor([0, 2]),
        repeats=2,
        source=InitSource.SPLIT,
        birth_step=600,
        child_means=torch.zeros((4, 3)),
    )
    clone_ids = ledger.append_children(
        parent_indices=torch.tensor([1]),
        repeats=1,
        source=InitSource.CLONE,
        birth_step=600,
        child_means=torch.tensor([[2.0, 0.0, 0.0]]),
    )

    assert split_ids.tolist() == [3, 4, 5, 6]
    assert clone_ids.tolist() == [7]
    assert ledger.parent_id.tolist() == [-1, -1, -1, 0, 2, 0, 2, 1]
    assert ledger.lineage_root_id.tolist() == [0, 1, 2, 0, 2, 0, 2, 1]
    assert ledger.actor_id.tolist() == [4, 4, 7, 4, 7, 4, 7, 4]
    assert ledger.init_source[-5:].tolist() == [4, 4, 4, 4, 5]
    assert torch.isnan(ledger.nearest_lidar_distance[3:7]).all()
    assert ledger.nearest_lidar_distance[7].item() == 0.0
    ledger.validate()


def test_screen_and_attributed_metrics_are_running_means() -> None:
    ledger = make_ledger()
    ledger.record_screen_statistics(
        indices=torch.tensor([0, 2]), gradients=torch.tensor([2.0, 4.0])
    )
    ledger.record_screen_statistics(
        indices=torch.tensor([0, 2]), gradients=torch.tensor([4.0, 8.0])
    )
    ledger.record_diagnostics(
        indices=torch.tensor([0, 2]),
        boundary_contribution=torch.tensor([0.2, 0.6]),
        photometric_residual=torch.tensor([0.4, float("nan")]),
    )
    ledger.record_diagnostics(
        indices=torch.tensor([0]),
        boundary_contribution=torch.tensor([0.4]),
        photometric_residual=torch.tensor([0.2]),
        depth_residual=torch.tensor([1.5]),
    )

    assert ledger.visibility_count.tolist() == [2, 0, 2]
    torch.testing.assert_close(ledger.screen_grad[[0, 2]], torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(
        ledger.boundary_contribution[[0, 2]], torch.tensor([0.3, 0.6])
    )
    assert ledger.photometric_residual_count.tolist() == [2, 0, 0]
    assert ledger.photometric_residual[0].item() == pytest.approx(0.3)
    assert ledger.depth_residual[0].item() == pytest.approx(1.5)


def test_prune_preserves_allocated_parent_references_and_roundtrips() -> None:
    ledger = make_ledger()
    ledger.append_children(
        parent_indices=torch.tensor([0, 1]),
        repeats=1,
        source=InitSource.CLONE,
        birth_step=700,
    )
    ledger.prune(torch.tensor([False, True, True, True, True]))
    ledger.validate()
    assert ledger.gaussian_id.tolist() == [1, 2, 3, 4]
    assert ledger.parent_id.tolist() == [-1, -1, 0, 1]

    restored = GaussianAncestryLedger.from_state_dict(
        ledger.state_dict(), device="cpu"
    )
    restored.validate()
    for name in ledger.state_dict()["fields"]:
        torch.testing.assert_close(
            getattr(restored, name), getattr(ledger, name), equal_nan=True
        )
    assert restored.next_gaussian_id == 5


def test_external_actor_replacement_is_auditable_clone() -> None:
    ledger = make_ledger()
    source_rows = ledger.select(torch.tensor([True, True, False]))
    ledger.prune(torch.tensor([False, False, True]))
    new_ids = ledger.append_external_clones(
        source_rows=source_rows,
        actor_id=9,
        birth_step=1000,
    )

    assert new_ids.tolist() == [3, 4]
    assert ledger.actor_id.tolist() == [7, 9, 9]
    assert ledger.parent_id.tolist() == [-1, 0, 1]
    assert ledger.lineage_root_id.tolist() == [2, 0, 1]
    assert ledger.init_source.tolist() == [1, 5, 5]
    ledger.validate()


def test_exact_nearest_lidar_materialization_is_per_actor() -> None:
    ledger = make_ledger()
    ledger.append_children(
        parent_indices=torch.tensor([0, 2]),
        repeats=1,
        source=InitSource.SPLIT,
        birth_step=600,
    )
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.5, 0.0],
        ]
    )
    result = ledger.materialize_nearest_lidar_distance(
        means=means,
        actor_ids=[4, 7],
        chunk_size=1,
        maximum_reference_points=2,
    )

    assert result["completed"] == {"4": 3, "7": 2}
    torch.testing.assert_close(
        ledger.nearest_lidar_distance,
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.5]),
    )


def test_rejects_duplicate_metric_indices_and_actor_drift() -> None:
    ledger = make_ledger()
    with pytest.raises(ValueError, match="unique"):
        ledger.record_screen_statistics(
            indices=torch.tensor([0, 0]), gradients=torch.tensor([1.0, 2.0])
        )
    with pytest.raises(ValueError, match="actor IDs"):
        ledger.validate(expected_actor_ids=torch.tensor([4, 7, 7]))
