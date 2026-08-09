from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from motion_proj.worldsim_v3.contribution_prune import (
    array_sha256,
    build_candidate_masks,
    build_candidate_registry,
    compare_metric_group,
    index_sha256,
    prune_checkpoint_state,
    select_largest_eligible_arm,
    stable_remove_indices,
)


def score_fixture(count: int) -> dict[str, np.ndarray]:
    return {
        "train_alpha_weight_sum": np.arange(count, dtype=np.float64),
        "train_visible_view_count": np.arange(count, dtype=np.int64),
        "learned_opacity": np.linspace(0.1, 0.9, count),
        "gaussian_ids": np.arange(100, 100 + count, dtype=np.int64),
    }


def checkpoint_fixture() -> OrderedDict:
    def model(count: int, rigid: bool) -> OrderedDict:
        state = OrderedDict(
            (
                name,
                torch.arange(count * width).reshape(count, width)
                if width > 1
                else torch.arange(count).reshape(count, 1),
            )
            for name, width in (
                ("_means", 3),
                ("_scales", 3),
                ("_quats", 4),
                ("_features_dc", 3),
                ("_features_rest", 6),
                ("_opacities", 1),
            )
        )
        state["worldsim_a2_ancestry"] = {
            "schema_version": 1,
            "reference_lidar_positions": torch.zeros(2, 3),
            "fields": {
                "gaussian_id": torch.arange(count),
                "visibility_count": torch.arange(count),
            },
        }
        if rigid:
            state["points_ids"] = torch.arange(count).remainder(2).reshape(-1, 1)
            state["instances_fv"] = torch.ones(2, 2, dtype=torch.bool)
        return state

    return OrderedDict(
        {
            "models": OrderedDict(
                {
                    "Background": model(5, False),
                    "RigidNodes": model(6, True),
                    "Sky": {"base": torch.ones(2)},
                }
            ),
            "step": torch.tensor(30_000),
        }
    )


def test_stable_remove_indices_uses_score_then_frozen_tie_breaks() -> None:
    removed = stable_remove_indices(
        train_alpha_weight_sum=np.array([1.0, 0.0, 0.0, 2.0]),
        train_visible_view_count=np.array([1, 2, 1, 0]),
        learned_opacity=np.array([0.1, 0.1, 0.9, 0.1]),
        gaussian_ids=np.array([10, 11, 12, 13]),
        asset_indices=np.arange(4),
        prune_fraction=0.5,
    )
    assert removed.tolist() == [1, 2]


def test_array_sha256_binds_dtype_shape_and_bytes() -> None:
    value = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert array_sha256(value) == array_sha256(value.copy())
    assert array_sha256(value) != array_sha256(value.astype(np.float32))
    assert array_sha256(value) != array_sha256(value.reshape(3, 2))


def test_build_candidate_masks_prunes_each_asset_independently() -> None:
    background = score_fixture(10)
    rigid = score_fixture(10)
    background_keep, rigid_keep, manifest = build_candidate_masks(
        background_scores=background,
        rigid_scores=rigid,
        rigid_point_ids=np.array([0] * 6 + [1] * 4),
        prune_fraction=0.5,
    )
    assert int(background_keep.sum()) == 5
    assert int(rigid_keep.sum()) == 5
    assert [row["removed_count"] for row in manifest] == [5, 3, 2]


def test_checkpoint_pruning_keeps_all_rows_aligned_and_invariants_exact() -> None:
    checkpoint = checkpoint_fixture()
    result = prune_checkpoint_state(
        checkpoint,
        torch.tensor([True, False, True, False, True]),
        torch.tensor([False, True, True, False, True, True]),
    )
    assert result["models"]["Background"]["_means"].shape[0] == 3
    assert result["models"]["Background"]["worldsim_a2_ancestry"]["fields"][
        "gaussian_id"
    ].shape[0] == 3
    assert result["models"]["RigidNodes"]["points_ids"].shape[0] == 4
    assert result["models"]["RigidNodes"]["worldsim_a2_ancestry"]["fields"][
        "visibility_count"
    ].shape[0] == 4
    assert result["models"]["Sky"] is checkpoint["models"]["Sky"]
    assert result["step"] is checkpoint["step"]


def test_candidate_registry_recomputes_actor_slices() -> None:
    source = {
        "actor_count": 2,
        "available_actor_count": 2,
        "empty_checkpoint_actor_count": 0,
        "checkpoint_sha256": "old",
        "actors": [
            {
                "rigid_model_index": 0,
                "availability": "available",
                "checkpoint_tensor_slice": {},
            },
            {
                "rigid_model_index": 1,
                "availability": "available",
                "checkpoint_tensor_slice": {},
            },
        ],
        "selected_smoke_actor": {"rigid_model_index": 1},
        "source": {"checkpoint": "old"},
    }
    result = build_candidate_registry(source, [0, 1, 0], "new")
    assert result["checkpoint_sha256"] == "new"
    assert result["actors"][0]["checkpoint_tensor_slice"]["gaussian_count"] == 2
    assert result["actors"][1]["checkpoint_tensor_slice"]["flat_indices_sha256"] == index_sha256([1])
    assert result["actor_registry_sha256"]


def test_metric_group_applies_directional_regression_budget() -> None:
    contracts = [
        {"name": "psnr", "direction": "higher", "maximum_regression": 0.1},
        {"name": "lpips", "direction": "lower", "maximum_regression": 0.002},
    ]
    rows = compare_metric_group(
        {"psnr": 20.0, "lpips": 0.1},
        {"psnr": 19.91, "lpips": 0.102},
        contracts,
    )
    assert all(row["passed"] for row in rows)
    failed = compare_metric_group(
        {"psnr": 20.0, "lpips": 0.1},
        {"psnr": 19.89, "lpips": 0.103},
        contracts,
    )
    assert not any(row["passed"] for row in failed)


def test_selection_uses_largest_fully_eligible_arm() -> None:
    common = {
        "candidate_checkpoint_reload_exact": True,
        "expected_counts_exact": True,
        "all_quality_safeguards_pass": True,
        "checkpoint_bytes_strictly_less_than_source": True,
        "source_inputs_unchanged": True,
        "resources_within_frozen_ceilings": True,
    }
    result = select_largest_eligible_arm(
        [
            {"id": "p1-source", "prune_fraction": 0.0},
            {"id": "p1-b05", "prune_fraction": 0.05, **common},
            {"id": "p1-b10", "prune_fraction": 0.10, **common},
            {
                "id": "p1-b20",
                "prune_fraction": 0.20,
                **{**common, "all_quality_safeguards_pass": False},
            },
        ]
    )
    assert result["selected_arm"] == "p1-b10"


def test_selection_falls_back_to_source_exact_alias() -> None:
    result = select_largest_eligible_arm(
        [
            {"id": "p1-source", "prune_fraction": 0.0},
            {"id": "p1-b05", "prune_fraction": 0.05},
        ]
    )
    assert result == {
        "selected_arm": "p1-source",
        "selected_prune_fraction": 0.0,
        "method_state": "rejected_quality_or_integrity_gate",
        "fallback_exact_alias": True,
    }
