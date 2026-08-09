from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from motion_proj.worldsim_v3.chunk_package import (
    SENTINEL_TAG,
    actor_memberships,
    build_skeleton,
    compare_checkpoint_states,
    materialize_chunk_package,
    reassemble_chunk_package,
    select_chunk_arm,
    static_chunk_id,
    static_memberships,
)


def protocol_fixture() -> dict:
    return {
        "selected_asset": {
            "checkpoint": {"path": "/source.pth", "sha256": "a" * 64, "bytes": 1},
            "actor_registry": {
                "path": "/registry.json",
                "sha256": "b" * 64,
                "bytes": 1,
            },
        },
        "row_tensor_schema": {
            "common_gaussian_row_tensors": [
                {"path": "_means", "dtype": "float32", "shape_tail": [3]},
                {"path": "_scales", "dtype": "float16", "shape_tail": [1]},
                {
                    "path": "worldsim_a2_ancestry.fields.gaussian_id",
                    "dtype": "int64",
                    "shape_tail": [],
                },
            ],
            "models": {
                "Background": {"row_count": 5, "additional_row_tensors": []},
                "RigidNodes": {
                    "row_count": 4,
                    "additional_row_tensors": [
                        {"path": "points_ids", "dtype": "int64", "shape_tail": [1]}
                    ],
                },
            },
        },
        "static_chunk_contract": {
            "origin_xy_m": [0.0, 0.0],
            "cell_size_m": 50.0,
        },
        "actor_chunk_contract": {
            "actor_index_domain_inclusive": [0, 2],
        },
        "package_contract": {"package_format": "worldsim_v3_chunk_package_v1"},
    }


def checkpoint_fixture() -> OrderedDict:
    background = OrderedDict(
        {
            "_means": torch.tensor(
                [
                    [-50.0, -50.0, 0.0],
                    [-0.1, -0.1, 1.0],
                    [0.0, 0.0, 2.0],
                    [49.9, 49.9, 3.0],
                    [50.0, 0.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            "_scales": torch.arange(5, dtype=torch.float16).reshape(5, 1),
            "worldsim_a2_ancestry": {
                "fields": {"gaussian_id": torch.arange(100, 105, dtype=torch.int64)},
                "schema_version": 1,
            },
            "shared": torch.tensor([3.0, 4.0]),
        }
    )
    rigid = OrderedDict(
        {
            "_means": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "_scales": torch.arange(4, dtype=torch.float16).reshape(4, 1),
            "worldsim_a2_ancestry": {
                "fields": {"gaussian_id": torch.arange(200, 204, dtype=torch.int64)},
                "schema_version": 1,
            },
            "points_ids": torch.tensor([[0], [1], [0], [2]], dtype=torch.int64),
            "instances_quats": torch.ones(2, 3, 4),
        }
    )
    return OrderedDict(
        {
            "models": OrderedDict(
                {
                    "Background": background,
                    "RigidNodes": rigid,
                    "Sky": {"weights": torch.ones(2, 3)},
                }
            ),
            "lpips": {"weight": torch.ones(1)},
            "step": torch.tensor(30_000),
            "label": "synthetic",
        }
    )


def test_grid_membership_uses_frozen_half_open_negative_cells() -> None:
    source = checkpoint_fixture()
    memberships = static_memberships(
        source["models"]["Background"]["_means"],
        protocol_fixture()["static_chunk_contract"],
    )
    assert [(ix, iy, indices.tolist()) for ix, iy, indices in memberships] == [
        (-1, -1, [0, 1]),
        (0, 0, [2, 3]),
        (1, 0, [4]),
    ]
    assert static_chunk_id(-1, 0) == "static-x-n0001-y-p0000"


def test_actor_membership_keeps_interleaved_indices_and_empty_asset() -> None:
    source = checkpoint_fixture()
    rows = actor_memberships(
        source["models"]["RigidNodes"]["points_ids"],
        protocol_fixture()["actor_chunk_contract"],
    )
    assert [(actor, indices.tolist()) for actor, indices in rows] == [
        (0, [0, 2]),
        (1, [1]),
        (2, [3]),
    ]
    changed = deepcopy(protocol_fixture())
    changed["actor_chunk_contract"]["actor_index_domain_inclusive"] = [0, 3]
    assert actor_memberships(
        source["models"]["RigidNodes"]["points_ids"],
        changed["actor_chunk_contract"],
    )[-1][1].numel() == 0


def test_skeleton_replaces_only_row_tensors_and_preserves_shared_state() -> None:
    source = checkpoint_fixture()
    skeleton, count = build_skeleton(source, protocol_fixture())
    assert count == 7
    assert skeleton["models"]["Background"]["_means"]["tag"] == SENTINEL_TAG
    assert skeleton["models"]["RigidNodes"]["points_ids"]["tag"] == SENTINEL_TAG
    assert torch.equal(
        skeleton["models"]["RigidNodes"]["instances_quats"],
        source["models"]["RigidNodes"]["instances_quats"],
    )
    assert skeleton["label"] == "synthetic"


def test_package_roundtrip_is_bitwise_exact(tmp_path: Path) -> None:
    source = checkpoint_fixture()
    protocol = protocol_fixture()
    root = tmp_path / "package"
    manifest = materialize_chunk_package(
        source,
        package_root=root,
        protocol=protocol,
        protocol_sha256="c" * 64,
        project_commit="d" * 40,
    )
    candidate, audit = reassemble_chunk_package(
        package_root=root,
        manifest=manifest,
        protocol=protocol,
    )
    comparison = compare_checkpoint_states(source, candidate)
    assert manifest["counts"] == {
        "static_assets": 3,
        "actor_assets": 3,
        "data_assets": 6,
        "payload_files": 7,
    }
    assert audit["manifest_records_exact"]
    assert audit["row_fields_exact"]
    assert audit["static_cell_membership_exact"]
    assert audit["actor_membership_exact"]
    assert audit["indices_unique_disjoint_exhaustive"]
    assert comparison["all_exact"]


def test_reassembly_rejects_payload_hash_drift(tmp_path: Path) -> None:
    source = checkpoint_fixture()
    protocol = protocol_fixture()
    root = tmp_path / "package"
    manifest = materialize_chunk_package(
        source,
        package_root=root,
        protocol=protocol,
        protocol_sha256="c" * 64,
        project_commit="d" * 40,
    )
    path = root / manifest["static_assets"][0]["path"]
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(RuntimeError, match="payload drift"):
        reassemble_chunk_package(
            package_root=root,
            manifest=manifest,
            protocol=protocol,
        )


def test_chunk_selection_is_fail_closed() -> None:
    eligible = {
        "exact_static_and_actor_asset_inventory": True,
        "exact_row_fields_and_shared_skeleton_without_duplication": True,
        "exact_package_manifest_hashes_bytes_counts_bounds_and_indices": True,
        "bitwise_exact_full_checkpoint_reassembly_and_reload": True,
        "p2_mixed_precision_runtime_adapter_exact": True,
        "source_baseline_replay_matches_p2_exact": True,
        "all_57_rgb_hashes_and_all_31_quality_endpoints_exact": True,
        "source_inputs_unchanged": True,
        "resources_within_frozen_ceilings": True,
    }
    assert select_chunk_arm(eligible) == {
        "selected_arm": "p3-chunk-package",
        "method_state": "selected_exact_chunk_package",
        "fallback_exact_alias": False,
    }
    for key in eligible:
        failed = dict(eligible)
        failed[key] = False
        assert select_chunk_arm(failed) == {
            "selected_arm": "p3-source",
            "method_state": "rejected_chunk_integrity_quality_or_resource_gate",
            "fallback_exact_alias": True,
        }
