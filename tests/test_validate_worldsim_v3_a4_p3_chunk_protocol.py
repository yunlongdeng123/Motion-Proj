from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from scripts.validate_worldsim_v3_a4_p3_chunk_protocol import (
    PROTOCOL,
    build_actor_inventory,
    build_static_inventory,
    directory_digest,
    discover_row_tensor_schema,
    validate_inputs,
    validate_schema,
    validate_source_layout,
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return OmegaConf.to_container(OmegaConf.load(PROTOCOL), resolve=True)


def test_frozen_protocol_schema_is_valid(protocol: dict) -> None:
    validate_schema(protocol)


def test_protocol_authorizes_only_exact_p3_packaging(protocol: dict) -> None:
    authorization = protocol["authorization"]
    assert authorization["p3_chunk_materialization_authorized"]
    assert authorization["p3_reassembled_candidate_render_authorized"]
    assert not authorization["training_authorized"]
    assert not authorization["optimizer_authorized"]
    assert not authorization["selective_chunk_render_authorized"]
    assert not authorization["view_dependent_chunk_culling_authorized"]
    assert not authorization["p4_lod_authorized"]


def test_protocol_rejects_grid_search_merge_or_sparse_drop(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["static_chunk_contract"]["cell_size_m"] = 25.0
    with pytest.raises(RuntimeError, match="cell size"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["static_chunk_contract"]["merge_policy"] = "merge_sparse"
    with pytest.raises(RuntimeError, match="merge policy"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["static_chunk_contract"]["sparse_chunks_preserved"] = False
    with pytest.raises(RuntimeError, match="sparse chunks"):
        validate_schema(changed)


def test_protocol_rejects_contiguous_actor_or_empty_actor_drift(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["actor_chunk_contract"]["contiguous_slice_assumption_allowed"] = True
    with pytest.raises(RuntimeError, match="contiguous assumption"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["actor_chunk_contract"]["empty_actor_indices"] = []
    with pytest.raises(RuntimeError, match="empty actor"):
        validate_schema(changed)


def test_protocol_rejects_row_schema_or_persistent_reassembly_drift(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["row_tensor_schema"]["common_gaussian_row_tensors"][1]["dtype"] = "float32"
    with pytest.raises(RuntimeError, match="common row tensor schema"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["package_contract"]["persistent_reassembled_checkpoint_forbidden"] = False
    with pytest.raises(RuntimeError, match="persistent reassembled checkpoint"):
        validate_schema(changed)


def test_protocol_rejects_quality_runtime_or_fallback_relaxation(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["quality_contract"]["candidate_source_replay_tolerance"]["psnr_absolute"] = 0.1
    with pytest.raises(RuntimeError, match="candidate tolerance"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["runtime_contract"]["performance_values_are_report_only_not_quality_selection"] = False
    with pytest.raises(RuntimeError, match="runtime report only"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["selection_contract"]["candidate_fail"]["selected_arm"] = "p3-chunk-package"
    with pytest.raises(RuntimeError, match="source fallback"):
        validate_schema(changed)


def test_protocol_rejects_recovery_or_resource_drift(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["recovery_contract"]["stage_order"].reverse()
    with pytest.raises(RuntimeError, match="stage order"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["resource_ceilings"]["run_bytes"] = 2_000_000_000
    with pytest.raises(RuntimeError, match="resource ceilings"):
        validate_schema(changed)


def test_frozen_inputs_and_p2_selection_are_exact(protocol: dict) -> None:
    audits = validate_inputs(protocol)
    assert len(audits) == 10
    assert audits["baseline_quality.actor_masks"]["file_count"] == 33


def test_directory_digest_binds_names_bytes_and_content(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"bb")
    first = directory_digest(tmp_path, "*.png")
    assert first["file_count"] == 2
    assert first["total_bytes"] == 3
    (tmp_path / "b.png").write_bytes(b"bc")
    assert directory_digest(tmp_path, "*.png")["sha256"] != first["sha256"]


def test_inventory_helpers_use_explicit_sorted_indices(protocol: dict) -> None:
    static_contract = deepcopy(protocol["static_chunk_contract"])
    static_contract["expected_source_inventory"]["boundary_band_m"] = 0.25
    means = torch.tensor(
        [
            [-0.1, -0.1, 0.0],
            [0.0, 0.0, 1.0],
            [49.9, 49.9, 2.0],
            [50.0, 0.0, 3.0],
            [-50.0, -50.0, 4.0],
        ],
        dtype=torch.float32,
    )
    static = build_static_inventory(means, static_contract)
    assert [row["id"] for row in static["rows"]] == [
        "static-x-n0001-y-n0001",
        "static-x-p0000-y-p0000",
        "static-x-p0001-y-p0000",
    ]
    assert [row["count"] for row in static["rows"]] == [2, 2, 1]

    actor_contract = deepcopy(protocol["actor_chunk_contract"])
    actor_contract["actor_index_domain_inclusive"] = [0, 2]
    actor = build_actor_inventory(torch.tensor([[0], [1], [0], [2]]), actor_contract)
    assert [row["count"] for row in actor["rows"]] == [2, 1, 1]
    assert not actor["rows"][0]["contiguous"]
    assert actor["rows"][1]["contiguous"]


def test_row_tensor_discovery_preserves_nested_provenance() -> None:
    model = {
        "_means": torch.zeros(3, 3),
        "shared": torch.zeros(2, 3),
        "nested": {"actor_id": torch.arange(3), "scalar": torch.tensor(1)},
    }
    assert discover_row_tensor_schema(model, 3) == [
        ("_means", "float32", [3]),
        ("nested.actor_id", "int64", []),
    ]


def test_frozen_source_layout_matches_50m_and_actor_facts(protocol: dict) -> None:
    audit = validate_source_layout(protocol)
    assert audit["static_inventory"]["occupied_chunk_count"] == 133
    assert audit["static_inventory"]["background_count"] == 1_205_164
    assert audit["actor_inventory"]["available_count"] == 23
    assert audit["actor_inventory"]["unavailable_count"] == 1
    assert audit["actor_inventory"]["actors"][14]["count"] == 0
