from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from motion_proj.worldsim_v32.integration import (
    build_chunk_protocol,
    discover_model_row_schema,
    extend_semantic_sidecar,
    validate_extended_semantic_sidecar,
)


def test_extend_semantic_sidecar_inserts_background_before_rigid(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    np.savez_compressed(
        source,
        labels=np.asarray([0, 1, 2, 3, 4], dtype=np.int8),
        posterior=np.asarray([0, 1, 2, 3, 4], dtype=np.float32),
        background_count=np.asarray(3, dtype=np.int64),
        rigid_point_ids=np.asarray([8, 9], dtype=np.int64),
    )
    audit = extend_semantic_sidecar(
        source,
        output,
        old_background_count=3,
        generated_background_count=2,
        rigid_count=2,
    )
    assert audit["new_total"] == 7
    with np.load(output, allow_pickle=False) as value:
        assert value["labels"].tolist() == [0, 1, 2, 0, 0, 3, 4]
        assert value["posterior"].tolist() == [0, 1, 2, 0, 0, 3, 4]
        assert int(value["background_count"]) == 5
        assert value["rigid_point_ids"].tolist() == [8, 9]
    assert validate_extended_semantic_sidecar(
        source,
        output,
        old_background_count=3,
        generated_background_count=2,
        rigid_count=2,
    )["all_exact"]


def test_discover_model_row_schema_ignores_non_row_tensor() -> None:
    model = OrderedDict(
        _means=torch.zeros(4, 3),
        _scales=torch.zeros(4, 3, dtype=torch.float16),
        global_value=torch.zeros(2),
        nested=OrderedDict(ids=torch.arange(4, dtype=torch.int64)),
    )
    rows = discover_model_row_schema(model)
    assert list(rows) == ["_means", "_scales", "nested.ids"]
    assert rows["_scales"]["dtype"] == "float16"


def test_build_chunk_protocol_splits_common_and_rigid_specific() -> None:
    common = {
        "_means": torch.zeros(3, 3),
        "_scales": torch.zeros(3, 3, dtype=torch.float16),
    }
    checkpoint = {
        "models": {
            "Background": OrderedDict(common),
            "RigidNodes": OrderedDict(
                _means=torch.zeros(4, 3),
                _scales=torch.zeros(4, 3, dtype=torch.float16),
                points_ids=torch.tensor([[0], [0], [1], [1]], dtype=torch.int64),
            ),
        }
    }
    protocol = build_chunk_protocol(
        checkpoint,
        checkpoint_record={"path": "x", "sha256": "a"},
        registry_record={"path": "r", "sha256": "b"},
    )
    common_paths = [
        row["path"] for row in protocol["row_tensor_schema"]["common_gaussian_row_tensors"]
    ]
    rigid_paths = [
        row["path"]
        for row in protocol["row_tensor_schema"]["models"]["RigidNodes"]["additional_row_tensors"]
    ]
    assert common_paths == ["_means", "_scales"]
    assert rigid_paths == ["points_ids"]
    assert protocol["actor_chunk_contract"]["actor_index_domain_inclusive"] == [0, 1]
