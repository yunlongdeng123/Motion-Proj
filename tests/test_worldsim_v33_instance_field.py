from pathlib import Path

import numpy as np
import pytest

from motion_proj.worldsim_v32.semantic_schema import (
    AMBIGUOUS,
    CORE_POSITIVE,
    NEGATIVE,
    SEMANTIC_POSITIVE,
)
from motion_proj.worldsim_v33.instance_field import (
    ActorSemanticSource,
    NO_INSTANCE,
    PROVENANCE_AMBIGUOUS_REASSIGNED,
    PROVENANCE_RIGID_CORE,
    atomic_save_instance_field,
    build_instance_field,
    load_instance_field,
    validate_instance_field,
)


def source(role: str, instance_id: int, rigid_index: int, labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    total, background_count = labels.size, 4
    return ActorSemanticSource(
        role=role,
        instance_id=instance_id,
        instance_token=f"token-{instance_id}",
        rigid_model_index=rigid_index,
        arrays={
            "labels": labels,
            "semantic_score": np.asarray(scores, dtype=np.float32),
            "num_positive_views": np.ones(total, dtype=np.int32) * 3,
            "num_negative_views": np.ones(total, dtype=np.int32),
            "visible_mass": np.ones(total, dtype=np.float32),
            "boundary_score": np.asarray([0.2, 0.2, 0.2, 0.2, 0.0, 0.0]),
            "background_count": np.asarray(background_count),
            "rigid_point_ids": np.asarray([5, 21], dtype=np.int64),
        },
    )


def sources():
    return [
        source(
            "high_support",
            13,
            5,
            [SEMANTIC_POSITIVE, AMBIGUOUS, NEGATIVE, NEGATIVE, CORE_POSITIVE, NEGATIVE],
            [0.8, 0.6, 0.1, 0.0, 1.0, 0.0],
        ),
        source(
            "boundary_support",
            41,
            21,
            [NEGATIVE, NEGATIVE, SEMANTIC_POSITIVE, SEMANTIC_POSITIVE, NEGATIVE, CORE_POSITIVE],
            [0.1, 0.0, 0.7, 0.78, 0.0, 1.0],
        ),
    ]


def build(arm="O3_dual_opacity_reassignment", allow=True):
    return build_instance_field(
        sources=sources(),
        arm=arm,
        allow_ambiguous_reassignment=allow,
        ambiguous_minimum_score=0.35,
        ambiguous_minimum_boundary_score=0.05,
        assignment_minimum_margin=0.05,
        rigid_core_opacity=0.98,
        unassigned_opacity=0.0001,
    )


def test_field_keeps_rigid_identity_and_adds_fail_closed_reassignment() -> None:
    field = build()
    np.testing.assert_array_equal(
        field["hard_instance_id"], [13, 13, 41, 41, 13, 41]
    )
    assert field["provenance"][1] == PROVENANCE_AMBIGUOUS_REASSIGNED
    assert field["provenance"][4] == PROVENANCE_RIGID_CORE
    assert field["provenance"][5] == PROVENANCE_RIGID_CORE


def test_o1_excludes_ambiguous_candidate() -> None:
    field = build(arm="O1_dual_opacity", allow=False)
    assert field["hard_instance_id"][1] == NO_INSTANCE
    assert not field["trainable"][1]


def test_heuristic_arm_has_no_trainable_parameter() -> None:
    field = build(arm="O0_heuristic", allow=False)
    assert not field["trainable"].any()


def test_field_round_trip_is_pickle_free(tmp_path: Path) -> None:
    field = build()
    path = tmp_path / "instance_field.npz"
    atomic_save_instance_field(path, field)
    loaded = load_instance_field(path)
    for name in field:
        np.testing.assert_array_equal(loaded[name], field[name])


def test_field_serialization_is_byte_exact(tmp_path: Path) -> None:
    field = build()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    atomic_save_instance_field(first, field)
    atomic_save_instance_field(second, field)
    assert first.read_bytes() == second.read_bytes()


def test_unknown_instance_id_fails_closed() -> None:
    field = build()
    field["hard_instance_id"][0] = 999
    with pytest.raises(ValueError, match="未知身份"):
        validate_instance_field(field)
