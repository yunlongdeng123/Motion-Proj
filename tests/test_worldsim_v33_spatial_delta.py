from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from motion_proj.worldsim_v33.spatial_delta import (
    ERASE_SCHEMA_VERSION,
    atomic_save_actor_insert_delta,
    atomic_save_erase_delta,
    build_actor_insert_delta,
    build_erase_delta,
    load_actor_insert_delta,
    load_erase_delta,
    ordered_stack_manifest,
    temporary_spatial_composition,
    validate_erase_delta,
    validate_stack_manifest,
)


def _field() -> dict[str, np.ndarray]:
    actor_ids = np.asarray([13, 41], dtype=np.int32)
    hard = np.asarray([13, -1, 41, 13, 13, 41, -1], dtype=np.int32)
    count = int(hard.size)
    return {
        "gaussian_id": np.arange(count, dtype=np.int64),
        "base_model": np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int8),
        "base_index": np.asarray([0, 1, 2, 3, 0, 1, 2], dtype=np.int64),
        "hard_instance_id": hard,
        "instance_opacity_logit": np.zeros(count, dtype=np.float32),
        "instance_opacity": np.full(count, 0.5, dtype=np.float32),
        "source_semantic_score": np.full(count, 0.5, dtype=np.float32),
        "num_positive_views": np.ones(count, dtype=np.int32),
        "num_negative_views": np.zeros(count, dtype=np.int32),
        "visibility_mass": np.ones(count, dtype=np.float32),
        "trainable": hard != -1,
        "provenance": np.asarray([2, 0, 2, 2, 1, 1, 0], dtype=np.uint8),
        "actor_instance_ids": actor_ids,
        "actor_tokens": np.asarray(["actor-13", "actor-41"], dtype="<U64"),
    }


def _actor_asset(count: int = 2) -> dict[str, np.ndarray]:
    return {
        "means": np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10,
        "scales": np.full((count, 3), 0.05, dtype=np.float32),
        "quats": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)
        ),
        "rgb": np.full((count, 3), 0.6, dtype=np.float32),
        "opacity": np.full(count, 0.8, dtype=np.float32),
        "target_lwh": np.asarray([4.2, 1.8, 1.6], dtype=np.float64),
    }


def _background_delta(count: int = 2) -> dict[str, np.ndarray]:
    return {
        "means": np.full((count, 3), 0.25, dtype=np.float32),
        "raw_scales": np.full((count, 3), -3.0, dtype=np.float32),
        "quats": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)
        ),
        "features_dc": np.zeros((count, 3), dtype=np.float32),
        "features_rest": np.zeros((count, 2, 3), dtype=np.float32),
        "raw_opacities": np.zeros((count, 1), dtype=np.float32),
        "source_flat_indices": np.arange(count, dtype=np.int64),
        "source_gaussian_ids": np.arange(100, 100 + count, dtype=np.int64),
        "feather_weight": np.ones(count, dtype=np.float32),
        "provenance_code": np.full(count, 2, dtype=np.uint8),
        "target_role": np.asarray(["high"] * count, dtype="<U32"),
        "donor_patch_id": np.asarray(["p0"] * count, dtype="<U40"),
        "donor_chunk_ids": np.asarray(["c0"] * count, dtype="<U256"),
    }


class _GaussianModel:
    def __init__(self, count: int) -> None:
        self._means = torch.nn.Parameter(torch.zeros(count, 3))
        self._scales = torch.nn.Parameter(torch.zeros(count, 3))
        self._quats = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
        )
        self._features_dc = torch.nn.Parameter(torch.zeros(count, 3))
        self._features_rest = torch.nn.Parameter(torch.zeros(count, 2, 3))
        self._opacities = torch.nn.Parameter(torch.zeros(count, 1))


class _RigidModel(_GaussianModel):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.point_ids = torch.asarray([[5], [7], [9]], dtype=torch.int64)[:count]
        self.instances_size = torch.ones(10, 3)
        self.sh_degree = 1


def _models() -> dict[str, object]:
    return {"Background": _GaussianModel(4), "RigidNodes": _RigidModel(3)}


def _identity_snapshot(models: dict[str, object]) -> dict[str, object]:
    output = {}
    for model_name in ("Background", "RigidNodes"):
        model = models[model_name]
        for attribute in (
            "_means",
            "_scales",
            "_quats",
            "_features_dc",
            "_features_rest",
            "_opacities",
        ):
            output[f"{model_name}.{attribute}"] = getattr(model, attribute)
    rigid = models["RigidNodes"]
    output["point_ids"] = rigid.point_ids
    output["instances_size"] = rigid.instances_size
    return output


def test_erase_delta_is_deterministic_and_bound_to_mask(tmp_path) -> None:
    delta = build_erase_delta(_field(), instance_id=13)
    assert str(delta["schema_version"].item()) == ERASE_SCHEMA_VERSION
    assert delta["source_flat_indices"].tolist() == [0, 3, 0]
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    atomic_save_erase_delta(first, delta)
    atomic_save_erase_delta(second, delta)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    restored = load_erase_delta(first)
    np.testing.assert_array_equal(restored["gaussian_ids"], delta["gaussian_ids"])


def test_erase_uses_s1_probability_for_background_but_keeps_rigid_core() -> None:
    field = _field()
    field["instance_opacity"] = np.asarray(
        [0.8, 0.5, 0.7, 0.2, 0.1, 0.9, 0.1], dtype=np.float32
    )
    field["instance_opacity_logit"] = np.log(
        field["instance_opacity"] / (1.0 - field["instance_opacity"])
    ).astype(np.float32)
    delta = build_erase_delta(
        field, instance_id=13, minimum_background_instance_opacity=0.5
    )
    assert delta["model_code"].tolist() == [0, 1]
    assert delta["source_flat_indices"].tolist() == [0, 0]
    assert str(delta["selection_policy"].item()) == (
        "rigid_core_plus_background_probability_ge"
    )


def test_erase_duplicate_and_out_of_bounds_fail_closed() -> None:
    delta = build_erase_delta(_field(), instance_id=13)
    duplicate = {name: value.copy() for name, value in delta.items()}
    duplicate["source_flat_indices"][1] = duplicate["source_flat_indices"][0]
    duplicate["model_code"][1] = duplicate["model_code"][0]
    with pytest.raises(ValueError, match="重复"):
        validate_erase_delta(duplicate)
    with pytest.raises(ValueError, match="越界"):
        validate_erase_delta(
            delta, model_counts={"Background": 2, "RigidNodes": 3}
        )


def test_erase_probability_policy_rejects_low_score_background() -> None:
    field = _field()
    delta = build_erase_delta(
        field, instance_id=13, minimum_background_instance_opacity=0.4
    )
    tampered = {name: value.copy() for name, value in delta.items()}
    background = np.flatnonzero(tampered["model_code"] == 0)
    tampered["selection_score"][background[0]] = 0.1
    with pytest.raises(ValueError, match="低于"):
        validate_erase_delta(tampered)


def test_actor_insert_records_per_gaussian_provenance(tmp_path) -> None:
    delta = build_actor_insert_delta(
        _actor_asset(),
        instance_id=13,
        instance_token="actor-13",
        rigid_model_index=5,
    )
    assert delta["provenance_code"].tolist() == [3, 3]
    assert delta["source_asset_gaussian_index"].tolist() == [0, 1]
    path = tmp_path / "actor.npz"
    atomic_save_actor_insert_delta(path, delta)
    restored = load_actor_insert_delta(path)
    np.testing.assert_array_equal(restored["means"], delta["means"])


def test_stack_is_canonicalized_and_rejects_order_or_duplicates() -> None:
    manifest = ordered_stack_manifest(
        stack_id="full",
        operations=[
            {"operation_id": "actor", "type": "INSERT_ACTOR"},
            {"operation_id": "erase", "type": "ERASE"},
            {"operation_id": "bg", "type": "INSERT_BACKGROUND"},
        ],
    )
    assert [row["type"] for row in manifest["operations"]] == [
        "ERASE",
        "INSERT_BACKGROUND",
        "INSERT_ACTOR",
    ]
    bad = {**manifest, "operations": list(reversed(manifest["operations"]))}
    with pytest.raises(ValueError, match="顺序"):
        validate_stack_manifest(bad)
    duplicate = {**manifest, "operations": manifest["operations"] * 2}
    with pytest.raises(ValueError, match="唯一"):
        validate_stack_manifest(duplicate)


def test_composition_keeps_base_rows_masks_exactly_and_restores_objects() -> None:
    models = _models()
    before = _identity_snapshot(models)
    erase = build_erase_delta(_field(), instance_id=13)
    actor = build_actor_insert_delta(
        _actor_asset(), instance_id=13, instance_token="actor-13", rigid_model_index=5
    )
    with temporary_spatial_composition(
        models,
        erase_delta=erase,
        background_delta=_background_delta(),
        actor_delta=actor,
    ) as audit:
        background, rigid = models["Background"], models["RigidNodes"]
        assert background._means.shape[0] == 6
        assert rigid._means.shape[0] == 5
        assert audit["base_rows_deleted"] == 0
        assert audit["effective_erased_opacity_nonzero"] == 0
        assert torch.equal(rigid.point_ids[:3], before["point_ids"])
        assert rigid.point_ids[-2:, 0].tolist() == [5, 5]
        assert torch.sigmoid(background._opacities[[0, 3]]).count_nonzero() == 0
        assert torch.sigmoid(rigid._opacities[[0]]).count_nonzero() == 0
    after = _identity_snapshot(models)
    assert all(after[name] is value for name, value in before.items())


def test_composition_rolls_back_on_exception() -> None:
    models = _models()
    before = _identity_snapshot(models)
    erase = build_erase_delta(_field(), instance_id=13)
    with pytest.raises(RuntimeError, match="sentinel"):
        with temporary_spatial_composition(models, erase_delta=erase):
            raise RuntimeError("sentinel")
    after = _identity_snapshot(models)
    assert all(after[name] is value for name, value in before.items())


def test_duplicate_background_source_is_rejected_before_mutation() -> None:
    models = _models()
    before = _identity_snapshot(models)
    delta = _background_delta()
    delta["source_gaussian_ids"][1] = delta["source_gaussian_ids"][0]
    with pytest.raises(ValueError, match="重复"):
        with temporary_spatial_composition(
            models,
            erase_delta=build_erase_delta(_field(), instance_id=13),
            background_delta=delta,
        ):
            pass
    after = _identity_snapshot(models)
    assert all(after[name] is value for name, value in before.items())
