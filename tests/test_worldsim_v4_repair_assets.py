from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from motion_proj.worldsim_v33.spatial_delta import build_erase_delta
from motion_proj.worldsim_v4.repair_assets import (
    atomic_save_repair_asset,
    build_repair_asset,
    load_repair_asset,
    temporary_repair_composition,
    verify_repair_asset_binding,
)


def _asset(count: int = 2) -> dict[str, np.ndarray]:
    return build_repair_asset(
        candidate_id="scene-0994-r0-observed",
        method="OBSERVED",
        provenance="observed_cross_view",
        means=np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10,
        raw_scales=np.full((count, 3), -3.0, dtype=np.float32),
        quats=np.tile(np.asarray([[1, 0, 0, 0]], dtype=np.float32), (count, 1)),
        features_dc=np.zeros((count, 3), dtype=np.float32),
        features_rest=np.zeros((count, 2, 3), dtype=np.float32),
        raw_opacities=np.zeros((count, 1), dtype=np.float32),
        confidence=np.full(count, 0.9, dtype=np.float32),
        source_frames=np.full(count, 97, dtype=np.int32),
        source_camera_ids=np.full(count, 2, dtype=np.int16),
        source_pixels_xy=np.arange(count * 2, dtype=np.int32).reshape(count, 2),
    )


def _field() -> dict[str, np.ndarray]:
    return {
        "gaussian_id": np.arange(5, dtype=np.int64),
        "base_model": np.asarray([0, 0, 0, 1, 1], dtype=np.int8),
        "base_index": np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
        "hard_instance_id": np.asarray([9, -1, -1, 9, -1], dtype=np.int32),
        "instance_opacity_logit": np.zeros(5, dtype=np.float32),
        "instance_opacity": np.full(5, 0.5, dtype=np.float32),
        "source_semantic_score": np.full(5, 0.5, dtype=np.float32),
        "num_positive_views": np.ones(5, dtype=np.int32),
        "num_negative_views": np.zeros(5, dtype=np.int32),
        "visibility_mass": np.ones(5, dtype=np.float32),
        "trainable": np.asarray([1, 0, 0, 1, 0], dtype=bool),
        "provenance": np.asarray([2, 0, 0, 1, 0], dtype=np.uint8),
        "actor_instance_ids": np.asarray([9], dtype=np.int32),
        "actor_tokens": np.asarray(["actor-9"], dtype="<U64"),
    }


class _Gaussian:
    def __init__(self, count: int) -> None:
        self._means = torch.nn.Parameter(torch.zeros(count, 3))
        self._scales = torch.nn.Parameter(torch.zeros(count, 3))
        self._quats = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
        )
        self._features_dc = torch.nn.Parameter(torch.zeros(count, 3))
        self._features_rest = torch.nn.Parameter(torch.zeros(count, 2, 3))
        self._opacities = torch.nn.Parameter(torch.zeros(count, 1))


class _Rigid(_Gaussian):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.point_ids = torch.arange(count, dtype=torch.int64)[:, None]


def _models() -> dict[str, object]:
    return {"Background": _Gaussian(3), "RigidNodes": _Rigid(2)}


def _references(models: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, model in models.items():
        for attribute in (
            "_means", "_scales", "_quats", "_features_dc", "_features_rest", "_opacities"
        ):
            result[f"{name}.{attribute}"] = getattr(model, attribute)
    result["RigidNodes.point_ids"] = models["RigidNodes"].point_ids
    return result


def test_repair_asset_is_deterministic_content_addressed_and_verified(tmp_path) -> None:
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    binding = atomic_save_repair_asset(first, _asset())
    atomic_save_repair_asset(second, _asset())
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    restored = verify_repair_asset_binding(binding)
    assert restored["method"].item() == "OBSERVED"
    assert binding.gaussian_count == 2
    first.write_bytes(first.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bytes"):
        verify_repair_asset_binding(binding)


def test_repair_asset_rejects_manifest_row_provenance_drift() -> None:
    asset = _asset()
    asset["point_provenance"][0] = 3
    with pytest.raises(ValueError, match="provenance"):
        load_repair_asset  # keep import exercised
        from motion_proj.worldsim_v4.repair_assets import validate_repair_asset

        validate_repair_asset(asset)


def test_temporary_repair_composition_is_atomic_and_restores_on_exception() -> None:
    models = _models()
    before = _references(models)
    erase = build_erase_delta(_field(), instance_id=9)
    with pytest.raises(RuntimeError, match="sentinel"):
        with temporary_repair_composition(models, erase_delta=erase, asset=_asset()) as audit:
            assert audit["method"] == "OBSERVED"
            assert models["Background"]._means.shape[0] == 5
            assert torch.sigmoid(models["Background"]._opacities[0]).count_nonzero() == 0
            assert torch.sigmoid(models["RigidNodes"]._opacities[0]).count_nonzero() == 0
            raise RuntimeError("sentinel")
    after = _references(models)
    assert all(after[name] is value for name, value in before.items())
