import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from motion_proj.train.pilot import (
    _aggregate_evaluation_rows,
    _record_backbone_provenance,
    _to_batch,
    capacity_decision,
    select_pair_indices,
)


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


def test_evaluation_uses_worst_pair_for_strict_gates():
    rows = [
        {
            "target_error": 1.0,
            "outside_teacher_drift_ratio": 0.01,
            "frame0_teacher_drift": 0.1,
            "target_roundtrip_max_error": 9.0e-6,
            "correction_direction_cosine": 0.2,
        },
        {
            "target_error": 3.0,
            "outside_teacher_drift_ratio": 0.03,
            "frame0_teacher_drift": 0.3,
            "target_roundtrip_max_error": 2.3e-5,
            "correction_direction_cosine": 0.4,
        },
    ]
    result = _aggregate_evaluation_rows(rows)
    assert result["target_error"] == 2.0
    assert result["outside_teacher_drift_ratio"] == 0.02
    assert result["outside_teacher_drift_ratio_max"] == 0.03
    assert result["frame0_teacher_drift"] == 0.3
    assert result["target_roundtrip_max_error"] == 2.3e-5
    assert result["correction_direction_cosine"] == pytest.approx(0.3)


def test_pilot_persists_lora_provenance(tmp_path: Path):
    class _Backbone:
        def adapter_metadata(self):
            return {
                "scope": "temporal_only",
                "rank": 16,
                "selected_module_names": ["down.temporal_transformer_blocks.0.attn1.to_q"],
                "selected_module_count": 1,
                "temporal_module_count": 1,
                "spatial_module_count": 0,
                "trainable_tensor_count": 2,
                "trainable_parameter_count": 128,
                "adapter_tensor_count": 2,
            }

    manifest = {"status": "running"}
    _record_backbone_provenance(
        tmp_path,
        manifest,
        _Backbone(),
        SimpleNamespace(model=SimpleNamespace(name="svd")),
    )

    persisted = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    selected = (tmp_path / "selected_modules.txt").read_text(encoding="utf-8").splitlines()
    assert selected == persisted["model"]["adapter"]["selected_module_names"]
    assert persisted["model"]["adapter"]["trainable_parameter_count"] == 128


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
