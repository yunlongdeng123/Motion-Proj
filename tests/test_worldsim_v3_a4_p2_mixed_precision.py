from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import pytest
import torch

from motion_proj.worldsim_v3.mixed_precision import (
    RENDER_PARAMETER_FIELDS,
    apply_runtime_parameter_dtypes,
    checkpoint_schema,
    conversion_audit,
    convert_checkpoint_state,
    install_fp32_renderer_input_adapter,
    persistent_parameter_inventory,
    renderer_adapter_summary,
    runtime_converted_field_audit,
    select_precision_arm,
    set_fp32_renderer_adapter_mode,
)


def checkpoint_fixture() -> OrderedDict:
    def model(count: int, *, rigid: bool) -> OrderedDict:
        state = OrderedDict(
            {
                "_means": torch.linspace(-3, 3, count * 3).reshape(count, 3),
                "_scales": torch.linspace(-2, 1, count * 3).reshape(count, 3),
                "_quats": torch.linspace(-1, 1, count * 4).reshape(count, 4),
                "_features_dc": torch.linspace(-0.5, 0.5, count * 3).reshape(count, 3),
                "_features_rest": torch.linspace(-0.2, 0.2, count * 6).reshape(count, 2, 3),
                "_opacities": torch.linspace(-4, 4, count).reshape(count, 1),
                "worldsim_a2_ancestry": {"fields": {"gaussian_id": torch.arange(count)}},
            }
        )
        if rigid:
            state.update(
                {
                    "points_ids": torch.arange(count).remainder(2).reshape(-1, 1),
                    "instances_quats": torch.ones(2, 2, 4),
                    "instances_trans": torch.ones(2, 2, 3),
                    "instances_size": torch.ones(2, 3),
                    "instances_fv": torch.ones(2, 2, dtype=torch.bool),
                }
            )
        return state

    return OrderedDict(
        {
            "models": {
                "Background": model(3, rigid=False),
                "RigidNodes": model(4, rigid=True),
                "Sky": {"base": torch.ones(2, 3)},
            },
            "lpips": {"weight": torch.ones(1)},
            "step": torch.tensor(30_000),
        }
    )


def test_conversion_changes_only_frozen_ten_fields() -> None:
    source = checkpoint_fixture()
    candidate = convert_checkpoint_state(source)
    audit = conversion_audit(source, candidate)
    assert audit["converted_field_set_exact"]
    assert audit["all_converted_fields_bitwise_exact"]
    assert audit["preserved_tensors_exact"]
    assert audit["checkpoint_schema_exact"]
    assert len(audit["converted_fields"]) == 10
    assert candidate["models"]["Background"]["_means"] is source["models"]["Background"]["_means"]
    assert candidate["models"]["Sky"] is source["models"]["Sky"]


def test_conversion_audit_detects_preserved_and_converted_drift() -> None:
    source = checkpoint_fixture()
    candidate = convert_checkpoint_state(source)
    candidate["models"]["Sky"] = {"base": torch.zeros(2, 3)}
    assert not conversion_audit(source, candidate)["preserved_tensors_exact"]
    candidate = convert_checkpoint_state(source)
    candidate["models"]["Background"]["_scales"][0, 0] += 1
    assert not conversion_audit(source, candidate)["all_converted_fields_bitwise_exact"]


def test_conversion_refuses_non_float32_or_nonfinite_source() -> None:
    source = checkpoint_fixture()
    source["models"]["Background"]["_scales"] = source["models"]["Background"][
        "_scales"
    ].half()
    with pytest.raises(ValueError, match="float32"):
        convert_checkpoint_state(source)
    source = checkpoint_fixture()
    source["models"]["Background"]["_scales"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        convert_checkpoint_state(source)


def test_checkpoint_schema_binds_container_order_and_tensor_shape_not_dtype() -> None:
    source = checkpoint_fixture()
    candidate = convert_checkpoint_state(source)
    assert checkpoint_schema(source) == checkpoint_schema(candidate)
    candidate["models"]["Background"]["_scales"] = torch.zeros(2, 3, dtype=torch.float16)
    assert checkpoint_schema(source) != checkpoint_schema(candidate)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for field in RENDER_PARAMETER_FIELDS:
            width = 4 if field == "_quats" else 3
            if field == "_opacities":
                width = 1
            self.register_parameter(field, torch.nn.Parameter(torch.ones(2, width)))
        self.register_parameter("_means", torch.nn.Parameter(torch.ones(2, 3)))

    def get_gaussians(self) -> dict[str, torch.Tensor]:
        colors = spherical_harmonics(
            0, self._means, self._features_dc[:, None, :]
        )
        return {
            "_means": self._means,
            "_scales": self._scales,
            "_quats": self._quats,
            "_rgbs": colors,
            "_opacities": self._opacities,
        }


def spherical_harmonics(
    degree: int,
    directions: torch.Tensor,
    coefficients: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> torch.Tensor:
    del degree, masks
    if directions.dtype != torch.float32 or coefficients.dtype != torch.float32:
        raise RuntimeError("fake kernel requires float32")
    return coefficients[:, 0, :] + directions * 0


@dataclass
class FakeGaussians:
    _means: torch.Tensor
    _scales: torch.Tensor
    _quats: torch.Tensor
    _rgbs: torch.Tensor
    _opacities: torch.Tensor


class FakeTrainer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.models = {"Background": FakeModel(), "RigidNodes": FakeModel()}

    def collect_gaussians(self) -> FakeGaussians:
        rows = [self.models[name].get_gaussians() for name in ("Background", "RigidNodes")]
        return FakeGaussians(
            **{
                field: torch.cat([row[field] for row in rows], dim=0)
                for field in ("_means", "_scales", "_quats", "_rgbs", "_opacities")
            }
        )


def test_runtime_dtype_and_renderer_adapter_contract() -> None:
    trainer = FakeTrainer()
    install_fp32_renderer_input_adapter(trainer)
    source_inventory = persistent_parameter_inventory(trainer)
    apply_runtime_parameter_dtypes(trainer, candidate=True)
    assert runtime_converted_field_audit(trainer, expected_dtype="float16")["exact"]
    inventory = persistent_parameter_inventory(trainer)
    assert inventory["bytes_by_dtype"]["float16"] > 0
    assert inventory["bytes_by_dtype"]["float32"] > 0
    assert inventory["total_bytes"] < source_inventory["total_bytes"]
    assert "models.Background._scales" in inventory["parameters"]
    set_fp32_renderer_adapter_mode(trainer, candidate=True)
    gaussians = trainer.collect_gaussians()
    assert all(
        getattr(gaussians, field).dtype == torch.float32
        for field in ("_means", "_scales", "_quats", "_rgbs", "_opacities")
    )
    assert renderer_adapter_summary(trainer)["all_renderer_inputs_float32"]
    assert renderer_adapter_summary(trainer)["spherical_harmonics_inputs_float32"]


def test_runtime_adapter_source_mode_preserves_float32() -> None:
    trainer = FakeTrainer()
    install_fp32_renderer_input_adapter(trainer)
    apply_runtime_parameter_dtypes(trainer, candidate=False)
    set_fp32_renderer_adapter_mode(trainer, candidate=False)
    trainer.collect_gaussians()
    summary = renderer_adapter_summary(trainer)
    assert summary["observation_count"] == 1
    assert summary["all_renderer_inputs_float32"]


def test_precision_selection_is_fail_closed() -> None:
    eligible = {
        "checkpoint_conversion_and_preservation_exact": True,
        "candidate_checkpoint_reload_and_runtime_dtype_exact": True,
        "renderer_input_float32_exact": True,
        "source_baseline_replay_exact": True,
        "all_quality_safeguards_pass": True,
        "checkpoint_bytes_strictly_less_than_source": True,
        "source_inputs_unchanged": True,
        "resources_within_frozen_ceilings": True,
    }
    assert select_precision_arm(eligible)["selected_arm"] == "p2-gs-param-fp16"
    for key in eligible:
        failed = dict(eligible)
        failed[key] = False
        result = select_precision_arm(failed)
        assert result == {
            "selected_arm": "p2-source",
            "method_state": "rejected_numeric_quality_integrity_or_resource_gate",
            "fallback_exact_alias": True,
        }
