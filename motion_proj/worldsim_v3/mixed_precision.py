"""A4-P2 混合精度检查点转换、运行时适配与确定性裁决工具。"""

from __future__ import annotations

from collections import OrderedDict
import math
import types
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from motion_proj.worldsim_v3.contribution_prune import tensor_sha256


RENDER_PARAMETER_FIELDS = (
    "_scales",
    "_quats",
    "_features_dc",
    "_features_rest",
    "_opacities",
)
CONVERTED_MODELS = ("Background", "RigidNodes")
RENDER_INPUT_FIELDS = ("_means", "_scales", "_quats", "_rgbs", "_opacities")


def recursive_tensor_rows(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    """递归记录 tensor 的 dtype、shape、字节数与内容指纹。"""
    if isinstance(value, torch.Tensor):
        return {
            prefix: {
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape),
                "bytes": int(value.numel() * value.element_size()),
                "sha256": tensor_sha256(value),
            }
        }
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            rows.update(recursive_tensor_rows(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            rows.update(recursive_tensor_rows(child, child_prefix))
    return rows


def checkpoint_schema(value: Any) -> Any:
    """生成忽略 tensor dtype、但绑定容器类型、键与 tensor shape 的结构签名。"""
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {
            "kind": type(value).__name__,
            "items": [(str(name), checkpoint_schema(child)) for name, child in value.items()],
        }
    if isinstance(value, list):
        return {"kind": "list", "items": [checkpoint_schema(child) for child in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [checkpoint_schema(child) for child in value]}
    return {"kind": type(value).__name__}


def converted_paths(
    models: Sequence[str] = CONVERTED_MODELS,
    fields: Sequence[str] = RENDER_PARAMETER_FIELDS,
) -> tuple[str, ...]:
    return tuple(f"models.{model}.{field}" for model in models for field in fields)


def convert_checkpoint_state(
    source: Mapping[str, Any],
    *,
    models: Sequence[str] = CONVERTED_MODELS,
    fields: Sequence[str] = RENDER_PARAMETER_FIELDS,
) -> MutableMapping[str, Any]:
    """仅替换冻结的十个 Gaussian 参数，其余对象保持只读共享。"""
    output: MutableMapping[str, Any] = type(source)(source.items())
    source_models = source["models"]
    output_models: MutableMapping[str, Any] = type(source_models)(source_models.items())
    for model_name in models:
        source_model = source_models[model_name]
        output_model: MutableMapping[str, Any] = type(source_model)(source_model.items())
        for field in fields:
            tensor = source_model[field]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                raise ValueError(f"A4-P2 source field must be float32 tensor: {model_name}.{field}")
            candidate = tensor.to(dtype=torch.float16)
            if not torch.isfinite(candidate).all().item():
                raise ValueError(f"A4-P2 FP16 conversion produced non-finite values: {model_name}.{field}")
            output_model[field] = candidate
        output_models[model_name] = output_model
    output["models"] = output_models
    return OrderedDict(output.items()) if isinstance(source, OrderedDict) else output


def _relative_error(source: torch.Tensor, roundtrip: torch.Tensor) -> float:
    denominator = source.detach().float().abs().clamp_min(1e-8)
    return float(((roundtrip - source.detach().float()).abs() / denominator).max().item())


def conversion_audit(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    models: Sequence[str] = CONVERTED_MODELS,
    fields: Sequence[str] = RENDER_PARAMETER_FIELDS,
) -> dict[str, Any]:
    """核对转换位级等价、误差有限以及非转换 tensor 完整保留。"""
    source_rows = recursive_tensor_rows(source)
    candidate_rows = recursive_tensor_rows(candidate)
    expected = set(converted_paths(models, fields))
    converted = {}
    for model_name in models:
        for field in fields:
            path = f"models.{model_name}.{field}"
            before = source["models"][model_name][field]
            after = candidate["models"][model_name][field]
            expected_half = before.to(torch.float16)
            roundtrip = after.float()
            delta = roundtrip - before.float()
            converted[path] = {
                "source_dtype": str(before.dtype).removeprefix("torch."),
                "candidate_dtype": str(after.dtype).removeprefix("torch."),
                "shape": list(before.shape),
                "source_bytes": int(before.numel() * before.element_size()),
                "candidate_bytes": int(after.numel() * after.element_size()),
                "source_sha256": tensor_sha256(before),
                "candidate_sha256": tensor_sha256(after),
                "expected_half_sha256": tensor_sha256(expected_half),
                "bitwise_equal_source_to_float16": torch.equal(after, expected_half),
                "source_min": float(before.min().item()),
                "source_max": float(before.max().item()),
                "max_absolute_error": float(delta.abs().max().item()),
                "mean_absolute_error": float(delta.abs().mean().item()),
                "root_mean_square_error": float(torch.sqrt(torch.mean(delta.square())).item()),
                "max_relative_error_eps_1e-8": _relative_error(before, roundtrip),
                "nonfinite_source_count": int((~torch.isfinite(before)).sum().item()),
                "nonfinite_candidate_count": int((~torch.isfinite(after)).sum().item()),
            }
    preserved_paths = sorted(set(source_rows) - expected)
    preserved_exact = (
        set(source_rows) == set(candidate_rows)
        and all(source_rows[path] == candidate_rows[path] for path in preserved_paths)
    )
    return {
        "converted_field_set_exact": set(converted) == expected,
        "converted_fields": converted,
        "all_converted_fields_bitwise_exact": all(
            row["bitwise_equal_source_to_float16"]
            and row["source_dtype"] == "float32"
            and row["candidate_dtype"] == "float16"
            and row["nonfinite_source_count"] == 0
            and row["nonfinite_candidate_count"] == 0
            and all(
                math.isfinite(float(row[name]))
                for name in (
                    "source_min",
                    "source_max",
                    "max_absolute_error",
                    "mean_absolute_error",
                    "root_mean_square_error",
                    "max_relative_error_eps_1e-8",
                )
            )
            for row in converted.values()
        ),
        "preserved_tensor_count": len(preserved_paths),
        "preserved_tensor_paths": preserved_paths,
        "preserved_tensors_exact": preserved_exact,
        "checkpoint_schema_exact": checkpoint_schema(source) == checkpoint_schema(candidate),
    }


def apply_runtime_parameter_dtypes(
    trainer: Any,
    *,
    candidate: bool,
    models: Sequence[str] = CONVERTED_MODELS,
    fields: Sequence[str] = RENDER_PARAMETER_FIELDS,
) -> dict[str, dict[str, Any]]:
    """把 runtime 中冻结的字段切换到目标 dtype，并返回逐字段审计。"""
    target = torch.float16 if candidate else torch.float32
    rows = {}
    for model_name in models:
        model = trainer.models[model_name]
        for field in fields:
            parameter = getattr(model, field)
            parameter.data = parameter.detach().to(dtype=target)
            rows[f"models.{model_name}.{field}"] = {
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "shape": list(parameter.shape),
                "bytes": int(parameter.numel() * parameter.element_size()),
            }
    return rows


def persistent_parameter_inventory(trainer: Any) -> dict[str, Any]:
    """汇总 trainer 的持久 Parameter 字节，并保留逐参数 dtype 账本。"""
    by_dtype: dict[str, int] = {}
    rows = {}
    for name, parameter in trainer.named_parameters():
        dtype = str(parameter.dtype).removeprefix("torch.")
        size = int(parameter.numel() * parameter.element_size())
        by_dtype[dtype] = by_dtype.get(dtype, 0) + size
        rows[name] = {"dtype": dtype, "shape": list(parameter.shape), "bytes": size}
    return {
        "bytes_by_dtype": dict(sorted(by_dtype.items())),
        "total_bytes": sum(by_dtype.values()),
        "parameters": rows,
    }


def runtime_converted_field_audit(
    trainer: Any,
    *,
    expected_dtype: str,
    models: Sequence[str] = CONVERTED_MODELS,
    fields: Sequence[str] = RENDER_PARAMETER_FIELDS,
) -> dict[str, Any]:
    rows = {}
    for model_name in models:
        model = trainer.models[model_name]
        for field in fields:
            parameter = getattr(model, field)
            rows[f"models.{model_name}.{field}"] = {
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "shape": list(parameter.shape),
                "finite": bool(torch.isfinite(parameter).all().item()),
            }
    return {
        "expected_dtype": expected_dtype,
        "fields": rows,
        "exact": len(rows) == len(models) * len(fields)
        and all(row["dtype"] == expected_dtype and row["finite"] for row in rows.values()),
    }


def install_fp32_renderer_input_adapter(trainer: Any) -> MutableMapping[str, Any]:
    """安装一次性适配器：仅候选 arm 在 collect 后将 gsplat 输入转为 FP32。"""
    existing = getattr(trainer, "_worldsim_p2_adapter_audit", None)
    if existing is not None:
        return existing
    original = trainer.collect_gaussians
    audit: MutableMapping[str, Any] = {
        "installed": True,
        "candidate_mode": False,
        "autocast_enabled": False,
        "observations": [],
        "spherical_harmonics_observations": [],
    }

    for model_name in CONVERTED_MODELS:
        model = trainer.models[model_name]
        original_model_collect = model.get_gaussians
        function_globals = original_model_collect.__func__.__globals__
        original_spherical_harmonics = function_globals.get("spherical_harmonics")
        if original_spherical_harmonics is None:
            continue

        def model_collect_with_fp32_sh(
            self: Any,
            *args: Any,
            _original: Any = original_model_collect,
            _globals: MutableMapping[str, Any] = function_globals,
            _spherical_harmonics: Any = original_spherical_harmonics,
            _model_name: str = model_name,
            **kwargs: Any,
        ) -> Any:
            previous = _globals["spherical_harmonics"]

            def spherical_harmonics_fp32(
                degree: int,
                directions: torch.Tensor,
                coefficients: torch.Tensor,
                masks: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if audit["candidate_mode"]:
                    directions = directions.float()
                    coefficients = coefficients.float()
                audit["spherical_harmonics_observations"].append(
                    {
                        "model": _model_name,
                        "candidate_mode": bool(audit["candidate_mode"]),
                        "directions_dtype": str(directions.dtype).removeprefix("torch."),
                        "coefficients_dtype": str(coefficients.dtype).removeprefix("torch."),
                    }
                )
                return _spherical_harmonics(degree, directions, coefficients, masks)

            _globals["spherical_harmonics"] = spherical_harmonics_fp32
            try:
                return _original(*args, **kwargs)
            finally:
                _globals["spherical_harmonics"] = previous

        model.get_gaussians = types.MethodType(model_collect_with_fp32_sh, model)

    def collect_with_fp32(self: Any, *args: Any, **kwargs: Any) -> Any:
        gaussians = original(*args, **kwargs)
        before = {
            field: str(getattr(gaussians, field).dtype).removeprefix("torch.")
            for field in RENDER_INPUT_FIELDS
        }
        if audit["candidate_mode"]:
            for field in RENDER_INPUT_FIELDS:
                setattr(gaussians, field, getattr(gaussians, field).float())
        after = {
            field: str(getattr(gaussians, field).dtype).removeprefix("torch.")
            for field in RENDER_INPUT_FIELDS
        }
        audit["observations"].append(
            {
                "candidate_mode": bool(audit["candidate_mode"]),
                "autocast_enabled": bool(torch.is_autocast_enabled()),
                "before": before,
                "after": after,
            }
        )
        return gaussians

    trainer.collect_gaussians = types.MethodType(collect_with_fp32, trainer)
    trainer._worldsim_p2_adapter_audit = audit
    return audit


def set_fp32_renderer_adapter_mode(trainer: Any, *, candidate: bool) -> None:
    audit = install_fp32_renderer_input_adapter(trainer)
    audit["candidate_mode"] = bool(candidate)
    audit["observations"] = []
    audit["spherical_harmonics_observations"] = []


def renderer_adapter_summary(trainer: Any) -> dict[str, Any]:
    audit = install_fp32_renderer_input_adapter(trainer)
    observations = list(audit["observations"])
    sh_observations = list(audit["spherical_harmonics_observations"])
    return {
        "installed": bool(audit["installed"]),
        "candidate_mode": bool(audit["candidate_mode"]),
        "autocast_enabled": bool(audit["autocast_enabled"]),
        "observation_count": len(observations),
        "observations": observations,
        "spherical_harmonics_observation_count": len(sh_observations),
        "spherical_harmonics_observations": sh_observations,
        "spherical_harmonics_inputs_float32": bool(sh_observations)
        and all(
            row["directions_dtype"] == "float32"
            and row["coefficients_dtype"] == "float32"
            for row in sh_observations
        ),
        "all_renderer_inputs_float32": bool(observations)
        and all(
            not row["autocast_enabled"]
            and all(dtype == "float32" for dtype in row["after"].values())
            for row in observations
        ),
    }


def select_precision_arm(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """仅当候选满足全部冻结门槛时选择 mixed-precision arm。"""
    required = (
        "checkpoint_conversion_and_preservation_exact",
        "candidate_checkpoint_reload_and_runtime_dtype_exact",
        "renderer_input_float32_exact",
        "source_baseline_replay_exact",
        "all_quality_safeguards_pass",
        "checkpoint_bytes_strictly_less_than_source",
        "source_inputs_unchanged",
        "resources_within_frozen_ceilings",
    )
    eligible = all(bool(candidate.get(name)) for name in required)
    if eligible:
        return {
            "selected_arm": "p2-gs-param-fp16",
            "method_state": "selected_mixed_precision_parameter_storage_fp32_render",
            "fallback_exact_alias": False,
        }
    return {
        "selected_arm": "p2-source",
        "method_state": "rejected_numeric_quality_integrity_or_resource_gate",
        "fallback_exact_alias": True,
    }
