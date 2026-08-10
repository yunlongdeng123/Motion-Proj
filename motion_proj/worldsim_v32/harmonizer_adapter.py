"""NVIDIA Harmonizer 非时序 JIT 的 StreetGS RGB 适配器。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


_TEX_LIBRARY: torch.library.Library | None = None
_TEX_RUNTIME: str | None = None


def rmsnorm_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool = False,
) -> torch.Tensor:
    """按 Transformer Engine RMSNorm 定义计算等价参考值。"""
    scale = weight + 1 if zero_centered_gamma else weight
    variance = input_tensor.float().square().mean(dim=-1, keepdim=True)
    normalized = input_tensor.float() * torch.rsqrt(variance + eps)
    return (normalized * scale.float()).to(dtype=input_tensor.dtype)


def register_tex_ts_rmsnorm_fallback() -> str:
    """缺少 NGC Transformer Engine 时注册导出图所需的唯一算子。"""
    global _TEX_LIBRARY, _TEX_RUNTIME
    if _TEX_RUNTIME is not None:
        return _TEX_RUNTIME
    try:
        getattr(torch.ops.tex_ts, "rmsnorm_fwd_inf_ts")
        _TEX_RUNTIME = "native_tex_ts"
        return _TEX_RUNTIME
    except AttributeError:
        pass

    library = torch.library.Library("tex_ts", "DEF")
    library.define(
        "rmsnorm_fwd_inf_ts(Tensor input, Tensor weight, float eps, "
        "int sm_margin, bool zero_centered_gamma) -> Tensor"
    )

    def rmsnorm_fwd_inf_ts(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> torch.Tensor:
        del sm_margin
        return rmsnorm_reference(input_tensor, weight, eps, zero_centered_gamma)

    library.impl(
        "rmsnorm_fwd_inf_ts",
        rmsnorm_fwd_inf_ts,
        "CompositeExplicitAutograd",
    )
    _TEX_LIBRARY = library
    _TEX_RUNTIME = "formula_equivalent_rmsnorm_v1"
    return _TEX_RUNTIME


def normalize_embedded_tensor_constant_devices(
    module: torch.jit.ScriptModule,
    device: torch.device,
) -> dict[str, int]:
    """修复旧式 einops trace 中形状标量被 map_location 移到 CUDA 的问题。"""
    counts = {
        "tensor_constants": 0,
        "shape_scalars_moved_to_cpu": 0,
        "model_constants_moved_to_cuda": 0,
    }

    def visit_block(block: Any) -> None:
        for node in block.nodes():
            if node.kind() == "prim::Constant" and node.hasAttribute("value"):
                if node.kindOf("value") == "t":
                    counts["tensor_constants"] += 1
                    value = node.t("value")
                    is_integral_scalar = value.numel() == 1 and not (
                        value.is_floating_point() or value.is_complex()
                    )
                    # 导出图只用整数 1/2 参与 einops 形状乘除；它们必须留在 CPU。
                    is_shape_scalar = is_integral_scalar and int(value.item()) in {1, 2}
                    if is_shape_scalar and value.device.type != "cpu":
                        node.t_("value", value.cpu())
                        counts["shape_scalars_moved_to_cpu"] += 1
                    elif not is_shape_scalar and value.device != device:
                        node.t_("value", value.to(device))
                        counts["model_constants_moved_to_cuda"] += 1
            for child in node.blocks():
                visit_block(child)

    visit_block(module.graph)
    return counts


def prepare_image(
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    model_size: tuple[int, int] = (1024, 576),
) -> torch.Tensor:
    """将 RGB 图像按冻结规则变为 B,C,V,H,W 且归一化到 [-1,1]。"""
    resized = image.convert("RGB").resize(model_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.mul(2.0).sub(1.0).to(device=device, dtype=dtype)
    return tensor.unsqueeze(2)


def restore_image(output: torch.Tensor, original_size: tuple[int, int]) -> Image.Image:
    """将 B,C,V,H,W 或 C,H,W 输出恢复到原始 RGB 尺寸。"""
    if output.ndim == 5:
        output = output[0, :, 0]
    elif output.ndim != 3:
        raise ValueError(f"不支持的 Harmonizer 输出维度：{tuple(output.shape)}")
    image = output.float().mul(0.5).add(0.5).clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array).resize(original_size, Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class HarmonizerLoadAudit:
    load_seconds: float
    operator_runtime: str
    constant_device_patch: dict[str, int]


class HarmonizerJITAdapter:
    """单 GPU、非时序 Harmonizer 推理封装；不接触任何 3D checkpoint。"""

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("正式 Harmonizer 推理必须使用 CUDA")
        torch.cuda.set_device(self.device)
        operator_runtime = register_tex_ts_rmsnorm_fallback()
        load_start = time.perf_counter()
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        constant_patch = normalize_embedded_tensor_constant_devices(
            self.model,
            self.device,
        )
        torch.cuda.synchronize(self.device)
        self.load_audit = HarmonizerLoadAudit(
            load_seconds=time.perf_counter() - load_start,
            operator_runtime=operator_runtime,
            constant_device_patch=constant_patch,
        )

    def infer(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        """推理单张图并返回恢复尺寸后的结果和资源统计。"""
        original_size = image.size
        input_tensor = prepare_image(image, self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(input_tensor)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        restored = restore_image(output, original_size)
        return restored, {
            "inference_seconds": elapsed,
            "model_input_shape": list(input_tensor.shape),
            "model_output_shape": list(output.shape),
        }


def validate_rmsnorm_operator(device: str = "cuda:0") -> dict[str, Any]:
    """独立计算参考式，验证注册算子在目标 GPU 上的数值一致性。"""
    runtime = register_tex_ts_rmsnorm_fallback()
    target = torch.device(device)
    generator = torch.Generator(device=target).manual_seed(3204)
    input_tensor = torch.randn(
        (8, 128),
        generator=generator,
        device=target,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (128,),
        generator=generator,
        device=target,
        dtype=torch.bfloat16,
    )
    expected = rmsnorm_reference(input_tensor, weight, 1e-6, False)
    actual = torch.ops.tex_ts.rmsnorm_fwd_inf_ts(input_tensor, weight, 1e-6, 0, False)
    max_abs_error = float((actual.float() - expected.float()).abs().max().item())
    return {
        "runtime": runtime,
        "dtype": str(input_tensor.dtype),
        "shape": list(input_tensor.shape),
        "max_abs_error": max_abs_error,
        "exact": bool(torch.equal(actual, expected)),
    }
