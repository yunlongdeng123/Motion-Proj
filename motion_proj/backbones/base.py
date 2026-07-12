"""SVD/OpenDWM 共用的稳定 diffusion backbone adapter 接口。"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Conditioning:
    """对 trainer 不透明的骨干条件包。"""

    data: dict[str, Any]


@dataclass(frozen=True)
class BackboneCapabilities:
    image_to_video: bool = True
    future_ego_control: bool = False
    layout_control: bool = False
    multi_camera_sync: bool = False
    generation: bool = False
    parameterizations: tuple[str, ...] = ("edm",)
    metadata: dict[str, Any] = field(default_factory=dict)


class DiffusionBackbone(abc.ABC):
    """trainer 唯一依赖的骨干协议，latent 统一为 ``[B,T,C,h,w]``。"""

    parameterization: str = "edm"
    p_mean: float = 0.7
    p_std: float = 1.6
    sigma_data: float = 0.5

    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(parameterizations=(self.parameterization,))

    @abc.abstractmethod
    def encode(self, frames: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def sample_sigmas(self, num: int, device) -> torch.Tensor: ...

    def add_noise(self, x0: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        while sigma.dim() < x0.dim():
            sigma = sigma.unsqueeze(-1)
        return x0 + sigma * noise

    def sample_training_sigma(self, n: int, device) -> torch.Tensor:
        return (torch.randn(n, device=device) * self.p_std + self.p_mean).exp()

    def predict_model_output(
        self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        """返回骨干原生参数化输出；旧骨干需在接入 V2 前显式实现。"""
        raise NotImplementedError("该骨干未实现原生 model output 预测")

    def x0_from_model_output(
        self, z: torch.Tensor, sigma: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("该骨干未实现 model output -> x0 变换")

    def model_output_from_x0(
        self, z: torch.Tensor, sigma: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("该骨干未实现 x0 -> model output 变换")

    def predict_x0(self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning) -> torch.Tensor:
        raw = self.predict_model_output(z, sigma, cond)
        return self.x0_from_model_output(z, sigma, raw)

    def predict_train(self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning) -> torch.Tensor:
        return self.predict_x0(z, sigma, cond)

    def anchor_predict_model_output(
        self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        raise NotImplementedError("该骨干未实现关闭 adapter 的原生 model output 预测")

    def anchor_predict_x0(
        self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        raw = self.anchor_predict_model_output(z, sigma, cond)
        return self.x0_from_model_output(z, sigma, raw)

    def predict_anchor(self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning) -> torch.Tensor:
        return self.anchor_predict_x0(z, sigma, cond)

    @abc.abstractmethod
    def build_conditioning(self, batch: dict) -> Conditioning: ...

    @abc.abstractmethod
    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...

    @abc.abstractmethod
    def save_adapter(self, path: str) -> None: ...

    @abc.abstractmethod
    def load_adapter(self, path: str) -> None: ...

    def adapter_state(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("该骨干未实现内存 adapter state 导出")

    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError("该骨干未实现内存 adapter state 加载")

    def adapter_metadata(self) -> dict[str, Any]:
        """返回可写入 run manifest 的 adapter 隔离与参数统计。"""
        state = self.adapter_state()
        parameters = self.trainable_parameters()
        return {
            "selected_module_names": [],
            "selected_module_count": 0,
            "temporal_module_count": 0,
            "spatial_module_count": 0,
            "trainable_tensor_count": len(parameters),
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "adapter_tensor_count": len(state),
        }

    def set_train_mode(self, enabled: bool = True) -> None:
        for parameter in self.trainable_parameters():
            parameter.requires_grad_(enabled)

    def training_module(self) -> torch.nn.Module:
        """返回 Accelerate 用于 accumulate/no_sync 的实际模块。"""
        raise NotImplementedError("骨干 adapter 必须暴露 training_module")

    def generation(self, cond_frame: torch.Tensor, **kwargs) -> torch.Tensor:
        generate = getattr(self, "generate", None)
        if not callable(generate):
            raise NotImplementedError("该骨干不支持生成")
        return generate(cond_frame, **kwargs)

    def eps_from_x0(self, z: torch.Tensor, sigma: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        while sigma.dim() < z.dim():
            sigma = sigma.unsqueeze(-1)
        return (z - x0) / sigma.clamp_min(1e-8)
