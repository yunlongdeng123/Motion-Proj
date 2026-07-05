"""Motion-Proj 使用的抽象视频扩散骨干网络接口。

训练代码与具体骨干网络无关：它只依赖这个接口。一个骨干网络必须能够
（1）在 RGB 片段与 latent 空间之间相互映射，（2）在给定 sigma 下加噪，
（3）从带噪 latent 产生*干净*预测 ``x0_hat``（模型预条件化在内部处理），
以及（4）从冻结的基座模型产生*anchor*（锚定）预测（禁用 LoRA/适配器）。

Latent 张量采用 ``[B, T, C, h, w]`` 的布局。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Conditioning:
    """骨干网络专属的条件包（对训练器保持不透明）。"""

    data: dict[str, Any]


class DiffusionBackbone(abc.ABC):
    parameterization: str = "edm"  # "edm"（通过预条件化得到 x0）或 "eps"
    # EDM 训练噪声的对数正态默认值（子类可覆盖）
    p_mean: float = 0.7
    p_std: float = 1.6
    sigma_data: float = 0.5

    # --------------------------------------------------------------- latent 空间
    @abc.abstractmethod
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """RGB ``[B,T,3,H,W]``（取值 [-1,1]）-> latent ``[B,T,C,h,w]``。"""

    @abc.abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latent ``[B,T,C,h,w]`` -> RGB ``[B,T,3,H,W]``（取值 [-1,1]）。"""

    # ----------------------------------------------------------------- 噪声模型
    @abc.abstractmethod
    def sample_sigmas(self, num: int, device) -> torch.Tensor:
        """返回升序的 sigma 调度表（其长度控制管道子集）。"""

    def add_noise(self, x0: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """EDM 前向：``z = x0 + sigma * noise``（alpha = 1）。"""
        while sigma.dim() < x0.dim():
            sigma = sigma.unsqueeze(-1)
        return x0 + sigma * noise

    def sample_training_sigma(self, n: int, device) -> torch.Tensor:
        """标准去噪损失 L_real 所用的对数正态 EDM 噪声采样。"""
        rnd = torch.randn(n, device=device)
        return (rnd * self.p_std + self.p_mean).exp()

    # ------------------------------------------------------------------- 去噪
    @abc.abstractmethod
    def predict_x0(
        self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        """经过预条件化的干净预测 ``x0_hat``（可训练路径，LoRA 开启）。"""

    @abc.abstractmethod
    def anchor_predict_x0(
        self, z: torch.Tensor, sigma: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        """来自冻结基座模型的干净预测（禁用 LoRA/适配器）。"""

    # ----------------------------------------------------------------- 条件
    @abc.abstractmethod
    def build_conditioning(self, batch: dict) -> Conditioning:
        """从 dataloader 批次构建条件包（不计算梯度）。"""

    # ----------------------------------------------------------------- 可训练部分
    @abc.abstractmethod
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        ...

    @abc.abstractmethod
    def save_adapter(self, path: str) -> None:
        ...

    @abc.abstractmethod
    def load_adapter(self, path: str) -> None:
        ...

    # --------------------------------------------------------------------- 辅助函数
    def eps_from_x0(self, z: torch.Tensor, sigma: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """在 EDM 下将 x0 目标转换为 eps 目标（``eps = (z - x0)/sigma``）。"""
        while sigma.dim() < z.dim():
            sigma = sigma.unsqueeze(-1)
        return (z - x0) / sigma.clamp_min(1e-8)
