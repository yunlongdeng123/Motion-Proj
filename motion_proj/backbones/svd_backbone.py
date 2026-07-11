"""实现 ``DiffusionBackbone`` 的 Stable Video Diffusion（img2vid）适配器。

我们将 SVD 封装为 EDM 风格的骨干网络，其 ``predict_x0`` 返回经过预条件化的
干净 latent。预条件化遵循 diffusers 的 ``EulerDiscreteScheduler`` 所使用的
SVD/EDM v-prediction 关系：

    c_in    = 1 / sqrt(sigma^2 + 1)
    c_noise = 0.25 * log(sigma)                 # unet 的“timestep”输入
    model   = unet(cat([c_in*z, img_latents], 2), c_noise, img_embed, add_time)
    x0_hat  = z / (sigma^2 + 1) - model * sigma / sqrt(sigma^2 + 1)   # v-pred

LoRA 被注入到 UNet 的注意力投影中；*anchor*（锚定）预测复用同一个 UNet，
只是禁用适配器（不在显存中保留第二份副本）。

注意（有权重时需在运行时验证）：SVD 的确切条件（未缩放的 VAE 条件 latent、
噪声增强、fps/motion-bucket 时间 id）可能需要做小幅调整。这里的数学/结构
对训练而言是正确的；条件包的构建方式对齐了 ``StableVideoDiffusionPipeline``。
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ..utils.logging import get_logger
from .base import BackboneCapabilities, Conditioning, DiffusionBackbone

log = get_logger(__name__)


def _expand(sigma: torch.Tensor, ndim: int) -> torch.Tensor:
    while sigma.dim() < ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class SVDBackbone(DiffusionBackbone):
    parameterization = "edm"

    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            image_to_video=True,
            generation=True,
            future_ego_control=False,
            layout_control=False,
            multi_camera_sync=False,
            parameterizations=("edm",),
            metadata={"family": "SVD", "research_role": "development_backbone"},
        )

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = "cuda"
        self.dtype = torch.bfloat16
        # EDM 训练噪声的对数正态分布（供 L_real 使用）；采用类 SVD 的默认值
        self.p_mean = 0.7
        self.p_std = 1.6
        self.sigma_data = 0.5
        # 条件默认值（对齐 SVD pipeline）
        self.fps = 7
        self.motion_bucket_id = 127
        self.noise_aug_strength = 0.02
        self._loaded = False
        self.unet = self.vae = self.image_encoder = self.feature_extractor = self.scheduler = None

    # --------------------------------------------------------------- 构建
    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from diffusers import (
            AutoencoderKLTemporalDecoder,
            EulerDiscreteScheduler,
            UNetSpatioTemporalConditionModel,
        )
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.device, self.dtype = device, dtype
        repo = self.cfg.pretrained
        log.info("Loading SVD components from %s", repo)
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(repo, subfolder="vae").to(device, dtype)
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(repo, subfolder="unet").to(device, dtype)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            repo, subfolder="image_encoder"
        ).to(device, dtype)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(repo, subfolder="feature_extractor")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        if self.cfg.get("freeze_vae", True):
            self.vae.eval()
        if self.cfg.get("gradient_checkpointing", True):
            self.unet.enable_gradient_checkpointing()
        if self.cfg.get("enable_xformers", True):
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:  # pragma: no cover
                log.warning("xformers attention not enabled: %s", e)

        if self.cfg.lora.enable:
            self._inject_lora()
        self._loaded = True
        return self

    def _inject_lora(self):
        from peft import LoraConfig

        lcfg = self.cfg.lora
        lora = LoraConfig(
            r=int(lcfg.rank),
            lora_alpha=int(lcfg.alpha),
            lora_dropout=float(lcfg.get("dropout", 0.0)),
            target_modules=list(lcfg.target_modules),
            init_lora_weights="gaussian",
        )
        if hasattr(self.unet, "add_adapter"):
            self.unet.add_adapter(lora)
            backend = "diffusers.add_adapter"
        else:
            from peft import inject_adapter_in_model

            self.unet = inject_adapter_in_model(lora, self.unet, adapter_name="default")
            backend = "peft.inject_adapter_in_model"
        # 将 LoRA 参数保持为 fp32，以便在 bf16 autocast 下稳定优化
        for _, p in self.unet.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        n = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        log.info("Injected LoRA via %s (rank=%d), trainable params: %.2fM", backend, lcfg.rank, n / 1e6)

    def _set_lora_enabled(self, enabled: bool) -> None:
        """兼容 diffusers adapter mixin 与 PEFT 原地注入两种 LoRA 后端。"""
        if not self.cfg.lora.enable or self.unet is None:
            return

        method_name = "enable_adapters" if enabled else "disable_adapters"
        model_method = getattr(self.unet, method_name, None)
        if callable(model_method):
            try:
                model_method()
                return
            except TypeError:
                if method_name == "enable_adapters":
                    model_method(enabled)
                    return

        toggled = 0
        for module in self.unet.modules():
            if module is self.unet:
                continue
            enable_method = getattr(module, "enable_adapters", None)
            if callable(enable_method):
                try:
                    enable_method(enabled)
                    toggled += 1
                    continue
                except TypeError:
                    pass
            module_method = getattr(module, method_name, None)
            if callable(module_method):
                module_method()
                toggled += 1

        if toggled == 0:
            log.warning("No LoRA adapter toggle method found; anchor prediction will include adapters")

    # --------------------------------------------------------------- latent 空间
    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:]).to(self.device, self.dtype)
        lat = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        return lat.reshape(b, t, *lat.shape[1:])

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        b, t = latents.shape[:2]
        z = latents.reshape(b * t, *latents.shape[2:]).to(self.device, self.dtype)
        z = z / self.vae.config.scaling_factor
        # 时序解码器的卷积需要 num_frames；逐帧解码没有问题
        imgs = self.vae.decode(z, num_frames=1).sample
        return imgs.reshape(b, t, *imgs.shape[1:]).clamp(-1, 1)

    # ----------------------------------------------------------------- 噪声模型
    def sample_sigmas(self, num: int, device) -> torch.Tensor:
        """升序的推理 sigma 调度表（用于定义低噪声管道）。"""
        self.scheduler.set_timesteps(num, device=device)
        sigmas = self.scheduler.sigmas[:-1].to(device)  # 丢弃末尾的 0
        return torch.sort(sigmas).values

    def sample_training_sigma(self, n: int, device) -> torch.Tensor:
        """标准去噪损失 L_real 所用的对数正态 EDM 噪声采样。"""
        rnd = torch.randn(n, device=device)
        return (rnd * self.p_std + self.p_mean).exp()

    # ------------------------------------------------------------------- 去噪
    def _unet_x0(self, z, sigma, cond: Conditioning) -> torch.Tensor:
        sigma_e = _expand(sigma, z.dim())
        c_in = 1.0 / (sigma_e**2 + 1.0).sqrt()
        c_noise = 0.25 * torch.log(sigma.clamp_min(1e-8))
        img_lat = cond.data["image_latents"]            # [B,T,C,h,w]
        model_in = torch.cat([z * c_in, img_lat], dim=2).to(self.dtype)
        out = self.unet(
            model_in,
            c_noise.to(self.dtype),
            encoder_hidden_states=cond.data["image_embeds"].to(self.dtype),
            added_time_ids=cond.data["added_time_ids"].to(self.dtype),
            return_dict=False,
        )[0]
        # v-prediction 转 x0
        denom = (sigma_e**2 + 1.0)
        x0 = z / denom - out * (sigma_e / denom.sqrt())
        return x0

    def predict_x0(self, z, sigma, cond: Conditioning) -> torch.Tensor:
        return self._unet_x0(z, sigma, cond)

    @torch.no_grad()
    def anchor_predict_x0(self, z, sigma, cond: Conditioning) -> torch.Tensor:
        if self.cfg.lora.enable:
            self._set_lora_enabled(False)
        try:
            return self._unet_x0(z, sigma, cond)
        finally:
            if self.cfg.lora.enable:
                self._set_lora_enabled(True)

    # ----------------------------------------------------------------- 条件
    @torch.no_grad()
    def build_conditioning(self, batch: dict) -> Conditioning:
        cond_frame = batch["cond_frame"].to(self.device, self.dtype)     # [B,3,H,W]，取值 [-1,1]
        b = cond_frame.shape[0]
        t = int(self.cfg.num_frames)

        # CLIP 图像嵌入
        # CLIPImageProcessor 会转 numpy；bf16 tensor 不能直接转 numpy。
        pixel = (batch["cond_frame"].detach().cpu().float() + 1.0) / 2.0
        proc = self.feature_extractor(
            images=pixel, do_rescale=False, return_tensors="pt"
        ).pixel_values.to(self.device, self.dtype)
        image_embeds = self.image_encoder(proc).image_embeds.unsqueeze(1)  # [B,1,D]

        # VAE 条件 latent（对齐 SVD：未缩放），在帧维度上复制
        img_lat = self.vae.encode(cond_frame).latent_dist.mode()           # [B,C,h,w]
        img_lat = img_lat.unsqueeze(1).repeat(1, t, 1, 1, 1)               # [B,T,C,h,w]

        added_time_ids = torch.tensor(
            [self.fps, self.motion_bucket_id, self.noise_aug_strength],
            device=self.device,
        ).repeat(b, 1)

        return Conditioning(
            data={
                "image_embeds": image_embeds,
                "image_latents": img_lat,
                "added_time_ids": added_time_ids,
            }
        )

    # ----------------------------------------------------------------- 可训练部分
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def set_train_mode(self, enabled: bool = True) -> None:
        self.unet.train(enabled)

    def training_module(self) -> torch.nn.Module:
        return self.unet

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu() for n, p in self.unet.named_parameters() if p.requires_grad}

    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        self.unet.load_state_dict(state, strict=False)

    def save_adapter(self, path: str) -> None:
        import os

        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(path), exist_ok=True)
        sd = self.adapter_state()
        save_file(sd, path)
        log.info("Saved adapter (%d tensors) -> %s", len(sd), path)

    def load_adapter(self, path: str) -> None:
        from safetensors.torch import load_file

        sd = load_file(path)
        missing, unexpected = self.unet.load_state_dict(sd, strict=False)
        log.info("Loaded adapter from %s (unexpected=%d)", path, len(unexpected))

    # ----------------------------------------------------------------- 生成
    @torch.no_grad()
    def generate(self, cond_frame: torch.Tensor, num_frames: int | None = None, **kw) -> torch.Tensor:
        """从一个条件帧采样出未来片段（供 replay mining 使用）。

        ``cond_frame``：``[3,H,W]``，取值 [-1,1]。返回 ``[K,3,H,W]``，取值 [-1,1]。
        使用（LoRA 增强后的）各组件构建一个 ``StableVideoDiffusionPipeline``。
        """
        from diffusers import StableVideoDiffusionPipeline
        from PIL import Image

        if getattr(self, "_pipe", None) is None:
            self._pipe = StableVideoDiffusionPipeline(
                vae=self.vae,
                image_encoder=self.image_encoder,
                unet=self.unet,
                scheduler=self.scheduler,
                feature_extractor=self.feature_extractor,
            )
        k = int(num_frames or self.cfg.num_frames)
        img = ((cond_frame + 1) / 2).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        # 组件以 bf16 加载，但 SVD pipeline 只对 fp16 VAE 做 force_upcast，会把 float32
        # 预处理后的图像直接喂给 bf16 VAE，导致 conv dtype 不匹配；用 autocast 统一精度。
        device_type = "cuda" if (torch.cuda.is_available() and str(self.device).startswith("cuda")) else "cpu"
        decode_chunk_size = int(kw.pop("decode_chunk_size", 4))
        with torch.autocast(device_type=device_type, dtype=self.dtype):
            out = self._pipe(Image.fromarray(img), num_frames=k, decode_chunk_size=decode_chunk_size, **kw)
        frames = out.frames[0]  # PIL 列表
        import numpy as np

        arr = np.stack([np.asarray(f) for f in frames], 0)  # [K,H,W,3]
        t = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
        return (t * 2 - 1).to(self.device)


def build_svd_backbone(cfg: Any, load: bool = True, device="cuda", dtype=torch.bfloat16) -> SVDBackbone:
    bk = SVDBackbone(cfg)
    if load:
        bk.load(device=device, dtype=dtype)
    return bk
