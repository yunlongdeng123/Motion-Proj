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
import re
from typing import Any

import torch
import torch.nn.functional as F

from ..utils.logging import get_logger
from .base import BackboneCapabilities, Conditioning, DiffusionBackbone

log = get_logger(__name__)

LORA_SCOPES = {"temporal_only", "spatial_only", "all_attention"}
DEFAULT_LORA_PROJECTIONS = ("to_q", "to_k", "to_v", "to_out.0")
SVD_GENERATION_PROTOCOLS = {"svd_legacy_unversioned", "svd_official_v1"}


def resolve_svd_generation_settings(cfg: Any) -> dict[str, Any]:
    """解析并校验 generation 协议；旧配置保持未版本化语义。"""
    generation = cfg.get("generation", {}) or {}
    protocol = str(generation.get("protocol", "svd_legacy_unversioned"))
    if protocol not in SVD_GENERATION_PROTOCOLS:
        raise ValueError(
            f"未知 SVD generation.protocol={protocol!r}; "
            f"allowed={sorted(SVD_GENERATION_PROTOCOLS)}"
        )
    settings = {
        "protocol": protocol,
        "fps": int(generation.get("fps", 7)),
        "motion_bucket_id": int(generation.get("motion_bucket_id", 127)),
        "noise_aug_strength": float(generation.get("noise_aug_strength", 0.02)),
        "min_guidance_scale": float(generation.get("min_guidance_scale", 1.0)),
        "max_guidance_scale": float(generation.get("max_guidance_scale", 3.0)),
    }
    if settings["fps"] <= 0:
        raise ValueError("model.generation.fps 必须大于 0")
    if settings["motion_bucket_id"] < 0:
        raise ValueError("model.generation.motion_bucket_id 必须非负")
    if not math.isfinite(settings["noise_aug_strength"]) or settings["noise_aug_strength"] < 0:
        raise ValueError("model.generation.noise_aug_strength 必须是有限非负数")
    if not all(math.isfinite(settings[key]) for key in ("min_guidance_scale", "max_guidance_scale")):
        raise ValueError("model.generation guidance scale 必须有限")
    if settings["min_guidance_scale"] <= 0 or settings["max_guidance_scale"] < settings["min_guidance_scale"]:
        raise ValueError("model.generation guidance scale 范围无效")
    return settings


def _expand(sigma: torch.Tensor, ndim: int) -> torch.Tensor:
    while sigma.dim() < ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def classify_attention_module_path(module_path: str) -> str | None:
    """依据完整模块路径区分 SVD 空间与时间 attention。"""
    segments = module_path.split(".")
    if "temporal_transformer_blocks" in segments:
        return "temporal"
    if "transformer_blocks" in segments:
        return "spatial"
    return None


def select_lora_module_names(
    unet: torch.nn.Module,
    scope: str,
    projections: tuple[str, ...] | list[str] = DEFAULT_LORA_PROJECTIONS,
) -> tuple[list[str], dict[str, str]]:
    """从注入前 UNet 中选择完整 Linear 路径，并拒绝无法分类的 attention。"""
    if scope not in LORA_SCOPES:
        raise ValueError(f"未知 LoRA scope={scope!r}; allowed={sorted(LORA_SCOPES)}")
    suffixes = tuple(str(value) for value in projections)
    candidates: dict[str, str] = {}
    ambiguous: list[str] = []
    for name, module in unet.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.endswith(suffixes):
            continue
        kind = classify_attention_module_path(name)
        if kind is None:
            ambiguous.append(name)
        else:
            candidates[name] = kind
    if ambiguous:
        raise RuntimeError(f"存在无法按完整路径分类的 attention 模块: {ambiguous}")

    selected = sorted(
        name
        for name, kind in candidates.items()
        if scope == "all_attention" or kind == scope.removesuffix("_only")
    )
    if not selected:
        raise RuntimeError(f"LoRA scope={scope} 未匹配到任何模块")
    return selected, {name: candidates[name] for name in selected}


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
        # 旧 checkpoint/config 没有 generation 字段时保留未版本化路径；新 run 必须显式声明。
        self._generation_settings = resolve_svd_generation_settings(cfg)
        self.fps = int(self._generation_settings["fps"])
        self.motion_bucket_id = int(self._generation_settings["motion_bucket_id"])
        self.noise_aug_strength = float(self._generation_settings["noise_aug_strength"])
        self.sigma_floor = float(cfg.get("sigma_floor", 1.0e-3))
        if not math.isfinite(self.sigma_floor) or self.sigma_floor <= 0:
            raise ValueError("model.sigma_floor 必须是有限正数")
        self._loaded = False
        self._lora_enabled = False
        self._selected_lora_module_names: tuple[str, ...] = ()
        self._selected_lora_module_kinds: dict[str, str] = {}
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
        scope = str(lcfg.get("scope", "all_attention"))
        projections = tuple(lcfg.get("projections", lcfg.get("target_modules", DEFAULT_LORA_PROJECTIONS)))
        selected, kinds = select_lora_module_names(self.unet, scope, projections)
        lora = LoraConfig(
            r=int(lcfg.rank),
            lora_alpha=int(lcfg.alpha),
            lora_dropout=float(lcfg.get("dropout", 0.0)),
            target_modules=selected,
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
        self._selected_lora_module_names = tuple(selected)
        self._selected_lora_module_kinds = kinds
        self._lora_enabled = True
        self._validate_lora_isolation()
        n = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        log.info(
            "Injected LoRA via %s (rank=%d, scope=%s), modules=%d, trainable params=%.2fM",
            backend,
            lcfg.rank,
            scope,
            len(selected),
            n / 1e6,
        )
        for name in selected:
            log.info("LoRA selected module [%s]: %s", kinds[name], name)

    @staticmethod
    def _lora_parameter_module(parameter_name: str) -> str | None:
        match = re.match(r"^(.*)\.lora_(?:A|B|embedding_A|embedding_B)\.", parameter_name)
        return match.group(1) if match else None

    def _validate_lora_isolation(self) -> None:
        trainable_names = [name for name, parameter in self.unet.named_parameters() if parameter.requires_grad]
        if not trainable_names:
            raise RuntimeError("LoRA 注入后没有可训练 tensor")
        non_lora = [name for name in trainable_names if self._lora_parameter_module(name) is None]
        if non_lora:
            raise RuntimeError(f"发现非 LoRA 可训练参数: {non_lora}")
        actual_modules = {
            module_name
            for name in trainable_names
            if (module_name := self._lora_parameter_module(name)) is not None
        }
        expected_modules = set(self._selected_lora_module_names)
        if actual_modules != expected_modules:
            raise RuntimeError(
                "LoRA 实际注入模块与完整路径清单不一致: "
                f"missing={sorted(expected_modules - actual_modules)}, "
                f"unexpected={sorted(actual_modules - expected_modules)}"
            )
        scope = str(self.cfg.lora.get("scope", "all_attention"))
        actual_kinds = {classify_attention_module_path(name) for name in actual_modules}
        if None in actual_kinds:
            raise RuntimeError("LoRA 实际注入到了无法分类的模块")
        if scope == "temporal_only" and actual_kinds != {"temporal"}:
            raise RuntimeError(f"temporal_only 混入非时间模块: {sorted(actual_modules)}")
        if scope == "spatial_only" and actual_kinds != {"spatial"}:
            raise RuntimeError(f"spatial_only 混入非空间模块: {sorted(actual_modules)}")

    def _set_lora_enabled(self, enabled: bool) -> None:
        """兼容 diffusers adapter mixin 与 PEFT 原地注入两种 LoRA 后端。"""
        if not self.cfg.lora.enable or self.unet is None:
            return

        if self._lora_enabled == enabled:
            return

        method_name = "enable_adapters" if enabled else "disable_adapters"
        model_method = getattr(self.unet, method_name, None)
        if callable(model_method):
            try:
                model_method()
                self._lora_enabled = enabled
                return
            except TypeError:
                if method_name == "enable_adapters":
                    model_method(enabled)
                    self._lora_enabled = enabled
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
            raise RuntimeError("未找到 LoRA adapter 开关；拒绝生成泄漏 student adapter 的 Base anchor")
        self._lora_enabled = enabled

    # --------------------------------------------------------------- 生成协议
    def generation_settings(self) -> dict[str, Any]:
        """返回新生成/run manifest 必须保存的显式 SVD 协议参数。"""
        return dict(self._generation_settings)

    def generation_protocol_metadata(self) -> dict[str, Any]:
        """导出协议与 scheduler 指纹；不重新标记既有 cache。"""
        import hashlib
        import json

        try:
            import diffusers

            diffusers_version = str(diffusers.__version__)
        except Exception:  # pragma: no cover - 仅在依赖异常时降级
            diffusers_version = "unknown"
        scheduler_config = dict(self.scheduler.config) if self.scheduler is not None else {}
        scheduler_raw = json.dumps(
            scheduler_config, sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")
        settings = self.generation_settings()
        return {
            "protocol": settings["protocol"],
            "diffusers_version": diffusers_version,
            "scheduler_config_fingerprint": hashlib.sha256(scheduler_raw).hexdigest(),
            "fps_input": settings["fps"],
            "fps_time_id": settings["fps"] - 1,
            "motion_bucket_id": settings["motion_bucket_id"],
            "noise_aug_strength": settings["noise_aug_strength"],
            "min_guidance_scale": settings["min_guidance_scale"],
            "max_guidance_scale": settings["max_guidance_scale"],
        }

    def _generation_pipeline(self):
        """延迟构造与官方 `StableVideoDiffusionPipeline` 同构的采样器。"""
        from diffusers import StableVideoDiffusionPipeline

        if getattr(self, "_pipe", None) is None:
            self._pipe = StableVideoDiffusionPipeline(
                vae=self.vae,
                image_encoder=self.image_encoder,
                unet=self.unet,
                scheduler=self.scheduler,
                feature_extractor=self.feature_extractor,
            )
        return self._pipe

    @staticmethod
    def _pipeline_image(cond_frame: torch.Tensor):
        """保持历史 wrapper 的 uint8/PIL 输入语义，以便精确审计旧 Base rollout。"""
        from PIL import Image

        if cond_frame.ndim != 3 or cond_frame.shape[0] != 3:
            raise ValueError("cond_frame 必须是 [3,H,W]")
        image = (
            ((cond_frame.detach() + 1.0) / 2.0)
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return Image.fromarray(image)

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
    def _safe_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(sigma).all()):
            raise ValueError("sigma 包含 NaN/Inf")
        return sigma.clamp_min(self.sigma_floor)

    def predict_model_output(self, z, sigma, cond: Conditioning) -> torch.Tensor:
        sigma_safe = self._safe_sigma(sigma)
        sigma_e = _expand(sigma_safe, z.dim())
        c_in = 1.0 / (sigma_e**2 + 1.0).sqrt()
        c_noise = 0.25 * torch.log(sigma_safe)
        img_lat = cond.data["image_latents"]            # [B,T,C,h,w]
        model_in = torch.cat([z * c_in, img_lat], dim=2).to(self.dtype)
        return self.unet(
            model_in,
            c_noise.to(self.dtype),
            encoder_hidden_states=cond.data["image_embeds"].to(self.dtype),
            added_time_ids=cond.data["added_time_ids"].to(self.dtype),
            return_dict=False,
        )[0]

    def x0_from_model_output(self, z, sigma, model_output) -> torch.Tensor:
        sigma_e = _expand(self._safe_sigma(sigma), z.dim())
        denom = sigma_e.square() + 1.0
        return z / denom - model_output * (sigma_e / denom.sqrt())

    def model_output_from_x0(self, z, sigma, x0) -> torch.Tensor:
        sigma_e = _expand(self._safe_sigma(sigma), z.dim())
        denom = sigma_e.square() + 1.0
        return (z / denom - x0) * (denom.sqrt() / sigma_e)

    @torch.no_grad()
    def anchor_predict_model_output(self, z, sigma, cond: Conditioning) -> torch.Tensor:
        was_enabled = self._lora_enabled
        if self.cfg.lora.enable and was_enabled:
            self._set_lora_enabled(False)
        try:
            return self.predict_model_output(z, sigma, cond)
        finally:
            if self.cfg.lora.enable and was_enabled:
                self._set_lora_enabled(True)

    # ----------------------------------------------------------------- 条件
    @torch.no_grad()
    def build_conditioning(self, batch: dict) -> Conditioning:
        """构造历史单步训练条件；不声称与完整 CFG rollout 等价。"""
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

    @torch.no_grad()
    def build_official_generation_conditioning(
        self,
        cond_frame: torch.Tensor,
        *,
        generator: torch.Generator,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> dict[str, Any]:
        """复现官方 SVD 的 image/CFG/time-ID 条件构造，供 parity 与新协议使用。

        此接口保留 condition-noise，调用者必须在同一个 generator 状态上继续创建
        initial video latent，才能和完整 pipeline 的随机序列逐位对齐。
        """
        if self._generation_settings["protocol"] != "svd_official_v1":
            raise RuntimeError("官方条件构造要求 model.generation.protocol=svd_official_v1")
        from diffusers.utils.torch_utils import randn_tensor

        pipe = self._generation_pipeline()
        settings = self.generation_settings()
        image = self._pipeline_image(cond_frame)
        num_frames = int(num_frames or self.cfg.num_frames)
        height = int(height or cond_frame.shape[-2])
        width = int(width or cond_frame.shape[-1])
        do_cfg = settings["max_guidance_scale"] > 1.0
        device = torch.device(self.device)

        image_embeds = pipe._encode_image(
            image, device, num_videos_per_prompt=1, do_classifier_free_guidance=do_cfg,
        )
        preprocessed = pipe.video_processor.preprocess(image, height=height, width=width).to(device)
        condition_noise = randn_tensor(
            preprocessed.shape, generator=generator, device=device, dtype=preprocessed.dtype,
        )
        noisy_image = preprocessed + settings["noise_aug_strength"] * condition_noise

        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float32)
        image_latents = pipe._encode_vae_image(
            noisy_image,
            device=device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        ).to(image_embeds.dtype)
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        repeated_image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        fps_time_id = settings["fps"] - 1
        added_time_ids = pipe._get_add_time_ids(
            fps_time_id,
            settings["motion_bucket_id"],
            settings["noise_aug_strength"],
            image_embeds.dtype,
            1,
            1,
            do_cfg,
        ).to(device)
        return {
            "image": image,
            "preprocessed_image": preprocessed,
            "condition_noise": condition_noise,
            "noisy_condition_image": noisy_image,
            "image_embeds": image_embeds,
            "image_latents": repeated_image_latents,
            "added_time_ids": added_time_ids,
            "fps_input": settings["fps"],
            "fps_time_id": fps_time_id,
            "num_frames": num_frames,
            "do_classifier_free_guidance": do_cfg,
        }

    # ----------------------------------------------------------------- 可训练部分
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def set_train_mode(self, enabled: bool = True) -> None:
        self.unet.train(enabled)

    def training_module(self) -> torch.nn.Module:
        return self.unet

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu() for n, p in self.unet.named_parameters() if p.requires_grad}

    def adapter_metadata(self) -> dict[str, Any]:
        if not self.cfg.lora.enable:
            return super().adapter_metadata()
        self._validate_lora_isolation()
        state = self.adapter_state()
        trainable = self.trainable_parameters()
        temporal_count = sum(
            self._selected_lora_module_kinds[name] == "temporal"
            for name in self._selected_lora_module_names
        )
        spatial_count = sum(
            self._selected_lora_module_kinds[name] == "spatial"
            for name in self._selected_lora_module_names
        )
        return {
            "scope": str(self.cfg.lora.get("scope", "all_attention")),
            "rank": int(self.cfg.lora.rank),
            "selected_module_names": list(self._selected_lora_module_names),
            "selected_module_count": len(self._selected_lora_module_names),
            "temporal_module_count": temporal_count,
            "spatial_module_count": spatial_count,
            "trainable_tensor_count": len(trainable),
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "adapter_tensor_count": len(state),
        }

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
        settings = self.generation_settings()
        protocol = str(kw.pop("protocol", settings["protocol"]))
        if protocol not in SVD_GENERATION_PROTOCOLS:
            raise ValueError(f"未知 generation protocol: {protocol}")
        if protocol != settings["protocol"]:
            raise ValueError(
                "调用参数 protocol 不得覆盖 model.generation.protocol；"
                "请使用独立配置版本化新的 generation 语义"
            )
        for key in (
            "fps", "motion_bucket_id", "noise_aug_strength",
            "min_guidance_scale", "max_guidance_scale",
        ):
            kw.setdefault(key, settings[key])
        pipe = self._generation_pipeline()
        k = int(num_frames or self.cfg.num_frames)
        image = self._pipeline_image(cond_frame)
        # 组件以 bf16 加载，但 SVD pipeline 只对 fp16 VAE 做 force_upcast，会把 float32
        # 预处理后的图像直接喂给 bf16 VAE，导致 conv dtype 不匹配；用 autocast 统一精度。
        device_type = "cuda" if (torch.cuda.is_available() and str(self.device).startswith("cuda")) else "cpu"
        decode_chunk_size = int(kw.pop("decode_chunk_size", 4))
        with torch.autocast(device_type=device_type, dtype=self.dtype):
            out = pipe(image, num_frames=k, decode_chunk_size=decode_chunk_size, **kw)
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
