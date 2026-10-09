"""Seen-to-Scene 官方组件的最小桥接；target 只进入监督分支。"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F


DEFAULT_EXTERNAL = Path("/root/autodl-tmp/external/seen_to_scene_v81")
DEFAULT_SVD = Path("/root/autodl-tmp/models/worldsim_v81/svd_xt_1_1")
DEFAULT_RAFT = Path("/root/autodl-tmp/models/worldsim_v77_poc_propainter/raft-things.pth")
DEFAULT_FCNET = Path("/root/autodl-tmp/models/worldsim_v77_poc_propainter/recurrent_flow_completion.pth")


@dataclass
class Components:
    vae: torch.nn.Module
    image_encoder: torch.nn.Module
    feature_extractor: object
    raft: torch.nn.Module
    fcnet: torch.nn.Module
    propagator: torch.nn.Module
    unet: torch.nn.Module
    scheduler: object
    flow_loss: torch.nn.Module


def official_modules(external: Path = DEFAULT_EXTERNAL):
    """使用固定修订的官方 checkout；不将第三方网络伪装成项目内实现。"""
    external = external.resolve()
    required = ("models/unet.py", "models/latent_warping.py", "models/recurrent_flow_completion.py",
                "models/bidirectional_flow_raft.py", "utils/loss.py")
    missing = [name for name in required if not (external / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Seen-to-Scene 官方源码不完整: {external}: {missing}")
    if str(external) not in sys.path:
        sys.path.insert(0, str(external))
    return tuple(importlib.import_module(name) for name in (
        "models.unet", "models.latent_warping", "models.recurrent_flow_completion",
        "models.bidirectional_flow_raft", "utils.loss"))


def configure_trainable(components: Components) -> dict[str, int]:
    """论文 P0：FCNet、传播/细化和 SVD temporal transformer 可训练。"""
    for module in (components.vae, components.image_encoder, components.raft, components.unet):
        module.requires_grad_(False)
        module.eval()
    components.fcnet.requires_grad_(True)
    components.propagator.requires_grad_(True)
    temporal = 0
    for name, parameter in components.unet.named_parameters():
        if "temporal_transformer_block" in name:
            parameter.requires_grad_(True)
            temporal += parameter.numel()
    if temporal == 0:
        raise RuntimeError("SVD UNet 没有 temporal_transformer_block；不能静默退化为无 SVD 训练")
    return {"svd_temporal": temporal,
            "fcnet": sum(p.numel() for p in components.fcnet.parameters()),
            "propagation_refinement": sum(p.numel() for p in components.propagator.parameters())}


def load_components(svd: Path = DEFAULT_SVD, raft_weight: Path = DEFAULT_RAFT,
                    fcnet_weight: Path = DEFAULT_FCNET, external: Path = DEFAULT_EXTERNAL,
                    device: str = "cuda") -> Components:
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    for path in (svd / "unet" / "config.json", svd / "vae" / "config.json",
                 svd / "image_encoder" / "config.json", svd / "scheduler" / "scheduler_config.json",
                 raft_weight, fcnet_weight):
        if not path.exists():
            raise FileNotFoundError(f"模型依赖缺失: {path}")
    unet_mod, prop_mod, fc_mod, raft_mod, loss_mod = official_modules(external)
    components = Components(
        vae=AutoencoderKLTemporalDecoder.from_pretrained(
            svd, subfolder="vae", variant="fp16", torch_dtype=torch.float32),
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            svd, subfolder="image_encoder", variant="fp16", torch_dtype=torch.float32),
        feature_extractor=CLIPImageProcessor.from_pretrained(svd, subfolder="feature_extractor"),
        raft=raft_mod.RAFT_bi(str(raft_weight)),
        fcnet=fc_mod.RecurrentFlowCompleteNet(str(fcnet_weight)),
        propagator=prop_mod.LatentPropagation(4),
        unet=unet_mod.UNetSpatioTemporalConditionModel.from_pretrained(
            svd, subfolder="unet", variant="fp16", torch_dtype=torch.float32),
        scheduler=EulerDiscreteScheduler.from_pretrained(svd, subfolder="scheduler"),
        flow_loss=loss_mod.FlowLoss(),
    )
    for module in (components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet):
        module.to(device)
    configure_trainable(components)
    return components


def validate_batch(batch: dict, require_target: bool = True) -> None:
    visible, mask = batch["visible_rgb"], batch["hole_mask"]
    if visible.ndim != 5 or visible.shape[2] != 3 or visible.shape[1] != 25 or visible.shape[-2:] != (256, 256):
        raise ValueError("visible_rgb 必须是 [B,25,3,256,256]")
    if mask.shape != (visible.shape[0], 25, 1, 256, 256):
        raise ValueError("hole_mask 形状错误")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("hole_mask 必须为 0/1，1 是洞区")
    if torch.any(visible.masked_select(mask.expand_as(visible).bool()) != 0):
        raise ValueError("可见输入含有洞区值")
    if require_target and batch["target_rgb"].shape != visible.shape:
        raise ValueError("target_rgb 形状错误")


@torch.no_grad()
def encode_video(vae: torch.nn.Module, video: torch.Tensor, *, scaled: bool,
                 sample: bool = False, chunk: int = 4) -> torch.Tensor:
    batch, frames = video.shape[:2]
    flat = video.flatten(0, 1)
    distributions = [vae.encode(flat[i:i + chunk]).latent_dist for i in range(0, len(flat), chunk)]
    parts = [dist.sample() if sample else dist.mode() for dist in distributions]
    latent = torch.cat(parts).reshape(batch, frames, *parts[0].shape[1:])
    return latent * vae.config.scaling_factor if scaled else latent


@torch.no_grad()
def image_embedding(components: Components, visible: torch.Tensor) -> torch.Tensor:
    image = F.interpolate(visible[:, 0].float(), (224, 224), mode="bicubic", align_corners=False)
    image = (image.clamp(-1, 1) + 1) / 2
    pixels = components.feature_extractor(images=image.cpu(), do_normalize=True,
                                          do_center_crop=False, do_resize=False,
                                          do_rescale=False, return_tensors="pt").pixel_values
    pixels = pixels.to(device=visible.device, dtype=next(components.image_encoder.parameters()).dtype)
    return components.image_encoder(pixels).image_embeds.unsqueeze(1)


@torch.no_grad()
def raft_flows(raft: torch.nn.Module, video: torch.Tensor, *, iters: int = 20,
               pair_chunk: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """小批次运行官方 RAFT，避免 24 帧双向相关体一次占满显存。"""
    batch, frames, channels, height, width = video.shape
    left = video[:, :-1].reshape(-1, channels, height, width).float()
    right = video[:, 1:].reshape(-1, channels, height, width).float()
    forward, backward = [], []
    for i in range(0, len(left), pair_chunk):
        a, b = left[i:i + pair_chunk], right[i:i + pair_chunk]
        forward.append(raft.fix_raft(a, b, iters=iters, test_mode=True)[1])
        backward.append(raft.fix_raft(b, a, iters=iters, test_mode=True)[1])
    return (torch.cat(forward).reshape(batch, frames - 1, 2, height, width),
            torch.cat(backward).reshape(batch, frames - 1, 2, height, width))


def completed_flows(components: Components, visible: torch.Tensor,
                    mask: torch.Tensor, *, raft_iters: int = 20, pair_chunk: int = 2):
    # 推理输入只能读取已经遮蔽的 RGB。完整视频光流由训练损失另行计算。
    input_flows = raft_flows(components.raft, visible, iters=raft_iters, pair_chunk=pair_chunk)
    predicted, _ = components.fcnet.forward_bidirect_flow(input_flows, mask)
    return components.fcnet.combine_flow(input_flows, predicted, mask)


def conditioning(components: Components, visible: torch.Tensor, mask: torch.Tensor,
                 flows: tuple[torch.Tensor, torch.Tensor], noise_strength: float | torch.Tensor = 0.02,
                 *, sample: bool = False):
    if float(noise_strength) > 0:
        visible = visible + torch.randn_like(visible) * noise_strength
    latent = encode_video(components.vae, visible, scaled=False, sample=sample)
    # 官方 LatentPropagation 的真实签名只有 cond_lat/flow_fw/flow_bw/masks。
    return components.propagator(latent, flows[0], flows[1], mask)[2]


def time_ids(unet: torch.nn.Module, batch: int, device: torch.device,
             dtype: torch.dtype, noise_strength: float | torch.Tensor) -> torch.Tensor:
    values = torch.tensor([6, 127, float(noise_strength)], device=device, dtype=dtype)
    if unet.config.addition_time_embed_dim * 3 != unet.add_embedding.linear_1.in_features:
        raise ValueError("SVD added time embedding 配置不匹配")
    return values.repeat(batch, 1)


def trainable_state(components: Components) -> dict:
    return {"unet_temporal": {name: p.detach().cpu() for name, p in components.unet.named_parameters()
                              if p.requires_grad},
            "fcnet": components.fcnet.state_dict(),
            "propagator": components.propagator.state_dict()}


def load_trainable_state(components: Components, state: dict) -> None:
    temporal = dict(components.unet.named_parameters())
    expected = {name for name, p in temporal.items() if p.requires_grad}
    if set(state["unet_temporal"]) != expected:
        raise ValueError("checkpoint 的 SVD temporal 参数与当前模型不一致")
    with torch.no_grad():
        for name, value in state["unet_temporal"].items():
            temporal[name].copy_(value)
    components.fcnet.load_state_dict(state["fcnet"], strict=True)
    components.propagator.load_state_dict(state["propagator"], strict=True)
