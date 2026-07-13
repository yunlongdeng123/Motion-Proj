"""逐样本临时写入、校验后原子提交的 projection cache。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import torch

from ..runtime.atomic import atomic_directory, atomic_write_json
from ..utils.io import ensure_dir

CACHE_SCHEMA_VERSION = 5
COMPLETE = "COMPLETE"

_FORMAL_V2_REQUIRED_METADATA = (
    "sample_id", "source", "parent_kind", "base_model_fingerprint", "adapter_loaded",
    "condition_id", "condition_frame", "generation_seed", "generation_sampler",
    "generation_steps", "generation_settings", "first_frame_frozen", "auditor_version",
    "projector_version", "geometry_mode", "uses_future_gt_ego", "uses_future_gt_track",
    "energy_before_by_component", "energy_after_by_component", "projector_diagnostics",
    "base_vae_fingerprint", "projected_vae_fingerprint", "vae_fingerprint",
    "static_valid_fraction", "object_valid_fraction", "static_confidence_mean",
    "object_confidence_mean",
)


class ProjectionCacheWriter:
    def __init__(self, cache_dir: str, store: str = "latent", overwrite: bool = False,
                 fingerprint: str | None = None, *, formal_v2: bool = False):
        if store not in ("latent", "rgb"):
            raise ValueError("store 必须是 latent 或 rgb")
        self.cache_dir = ensure_dir(cache_dir)
        self.store = store
        self.overwrite = bool(overwrite)
        self.fingerprint = fingerprint
        self.formal_v2 = bool(formal_v2)

    def sample_dir(self, sample_id: str) -> str:
        return os.path.join(self.cache_dir, sample_id)

    def exists(self, sample_id: str) -> bool:
        directory = self.sample_dir(sample_id)
        try:
            with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            return False
        required = ["clean.pt", "y.pt", "x_dagger.pt", "mask.pt", "metadata.json", COMPLETE]
        if self.store == "latent":
            required.append("context.pt")
        return (
            all(os.path.isfile(os.path.join(directory, name)) for name in required)
            and meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION
            and (self.fingerprint is None or meta.get("cache_fingerprint") == self.fingerprint)
        )

    @staticmethod
    def _validate(
        clean: torch.Tensor,
        y: torch.Tensor,
        x_dagger: torch.Tensor,
        mask: torch.Tensor,
        context: dict | None,
        latent_flow: torch.Tensor | None,
        flow_confidence: torch.Tensor | None,
    ) -> None:
        if clean.shape != y.shape or y.shape != x_dagger.shape:
            raise ValueError("clean、y 与 x_dagger shape 不一致")
        if mask.shape[0] != y.shape[0] or mask.shape[-2:] != y.shape[-2:]:
            raise ValueError("mask 与目标的时间/空间 shape 不一致")
        for name, tensor in (
            ("clean", clean),
            ("y", y),
            ("x_dagger", x_dagger),
            ("mask", mask),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} 包含 NaN/Inf")
        if bool((mask < 0).any()) or bool((mask > 1).any()):
            raise ValueError("mask 超出 [0,1]")
        if context is not None and not all(bool(torch.isfinite(v).all()) for v in context.values()):
            raise ValueError("context 包含 NaN/Inf")
        if latent_flow is not None:
            if latent_flow.shape != (y.shape[0] - 1, y.shape[-2], y.shape[-1], 2):
                raise ValueError("latent_flow shape 必须为 [T-1,H,W,2]")
            if not bool(torch.isfinite(latent_flow).all()):
                raise ValueError("latent_flow 包含 NaN/Inf")
        if flow_confidence is not None:
            expected = (y.shape[0] - 1, 1, y.shape[-2], y.shape[-1])
            if flow_confidence.shape != expected:
                raise ValueError("flow_confidence shape 必须为 [T-1,1,H,W]")
            if not bool(torch.isfinite(flow_confidence).all()):
                raise ValueError("flow_confidence 包含 NaN/Inf")
            if bool((flow_confidence < 0).any()) or bool((flow_confidence > 1).any()):
                raise ValueError("flow_confidence 超出 [0,1]")

    @staticmethod
    def _validate_component_tensor(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
        if value.shape != (reference.shape[0], 1, reference.shape[-2], reference.shape[-1]):
            raise ValueError(f"{name} shape 必须为 [T,1,H,W]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} 包含 NaN/Inf")
        if bool((value < 0).any()) or bool((value > 1).any()):
            raise ValueError(f"{name} 超出 [0,1]")

    @classmethod
    def _validate_formal_v2(
        cls,
        sample_id: str,
        y: torch.Tensor,
        x_dagger: torch.Tensor,
        mask: torch.Tensor,
        metadata: dict,
        static_mask: torch.Tensor | None,
        object_mask: torch.Tensor | None,
        static_confidence: torch.Tensor | None,
        object_confidence: torch.Tensor | None,
        base_rgb: torch.Tensor | None,
        projected_rgb: torch.Tensor | None,
        base_latent: torch.Tensor | None,
        projected_latent: torch.Tensor | None,
        latent_residual: torch.Tensor | None,
        store: str,
    ) -> None:
        missing = [name for name in _FORMAL_V2_REQUIRED_METADATA if name not in metadata]
        if missing:
            raise ValueError(f"正式 V2 cache 缺少 metadata: {', '.join(missing)}")
        if metadata["sample_id"] != sample_id:
            raise ValueError("正式 V2 cache 的 sample_id 必须与目录一致")
        if metadata["source"] != "replay_v2":
            raise ValueError("正式 V2 cache 的 source 必须是 replay_v2")
        if metadata["parent_kind"] != "base" or bool(metadata["adapter_loaded"]):
            raise ValueError("正式 V2 cache 必须来自无 adapter 的 Base parent")
        if bool(metadata["uses_future_gt_ego"]) or bool(metadata["uses_future_gt_track"]):
            raise ValueError("正式 V2 cache 禁止 future GT ego/track")
        if not bool(metadata["first_frame_frozen"]):
            raise ValueError("正式 V2 cache 必须声明首帧冻结")
        if int(metadata["generation_steps"]) <= 0:
            raise ValueError("正式 V2 cache 的 generation_steps 必须为正")
        if not metadata["base_vae_fingerprint"] or (
            metadata["base_vae_fingerprint"] != metadata["projected_vae_fingerprint"]
        ):
            raise ValueError("正式 V2 cache 的 Base/projected VAE fingerprint 必须一致且非空")
        if metadata["vae_fingerprint"] != metadata["base_vae_fingerprint"]:
            raise ValueError("正式 V2 cache 的 VAE fingerprint 必须与 Base/projected 一致")
        components = (static_mask, object_mask, static_confidence, object_confidence)
        if any(value is None for value in components):
            raise ValueError("正式 V2 cache 必须保存 static/object mask 与 confidence")
        assert static_mask is not None and object_mask is not None
        assert static_confidence is not None and object_confidence is not None
        for name, value in (
            ("static_mask", static_mask), ("object_mask", object_mask),
            ("static_confidence", static_confidence), ("object_confidence", object_confidence),
        ):
            cls._validate_component_tensor(name, value, y)
            if bool(value[0].ne(0).any()):
                raise ValueError(f"正式 V2 cache 的 {name} 首帧必须为零")
        if bool(mask[0].ne(0).any()) or not torch.equal(y[0], x_dagger[0]):
            raise ValueError("正式 V2 cache 必须首帧 mask 为零且 projected 与 Base 一致")
        if not bool((static_mask.gt(0) | object_mask.gt(0)).any()):
            raise ValueError("正式 V2 cache 不得把空 component mask 标记为有效")
        if base_rgb is None or projected_rgb is None:
            raise ValueError("正式 V2 cache 必须保存 Base 与 projected RGB")
        if base_rgb.shape != projected_rgb.shape or base_rgb.ndim != 4 or base_rgb.shape[0] != y.shape[0]:
            raise ValueError("Base/projected RGB 必须是同 shape 的 [T,C,H,W]")
        if not bool(torch.isfinite(base_rgb).all()) or not bool(torch.isfinite(projected_rgb).all()):
            raise ValueError("Base/projected RGB 包含 NaN/Inf")
        if not torch.equal(base_rgb[0], projected_rgb[0]):
            raise ValueError("正式 V2 cache 的 projected RGB 首帧必须与 Base 一致")
        if base_latent is None or projected_latent is None or latent_residual is None:
            raise ValueError("正式 V2 cache 必须保存 Base/projected latent 与 residual")
        if base_latent.shape != projected_latent.shape or base_latent.shape != latent_residual.shape:
            raise ValueError("Base/projected latent 与 residual shape 必须一致")
        if base_latent.ndim != 4 or base_latent.shape[0] != y.shape[0]:
            raise ValueError("Base/projected latent 必须与 RGB 保持相同 frame count")
        if not all(bool(torch.isfinite(value).all()) for value in
                   (base_latent, projected_latent, latent_residual)):
            raise ValueError("Base/projected latent 或 residual 包含 NaN/Inf")
        if not torch.allclose(latent_residual, projected_latent - base_latent, atol=1e-6, rtol=1e-5):
            raise ValueError("latent_residual 必须等于 projected_latent - base_latent")
        if store == "latent" and (
            not torch.equal(y, base_latent) or not torch.equal(x_dagger, projected_latent)
        ):
            raise ValueError("latent cache 的 y/x_dagger 必须分别等于 Base/projected latent")
        if store == "rgb" and (
            not torch.equal(y, base_rgb) or not torch.equal(x_dagger, projected_rgb)
        ):
            raise ValueError("RGB cache 的 y/x_dagger 必须分别等于 Base/projected RGB")
        actual_quality = {
            "static_valid_fraction": float(static_mask.gt(0).float().mean()),
            "object_valid_fraction": float(object_mask.gt(0).float().mean()),
            "static_confidence_mean": float(static_confidence.mean()),
            "object_confidence_mean": float(object_confidence.mean()),
        }
        for name, value in actual_quality.items():
            declared = metadata[name]
            if not isinstance(declared, (int, float)) or not 0.0 <= float(declared) <= 1.0:
                raise ValueError(f"正式 V2 cache 的 {name} 必须位于 [0,1]")
            if abs(float(declared) - value) > 1e-6:
                raise ValueError(f"正式 V2 cache 的 {name} 与保存 tensor 不一致")
        before = metadata["energy_before_by_component"]
        after = metadata["energy_after_by_component"]
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("正式 V2 cache 的 component energy 必须是字典")
        confidence = {"static": float(static_confidence.mean()), "object": float(object_confidence.mean())}
        for component, value in before.items():
            if component not in after or not isinstance(value, (int, float)) or not isinstance(after[component], (int, float)):
                raise ValueError("正式 V2 cache 的 component energy 不完整")
            if confidence.get(component, 0.0) >= 0.5 and float(after[component]) > float(value) + 1e-6:
                raise ValueError(f"正式 V2 cache 的高置信 {component} energy 上升")

    def write(self, sample_id: str, y: torch.Tensor, x_dagger: torch.Tensor, mask: torch.Tensor,
              metadata: dict, context: dict | None = None, clean: torch.Tensor | None = None,
              latent_flow: torch.Tensor | None = None,
              flow_confidence: torch.Tensor | None = None, *, source: str | None = None,
              generation_seed: int | None = None, parent_checkpoint: str | None = None,
              source_fingerprint: str | None = None, static_mask: torch.Tensor | None = None,
              object_mask: torch.Tensor | None = None, static_confidence: torch.Tensor | None = None,
              object_confidence: torch.Tensor | None = None, base_rgb: torch.Tensor | None = None,
              projected_rgb: torch.Tensor | None = None, base_latent: torch.Tensor | None = None,
              projected_latent: torch.Tensor | None = None, latent_residual: torch.Tensor | None = None) -> str:
        if self.exists(sample_id) and not self.overwrite:
            return self.sample_dir(sample_id)
        clean = y if clean is None else clean
        self._validate(clean, y, x_dagger, mask, context, latent_flow, flow_confidence)
        if self.formal_v2:
            self._validate_formal_v2(
                sample_id, y, x_dagger, mask, metadata, static_mask, object_mask,
                static_confidence, object_confidence, base_rgb, projected_rgb,
                base_latent, projected_latent, latent_residual, self.store,
            )
        target = self.sample_dir(sample_id)
        if os.path.exists(target):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            os.replace(target, f"{target}.stale-{stamp}")
        with atomic_directory(target) as tmp:
            torch.save(clean.detach().cpu(), os.path.join(tmp, "clean.pt"))
            torch.save(y.detach().cpu(), os.path.join(tmp, "y.pt"))
            torch.save(x_dagger.detach().cpu(), os.path.join(tmp, "x_dagger.pt"))
            torch.save(mask.detach().cpu(), os.path.join(tmp, "mask.pt"))
            if context is not None:
                torch.save({key: value.detach().cpu() for key, value in context.items()}, os.path.join(tmp, "context.pt"))
            if latent_flow is not None:
                torch.save(latent_flow.detach().cpu(), os.path.join(tmp, "latent_flow.pt"))
            if flow_confidence is not None:
                torch.save(flow_confidence.detach().cpu(), os.path.join(tmp, "flow_confidence.pt"))
            for name, value in (
                ("static_mask", static_mask), ("object_mask", object_mask),
                ("static_confidence", static_confidence), ("object_confidence", object_confidence),
                ("base_rgb", base_rgb), ("projected_rgb", projected_rgb),
                ("base_latent", base_latent), ("projected_latent", projected_latent),
                ("latent_residual", latent_residual),
            ):
                if value is not None:
                    torch.save(value.detach().cpu(), os.path.join(tmp, f"{name}.pt"))
            meta = dict(metadata)
            meta.update({"store": self.store, "cache_schema_version": CACHE_SCHEMA_VERSION,
                         "cache_fingerprint": self.fingerprint,
                         "formal_v2": self.formal_v2,
                         "source": source or meta.get("source"),
                         "generation_seed": generation_seed,
                         "parent_checkpoint": parent_checkpoint,
                         "source_fingerprint": source_fingerprint})
            atomic_write_json(os.path.join(tmp, "metadata.json"), meta)
            with open(os.path.join(tmp, COMPLETE), "w", encoding="utf-8") as handle:
                handle.write((self.fingerprint or "unversioned") + "\n")
        return target
