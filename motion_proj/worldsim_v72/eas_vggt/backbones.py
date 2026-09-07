"""VGGT 与 Pi3X 官方实现的薄适配层。"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from motion_proj.worldsim_v72.data.camera_schema import CameraWindow
from motion_proj.worldsim_v72.eas_vggt.types import BackboneGeometry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _add_import_root(root: Path) -> None:
    value = str(Path(root).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _last_feature_grid(tokens: list[Any], patch_start: int, count: int, height: int, width: int) -> np.ndarray:
    value = next(item for item in reversed(tokens) if item is not None)
    patches = value[:, patch_start:, :].reshape(count, height // 14, width // 14, -1)
    return patches.detach().to(dtype=torch.float16, device="cpu").numpy()


class VGGTBackbone:
    backbone_id = "facebook/VGGT-1B"

    def __init__(self, repository_root: Path, checkpoint: Path, device: str = "cuda") -> None:
        self.repository_root = Path(repository_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)

    def infer(
        self,
        window: CameraWindow,
        images: torch.Tensor,
        model_from_original_px: np.ndarray,
    ) -> BackboneGeometry:
        _add_import_root(self.repository_root)
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        model = VGGT(enable_track=False).eval()
        incompatible = model.load_state_dict(load_file(str(self.checkpoint)), strict=False)
        model = model.to(self.device)
        images = images.to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            tokens, patch_start = model.aggregator(images[None])
            pose_encoding = model.camera_head(tokens)[-1]
            points, confidence = model.point_head(tokens, images=images[None], patch_start_idx=patch_start)
        extrinsic, intrinsics = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
        camera_from_reference = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype)[None, None].repeat(1, len(window.frames), 1, 1)
        camera_from_reference[..., :3, :4] = extrinsic
        reference_from_camera = torch.linalg.inv(camera_from_reference)
        points_np = points[0].float().cpu().numpy()
        confidence_np = confidence[0].float().cpu().numpy()
        feature_grid = _last_feature_grid(tokens, patch_start, len(window.frames), images.shape[-2], images.shape[-1])
        result = BackboneGeometry(
            backbone_id=self.backbone_id,
            checkpoint_id=f"sha256:{_sha256(self.checkpoint)}",
            repository_commit=_repository_commit(self.repository_root),
            frame_ids=np.asarray([frame.frame_id for frame in window.frames]),
            image_sha256=np.asarray([frame.image_sha256 for frame in window.frames]),
            model_from_original_px=np.asarray(model_from_original_px, dtype=np.float64),
            points_reference=points_np,
            confidence=confidence_np,
            valid_mask=np.isfinite(points_np).all(axis=-1) & np.isfinite(confidence_np),
            reference_from_camera_opencv=reference_from_camera[0].float().cpu().numpy(),
            intrinsics_model_px=intrinsics[0].float().cpu().numpy(),
            feature_grid=feature_grid,
            scale_status="arbitrary",
            provenance={
                "window_fingerprint": window.fingerprint,
                "official_adapter": True,
                "unexpected_key_count": len(incompatible.unexpected_keys),
                "missing_key_count": len(incompatible.missing_keys),
                "payload_role": "build_input",
            },
        )
        del model, images, tokens, pose_encoding, points, confidence
        torch.cuda.empty_cache()
        return result

class Pi3XBackbone:
    backbone_id = "yyfz233/Pi3X"

    def __init__(self, repository_root: Path, checkpoint: Path, device: str = "cuda") -> None:
        self.repository_root = Path(repository_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)

    def infer(
        self,
        window: CameraWindow,
        images: torch.Tensor,
        model_from_original_px: np.ndarray,
    ) -> BackboneGeometry:
        _add_import_root(self.repository_root)
        from safetensors.torch import load_file
        from pi3.models.pi3x import Pi3X
        from pi3.utils.geometry import recover_intrinsic_from_rays_d

        model = Pi3X(use_multimodal=True).eval()
        incompatible = model.load_state_dict(load_file(str(self.checkpoint)), strict=False)
        model.disable_multimodal(free_cuda_cache=False)
        model = model.to(self.device)
        images = images[None].to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            normalized = (images - model.image_mean) / model.image_std
            batch, count, _, height, width = normalized.shape
            hidden, poses, _, pose_mask, _ = model.encode(normalized, with_prior=False)
            hidden = hidden.reshape(batch, count, -1, model.dec_embed_dim)
            hidden, position = model.decode(hidden, count, height, width, poses, pose_mask)
            outputs = model.forward_head(hidden, position, batch, count, height, width, height // 14, width // 14)
        rays = torch.nn.functional.normalize(outputs["local_points"], dim=-1)
        intrinsics = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)
        feature_grid = hidden[:, model.patch_start_idx :, :].reshape(count, height // 14, width // 14, -1)
        feature_grid_np = feature_grid.detach().to(dtype=torch.float16, device="cpu").numpy()
        points_np = outputs["points"][0].float().cpu().numpy()
        confidence_np = torch.sigmoid(outputs["conf"][0, ..., 0]).float().cpu().numpy()
        result = BackboneGeometry(
            backbone_id=self.backbone_id,
            checkpoint_id=f"sha256:{_sha256(self.checkpoint)}",
            repository_commit=_repository_commit(self.repository_root),
            frame_ids=np.asarray([frame.frame_id for frame in window.frames]),
            image_sha256=np.asarray([frame.image_sha256 for frame in window.frames]),
            model_from_original_px=np.asarray(model_from_original_px, dtype=np.float64),
            points_reference=points_np,
            confidence=confidence_np,
            valid_mask=np.isfinite(points_np).all(axis=-1) & np.isfinite(confidence_np),
            reference_from_camera_opencv=outputs["camera_poses"][0].float().cpu().numpy(),
            intrinsics_model_px=intrinsics[0].float().cpu().numpy(),
            feature_grid=feature_grid_np,
            scale_status="approximate_metric",
            provenance={
                "window_fingerprint": window.fingerprint,
                "official_adapter": True,
                "multimodal_conditions_used": False,
                "unexpected_key_count": len(incompatible.unexpected_keys),
                "missing_key_count": len(incompatible.missing_keys),
                "payload_role": "build_input",
            },
        )
        del model, images, hidden, outputs, rays
        torch.cuda.empty_cache()
        return result
