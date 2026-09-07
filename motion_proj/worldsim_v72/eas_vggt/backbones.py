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
    if value.ndim != 4 or value.shape[:2] != (1, count):
        raise ValueError(f"expected VGGT tokens [1,{count},N,C], got {tuple(value.shape)}")
    patches = value[:, :, patch_start:, :].reshape(count, height // 14, width // 14, value.shape[-1])
    return patches.detach().to(dtype=torch.float16, device="cpu").numpy()


def _validate_checkpoint_keys(
    incompatible: Any,
    *,
    allowed_unexpected_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    disallowed = [
        name for name in unexpected if not any(name.startswith(prefix) for prefix in allowed_unexpected_prefixes)
    ]
    if missing or disallowed:
        raise RuntimeError(
            f"checkpoint contract mismatch: missing={missing[:8]}, unexpected={disallowed[:8]}"
        )
    return {
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "allowed_unexpected_prefixes": list(allowed_unexpected_prefixes),
    }


class VGGTBackbone:
    backbone_id = "facebook/VGGT-1B"

    def __init__(
        self,
        repository_root: Path,
        checkpoint: Path,
        device: str = "cuda",
        keep_loaded: bool = False,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.keep_loaded = bool(keep_loaded)
        self._model: Any | None = None
        self._checkpoint_contract: dict[str, Any] | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        _add_import_root(self.repository_root)
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT

        model = VGGT(enable_track=False).eval()
        incompatible = model.load_state_dict(load_file(str(self.checkpoint)), strict=False)
        self._checkpoint_contract = _validate_checkpoint_keys(
            incompatible,
            allowed_unexpected_prefixes=("track_head.",),
        )
        self._model = model.to(self.device)
        return self._model

    def close(self) -> None:
        self._model = None
        torch.cuda.empty_cache()

    def infer(
        self,
        window: CameraWindow,
        images: torch.Tensor,
        model_from_original_px: np.ndarray,
    ) -> BackboneGeometry:
        model = self._load_model()
        from vggt.utils.geometry import closed_form_inverse_se3
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = images.to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            tokens, patch_start = model.aggregator(images[None])
            pose_encoding = model.camera_head(tokens)[-1]
            points, confidence = model.point_head(tokens, images=images[None], patch_start_idx=patch_start)
        extrinsic, intrinsics = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
        camera_from_reference = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype)[None, None].repeat(1, len(window.frames), 1, 1)
        camera_from_reference[..., :3, :4] = extrinsic
        reference_from_camera = closed_form_inverse_se3(camera_from_reference.flatten(0, 1)).reshape_as(camera_from_reference)
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
                **dict(self._checkpoint_contract or {}),
                "payload_role": "build_input",
            },
        )
        del images, tokens, pose_encoding, points, confidence
        if not self.keep_loaded:
            self.close()
        return result

class Pi3XBackbone:
    backbone_id = "yyfz233/Pi3X"

    def __init__(
        self,
        repository_root: Path,
        checkpoint: Path,
        device: str = "cuda",
        keep_loaded: bool = False,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.keep_loaded = bool(keep_loaded)
        self._model: Any | None = None
        self._checkpoint_contract: dict[str, Any] | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        _add_import_root(self.repository_root)
        from safetensors.torch import load_file
        from pi3.models.pi3x import Pi3X

        model = Pi3X(use_multimodal=True).eval()
        incompatible = model.load_state_dict(load_file(str(self.checkpoint)), strict=False)
        self._checkpoint_contract = _validate_checkpoint_keys(incompatible)
        model.disable_multimodal(free_cuda_cache=False)
        self._model = model.to(self.device)
        return self._model

    def close(self) -> None:
        self._model = None
        torch.cuda.empty_cache()

    def infer(
        self,
        window: CameraWindow,
        images: torch.Tensor,
        model_from_original_px: np.ndarray,
    ) -> BackboneGeometry:
        model = self._load_model()
        from pi3.utils.geometry import recover_intrinsic_from_rays_d

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
                **dict(self._checkpoint_contract or {}),
                "payload_role": "build_input",
            },
        )
        del images, hidden, outputs, rays
        if not self.keep_loaded:
            self.close()
        return result


class MapAnythingBackbone:
    """Official MapAnything with calibrated metric poses as its native prompted baseline."""

    backbone_id = "facebook/map-anything"

    def __init__(
        self,
        repository_root: Path,
        checkpoint: Path,
        device: str = "cuda",
        keep_loaded: bool = False,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.keep_loaded = bool(keep_loaded)
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            _add_import_root(self.repository_root)
            from mapanything.models import MapAnything

            self._model = MapAnything.from_pretrained(str(self.checkpoint.parent)).eval().to(self.device)
        return self._model

    def close(self) -> None:
        self._model = None
        torch.cuda.empty_cache()

    def infer(
        self,
        window: CameraWindow,
        images: torch.Tensor,
        model_from_original_px: np.ndarray,
    ) -> BackboneGeometry:
        del images, model_from_original_px
        from PIL import Image
        from mapanything.utils.image import preprocess_inputs

        raw_views = []
        for frame in window.frames:
            with Image.open(frame.image_path) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.uint8)
            actual_h, actual_w = image.shape[:2]
            original_w, original_h = map(int, frame.original_size_wh)
            actual_from_original = np.asarray(
                [[actual_w / original_w, 0.0, 0.0], [0.0, actual_h / original_h, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            raw_views.append(
                {
                    "img": image,
                    "intrinsics": np.asarray(
                        actual_from_original @ frame.intrinsics_px,
                        dtype=np.float32,
                    ),
                    "camera_poses": np.asarray(frame.world_from_camera_opencv, dtype=np.float32),
                    "is_metric_scale": True,
                }
            )
        views = preprocess_inputs(raw_views, resolution_set=518, norm_type="dinov2", patch_size=14)
        feature_grid = torch.nn.functional.avg_pool2d(
            torch.cat([view["img"] for view in views]), kernel_size=14, stride=14
        ).permute(0, 2, 3, 1).to(dtype=torch.float16, device="cpu").numpy()
        model = self._load_model()
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
        points = np.stack([value["pts3d"][0].float().cpu().numpy() for value in predictions])
        confidence = np.stack([value["conf"][0].float().cpu().numpy() for value in predictions])
        masks = np.stack([value["mask"][0].bool().cpu().numpy().squeeze(-1) for value in predictions])
        poses = np.stack([value["camera_poses"][0].float().cpu().numpy() for value in predictions])
        intrinsics = np.stack([value["intrinsics"][0].float().cpu().numpy() for value in predictions])
        transforms = np.stack(
            [intrinsics[index] @ np.linalg.inv(frame.intrinsics_px) for index, frame in enumerate(window.frames)]
        )
        result = BackboneGeometry(
            backbone_id=self.backbone_id,
            checkpoint_id=f"sha256:{_sha256(self.checkpoint)}",
            repository_commit=_repository_commit(self.repository_root),
            frame_ids=np.asarray([frame.frame_id for frame in window.frames]),
            image_sha256=np.asarray([frame.image_sha256 for frame in window.frames]),
            model_from_original_px=transforms,
            points_reference=points,
            confidence=confidence,
            valid_mask=masks & np.isfinite(points).all(axis=-1) & np.isfinite(confidence),
            reference_from_camera_opencv=poses,
            intrinsics_model_px=intrinsics,
            feature_grid=feature_grid,
            scale_status="metric_aligned",
            provenance={
                "window_fingerprint": window.fingerprint,
                "official_adapter": True,
                "payload_role": "build_input",
                "input_modalities": ["rgb", "intrinsics", "metric_camera_pose"],
                "native_feature_exported": False,
                "feature_grid": "common_rgb_patch_mean_diagnostic_only",
            },
        )
        del predictions, views, feature_grid
        if not self.keep_loaded:
            self.close()
        return result
