"""WorldSim V4 冻结图像评测口径。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from skimage.metrics import structural_similarity


REGION_NAMES = ("global", "static", "actor", "boundary", "edit_roi")


class EvaluationError(ValueError):
    """输入不满足 V4 图像评测合同。"""


@dataclass(frozen=True)
class LpipsRegionProtocol:
    crop_padding_pixels: int = 8
    minimum_side_pixels: int = 64


def _rgb01(image: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise EvaluationError(f"{label} 必须是 HxWx3 RGB")
    value = value.astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise EvaluationError(f"{label} 含 NaN/Inf")
    if value.size and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
        raise EvaluationError(f"{label} 必须位于 [0,1]")
    return value


def _mask(mask: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape:
        raise EvaluationError(f"{label} shape {value.shape} != {shape}")
    return value.astype(bool, copy=False)


def masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    pred = _rgb01(prediction, "prediction")
    ref = _rgb01(target, "target")
    if pred.shape != ref.shape:
        raise EvaluationError("prediction 与 target shape 不一致")
    region = _mask(mask, pred.shape[:2], "mask")
    if not region.any():
        return None
    mse = float(np.mean(np.square(pred[region] - ref[region]), dtype=np.float64))
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    pred = _rgb01(prediction, "prediction")
    ref = _rgb01(target, "target")
    if pred.shape != ref.shape:
        raise EvaluationError("prediction 与 target shape 不一致")
    region = _mask(mask, pred.shape[:2], "mask")
    if not region.any():
        return None
    minimum = min(pred.shape[:2])
    if minimum < 3:
        raise EvaluationError("SSIM 要求图像最短边至少为 3")
    win_size = min(11, minimum if minimum % 2 else minimum - 1)
    _, score_map = structural_similarity(
        ref,
        pred,
        data_range=1.0,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        win_size=win_size,
        full=True,
    )
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=-1)
    return float(np.mean(score_map[region], dtype=np.float64))


def _tight_bbox(mask: np.ndarray, padding: int) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(mask.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(mask.shape[1], int(xs.max()) + padding + 1)
    return slice(y0, y1), slice(x0, x1)


def masked_lpips_alex(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    model: Callable[[Any, Any], Any],
    protocol: LpipsRegionProtocol = LpipsRegionProtocol(),
) -> float | None:
    """只保留区域内差异后，在 tight bbox 上计算 LPIPS-Alex。"""
    import torch
    import torch.nn.functional as functional

    pred = _rgb01(prediction, "prediction")
    ref = _rgb01(target, "target")
    if pred.shape != ref.shape:
        raise EvaluationError("prediction 与 target shape 不一致")
    region = _mask(mask, pred.shape[:2], "mask")
    if not region.any():
        return None
    masked_pred = np.where(region[..., None], pred, ref)
    rows, cols = _tight_bbox(region, protocol.crop_padding_pixels)

    def tensor(image: np.ndarray) -> Any:
        value = torch.from_numpy(image[rows, cols].transpose(2, 0, 1)).float()[None]
        if min(value.shape[-2:]) < protocol.minimum_side_pixels:
            scale = protocol.minimum_side_pixels / min(value.shape[-2:])
            size = tuple(max(protocol.minimum_side_pixels, int(round(x * scale))) for x in value.shape[-2:])
            value = functional.interpolate(value, size=size, mode="bilinear", align_corners=False)
        return value.mul(2.0).sub(1.0)

    pred_tensor = tensor(masked_pred)
    ref_tensor = tensor(ref)
    try:
        parameter = next(model.parameters())  # type: ignore[attr-defined]
        pred_tensor = pred_tensor.to(parameter.device)
        ref_tensor = ref_tensor.to(parameter.device)
    except (AttributeError, StopIteration):
        pass
    with torch.inference_mode():
        result = model(pred_tensor, ref_tensor)
    if hasattr(result, "detach"):
        result = result.detach().cpu().item()
    return float(result)


def evaluate_frame(
    prediction: np.ndarray,
    target: np.ndarray | None,
    region_masks: Mapping[str, np.ndarray],
    *,
    lpips_model: Callable[[Any, Any], Any] | None = None,
    lpips_protocol: LpipsRegionProtocol = LpipsRegionProtocol(),
) -> dict[str, Any]:
    pred = _rgb01(prediction, "prediction")
    if target is None:
        return {
            "ground_truth": False,
            "regions": {
                name: {"status": "undefined_no_ground_truth", "pixel_count": None, "psnr": None, "ssim": None, "lpips_alex": None}
                for name in REGION_NAMES
            },
        }
    ref = _rgb01(target, "target")
    if pred.shape != ref.shape:
        raise EvaluationError("prediction 与 target shape 不一致")
    masks: dict[str, np.ndarray] = {"global": np.ones(pred.shape[:2], dtype=bool)}
    unknown = sorted(set(region_masks) - set(REGION_NAMES))
    if unknown:
        raise EvaluationError(f"未知 region：{unknown}")
    masks.update({name: _mask(value, pred.shape[:2], name) for name, value in region_masks.items()})
    rows: dict[str, Any] = {}
    for name in REGION_NAMES:
        if name not in masks:
            rows[name] = {"status": "undefined_mask_missing", "pixel_count": None, "psnr": None, "ssim": None, "lpips_alex": None}
            continue
        region = masks[name]
        count = int(region.sum())
        if count == 0:
            rows[name] = {"status": "undefined_empty_region", "pixel_count": 0, "psnr": None, "ssim": None, "lpips_alex": None}
            continue
        rows[name] = {
            "status": "done" if lpips_model is not None else "done_lpips_unavailable",
            "pixel_count": count,
            "psnr": masked_psnr(pred, ref, region),
            "ssim": masked_ssim(pred, ref, region),
            "lpips_alex": None if lpips_model is None else masked_lpips_alex(pred, ref, region, lpips_model, lpips_protocol),
        }
    return {"ground_truth": True, "regions": rows}
