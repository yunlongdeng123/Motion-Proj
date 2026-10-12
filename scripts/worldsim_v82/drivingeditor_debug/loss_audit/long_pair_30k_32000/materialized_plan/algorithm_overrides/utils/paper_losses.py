"""Final-paper losses, isolated from the released DrivingEditor objective.

Paper: DrivingEditor, TIP35(2026), printed1700-1703, equations7-16.
The paper does not report all scalar weights or exact NPS sampling offsets.
The choices below are explicit and covered by independent numerical tests:
* Eq13 is applied on the dynamic mask, normalized by full image area. This
  follows the foreground prose; the displayed equation omits an explicit mask.
* Eq9 visits both checkerboard parities with four axial one-pixel neighbors.
  Invalid image-edge neighbors are omitted. Distance is the Euclidean distance
  to the nearest static pixel center; no-static-reference masks contribute zero.
* NPS uses the equation's sum, not a mean over foreground pixels or neighbors.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _mask_like(mask, reference):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape != (1, reference.shape[-2], reference.shape[-1]):
        raise ValueError("A binary mask must have shape1xHxW matching the render")
    return (mask > 0.5).to(device=reference.device, dtype=reference.dtype)


def sky_loss(static_alpha, sky_mask, eps=1e-4):
    """Eq7: -sum(o_sky*log(1-o_sky))/(H*W), o_sky=o_static*Msky."""
    mask = _mask_like(sky_mask, static_alpha)
    opacity = static_alpha.clamp(0.0, 1.0 - eps) * mask
    return -(opacity * torch.log1p(-opacity)).mean()


def dynamic_opacity_loss(dynamic_alpha, dynamic_mask, eps=1e-4):
    """Eq13 on foreground pixels, with the printed full-image normalization."""
    mask = _mask_like(dynamic_mask, dynamic_alpha)
    opacity = dynamic_alpha.clamp(eps, 1.0 - eps)
    return -(mask * (1.0 - opacity) * torch.log(opacity)).mean()


def boundary_distance(dynamic_mask):
    """Return detached1xHxW pixel distances; cache this immutable per-camera map."""
    if dynamic_mask.ndim == 2:
        dynamic_mask = dynamic_mask.unsqueeze(0)
    if dynamic_mask.ndim != 3 or dynamic_mask.shape[0] != 1:
        raise ValueError("Expected a1xHxW dynamic mask")
    mask = dynamic_mask.detach().to(device="cpu").numpy()[0] > 0.5
    if not mask.any() or mask.all():
        distances = np.zeros(mask.shape, dtype=np.float64)
    else:
        distances = distance_transform_edt(mask)
    return torch.as_tensor(distances[None], device=dynamic_mask.device,
                           dtype=dynamic_mask.dtype)


def nps_loss(static_features, dynamic_mask, distances=None):
    """Eq9's sum of four-neighbor vector L2 discrepancies / boundary distance."""
    if static_features.ndim != 3:
        raise ValueError("ExpectedCxHxW opacity-normalized static features")
    mask = _mask_like(dynamic_mask, static_features)
    if distances is None:
        distances = boundary_distance(mask)
    if distances.shape != mask.shape:
        raise ValueError("NPS distance map and mask dimensions differ")
    distances = distances.to(device=static_features.device, dtype=static_features.dtype)
    weights = mask[0] * (distances[0] > 0).to(static_features.dtype)
    weights = weights / distances[0].clamp_min(1.0)
    # Four axial neighbors are opposite-color neighbors in red/black sampling.
    # Evaluating every source pixel visits both parities without an order effect.
    horizontal = torch.linalg.vector_norm(static_features[:, :, 1:] - static_features[:, :, :-1], dim=0)
    vertical = torch.linalg.vector_norm(static_features[:, 1:, :] - static_features[:, :-1, :], dim=0)
    result = static_features.sum() * 0.0
    result = result + (horizontal * weights[:, 1:]).sum() + (horizontal * weights[:, :-1]).sum()
    result = result + (vertical * weights[1:, :]).sum() + (vertical * weights[:-1, :]).sum()
    return result


def sparse_inverse_depth_loss(rendered_depth, lidar_depth, eps=1e-5):
    """Eq16: valid sparse inverse-depth residuals summed / full image area."""
    if rendered_depth.shape != lidar_depth.shape or rendered_depth.shape[0] != 1:
        raise ValueError("Depth maps must have matching1xHxW shapes")
    valid = (lidar_depth > 0) & torch.isfinite(lidar_depth)
    # Invalid zero depth never enters a reciprocal, keeping gradients finite.
    safe_lidar = torch.where(valid, lidar_depth, torch.ones_like(lidar_depth))
    safe_rendered = torch.where(valid, rendered_depth, torch.ones_like(rendered_depth))
    residual = (safe_lidar.add(eps).reciprocal() - safe_rendered.add(eps).reciprocal()).abs()
    return (residual * valid.to(residual.dtype)).sum() / rendered_depth.shape[-2] / rendered_depth.shape[-1]


def compose_paper_loss(global_rgb, dynamic_rgb, sky, nps, opacity, depth,
                       lambda_sky, lambda_nps, lambda_opacity, lambda_depth,
                       lambda_dynamic_rgb=1.0):
    """Eqs10,15-16: no legacy5x photometric multipliers or hidden PVG terms."""
    return (global_rgb + lambda_dynamic_rgb * dynamic_rgb + lambda_sky * sky + lambda_nps * nps
            + lambda_opacity * opacity + lambda_depth * depth)


def diagnostic_dynamic_background_opacity_loss(dynamic_alpha, dynamic_mask, eps=1e-4):
    """Experimental background-domain complement, NOT a verified author term.

    Retains Eq13 on the foreground. Outside Md, use the Eq7-shaped soft
    zero-opacity penalty -a*log((1-a+eps)/(1+eps)), normalized by full image area.
    This stabilizes the logarithm without a high-alpha zero-gradient plateau.
    Mask uncertainty and the extra coefficient require a controlled diagnosis.
    No old100*MSE or old composite/static RGB objective is restored.
    """
    foreground = _mask_like(dynamic_mask, dynamic_alpha)
    opacity = dynamic_alpha.clamp(0.0, 1.0)
    return -((1.0 - foreground) * opacity * torch.log1p(-opacity / (1.0 + eps))).mean()
