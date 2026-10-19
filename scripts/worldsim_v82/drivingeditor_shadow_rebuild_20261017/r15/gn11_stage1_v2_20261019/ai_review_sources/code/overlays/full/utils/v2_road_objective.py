"""Static-only road reconstruction objective, separate from ownership losses."""
import torch


def road_objective(static_rgb, static_depth_sum, static_alpha, target, weights=None):
    rgb=target['rgb'].detach()
    depth=target['depth'].detach()
    provenance=target['provenance'].detach()
    confidence=target['confidence'].detach()
    if not torch.isfinite(rgb).all() or not torch.isfinite(depth).all():
        raise ValueError('nonfinite road target')
    observed=(provenance==1).to(static_rgb)*confidence
    prior=(provenance==2).to(static_rgb)*confidence
    def average(value, weights):
        return (value*weights).sum()/weights.sum().clamp_min(1)
    rgb_error=(static_rgb-rgb).abs().mean(0,keepdim=True)
    prediction_depth=static_depth_sum/static_alpha.clamp_min(1e-5)
    # Observed targets use measured surface support; prior depth is explicitly a surface prior.
    inverse_error=(1/prediction_depth.clamp_min(1e-5)-1/depth.clamp_min(1e-5)).abs()
    raw=dict(observed_rgb=average(rgb_error,observed),
             prior_rgb=average(rgb_error,prior),
             observed_depth=average(inverse_error,observed),
             prior_depth=average(inverse_error,prior))
    # Fixed initial candidate weights, all independently logged and frozen in config.
    weights=weights or dict(observed_rgb=1.,prior_rgb=.1,observed_depth=.1,prior_depth=.01)
    if set(weights)!=set(raw) or any(not torch.isfinite(torch.tensor(v)) or v<0 for v in weights.values()):
        raise ValueError('explicit finite nonnegative road weights required')
    total=sum(raw[k]*weights[k] for k in raw)
    if not torch.isfinite(total): raise FloatingPointError('road objective nonfinite')
    return total,raw
