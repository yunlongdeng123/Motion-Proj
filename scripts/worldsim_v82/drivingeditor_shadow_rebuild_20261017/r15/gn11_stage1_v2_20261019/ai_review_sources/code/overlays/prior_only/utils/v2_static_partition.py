"""GN11 v2 ownership constraints. No road appearance or prior-fill objective here.

The static adapter applies only to the native static class. The dynamic class
must continue to inherit the unmodified native base (see build_candidate.py).
"""
import torch
import torch.nn.functional as F


class StaticV2Mixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_grad = False
        self.no_time_split = True

    def get_xyz_SHM(self, t, times=1):
        return self._xyz

    def get_marginal_t(self, timestamp):
        return torch.ones_like(self._opacity)

    @property
    def get_inst_velocity(self):
        return torch.zeros_like(self._velocity)

    @property
    def get_scaling_t(self):
        # Constant feature retained for native renderer/checkpoint compatibility.
        return torch.full_like(self._scaling_t, 2.0)

    def enforce_static(self):
        self.t_grad = False
        self.no_time_split = True
        with torch.no_grad():
            self._velocity.zero_()
        for field in ('_velocity', '_t', '_scaling_t'):
            getattr(self, field).requires_grad_(False)
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get('name') in ('velocity', 't', 'scaling_t'):
                    group['lr'] = 0.0

    def create_from_pcd(self, *args, **kwargs):
        result = super().create_from_pcd(*args, **kwargs)
        self.enforce_static()
        return result

    def training_setup(self, *args, **kwargs):
        result = super().training_setup(*args, **kwargs)
        self.enforce_static()
        return result

    def update_learning_rate(self, *args, **kwargs):
        result = super().update_learning_rate(*args, **kwargs)
        self.enforce_static()
        return result

    def restore(self, *args, **kwargs):
        result = super().restore(*args, **kwargs)
        self.enforce_static()
        return result

    def densify_and_prune(self, *args, **kwargs):
        self.enforce_static()
        result = super().densify_and_prune(*args, **kwargs)
        self.enforce_static()
        return result

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # Native code reads _t.grad.clone(), which is absent for true static GS.
        if viewspace_point_tensor.grad is None:
            raise RuntimeError('missing actual raster screen gradient')
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.t_gradient_accum.zero_()


def route_depth(joint_depth, dynamic_depth_sum, dynamic_alpha, selected_mask):
    """Body depth has no connection to static geometry; no image alpha erasure."""
    return torch.where(selected_mask.bool(),
                       dynamic_depth_sum / dynamic_alpha.clamp_min(1e-5), joint_depth)


def observed_rgb_loss(prediction, target, selected_mask, ssim_fn, dssim=0.2):
    """Zero *direct pixel* gradient inside selected masks, including SSIM input.

    Neighboring visible pixels may still constrain shared Gaussian parameters.
    The physical forward render itself remains unmasked for evaluation.
    """
    outside = 1 - selected_mask.to(prediction)
    a, b = prediction * outside, target * outside
    return (1-dssim) * F.l1_loss(a, b) + dssim * (1-ssim_fn(a, b))


def vehicle_volume_opacity(xyz, opacity, centers, axes, half_sizes,
                           world_up, ground_offset, road_clearance,
                           softness, chunk_size=131072):
    """Soft occupancy exclusion ONLY within measured selected vehicle volumes.

    All geometry is in model coordinates; axes columns are OBB unit axes.
    A protected slab above the measured road prevents penalizing road Gaussians.
    This is a conservative center test, not exact splat-contribution attribution.
    No hard pruning, mask alpha zeroing, or synthetic vehicle-point initialization.
    """
    if centers.numel() == 0:
        return opacity.sum() * 0
    if softness <= 0 or road_clearance <= 0:
        raise ValueError('positive scale-calibrated margins required')
    total = opacity.sum() * 0
    weight_sum = torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    for start in range(0, len(xyz), chunk_size):
        points = xyz[start:start+chunk_size]
        # N x B x 3, bounded to avoid full-scene covariance-sized allocations.
        local = torch.einsum('nbi,bij->nbj', points[:, None] - centers, axes)
        distance = (half_sizes[None] - local.abs()).amin(-1)
        # Compact support: no tiny positive tail penalizing the entire background.
        membership = (distance / softness).clamp(0, 1).amax(1)
        height = points @ world_up + ground_offset
        membership = membership * ((height-road_clearance)/softness).clamp(0, 1)
        weights = membership.detach()
        total = total + (opacity[start:start+chunk_size, 0] * weights).sum()
        weight_sum = weight_sum + weights.sum()
    # Detached geometry assignment avoids pushing points out of cars to evade loss.
    return total / weight_sum.clamp_min(1)


def independent_road_backward(ownership_loss, road_loss, static_parameters):
    """Two named objectives; road gradients may reach static parameters ONLY.

    Both accumulate before a single native optimizer step; no hidden extra update.
    Environment and dynamic optimizers receive only ownership/reconstruction loss.
    """
    params = [p for p in static_parameters if p.requires_grad]
    road_grads = torch.autograd.grad(road_loss, params, retain_graph=True,
                                    allow_unused=True) if params and road_loss.requires_grad else [None]*len(params)
    ownership_loss.backward()
    with torch.no_grad():
        for p, g in zip(params, road_grads):
            if g is not None:
                if not torch.isfinite(g).all():
                    raise FloatingPointError('non-finite independent road gradient')
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.add_(g)
