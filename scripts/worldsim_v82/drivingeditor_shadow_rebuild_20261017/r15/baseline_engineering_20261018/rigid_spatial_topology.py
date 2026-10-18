"""Spatial part of pinned native densification for the explicit rigid adapter.

Candidate for the new formal instance phase; not installed in any running job.
Native temporal coordinates, time splits and velocity are intentionally absent.
No R15 geometry clamps, per-event caps or ownership transfers are introduced.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SpatialPolicy:
    scene_extent: float
    percent_dense: float = .01
    gradient_threshold: float = .00017
    min_opacity: float = .005
    big_point_threshold: float = .1
    contracted: bool = True
    from_step: int = 500
    until_step: int = 15000
    interval: int = 100
    opacity_reset_interval: int = 3000
    dynamic_point_limit: int = 1000000
    static_point_limit: int = 3000000

    def __post_init__(self):
        for name in ['scene_extent','percent_dense','gradient_threshold','min_opacity','big_point_threshold']:
            value = getattr(self,name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('positive finite policy required: '+name)
        if not 0 < self.min_opacity < 1 or self.interval <= 0 or self.from_step < 0 or self.until_step <= self.from_step:
            raise ValueError('invalid opacity or schedule')
        # Keep the actual pinned native point-limit condition. Other sentinel
        # combinations are not silently reinterpreted as an unlimited policy.
        if self.static_point_limit >= 0 and self.dynamic_point_limit < 1:
            raise ValueError('unsupported native sentinel combination')

    def accumulation_active(self, step, points):
        return step < self.until_step and (self.static_point_limit < 0 or points < self.dynamic_point_limit)

    def mutation_due(self, step, points):
        return self.accumulation_active(step,points) and step > self.from_step and step % self.interval == 0


def world_rows(model, timestamp):
    identity_rotation = torch.zeros_like(model.edit_rotation)
    identity_rotation[:,0] = 1
    if bool(model.deleted.any()) or bool((model.replacement_source >= 0).any()) or bool(model.edit_translation.count_nonzero()) or not torch.equal(model.edit_rotation,identity_rotation):
        raise ValueError('training topology rejects edited scene state')
    world = model.evaluate(timestamp)
    if len(world.row_ids) != model.num_points or not torch.equal(world.row_ids.sort().values,
                                                               torch.arange(model.num_points,device=model.local_xyz.device)):
        raise ValueError('formal topology requires the complete unedited valid rigid scene at the sampled frame')
    xyz = torch.empty_like(model.local_xyz)
    xyz[world.row_ids] = world.xyz
    return xyz


def contracted_factor(xyz, policy):
    if policy.contracted:
        factor = xyz.norm(dim=-1)*policy.scene_extent - 1
        return torch.where(factor <= 1, torch.ones_like(factor), factor)/policy.scene_extent
    return torch.ones_like(xyz[:,0])/policy.scene_extent


def reset_statistics(model):
    model.xyz_gradient_accum.zero_()
    model.denom.zero_()
    model.max_radii2D.zero_()


@torch.no_grad()
def native_append(model, source, optimizer, offsets=None):
    """Native append replaces Parameters and resets statistics even if empty.

This preserves the native loss-gradient lifecycle on a densification update;
it does not pretend old gradients survive Parameter replacement.
"""
    if source.numel():
        added = model.clone_points(source, offsets=offsets, optimizer=optimizer)
    else:
        all_rows = torch.arange(model.num_points,device=model.local_xyz.device)
        model._replace_point_rows(all_rows,model.num_points,optimizer)
        added = all_rows[:0]
    reset_statistics(model)
    return added


@torch.no_grad()
def spatial_mutation(model, optimizer, policy, timestamp, *, max_screen_radius=None):
    """Clone small points, split anisotropic large ones, then native prune.

World distance selects thresholds; samples/covariance stay in the actor basis.
Native append resets projected-radius history before the prune predicate. This
port preserves and reports that behavior rather than claiming an effective
screen-radius guard that the pinned ordering does not provide.
"""
    from rigid_model import quaternion_to_matrix
    original_count = model.num_points
    if original_count == 0:
        raise ValueError('no rigid geometry')
    if bool(((model.denom == 0) & (model.xyz_gradient_accum != 0)).any()):
        raise ValueError('inconsistent accumulated gradient/count')
    average = model.xyz_gradient_accum/model.denom
    average[torch.isnan(average)] = 0
    if not bool(torch.isfinite(average).all()):
        raise FloatingPointError('nonfinite topology statistic')
    scaling = model._scaling.exp()
    limit = policy.percent_dense*policy.scene_extent*contracted_factor(world_rows(model,timestamp),policy)
    # Native dynamic branch halves the gradient threshold.
    high = average.flatten() >= policy.gradient_threshold*.5
    small = scaling.amax(-1) <= limit
    clone = torch.nonzero(high & small,as_tuple=False).flatten()
    split = torch.nonzero(high & ~small,as_tuple=False).flatten()
    native_append(model,clone,optimizer)
    if split.numel():
        source = split.repeat(2)
        std = model._scaling[split].exp().repeat(2,1)
        sample = torch.normal(mean=torch.zeros_like(std),std=std)
        rotations = quaternion_to_matrix(model._rotation[split]).repeat(2,1,1)
        offsets = torch.bmm(rotations,sample.unsqueeze(-1)).squeeze(-1)
        child_scale = torch.log(std/(2*.8))
        children = native_append(model,source,optimizer,offsets=offsets)
        model._scaling[children] = child_scale
    else:
        native_append(model,split,optimizer)
    parents = torch.zeros(model.num_points,dtype=torch.bool,device=model.local_xyz.device)
    parents[split] = True
    model.prune_points(parents,optimizer=optimizer)
    remove = model._opacity.sigmoid().flatten() < policy.min_opacity
    world_large = torch.zeros_like(remove)
    if max_screen_radius is not None:
        threshold = policy.big_point_threshold*policy.scene_extent*contracted_factor(world_rows(model,timestamp),policy)
        world_large = model._scaling.exp().amax(-1) > threshold
        remove |= (model.max_radii2D > max_screen_radius) | world_large
    opacity_count = int((model._opacity.sigmoid().flatten() < policy.min_opacity).sum())
    prune_count = int(remove.sum())
    model.prune_points(remove,optimizer=optimizer)
    reset_statistics(model)
    return dict(original_points=original_count, cloned=int(clone.numel()), split_parents=int(split.numel()),
        split_children=int(split.numel()*2), pruned_after_split=prune_count,
        low_opacity_predicate_count=opacity_count, world_large_predicate_count=int(world_large.sum()),
        final_points=model.num_points, native_append_resets_screen_radius_history=True,
        temporal_PVG_parameters_or_time_splits=False, unused_native_temporal_RNG_draws_ported=False,
        owner_transfers=0)


@torch.no_grad()
def after_backward(model, optimizer, package, policy, timestamp, step):
    """Collect only the original joint-render screen gradient, before Adam."""
    if not policy.accumulation_active(step,model.num_points):
        return dict(active=False,mutated=False)
    gradient = package['viewspace_points'].grad
    if gradient is None:
        raise ValueError('missing actual full-scene screen-space gradient')
    start = package['static_point_count']
    rows = package['instance_row_ids']
    radii = package['radii'][start:]
    if gradient[start:].shape[0] != len(rows) or len(radii) != len(rows):
        raise ValueError('rendered dynamic rows differ from topology gradient rows')
    model.accumulate_densification_stats(rows,gradient[start:],visible=radii>0,radii=radii)
    if not policy.mutation_due(step,model.num_points):
        return dict(active=True,mutated=False)
    result = spatial_mutation(model,optimizer,policy,timestamp,
        max_screen_radius=10 if step > policy.opacity_reset_interval else None)
    result.update(active=True,mutated=True,step=step)
    return result
