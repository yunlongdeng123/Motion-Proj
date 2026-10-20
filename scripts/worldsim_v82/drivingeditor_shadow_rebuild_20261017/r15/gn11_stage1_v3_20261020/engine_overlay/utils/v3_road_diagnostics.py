"""同一世界道路单元在三种物理合成中的可见贡献，不更改训练前向。"""

import torch
from utils.v3_multiview_road import sample_map


@torch.no_grad()
def road_visibility(package, road_package, road_mask, unit_ids, uv, supported, provenance, query, geometric_in_view,
                    contribution_renderer):
    flags = road_mask.to(package['background_means3D'])[:, None]
    static_map, joint_map = [contribution_renderer(flags, joint) for joint in [False, True]]
    for result, name in [(static_map, 'static_alpha'), (joint_map, 'alpha')]:
        if not torch.allclose(result[-1:], package[name], atol=2e-5, rtol=2e-5):
            raise RuntimeError('road contribution replay changed original transmittance')
    alone = sample_map(road_package['alpha'], uv)[:, 0]
    background = sample_map(static_map[:1], uv)[:, 0]
    joint = sample_map(joint_map[:1], uv)[:, 0]
    shape = package['alpha'].shape[-2:]
    in_view = (geometric_in_view & torch.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= shape[1]-1)
               & (uv[:, 1] >= 0) & (uv[:, 1] <= shape[0]-1) & supported)
    arrays = dict(unit_id=unit_ids, raster_uv=uv, supported=supported, provenance=provenance,
                  is_query=query, in_view=in_view, road_only_alpha=alone,
                  road_alpha_in_background=background, road_alpha_in_joint=joint,
                  background_attenuation=(alone-background), joint_attenuation=(alone-joint))
    cohorts = {}
    for name, domain in [('observed', provenance == 1), ('prior', provenance == 2), ('unknown', provenance == 0)]:
        valid = in_view & query & domain
        count = int(valid.sum())
        cohorts[name] = dict(units=count, **{
            key: float(value[valid].mean()) if count else None for key, value in [
                ('road_only_alpha', alone), ('road_alpha_in_background', background),
                ('road_alpha_in_joint', joint), ('background_attenuation', alone-background),
                ('joint_attenuation', alone-joint)]})
    return arrays, dict(cohorts=cohorts, raster_calls=3, original_alpha_replay_checked=True,
                       all_road_gaussians_sampled_at_unit_anchor_not_isolated_per_unit_splat=True,
                       unit_projection_sampling_not_unique_pixel_area=True,
                       attenuation_is_not_illegal_occlusion_ground_truth=True,
                       explanation='normal target vehicles and legitimate non-edit foreground may hide road; no loss or culling')
