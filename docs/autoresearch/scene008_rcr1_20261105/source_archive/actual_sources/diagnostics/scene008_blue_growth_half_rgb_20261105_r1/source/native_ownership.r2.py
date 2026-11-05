"""Scene008 native training: separate appearance, shared visibility, background growth."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dual_actor_runtime import actor_ids
from evaluate_a6_probe import MARKER_ALPHA, marker_kwargs, marker_positions

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OBS = CAMPAIGN / 'scene008_instances_20261105_r1/observations/frames'
REF = CAMPAIGN / 'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
ACTORS = (6, 37)
MARGIN_M = .30


def average(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def guarded(value, mask):
    return torch.where(mask, value, 0)


class Ownership:
    def __init__(self):
        with np.load(REF, allow_pickle=False) as archive:
            self.reference = {name: archive[name] for name in ('depth_m', 'meter_to_model', 'model_time')}
        self.scale = float(self.reference['meter_to_model'])
        self.observations = OBS
        self.frames = {}
        self.markers = {}

    def evidence(self, camera):
        raw = int(camera.image_name[-3:])
        if raw not in self.frames:
            with np.load(OBS / f'{raw:03d}.npz', allow_pickle=False) as data:
                keys = ('visible', 'free', 'depth_m', 'depth_valid', 'visible_reviewed',
                        'occluded', 'protect', 'road', 'box_candidate', 'model_time', 'model_to_camera')
                obs = {name: data[name] for name in keys}
            if abs(float(obs['model_time']) - camera.timestamp) > 1e-6:
                raise ValueError(f'raw{raw} timestamp mismatch')
            if not np.allclose(obs['model_to_camera'].T, camera.world_view_transform.cpu().numpy(), atol=1e-6):
                raise ValueError(f'raw{raw} view mismatch')
            if obs['visible'].shape != (2, camera.image_height, camera.image_width):
                raise ValueError(f'raw{raw} shape mismatch')
            if (obs['visible'][0] & obs['visible'][1]).any():
                raise ValueError(f'raw{raw} multiple visible owners')
            # The first two fixed layers contain measured a6 surfaces. The second
            # comes from other times, where the same physical surface was visible.
            # Keep the nearest a37 measurement only where a6 has no usable depth.
            fixed = self.reference['depth_m'][raw - 89, :2]
            a6_cut = np.maximum(obs['depth_m'][0], fixed.max(axis=0))
            a37_cut = obs['depth_m'][1]
            # Keep the deeper a6 cutoff where it is supported, including through
            # the a37 silhouette. No generated depths enter the training loss.
            cutoff = np.maximum(a6_cut, a37_cut)
            y, x = np.nonzero((cutoff - MARGIN_M) * self.scale > .2)
            z = (cutoff[y, x] - MARGIN_M) * self.scale
            xyz = marker_positions(camera.world_view_transform.cpu().numpy(),
                                   camera.projection_matrix.cpu().numpy(), np.column_stack((x, y)), z)
            self.markers[raw] = tuple(torch.as_tensor(value, device='cuda') for value in (y, x, z, xyz))
            fields = ('visible', 'free', 'depth_m', 'depth_valid', 'visible_reviewed', 'occluded', 'protect', 'road', 'box_candidate')
            frame = {key: torch.as_tensor(obs[key], device='cuda') for key in fields}
            # Observed road and protected surfaces outside the expanded actor
            # footprint are safe isolated negatives, including oversize tails
            # far beyond the narrow boundary band used by the original probe.
            allowed = frame['box_candidate'] | frame['visible_reviewed']
            allowed = F.max_pool2d(allowed[None].float(), 5, 1, 2)[0].bool()
            frame['safe_background'] = (frame['road'] | frame['protect']) & ~allowed
            self.frames[raw] = frame
        return self.frames[raw], self.markers[raw]

    @staticmethod
    def draw(camera, args, static, dynamic, background, env, other, dyn_other, shift=None,
             training=False, hide=None, markers=None):
        import math
        from gaussian_renderer import (GaussianRasterizationSettings, GaussianRasterizer,
                                       merge_kwargs, prepare_rasterization)
        settings = GaussianRasterizationSettings(
            image_height=camera.image_height, image_width=camera.image_width,
            tanfovx=math.tan(-.5 if args.neg_fov else camera.FoVx * .5),
            tanfovy=math.tan(-.5 if args.neg_fov else camera.FoVy * .5),
            bg=background, scale_modifier=1., viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform, sh_degree=static.active_sh_degree,
            campos=camera.camera_center, prefiltered=False, debug=args.debug)
        rasterizer = GaussianRasterizer(raster_settings=settings)

        def prep(model, features, is_dynamic):
            return prepare_rasterization(camera, model, args, time_shift=shift, other=features, dynamic=is_dynamic)

        def draw(kw):
            return rasterizer(**kw)

        base_t = static._t
        # The native growth accumulator reads _t.grad. Keep a separate alias
        # to read only gradients through the background-supported render.
        growth_t = base_t + 0 if training else base_t
        if training:
            growth_t.retain_grad()
        static._t = growth_t
        try:
            sk = prep(static, other, False)
        finally:
            static._t = base_t
        dk = prep(dynamic, dyn_other, True)
        ids = actor_ids(dynamic)
        actor_rows = [ids == owner for owner in ACTORS]
        count = sum(v.shape[1] for v in other)
        names = ('means3D', 'scales', 'rotations', 'cov3D_precomp', 'opacities', 'mask')
        if training:
            # Joint gradients may change support and opacity but cannot fit SH
            # or update the environment. A distinct temporal graph isolates growth.
            joint_static = prep(static, other, False)
        else:
            joint_static = sk
        sf, df = joint_static['features'], dk['features']
        df_owner = torch.stack(actor_rows, 1).to(df.dtype)
        sf_owner = sf.new_zeros((len(sf), 2))

        def tagged(kw, features, flags):
            z = features[:, -2:-1]
            return dict(kw, features=torch.cat((features[:, :count], flags, flags * z, features[:, -2:]), 1))

        js = tagged(joint_static, sf, sf_owner)
        jd = tagged(dk, df, df_owner)
        if hide is not None:
            jd['mask'] = jd['mask'] & (ids != hide)
        joint = merge_kwargs(js, jd)
        joint['means2D'] = torch.zeros_like(joint['means3D'])
        _, image, packed, radii = draw(joint)
        velocity, owner, owner_z, depth, alpha = packed.split([count, 2, 2, 1, 1], dim=0)
        directions = camera.get_world_directions(training).permute(1, 2, 0)
        env_rgb = env(directions).permute(2, 0, 1)
        result = dict(render=image + (1-alpha) * env_rgb, alpha=alpha, depth=depth,
                      owner=owner, owner_z=owner_z, feature=velocity, radii=radii,
                      env_map=(1-alpha) * env_rgb)
        if hide is not None:
            return result

        def screen(kw):
            screen_xy = torch.zeros_like(kw['means3D'], requires_grad=training) + 0
            if training:
                screen_xy.retain_grad()
            return screen_xy

        sk['means2D'] = screen(sk)
        _, static_rgb, static_maps, static_radii = draw(sk)
        result.update(render_s=static_rgb + (1-static_maps[-1:]) * env_rgb,
                      static_feature=static_maps[:count], static_depth=static_maps[-2:-1],
                      static_alpha=static_maps[-1:], static_growth=sk['means2D'],
                      static_growth_t=growth_t, static_radii=static_radii)
        dyn_xy = screen(dk)
        isolated = []
        dyn_radii = torch.zeros_like(radii[len(sf):])
        for rows in actor_rows:
            # Gather the actor's native rows, using a common screen tensor so
            # its retained gradient is in the original dynamic row order.
            local = {key: (None if value is None else value[rows].contiguous()) for key, value in dk.items()}
            local['means2D'] = dyn_xy[rows].contiguous()
            _, rgb, maps, local_radii = draw(local)
            dyn_radii[rows] = local_radii
            isolated.append((rgb, maps[-2:-1], maps[-1:]))
        isolated_rgb = isolated[0][0] + isolated[1][0]
        isolated_alpha = 1 - (1-isolated[0][2]) * (1-isolated[1][2])
        result.update(isolated=isolated, dynamic_growth=dyn_xy, dynamic_radii=dyn_radii,
                      render_d=isolated_rgb, dynamic_alpha=isolated_alpha,
                      env_map_s=(1-static_maps[-1:]) * env_rgb)
        if markers is not None:
            y, x, z, xyz = markers
            # Fixed probes use a separate pass. This penalty can only decrease
            # static opacity ahead of the observed surface. It has no RGB term.
            native = {key: joint_static[key] for key in (*names, 'features', 'shs', 'colors_precomp')}
            native['features'] = sf.new_zeros((len(sf), 3))
            native['means2D'] = torch.zeros_like(native['means3D'])
            marker = marker_kwargs(native, xyz, z, torch)
            prefix = {key: None if value is None else torch.cat((value, marker[key]), 0).contiguous()
                      for key, value in native.items()}
            _, _, prefix_map, _ = draw(prefix)
            result['prefix'] = 1 - prefix_map[2, y, x] / MARKER_ALPHA
        return result

    def terms(self, result, evidence, target, native_depth, sky_mask, union, args, iteration):
        from utils.hybrid30k_losses import (combined_opacity_entropy, lidar_decay_factor,
                                           photometric_rgb_loss, static_duration_loss, static_velocity_loss)
        from utils.loss_utils import ssim
        from utils.paper_losses import sky_loss
        import kornia
        visible, free, depth_valid = (evidence[key] for key in ('visible', 'free', 'depth_valid'))
        collar = F.max_pool2d(evidence['visible_reviewed'].any(0)[None, None].float(), 5, 1, 2)[0].bool()
        background = ~collar & (union == 0)
        rgb_loss = lambda image, mask: photometric_rgb_loss(guarded(image, mask), guarded(target, mask),
                                                            ssim, args.lambda_dssim)
        static_rgb = rgb_loss(result['render_s'], background)
        actor_rgb = [rgb_loss(layer[0], visible[i]) for i, layer in enumerate(result['isolated'])]
        iso_alpha = torch.cat([layer[2] for layer in result['isolated']], dim=0)
        iso_inv = torch.cat([layer[2] / layer[1].clamp_min(1e-5) for layer in result['isolated']], dim=0)
        static_alpha = result['static_alpha']
        static_inv = static_alpha.square() / result['static_depth'].clamp_min(1e-5) + (1-static_alpha) / 900
        native_depth = native_depth.to(target.device)
        bg_depth = average((static_inv - native_depth.clamp_min(1e-5).reciprocal()).abs(),
                           background & (native_depth > 0))
        depth = evidence['depth_m'] * self.scale
        actor_depth = average((iso_inv - depth.clamp_min(1e-5).reciprocal()).abs(), depth_valid)
        w = result['owner']
        joint_depth = average((w / result['owner_z'].clamp_min(1e-5) - depth.clamp_min(1e-5).reciprocal()).abs(),
                              depth_valid & (w.detach() > .1))
        coverage = ((1-iso_alpha).square() * visible + iso_alpha.square() * (free | evidence['safe_background'])).sum() / union.numel()
        ownership = average(-(w + .01).log(), visible) + average(w.square(), free)
        prefix = result['prefix'].square().mean() if result['prefix'].numel() else w.sum() * 0
        feature = result['static_feature'] / static_alpha.clamp_min(1e-5)
        smooth_depth = kornia.losses.inverse_depth_smoothness_loss(guarded(static_inv, background)[None], target[None])
        smooth_velocity = kornia.losses.inverse_depth_smoothness_loss(guarded(feature[1:], background)[None], target[None])
        return {
            'static_rgb': (1 + args.hybrid_lambda_static_rgb) * static_rgb,
            'rgb_a6': args.hybrid_lambda_dynamic_rgb * actor_rgb[0],
            'rgb_a37': args.hybrid_lambda_dynamic_rgb * actor_rgb[1],
            'coverage': args.hybrid_lambda_dynamic_opacity * coverage,
            'ownership': ownership,
            'static_prefix': .5 * prefix,
            'depth_background': args.paper_lambda_depth * lidar_decay_factor(iteration, args.lidar_decay) * bg_depth,
            'depth_actor': .10 * actor_depth,
            'depth_joint': .10 * joint_depth,
            'sky': args.paper_lambda_sky * sky_loss(static_alpha, background & sky_mask),
            'duration': args.lambda_t_reg * static_duration_loss(guarded(feature[:1], background), args.hybrid_time_reg_eps),
            'velocity': args.lambda_v_reg * static_velocity_loss(guarded(feature[1:], background)),
            'smooth_depth': args.lambda_inv_depth * smooth_depth,
            'smooth_velocity': args.lambda_v_smooth * smooth_velocity,
            'background_entropy': args.lambda_opacity_entropy * combined_opacity_entropy(guarded(static_alpha, background)),
        }


    @staticmethod
    def audit_gradients(terms, static, dynamic, result, args, iteration):
        import json

        ids = actor_ids(dynamic)
        retained = (result['static_growth'], result['static_growth_t'], result['dynamic_growth'])
        audit = {}
        for name in ('rgb_a6', 'rgb_a37', 'static_rgb', 'ownership', 'static_prefix'):
            inputs = (static._features_dc, dynamic._features_dc, retained[0], retained[1], static._opacity)
            gradients = torch.autograd.grad(terms[name], inputs, allow_unused=True, retain_graph=True)
            gs, gd, xy, t, opa = gradients

            def mag(value):
                return 0.0 if value is None or value.numel() == 0 else value.abs().max().item()

            audit[name] = dict(static_SH=mag(gs), a6_SH=mag(None if gd is None else gd[ids == 6]),
                               a37_SH=mag(None if gd is None else gd[ids == 37]),
                               growth_xy=mag(xy), growth_t=mag(t), static_opacity=mag(opa))
            for value in retained:
                value.grad = None
        for source, forbidden in {
            'rgb_a6': ('static_SH', 'a37_SH', 'growth_xy', 'growth_t'),
            'rgb_a37': ('static_SH', 'a6_SH', 'growth_xy', 'growth_t'),
            'static_rgb': ('a6_SH', 'a37_SH'),
            'ownership': ('static_SH', 'a6_SH', 'a37_SH', 'growth_xy', 'growth_t'),
            'static_prefix': ('static_SH', 'a6_SH', 'a37_SH', 'growth_xy', 'growth_t'),
        }.items():
            if any(audit[source][field] != 0 for field in forbidden):
                raise RuntimeError(f'Unexpected gradient path {source}: {audit[source]}')
        with (Path(args.model_path) / f'gradient_routing_{iteration:05d}.json').open('w') as handle:
            json.dump(audit, handle, indent=2, allow_nan=False)

