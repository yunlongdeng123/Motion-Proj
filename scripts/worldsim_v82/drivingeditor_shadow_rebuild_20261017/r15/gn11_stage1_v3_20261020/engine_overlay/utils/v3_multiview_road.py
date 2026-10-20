"""真实来源逐帧优化共享世界道路高斯；硬几何门限不能被可靠性学习反转。"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.v3_road_contract import ROAD_POLICY, ROAD_VERSION


def sample_map(image, uv):
    height, width = image.shape[-2:]
    grid = uv.clone()
    grid[:, 0] = 2*grid[:, 0]/(width-1)-1
    grid[:, 1] = 2*grid[:, 1]/(height-1)-1
    values = F.grid_sample(image[None], grid[None, :, None], mode='bilinear',
                           padding_mode='zeros', align_corners=True)
    return values[0, :, :, 0].T


def robust_rgb(prediction, target, delta=.05):
    return (torch.sqrt((prediction-target).square()+delta**2)-delta).mean(-1)


def updated_reliability(color_error, depth_error_m, policy=ROAD_POLICY):
    # detach + 下界：不能通过降低自身权重关闭难例；也不能把硬拒绝提升成真观测。
    color = 1/(1+(color_error.detach()/policy['reliability_color_scale']).square())
    depth = 1/(1+(depth_error_m.detach()/policy['depth_tolerance_m']).square())
    return (color*depth).clamp(min=policy['reliability_floor'], max=1.)


def quaternion_from_normal(normal):
    quaternion = torch.cat((1+normal[:, 2:], -normal[:, 1:2], normal[:, :1], torch.zeros_like(normal[:, :1])), 1)
    opposite = quaternion.norm(dim=1) < 1e-8
    quaternion[opposite] = normal.new_tensor([0., 1., 0., 0.])
    return F.normalize(quaternion, dim=1)


def install_road_units(model, arrays, args):
    if getattr(model, '_v3_road_ids', None) is not None:
        raise RuntimeError('fresh first-stage run must not install road primitives twice')
    if getattr(model, '_sgcm_lineage_ids', None) is not None:
        raise RuntimeError('install fixed road units before initializing SGCM lineage')
    supported = np.flatnonzero(arrays['surface_supported'])
    count, original = len(supported), len(model._xyz)
    if not count or original+count > int(args.densify_until_num_points):
        raise RuntimeError('world road units do not fit the unchanged total static point budget')
    tensor = lambda value: torch.as_tensor(value, dtype=model._xyz.dtype, device=model._xyz.device)
    scale = float(args.v3_road_meter_to_model)
    xyz = tensor(arrays['model_xyz'][supported])
    color = tensor(ROAD_POLICY['initial_color']).expand(count, 3)
    features = ((color-.5)/.28209479177387814)[:, None]
    rest = model._features_rest.new_zeros((count, *model._features_rest.shape[1:]))
    sigma = xyz.new_tensor([ROAD_POLICY['cell_m']*.65, ROAD_POLICY['cell_m']*.65,
                            ROAD_POLICY['normal_sigma_m']])*scale
    model.densification_postfix(
        xyz, features, rest, xyz.new_zeros(count, 1), sigma.log().expand(count, 3).clone(),
        quaternion_from_normal(tensor(arrays['normal_model'][supported])),
        xyz.new_full((count, 1), float(sum(args.time_duration)/2)),
        xyz.new_full((count, 1), np.log(2.)), xyz.new_zeros(count, 3))
    model._v3_road_ids = torch.cat((torch.full((original,), -1, device=xyz.device, dtype=torch.long),
                                   torch.as_tensor(supported, device=xyz.device, dtype=torch.long)))
    return count


def normal_from_quaternion(quaternion):
    w, x, y, z = F.normalize(quaternion, dim=1).unbind(1)
    return torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), 1)


def geometry_objective(model, arrays, scale, policy=ROAD_POLICY):
    rows = torch.nonzero(model.road_unit_mask, as_tuple=False).flatten()
    ids = model._road_ids()[rows]
    anchor, normal = arrays['model_xyz'][ids], arrays['normal_model'][ids]
    delta = (model._xyz[rows]-anchor)/scale
    surface = (delta*normal).sum(1).abs().mean()
    normal_loss = (1-(normal_from_quaternion(model._rotation[rows])*normal).sum(1).abs()).mean()
    sigma = model.get_scaling[rows]/scale
    thin = F.relu(sigma[:, 2]-policy['maximum_normal_sigma_m']).square().mean()
    # 固定单元身份，不靠时间变化迁移；位移/大尺度越域均受可解释的几何约束。
    anchor_loss = F.relu(delta.norm(dim=1)-policy['maximum_unit_drift_m']).square().mean()
    anchor_loss = anchor_loss+F.relu(sigma[:, :2]-policy['cell_m']).square().mean()
    total = (policy['surface_weight']*surface+policy['normal_weight']*normal_loss
             +policy['thin_weight']*thin+policy['anchor_weight']*anchor_loss)
    return total, dict(surface=surface, normal=normal_loss, thin=thin, anchor=anchor_loss)


class MultiViewRoad:
    def __init__(self, bundle_path, scene, args, output):
        self.path = Path(bundle_path)
        self.meta = json.loads(self.path.read_text(encoding='utf-8'))
        if (self.meta['status'] != 'CPU_MULTIVIEW_ROAD_INPUT_VERIFIED'
                or self.meta['version'] != ROAD_VERSION or self.meta['policy'] != ROAD_POLICY):
            raise RuntimeError('wrong multiview road contract')
        with np.load(self.path.parent/self.meta['units'], allow_pickle=False) as archive:
            self.numpy = {key: archive[key].copy() for key in archive.files}
        self.scale = float(self.meta['meter_to_model'])
        args.v3_road_meter_to_model = self.scale
        self.seed_count = install_road_units(scene.gaussians, self.numpy, args)
        device = scene.gaussians._xyz.device
        self.data = {key: torch.as_tensor(value, device=device) for key, value in self.numpy.items()
                     if key in {'model_xyz', 'normal_model', 'source_rgb', 'source_uv', 'source_depth',
                                'base_weight', 'provenance', 'is_query', 'edges', 'surface_supported'}}
        for key in ['model_xyz', 'normal_model']:
            self.data[key] = self.data[key].to(dtype=scene.gaussians._xyz.dtype)
        from utils.hybrid30k_runtime import camera_id
        cameras = {Path(camera_id(camera)).stem: camera for camera in scene.getTrainCameras()}
        self.cameras = [cameras[frame] for frame in self.meta['frame_order']]
        raster_uv = []
        homogeneous = torch.cat((self.data['model_xyz'], torch.ones_like(self.data['model_xyz'][:, :1])), 1)
        for frame, camera in zip(self.meta['frame_order'], self.cameras):
            intrinsic = np.asarray(self.meta['frames'][frame]['intrinsic'])
            actual = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.]])
            if not np.allclose(actual, intrinsic, atol=1e-3, rtol=1e-5):
                raise RuntimeError('source/raw calibration and actual native raster camera differ')
            clip = homogeneous @ camera.full_proj_transform.to(homogeneous)
            ndc = clip[:, :2]/clip[:, 3:].clamp_min(1e-8)
            # 使用当前原生光栅器的 NDC->pixel 约定；不把 raw RGB 的 uv
            # 无条件当成光栅坐标（原生可能存在半像素中心差）。
            raster_uv.append(((ndc+1)*homogeneous.new_tensor([camera.image_width, camera.image_height])-1)/2)
        self.raster_uv = torch.stack(raster_uv)
        self.exclusions = [dict(entry) for entry in args.get('v3_road_source_exclusions', [])]
        for exclusion in self.exclusions:
            frame = int(exclusion['source_raw'])-89
            if not 0 <= frame < 49:
                raise ValueError('counterfactual source is outside the legal49 window')
            keep = ((self.numpy['query_frame'] == int(exclusion['query_raw'])-89)
                    & (self.numpy['query_actor'] == int(exclusion['actor'])))
            ids = np.unique(self.numpy['query_unit_id'][keep])
            self.data['base_weight'][frame, torch.as_tensor(ids, device=device)] = 0
        self.reliability = torch.ones_like(self.data['base_weight'])
        shape = self.numpy['base_weight'].shape
        self.visits = np.zeros(shape, np.int32)
        self.gradient_l2_cumulative = np.zeros(shape, np.float64)
        self.gradient_samples = np.zeros(shape, np.int32)
        self.output = Path(output)
        from omegaconf import OmegaConf
        resolved = OmegaConf.to_container(args, resolve=True) if OmegaConf.is_config(args) else dict(vars(args))
        self.paired_training_config = {key: value for key, value in resolved.items()
                                       if key not in {'model_path', 'v3_road_source_exclusions'}}
        self.last_iteration = 0
        self.last_terms = {}

    def objective(self, model, args, iteration, weights):
        from gaussian_renderer import render_road_only
        return self.objective_with_renderer(model, args, iteration, weights, render_road_only)

    def objective_with_renderer(self, model, args, iteration, weights, render_fn):
        data, policy = self.data, ROAD_POLICY
        self.arm_weights = dict(weights)
        common, geometry_terms = geometry_objective(model, data, self.scale)
        rows = torch.nonzero(model.road_unit_mask, as_tuple=False).flatten()
        ids = model._road_ids()[rows]
        color = (model._features_dc[rows, 0]*.28209479177387814+.5).clamp_min(0)
        unit_color = color.new_zeros((len(data['provenance']), 3)).index_copy(0, ids, color)
        edges = data['edges'].long()
        prior = (data['provenance'] == 2) & data['is_query']
        edge_valid = (data['surface_supported'][edges].all(1) & prior[edges].any(1)
                      & (data['provenance'][edges] != 0).all(1))
        continuity = (unit_color[edges[:, 0]]-unit_color[edges[:, 1]]).square().mean(1)
        continuity = (continuity*edge_valid).sum()/edge_valid.sum().clamp_min(1)
        prior_alpha = model.get_opacity[rows, 0]
        prior_coverage = (F.relu(.95-prior_alpha).square()*prior[ids]).sum()/prior[ids].sum().clamp_min(1)
        prior_depth = ((model._xyz[rows]-data['model_xyz'][ids]).norm(dim=1)/self.scale*prior[ids]).sum()/prior[ids].sum().clamp_min(1)
        total = common+weights['prior_rgb']*(continuity+policy['coverage_weight']*prior_coverage)+weights['prior_depth']*prior_depth
        batch = [(iteration-1)*policy['source_batch_size']+index for index in range(policy['source_batch_size'])]
        frame_ids = [index % len(self.cameras) for index in batch]
        if getattr(args, 'v3_zero_update_gate', False):
            query_observed = ((data['provenance'] == 1) & data['is_query'])[None]
            scores = (data['base_weight']*query_observed).sum(1)
            frame_ids = torch.argsort(scores, descending=True, stable=True)[:policy['source_batch_size']].tolist()
        self.last_frames = frame_ids
        diagnostics = iteration == 1 or iteration % policy['audit_interval'] == 0
        raw_totals = {key: total.detach()*0 for key in ['mv_rgb', 'mv_depth', 'prior_context_rgb', 'coverage']}
        for frame_index in frame_ids:
            package = render_fn(self.cameras[frame_index], model, args)
            if not all(torch.isfinite(package[key]).all() for key in ['rgb', 'alpha', 'depth_sum']):
                raise FloatingPointError('nonfinite actual road raster output')
            uv = self.raster_uv[frame_index] if hasattr(self, 'raster_uv') else data['source_uv'][frame_index]
            alpha = sample_map(package['alpha'], uv)[:, 0]
            predicted = sample_map(package['rgb'], uv)/alpha[:, None].clamp_min(1e-5)
            depth = sample_map(package['depth_sum'], uv)[:, 0]/alpha.clamp_min(1e-5)
            color_error = robust_rgb(predicted, data['source_rgb'][frame_index], policy['robust_rgb_delta'])
            depth_error = (depth-data['source_depth'][frame_index]).abs()/self.scale
            reliability = updated_reliability(color_error, depth_error)
            self.reliability[frame_index] = reliability
            base = data['base_weight'][frame_index]
            observed = (data['provenance'] == 1) & data['is_query']
            boundary = (data['provenance'] == 1) & ~data['is_query']
            # 周期等频 source batch，F/B 保持全体来源加权目标的量级；
            # 分母用最近一次可靠性快照（stop-grad），不逐帧单独归一化。
            def source_term(error, domain):
                numerator = (error*base*reliability*domain).sum()
                denominator = (data['base_weight']*self.reliability*domain[None]).sum().clamp_min(1e-8)
                return numerator/denominator*(len(self.cameras)/len(frame_ids))
            rgb = source_term(color_error, observed)
            boundary_rgb = source_term(color_error, boundary)
            depth_loss = source_term(depth_error, observed)
            coverage = source_term(F.relu(.95-alpha).square(), observed)
            rgb_term = weights['observed_rgb']*rgb+weights['prior_rgb']*boundary_rgb
            total = total+rgb_term+weights['observed_depth']*depth_loss+weights['observed_rgb']*policy['coverage_weight']*coverage
            self.visits[frame_index] += ((base > 0) & ((observed & (weights['observed_rgb'] > 0))
                                                     | (boundary & (weights['prior_rgb'] > 0)))).detach().cpu().numpy()
            if diagnostics and rgb_term.requires_grad:
                proxy = package['proxy']
                old_gradient = None if proxy.grad is None else proxy.grad.clone()
                gradient = torch.autograd.grad(rgb_term, model._features_dc, retain_graph=True, allow_unused=True)[0]
                proxy.grad = old_gradient
                if gradient is not None:
                    norm = gradient[rows].flatten(1).norm(dim=1).detach().cpu().numpy()
                    supported_ids = ids.detach().cpu().numpy()
                    self.gradient_l2_cumulative[frame_index, supported_ids] += norm
                    self.gradient_samples[frame_index, supported_ids] += 1
            for key, value in [('mv_rgb', rgb), ('mv_depth', depth_loss), ('prior_context_rgb', boundary_rgb), ('coverage', coverage)]:
                raw_totals[key] += value.detach()
        if not torch.isfinite(total):
            raise FloatingPointError('nonfinite world road objective')
        self.last_iteration = iteration
        self.last_terms = {key: float(value.detach()) for key, value in {**geometry_terms, **raw_totals, 'prior_continuity': continuity}.items()}
        return total, self.last_terms

    def export(self, model, iteration, optimizer_updates):
        destination = self.output/'road_audit'/f'iteration_{iteration:06d}'
        destination.mkdir(parents=True, exist_ok=True)
        rows = torch.nonzero(model.road_unit_mask, as_tuple=False).flatten()
        ids = model._road_ids()[rows].detach().cpu().numpy()
        effective = self.data['base_weight']*self.reliability
        observed = (self.data['provenance'] == 1) & self.data['is_query']
        context = (self.data['provenance'] == 1) & ~self.data['is_query']
        coefficients = torch.zeros_like(effective)
        for mask, key in [(observed, 'observed_rgb'), (context, 'prior_rgb')]:
            masked = effective*mask[None]
            coefficients += self.arm_weights[key]*masked/masked.sum().clamp_min(1e-8)
        np.savez_compressed(destination/'optimization.npz', unit_ids=ids,
                            model_xyz=model._xyz[rows].detach().cpu().numpy(),
                            color=(model._features_dc[rows, 0]*.28209479177387814+.5).detach().cpu().numpy(),
                            reliability=self.reliability.detach().cpu().numpy(),
                            effective_base_weight=self.data['base_weight'].detach().cpu().numpy(),
                            effective_rgb_weight=coefficients.detach().cpu().numpy(),
                            objective_visits=self.visits, gradient_l2_cumulative=self.gradient_l2_cumulative,
                            gradient_samples=self.gradient_samples)
        metadata = dict(version=ROAD_VERSION, input_bundle=str(self.path), iteration=iteration,
                        optimizer_updates=optimizer_updates, source_exclusions=self.exclusions,
                        paired_training_config=self.paired_training_config,
                        gradient_definition='sampled isolated weighted road-RGB gradient L2 per road-unit DC parameter; not causal contribution',
                        diagnostics_interval=ROAD_POLICY['audit_interval'], arm_weights=self.arm_weights,
                        effective_weight_definition='last reliability snapshot, true provenance/domain and arm coefficient, full49 normalization; batch uses F/B factor',
                        counterfactual_required_for_causal_claim=True, quality_accepted=False)
        (destination/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return destination

    def capture_runtime_state(self):
        return dict(version=ROAD_VERSION, input_bundle=str(self.path), source_exclusions=self.exclusions,
                    reliability=self.reliability.detach().cpu().clone(),
                    visits=self.visits.copy(), gradient_l2_cumulative=self.gradient_l2_cumulative.copy(),
                    gradient_samples=self.gradient_samples.copy(), last_iteration=self.last_iteration,
                    last_frames=list(getattr(self, 'last_frames', [])),
                    arm_weights=dict(getattr(self, 'arm_weights', {})))

    def restore_runtime_state(self, state):
        if (state['version'] != ROAD_VERSION or state['input_bundle'] != str(self.path)
                or state['source_exclusions'] != self.exclusions):
            raise ValueError('road runtime state belongs to another input/source contract')
        if state['reliability'].shape != self.reliability.shape:
            raise ValueError('road runtime state has different world units or sources')
        self.reliability = state['reliability'].to(self.reliability).clone()
        for name, key in [('visits', 'visits'), ('gradient_l2_cumulative', 'gradient_l2_cumulative'),
                          ('gradient_samples', 'gradient_samples')]:
            value = np.asarray(state[key])
            if value.shape != self.visits.shape or not np.isfinite(value).all():
                raise ValueError('invalid road runtime statistic '+key)
            setattr(self, name, value.copy())
        if not torch.isfinite(self.reliability).all() or (self.reliability < ROAD_POLICY['reliability_floor']).any() or (self.reliability > 1).any():
            raise ValueError('invalid restored reliability')
        self.last_iteration = int(state['last_iteration'])
        self.last_frames = list(state['last_frames'])
        self.arm_weights = dict(state['arm_weights'])
