"""V4 独立道路：固定几何，只训练 DC/opacity；不持有 V1 参数。"""

from collections import deque
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VERSION = 'GN11_V4_INDEPENDENT_ROAD_20261022'
INPUT_VERSION = 'V3_WORLD_ROAD_MULTIVIEW_R2_20261021'
C0 = .28209479177387814
POLICY = dict(cell_m=.2, normal_sigma_m=.015, tangent_sigma_cells=.65,
              initial_color=[.18, .18, .18], initial_opacity=.5,
              coverage_target=.95, coverage_weight=.1, prior_weight=.1,
              robust_delta=.05, source_batch_size=2, minimum_views=2,
              dc_lr=.0025, opacity_lr=.005, log_interval=100)
FRAMES = [f'0008{raw:03d}' for raw in range(89, 138)]


def sample_map(image, uv):
    height, width = image.shape[-2:]
    grid = torch.nan_to_num(uv).clone()
    grid[:, 0] = 2*grid[:, 0]/max(width-1, 1)-1
    grid[:, 1] = 2*grid[:, 1]/max(height-1, 1)-1
    return F.grid_sample(image[None], grid[None, :, None], align_corners=True,
                         mode='bilinear', padding_mode='zeros')[0, :, :, 0].T


def connected_to_seeds(eligible, edges, seeds):
    """只沿受支持且非 UNKNOWN 的四邻边传播，不跨无支撑缺口。"""
    adjacency = [[] for _ in eligible]
    for a, b in edges:
        if eligible[a] and eligible[b]:
            adjacency[a].append(int(b))
            adjacency[b].append(int(a))
    reached = np.asarray(seeds, bool).copy() & eligible
    queue = deque(np.flatnonzero(reached).tolist())
    while queue:
        a = queue.popleft()
        for b in adjacency[a]:
            if not reached[b]:
                reached[b] = True
                queue.append(b)
    return reached


def classify_units(arrays):
    supported = np.asarray(arrays['surface_supported'], bool)
    weights = np.asarray(arrays['base_weight'])
    provenance = arrays['provenance']
    observed = supported & (provenance == 1) & ((weights > 0).sum(0) >= POLICY['minimum_views'])
    eligible = supported & ((provenance == 2) | observed)
    reached = connected_to_seeds(eligible, arrays['edges'], observed)
    return observed, reached & ~observed


def load_bundle(path):
    path = Path(path)
    meta = json.loads(path.read_text(encoding='utf-8'))
    if (meta.get('status') != 'CPU_MULTIVIEW_ROAD_INPUT_VERIFIED'
            or meta.get('version') != INPUT_VERSION or meta.get('frame_order') != FRAMES):
        raise ValueError('必须使用已核对的同49帧道路输入，不自动改用199帧')
    with np.load(path.parent/meta['units'], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    count = len(arrays['model_xyz'])
    weights = arrays['base_weight']
    if weights.shape != (49, count) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('非法道路来源权重')
    edges = arrays['edges']
    if edges.ndim != 2 or edges.shape[1] != 2 or ((edges < 0) | (edges >= count)).any():
        raise ValueError('非法道路邻接表')
    for key in ('model_xyz', 'normal_model', 'source_rgb', 'source_depth'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError('道路输入非有限值：'+key)
    gates = [value for key, value in arrays.items() if key.startswith('gate_')]
    if len(gates) != 9 or any(value.shape != weights.shape for value in gates):
        raise ValueError('缺少原道路九项来源门限')
    if ((weights > 0) & ~np.logical_and.reduce(gates)).any():
        raise ValueError('可靠来源不能越过几何/遮挡门限')
    if (arrays['source_rgb'].shape != (49, count, 3) or arrays['source_depth'].shape != weights.shape
            or (arrays['source_depth'][weights > 0] <= 0).any()
            or (np.linalg.norm(arrays['normal_model'][arrays['surface_supported']], axis=1) < .99).any()):
        raise ValueError('道路来源或法线形状/有效域不匹配')
    if not np.isfinite(meta['meter_to_model']) or meta['meter_to_model'] <= 0:
        raise ValueError('非法米制缩放')
    return meta, arrays


class RoadLayer(nn.Module):
    def __init__(self, arrays, scale, arm, device='cpu'):
        super().__init__()
        if arm not in ('R', 'RP'):
            raise ValueError('道路层只适用于 R/RP')
        self.arm = arm
        ids = np.flatnonzero(arrays['surface_supported'])
        observed, prior = classify_units(arrays)
        self.unit_count = len(observed)
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
        self.register_buffer('unit_ids', torch.as_tensor(ids, dtype=torch.long, device=device))
        self.register_buffer('xyz', tensor(arrays['model_xyz'][ids]))
        normal = F.normalize(tensor(arrays['normal_model'][ids]), dim=1)
        quaternion = torch.cat((1+normal[:, 2:], -normal[:, 1:2], normal[:, :1],
                                torch.zeros_like(normal[:, :1])), 1)
        quaternion[quaternion.norm(dim=1) < 1e-8] = tensor([0., 1., 0., 0.])
        self.register_buffer('rotation', F.normalize(quaternion, dim=1))
        sigma = tensor([POLICY['cell_m']*POLICY['tangent_sigma_cells']]*2
                       + [POLICY['normal_sigma_m']])*scale
        self.register_buffer('scaling', sigma.expand(len(ids), 3).clone())
        self.register_buffer('observed', torch.as_tensor(observed[ids], device=device))
        self.register_buffer('prior', torch.as_tensor(prior[ids] & (arm == 'RP'), device=device))
        self.register_buffer('source_seen', torch.zeros(len(ids), dtype=torch.bool, device=device))
        self.register_buffer('appearance_updated', torch.zeros(len(ids), dtype=torch.bool, device=device))
        initial = tensor(POLICY['initial_color']).expand(len(ids), 3)
        self.dc = nn.Parameter(((initial-.5)/C0)[:, None].clone())
        self.opacity = nn.Parameter(torch.full((len(ids), 1),
            float(np.log(POLICY['initial_opacity']/(1-POLICY['initial_opacity']))), device=device))
        self.register_buffer('steps', torch.zeros((), dtype=torch.long, device=device))

    @property
    def active(self):
        return self.observed | self.prior

    def output_mask(self):
        # 原始灰色不是已恢复的路面；先验还必须实际接收过颜色更新。
        return ((self.observed & self.source_seen) | (self.prior & self.appearance_updated)) & (self.steps > 0)

    def colors(self):
        return (self.dc[:, 0]*C0+.5).clamp_min(0)


class RoadTraining:
    def __init__(self, args, scene, first_iter):
        self.path = str(Path(args.v4_bundle).resolve())
        self.meta, self.arrays = load_bundle(self.path)
        self.arm = args.v4_arm
        self.scale = float(self.meta['meter_to_model'])
        self.layer = RoadLayer(self.arrays, self.scale, self.arm, scene.gaussians.get_xyz.device)
        self.optimizer = torch.optim.Adam([
            dict(params=[self.layer.dc], lr=POLICY['dc_lr'], name='road_dc'),
            dict(params=[self.layer.opacity], lr=POLICY['opacity_lr'], name='road_opacity')], eps=1e-15)
        self.sampler = random.Random(int(args.seed)+104729)
        self.stack = []
        self.cumulative_seconds = 0.
        self.output = Path(args.model_path)
        lookup = {Path(camera.image_path).parent.name+'/'+camera.image_name: camera
                  for camera in scene.getTrainCameras()}
        self.cameras = [lookup['image_0/'+frame] for frame in FRAMES]
        device = self.layer.xyz.device
        ids = self.layer.unit_ids.detach().cpu().numpy()
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
        self.weights = tensor(self.arrays['base_weight'][:, ids])*self.layer.observed[None]
        self.rgb = tensor(self.arrays['source_rgb'][:, ids])
        self.depth = tensor(self.arrays['source_depth'][:, ids])
        homogeneous = torch.cat((self.layer.xyz, torch.ones_like(self.layer.xyz[:, :1])), 1)
        self.uv = []
        for frame, camera in zip(FRAMES, self.cameras):
            intrinsic = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.]])
            if not np.allclose(intrinsic, self.meta['frames'][frame]['intrinsic'], atol=1e-3, rtol=1e-5):
                raise ValueError('原生相机与道路来源标定不匹配')
            expected_view = np.asarray(self.meta['frames'][frame]['world_to_camera']) @ np.linalg.inv(
                np.asarray(self.meta['world_to_model']))
            # 原始相机深度以米计；V1 Camera 在归一化模型坐标中输出模型深度。
            expected_view[:3] *= self.scale
            if not np.allclose(camera.world_view_transform.cpu().numpy(), expected_view.T, atol=2e-4, rtol=1e-5):
                raise ValueError('V1 相机与道路模型坐标不匹配')
            clip = homogeneous @ camera.full_proj_transform.to(homogeneous)
            uv = ((clip[:, :2]/clip[:, 3:].clamp_min(1e-8)+1)*homogeneous.new_tensor(
                [camera.image_width, camera.image_height])-1)/2
            self.uv.append(uv)
        reverse = np.full(self.layer.unit_count, -1, np.int64)
        reverse[ids] = np.arange(len(ids))
        edges = reverse[self.arrays['edges']]
        keep = (edges >= 0).all(1)
        edges = edges[keep]
        active = self.layer.active.cpu().numpy()
        prior = self.layer.prior.cpu().numpy()
        keep = active[edges].all(1) & prior[edges].any(1)
        self.edges = torch.as_tensor(edges[keep], dtype=torch.long, device=device)
        if first_iter:
            self.restore(self.output/f'road_chkpnt{first_iter}.pth', first_iter)
        self.write_json('v4_road_contract.json', dict(version=VERSION, arm=self.arm,
            input_bundle=self.path, policy=POLICY, points=len(ids),
            observed=int(self.layer.observed.sum()), connected_prior=int(self.layer.prior.sum()),
            unknown_supported=int((~self.layer.active).sum()),
            unsupported=int((~self.arrays['surface_supported']).sum()),
            V1_point_budget_unchanged=True, equal_steps_not_equal_compute=True,
            trainable_parameters=['dc', 'opacity'], geometry_fixed=True))

    def write_json(self, name, value):
        (self.output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    def sample(self):
        result = []
        for _ in range(POLICY['source_batch_size']):
            if not self.stack:
                self.stack = list(range(len(self.cameras)))
            result.append(self.stack.pop(self.sampler.randrange(len(self.stack))))
        return result

    def objective(self, frame_ids, render_fn):
        rgb_loss = self.layer.dc.sum()*0
        coverage = self.layer.opacity.sum()*0
        denominator = self.weights.sum().clamp_min(1e-8)
        factor = len(self.cameras)/len(frame_ids)
        depth_numerator = rgb_loss.detach()*0
        source_weight = 0.
        for index in frame_ids:
            package = render_fn(self.cameras[index], self.layer, self.args, for_training=True)
            if not all(torch.isfinite(package[key]).all() for key in ('rgb', 'alpha', 'depth_sum')):
                raise FloatingPointError('道路光栅输出非有限值')
            alpha = sample_map(package['alpha'], self.uv[index])[:, 0]
            predicted = sample_map(package['rgb'], self.uv[index])/alpha[:, None].clamp_min(1e-5)
            error = (torch.sqrt((predicted-self.rgb[index]).square()+POLICY['robust_delta']**2)
                     -POLICY['robust_delta']).mean(1)
            weight = self.weights[index]
            rgb_loss = rgb_loss+(error*weight).sum()/denominator*factor
            coverage = coverage+(F.relu(POLICY['coverage_target']-alpha).square()*weight).sum()/denominator*factor
            depth = sample_map(package['depth_sum'], self.uv[index])[:, 0]/alpha.clamp_min(1e-5)
            depth_numerator += (((depth.detach()-self.depth[index]).abs()/self.scale)*weight).sum()
            source_weight += float(weight.sum())
            self.layer.source_seen |= weight > 0
        continuity = self.layer.dc.sum()*0
        if self.arm == 'RP' and len(self.edges):
            color = self.layer.colors()
            continuity = (color[self.edges[:, 0]]-color[self.edges[:, 1]]).square().mean()
            opacity = self.layer.opacity.sigmoid()
            continuity = continuity+(opacity[self.edges[:, 0]]-opacity[self.edges[:, 1]]).square().mean()
        total = rgb_loss+POLICY['coverage_weight']*coverage+POLICY['prior_weight']*continuity
        return total, dict(rgb=float(rgb_loss.detach()), coverage=float(coverage.detach()),
            prior_continuity=float(continuity.detach()),
            depth_consistency_m=float(depth_numerator)/source_weight if source_weight else None)

    def step(self, iteration, render_fn=None):
        from utils.hybrid30k_runtime import preserve_rng
        if iteration != int(self.layer.steps)+1:
            raise ValueError('每个 V1 step 只允许一次道路 step，不追加步数')
        if render_fn is None:
            from utils.v4_road_render import render_road
            render_fn = render_road
        with preserve_rng(), torch.enable_grad():
            if self.layer.xyz.is_cuda:
                torch.cuda.synchronize(self.layer.xyz.device)
            started = time.perf_counter()
            frame_ids = self.sample()
            self.optimizer.zero_grad(set_to_none=True)
            total, terms = self.objective(frame_ids, render_fn)
            if not torch.isfinite(total):
                raise FloatingPointError('道路目标非有限值')
            total.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.layer.parameters()):
                raise FloatingPointError('道路梯度缺失或非有限值')
            changed = self.layer.dc.grad.detach().abs().flatten(1).sum(1) > 0
            self.optimizer.step()
            with torch.no_grad():
                self.layer.appearance_updated |= changed
                self.layer.steps.fill_(iteration)
            self.optimizer.zero_grad(set_to_none=True)
            # 同步计时只记道路成本，不改写 V1 原有迭代时间字段。
            if self.layer.xyz.is_cuda:
                torch.cuda.synchronize(self.layer.xyz.device)
            elapsed = time.perf_counter()-started
            self.cumulative_seconds += elapsed
            if iteration == 1 or iteration % POLICY['log_interval'] == 0:
                from utils.hybrid30k_runtime import append_json
                append_json(self.output/'v4_road_steps.jsonl', dict(iteration=iteration,
                    frame_ids=[FRAMES[i] for i in frame_ids], road_points=len(self.layer.xyz),
                    rendered_points=int(self.layer.output_mask().sum()), loss=float(total.detach()),
                    road_seconds=elapsed, road_seconds_cumulative=self.cumulative_seconds, **terms))
            if iteration in set(self.args.checkpoint_iterations) | set(self.args.save_iterations):
                self.save(iteration)

    def save(self, iteration):
        if int(self.layer.steps) != iteration:
            raise ValueError('不能发布尚未完成道路更新的节点')
        destination = self.output/f'road_chkpnt{iteration}.pth'
        temporary = destination.with_suffix('.partial')
        torch.save(dict(version=VERSION, policy=POLICY, arm=self.arm, bundle=self.path,
            iteration=iteration, layer=self.layer.state_dict(), optimizer=self.optimizer.state_dict(),
            sampler=self.sampler.getstate(), stack=list(self.stack),
            cumulative_seconds=self.cumulative_seconds), temporary)
        temporary.replace(destination)
        self.write_json(f'v4_checkpoint{iteration}.json', dict(version=VERSION, arm=self.arm,
            iteration=iteration, V1_updates=iteration, road_updates=iteration,
            road_checkpoint=destination.name, V1_checkpoint=f'chkpnt{iteration}.pth',
            road_points=len(self.layer.xyz), road_seconds_cumulative=self.cumulative_seconds))

    def restore(self, path, iteration):
        state = torch.load(path, map_location=self.layer.xyz.device)
        if (state['version'] != VERSION or state['policy'] != POLICY or state['arm'] != self.arm
                or state['bundle'] != self.path or state['iteration'] != iteration):
            raise ValueError('道路 sidecar 与 V1 checkpoint 不匹配')
        for name in ('xyz', 'scaling', 'rotation', 'unit_ids', 'observed', 'prior'):
            if not torch.equal(getattr(self.layer, name), state['layer'][name]):
                raise ValueError('固定道路初始化发生变化：'+name)
        self.layer.load_state_dict(state['layer'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.sampler.setstate(state['sampler'])
        self.stack = list(state['stack'])
        self.cumulative_seconds = float(state['cumulative_seconds'])
        if int(self.layer.steps) != iteration:
            raise ValueError('道路 step 计数不匹配')


def create_road_training(args, scene, first_iter=0):
    arm = args.get('v4_arm', 'B')
    if arm == 'B':
        return None
    if arm not in ('R', 'RP'):
        raise ValueError('未知 V4 实验组')
    from utils.hybrid30k_runtime import preserve_rng
    with preserve_rng():
        training = RoadTraining(args, scene, first_iter)
        training.args = args
        return training
