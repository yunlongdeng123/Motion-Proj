"""V4.1：V1 每步完成后，密集合成损失只更新道路 DC/opacity。"""

from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve
from utils.v4_road import RoadTraining, POLICY as BASE_POLICY, FRAMES, sample_map
from utils.v41_objectives import VERSION, POLICY, dense_objective, deletion_objective
from utils.v41_sources import EVIDENCE_VERSION, SOURCE_POLICY, validate_label_sources


class DenseRoadTraining(RoadTraining):
    def __init__(self, args, scene, first_iter=0):
        super().__init__(args, scene, 0)
        self.args = args
        self.dense_path = str(Path(args.v41_dense_manifest).resolve())
        meta = json.loads(Path(self.dense_path).read_text(encoding='utf-8'))
        if (meta['version'] != EVIDENCE_VERSION or meta['frame_order'] != FRAMES
                or meta['policy'].get('source_expansion') != SOURCE_POLICY
                or meta['source_bundle'] != self.path or not meta['dense_domain_ready']):
            raise ValueError('密集监督输入未通过49帧证据检查')
        self.mask_policy = meta['policy']
        self.dense = []
        for frame, record in zip(FRAMES, meta['frames']):
            with np.load(Path(self.dense_path).parent/'frames'/f'{frame}.npz', allow_pickle=False) as z:
                source_keys = ('deletion_source_ids','deletion_label_kind','deletion_legacy_valid',
                    'deletion_validation_source_ids','deletion_fit_count','deletion_fit_inliers',
                    'deletion_fit_span','deletion_validation_count','deletion_validation_inliers')
                row = {key: z[key].copy() for key in ('visible', 'protect', 'ssim_valid', 'unit_id', 'vehicle_body',
                    'vehicle_edge', 'deletion_domains', 'deletion_target_rgb', *source_keys)}
            validate_label_sources(row,len(self.dense))
            visible, protect, windows = [row[k] for k in ('visible', 'protect', 'ssim_valid')]
            if any(x.shape != (320, 480) or x.dtype != bool for x in (visible, protect, windows)):
                raise ValueError('密集掩码形状/类型错误')
            count = convolve(visible.astype(np.float32), np.ones((11, 11), np.float32), mode='constant', cval=0)
            interior = np.zeros_like(visible); interior[5:-5, 5:-5] = True
            expected_windows = (count >= 121*POLICY['ssim_min_valid_fraction']) & visible & interior
            if (not np.array_equal(protect, ~visible) or not np.array_equal(windows, expected_windows)):
                raise ValueError('保护域或 SSIM 窗口越界')
            if visible.sum() != record['visible_pixels'] or windows.sum() != record['ssim_windows']:
                raise ValueError('密集证据计数不匹配')
            if not visible.any() or not windows.any():
                raise ValueError('不能用无有效密集监督的帧冒充通过')
            bodies, edges = row['vehicle_body'], row['vehicle_edge']
            if any(x.shape != (6, 320, 480) or x.dtype != bool for x in (bodies, edges)):
                raise ValueError('车身域须为五身份加其他车辆')
            vehicle = bodies.any(0) | edges.any(0)
            if (vehicle & ~protect).any() or (vehicle & visible).any():
                raise ValueError('车身域未被保护或进入道路真值')
            ids = np.unique(row['unit_id'][visible])
            eligible = np.flatnonzero(self.arrays['surface_supported'] & (self.arrays['provenance'] == 1)
                                      & ((self.arrays['base_weight'] > 0).sum(0) >= 2))
            if not np.isin(ids, eligible).all():
                raise ValueError('未知或先验单元不能获得真实 RGB 标签')
            deletion = row['deletion_domains']; deletion_valid = deletion.any(0)
            if (deletion.shape != (5, 320, 480) or deletion.dtype != bool
                    or row['deletion_target_rgb'].shape != (3, 320, 480)
                    or not np.isfinite(row['deletion_target_rgb']).all()
                    or (deletion_valid & ~vehicle).any()
                    or not np.isin(row['unit_id'][deletion_valid], eligible).all()):
                raise ValueError('删车监督须来自已验收跨帧来源和已观测 R 几何')
            for key in source_keys:del row[key]
            ids = np.union1d(ids, row['unit_id'][deletion_valid])
            row['observed_rows'] = torch.as_tensor(np.flatnonzero(np.isin(self.layer.unit_ids.cpu().numpy(), ids)))
            for key in ('visible', 'protect', 'ssim_valid', 'vehicle_body', 'vehicle_edge', 'deletion_domains', 'deletion_target_rgb'):
                row[key] = torch.from_numpy(row[key])
            del row['unit_id']
            self.dense.append(row)
        self.write_json('v41_road_contract.json', dict(version=VERSION, arm=self.arm,
            input_bundle=self.path, dense_manifest=self.dense_path, dense_policy=self.mask_policy,
            base_policy=BASE_POLICY, policy=POLICY, points=len(self.layer.xyz),
            observed=int(self.layer.observed.sum()), connected_prior=int(self.layer.prior.sum()),
            trainable_parameters=['dc', 'opacity'], geometry_fixed=True,
            joint_raster_training=True, V1_inputs_detached=True, equal_steps_not_equal_compute=True))
        if first_iter:
            self.restore(self.output/f'road_chkpnt{first_iter}.pth', first_iter)

    def write_json(self, name, value):
        # 父类初始化写出的旧合同立即由完整 V4.1 合同替换，不混用旧 manifest 名称。
        super().write_json(name.replace('v4_', 'v41_'), value)

    def coverage(self, index, package):
        alpha = sample_map(package['alpha'], self.uv[index])[:, 0]
        weight = self.weights[index]
        loss = (F.relu(BASE_POLICY['coverage_target']-alpha).square()*weight).sum()
        loss = loss/self.weights.sum().clamp_min(1e-8)*len(self.cameras)
        self.layer.source_seen |= weight > 0
        return loss

    def continuity(self):
        loss = self.layer.dc.sum()*0+self.layer.opacity.sum()*0
        if self.arm == 'RP' and len(self.edges):
            a, b = self.edges[:, 0], self.edges[:, 1]
            loss = (self.layer.colors()[a]-self.layer.colors()[b]).square().mean()
            loss = loss+(self.layer.opacity.sigmoid()[a]-self.layer.opacity.sigmoid()[b]).square().mean()
        return loss

    def step(self, iteration, static, dynamic, background, environment, render_fn=None, pair_fn=None, deleted_fn=None):
        from utils.hybrid30k_runtime import preserve_rng, append_json
        from utils.v4_road_render import render_road
        from utils.v41_render import render_training_pair, render_training_deleted
        if iteration != int(self.layer.steps)+1:
            raise ValueError('每个 V1 step 只能对应一次道路 step')
        render_fn = render_fn or render_road
        pair_fn = pair_fn or render_training_pair
        deleted_fn = deleted_fn or render_training_deleted
        with preserve_rng(), torch.enable_grad():
            device = self.layer.xyz.device
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            selected = self.sample()
            self.optimizer.zero_grad(set_to_none=True)
            logs, total_value = {}, 0.
            # 逐帧 backward，仅做一次道路 optimizer.step，避免同时保存两套全场景光栅反传缓存。
            for index in selected:
                camera = self.cameras[index]
                target = camera.original_image.to(device).detach()
                masks = self.dense[index]
                independent = render_fn(camera, self.layer, self.args, for_training=True)
                native, joint = pair_fn(camera, static, dynamic, self.layer, self.args, background, environment)
                tensors = [target, native, joint['rgb'], joint['alpha'], independent['rgb'], independent['alpha']]
                if any(not torch.isfinite(value).all() for value in tensors):
                    raise FloatingPointError('V4.1 前向含非有限值')
                loss, terms = dense_objective(joint['rgb'], native, independent, target,
                    masks['visible'].to(device), masks['protect'].to(device), masks['ssim_valid'].to(device),
                    masks['vehicle_body'].to(device), masks['vehicle_edge'].to(device))
                cover = self.coverage(index, independent)
                loss = loss+BASE_POLICY['coverage_weight']*cover
                if not torch.isfinite(loss):
                    raise FloatingPointError('V4.1 道路损失非有限')
                (loss/len(selected)).backward()
                total_value += float(loss.detach())/len(selected)
                terms['coverage'] = cover.detach()
                for key, value in terms.items():
                    logs[key] = logs.get(key, 0.)+float(value.detach())/len(selected)
                self.layer.source_seen[masks['observed_rows'].to(device)] = True
                del loss, terms, independent, joint, native
                # 完整图反传已释放，再做删车前向；同一采样、同一次 Adam step，无新增训练步。
                if masks['deletion_domains'].any():
                    deleted = deleted_fn(camera, static, self.layer, self.args, background, environment)
                    if any(not torch.isfinite(deleted[k]).all() for k in ('rgb', 'road_visible_alpha')):
                        raise FloatingPointError('删车前向含非有限值')
                    deletion_loss, deletion_terms = deletion_objective(deleted,
                        masks['deletion_target_rgb'].to(device), masks['deletion_domains'].to(device))
                    if not torch.isfinite(deletion_loss): raise FloatingPointError('删车损失非有限')
                    if iteration == 1 or iteration % BASE_POLICY['log_interval'] == 0:
                        gradients = torch.autograd.grad(deletion_loss, (self.layer.dc, self.layer.opacity),
                            retain_graph=True, allow_unused=False)
                        if any(not torch.isfinite(g).all() for g in gradients):
                            raise FloatingPointError('删车梯度非有限')
                        for name, gradient in zip(('dc', 'opacity'), gradients):
                            deletion_terms['deletion_'+name+'_gradient_l1'] = gradient.detach().abs().sum()
                    (deletion_loss/len(selected)).backward()
                    total_value += float(deletion_loss.detach())/len(selected)
                    for key, value in deletion_terms.items():
                        logs[key] = logs.get(key, 0.)+float(value)/len(selected)
                    del deleted, deletion_loss, deletion_terms
                else:
                    for key in ('deletion_rgb', 'deletion_effective_coverage', 'deletion_actual_contribution', 'deletion_pixels'):
                        logs.setdefault(key, 0.)
            prior = self.continuity()*BASE_POLICY['prior_weight']
            prior.backward()
            total_value += float(prior.detach())
            logs['weighted_prior'] = float(prior.detach())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.layer.parameters()):
                raise FloatingPointError('道路梯度缺失或非有限')
            logs['dc_gradient_l1'] = float(self.layer.dc.grad.detach().abs().sum())
            logs['opacity_gradient_l1'] = float(self.layer.opacity.grad.detach().abs().sum())
            changed = self.layer.dc.grad.detach().abs().flatten(1).sum(1) > 0
            self.optimizer.step()
            with torch.no_grad():
                self.layer.appearance_updated |= changed
                self.layer.steps.fill_(iteration)
            self.optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-started
            self.cumulative_seconds += elapsed
            if iteration == 1 or iteration % BASE_POLICY['log_interval'] == 0:
                append_json(self.output/'v41_road_steps.jsonl', dict(iteration=iteration,
                    frame_ids=[FRAMES[i] for i in selected], road_points=len(self.layer.xyz),
                    rendered_points=int(self.layer.output_mask().sum()), loss=total_value,
                    road_seconds=elapsed, road_seconds_cumulative=self.cumulative_seconds,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None, **logs))
            if iteration in set(self.args.checkpoint_iterations) | set(self.args.save_iterations):
                self.save(iteration)

    def state(self, iteration):
        if int(self.layer.steps) != iteration:
            raise ValueError('节点尚未完成道路更新')
        return dict(version=VERSION, policy=POLICY, base_policy=BASE_POLICY, arm=self.arm,
            bundle=self.path, dense_manifest=self.dense_path, mask_policy=self.mask_policy,
            iteration=iteration, layer=self.layer.state_dict(), optimizer=self.optimizer.state_dict(),
            sampler=self.sampler.getstate(), stack=list(self.stack), cumulative_seconds=self.cumulative_seconds)

    def save(self, iteration):
        payload = self.state(iteration)
        components = [self.output/f'{prefix}{iteration}.pth' for prefix in
            ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]
        if not all(p.is_file() and p.stat().st_size > 0 for p in components):
            raise ValueError('原生四件未完整保存，不发布完整 V4.1 节点')
        destination = self.output/f'road_chkpnt{iteration}.pth'
        temporary = destination.with_suffix('.partial')
        torch.save(payload, temporary); temporary.replace(destination)
        self.write_json(f'v41_checkpoint{iteration}.json', dict(version=VERSION, arm=self.arm,
            iteration=iteration, V1_updates=iteration, road_updates=iteration,
            files=[p.name for p in components]+[destination.name], road_points=len(self.layer.xyz),
            dense_manifest=self.dense_path, road_seconds_cumulative=self.cumulative_seconds))

    def restore(self, path, iteration):
        state = torch.load(path, map_location=self.layer.xyz.device, weights_only=False)
        expected = dict(version=VERSION, policy=POLICY, base_policy=BASE_POLICY, arm=self.arm,
            bundle=self.path, dense_manifest=self.dense_path, mask_policy=self.mask_policy, iteration=iteration)
        if any(state.get(k) != value for k, value in expected.items()):
            raise ValueError('V4.1 道路节点与当前方法/监督数据不一致；不能静默续用 V4')
        for name in ('xyz', 'scaling', 'rotation', 'unit_ids', 'observed', 'prior'):
            if not torch.equal(getattr(self.layer, name), state['layer'][name]):
                raise ValueError('固定几何/分类发生变化：'+name)
        self.layer.load_state_dict(state['layer'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.sampler.setstate(state['sampler']); self.stack = list(state['stack'])
        self.cumulative_seconds = float(state['cumulative_seconds'])
        if int(self.layer.steps) != iteration:
            raise ValueError('道路更新数与恢复节点不一致')


def create_road_training(args, scene, first_iter=0):
    from utils.hybrid30k_runtime import preserve_rng
    if args.v4_arm != 'R':
        raise ValueError('当前仅固定几何 V4.1-R；RP 与几何补全未进入执行范围')
    with preserve_rng():
        return DenseRoadTraining(args, scene, first_iter)


class SmokeStop(Exception):
    """完整节点保存后退出短程，不改变原30K学习率计划。"""


STOP_AFTER = None  # 执行器边界，不进入原生可恢复训练配置。


def after_step(args, iteration):
    if STOP_AFTER == iteration:
        if not (Path(args.model_path)/f'v41_checkpoint{iteration}.json').is_file():
            raise RuntimeError('smoke 不得停在无完整节点处')
        raise SmokeStop(iteration)
