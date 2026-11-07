"""冻结正式 v44 底座，仅追加 Full5 的 a5/a18/a26 原生动态行。

这是一种显式混合表示：a6 仍为原 C2/SurfaceLOD，a37 仍走原 v44 路径。
不迁移旧 static gates，不改训练、权重、颜色场或像素，不推断新 actor 标签。
"""
from pathlib import Path
import hashlib
import json
import sys

from .ownership import actor_ids, install_identity_serialization
from .scene_io import checkpoint_actors, camera_headers


CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
V44_ROOT = CAMPAIGN / 'scene008_v44_rgb_asset_20261107_r1'
FULL5_MODEL = CAMPAIGN / 'scene008_full_dynamic_edit_20261107_r1/inference10k'
V44_RUNTIME_SHA256 = '245336e5bedeb2e994d690c14df88e53c8c1303242b626b636bc7ad9de718ad9'
ADDED_ACTOR_IDS = (5, 18, 26)
GROUPS = ('asset', 'a37', 'road', 'native_static', 'a5', 'a18', 'a26')
FEATURE_COUNT = 24


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _identity(path):
    value = Path(path).stat()
    return (value.st_size, value.st_mtime_ns)


def _tensors(model):
    return {name: value for name, value in vars(model).items()
            if hasattr(value, 'detach') and hasattr(value, 'data_ptr')}


def _versions(model):
    return {name: (value.data_ptr(), tuple(value.shape), str(value.dtype), value._version)
            for name, value in _tensors(model).items()}


class Full5V44Integration:
    """先正常构造 V44Runtime，再传入本适配器；每个 worker 仅构造一套底座。

    render_scene(raw, delete=False) 只改变原 v44 的 a37 存在状态。
    输出七组 joint alpha*T 以及三个新增 actor 的 joint RGB*T；不伪造四组 RGB。
    """

    def __init__(self, baseline, full5_model=FULL5_MODEL, iteration=10000,
                 full5_transform_path=None):
        self.base = baseline
        self.model_path = Path(full5_model).resolve()
        self.iteration = int(iteration)
        if self.iteration != 10000:
            raise ValueError('此固定 Scene008 接入只接受已验收 Full5 10K')
        module = sys.modules[type(baseline).__module__]
        runtime_path = Path(module.__file__).resolve()
        if runtime_path != V44_ROOT / 'source/v44_runtime.py' or _sha(runtime_path) != V44_RUNTIME_SHA256:
            raise ValueError('必须传入已展示的正式 v44，不能使用 R0 候选或另一套 runtime')
        if (Path(baseline.config.contract_path).resolve() != V44_ROOT / 'contract/input_contract.json'
                or baseline.config.sampler != 'surface_lod' or not baseline.config.transfer_fine):
            raise ValueError('v44 合同、C2 SurfaceLOD 与冻结 fine transfer 必须保持正式版本')
        self._runtime_module = module
        self.np, self.torch = baseline.np, baseline.torch
        torch, np = self.torch, self.np
        from omegaconf import OmegaConf
        from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
        import scene.gaussian_model as native_module
        if Path(native_module.__file__).resolve() != baseline.config.path('engine') / 'scene/gaussian_model.py':
            raise ValueError('Full5 与 v44 未加载同一个已冻结原生引擎')

        # 所有坐标/相机/时间元数据精确匹配；不用点云或 actor mask 选身份。
        config_path = self.model_path / 'effective_config.json'
        checkpoint_path = self.model_path / f'dynamic_chkpnt{self.iteration}.pth'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        self.full5_pipe = OmegaConf.create(config)
        self.headers = camera_headers(self.model_path)
        for name in ['cameras.json', 'actual_scene_binding.json']:
            if json.loads((self.model_path / name).read_text()) != json.loads((baseline.parent / name).read_text()):
                raise ValueError('Full5 与 v44 的49帧相机、时间或锚点不一致：' + name)
        transform_path = (Path(full5_transform_path).resolve() if full5_transform_path is not None
                          else Path(config['source_path']) / 'transform_matrix.txt')
        transform = np.loadtxt(transform_path)
        expected = np.asarray(baseline.exp.meta['world_to_model'])
        if not np.array_equal(transform, expected):
            raise ValueError('Full5 world_to_model 与 v44 不一致，不能直接拼接高斯')
        units = np.linalg.norm(transform[:3, :3], axis=0)
        if not np.allclose(units, baseline.scale, atol=1e-14, rtol=0):
            raise ValueError('Full5 与 v44 的米制比例不同')
        self.coordinate_binding = dict(saved_cameras_49_exact=True, saved_timestamps_and_anchor_exact=True,
            world_to_model_exact=True, world_to_model=transform.tolist(), meter_to_model=float(baseline.scale),
            transform_metadata_path=str(transform_path), transform_sha256=_sha(transform_path))

        info = checkpoint_actors(self.model_path, self.iteration)
        if info['dynamic_actor_ids'] != [5, 6, 18, 26, 37]:
            raise ValueError('Full5 checkpoint 必须自带完整五身份')
        # 只给私有子类安装新恢复方法；原 v44 的共享类/旧 lineage hook 均不改。
        shared_methods = (Dynamic_GaussianModel.capture, Dynamic_GaussianModel.restore,
                          GaussianModel.capture, GaussianModel.restore)
        class IdentityDynamic(Dynamic_GaussianModel):
            pass
        install_identity_serialization(GaussianModel, IdentityDynamic)
        with torch.no_grad():
            self.dynamic = IdentityDynamic(self.full5_pipe)
            payload, step = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False, mmap=True)
            if int(step) != self.iteration:
                raise ValueError('Full5 checkpoint step 不匹配')
            values = list(payload)
            for index in range(1, 11):
                values[index] = values[index].detach().cuda()
            self.dynamic.time_duration = tuple(config['time_duration'])
            self.dynamic.restore(tuple(values), training_args=None)
            if getattr(self.dynamic, 'optimizer', None) is not None:
                raise ValueError('接入只能推理，不能创建优化器')
            for value in _tensors(self.dynamic).values():
                value.requires_grad_(False)
            self.owners = actor_ids(self.dynamic).detach().clone()
            if not torch.equal(self.owners.cpu(), payload[14]['dynamic_actor_ownership_v1']['actor_ids'].cpu()):
                raise ValueError('恢复后的 Full5 actor ID 与 checkpoint 不同')
            tensor_fields = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation',
                             '_opacity', '_t', '_scaling_t', '_velocity', 'max_radii2D']
            for index, name in enumerate(tensor_fields, 1):
                actual = getattr(self.dynamic, name)
                if (actual.dtype != payload[index].dtype or actual.shape != payload[index].shape
                        or not torch.equal(actual.cpu(), payload[index].cpu())):
                    raise ValueError('Full5 参数恢复发生数值变化')
            for index, name in [(0, 'active_sh_degree'), (15, 'spatial_lr_scale'),
                                (16, 'T'), (17, 'velocity_decay')]:
                if getattr(self.dynamic, name) != payload[index]:
                    raise ValueError('Full5 原生标量恢复发生变化：' + name)
            self.added_mask = torch.zeros_like(self.owners, dtype=torch.bool)
            for actor in ADDED_ACTOR_IDS:
                self.added_mask |= self.owners == actor
            self.owner_rows = {str(actor): int((self.owners == actor).sum()) for actor in info['dynamic_actor_ids']}
            self.added_rows = {str(actor): self.owner_rows[str(actor)] for actor in ADDED_ACTOR_IDS}
            if not all(self.added_rows.values()):
                raise ValueError('新增 actor 在此 checkpoint 中无真实行')
            del payload, values
        if shared_methods != (Dynamic_GaussianModel.capture, Dynamic_GaussianModel.restore,
                              GaussianModel.capture, GaussianModel.restore):
            raise RuntimeError('不允许修改 v44 共享模型类的 capture/restore')
        self._shared_classes = (Dynamic_GaussianModel, GaussianModel)
        self._shared_methods = shared_methods
        self._before_versions = _versions(self.dynamic)
        self._before_hash = module.tensor_hash(_tensors(self.dynamic))
        self._input_files = [runtime_path, config_path, checkpoint_path, transform_path,
                             self.model_path / 'cameras.json', self.model_path / 'actual_scene_binding.json']
        self._file_identities = {str(path): _identity(path) for path in self._input_files}
        self.render_calls = 0
        self.camera_checks = {}
        baseline.assert_unchanged(full=True)

    def __getattr__(self, name):
        return getattr(self.__dict__['base'], name)

    def assert_unchanged(self, full=False):
        self.base.assert_unchanged(full=full)
        if _versions(self.dynamic) != self._before_versions:
            raise RuntimeError('Full5 动态模型参数/身份/谱系被改写')
        dynamic_class, gaussian_class = self._shared_classes
        if self._shared_methods != (dynamic_class.capture, dynamic_class.restore,
                                    gaussian_class.capture, gaussian_class.restore):
            raise RuntimeError('共享原生类方法被替换')
        for path, identity in self._file_identities.items():
            if _identity(path) != identity:
                raise RuntimeError('冻结输入文件发生变化：' + path)
        if full:
            if self._runtime_module.tensor_hash(_tensors(self.dynamic)) != self._before_hash:
                raise RuntimeError('Full5 动态模型内容发生变化')
            if not self.torch.equal(actor_ids(self.dynamic), self.owners):
                raise RuntimeError('Full5 身份发生变化')
        return True

    def _added_native_rows(self, camera):
        rows = self.base.prepare(camera, self.dynamic, self.full5_pipe, dynamic=True)
        if rows['colors_precomp'] is None:
            direction = rows['means3D'] - camera.camera_center
            direction = direction / direction.norm(dim=-1, keepdim=True)
            colour = (self.base.eval_sh(self.dynamic.active_sh_degree,
                      rows['shs'].transpose(1, 2), direction) + .5).clamp_min(0)
        else:
            colour = rows['colors_precomp']
        rows = dict(rows, means2D=self.torch.zeros_like(rows['means3D']), shs=None, colors_precomp=colour)
        return {key: None if value is None else value[self.added_mask].contiguous()
                for key, value in rows.items()}

    def scene(self, raw, delete=False):
        if type(delete) is not bool:
            raise TypeError('delete 必须为 bool，仅表示原 v44 的 a37 状态')
        self.assert_unchanged()
        torch, np = self.torch, self.np
        with torch.no_grad():
            scene = self.base.scene(raw, delete=delete)
            camera, original = scene['cam'], scene['rows']
            header = self.headers[raw]
            view_error = float(np.abs(camera.world_view_transform.detach().cpu().numpy().T - header['W']).max())
            if view_error > 1e-6 or float(camera.timestamp) != header['timestamp']:
                raise ValueError('实际 v44 Camera 与 Full5 保存的相机/时间不匹配')
            intrinsic_error = max(abs(float(getattr(camera, name)) - header[name])
                                  for name in ['fx', 'fy', 'cx', 'cy'])
            for name in ['fx', 'fy', 'cx', 'cy']:
                if abs(float(getattr(camera, name)) - header[name]) > 1e-9:
                    raise ValueError('实际相机内参不匹配：' + name)
            self.camera_checks[str(raw)] = dict(world_view_max_abs_error=view_error,
                actual_timestamp_exact=True, actual_intrinsics_max_abs_error=intrinsic_error,
                saved_camera_and_world_transform_exact=True)
            # 保留原四组的 alpha 标签；不把新增车辆混标为 a37 或 static。
            feature = torch.zeros_like(original['features'])
            if feature.shape[1] != FEATURE_COUNT:
                raise ValueError('正式 v44 feature ABI 发生变化')
            feature[:, :6] = original['features'][:, :6]
            base_rows = dict(original, features=feature)
            added = self._added_native_rows(camera)
            added_owners = self.owners[self.added_mask]
            feature = added['features'].new_zeros((len(added_owners), FEATURE_COUNT))
            feature[:, :2] = added['features'][:, :2]
            for index, actor in enumerate(ADDED_ACTOR_IDS):
                selected = added_owners == actor
                feature[selected, 6 + index] = 1
                feature[selected, 9 + 3 * index:12 + 3 * index] = added['colors_precomp'][selected]
            added = dict(added, features=feature)
            joint = self.base.merge(base_rows, added)
            count = len(original['mask'])
            for name, value in original.items():
                if name != 'features' and value is not None and not torch.equal(joint[name][:count], value):
                    raise RuntimeError('原 v44 行前缀被改写：' + name)
            meta = dict(scene['meta'], groups=list(GROUPS), delete_actor_ids=[37] if delete else [],
                full5_dynamic_actor_ids=[5, 6, 18, 26, 37], full5_added_actor_ids=list(ADDED_ACTOR_IDS),
                full5_added_rows=self.added_rows, full5_native_a6_rows_added=0, full5_native_a37_rows_added=0,
                frozen_v44_geometric_row_prefix_exact=True, old_static_gate_indices_not_transferred=True,
                full5_native_pipe_used=True, actual_camera_binding=self.camera_checks[str(raw)],
                representation='original_v44_static_env_road_fine_C2_a37_plus_Full5_a5_a18_a26',
                old_static_learned_vehicle_content_may_overlap=True,
                attribution='one native shared depth sort; seven alphaT plus three added actor RGBT',
                original_four_group_RGB_not_recorded=True, screen_mask_or_geometry_based_owner_guess=False)
            return dict(scene, rows=joint, meta=meta)

    def render(self, raw, scene):
        torch = self.torch
        with torch.no_grad():
            out = self.base.raster(scene['cam'], self.base.pipe, 0, scene['rows'], scene['bg'])
            self.render_calls += 1
            if out['extra'].shape[0] < 16:
                raise RuntimeError('原生 feature ABI 不能承载七组 alpha 和新增 RGB')
            out['rgb_premultiplied'] = out['rgb']
            out['group_alpha_T'] = out['extra'][:7]
            out['component_alpha_T'] = out['group_alpha_T']
            out['added_owner_rgb_T'] = out['extra'][7:16].reshape(3, 3, *out['alpha'].shape[-2:])
            out['added_owner_rgb_actor_ids'] = list(ADDED_ACTOR_IDS)
            out['env_rgb_T'] = (1 - out['alpha']) * scene['env']
            out['constant_background_rgb_T'] = (1 - out['alpha']) * scene['bg'][:, None, None]
            out['rgb'] = out['rgb_premultiplied'] + out['env_rgb_T']
            error = float((out['group_alpha_T'].sum(0) - out['alpha'][0]).abs().max())
            if error >= 2e-5:
                raise RuntimeError('七组 joint alphaT 不守恒：' + str(error))
            if not all(bool(torch.isfinite(out[key]).all()) for key in
                       ['rgb', 'alpha', 'depth_sum', 'extra', 'added_owner_rgb_T']):
                raise RuntimeError('原生 joint 渲染非有限值')
            out['group_conservation_maxabs'] = error
            out['meta'] = scene['meta']
            return out

    def render_scene(self, raw, delete=False):
        return self.render(raw, self.scene(raw, delete=delete))

    def decompose(self, raw, delete=False):
        return self.render_scene(raw, delete=delete)

    def receipt(self):
        self.assert_unchanged(full=True)
        return dict(status='PASS_V44_FULL5_METADATA_INTEGRATION_ENGINEERING',
            frozen_v44=self.base.receipt(), full5_model=str(self.model_path), iteration=self.iteration,
            dynamic_actor_ids=[5, 6, 18, 26, 37], source_owner_schema='dynamic_actor_ownership_v1',
            current_owner_rows=self.owner_rows, added_actor_ids=list(ADDED_ACTOR_IDS), added_rows=self.added_rows,
            Full5_static_or_environment_loaded=False, Full5_a6_or_a37_added=False,
            source_model_tensor_hash_unchanged=self._before_hash, coordinate_binding=self.coordinate_binding,
            camera_checks=self.camera_checks, shared_native_class_methods_unchanged=True,
            full5_restore_all_parameters_and_current_owner_ids_exact=True,
            all_original_v44_parameters_assets_road_gates_and_fine_preserved=True,
            old_static_gates_applied_only_to_original_static=True, feature_count=FEATURE_COUNT,
            joint_alphaT_groups=list(GROUPS), added_owner_RGBT_actor_ids=list(ADDED_ACTOR_IDS),
            original_four_group_RGB_available=False, render_calls=self.render_calls,
            original_static_vehicle_overlap_requires_visual_pilot=True,
            gradient_updates=0, optimizer_created=False, colour_fit_steps=0, calibration_solves=0,
            quality_accepted=False, human_verdict=None)
