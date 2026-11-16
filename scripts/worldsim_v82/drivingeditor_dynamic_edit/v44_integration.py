# Historical Scene008 replay only. Use v44_cli / SceneManifest for new scenes.
"""冻结正式 v44 底座，默认只追加 Full5 的 a5 原生动态行。

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
                 full5_transform_path=None, added_actor_ids=(5,),
                 retire_legacy_static_actor5=False, legacy_static_actor5_audit_path=None):
        self.base = baseline
        if type(retire_legacy_static_actor5) is not bool:
            raise TypeError('retire_legacy_static_actor5 必须为 bool')
        self.retire_legacy_static_actor5 = retire_legacy_static_actor5
        requested = tuple(added_actor_ids)
        if (not requested or any(type(actor) is not int for actor in requested)
                or len(set(requested)) != len(requested)
                or not set(requested).issubset(ADDED_ACTOR_IDS)):
            raise ValueError('新增 actor 必须是 [5,18,26] 的非空、无重复整数子集')
        self.added_actor_ids = tuple(sorted(requested))
        if retire_legacy_static_actor5 and self.added_actor_ids != (5,):
            raise ValueError('此隔离退役入口只接受追加 a5，其他车辆保持原 v44')
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
            for actor in self.added_actor_ids:
                self.added_mask |= self.owners == actor
            self.owner_rows = {str(actor): int((self.owners == actor).sum()) for actor in info['dynamic_actor_ids']}
            self.added_rows = {str(actor): self.owner_rows[str(actor)] for actor in self.added_actor_ids}
            if not all(self.added_rows.values()):
                raise ValueError('新增 actor 在此 checkpoint 中无真实行')
            del payload, values
        if shared_methods != (Dynamic_GaussianModel.capture, Dynamic_GaussianModel.restore,
                              GaussianModel.capture, GaussianModel.restore):
            raise RuntimeError('不允许修改 v44 共享模型类的 capture/restore')
        self.legacy_static_row_offset = 0
        static_count = len(baseline.n.static.get_xyz)
        self._legacy_retirement_mask = torch.zeros(static_count, dtype=torch.bool,
                                                   device=baseline.n.static.get_xyz.device)
        self._legacy_retirement_rows = torch.empty(0, dtype=torch.int64,
                                                   device=baseline.n.static.get_xyz.device)
        self._legacy_retirement_input_files = []
        self._legacy_retirement_proof = None
        self.retirement_checks = {}
        if retire_legacy_static_actor5:
            audit_path = (Path(legacy_static_actor5_audit_path).resolve()
                          if legacy_static_actor5_audit_path is not None else
                          CAMPAIGN / 'scene008_v44_full5_integration_20261107_r1/review/legacy_static_actor5/audit.json')
            audit = json.loads(audit_path.read_text())
            if (audit['status'] != 'PASS_EXACT_SOURCE_ROWS_AND_PERSISTENT_STATIC_ACTOR5_LINEAGE'
                    or Path(audit['native_parent']).resolve() != baseline.parent
                    or audit['current_static_rows'] != static_count
                    or audit['exact_actor5_sampled_rows'] != 1496
                    or audit['exact_actor5_unique_xyz_time'] != 1442
                    or not audit['initial_xyz_hash_matches_executed_training_assert']
                    or not audit['initial_time_hash_matches_executed_training_assert']
                    or not audit['after_a6_transfer_xyz_time_hashes_exact']):
                raise ValueError('旧 static 来源审计与实际 v44 不匹配')
            asset_path = Path(audit['evidence_asset']).resolve()
            if _sha(asset_path) != audit['evidence_asset_sha256']:
                raise ValueError('旧 a5 来源行资产变化')
            with np.load(asset_path, allow_pickle=False) as saved:
                seeds = saved['initial_static_lineage_ids'].copy()
                expected_rows = saved['current_static_retirement_rows'].copy()
            lineage = getattr(baseline.n.static, '_sgcm_lineage_ids', None)
            if not isinstance(lineage, torch.Tensor) or lineage.dtype != torch.int64 or lineage.shape != (static_count,):
                raise ValueError('旧 static 缺少可靠的逐行谱系')
            actual_lineage = lineage.detach().cpu().contiguous().numpy()
            if hashlib.sha256(actual_lineage.tobytes()).hexdigest() != audit['current_static_lineage_sha256']:
                raise ValueError('实际旧 static 谱系行序与已证来源资产不同')
            if seeds.dtype != np.int64 or len(seeds) != 1496 or len(np.unique(seeds)) != len(seeds):
                raise ValueError('旧 a5 初始来源 ID 集合无效')
            retired = np.isin(actual_lineage, seeds)
            retired_rows = np.flatnonzero(retired)
            if (not np.array_equal(retired_rows, expected_rows)
                    or len(retired_rows) != audit['current_static_actor5_descendant_rows']):
                raise ValueError('旧 a5 后代集合无法从实际谱系精确复现')
            checkpoint_path_static = Path(audit['static_checkpoint_identity']['path'])
            if (checkpoint_path_static != baseline.parent / f'chkpnt{baseline.iteration}.pth'
                    or _identity(checkpoint_path_static) !=
                    (audit['static_checkpoint_identity']['bytes'], audit['static_checkpoint_identity']['mtime_ns'])):
                raise ValueError('旧 static checkpoint 不是来源审计绑定的权重')
            gate_ids = baseline.exp.repair.ids.detach().cpu().numpy()
            intersection = len(np.intersect1d(gate_ids, retired_rows))
            if intersection != audit['road_gate_actor5_descendant_intersection']:
                raise ValueError('实际 road gates 与 static 来源集合不匹配')
            self._legacy_retirement_mask = torch.as_tensor(retired, device=lineage.device)
            self._legacy_retirement_rows = torch.as_tensor(retired_rows, device=lineage.device)
            self._legacy_retirement_input_files = [audit_path, asset_path]
            self._legacy_retirement_proof = dict(audit_path=str(audit_path), audit_sha256=_sha(audit_path),
                asset_path=str(asset_path), asset_sha256=_sha(asset_path), source_seed_rows=len(seeds),
                current_descendant_rows=len(retired_rows), road_gate_intersection=intersection,
                static_lineage_sha256=audit['current_static_lineage_sha256'],
                source_descendant_identity_is_not_semantic_ground_truth=True)
        self._shared_classes = (Dynamic_GaussianModel, GaussianModel)
        self._shared_methods = shared_methods
        self._before_versions = _versions(self.dynamic)
        self._before_hash = module.tensor_hash(_tensors(self.dynamic))
        self._input_files = [runtime_path, config_path, checkpoint_path, transform_path,
                             self.model_path / 'cameras.json', self.model_path / 'actual_scene_binding.json']
        self._input_files += self._legacy_retirement_input_files
        self._file_identities = {str(path): _identity(path) for path in self._input_files}
        self.render_calls = 0
        self.camera_checks = {}
        baseline.assert_unchanged(full=True)

    @property
    def legacy_static_actor5_retirement_mask(self):
        """原 static 行序的严格目标集合；返回副本，不允许调用方改内部筛选。"""
        return self._legacy_retirement_mask.clone()

    @property
    def legacy_static_actor5_retirement_rows(self):
        return self._legacy_retirement_rows.clone()

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

    def _added_row_actor_ids(self):
        return self.owners[self.added_mask]

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
            if self.retire_legacy_static_actor5:
                static_count = len(self._legacy_retirement_mask)
                if (not bool((original['features'][:static_count, 5] == 1).all())
                        or bool((original['features'][static_count:, 5] != 0).any())):
                    raise ValueError('旧 static 不再是原 v44 packet 的已绑定前缀')
                # base.scene 已先按原索引应用 gates。只相交临时渲染 mask，不改 opacity/参数/行序。
                expected = torch.zeros_like(original['mask'])
                expected[:static_count] = self._legacy_retirement_mask
                new_mask = original['mask'] & ~expected
                if not torch.equal(original['mask'] ^ new_mask, original['mask'] & expected):
                    raise RuntimeError('mask 差异不等于原活跃 mask 与严格来源后代集合的交集')
                base_rows['mask'] = new_mask
                check = dict(target_descendant_rows=int(expected.sum()),
                    originally_active_retired_rows=int((original['mask'] & expected).sum()),
                    actual_changed_mask_rows=int((original['mask'] ^ new_mask).sum()),
                    static_prefix_offset=0, unchanged_outside_strict_target=True,
                    old_gates_applied_before_mask=True)
                if str(raw) in self.retirement_checks and self.retirement_checks[str(raw)] != check:
                    raise RuntimeError('两编辑状态的静态退役集合/原活跃 mask 不一致')
                self.retirement_checks[str(raw)] = check
            added = self._added_native_rows(camera)
            added_owners = self._added_row_actor_ids()
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
                excepted = name == 'features' or (name == 'mask' and self.retire_legacy_static_actor5)
                if not excepted and value is not None and not torch.equal(joint[name][:count], value):
                    raise RuntimeError('原 v44 行前缀被改写：' + name)
            meta = dict(scene['meta'], groups=list(GROUPS), delete_actor_ids=[37] if delete else [],
                full5_dynamic_actor_ids=[5, 6, 18, 26, 37], full5_added_actor_ids=list(self.added_actor_ids),
                full5_added_rows=self.added_rows, full5_native_a6_rows_added=0, full5_native_a37_rows_added=0,
                frozen_v44_geometric_row_prefix_exact=True, old_static_gate_indices_not_transferred=True,
                original_v44_prefix_mask_unchanged=not self.retire_legacy_static_actor5,
                retire_legacy_static_actor5=self.retire_legacy_static_actor5,
                legacy_static_retirement=self.retirement_checks.get(str(raw)),
                legacy_static_actor5_source_proof=self._legacy_retirement_proof,
                full5_native_pipe_used=True, actual_camera_binding=self.camera_checks[str(raw)],
                representation='original_v44_static_env_road_fine_C2_a37_plus_selected_Full5_actors',
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
            out['active_added_actor_ids'] = list(self.added_actor_ids)
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
            current_owner_rows=self.owner_rows, added_actor_ids=list(self.added_actor_ids), added_rows=self.added_rows,
            Full5_static_or_environment_loaded=False, Full5_a6_or_a37_added=False,
            source_model_tensor_hash_unchanged=self._before_hash, coordinate_binding=self.coordinate_binding,
            camera_checks=self.camera_checks, shared_native_class_methods_unchanged=True,
            full5_restore_all_parameters_and_current_owner_ids_exact=True,
            all_original_v44_parameters_assets_road_gates_and_fine_preserved=True,
            retire_legacy_static_actor5=self.retire_legacy_static_actor5,
            legacy_static_actor5_source_proof=self._legacy_retirement_proof,
            legacy_static_retirement_checks=self.retirement_checks,
            original_v44_packet_mask_unchanged=not self.retire_legacy_static_actor5,
            old_static_gates_applied_only_to_original_static=True, feature_count=FEATURE_COUNT,
            joint_alphaT_groups=list(GROUPS), added_owner_RGBT_slot_actor_ids=list(ADDED_ACTOR_IDS),
            unselected_actor_feature_slots_zero=True,
            original_four_group_RGB_available=False, render_calls=self.render_calls,
            original_static_vehicle_overlap_requires_visual_pilot=True,
            gradient_updates=0, optimizer_created=False, colour_fit_steps=0, calibration_solves=0,
            quality_accepted=False, human_verdict=None)


class SemanticActor5V44Integration(Full5V44Integration):
    """显式 BUILD 可见贡献校准模式；原来源身份不覆写，旧 v44 static 不退役。

    static / dynamic 的固定行集合来自一个已封存 sidecar，而非查询帧 mask。
    估计语义5的行统一参加 a5 group，来源 actor/lineage 仍单独记录。
    """

    def __init__(self, baseline, semantic_sidecar_path, full5_model=FULL5_MODEL,
                 iteration=10000, full5_transform_path=None):
        self._semantic_ready = False
        super().__init__(baseline, full5_model=full5_model, iteration=iteration,
            full5_transform_path=full5_transform_path, added_actor_ids=(5,),
            retire_legacy_static_actor5=False)
        np, torch = self.np, self.torch
        manifest_path = Path(semantic_sidecar_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        build_raw_ids = manifest.get('raw_ids')
        valid_build_frames = (type(build_raw_ids) is list
                              and all(type(raw) is int for raw in build_raw_ids)
                              and build_raw_ids in ([94, 105, 130, 137], list(range(89, 138))))
        if (manifest.get('schema') != 'scene008.semantic_actor5.visible_mass.v1'
                or manifest.get('iteration') != self.iteration
                or not valid_build_frames
                or manifest.get('threshold') != dict(purity_strictly_greater_than=.5,
                                                    total_mass_strictly_greater_than=0.)
                or manifest.get('gaussian_parameter_updates') != 0):
            raise ValueError('语义 sidecar 的模型、BUILD帧或固定多数归属协议不匹配')
        selection = Path(manifest['selection_file'])
        selection = selection if selection.is_absolute() else manifest_path.parent / selection
        selection = selection.resolve()
        if _sha(selection) != manifest['selection_sha256']:
            raise ValueError('语义 sidecar 行/质量数组已变化')
        for name in [f'chkpnt{self.iteration}.pth', f'dynamic_chkpnt{self.iteration}.pth']:
            if _sha(self.model_path / name) != manifest['gaussian_files_sha256'][name]:
                raise ValueError('语义校准指向另一份 Full5 权重：' + name)

        from scene.gaussian_model import GaussianModel
        with torch.no_grad():
            self.semantic_static = GaussianModel(self.full5_pipe)
            payload, step = torch.load(str(self.model_path / f'chkpnt{self.iteration}.pth'),
                                      map_location='cpu', weights_only=False, mmap=True)
            if int(step) != self.iteration or len(payload) != 18:
                raise ValueError('语义模式 Full5 static checkpoint 不匹配')
            values = list(payload)
            for index in range(1, 11):
                values[index] = values[index].detach().cuda()
            self.semantic_static.time_duration = tuple(self.full5_pipe.time_duration)
            self.semantic_static.restore(tuple(values), training_args=None)
            fields = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation',
                      '_opacity', '_t', '_scaling_t', '_velocity', 'max_radii2D']
            for index, name in enumerate(fields, 1):
                actual = getattr(self.semantic_static, name)
                if (actual.dtype != payload[index].dtype or actual.shape != payload[index].shape
                        or not torch.equal(actual.cpu(), payload[index].cpu())):
                    raise ValueError('Full5 static 参数恢复发生数值变化：' + name)
            for index, name in [(0, 'active_sh_degree'), (15, 'spatial_lr_scale'),
                                (16, 'T'), (17, 'velocity_decay')]:
                if getattr(self.semantic_static, name) != payload[index]:
                    raise ValueError('Full5 static 标量恢复发生变化：' + name)
            if getattr(self.semantic_static, 'optimizer', None) is not None:
                raise ValueError('语义接入禁止创建优化器')
            for value in _tensors(self.semantic_static).values():
                value.requires_grad_(False)
            del payload, values
        self._semantic_static_versions = _versions(self.semantic_static)
        self._semantic_static_hash = self._runtime_module.tensor_hash(_tensors(self.semantic_static))
        selected = {}
        with np.load(selection, allow_pickle=False) as saved:
            for branch, model in [('static', self.semantic_static), ('dynamic', self.dynamic)]:
                count = len(model.get_xyz)
                if manifest['branch_rows'][branch] != count:
                    raise ValueError('语义 sidecar 高斯行数不匹配：' + branch)
                rows = saved[branch + '_row_ids'].copy()
                inside, total = saved[branch + '_inside_mass'], saved[branch + '_total_mass']
                if (rows.dtype != np.int64 or rows.ndim != 1 or not np.array_equal(rows, np.unique(rows))
                        or (len(rows) and (rows.min() < 0 or rows.max() >= count))
                        or inside.shape != (count,) or total.shape != (count,)
                        or not np.isfinite(inside).all() or not np.isfinite(total).all()
                        or (inside < 0).any() or (total < 0).any()
                        or (inside > total).any()):
                    raise ValueError('语义 sidecar 行号或可见质量无效：' + branch)
                purity = np.divide(inside, total, out=np.zeros_like(total), where=total > 0)
                expected = np.flatnonzero((total > 0) & (purity > .5))
                if not np.array_equal(rows, expected):
                    raise ValueError('语义行集合不等于预定 purity>.5 且 mass>0：' + branch)
                selected[branch] = torch.as_tensor(rows, dtype=torch.int64, device=model.get_xyz.device)
        self._semantic_static_rows = selected['static']
        self._semantic_dynamic_rows = selected['dynamic']
        self._semantic_static_mask = torch.zeros(len(self.semantic_static.get_xyz), dtype=torch.bool,
                                                device=self.semantic_static.get_xyz.device)
        self._semantic_static_mask[self._semantic_static_rows] = True
        self.added_mask = torch.zeros_like(self.owners, dtype=torch.bool)
        self.added_mask[self._semantic_dynamic_rows] = True
        self.semantic_branch_rows = {name: len(value) for name, value in selected.items()}
        if not sum(self.semantic_branch_rows.values()):
            raise ValueError('校准未找到任何已观测多数归属 a5 的高斯；禁止猜填')
        self.semantic_dynamic_source_counts = {str(actor): int((self.owners[self.added_mask] == actor).sum())
                                               for actor in [5, 6, 18, 26, 37]}
        self.added_rows = {'5': sum(self.semantic_branch_rows.values())}
        self._semantic_manifest = manifest
        self._semantic_manifest_path = manifest_path
        self._semantic_selection_path = selection
        self._semantic_files = [manifest_path, selection, self.model_path / f'chkpnt{self.iteration}.pth']
        self._semantic_file_identities = {str(path): _identity(path) for path in self._semantic_files}
        self._semantic_ready = True
        self.assert_unchanged(full=True)

    @property
    def semantic_static_row_ids(self):
        return self._semantic_static_rows.clone()

    @property
    def semantic_dynamic_row_ids(self):
        return self._semantic_dynamic_rows.clone()

    def assert_unchanged(self, full=False):
        super().assert_unchanged(full=full)
        if self.__dict__.get('_semantic_ready', False):
            if _versions(self.semantic_static) != self._semantic_static_versions:
                raise RuntimeError('Full5 static 参数或来源谱系被改写')
            for path, identity in self._semantic_file_identities.items():
                if _identity(path) != identity:
                    raise RuntimeError('语义 sidecar 或 Full5 static 输入变化：' + path)
            if full and self._runtime_module.tensor_hash(_tensors(self.semantic_static)) != self._semantic_static_hash:
                raise RuntimeError('Full5 static 参数内容变化')
        return True

    def _added_native_rows(self, camera):
        rows = self.base.prepare(camera, self.semantic_static, self.full5_pipe, dynamic=False)
        if rows['colors_precomp'] is None:
            direction = rows['means3D'] - camera.camera_center
            direction = direction / direction.norm(dim=-1, keepdim=True)
            colour = (self.base.eval_sh(self.semantic_static.active_sh_degree,
                      rows['shs'].transpose(1, 2), direction) + .5).clamp_min(0)
        else:
            colour = rows['colors_precomp']
        rows = dict(rows, means2D=self.torch.zeros_like(rows['means3D']), shs=None, colors_precomp=colour)
        static_rows = {key: None if value is None else value[self._semantic_static_mask].contiguous()
                       for key, value in rows.items()}
        dynamic_rows = super()._added_native_rows(camera)
        return self.base.merge(static_rows, dynamic_rows)

    def _added_row_actor_ids(self):
        return self.owners.new_full((self.added_rows['5'],), 5)

    def scene(self, raw, delete=False):
        scene = super().scene(raw, delete=delete)
        scene['meta'].update(semantic_actor5_mode=True,
            semantic_manifest_sha256=_sha(self._semantic_manifest_path),
            semantic_selection_sha256=self._semantic_manifest['selection_sha256'],
            semantic_calibration_BUILD_raws=self._semantic_manifest['raw_ids'],
            semantic_row_order='selected Full5 static rows, then selected Full5 dynamic rows',
            full5_added_branch_rows=self.semantic_branch_rows,
            full5_added_dynamic_source_actor_rows=self.semantic_dynamic_source_counts,
            full5_native_a6_rows_added=self.semantic_dynamic_source_counts['6'],
            full5_native_a37_rows_added=self.semantic_dynamic_source_counts['37'],
            source_actor_ancestry_preserved=True, semantic_identity_is_BUILD_majority_estimate=True,
            selected_rows_fixed_across_query_views=True, retire_legacy_static_actor5=False)
        return scene

    def receipt(self):
        result = super().receipt()
        result.update(semantic_actor5_mode=True,
            semantic_manifest=str(self._semantic_manifest_path),
            semantic_manifest_sha256=_sha(self._semantic_manifest_path),
            semantic_selection_sha256=self._semantic_manifest['selection_sha256'],
            full5_selected_branch_rows=self.semantic_branch_rows,
            full5_selected_dynamic_source_actor_rows=self.semantic_dynamic_source_counts,
            Full5_static_or_environment_loaded=True, Full5_static_loaded=True, Full5_environment_loaded=False,
            Full5_a6_or_a37_added=bool(self.semantic_dynamic_source_counts['6'] or self.semantic_dynamic_source_counts['37']),
            source_actor_ancestry_preserved=True, semantic_identity_is_BUILD_majority_estimate=True,
            original_v44_static_retirement=False,
            full5_static_parameters_and_lineage_unchanged=True)
        return result
