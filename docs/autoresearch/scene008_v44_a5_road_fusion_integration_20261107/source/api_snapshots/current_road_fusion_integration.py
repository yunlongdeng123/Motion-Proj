"""冻结 Semantic-a5 V44 场景与持久三维道路材质的只读组合。

AA/C 均追加相同40392行；C仅替换已烘焙的颜色。原场景的物理前缀不改，
诊断特征显式重排到24通道，所有图元仍在一次原生深度排序中合成。
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
AA_MANIFEST = CAMPAIGN / 'scene008_road_seam_contact_20261107_r1/engineer/canonical_n2_mip_candidate_v2/scene_manifest.json'
C_MANIFEST = CAMPAIGN / 'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/scene_manifest.json'
GROUPS = ('asset', 'a37', 'road', 'native_static', 'a5', 'a18', 'a26', 'newroad')
FEATURE_COUNT = 24
PHYSICAL_FIELDS = ('means3D', 'means2D', 'shs', 'colors_precomp', 'opacities',
                   'scales', 'rotations', 'cov3D_precomp', 'mask')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def _module(path):
    path = Path(path).resolve()
    name = '_road_fusion_pinned_' + _sha(path)[:20]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_road_payload(manifest_path, verify_dependencies=True):
    """CPU读取并验证显式道路包，不创建模型或CUDA上下文。"""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest['status'] != 'GEOMETRY_SUPPORT_RESEARCH_CANDIDATE'
            or manifest['adoption_claim'] or not manifest['native_capture_verified']):
        raise ValueError('只接受已核验、明确未采纳的原生道路tuple包')
    files = {str(path): _sha(path)}
    asset_path = path.parent / manifest['new_road']['file']
    if _sha(asset_path) != manifest['new_road']['sha256']:
        raise ValueError('道路资产SHA不符')
    files[str(asset_path)] = _sha(asset_path)
    for item in manifest['replay_sources'].values():
        source = path.parent / item['file']
        if _sha(source) != item['sha256']:
            raise ValueError('道路重放源码SHA不符：' + str(source))
        files[str(source)] = item['sha256']
    if verify_dependencies:
        for item in manifest['runtime_dependencies'].values():
            source = Path(item['path'])
            if source.stat().st_size != item['bytes'] or _sha(source) != item['sha256']:
                raise ValueError('冻结道路依赖不符：' + str(source))
            files[str(source)] = item['sha256']
    with np.load(asset_path, allow_pickle=False) as saved:
        asset = {key: saved[key].copy() for key in saved.files}
    n = len(asset['active'])
    if n != 40392 or n != manifest['new_road']['rows'] or not asset['active'].all():
        raise ValueError('必须保留完整40392行道路支撑')
    for key, width in [('render_model_xyz', 3), ('render_scales_model', 3),
                       ('render_rotation_wxyz', 4), ('render_opacity_logit', 1),
                       ('render_colours_srgb', 3)]:
        a = asset[key]
        if a.shape != (n, width) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('非法持久道路tuple：' + key)
    if (np.any(asset['render_colours_srgb'] < 0) or np.any(asset['render_colours_srgb'] > 1)
            or not np.array_equal(asset['child_to_parent_ids'], np.repeat(np.arange(10098), 4))):
        raise ValueError('道路材质或父子映射不符')
    return manifest, asset, files


class RoadFusionV44Integration:
    """传入已加载的SemanticActor5V44Integration，不重复加载第二套v44。"""
    def __init__(self, semantic_runtime, road_manifest_path, verify_dependencies=True):
        from .v44_integration import SemanticActor5V44Integration
        if not isinstance(semantic_runtime, SemanticActor5V44Integration):
            raise TypeError('基线必须为当前显式SemanticActor5V44Integration')
        self.semantic = semantic_runtime
        self.base = semantic_runtime.base
        self.np, self.torch = semantic_runtime.np, semantic_runtime.torch
        self.manifest_path = Path(road_manifest_path).resolve()
        self.manifest, self.asset, self._files = load_road_payload(self.manifest_path, verify_dependencies)
        if self.manifest['frozen_v44_binding_sha256'] != self.base.binding_sha256:
            raise ValueError('道路包与冻结v44底座不一致')
        if not np.array_equal(self.asset['world_to_model'], np.asarray(self.base.exp.meta['world_to_model'])):
            raise ValueError('道路与当前v44 world_to_model不同')
        if not np.array_equal(self.asset['coordinate_anchor'], np.asarray(self.base.exp.meta['coordinate_anchor'])):
            raise ValueError('道路与当前v44 anchor不同')
        if abs(float(self.asset['meter_to_model']) - float(self.base.scale)) > 1e-14:
            raise ValueError('道路与当前v44米制比例不同')
        source = _module(self.manifest_path.parent / 'canonical_road_runtime.py')
        self.extension = source.ExactSurface(self.base, self.asset)
        self.parent_indices = self.torch.as_tensor(self.asset['child_to_parent_ids'],
            device=self.base.n.bg.device, dtype=self.torch.long)
        policy = self.manifest['renderer_policy']
        if not policy['relative_mip']:
            raise ValueError('AA/C已核验relativeMip策略不可静默关闭')
        mip = _module(self.manifest_path.parent / policy['source_file'])
        self.opacity_fn = source.exact_relative_mip(mip,
            self.manifest_path.parent / self.manifest['new_road']['file'], self.asset, policy)
        if (bool(self.base.pipe.neg_fov) != bool(policy['legacy_neg_fov'])
                or bool(self.base.pipe.compute_cov3D_python) != bool(policy['normalize_quaternion'])):
            raise ValueError('原相机/协方差规则不同')
        self._file_identity = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in self._files}
        self.render_calls = 0
        self.assert_unchanged(full=True)

    def __getattr__(self, name):
        return getattr(self.semantic, name)

    def extension_rows(self, raw):
        camera = self.base.raw(raw)['cam']
        rows = self.base.road_kwargs(camera, self.extension, self.base.pipe,
            sh_channels=1, precomputed_color=True)
        rows = dict(rows, opacities=self.opacity_fn(camera, rows, self.parent_indices))
        feature = rows['features'].new_zeros((len(rows['mask']), FEATURE_COUNT))
        feature[:, :2] = rows['features'][:, :2]
        feature[:, 9] = 1  # extra[7] = newroad alpha*T
        feature[:, 13:16] = rows['colors_precomp']
        return dict(rows, features=feature)

    def scene(self, raw, delete=False):
        if type(delete) is not bool:
            raise TypeError('delete必须是bool，仅改变原v44 a37')
        self.assert_unchanged()
        with self.torch.no_grad():
            scene = self.semantic.scene(raw, delete=delete)
            original = scene['rows']
            f = original['features']
            if f.shape[1] != FEATURE_COUNT or bool((f[:, 7:9] != 0).any()) or bool((f[:, 12:18] != 0).any()):
                raise ValueError('仅支持已封存的a5-only语义特征布局，18/26必须为空')
            features = self.torch.zeros_like(f)
            features[:, :9] = f[:, :9]  # depth,alpha,原七组alpha
            features[:, 10:13] = f[:, 9:12]  # semantic5 RGBT，公开输出仍保留三actor槽
            prefix = dict(original, features=features)
            extension = self.extension_rows(raw)
            rows = self.base.merge(prefix, extension)
            n = len(original['mask'])
            for key, value in original.items():
                if key != 'features' and value is not None and not self.torch.equal(rows[key][:n], value):
                    raise RuntimeError('最新Semantic场景的物理前缀改变：' + key)
            meta = dict(scene['meta'], groups=list(GROUPS), latest_semantic_prefix_rows=n,
                latest_semantic_physical_prefix_exact=True, road_fusion_asset_rows=len(extension['mask']),
                road_fusion_manifest=str(self.manifest_path), road_fusion_manifest_sha256=_sha(self.manifest_path),
                road_fusion_asset_sha256=self.manifest['new_road']['sha256'],
                road_fusion_material_variant=self.manifest['new_road'].get('material_variant', 'AA_parent'),
                road_fusion_geometry_added=True, old_road_and_fine_colours_preserved=True,
                road_always_enabled=True, original_v44_static_retirement=False,
                feature_layout='depth0 alpha1 groups2:10 semantic5RGB10:13 newroadRGB13:16 zeros16:24',
                single_native_joint_sort=True, original_parameter_updates=0)
            return dict(scene, rows=rows, meta=meta)

    def render(self, raw, scene):
        with self.torch.no_grad():
            out = self.base.raster(scene['cam'], self.base.pipe, 0, scene['rows'], scene['bg'])
            self.render_calls += 1
            if out['extra'].shape[0] < 14:
                raise RuntimeError('24通道原生feature ABI不符')
            out['rgb_premultiplied'] = out['rgb']
            out['group_alpha_T'] = out['extra'][:8]
            out['component_alpha_T'] = out['group_alpha_T']
            rgb = out['extra'][8:11]
            out['added_owner_rgb_T'] = self.torch.stack((rgb, self.torch.zeros_like(rgb), self.torch.zeros_like(rgb)))
            out['added_owner_rgb_actor_ids'] = [5, 18, 26]
            out['newroad_rgb_T'] = out['extra'][11:14]
            out['env_rgb_T'] = (1-out['alpha']) * scene['env']
            out['constant_background_rgb_T'] = (1-out['alpha']) * scene['bg'][:, None, None]
            out['rgb'] = out['rgb_premultiplied'] + out['env_rgb_T']
            error = float((out['group_alpha_T'].sum(0)-out['alpha'][0]).abs().max())
            if error >= 2e-5:
                raise RuntimeError('八组joint alphaT不守恒：' + str(error))
            for key in ['rgb', 'alpha', 'depth_sum', 'extra', 'added_owner_rgb_T', 'newroad_rgb_T']:
                if not bool(self.torch.isfinite(out[key]).all()):
                    raise RuntimeError('非有限渲染：' + key)
            out['group_conservation_maxabs'] = error
            out['meta'] = scene['meta']
            return out

    def render_scene(self, raw, delete=False):
        return self.render(raw, self.scene(raw, delete=delete))

    def assert_unchanged(self, full=False):
        self.semantic.assert_unchanged(full=full)
        if not self.extension.unchanged():
            raise RuntimeError('持久道路tensor被改写')
        for path, before in self._file_identity.items():
            p = Path(path)
            if (p.stat().st_size, p.stat().st_mtime_ns) != before:
                raise RuntimeError('道路依赖文件改变：' + path)
            if full and _sha(p) != self._files[path]:
                raise RuntimeError('道路依赖字节改变：' + path)
        return True

    def receipt(self):
        self.assert_unchanged(full=True)
        return dict(status='PASS_READONLY_ROAD_FUSION_COMPOSITION_ENGINEERING',
            native_semantic=self.semantic.receipt(), scene_manifest=str(self.manifest_path),
            scene_manifest_sha256=_sha(self.manifest_path), road_asset_sha256=self.manifest['new_road']['sha256'],
            road_rows=40392, original_parameter_updates=0, optimizer_created=False,
            latest_semantic_physical_prefix_exact=True, original_road_fine_and_gates_unchanged=True,
            shared_native_depth_sort=True, screen_colour_composite=False,
            road_always_enabled=True, keep_delete_only_a37=True,
            renderer_relative_mip_receipt=self.opacity_fn.receipt(),
            feature_count=FEATURE_COUNT, joint_alphaT_groups=list(GROUPS),
            unused_added_owner_rgb_actor_slots=[18,26], render_calls=self.render_calls,
            adoption_claim=False, human_verdict=None)
