"""固定BUILD可见质量清单，仅过滤追加道路的临时原生mask。"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .road_fusion_integration import RoadFusionV44Integration

SCHEMA = 'fixed-road-support-v1'
CALIBRATION_RAW_IDS = [94, 105, 128, 137]


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def runtime_support_binding(road_runtime):
    semantic = road_runtime.semantic
    return dict(v44_binding_sha256=road_runtime.base.binding_sha256,
        v44_contract_sha256=_sha(road_runtime.base.config.contract_path),
        semantic_manifest_sha256=_sha(semantic._semantic_manifest_path),
        semantic_selection_sha256=semantic._semantic_manifest['selection_sha256'],
        full5_gaussian_files_sha256=semantic._semantic_manifest['gaussian_files_sha256'])


def read_support_sidecar(manifest_path, expected_asset_sha256, expected_rows, expected_binding):
    """CPU严格验证完整predicate，不根据照片或query掩码重新筛点。"""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA:
        raise ValueError('未知fixed-road-support schema')
    if manifest.get('calibration_raw_ids') != CALIBRATION_RAW_IDS:
        raise ValueError('必须绑定明确的4帧BUILD资格域')
    if manifest.get('asset_sha256') != expected_asset_sha256 or manifest.get('row_count') != expected_rows:
        raise ValueError('道路资产/行数与固定清单不符')
    if manifest.get('model_binding') != expected_binding:
        raise ValueError('父v44/Full5/语义身份绑定已改变，必须重新资格评定')
    if manifest.get('threshold') != dict(inside_strictly_greater_than=0., outside_equal=0.):
        raise ValueError('不允许更改inside>0且outside==0判据')
    if manifest.get('model_updates') != 0 or manifest.get('colour_updates') != 0:
        raise ValueError('资格过程不能更新模型或颜色')
    if (manifest.get('states') != ['keep', 'delete']
            or manifest.get('calibration_support_mode') != 'full40392_unfiltered'):
        raise ValueError('必须显式绑定两state与首次完整40392支撑资格场景')
    relative = Path(manifest['selection_file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('selection必须为sidecar内相对文件')
    npz_path = path.parent / relative
    if _sha(npz_path) != manifest['selection_sha256']:
        raise ValueError('固定道路清单SHA不符')
    with np.load(npz_path, allow_pickle=False) as saved:
        rows = saved['safe_child_row_ids'].copy()
        inside = saved['inside_mass'].copy()
        outside = saved['outside_mass'].copy()
    if rows.ndim != 1 or rows.dtype != np.int64 or not np.array_equal(rows, np.unique(rows)):
        raise ValueError('safe_child_row_ids必须是排序去重的int64行号')
    for name, a in [('inside_mass', inside), ('outside_mass', outside)]:
        if a.shape != (expected_rows,) or a.dtype != np.float64 or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('非法完整逐行质量：' + name)
    expected = np.flatnonzero((inside > 0) & (outside == 0))
    if not np.array_equal(rows, expected):
        raise ValueError('固定行号不等于完整inside/outside的精确predicate')
    if manifest.get('safe_rows') != len(rows):
        raise ValueError('固定行数声明不符')
    return manifest, rows, inside, outside, npz_path


class FixedRoadSupportV44Integration(RoadFusionV44Integration):
    """包装已加载的AA/C实例；不修改它或原Semantic实例的任何方法/参数。"""
    def __init__(self, road_runtime, support_manifest_path):
        if not isinstance(road_runtime, RoadFusionV44Integration):
            raise TypeError('必须传入已验证的RoadFusionV44Integration')
        self.road = road_runtime
        self._support_manifest_path = Path(support_manifest_path).resolve()
        result = read_support_sidecar(self._support_manifest_path,
            self.road.manifest['new_road']['sha256'], len(self.road.asset['active']),
            runtime_support_binding(self.road))
        self._support_manifest, self._support_rows_cpu, self._inside, self._outside, npz_path = result
        self._support_files = {str(p): (_sha(p), p.stat().st_size, p.stat().st_mtime_ns)
                               for p in [self._support_manifest_path, npz_path]}
        torch = self.road.torch
        self._support_mask = torch.zeros(len(self.road.asset['active']), dtype=torch.bool,
            device=self.road.base.n.bg.device)
        self._support_mask[torch.as_tensor(self._support_rows_cpu, dtype=torch.long,
            device=self._support_mask.device)] = True
        self._support_mask_version = (id(self._support_mask), self._support_mask._version)
        self.render_calls = 0
        self.assert_unchanged(full=True)

    def __getattr__(self, name):
        return getattr(self.road, name)

    @property
    def fixed_support_child_mask(self):
        return self._support_mask.detach().clone()

    @property
    def fixed_support_child_row_ids(self):
        return self._support_rows_cpu.copy()

    def extension_rows(self, raw):
        original = self.road.extension_rows(raw)
        if original['mask'].shape != self._support_mask.shape:
            raise ValueError('追加道路拓扑与固定mask不符')
        # 一切物理字段保持原tensor；只构造临时mask交集，同一清单用于KEEP/DELETE。
        return dict(original, mask=original['mask'] & self._support_mask)

    def scene(self, raw, delete=False):
        scene = RoadFusionV44Integration.scene(self, raw, delete=delete)
        offset = scene['meta']['latest_semantic_prefix_rows']
        if not self.torch.equal(scene['rows']['mask'][offset:], self.road.extension.active & self._support_mask):
            raise RuntimeError('实际追加道路mask不等于原mask与固定资格清单交集')
        scene['meta'].update(fixed_road_support=True,
            support_manifest_sha256=_sha(self._support_manifest_path),
            support_selection_sha256=self._support_manifest['selection_sha256'],
            support_calibration_raw_ids=CALIBRATION_RAW_IDS,
            support_safe_rows=len(self._support_rows_cpu), support_total_rows=len(self._support_mask),
            support_fixed_across_queries_and_states=True,
            support_predicate='inside_mass>0 and outside_mass==0',
            road_geometry_tensor_and_opacity_unchanged=True,
            road_temporary_mask_intersection_only=True)
        return scene

    def assert_unchanged(self, full=False):
        self.road.assert_unchanged(full=full)
        if self._support_mask_version != (id(self._support_mask), self._support_mask._version):
            raise RuntimeError('固定道路资格mask被改写')
        for name, (sha, size, mtime) in self._support_files.items():
            p = Path(name)
            if (p.stat().st_size, p.stat().st_mtime_ns) != (size, mtime) or (full and _sha(p) != sha):
                raise RuntimeError('固定道路sidecar改变：' + name)
        return True

    def receipt(self):
        result = RoadFusionV44Integration.receipt(self)
        result.update(fixed_road_support=True, support_manifest=str(self._support_manifest_path),
            support_manifest_sha256=_sha(self._support_manifest_path),
            support_selection_sha256=self._support_manifest['selection_sha256'],
            support_safe_rows=len(self._support_rows_cpu), support_total_rows=len(self._support_mask),
            support_calibration_raw_ids=CALIBRATION_RAW_IDS,
            support_fixed_across_queries_and_states=True,
            support_predicate='inside_mass>0 and outside_mass==0',
            support_model_binding=runtime_support_binding(self.road),
            original_base_and_road_masks_unmodified=True,
            road_temporary_mask_intersection_only=True, topology_and_future_query_not_recalibrated=True)
        return result
