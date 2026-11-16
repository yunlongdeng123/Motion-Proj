"""固定支撑上的持久shared-parent RGB导出/只读接线；本模块不执行拟合。"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .fixed_road_support import FixedRoadSupportV44Integration

SCHEMA = 'fixed-support-shared-road-rgb-v1'
COLOUR_FIELDS = ('render_colours_srgb', 'parent_render_colours_srgb', 'material_parent_colours_srgb')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def noncolour_payload_sha256(asset):
    """连opacity、relativeMip父tuple、mask/行序及所有原provenance也固定。"""
    h = hashlib.sha256()
    for key in sorted(set(asset)-set(COLOUR_FIELDS)):
        a = np.ascontiguousarray(asset[key])
        h.update(key.encode('utf-8'));h.update(str((a.shape,a.dtype.str)).encode('ascii'));h.update(a.tobytes())
    return h.hexdigest()


def bake_shared_parent_material(source_asset_path, support_manifest_path, parent_rgb,
                                eligible_parent_ids, output_dir):
    """CPU重烘焙显式RGB表；调用者另行授权/登记拟合，本函数没有优化器。"""
    source_asset_path = Path(source_asset_path).resolve()
    support_manifest_path = Path(support_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    source = _load(source_asset_path)
    support = json.loads(support_manifest_path.read_text(encoding='utf-8'))
    if support['asset_sha256'] != _sha(source_asset_path):
        raise ValueError('支撑资格必须属于该原C资产')
    selection_path = support_manifest_path.parent / support['selection_file']
    if _sha(selection_path) != support['selection_sha256']:
        raise ValueError('支撑清单SHA不符')
    safe = _load(selection_path)['safe_child_row_ids']
    mapping = source['child_to_parent_ids']
    if not np.array_equal(mapping, np.repeat(np.arange(10098), 4)):
        raise ValueError('未知父子映射')
    eligible = np.asarray(eligible_parent_ids)
    if eligible.dtype != np.int64 or eligible.ndim != 1 or not np.array_equal(eligible,np.unique(eligible)):
        raise ValueError('eligible父行号必须排序去重int64')
    if not np.isin(eligible, np.unique(mapping[safe])).all():
        raise ValueError('颜色更新不能扩到没有固定safe child的父格')
    rgb = np.asarray(parent_rgb)
    if rgb.shape != (10098,3) or rgb.dtype != np.float32 or not np.isfinite(rgb).all() or np.any((rgb<0)|(rgb>1)):
        raise ValueError('必须提供完整finite float32 [10098,3] RGB表')
    initial = source['parent_render_colours_srgb']
    frozen = np.ones(10098,bool);frozen[eligible]=False
    if not np.array_equal(rgb[frozen], initial[frozen]):
        raise ValueError('未登记父格的RGB必须逐值保持原C')
    target = {k:v.copy() for k,v in source.items()}
    target['parent_render_colours_srgb'] = rgb.copy()
    target['material_parent_colours_srgb'] = rgb.copy()
    target['render_colours_srgb'] = rgb[mapping].copy()
    if noncolour_payload_sha256(target) != noncolour_payload_sha256(source):
        raise RuntimeError('颜色导出改变了非颜色字段')
    output_dir.mkdir(parents=True,exist_ok=True)
    asset_path = output_dir/'new_road_shared_rgb_candidate.npz'
    manifest_path = output_dir/'shared_material_manifest.json'
    if asset_path.exists() or manifest_path.exists() or asset_path == source_asset_path:
        raise FileExistsError('不得覆盖原资产或既存候选')
    np.savez_compressed(asset_path,**target)
    manifest = dict(schema=SCHEMA,asset_file=asset_path.name,asset_sha256=_sha(asset_path),
        source_asset=str(source_asset_path),source_asset_sha256=_sha(source_asset_path),
        support_manifest=str(support_manifest_path),support_manifest_sha256=_sha(support_manifest_path),
        support_selection_sha256=support['selection_sha256'],
        noncolour_payload_sha256=noncolour_payload_sha256(source),
        eligible_parent_row_ids=eligible.tolist(),changed_parent_rows=int(np.any(rgb!=initial,axis=1).sum()),
        parent_count=10098,child_count=40392,shared_four_child_rgb=True,
        current_material_authority='this manifest and baked render_colours_srgb; other old material metadata remains source provenance',
        geometry_updates=0,opacity_updates=0,mask_changes=0,optimizer_steps_executed_by_exporter=0,
        quality_accepted=False,human_verdict=None)
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return manifest_path


class SharedRoadMaterialV44Integration(FixedRoadSupportV44Integration):
    """保留原C与固定mask，只在临时packet取出已烘焙的另一个RGB表。"""
    def __init__(self, fixed_runtime, material_manifest_path):
        if not isinstance(fixed_runtime,FixedRoadSupportV44Integration):
            raise TypeError('必须显式使用已绑定的固定道路支撑实例')
        self.fixed = fixed_runtime
        self._material_manifest_path = Path(material_manifest_path).resolve()
        manifest = json.loads(self._material_manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA:
            raise ValueError('未知持久shared-RGB schema')
        source_path = self.fixed.road.manifest_path.parent/self.fixed.road.manifest['new_road']['file']
        if manifest['source_asset_sha256'] != _sha(source_path):
            raise ValueError('RGB派生资产的原C父包不符')
        if manifest['support_manifest_sha256'] != _sha(self.fixed._support_manifest_path):
            raise ValueError('不得把材质派生资产绑定到另一份资格清单')
        relative=Path(manifest['asset_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('派生资产必须在材质包内')
        asset_path=self._material_manifest_path.parent/relative
        if _sha(asset_path)!=manifest['asset_sha256']:
            raise ValueError('派生资产SHA不符')
        target=_load(asset_path);source=self.fixed.road.asset
        if set(target)!=set(source) or noncolour_payload_sha256(target)!=noncolour_payload_sha256(source):
            raise ValueError('派生资产改变了几何/opacity/mask/relativeMip/来源字段')
        if manifest['noncolour_payload_sha256']!=noncolour_payload_sha256(source):
            raise ValueError('非颜色payload指纹不符')
        mapping=source['child_to_parent_ids'];rgb=target['parent_render_colours_srgb']
        eligible=np.asarray(manifest['eligible_parent_row_ids'],np.int64)
        if not np.array_equal(eligible,np.unique(eligible)) or not np.isin(eligible,np.unique(mapping[self.fixed._support_rows_cpu])).all():
            raise ValueError('登记父格越过safe支撑范围')
        frozen=np.ones(10098,bool);frozen[eligible]=False
        if (rgb.shape!=(10098,3) or rgb.dtype!=np.float32 or not np.isfinite(rgb).all()
                or np.any((rgb<0)|(rgb>1)) or not np.array_equal(rgb[frozen],source['parent_render_colours_srgb'][frozen])
                or not np.array_equal(target['render_colours_srgb'],rgb[mapping])
                or not np.array_equal(target['material_parent_colours_srgb'],rgb)):
            raise ValueError('派生RGB范围、冻结父格或四子继承不符')
        self._material_colours=self.fixed.torch.as_tensor(target['render_colours_srgb'].copy(),
            device=self.fixed.base.n.bg.device)
        self._material_version=(id(self._material_colours),self._material_colours._version)
        self._material_manifest=manifest
        self._material_files={str(p):(_sha(p),p.stat().st_size,p.stat().st_mtime_ns)
                              for p in [self._material_manifest_path,asset_path]}
        self.render_calls=0
        self.assert_unchanged(full=True)

    def __getattr__(self,name):
        return getattr(self.fixed,name)

    def extension_rows(self,raw):
        original=self.fixed.extension_rows(raw)
        feature=original['features'].clone()
        feature[:,13:16]=self._material_colours
        return dict(original,colors_precomp=self._material_colours,features=feature)

    def scene(self,raw,delete=False):
        scene=FixedRoadSupportV44Integration.scene(self,raw,delete=delete)
        scene['meta'].update(shared_road_material_override=True,
            baked_material_asset_sha256=self._material_manifest['asset_sha256'],
            baked_material_source_C_sha256=self._material_manifest['source_asset_sha256'],
            material_qualification_inheritance='all original noncolour arrays exact, including geometry/opacity/fixedmask; RGB quality requires independent gates')
        return scene

    def assert_unchanged(self,full=False):
        self.fixed.assert_unchanged(full=full)
        if self._material_version!=(id(self._material_colours),self._material_colours._version):
            raise RuntimeError('持久材质颜色被改写')
        for name,(sha,size,mtime) in self._material_files.items():
            p=Path(name)
            if (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime) or (full and _sha(p)!=sha):
                raise RuntimeError('持久材质包被改写')
        return True

    def receipt(self):
        result=FixedRoadSupportV44Integration.receipt(self)
        result.update(shared_road_material_override=True,
            baked_material_manifest=str(self._material_manifest_path),
            baked_material_manifest_sha256=_sha(self._material_manifest_path),
            baked_material_asset_sha256=self._material_manifest['asset_sha256'],
            all_original_noncolour_payload_exact=True,
            original_C_colours_and_geometry_tensors_unmodified=True,
            geometry_opacity_and_fixed_support_identical=True,material_quality_requires_registered_gates=True)
        return result
