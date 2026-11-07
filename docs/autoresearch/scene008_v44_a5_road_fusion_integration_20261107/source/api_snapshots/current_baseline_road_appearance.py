"""现有native-static/oldroad的固定行RGB残差，不追加或过滤任何几何。"""
from pathlib import Path
import hashlib
import json

import numpy as np

QUALIFICATION_SCHEMA = 'fixed-baseline-road-appearance-qualification-v1'
MATERIAL_SCHEMA = 'fixed-baseline-road-appearance-rgb-v1'
BRANCH_ROWS = dict(native_static=1042776,oldroad=143248)
CALIBRATION_RAW_IDS = [94,105,128,137]
BRANCH_KEYS = dict(native_static='static',oldroad='road')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):
            h.update(block)
    return h.hexdigest()


def baseline_appearance_binding(semantic):
    """实际静态参数含SH/time逐值指纹；道路/fine/gates由原v44强绑定。"""
    static=semantic.base.n.static
    keys=['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_velocity']
    return dict(v44_binding_sha256=semantic.base.binding_sha256,
        v44_contract_sha256=_sha(semantic.base.config.contract_path),
        native_static_tensor_sha256=semantic._runtime_module.tensor_hash({k:getattr(static,k) for k in keys}),
        oldroad_repair_state_sha256=semantic.base.repair_hash,
        oldroad_final_colour_sha256=semantic.base.road_colour_hash,
        semantic_manifest_sha256=_sha(semantic._semantic_manifest_path),
        semantic_selection_sha256=semantic._semantic_manifest['selection_sha256'],
        full5_gaussian_files_sha256=semantic._semantic_manifest['gaussian_files_sha256'],
        branch_rows=BRANCH_ROWS)


def read_qualification(path,expected_binding):
    path=Path(path).resolve();m=json.loads(path.read_text(encoding='utf-8'))
    if (m.get('schema')!=QUALIFICATION_SCHEMA or m.get('branch_rows')!=BRANCH_ROWS
            or m.get('calibration_raw_ids')!=CALIBRATION_RAW_IDS or m.get('states')!=['keep','delete']
            or m.get('calibration_scene')!='latest_semantic_without_added_road_geometry'):
        raise ValueError('未知现有geometry资格域/拓扑/schema')
    if m.get('model_binding')!=expected_binding:
        raise ValueError('父模型、SH/time或语义身份绑定已变化')
    if m.get('threshold')!=dict(inside_strictly_greater_than=0.,outside_equal=0.):
        raise ValueError('资格必须是inside>0且outside==0')
    if m.get('model_updates')!=0 or m.get('colour_updates')!=0:
        raise ValueError('资格不能更改模型/颜色')
    relative=Path(m['selection_file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('qualification选择文件必须在包内')
    npz=path.parent/relative
    if _sha(npz)!=m['selection_sha256']:
        raise ValueError('qualification数组SHA不符')
    with np.load(npz,allow_pickle=False) as z:
        payload={k:z[k].copy() for k in z.files}
    safe={}
    for branch,n in BRANCH_ROWS.items():
        key=BRANCH_KEYS[branch];ids=payload[key+'_safe_row_ids']
        inside,outside=payload[key+'_inside_mass'],payload[key+'_outside_mass']
        if ids.dtype!=np.int64 or ids.ndim!=1 or not np.array_equal(ids,np.unique(ids)):
            raise ValueError('资格行号必须排序去重int64：'+branch)
        for a in (inside,outside):
            if a.shape!=(n,) or a.dtype!=np.float64 or not np.isfinite(a).all() or np.any(a<0):
                raise ValueError('完整逐row质量非法：'+branch)
        if not np.array_equal(ids,np.flatnonzero((inside>0)&(outside==0))):
            raise ValueError('精确predicate与资格行号不同：'+branch)
        if m['safe_rows'][branch]!=len(ids):
            raise ValueError('safe计数声明不符：'+branch)
        safe[branch]=ids
    return m,safe,npz


def write_appearance_material(qualification_path,deltas_by_branch,output_dir):
    """只导出给定固定残差，没有优化器、拟合或自动选点。"""
    qualification_path=Path(qualification_path).resolve()
    q=json.loads(qualification_path.read_text(encoding='utf-8'))
    q,safe,_=read_qualification(qualification_path,q['model_binding'])
    arrays={}
    for branch in BRANCH_ROWS:
        d=np.asarray(deltas_by_branch[branch]);key=BRANCH_KEYS[branch]
        if d.shape!=(len(safe[branch]),3) or d.dtype!=np.float32 or not np.isfinite(d).all():
            raise ValueError('δ必须是已排序safe row对应的finite float32[N,3]：'+branch)
        arrays[key+'_row_ids']=safe[branch]
        arrays[key+'_rgb_delta']=d.copy()
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    npz=output_dir/'fixed_rgb_residual.npz';path=output_dir/'appearance_manifest.json'
    if npz.exists() or path.exists():
        raise FileExistsError('不得覆盖既存材质候选')
    np.savez_compressed(npz,**arrays)
    m=dict(schema=MATERIAL_SCHEMA,material_file=npz.name,material_sha256=_sha(npz),
        qualification_manifest=str(qualification_path),qualification_manifest_sha256=_sha(qualification_path),
        qualification_selection_sha256=q['selection_sha256'],model_binding=q['model_binding'],
        branch_rows=BRANCH_ROWS,apply_rule='selected colors_precomp = clamp_min(existing colors_precomp + fixed branch-row RGB delta, 0); no upper clamp',
        same_delta_for_all_cameras_times_and_keep_delete=True,geometry_updates=0,opacity_updates=0,mask_changes=0,
        SH_coefficients_unchanged=True,existing_SH_direction_dependence_preserved=True,
        nonnegative_projection_after_original_SH_and_fine=True,projection_can_make_view_dependence_nonlinear=True,
        old_fine_applied_before_residual=True,optimizer_steps_executed_by_exporter=0,
        quality_accepted=False,human_verdict=None)
    path.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


class BaselineRoadAppearanceV44Integration:
    """在实际joint packet中加残差；原参数、geometry、mask和features保持。"""
    def __init__(self,semantic_runtime,qualification_path,material_manifest_path):
        from .v44_integration import SemanticActor5V44Integration
        if not isinstance(semantic_runtime,SemanticActor5V44Integration):
            raise TypeError('只接受最新显式Semantic-a5基线，不接受追加道路场景')
        self.semantic=semantic_runtime
        self._qualification_path=Path(qualification_path).resolve()
        self._material_path=Path(material_manifest_path).resolve()
        self.binding=baseline_appearance_binding(semantic_runtime)
        self.qualification,self.safe,npz=read_qualification(self._qualification_path,self.binding)
        material=json.loads(self._material_path.read_text(encoding='utf-8'))
        if (material.get('schema')!=MATERIAL_SCHEMA or material.get('model_binding')!=self.binding
                or material.get('qualification_manifest_sha256')!=_sha(self._qualification_path)
                or material.get('branch_rows')!=BRANCH_ROWS
                or material.get('apply_rule')!='selected colors_precomp = clamp_min(existing colors_precomp + fixed branch-row RGB delta, 0); no upper clamp'):
            raise ValueError('固定RGB材质合同与资格/父模型不同')
        relative=Path(material['material_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('材质文件必须在包内')
        material_npz=self._material_path.parent/relative
        if _sha(material_npz)!=material['material_sha256']:
            raise ValueError('固定RGB材质SHA不符')
        with np.load(material_npz,allow_pickle=False) as z:
            values={k:z[k].copy() for k in z.files}
        torch=self.semantic.torch;device=self.semantic.base.n.bg.device
        self._ids,self._delta={},{}
        for branch in BRANCH_ROWS:
            key=BRANCH_KEYS[branch];ids=values[key+'_row_ids'];delta=values[key+'_rgb_delta']
            if not np.array_equal(ids,self.safe[branch]) or delta.shape!=(len(ids),3) or delta.dtype!=np.float32 or not np.isfinite(delta).all():
                raise ValueError('固定δ不是同一safe-row顺序：'+branch)
            self._ids[branch]=torch.as_tensor(ids.copy(),device=device,dtype=torch.long)
            self._delta[branch]=torch.as_tensor(delta.copy(),device=device)
        self._versions=[(t,id(t),t._version) for t in [*self._ids.values(),*self._delta.values()]]
        paths=[self._qualification_path,npz,self._material_path,material_npz]
        self._files={str(p):(_sha(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
        self.material=material;self.render_calls=0;self.assert_unchanged(full=True)

    def __getattr__(self,name):
        return getattr(self.semantic,name)

    def scene(self,raw,delete=False):
        self.assert_unchanged()
        with self.torch.no_grad():
            scene=self.semantic.scene(raw,delete=delete);original=scene['rows']
            colours=original['colors_precomp'].clone();f=original['features']
            for branch,column in [('native_static',5),('oldroad',4)]:
                rows=self.torch.nonzero(f[:,column]==1,as_tuple=False).flatten()
                if len(rows)!=BRANCH_ROWS[branch] or (len(rows)>1 and not bool((rows[1:]-rows[:-1]==1).all())):
                    raise ValueError('实际joint branch行数/连续顺序发生变化：'+branch)
                if branch=='native_static' and not self.torch.equal(rows,self.torch.arange(len(rows),device=rows.device)):
                    raise ValueError('native-static必须是原joint packet前缀')
                take=rows.index_select(0,self._ids[branch])
                colours[take]=(colours[take]+self._delta[branch]).clamp_min(0)
            if not bool(self.torch.isfinite(colours).all()):
                raise ValueError('固定δ导致非有限RGB')
            rows=dict(original,colors_precomp=colours)
            for key,value in original.items():
                if key!='colors_precomp' and value is not None and rows[key] is not value:
                    raise RuntimeError('现有geometry/opacity/mask/features的tensor身份改变：'+key)
            meta=dict(scene['meta'],fixed_baseline_road_appearance=True,
                appearance_material_sha256=self.material['material_sha256'],
                qualification_manifest_sha256=_sha(self._qualification_path),
                appearance_branch_safe_rows={k:len(v) for k,v in self.safe.items()},
                new_geometry_rows=0,all_geometry_opacity_masks_and_features_unchanged=True,
                original_SH_coefficients_unchanged=True,existing_SH_direction_dependence_preserved=True,
                nonnegative_projection_after_original_SH_and_fine=True,no_upper_RGB_clamp=True,
                projection_can_make_view_dependence_nonlinear=True,
                same_fixed_row_delta_for_all_views_and_states=True,old_fine_applied_before_residual=True)
            return dict(scene,rows=rows,meta=meta)

    def render(self,raw,scene):
        result=self.semantic.render(raw,scene);self.render_calls+=1;return result

    def render_scene(self,raw,delete=False):
        return self.render(raw,self.scene(raw,delete=delete))

    def assert_unchanged(self,full=False):
        self.semantic.assert_unchanged(full=full)
        if any((id(t),t._version)!=(identity,version) for t,identity,version in self._versions):
            raise RuntimeError('固定RGB残差/行号被改写')
        for name,(sha,size,mtime) in self._files.items():
            p=Path(name)
            if (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime) or (full and _sha(p)!=sha):
                raise RuntimeError('固定外观sidecar文件改变：'+name)
        return True

    def receipt(self):
        self.assert_unchanged(full=True)
        return dict(status='PASS_READONLY_FIXED_BASELINE_ROAD_APPEARANCE_ENGINEERING',
            native_semantic=self.semantic.receipt(),model_binding=self.binding,
            qualification_manifest=str(self._qualification_path),qualification_manifest_sha256=_sha(self._qualification_path),
            appearance_manifest=str(self._material_path),appearance_manifest_sha256=_sha(self._material_path),
            appearance_material_sha256=self.material['material_sha256'],
            branch_safe_rows={k:len(v) for k,v in self.safe.items()},new_geometry_rows=0,
            geometry_opacity_mask_features_and_original_parameters_unchanged=True,
            SH_coefficients_unchanged=True,existing_SH_direction_dependence_preserved=True,
            nonnegative_projection_after_original_SH_and_fine=True,no_upper_RGB_clamp=True,
            projection_can_make_view_dependence_nonlinear=True,
            old_fine_applied_before_residual=True,keep_delete_share_one_fixed_RGB_delta=True,
            render_calls=self.render_calls,optimizer_created=False,model_updates=0,
            qualification_is_BUILD_only_and_not_independent_generalization=True,
            quality_accepted=False,human_verdict=None)
