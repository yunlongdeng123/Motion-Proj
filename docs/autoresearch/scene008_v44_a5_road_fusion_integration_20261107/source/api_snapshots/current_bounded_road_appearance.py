"""固定几何的逐域解析颜色预算；所有质量矩阵只在BUILD生成。"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .baseline_road_appearance import (BaselineRoadAppearanceV44Integration,
    baseline_appearance_binding, BRANCH_ROWS, CALIBRATION_RAW_IDS)

QUALIFICATION_SCHEMA='bounded-road-appearance-qualification-v1'
MATERIAL_SCHEMA='bounded-road-appearance-rgb-v1'
EPSILON=5.e-7
ORIGINAL_TOLERANCE=1.e-6
FEASIBILITY_CONTRACT_SHA256='81194d78c1aa130485a8907f40278924361348b755bd64d8e3e16f346d108417'
DOMAINS={'a5_full','a5_upper','a6_visible','protected_nonroad','a37_visible_keep'}
APPLY_RULE='selected colors_precomp = clamp_min(existing colors_precomp + fixed branch-row RGB delta, 0); no upper clamp'


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()


def _relative(parent,value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts:raise ValueError('数组文件必须为包内相对路径')
    return parent/path


def recompute_constraint_weights(domain_mass,domain_records,epsilon=EPSILON):
    """w_r=max_m A_mr/(3*N_m*epsilon)，不接受空域作为零分母。"""
    mass=np.asarray(domain_mass)
    if (mass.ndim!=2 or mass.dtype!=np.float64 or len(domain_records)!=len(mass)
            or not len(mass) or not np.isfinite(mass).all() or np.any(mass<0)):
        raise ValueError('逐域完整质量矩阵非法')
    if epsilon!=EPSILON:raise ValueError('解析预算不能改变')
    keys=set()
    for row,record in zip(mass,domain_records):
        key=(record['raw'],record['state'],record['domain']);pixels=record['pixels']
        if (key in keys or key[0] not in CALIBRATION_RAW_IDS or key[1] not in ['keep','delete']
                or key[2] not in DOMAINS or type(pixels) is not int or not 0<pixels<=153600
                or (key[2]=='a37_visible_keep' and key[1]!='keep')):
            raise ValueError('重复、未知或空保护域')
        keys.add(key)
    counts=np.asarray([record['pixels'] for record in domain_records],dtype=np.float64)
    return np.max(mass/(3.0*counts[:,None]*epsilon),axis=0)


def weighted_l1_cost(delta,weights):
    delta=np.asarray(delta);weights=np.asarray(weights)
    if delta.ndim!=2 or delta.shape!=(len(weights),3) or not np.isfinite(delta).all():
        raise ValueError('非法逐行RGB residual')
    if weights.dtype!=np.float64 or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('非法逐行预算权重')
    return float(np.sum(weights*np.abs(delta.astype(np.float64)).sum(1),dtype=np.float64))


def project_box_weighted_l1_cpu(delta,weights):
    """数值工具：先box[-1,1]，再加权L1 soft-threshold；不是一次fit。"""
    raw=np.asarray(delta,dtype=np.float64)
    if not np.isfinite(raw).all():raise ValueError('projection不能掩盖非有限更新')
    z=np.clip(raw,-1.,1.)
    weights=np.asarray(weights,dtype=np.float64)
    if z.shape!=(len(weights),3) or np.any(weights<0) or not np.isfinite(z).all() or not np.isfinite(weights).all():
        raise ValueError('projection输入非法')
    positive=weights>0
    if weighted_l1_cost(z,weights)>1.:
        low=0.;high=float(np.max(np.abs(z)))/float(weights.max())
        # 极小但非零的alpha尾不能把数值括区扩大到1/w_min，丢失根精度。
        while weighted_l1_cost(np.sign(z)*np.maximum(np.abs(z)-high*weights[:,None],0.),weights)>1.:
            high*=2.
            if not np.isfinite(high):raise ValueError('projection根无有限括区')
        for _ in range(80):
            middle=(low+high)/2
            x=np.sign(z)*np.maximum(np.abs(z)-middle*weights[:,None],0.)
            if weighted_l1_cost(x,weights)>1.:low=middle
            else:high=middle
        z=np.sign(z)*np.maximum(np.abs(z)-high*weights[:,None],0.)
    result=z.astype(np.float32)
    # 只处理float32舍入造成的轻微越界；零成本行保持原box结果。
    for _ in range(4):
        cost=weighted_l1_cost(result,weights)
        if cost<=1.:break
        result[positive]*=np.float32(np.nextafter(np.float32(1),np.float32(0))/cost)
    if weighted_l1_cost(result,weights)>1.:raise ValueError('projection数值预算未满足')
    return result


def read_bounded_qualification(path,expected_binding=None):
    path=Path(path).resolve();m=json.loads(path.read_text(encoding='utf-8'))
    if (m.get('schema')!=QUALIFICATION_SCHEMA or m.get('branch_rows')!=BRANCH_ROWS
            or m.get('calibration_raw_ids')!=CALIBRATION_RAW_IDS or m.get('states')!=['keep','delete']
            or m.get('epsilon')!=EPSILON or m.get('original_tolerance')!=ORIGINAL_TOLERANCE
            or m.get('feasibility_contract_sha256')!=FEASIBILITY_CONTRACT_SHA256
            or m.get('model_updates')!=0 or m.get('colour_updates')!=0
            or m.get('base_RGB_nonnegative_premise') is not True):
        raise ValueError('未知模型拓扑、BUILD域、预算或可行性合同')
    if expected_binding is not None and m.get('model_binding')!=expected_binding:
        raise ValueError('父模型、SH/time或身份绑定已变化')
    matrix_path=_relative(path.parent,m['mass_matrix_file'])
    if _sha(matrix_path)!=m['mass_matrix_sha256']:raise ValueError('贡献矩阵SHA不符')
    with np.load(matrix_path,allow_pickle=False) as z:
        inside=z['inside_mass'].copy();mass=z['domain_mass'].copy()
        candidates=z['candidate_row_ids'].copy();stored_weights=z['weights'].copy()
    total=sum(BRANCH_ROWS.values())
    if (inside.shape!=(total,) or inside.dtype!=np.float64 or not np.isfinite(inside).all()
            or np.any(inside<0) or mass.shape[1:]!=(total,)):
        raise ValueError('inside/贡献矩阵行序不符')
    if (candidates.dtype!=np.int64 or candidates.ndim!=1
            or not np.array_equal(candidates,np.flatnonzero(inside>0))):
        raise ValueError('候选必须精确等于inside>0的完整global branch行序')
    weights=recompute_constraint_weights(mass,m['domain_records'])
    measured={(r['raw'],r['state'],r['domain']) for r in m['domain_records']}
    empty=m.get('empty_domains',[])
    empty_keys={(r['raw'],r['state'],r['domain']) for r in empty}
    all_domains={(raw,state,domain) for raw in CALIBRATION_RAW_IDS
                 for state in ['keep','delete'] for domain in DOMAINS}
    if (len(empty_keys)!=len(empty) or measured&empty_keys or measured|empty_keys!=all_domains
            or any(r.get('pixels')!=0 for r in empty)):
        raise ValueError('原40个view/state/domain必须逐一归为非空约束或显式空域')
    if stored_weights.dtype!=np.float64 or not np.array_equal(stored_weights,weights):
        raise ValueError('保存的w与完整矩阵/N/预算重算不同')
    if m.get('candidate_rows')!=len(candidates):raise ValueError('候选行数声明不符')
    return m,candidates,weights,inside,mass,matrix_path


def write_bounded_appearance_material(qualification_path,delta,output_dir):
    """CPU导出固定δ，完整矩阵重算w并实查预算；不创建优化器。"""
    qualification_path=Path(qualification_path).resolve()
    q,ids,w,_,_,matrix_path=read_bounded_qualification(qualification_path)
    delta=np.asarray(delta)
    if (delta.dtype!=np.float32 or delta.shape!=(len(ids),3) or not np.isfinite(delta).all()
            or np.any(np.abs(delta)>1)):
        raise ValueError('δ必须为候选顺序finite float32[N,3]且在[-1,1]')
    cost=weighted_l1_cost(delta,w[ids])
    if cost>1.:raise ValueError('导出δ违反解析weighted L1预算')
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    npz=output_dir/'bounded_rgb_residual.npz';path=output_dir/'appearance_manifest.json'
    if npz.exists() or path.exists():raise FileExistsError('不得覆盖既存材质')
    np.savez_compressed(npz,global_branch_row_ids=ids,rgb_delta=delta)
    material=dict(schema=MATERIAL_SCHEMA,material_file=npz.name,material_sha256=_sha(npz),
        qualification_manifest_sha256=_sha(qualification_path),mass_matrix_sha256=_sha(matrix_path),
        model_binding=q['model_binding'],branch_rows=BRANCH_ROWS,apply_rule=APPLY_RULE,
        epsilon=EPSILON,original_tolerance=ORIGINAL_TOLERANCE,weighted_l1_cost=cost,
        parameter_bounds=[-1.,1.],same_delta_all_cameras_times_states=True,
        nonnegative_projection_after_original_SH_and_fine=True,
        geometry_updates=0,opacity_updates=0,mask_changes=0,exporter_optimizer_steps=0,
        constraint_scope='calibrated four BUILD views only; future queries use fixed row delta',
        quality_accepted=False,human_verdict=None)
    path.write_text(json.dumps(material,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


class BoundedRoadAppearanceV44Integration(BaselineRoadAppearanceV44Integration):
    """新F入口，旧415和E资格/材料不变；未来查询不按当帧GT选择行。"""
    def __init__(self,semantic_runtime,qualification_path,material_manifest_path):
        from .v44_integration import SemanticActor5V44Integration
        if not isinstance(semantic_runtime,SemanticActor5V44Integration):
            raise TypeError('必须为原最新Semantic-a5基线，不含新增road或E residual')
        self.semantic=semantic_runtime;self.binding=baseline_appearance_binding(semantic_runtime)
        self._qualification_path=Path(qualification_path).resolve()
        self._material_path=Path(material_manifest_path).resolve()
        q,ids,weights,_,_,matrix_path=read_bounded_qualification(self._qualification_path,self.binding)
        self.qualification=q
        material=json.loads(self._material_path.read_text(encoding='utf-8'))
        if (material.get('schema')!=MATERIAL_SCHEMA or material.get('model_binding')!=self.binding
                or material.get('qualification_manifest_sha256')!=_sha(self._qualification_path)
                or material.get('mass_matrix_sha256')!=_sha(matrix_path)
                or material.get('epsilon')!=EPSILON or material.get('original_tolerance')!=ORIGINAL_TOLERANCE
                or material.get('apply_rule')!=APPLY_RULE or material.get('parameter_bounds')!=[-1.,1.]):
            raise ValueError('未知F材质绑定、预算或颜色规则')
        material_npz=_relative(self._material_path.parent,material['material_file'])
        if _sha(material_npz)!=material['material_sha256']:raise ValueError('δ文件SHA不符')
        with np.load(material_npz,allow_pickle=False) as z:
            saved_ids=z['global_branch_row_ids'].copy();delta=z['rgb_delta'].copy()
        if (saved_ids.dtype!=np.int64 or not np.array_equal(ids,saved_ids) or delta.shape!=(len(ids),3) or delta.dtype!=np.float32
                or not np.isfinite(delta).all() or np.any(np.abs(delta)>1)):
            raise ValueError('δ行序/范围不符')
        self.weighted_cost=weighted_l1_cost(delta,weights[ids])
        if self.weighted_cost>1. or not np.isclose(self.weighted_cost,material['weighted_l1_cost'],rtol=1e-13,atol=1e-13):
            raise ValueError('载入δ违反重算weighted L1预算')
        torch=self.semantic.torch;device=self.semantic.base.n.bg.device;ns=BRANCH_ROWS['native_static']
        self.safe={'native_static':ids[ids<ns],'oldroad':ids[ids>=ns]-ns}
        self._ids={k:torch.as_tensor(v.copy(),device=device,dtype=torch.long) for k,v in self.safe.items()}
        self._delta={'native_static':torch.as_tensor(delta[ids<ns].copy(),device=device),
                     'oldroad':torch.as_tensor(delta[ids>=ns].copy(),device=device)}
        self._versions=[(t,id(t),t._version) for t in [*self._ids.values(),*self._delta.values()]]
        paths=[self._qualification_path,matrix_path,self._material_path,material_npz]
        self._files={str(p):(_sha(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
        self.material=material;self.render_calls=0;self.assert_unchanged(full=True)

    def scene(self,raw,delete=False):
        self.assert_unchanged()
        with self.torch.no_grad():
            scene=self.semantic.scene(raw,delete=delete);original=scene['rows'];f=original['features']
            colour=original['colors_precomp'].clone()
            for branch,column in [('native_static',5),('oldroad',4)]:
                rows=self.torch.nonzero(f[:,column]==1,as_tuple=False).flatten()
                if len(rows)!=BRANCH_ROWS[branch] or (len(rows)>1 and not bool((rows[1:]-rows[:-1]==1).all())):
                    raise ValueError('原branch拓扑/顺序改变：'+branch)
                if branch=='native_static' and not self.torch.equal(rows,self.torch.arange(len(rows),device=rows.device)):
                    raise ValueError('native-static前缀改变')
                take=rows.index_select(0,self._ids[branch]);base=colour[take]
                if not bool((base>=0).all()):raise ValueError('Lipschitz证明所需原branch颜色非负假设不满足')
                colour[take]=(base+self._delta[branch]).clamp_min(0)
            if not bool(self.torch.isfinite(colour).all()):raise ValueError('非有限RGB')
            rows=dict(original,colors_precomp=colour)
            if any(value is not None and key!='colors_precomp' and rows[key] is not value for key,value in original.items()):
                raise RuntimeError('geometry/opacity/mask/features改变')
            meta=dict(scene['meta'],bounded_baseline_road_appearance=True,
                appearance_material_sha256=self.material['material_sha256'],
                qualification_manifest_sha256=_sha(self._qualification_path),
                appearance_branch_candidate_rows={k:len(v) for k,v in self.safe.items()},
                weighted_l1_cost=self.weighted_cost,per_measured_domain_analytic_MAE_bound=EPSILON*self.weighted_cost,
                original_protection_tolerance=ORIGINAL_TOLERANCE,BUILD_bound_not_unseen_view_guarantee=True,
                new_geometry_rows=0,all_geometry_opacity_masks_and_features_unchanged=True,
                original_SH_coefficients_unchanged=True,old_fine_applied_before_residual=True,
                same_fixed_row_delta_all_views_and_states=True,nonnegative_baseline_colours_checked=True)
            return dict(scene,rows=rows,meta=meta)

    def receipt(self):
        result=super().receipt();result.pop('branch_safe_rows',None)
        result.update(status='PASS_READONLY_BOUNDED_BASELINE_ROAD_APPEARANCE_ENGINEERING',
            branch_candidate_rows={k:len(v) for k,v in self.safe.items()},
            weighted_l1_cost=self.weighted_cost,per_measured_domain_analytic_MAE_bound=EPSILON*self.weighted_cost,
            epsilon=EPSILON,original_tolerance=ORIGINAL_TOLERANCE,
            bound_scope='only recorded four BUILD view/state domains; unknown or49-frame views require actual evaluation',
            all_point_colour_change_is_not_claimed_zero=True)
        return result
