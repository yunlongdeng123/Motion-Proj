"""独立只读核对两parent的同95观测材质激活；原字节/旧B32类别不改变。"""
from pathlib import Path
import argparse,json,os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np


def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def audit_pair(parent,child,ids,b32):
    checks={};ledger=[];B32_fields=[]
    with np.load(parent/'road.npz',allow_pickle=False) as a,np.load(child/'road.npz',allow_pickle=False) as b,np.load(parent/'provenance.npz',allow_pickle=False) as p:
        oldactive=a['active'];newactive=b['active'];oldrgb=a['reference_srgb'];newrgb=b['reference_srgb']
        checks['same_road_keys']=a.files==b.files
        checks['only95_active_changes']=np.array_equal(np.flatnonzero(oldactive!=newactive),ids)
        checks['95_false_to_true']=bool(not oldactive[ids].any() and newactive[ids].all())
        outside=np.ones(len(oldactive),bool);outside[ids]=False
        checks['outside95_RGB_shape_dtype_bytes_exact']=exact(oldrgb[outside],newrgb[outside])
        checks['new95_RGB_from_immutable_observed_exact']=exact(newrgb[ids],p['observed_reference_srgb'][ids])
        checks['old_active_RGB_bytes_exact']=exact(oldrgb[oldactive],newrgb[oldactive])
        for key in a.files:
            if key not in ['active','reference_srgb']:checks['road_frozen_bytes_'+key]=exact(a[key],b[key])
        new_rgb_change=int(np.any(oldrgb!=newrgb,axis=1).sum())
        counts=dict(new_active=int((~oldactive&newactive).sum()),deactivated=int((oldactive&~newactive).sum()),RGB_changed_rows=new_rgb_change,
            parent_active_rows=int(oldactive.sum()),child_active_rows=int(newactive.sum()))
    # 源观测与历史覆盖/几何证据全部冻结；有效运行状态另字段保存。
    effective={'effective_row_rgb','effective_active','effective_model_xyz','effective_sigma_m','effective_road_opacity',
        'effective_coverage_requested','effective_material_source','effective_material_available','effective_material_unknown',
        'effective_coverage_legal','effective_material_source_raw','effective_material_source_uv_native','allowed_color_refine'}
    with np.load(parent/'provenance.npz',allow_pickle=False) as a,np.load(child/'provenance.npz',allow_pickle=False) as b:
        for key in a.files:
            if key.startswith('activation_') or key.startswith('parent_') or key in effective:continue
            checks['prior_bytes_'+key]=exact(a[key],b[key]);ledger.append(key)
            if key.startswith('B32_'):B32_fields.append(key)
        checks['95_observed_ledger_qualification']=bool(a['known'][ids].all() and (a['source'][ids]==1).all() and not a['unknown'][ids].any())
        for key in effective&set(a.files)&set(b.files):
            # 历史B32的所有有效字段也必须保存。新95可更新同名当前状态。
            checks['B32_effective_preserved_'+key]=exact(a[key][b32],b[key][b32]) if a[key].shape and a[key].shape[0]==len(outside) else exact(a[key],b[key])
        coverage=a['effective_coverage_requested']
        checks['all_parent_effective_requests_preserved']=bool(b['effective_coverage_requested'][coverage].all())
        checks['new95_effective_requests_set']=bool(b['effective_coverage_requested'][ids].all())
        if B32_fields:
            checks['B32_original_ledger_unknown_source0']=bool(not b['known'][b32].any() and b['unknown'][b32].all() and (b['source'][b32]==0).all())
            checks['B32_effective_material_source3']=bool((b['effective_material_source'][b32]==3).all())
            checks['B32_effective_available']=bool(b['effective_material_available'][b32].all() and not b['effective_material_unknown'][b32].any())
            checks['B32_pseudo_source_raw110']=bool((b['effective_material_source_raw'][b32]==110).all())
            counts['retained_B32_pseudo_rows']=len(b32)
        else:counts['retained_B32_pseudo_rows']=0
    return dict(parent_asset=str(parent),tested_asset=str(child),checks=checks,passed=all(checks.values()),counts=counts,
        preserved_prior_fields=ledger,preserved_B32_fields=B32_fields)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();N=a.campaign
    with np.load(N/'footprint_fixed_sigma_cpu_r3/observed_inactive_fixed_sigma_candidates.npz',allow_pickle=False) as z:ids=z['proposed_world_rows'].copy()
    with np.load(N/'supervision_B32_r1/activation_pseudo_init.npz',allow_pickle=False) as z:b32=z['proposed_world_rows'].copy()
    if len(np.unique(ids))!=95 or len(np.intersect1d(ids,b32)):raise ValueError('same95/B32 scope changed')
    pairs=[(N/'native_r1/asset_A1_guarded',N/'A1_beam95_native_r1/asset_guarded'),
        (N/'B32_native_r1/asset_B32_guarded',N/'B32_beam95_native_r1/asset_guarded')]
    results=[audit_pair(p,c,ids,b32) for p,c in pairs]
    receipt=dict(schema='same95_activation_source_bytes_independent_CPU.v1',status='PASS' if all(r['passed'] for r in results) else 'FAIL',
        checks=sum(len(r['checks']) for r in results),same_new_world_rows=95,rows=results,CPU_only=True,
        comparison='full shape/dtype/bytes, including NaN payloads; active/RGB delta explicitly scoped',
        limits=['CPU source/bytes audit only; native K/alphaT/support/public reload are independently recorded by renderer/QA',
            'same95 membership inherited original A1 residual73; no arbitrary radius or new geometry',
            'B32 remains effective pseudo material, original observed source0/unknown classification immutable',
            'new observed RGB source nativeUV was historically not retained; frame/world lineage is preserved, no pixel-GT claim'],
        GPU_operations=0,optimizer_updates=0,old_assets_modified=False,human_verdict=None)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'asset_bytes_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],checks=receipt['checks'],rows=[r['counts'] for r in results])))
    if receipt['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
