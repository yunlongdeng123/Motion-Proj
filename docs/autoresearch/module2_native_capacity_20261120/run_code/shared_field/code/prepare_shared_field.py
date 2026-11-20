"""CPU-only common avg-A200/H0 export and seen-mask geometric bandwidth audit.

No target RGB, held-domain statistics, native context or optimizer are read here.
The frozen existing A200 was historically exposed to both views; this is not blind.
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np
import torch
from scipy import sparse
from shared_field_gs_read import file_sha,load_common_avg_model


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def export_initial(q,asset,output):
    """Exact zero-H fast path avoids loading the 20M-entry high matrix on CPU."""
    q=Path(q);g=q/'gpu_20261119';asset=Path(asset);output=Path(output)
    if output.exists():raise FileExistsError('preserve prior preparation: '+str(output))
    output.mkdir(parents=True)
    field=g/'formal_r1/avg_common_A200/field.pt';permission=q/'permission/final_r1/teacher_evidence.npz'
    runtime=q/'code/world_texture_runtime.py';operator=q/'permission/avg_sparse_gh31_r1/operator'
    scope=g/'native_qualification_r2/basis/110/row_scope.npz';road=asset/'road.npz'
    model,binding=load_common_avg_model(field,permission,runtime)
    with np.load(scope,allow_pickle=False)as z:
        active_ids=z['active_world_rows'].copy();eligible=z['eligible_active'].copy()
    with np.load(road,allow_pickle=False)as z:
        parent_ids=np.flatnonzero(z['active']);parent_rgb=z['reference_srgb'][parent_ids].copy();world_count=len(z['active'])
    if not np.array_equal(active_ids,parent_ids):raise ValueError('scope and parent active topology differ')
    with np.load(operator/'constants.npz',allow_pickle=False)as z:base=z['base'].copy();row_ids=z['row_ids'].copy()
    if not np.array_equal(np.sort(row_ids),active_ids[eligible]):raise ValueError('operator rows != exact common scope')
    positions=np.searchsorted(active_ids,row_ids)
    KL=sparse.load_npz(operator/'low.npz')
    with torch.no_grad():L,H=model.fields()
    if bool((H!=0).any()):raise ValueError('H initial is not exactly zero')
    values=np.clip(base+KL@L.numpy().reshape(-1,3),0.,1.).astype(np.float32)
    ci0=parent_rgb.copy();ci0[positions]=values
    world_eligible=np.zeros(world_count,bool);world_eligible[active_ids]=eligible
    path=output/'initial_ci.npz'
    np.savez_compressed(path,ci0=ci0,parent_rgb=parent_rgb,active_world_rows=active_ids,
        eligible_active=eligible,world_eligible=world_eligible,operator_row_ids=row_ids)
    fieldout=output/'initial_field.pt'
    model.save(fieldout,dict(role='CPU_prepared_initialization',reader='avg',cursor=200,
        H_initial_exact_zero=True,source_common_avg_A200=str(field),ordinary_GS_primary=True,binding=binding))
    with np.load(path,allow_pickle=False)as z:
        reload_exact=all(np.array_equal(z[k],v)for k,v in dict(ci0=ci0,parent_rgb=parent_rgb,
            active_world_rows=active_ids,eligible_active=eligible,world_eligible=world_eligible,operator_row_ids=row_ids).items())
    state=torch.load(fieldout,map_location='cpu',weights_only=False)['state']
    field_exact=all(torch.equal(v,state[k])for k,v in model.state_dict().items())
    fixed_paths=[Path(__file__),Path(__file__).with_name('shared_field_gs_read.py'),road,scope,
        operator/'read.json',operator/'constants.npz',operator/'low.npz']
    receipt=dict(schema='native_capacity_shared_initial.v1',complete=True,GPU_executed=False,
        native_forward_calls=0,optimizer_updates=0,active_rows=len(active_ids),scope_rows=int(eligible.sum()),
        topology_rows=world_count,readonly_rows=int((~eligible).sum()),readonly_RGB_exact=bool(np.array_equal(ci0[~eligible],parent_rgb[~eligible])),
        common_L_exact=True,H_initial_exact_zero=True,RGB_range=[float(ci0.min()),float(ci0.max())],
        changed_RGB_rows=int(np.any(ci0!=parent_rgb,axis=1).sum()),NPZ_reload_exact=reload_exact,field_reload_exact=field_exact,
        ci0_formula='parent_RGB; eligible rows <- float32(clip(B+KL@L_A200,0,1)), H=0 exactly',
        shared_initialization_for=['G-RGB','G-DETAIL','independent_ci_box_control'],
        native_initial_F0='pending actual 1920 ordinary GS rendering of this ci0; pixel low_A200 is not F0',
        scope_qualification='inherited candidate89887; new full-resolution K qualification belongs to new contract',
        prior_exposure='old A0 and avg A200 exposed to both views; current seen/held split is not blind',
        binding=binding,source_SHA256={str(p):file_sha(p)for p in fixed_paths},
        outputs={str(p):file_sha(p)for p in(path,fieldout)})
    if not(reload_exact and field_exact and receipt['readonly_RGB_exact']):raise RuntimeError('initial save/reload failed')
    save_json(output/'initial_receipt.json',receipt)
    return receipt


def quantiles(a):
    a=np.asarray(a)
    if not len(a):return dict(n=0)
    return dict(n=len(a),min=float(a.min()),p01=float(np.quantile(a,.01)),p50=float(np.quantile(a,.5)),
        p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()))


def audit_seen_footprints(q,output):
    """C_T=J_uv J_uv^T/12 from frozen seen contexts; no color/held labels."""
    q=Path(q);g=q/'gpu_20261119';b=g/'native_capacity_20261120_r1';high=g/'quality_bench_20261120_r1/highres'
    output=Path(output)
    if output.exists():raise FileExistsError('preserve prior audit: '+str(output))
    output.mkdir(parents=True);details={};bindings={}
    pooled={role:[]for role in('A','C','S')}
    for raw in(86,110):
        sp=b/f'qa/source_split_r1/{raw:03d}.npz';mp=high/f'maps/{raw:03d}.npz'
        with np.load(mp,allow_pickle=False)as z:
            ids=z['pixel_ids'];J=z['pixel_uv_jacobian'];hw=tuple(z['image_hw'])
        if J.shape!=(len(ids),2,2)or np.any(np.diff(ids)<=0):raise ValueError('compact map identity differs')
        covariance=np.einsum('nij,nkj->nik',J,J)/12.
        finite=np.isfinite(covariance).all((1,2));sigma=np.full((len(ids),2),np.nan)
        sigma[finite]=np.sqrt(np.linalg.eigvalsh(covariance[finite]).clip(0))
        with np.load(sp,allow_pickle=False)as z:
            if not np.array_equal(z['pixel_ids'],ids)or tuple(z['image_hw'])!=hw:raise ValueError('split/map pixel identity differs')
            frame={}
            for role in('A','C','S'):
                context=z[role+'_seen_context'];score=z[role+'_seen'];selected=context.ravel()[ids];v=sigma[selected&finite]
                frame[role]=dict(seen_score=int(score.sum()),seen_context=int(context.sum()),
                    seen_context_geometry_hit=int(selected.sum()),seen_context_geometry_missing=int(context.sum()-selected.sum()),
                    finite_footprints=len(v),minor_sigma_2cm_cells=quantiles(v[:,0]),major_sigma_2cm_cells=quantiles(v[:,1]),
                    major_sigma_le_half_cell=int((v[:,1]<=.5).sum()),minor_sigma_le_half_cell=int((v[:,0]<=.5).sum()))
                pooled[role].append(v)
        details[str(raw)]=frame;bindings.update({str(p):file_sha(p)for p in(sp,mp)})
    summary={role:dict(minor=quantiles(np.concatenate(parts)[:,0]),major=quantiles(np.concatenate(parts)[:,1]))
             for role,parts in pooled.items()}
    receipt=dict(schema='native_capacity_seen_teacher_footprints.v1',complete=True,GPU_executed=False,
        RGB_teacher_read=False,held_statistics_used=False,selection='exact frozen A/C/S_seen_context; geometry missing retained in counts',
        formula='C_T=J_uv J_uv^T/12; sqrt(eigenvalues), nominal 2cm chart cells',
        scope_note='footprint availability is descriptive, not a new ordinary GS fit or write gate',frames=details,pooled_seen=summary,
        minimal_band_rule=dict(H_grid_m=.02,H_highpass_sigma_m=.06,extra_source_blur=False,
            reason='anisotropic per-pixel footprints; avoid global isotropic second filtering before fixed physical-Sigma AA',
            frozen_GS_Sigma_antialias='existing GH31 source-normalized average',both_loss_arms_identical=True),
        source_SHA256=bindings)
    save_json(output/'teacher_footprint_seen.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initial','footprints']);p.add_argument('--q',required=True)
    p.add_argument('--asset');p.add_argument('--output',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('explicit CUDA_VISIBLE_DEVICES=\'\' required for CPU-only preparation')
    torch.set_num_threads(1);started=time.time()
    if a.mode=='initial':
        if not a.asset:p.error('--asset required')
        r=export_initial(a.q,a.asset,a.output)
    else:r=audit_seen_footprints(a.q,a.output)
    print(json.dumps(dict(complete=r['complete'],mode=a.mode,output=a.output,wall_s=time.time()-started)))


if __name__=='__main__':main()
