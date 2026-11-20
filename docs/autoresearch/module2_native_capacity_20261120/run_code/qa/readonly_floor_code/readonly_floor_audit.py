"""CPU-only fixed-domain readonly/color-range audit; never edits evaluation_r1.

Pointwise RGB intervals relax all spatial ci coupling. Finite native feature
arithmetic has no rigorous global error bound here, so floors are numeric proxies,
not a strict objective certificate or a semantic ghost detector. No HF bound.
"""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
from PIL import Image

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
B=Q/'gpu_20261119/native_capacity_20261120_r1';HIGH=Q/'gpu_20261119/quality_bench_20261120_r1/highres'
PARENT=Q.parent/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
REAL_ARMS=['F0','G_RGB','G_DETAIL','G_ORACLE']

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in(z.files if keys is None else keys)}
def quantile(a):
    a=np.asarray(a,np.float64)
    return dict(n=int(a.size),mean=float(a.mean())if a.size else None,
        **{k:(float(np.quantile(a,q))if a.size else None)for k,q in [('min',0),('p01',.01),('p05',.05),('p10',.1),('p25',.25),('p50',.5),('p75',.75),('p90',.9),('p95',.95),('p99',.99),('max',1)]})

def interval_floor(teacher_P,readonly,weight,alpha):
    target=np.asarray(teacher_P,np.float64);lo=np.asarray(readonly,np.float64);we=np.asarray(weight,np.float64)
    if target.shape!=lo.shape or target.shape[0]!=3 or we.shape!=lo.shape[1:]or alpha.shape!=we.shape:raise ValueError('CHW/HW interval shape mismatch')
    if not np.isfinite(target).all()or not np.isfinite(lo).all()or not np.isfinite(we).all():raise ValueError('finite intervals required')
    if np.any(we<0):raise ValueError('unit eligible weights must be nonnegative')
    hi=lo+we[None];projection=np.minimum(np.maximum(target,lo),hi);residual=target-projection
    straight=np.divide(residual,np.asarray(alpha,np.float64)[None],out=np.zeros_like(residual),where=np.asarray(alpha)[None]>0)
    return dict(residual=residual,straight_residual=straight,too_dark=(target<lo),too_bright=(target>hi))

def floor_summary(floor,mask):
    n=int(mask.sum())
    if not n:return dict(pixels=0,premult_MAE_lower_proxy=None,premult_MSE_lower_proxy=None,straight_MAE_lower_proxy=None,straight_MSE_lower_proxy=None,status='N/A_no_pixels')
    r=floor['residual'][:,mask];s=floor['straight_residual'][:,mask]
    return dict(pixels=n,premult_MAE_lower_proxy=float(np.abs(r).mean()),premult_MSE_lower_proxy=float((r*r).mean()),
        straight_MAE_lower_proxy=float(np.abs(s).mean()),straight_MSE_lower_proxy=float((s*s).mean()),
        pixels_target_below_readonly=int(floor['too_dark'][:,mask].any(0).sum()),
        pixels_target_above_readonly_plus_We=int(floor['too_bright'][:,mask].any(0).sum()),
        pixels_any_exact_interval_violation=int((np.abs(r).max(0)>0).sum()),
        pixels_straight_violation_gt1code=int((np.abs(s).max(0)>1/255).sum()),status='numeric_pointwise_RGB_proxy_only')

def mse_ratio(pred,target,alpha,floor,mask):
    n=int(mask.sum())
    if not n:return dict(pixels=0,premult_RGB_MSE=None,straight_RGB_MSE=None,premult_actual_over_floor=None,straight_actual_over_floor=None)
    e=np.asarray(pred,np.float64)[:,mask]-np.asarray(target,np.float64)[:,mask];a=alpha[mask]
    pm=float((e*e).mean());sm=float(((e/a[None])**2).mean());p0=floor['premult_MSE_lower_proxy'];s0=floor['straight_MSE_lower_proxy']
    return dict(pixels=n,premult_RGB_MSE=pm,straight_RGB_MSE=sm,premult_actual_over_floor=pm/p0 if p0 and p0>0 else None,
        straight_actual_over_floor=sm/s0 if s0 and s0>0 else None,
        zero_floor_ratio_policy='null when proxyfloor iszero; not infinity or successful fit')

def save_crop(path,rgb,bbox):
    x0,y0,x1,y1=bbox
    if rgb.ndim==3 and rgb.shape[0]==3:rgb=rgb.transpose(1,2,0)
    Image.fromarray(np.rint(np.clip(rgb[y0:y1,x0:x1],0,1)*255).astype(np.uint8)).save(path)

def category_overlay(rgb,domain,seen,held):
    base=np.clip(np.asarray(rgb).transpose(1,2,0),0,1).copy();other=domain&~seen&~held
    for mask,color in [(seen,[.10,.85,.35]),(held,[.75,.20,.90]),(other,[1.,.50,.05])]:
        base[mask]=.62*base[mask]+.38*np.asarray(color)
    return base,other

def audit(output):
    output=Path(output)
    if output.exists():raise FileExistsError('new audit directory required; original evaluation stays untouched')
    started=time.perf_counter();output.mkdir(parents=True);(output/'crops').mkdir();(output/'contexts').mkdir()
    manifest_path=B/'public1920/export_r1/packets_manifest.json';vis_path=B/'public1920/export_r1/visibility_manifest_r1.json';plan_path=B/'qa/final_eval_plan_r1.json'
    public=read(manifest_path);vis=read(vis_path);plan=read(plan_path)
    assert vis['complete']and vis['CPU_only']and vis['formal_scope_equal_old_scope']and sha(manifest_path)==vis['public_manifest_SHA256']
    packets={(r['arm'],r['raw']):r for r in public['rows']};sources={str(p):sha(p)for p in [manifest_path,vis_path,plan_path]}
    permission=load(Q/'permission/final_r1/teacher_evidence.npz',['active_world_rows','common_write_active']);active=permission['active_world_rows'];eligible=permission['common_write_active'];readonly_world=active[~eligible]
    original=load(PARENT/'road.npz');f0asset=Path(packets['F0',86]['asset_dir']);current=load(f0asset/'road.npz')
    unchanged={k:(v.shape==current[k].shape and v.dtype==current[k].dtype and v.tobytes()==current[k].tobytes())for k,v in original.items()if k!='reference_srgb'}
    if not all(unchanged.values()):raise ValueError('readonly basis geometry/opacity/source asset differs')
    readonly_rgb_exact=np.array_equal(original['reference_srgb'][readonly_world],current['reference_srgb'][readonly_world])
    if not readonly_rgb_exact:raise ValueError('currentF0 readonly colors differ from the cached native basis authority')
    if not np.array_equal(np.flatnonzero(current['active']),active):raise ValueError('active row identity differs')
    for p in [PARENT/'road.npz',f0asset/'road.npz',Q/'permission/final_r1/teacher_evidence.npz']:sources[str(p)]=sha(p)
    roipath=HIGH/'previews/roi_manifest.json';rois=read(roipath);sources[str(roipath)]=sha(roipath)
    rows=[];tile_rows=[];bindings=[];crop_rows=[]
    for raw in [86,110]:
        splitpath=B/f'qa/source_split_r1/{raw:03d}.npz';targetpath=HIGH/f'targets/{raw:03d}.npz';basispath=HIGH/f'native_r1/{raw:03d}/road_basis.npz'
        split=load(splitpath);basis=load(basispath,['baseline_rgb','eligible_rgb','readonly_rgb','direct_readonly_rgb','eligible_weight','alpha','K'])
        f0=load(packets['F0',raw]['path'],['road_rgb','road_alpha','K']);detail=load(packets['G_DETAIL',raw]['path'],['road_rgb'])['road_rgb']
        endpoint_predictions={arm:load(packets[arm,raw]['path'],['road_rgb','road_alpha'])for arm in REAL_ARMS}
        if not np.array_equal(f0['road_alpha'],basis['alpha'])or not np.array_equal(f0['road_alpha'],split['fixed_alpha'])or not np.array_equal(f0['K'],basis['K']):raise ValueError('basis native camera/alpha/K differs')
        b=basis['readonly_rgb'].astype(np.float64);w=basis['eligible_weight'].astype(np.float64);a=f0['road_alpha'].astype(np.float64)
        old_reconstruction=basis['readonly_rgb']+basis['eligible_rgb'];old_float_error=float(np.abs(old_reconstruction-basis['baseline_rgb']).max())
        old_double_error=float(np.abs(b+basis['eligible_rgb'].astype(np.float64)-basis['baseline_rgb']).max())
        direct_error=float(np.abs(b-basis['direct_readonly_rgb']).max())
        E0=f0['road_rgb'].astype(np.float64)-b
        initial_below=np.maximum(-E0,0.);initial_above=np.maximum(E0-w[None],0.)
        constant_receipt=read(HIGH/f'native_r1/{raw:03d}/road_constant_receipt.json')
        binding=dict(raw=raw,eligible_rows=int(eligible.sum()),readonly_rows=len(readonly_world),readonly_row_RGB_bitexact=True,all_nonRGB_fields_exact=True,
            basis_original_native_reconstruction_float32_max=old_float_error,basis_original_native_reconstruction_float64_max=old_double_error,
            algebraic_readonly_vs_independent_native_direct_readonly_max=direct_error,original_constant_probe=constant_receipt,
            initial_E0_definition='actual publicF0 P minus immutable native readonly contribution; algebraic residual, not a newly rendered eligible feature',
            initial_P_recomposition_float64_max=float(np.abs(b+E0-f0['road_rgb']).max()),
            initial_E0_lower_interval_violation_max=float(initial_below.max()),initial_E0_upper_interval_violation_max=float(initial_above.max()),
            fresh_F0_eligible_feature_pass=False,
            precision_note='Old native decomposition/constant identity plus byte-exact readonly rows and samecamera/alpha/order qualify reuse. Recomposition of residualE0 is algebraic, not an independent forward proof. Floors remain numerical proxies.')
        bindings.append(binding)
        masks={};targets=load(targetpath,['T_C','T_S'])
        ratio=np.divide(w,a,out=np.zeros_like(w),where=a>0)
        for role in ['C','S']:
            domain=split[role+'_pre_alpha'];seen=split[role+'_seen_context'];held=split[role+'_held_context'];held_world=split[role+'_held']
            full=domain&(a>=.95)&~split['K'];unmapped=full&(split['world_code']==0)
            categories=dict(full_stable_alpha95=full,seen=seen,held=held,unmapped_stable_alpha95=unmapped)
            if not np.array_equal(full,seen|held|unmapped):raise ValueError('full stable domain does not partition into seen/held/unmapped')
            teacherP=targets['T_'+role].astype(np.float64)*a[None];floor=interval_floor(teacherP,b,w,a)
            for name,mask in categories.items():
                fs=floor_summary(floor,mask);n=int(mask.sum())
                row=dict(raw=raw,source=role,split=name,We_over_alpha=quantile(ratio[mask]),
                    We_over_alpha_below={str(c):int((ratio[mask]<c).sum())for c in [.95,.5,.05]},
                    We_over_alpha_above1=int((ratio[mask]>1).sum()),floor=fs,endpoints=[],
                    teacher_role='frozen_pseudo_teacher_not_hidden_GT',held_prior_preexposed=True,blind_test=False)
                for arm in REAL_ARMS:
                    pred=endpoint_predictions[arm]
                    if not np.array_equal(pred['road_alpha'],a):raise ValueError('endpoint alpha differs')
                    row['endpoints'].append(dict(arm=arm,**mse_ratio(pred['road_rgb'],teacherP,a,fs,mask)))
                rows.append(row)
                for y0 in range(0,mask.shape[0],128):
                    for x0 in range(0,mask.shape[1],128):
                        sl=np.s_[y0:y0+128,x0:x0+128];m=mask[sl]
                        if not m.any():continue
                        fsmall={k:v[:,sl[0],sl[1]]for k,v in floor.items()};f=floor_summary(fsmall,m)
                        tile_rows.append(dict(raw=raw,source=role,split=name,tile_id=[y0//128,x0//128],bounds_y0_y1_x0_x1=[y0,min(y0+128,mask.shape[0]),x0,min(x0+128,mask.shape[1])],**f))
            roi=next(r for r in rois['ROIs']if r['raw']==raw and r['id']==role+'_native_source');box=roi['bbox_xyxy'];stem=f'{raw:03d}_{role}'
            save_crop(output/'crops'/f'{stem}_readonly_P.png',b,box)
            save_crop(output/'crops'/f'{stem}_eligible_We_over_alpha.png',np.repeat(ratio[None],3,axis=0),box)
            overlay,unqualified=category_overlay(detail,domain,seen,held)
            save_crop(output/'crops'/f'{stem}_training_context_overlay.png',overlay,box)
            world_overlay=np.clip(detail.transpose(1,2,0),0,1).copy();world_overlay[held_world]=.62*world_overlay[held_world]+.38*np.asarray([.75,.2,.90])
            save_crop(output/'crops'/f'{stem}_world_held_overlay.png',world_overlay,box)
            masks.update({role+'_actual_seen_context':seen,role+'_actual_held_context':held,role+'_world_held':held_world,role+'_unqualified_pre_alpha':unqualified})
            crop_rows.append(dict(raw=raw,source=role,bbox_xyxy=box,selection='unchanged frozen source-mask bbox+32px; no error-based crop',
                actual_seen_context=int(seen.sum()),held_stable_context=int(held.sum()),held_world_pre_alpha=int(held_world.sum()),
                unqualified_pre_alpha=int(unqualified.sum()),full_pre_alpha=int(domain.sum()),
                held_fraction_of_canonical_source=float(held_world.sum()/((domain&(split['world_code']>0)).sum()))if(domain&(split['world_code']>0)).any()else None,
                files=[f'crops/{stem}_{s}.png'for s in ['readonly_P','eligible_We_over_alpha','training_context_overlay','world_held_overlay']]))
        np.savez_compressed(output/'contexts'/f'{raw:03d}.npz',**masks)
        for p in [splitpath,targetpath,basispath,Path(packets['F0',raw]['path']),Path(packets['G_DETAIL',raw]['path'])]:sources[str(p)]=sha(p)
    for arm in REAL_ARMS:
        for raw in [86,110]:sources[packets[arm,raw]['path']]=sha(packets[arm,raw]['path'])
    write(output/'results.json',dict(schema='fixed_native_readonly_pointwise_RGB_floor.v1',complete=True,rows=rows,bindings=bindings,crops=crop_rows,source_SHA256=sources,
        readonly_row_RGB_exact=readonly_rgb_exact,nonRGB_fields=unchanged,seconds=time.perf_counter()-started,code_SHA256=sha(__file__),
        interval='readonly_rgb <= premult P <= readonly_rgb+eligible_unit_alphaT perRGB channel, ci in[0,1], fixed ordinaryGS positive weights',
        lower_proxy='distance of alpha*T to this pointwise interval; it relaxes cross-pixel sharedci coupling, RGB only, no HF lower bound',
        strict_global_certificate=False,semantic_ghost_detection=False,HF_bound_claim=False,
        numerical_caveat='native float32 feature decomposition and constant-color replay errors reported, not a rigorous global error envelope',
        endpoint_scope='F0+3 real formal arms only. IDEAL_ORACLE deliberately uses a different synthetic target and is not ranked against real pseudo teachers.',
        overlay_legend=dict(green='actual seen training context',purple='actual held stable context',orange='inside C/S pre-alpha but outside seen/held stable contexts',separate_world_overlay='all canonical2m held world source, including lowalpha; not a new GT split'),
        image_units='native1:1 crops; readonly is premult RGB. We/alpha is clipped0..1 only for grayscale display; numerical ratio is never clipped.',
        BUILD_QUERY='Main arms intentionally exclude new held labels. Full qualitative residual includes held and unqualified areas; it is not an all-labels BUILD capacity result.',
        original_evaluation_modified=False,GPU_executed=False,native_calls=0,optimizer_updates=0,human_verdict=None,production_promoted=False))
    write(output/'tiles.json',dict(rows=tile_rows,layout='128native pixels, fixed global origin0',worst=[r for raw in [86,110]for role in ['C','S']for split in ['full_stable_alpha95','seen','held']for r in sorted([r for r in tile_rows if r['raw']==raw and r['source']==role and r['split']==split],key=lambda r:r['straight_MSE_lower_proxy'],reverse=True)[:5]],
        selection='posthoc diagnostic ranking of all fixed source tiles by interval floor; not new training ROI or validation selection'))
    with(output/'interval_floors.csv').open('w',newline='',encoding='utf-8-sig')as f:
        flat=[]
        for r in rows:
            for e in r['endpoints']:flat.append(dict(raw=r['raw'],source=r['source'],split=r['split'],arm=e['arm'],pixels=r['floor']['pixels'],**{k:v for k,v in r['floor'].items()if k.endswith('_proxy')},**{k:v for k,v in e.items()if k not in ['arm','pixels','zero_floor_ratio_policy']}))
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(completed=True,rows=len(rows),output=str(output),seconds=time.perf_counter()-started,bindings=bindings)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'readonly_floor_audit_r1');a=p.parse_args();audit(a.output)
