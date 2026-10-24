"""V4.2设计用CPU审计：重算门限、逐门消融、异块颜色复核；不修改训练输入。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, importlib.util, json, time
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, laplace, distance_transform_edt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); start = time.perf_counter()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    sys.path.insert(0, str(a.campaign/'tools'))
    from prepare_v41_dense import cloud_and_boxes
    from v41_dense_geometry import measured_patch_evidence, project
    from prepare_v41_revision import native_rgb, sample_rgb
    base = a.repo/'scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020'
    spec = importlib.util.spec_from_file_location('legacy_geometry', base/'engine_overlay/utils/v3_road_geometry.py')
    g = importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    dense = Path(str(a.campaign)+'_dense_r7')
    dm = json.loads((dense/'manifest.json').read_text())
    bundle = Path(dm['source_bundle']);m = json.loads(bundle.read_text())
    with np.load(bundle.parent/m['units']) as z: u={k:z[k] for k in z.files}
    supported=u['surface_supported']; old=u['base_weight']>0; old_active=old.sum(0)>=2
    gates={k.removeprefix('gate_'):v for k,v in u.items() if k.startswith('gate_')}
    assert np.array_equal(np.logical_and.reduce(list(gates.values())),old)
    variants={'baseline':old, 'minimum_views_1':old}
    for drop in gates:
        variants['drop_'+drop]=np.logical_and.reduce([v for k,v in gates.items() if k!=drop]) & supported
    for names in [('road_support','measured_depth_match'),('boxes_clear','measured_depth_match'),('boxes_clear','mask_clear')]:
        variants['drop_'+'_and_'.join(names)]=np.logical_and.reduce([v for k,v in gates.items() if k not in names]) & supported
    conditions={name:[] for name in ['mask_margin_0','mask_margin_1','depth_0.15','depth_0.5','depth_0.8','cosine_0.06','area_0.1','quality_0','sparse_radius_3','native_pixel_convention','patch_strict','patch_no_box','patch_geometry_without_angle']}
    mismatches={k:0 for k in gates}; colors=[]; measurements=[]; pooled_ground=[]
    anchor=np.asarray(m['coordinate_anchor']);inv=np.linalg.inv(anchor)
    plane=np.asarray(json.loads(Path(m['original_readiness']).read_text())['SGCM_geometry']['ground_fit']['plane_anchor'])
    source=Path(m['original_engineering']).parent/'inputs/scene0008_089_137'
    rgbs=[np.asarray(Image.open(m['frames'][fid]['original_rgb']).convert('RGB').resize((480,320),Image.Resampling.LANCZOS)) for fid in m['frame_order']]
    sharp=np.array([np.var(laplace(rgb.astype(float).mean(2)/255.)) for rgb in rgbs]);quality=np.minimum(sharp/max(np.median(sharp),1e-9),1.)
    def rest(i,replace):return np.logical_and.reduce([replace.get(k,v[i]) for k,v in gates.items()]) & supported
    for i,fid in enumerate(m['frame_order']):
        fm=m['frames'][fid];K=np.asarray(fm['intrinsic']);w2c=np.asarray(fm['world_to_camera'])
        cloud,ground,boxes=cloud_and_boxes(source,fid,inv,plane);pooled_ground.append(g.transform(cloud[ground],inv))
        mask=np.zeros((320,480),bool)
        for path in fm['occluder_masks']['paths']:
            if path.endswith('.npz'):
                with np.load(path) as z: im=Image.fromarray(z['masks'].astype(bool).any(0))
            else:im=Image.open(path)
            ar=np.asarray(im.resize((480,320),Image.Resampling.NEAREST));mask|=(ar.any(2) if ar.ndim==3 else ar)>0
        dep,valid,road=g.sparse_depth_maps(cloud,ground,K,w2c)
        frame=dict(K=K,w2c=w2c,depth=dep,valid=valid,road=road,boxes=boxes,occluder=mask,rgb=rgbs[i],quality=quality[i])
        _,uv,_,_,calc=g.source_observation(u['world'],u['normal_world'],supported,frame,m['policy'])
        for k in gates:mismatches[k]+=int((calc[k]!=gates[k][i]).sum())
        for margin in [0,1]:
            guarded=binary_dilation(mask,iterations=margin) if margin else mask
            conditions[f'mask_margin_{margin}'].append(rest(i,{'mask_clear':g.sample(guarded,uv)==0}))
        zq=g.transform(u['world'],w2c)[:,2]; measured=g.sample(dep,uv); depth_valid=g.sample(valid,uv)>.99
        for tolerance in [.15,.5,.8]:conditions[f'depth_{tolerance}'].append(rest(i,{'measured_depth_match':depth_valid & (np.abs(measured-zq)<tolerance)}))
        origin=np.linalg.inv(w2c)[:3,3];r=origin-u['world'];dist=np.linalg.norm(r,axis=1)
        cosine=np.abs((r/np.maximum(dist[:,None],1e-9)*u['normal_world']).sum(1))
        area=K[0,0]*K[1,1]*.2**2*cosine/np.maximum(zq**2,1e-6)
        conditions['cosine_0.06'].append(rest(i,{'angle_ok':cosine>=.06}))
        conditions['area_0.1'].append(rest(i,{'resolution_ok':area>=.1}))
        conditions['quality_0'].append(rest(i,{'quality_ok':np.ones(len(supported),bool)}))
        # 单独扩大稀疏深度近邻半径；允许跨边界，故只作为风险明确的诊断上限。
        cuv,cz,inside=g.project(cloud,K,w2c);ids=np.flatnonzero(inside);ids=ids[np.argsort(cz[ids])[::-1]]
        px=np.floor(cuv[ids]+.5).astype(int); nearest=np.full((320,480),np.inf);gr=np.zeros((320,480),bool)
        for j,(x,y) in zip(ids,px):nearest[y,x]=cz[j];gr[y,x]=ground[j]
        dd,ii=distance_transform_edt(~np.isfinite(nearest),return_indices=True);vv=dd<=3
        d3=np.where(vv,nearest[tuple(ii)],0);r3=gr[tuple(ii)]&vv
        conditions['sparse_radius_3'].append(rest(i,{'road_support':g.sample(r3,uv)>.99,'measured_depth_match':(g.sample(vv,uv)>.99)&(np.abs(g.sample(d3,uv)-zq)<.3)}))
        # 原中心门限为K投影直接采样；原生相机以像素中心+0.5，单列半像素约定。
        nuv,nz=project(u['world'],K,w2c)
        npx=np.floor(cuv[ids]-.5+.5).astype(int);nin=(npx[:,0]>=0)&(npx[:,0]<480)&(npx[:,1]>=0)&(npx[:,1]<320)
        nd=np.full((320,480),np.inf);ng=np.zeros((320,480),bool)
        for j,(x,y) in zip(ids[nin],npx[nin]):nd[y,x]=cz[j];ng[y,x]=ground[j]
        nn,ni=distance_transform_edt(~np.isfinite(nd),return_indices=True);nv=nn<=1.5;nd=np.where(nv,nd[tuple(ni)],0);ng=ng[tuple(ni)]&nv
        conditions['native_pixel_convention'].append(rest(i,{'in_view':(nz>.1)&(nuv[:,0]>=0)&(nuv[:,0]<479)&(nuv[:,1]>=0)&(nuv[:,1]<319),'mask_clear':g.sample(binary_dilation(mask,iterations=2),nuv)==0,'road_support':g.sample(ng,nuv)>.99,'measured_depth_match':(g.sample(nv,nuv)>.99)&(np.abs(g.sample(nd,nuv)-nz)<.3)}))
        ok,pdepth,_,conflict=measured_patch_evidence(cloud,ground,K,w2c,u['world'],nuv,u['normal_world'])
        guard_clear=g.sample(binary_dilation(mask,iterations=2),nuv)<=1e-7
        iv=(nz>.1)&(nuv[:,0]>=0)&(nuv[:,0]<479)&(nuv[:,1]>=0)&(nuv[:,1]<319)
        patch=ok & supported & guard_clear & iv & gates['angle_ok'][i]&gates['resolution_ok'][i]&gates['quality_ok'][i]
        conditions['patch_strict'].append(patch & gates['boxes_clear'][i]);conditions['patch_no_box'].append(patch)
        conditions['patch_geometry_without_angle'].append(ok & supported & guard_clear & iv & gates['boxes_clear'][i])
        colors.append(sample_rgb(native_rgb(fm['original_rgb']),nuv));measurements.append(conflict)
        if i%10==0:print(json.dumps({'frame':fid,'stage':'CPU source replay'}),flush=True)
    variants.update({k:np.asarray(v) for k,v in conditions.items()})
    # 框相交只在另一个原本框外来源存在时可复核；新候选不能循环担保。
    strict=variants['patch_strict'];clear_elsewhere=(strict.sum(0)[None]-strict)>0
    variants['patch_box_recheck']=strict | (variants['patch_no_box']&clear_elsewhere)
    colors=np.asarray(colors); measurements=np.asarray(measurements)
    split=(np.arange(49)//5)%2==0
    def crosscheck(accept):
        fit=accept & split[:,None];test=accept & ~split[:,None]
        nfit=fit.sum(0);ntest=test.sum(0)
        fc=np.sum(colors*fit[:,:,None],axis=0)/np.maximum(nfit[:,None],1)
        tc=np.sum(colors*test[:,:,None],axis=0)/np.maximum(ntest[:,None],1)
        err=np.abs(fc-tc).mean(1);eligible=(nfit>=2)&(ntest>=2)
        return eligible, eligible&(err<=.05),err
    patch_e,patch_c,patch_err=crosscheck(variants['patch_box_recheck'])
    relaxed_e,relaxed_c,relaxed_err=crosscheck(variants['patch_geometry_without_angle'])
    rows={};selected={};actors={str(actor):np.unique(u['query_unit_id'][u['query_actor']==actor]) for actor in m['actors']}
    for name,accept in variants.items():
        minviews=1 if name=='minimum_views_1' else 2
        chosen=(accept.sum(0)>=minviews)&supported;new=chosen & ~old_active;selected[name]=chosen
        own_e,own_c,own_err=crosscheck(accept)
        rows[name]={'eligible_supported_units':int(chosen.sum()),'added_vs_1919':int(new.sum()),'lost_vs_1919':int((old_active & ~chosen).sum()),'source_observations':int(accept[:,supported].sum()),'added_with_disjoint_block_patch_color_check':int((new & patch_c).sum()),'added_with_two_strict_patch_views':int((new & (strict.sum(0)>=2)).sum()),'added_own_gate_block_testable':int((new & own_e).sum()),'added_own_gate_block_color_consistent':int((new & own_c).sum()),'added_patch_no_angle_testable':int((new & relaxed_e).sum()),'added_patch_no_angle_color_consistent':int((new & relaxed_c).sum()),'added_local_x_quantiles_m':np.quantile(u['xy_local'][new,0],[0,.25,.5,.75,1]).tolist() if new.any() else [],'by_actor_query_unit_union':{actor:{'total':len(ids),'supported':int(supported[ids].sum()),'eligible':int(chosen[ids].sum()),'added':int(new[ids].sum())} for actor,ids in actors.items()}}
    allground=np.concatenate(pooled_ground);_,ix=np.unique(np.floor(allground[:,:2]/.1).astype(int),axis=0,return_index=True);allground=allground[ix]
    from scipy.spatial import cKDTree
    tree=cKDTree(allground[:,:2]);radius={}
    for rad in [.6,.8,1.,1.2,1.6]:
        distances,_=tree.query(u['xy_local'],k=8,distance_upper_bound=rad);s=np.isfinite(distances).sum(1)>=3
        radius[str(rad)]={'geometric_candidates':int(s.sum()),'added_vs_0_8':int((s&~supported).sum()),'warning':'support radius only; no new height/normal/color or visibility validation'}
    funnel=[];cumulative=np.ones_like(old)
    for name,gate in gates.items():
        cumulative &= gate;funnel.append({'gate':name,'accepted_observations':int(cumulative.sum()),'units_two_views':int((cumulative.sum(0)>=2).sum())})
    # 保存逐点表，便于定位实际新增/排除点；不是仅输出总数。
    np.savez_compressed(a.output/'audit_arrays.npz',unit_ids=u['unit_ids'],xy=u['xy_local'],world=u['world'],supported=supported,baseline_active=old_active,patch_crosscheck=patch_c,patch_crosscheck_error=patch_err,**{'source_count_'+k:v.sum(0) for k,v in variants.items()},**{'eligible_'+k:v for k,v in selected.items()})
    result=dict(version='V42_SOURCE_GATE_AUDIT_V2',bundle=str(bundle),gate_replay_mismatch_counts=mismatches,baseline={'all_units':len(supported),'created':int(supported.sum()),'two_or_more':int(old_active.sum()),'one':int(((old.sum(0)==1)&supported).sum()),'zero':int(((old.sum(0)==0)&supported).sum())},variants=rows,ordered_funnel=funnel,geometric_support_radius=radius,patch_crosscheck={'two_temporal_blocks_with_two_sources_each':int(patch_e.sum()),'block_rgb_mae_le_0_05':int(patch_c.sum()),'no_angle_testable':int(relaxed_e.sum()),'no_angle_color_consistent':int(relaxed_c.sum()),'note':'disjoint source-block diagnostic, not independent ground truth; geometry derived from all49 frames. Original-angle crosscheck cannot reject angle-relaxed candidates; own-gate and no-angle patch checks reported separately.'},quality_accepted=False,human_verdict=None,GPU_used=False,optimizer_updates=0,elapsed_seconds=time.perf_counter()-start,limitations=['More eligible points is not better reconstruction. No retraining or renderer quality measured.','Drop-gate variants are diagnostic upper bounds, not safe policies.','Threshold-only candidates still need visibility, geometric and appearance verification.','Surface radius increase changes potential geometry; no inserted Gaussian or new labels here.','RGB block check is a proxy and not proof for truly hidden texture.','Patch crosschecks retain small-triangle and mask constraints; missing confirmation is inconclusive, not evidence a candidate is false.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')
    print(json.dumps({'summary':str(a.output/'summary.json'),'replay_mismatches':mismatches,'variants':{k:{q:v[q] for q in ['eligible_supported_units','added_vs_1919','added_with_disjoint_block_patch_color_check']} for k,v in rows.items()},'elapsed':result['elapsed_seconds']}),flush=True)

if __name__=='__main__': main()
