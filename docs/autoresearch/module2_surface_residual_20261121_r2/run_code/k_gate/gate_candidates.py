"""Native joint-K feasibility only: two fixed states, zero parameter updates."""
from pathlib import Path
import copy,hashlib,json,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_surface_residual_20261121_r2';OUT=ROOT/'k_gate'
PARENT=C/'road_surface_adaptive_20261121_r1/appearance_fit/asset'
R1=PARENT.parents[1];REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
FRAMES=(86,110,134,179,198);OWNER='residual_r2_K_gate'
sys.path[:0]=[str(REPO/'scripts/worldsim_v82'),str(C/'road_capacity_gpu_20261118_r1/code')]
import torch

def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def tensor(a,dtype=torch.float32):return torch.as_tensor(np.array(a,copy=True),device='cuda',dtype=dtype)

def probe_asset(d):
    from drivingeditor_dynamic_edit.persistent_road_handoff import save_persistent_road_handoff
    asset=OUT/'probe_asset';r=load(PARENT/'road.npz');p=load(PARENT/'provenance.npz');n=len(r['active'])
    kind=d['candidate_kind'];act=np.flatnonzero(kind==0);app=np.flatnonzero(kind==1);m=len(app)
    worldids=np.full(len(kind),-1,np.int64);worldids[act]=d['existing_world_rows'][act]
    assert not r['active'][worldids[act]].any()
    assert len(np.unique(worldids[act]))==len(act)
    worldids[app]=n+np.arange(m)
    oldactive=r['active'].copy();r['active'][worldids[act]]=True
    # The activation initial state retains its existing opacity and scales.
    initial_opacity=np.full(len(kind),.1,np.float32);initial_opacity[act]=r['road_opacity'][worldids[act]]
    initial_sigma=d['scales_m'].copy();initial_sigma[act]=r['gaussian_sigma_m'][worldids[act]]
    T=r['model_from_world'];new=dict(active=np.ones(m,bool),model_xyz=d['world_xyz'][app]@T[:3,:3].T+T[:3,3],
        normal_world=d['normal_world'][app],gaussian_sigma_m=initial_sigma[app],reference_srgb=np.full((m,3),.5,np.float32),
        road_opacity=initial_opacity[app],minimum_model_depth=np.zeros(m,np.float32))
    for k,v in list(r.items()):
        if v.ndim and v.shape[0]==n:r[k]=np.concatenate([v,new[k].astype(v.dtype)])
    for k,v in list(p.items()):
        if v.ndim and v.shape[0]==n:
            fill=-1 if v.dtype.kind=='i' else 0;p[k]=np.concatenate([v,np.full((m,*v.shape[1:]),fill,dtype=v.dtype)])
    p['world_xyz'][n:]=d['world_xyz'][app];p['known'][n:]=False;p['unknown'][n:]=True
    p['source'][n:]=0;p['confidence'][n:]=0;p['observed_reference_srgb'][n:]=np.nan
    p['world_keys'][n:]=np.c_[np.full(m,-2),-1-np.arange(m)]
    p['residual_r2_candidate_index']=np.full(n+m,-1,np.int64);p['residual_r2_candidate_index'][worldids]=np.arange(len(kind))
    assert np.array_equal(r['gaussian_sigma_m'][:n][oldactive],load(PARENT/'road.npz')['gaussian_sigma_m'][oldactive])
    asset.mkdir(parents=True,exist_ok=False);np.savez_compressed(asset/'road.npz',**r);np.savez_compressed(asset/'provenance.npz',**p)
    save_persistent_road_handoff(PARENT/'persistent/scene.json',PARENT/'persistent/proof.json',asset/'road.npz',asset/'persistent',
        training_record=dict(role='K_gate_only',optimizer_updates=0,activation_rows=len(act),appended_probe_rows=m,
            spatial_new_rows=int((~d['opacity_stack_on_parent_active'][app]).sum()),coincident_stack_rows=int(d['opacity_stack_on_parent_active'][app].sum())))
    return asset,worldids,initial_opacity,initial_sigma

def main():
    if (OUT/'started.json').exists():raise FileExistsError('preserve prior gate; no implicit replay')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);start=time.monotonic();ctx=None;nativeB=0
    d=load(ROOT/'proposal/candidates.npz');N=len(d['candidate_kind']);kind=d['candidate_kind'];stack=d['opacity_stack_on_parent_active'].astype(bool)
    assert not np.any(stack&(kind==0))
    frozen=[PARENT/'road.npz',PARENT/'provenance.npz',R1/'status.json',R1/'resource_ledger.json',R1/'receipt.json',ROOT/'proposal/candidates.npz',ROOT/'proposal/manifest.json']
    binding={str(p):sha(p) for p in frozen}
    write(OUT/'started.json',dict(optimizer_updates=0,input_SHA256=binding,states=['initial','max'],max_nativeF=180))
    write(OUT/'gpu_lease.json',dict(active=True,owner=OWNER,gpu_authorized=True,optimizer_updates_allowed=0,GPU_keep_on=True))
    try:
        # Same-center is a point-type label, never a Poisson radius rejection.
        from scipy.spatial import cKDTree
        base=load(PARENT/'road.npz');T=base['model_from_world']
        model=(d['world_xyz']@T[:3,:3].T+T[:3,3]).astype(np.float32)
        nearest,index=cKDTree(base['model_xyz'][base['active']].astype(np.float32).astype(np.float64)).query(model.astype(np.float64),k=1)
        coincident=nearest==0
        assert np.array_equal(coincident[kind==1],stack[kind==1]),'CPU F32 same-center labels differ'
        asset,worldids,initial_opacity,initial_sigma=probe_asset(d)
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from road_capacity_native import NativeRoadContext
        scene,m,loader=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
        for obj in (scene.static,scene.dynamic,scene.environment):
            for p in obj.parameters() if hasattr(obj,'parameters') else ():p.requires_grad_(False)
        assert all((h['width'],h['height'])==(1152,768) for r,h in m.camera_headers().items() if r in FRAMES)
        ctx=NativeRoadContext(scene,m,loader,asset,FRAMES,OUT/'gpu_lease.json',OWNER)
        positions=np.searchsorted(ctx.active_world_rows,worldids);assert np.array_equal(ctx.active_world_rows[positions],worldids)
        pos=tensor(positions,torch.long);base_packet=ctx.base_packet(86)
        cap=np.repeat(np.min(d['max_sigma_m'],axis=1)[:,None],2,axis=1)
        mu=float(ctx.road['meter_to_model']);maskK={r:tensor(ctx.K_masks[r],torch.bool) for r in FRAMES}
        parent={r:load(R1/'appearance_fit/rendered'/f'{r:03d}.npz') for r in FRAMES}
        vjp_mass=np.zeros((N,2,5),np.float64);unsafe=np.zeros((N,2,5),bool);reason=np.zeros(N,np.uint8);calls=[]

        def render(raw,alive,state,features=True,need_grad=False):
            nonlocal nativeB
            if ctx.native_calls>=180:raise RuntimeError('finite K-gate nativeF cap reached; no training')
            rows,owners,camera,_=scene._rows(raw,(10,),(),False)
            rows={k:(v.detach() if torch.is_tensor(v) else v) for k,v in rows.items()}
            rows['mask']=rows['mask']&~scene._contains(owners,(10,));offset=len(rows['means3D'])-len(ctx.base_rgb)
            assert torch.equal(rows['means3D'][offset:],ctx.base_packet(raw)['means3D'])
            fullpos=offset+pos;alive_gpu=tensor(alive,torch.bool)
            rows['mask']=rows['mask'].clone();rows['mask'][fullpos]=rows['mask'][fullpos]&alive_gpu
            op=base_packet['opacities'].reshape(-1)[pos] if state==0 else torch.ones(N,device='cuda')
            newscale=base_packet['scales'][pos] if state==0 else torch.cat([tensor(cap)*mu,base_packet['scales'][pos,2:]],1)
            rows['opacities']=rows['opacities'].index_copy(0,fullpos,op[:,None])
            rows['scales']=rows['scales'].index_copy(0,fullpos,newscale)
            colors=torch.full((N,3),.5,device='cuda',requires_grad=need_grad)
            rows['colors_precomp']=rows['colors_precomp'].index_copy(0,fullpos,colors)
            if features:
                f=torch.zeros((len(rows['means3D']),24),device='cuda');f[:,:2]=rows['features'][:,:2];f[fullpos,2]=alive_gpu.float();rows['features']=f
            ctx.native_calls+=1
            with torch.set_grad_enabled(need_grad):
                out=scene._bindings().raster(camera,scene.pipe,0,rows,scene._background)
                rgb=out['rgb']+(1-out['alpha'])*scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                K=maskK[raw];unit=out['extra'][0] if features else None
                maximum=float(unit[K].abs().max()) if features else None
                mass=None;err=None
                if need_grad and maximum and maximum>0:
                    g=torch.autograd.grad(rgb[0,K].sum(),colors)[0][:,0];nativeB+=1
                    if not torch.isfinite(g).all() or float(g.min()) < -1e-9:raise RuntimeError('invalid native color VJP attribution')
                    mass=g.detach().cpu().numpy().astype(np.float64)
                    expected=float(unit[K].sum());err=abs(float(mass.sum())-expected)/max(abs(expected),1e-12)
                    if err>5e-4:raise RuntimeError('native VJP does not conserve unit alphaT '+str(err))
                rgbmax=float((rgb-tensor(parent[raw]['joint_rgb'])).abs()[:,K].max())
                amax=float((out['alpha'].squeeze()-tensor(parent[raw]['joint_alpha'])).abs()[K].max())
            calls.append(dict(raw=raw,state=['initial','max'][state],feature_unit=features,alive=int(alive.sum()),
                unit_K_max=maximum,joint_K_RGB_max=rgbmax,joint_K_alpha_max=amax,VJP_relative_error=err))
            return maximum,rgbmax,amax,mass

        # Native2 baseline checks protect against a feature/row-topology ABI mismatch.
        empty=np.zeros(N,bool)
        for raw in FRAMES:
            _,rgb,alpha,_=render(raw,empty,0,features=False)
            if rgb!=0 or alpha!=0:raise RuntimeError('masked-candidate native baseline differs from pinned parent')

        def bad(raw,ids,state):
            alive=np.zeros(N,bool);alive[ids]=True
            u,r,a,_=render(raw,alive,state,features=True,need_grad=False)
            return u!=0 or r!=0 or a!=0
        def locate_silent(raw,ids,state):
            # Only used for actual RGB/alpha changes without visible new unit mass.
            if len(ids)==1:return ids
            half=len(ids)//2
            for part in (ids[:half],ids[half:]):
                if len(part) and bad(raw,part,state):return locate_silent(raw,part,state)
            # No single half reproduces the observed change: preserve the interaction
            # group as unresolved instead of claiming every member caused the change.
            return np.zeros(0,np.int64)

        def gate(group,label):
            alive=group.copy();unresolved=[]
            for iteration in range(12):
                remove=np.zeros(N,bool);silent=[];all_zero=True
                for state in (0,1):
                    for vi,raw in enumerate(FRAMES):
                        u,r,a,mass=render(raw,alive,state,features=True,need_grad=True)
                        if mass is not None:
                            hit=alive&(mass>0);vjp_mass[:,state,vi]=np.maximum(vjp_mass[:,state,vi],mass)
                            unsafe[hit,state,vi]=True;remove|=hit;reason[hit]=1
                        if u!=0 or r!=0 or a!=0:all_zero=False
                        if u==0 and (r!=0 or a!=0):silent.append((raw,state,vi))
                if all_zero:
                    # Independent unmarked native RGB/alpha check at the same two states.
                    for state in (0,1):
                        for raw in FRAMES:
                            _,r,a,_=render(raw,alive,state,features=False)
                            if r!=0 or a!=0:raise RuntimeError('final ordinary-native K RGB/alpha is not exact0')
                    return alive,unresolved
                if not remove.any() and silent:
                    raw,state,vi=silent[0];ids=np.flatnonzero(alive)
                    # Projection intersection only narrows attribution; it never decides rejection.
                    focus=ids[d['projection_K_conflict'][ids,vi]]
                    if not len(focus):focus=ids
                    culprit=locate_silent(raw,focus,state)
                    if len(culprit):remove[culprit]=True;unsafe[culprit,state,vi]=True;reason[culprit]=2
                    else:
                        unresolved.extend(focus.tolist());remove[focus]=True;reason[focus]=3
                if not remove.any():raise RuntimeError('nonzero K result without attributable candidate')
                alive &= ~remove
                write(OUT/(label+'_progress.json'),dict(iteration=iteration+1,remaining=int(alive.sum()),removed=int(remove.sum()),nativeF=ctx.native_calls))
            raise RuntimeError('finite fixed-point K gate did not stabilize')

        # Same-center opacity stacks are diagnostic-only and cannot mask a spatial candidate.
        spatial_safe,unresolved_a=gate(~stack,'spatial_and_activation')
        stack_safe,unresolved_b=gate(stack,'coincident_stack_diagnostic')
        safe=spatial_safe|stack_safe
        def subset(mask,path):
            values={k:(v[mask] if v.ndim and len(v)==N else v) for k,v in d.items()}
            values['original_candidate_index']=np.flatnonzero(mask)
            np.savez_compressed(path,**values)
        subset(spatial_safe,OUT/'safe_candidates.npz');subset(stack_safe,OUT/'safe_stack_diagnostic.npz');subset(~safe,OUT/'rejected_candidates.npz')
        np.savez_compressed(OUT/'per_candidate_K_evidence.npz',candidate_id=d['candidate_id'],safe=safe,
            training_eligible=spatial_safe,unsafe_state_view=unsafe,VJP_K_mass=vjp_mass,rejection_reason=reason,
            coincident_with_parent_F32=coincident,projection_K_conflict=d['projection_K_conflict'])
        counts={}
        for name,mask in [('existing_inactive',kind==0),('spatially_new_append',(kind==1)&~stack),('coincident_opacity_stack',stack)]:
            counts[name]=dict(proposed=int(mask.sum()),safe=int((mask&safe).sum()),not_qualified=int((mask&~safe).sum()))
        byview={str(r):{name:int((mask&unsafe[:,:,vi].any(1)).sum()) for name,mask in [('existing_inactive',kind==0),('spatially_new_append',(kind==1)&~stack),('coincident_opacity_stack',stack)]} for vi,r in enumerate(FRAMES)}
        summary=dict(complete=True,counts=counts,unsafe_view_counts=byview,
            safe_projection_conflict_retained=int((safe&d['projection_K_conflict'].any(1)).sum()),
            training_eligible=int(spatial_safe.sum()),stacks_excluded_from_training=True,
            unresolved_joint_interaction_candidate_ids=unresolved_a+unresolved_b,
            nativeF=ctx.native_calls,nativeB=nativeB,optimizer_updates=0,states=['future_initial','opacity1_original_max_sigma'],
            gate='all5 unit contribution and jointRGB/alpha exact0 in both fixed states; rejected set removed and surviving set revalidated',
            boundary='two fixed states are not an entire parameter-interval theorem; future training requires real checkpoint/final K validation',
            no_CPU_projection_rejection=True,no_training_started=True,GPU_keep_on=True,seconds=time.monotonic()-start)
        if {str(p):sha(p) for p in frozen}!=binding:raise RuntimeError('first-wave/input files changed')
        write(OUT/'native_calls.json',dict(rows=calls,nativeF=ctx.native_calls,nativeB=nativeB));write(OUT/'summary.json',summary)
        write(OUT/'receipt.json',dict(**summary,input_SHA256=binding,first_wave_formal_unchanged=True,source_code=str(Path(__file__)),source_code_SHA256=sha(__file__)))
        print(json.dumps(summary),flush=True)
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),nativeF=0 if ctx is None else ctx.native_calls,nativeB=nativeB,optimizer_updates=0));raise
    finally:
        if ctx is not None:ctx.close()
        lease=read(OUT/'gpu_lease.json');lease.update(active=False);write(OUT/'gpu_lease.json',lease)

if __name__=='__main__':main()
