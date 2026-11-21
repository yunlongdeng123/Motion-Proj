"""Fixed R2 inputs and max-state gate for reusing R1-added rows; no training."""
from pathlib import Path
import hashlib,json,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_surface_residual_20261121_r2';R1=C/'road_surface_adaptive_20261121_r1'
PARENT=R1/'appearance_fit/asset';OUT=ROOT/'train_inputs'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');FRAMES=(86,110,134,179,198)
sys.path[:0]=[str(R1/'code'),str(REPO/'scripts/worldsim_v82'),str(C/'road_capacity_gpu_20261118_r1/code')]
from run_adaptive import geometry_render
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists() and read(OUT/'manifest.json').get('complete'):return
    d=load(ROOT/'k_gate/safe_candidates.npz');kind=d['candidate_kind']
    assert int((kind==0).sum())==13 and int((kind==1).sum())==1071
    assert not d['opacity_stack_on_parent_active'].any()
    for name,mask in [('activation',kind==0),('append',kind==1)]:
        np.savez_compressed(OUT/(name+'.npz'),**{k:v[mask] if v.ndim and len(v)==len(kind) else v for k,v in d.items()})
    raw=load(ROOT/'proposal/candidates.npz');stack=raw['opacity_stack_on_parent_active']
    ids=raw['parent_active_F32_model_center_world_row'][stack].astype(np.int64)
    original_candidate=raw['candidate_id'][stack]
    p=load(PARENT/'provenance.npz');road=load(PARENT/'road.npz')
    newgroup=p['adaptive_activate_existing']|p['adaptive_append_new_geometry'];eligible=newgroup[ids]&road['active'][ids]
    ids=ids[eligible];original_candidate=original_candidate[eligible]
    assert len(np.unique(ids))==len(ids)
    old_total=int(read(R1/'proposal/manifest.json')['parent_total'])
    oldapp=load(R1/'proposal/append.npz');oldact=load(R1/'proposal/activation.npz')
    expected=read(R1/'dense_init/asset/extension_receipt.json')['append_source_SHA256']
    assert sha(R1/'proposal/append.npz')==expected
    cap=[];minimum=[];original_sigma=[];source_index=[]
    for row in ids:
        if p['adaptive_append_new_geometry'][row]:
            ix=int(row-old_total);s=oldapp;tag='append'
        else:
            found=np.flatnonzero(oldact['existing_world_rows']==row);assert len(found)==1;ix=int(found[0]);s=oldact;tag='activation'
        assert np.max(np.abs(s['world_xyz'][ix]-p['world_xyz'][row]))<1e-6
        c=np.repeat(np.min(s['max_sigma_m'][ix]),2);sig=s['scales_m'][ix]
        cap.append(c);minimum.append(sig[:2]*.5);original_sigma.append(sig);source_index.append(dict(world_row=int(row),source=tag,index=ix))
    cap=np.asarray(cap);minimum=np.asarray(minimum);original_sigma=np.asarray(original_sigma)
    proposed=dict(ids=ids,world=p['world_xyz'][ids],normal=road['normal_world'][ids],sigma=road['gaussian_sigma_m'][ids],
        cap=cap,minimum=minimum,original_sigma=original_sigma,source_candidate_id=original_candidate)
    np.savez_compressed(OUT/'reuse_proposed.npz',**proposed)
    write(OUT/'reuse_caps_authority.json',dict(R1_provenance_SHA256=sha(PARENT/'provenance.npz'),
        original_append_proposal_SHA256=expected,original_activation_proposal_SHA256=sha(R1/'proposal/activation.npz'),
        interpretation='R1 provenance identity resolves original immutable proposal cap; generic legacy phase1 cap is not used',rows=source_index))
    lease=OUT/'reuse_K/gpu_lease.json';write(lease,dict(active=True,owner='r2_reuse_gate',gpu_authorized=True,optimizer_updates_allowed=0))
    ctx=None;start=time.monotonic();checks=[];passed=True
    try:
        import torch
        torch.set_num_threads(2)
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from road_capacity_native import NativeRoadContext
        scene,m,loader=load_persistent_handoff(PARENT/'persistent/scene.json',PARENT/'persistent/proof.json')
        for obj in (scene.static,scene.dynamic,scene.environment):
            for par in obj.parameters() if hasattr(obj,'parameters') else ():par.requires_grad_(False)
        ctx=NativeRoadContext(scene,m,loader,PARENT,FRAMES,lease,'r2_reuse_gate')
        positions=np.searchsorted(ctx.active_world_rows,ids);assert np.array_equal(ctx.active_world_rows[positions],ids)
        pos=torch.as_tensor(positions,device='cuda',dtype=torch.long);base=ctx.base_packet(86)
        opacity=base['opacities'].detach().reshape(-1).clone().index_copy(0,pos,torch.ones(len(ids),device='cuda'))
        sig=torch.cat([torch.as_tensor(cap,device='cuda',dtype=torch.float32)*float(road['meter_to_model']),base['scales'][pos,2:]],1)
        scales=base['scales'].detach().clone().index_copy(0,pos,sig)
        with torch.no_grad():
            for frame in FRAMES:
                out=geometry_render(ctx,frame,opacity,scales,joint=True,feature_ids=pos);K=ctx.K_masks[frame]
                parent=load(R1/'appearance_fit/rendered'/f'{frame:03d}.npz')
                unit=float(out['new_weight'][K].abs().max());rgb=float(np.abs(out['rgb'].cpu().numpy()-parent['joint_rgb'])[:,K].max())
                alpha=float(np.abs(out['alpha'].cpu().numpy()-parent['joint_alpha'])[K].max())
                checks.append(dict(raw=frame,unit_K_max=unit,joint_K_RGB_max=rgb,joint_K_alpha_max=alpha))
                passed &= unit==0 and rgb==0 and alpha==0
        write(OUT/'reuse_K/receipt.json',dict(complete=True,passed=passed,proposed_reuse=len(ids),checks=checks,
            nativeF=ctx.native_calls,nativeB=0,optimizer_updates=0,seconds=time.monotonic()-start,
            failure_policy='if group fails, all proposed reuse rows retain parent values; C expansion still proceeds'))
        keep=np.ones(len(ids),bool) if passed else np.zeros(len(ids),bool)
        reuse={k:v[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(ids) else v for k,v in proposed.items()}
        np.savez_compressed(OUT/'reuse_existing.npz',**reuse)
        actids=d['existing_world_rows'][kind==0].astype(np.int64);appids=len(road['active'])+np.arange(int((kind==1).sum()))
        allowed=np.r_[actids,appids,reuse['ids']];assert len(np.unique(allowed))==len(allowed)
        np.savez_compressed(OUT/'allowed_color_world_rows.npz',world_row_ids=allowed,
            group=np.r_[np.full(len(actids),1,np.uint8),np.full(len(appids),2,np.uint8),np.full(len(reuse['ids']),3,np.uint8)])
        write(OUT/'manifest.json',dict(complete=True,parent=str(PARENT),parent_total=len(road['active']),parent_active=int(road['active'].sum()),
            activated_existing=len(actids),appended_new_geometry=len(appids),reuse_existing_newgroup=len(reuse['ids']),
            allowed_color_world_rows=str(OUT/'allowed_color_world_rows.npz'),source_SHA256={str(x):sha(x) for x in [
                ROOT/'k_gate/safe_candidates.npz',OUT/'activation.npz',OUT/'append.npz',OUT/'reuse_existing.npz',OUT/'allowed_color_world_rows.npz',OUT/'reuse_caps_authority.json']},
            no_new_same_center_stack=True))
        print(json.dumps(dict(complete=True,activation=len(actids),true_spatial_append=len(appids),reuse=len(reuse['ids']),reuse_K_pass=passed)),flush=True)
    except Exception:
        write(OUT/'reuse_K/failed.json',dict(error=traceback.format_exc(),nativeF=0 if ctx is None else ctx.native_calls,optimizer_updates=0));raise
    finally:
        if ctx is not None:ctx.close()
        state=read(lease);state['active']=False;write(lease,state)
if __name__=='__main__':main()
