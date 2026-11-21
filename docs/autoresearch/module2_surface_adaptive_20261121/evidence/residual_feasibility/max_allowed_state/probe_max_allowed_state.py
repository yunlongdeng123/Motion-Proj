"""One optimistic fixed-footprint alpha feasibility probe; zero updates, no asset edits."""
from pathlib import Path
import hashlib,json,sys,time,traceback
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_adaptive_20261121_r1')
OUT=ROOT/'residual_feasibility/max_allowed_state'
ASSET=ROOT/'appearance_fit/asset'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
FRAMES=(86,110,134,179,198);OWNER='surface_max_allowed_state'
sys.path[:0]=[str(ROOT/'code'),str(REPO/'scripts/worldsim_v82'),
    '/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1/code']
from run_adaptive import geometry_render

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    if (OUT/'receipt.json').exists():raise FileExistsError('preserve completed one-shot diagnostic')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'started.json').exists():raise FileExistsError('preserve failed one-shot diagnostic and its compute')
    frozen=[ROOT/'status.json',ROOT/'receipt.json',ROOT/'resource_ledger.json',ROOT/'contract.json',
        ROOT/'proposal/activation.npz',ROOT/'proposal/append.npz',
        ASSET/'road.npz',ASSET/'provenance.npz',ASSET/'persistent/scene.json',
        ASSET/'persistent/proof.json',ASSET/'persistent/static_opacity.pt',ROOT/'code/run_adaptive.py']
    frozen += [ROOT/'appearance_fit/rendered'/f'{raw:03d}.npz' for raw in FRAMES]
    frozen += [ROOT/'proposal/domains'/f'{raw:03d}.npz' for raw in FRAMES]
    binding={str(p):sha(p) for p in frozen};start=time.monotonic();ctx=None;rows=[];Kproof=[]
    write(OUT/'started.json',dict(time=time.time(),optimizer_updates=0,asset=str(ASSET),input_SHA256=binding))
    write(OUT/'gpu_lease.json',dict(active=True,owner=OWNER,gpu_authorized=True,optimizer_updates_allowed=0,power_policy='keep_on'))
    try:
        import torch
        torch.set_num_threads(2)
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from road_capacity_native import NativeRoadContext
        scene,m,loader=load_persistent_handoff(ASSET/'persistent/scene.json',ASSET/'persistent/proof.json')
        for component in (scene.static,scene.dynamic,scene.environment):
            for p in component.parameters() if hasattr(component,'parameters') else ():p.requires_grad_(False)
        headers=m.camera_headers()
        assert all(headers[r]['width']==1152 and headers[r]['height']==768 for r in FRAMES)
        ctx=NativeRoadContext(scene,m,loader,ASSET,FRAMES,OUT/'gpu_lease.json',OWNER)
        road=ctx.road;prior=ctx.provenance
        parent_asset=Path(read(ROOT/'contract.json')['parent'])
        n0=int(read(parent_asset/'persistent/proof.json')['road']['total_rows'])
        activation=load(ROOT/'proposal/activation.npz');append=load(ROOT/'proposal/append.npz')
        actids=activation['existing_world_rows'].astype(np.int64)
        appids=np.arange(n0,n0+len(append['world_xyz']),dtype=np.int64)
        assert n0+len(appids)==len(road['active'])
        expected=np.flatnonzero(prior['adaptive_activate_existing']|prior['adaptive_append_new_geometry'])
        world_ids=np.r_[actids,appids]
        assert np.array_equal(np.sort(world_ids),expected)
        positions=np.searchsorted(ctx.active_world_rows,world_ids)
        assert np.array_equal(ctx.active_world_rows[positions],world_ids)
        pos=torch.as_tensor(positions,device='cuda',dtype=torch.long)
        native=ctx.base_packet(86)
        op=native['opacities'].detach().reshape(-1).clone()
        scales=native['scales'].detach().clone()
        # Match the exact float32 cap expression used by the trained coverage guard.
        cap=np.r_[np.repeat(np.min(activation['max_sigma_m'],axis=1)[:,None],2,axis=1),
                  np.repeat(np.min(append['max_sigma_m'],axis=1)[:,None],2,axis=1)]
        cap_gpu=torch.as_tensor(cap,device='cuda',dtype=torch.float32)
        changed=torch.cat([cap_gpu*float(road['meter_to_model']),scales[pos,2:]],1)
        op=op.index_copy(0,pos,torch.ones(len(pos),device='cuda'))
        scales=scales.index_copy(0,pos,changed)
        untouched=torch.ones(len(ctx.base_rgb),device='cuda',dtype=torch.bool);untouched[pos]=False
        assert torch.equal(op[untouched],native['opacities'].reshape(-1)[untouched])
        assert torch.equal(scales[untouched],native['scales'][untouched])
        assert torch.equal(scales[:,2],native['scales'][:,2])
        np.savez_compressed(OUT/'max_state_parameters.npz',world_row_ids=world_ids,active_positions=positions,
            opacity=np.ones(len(pos),np.float32),tangent_sigma_m=cap,normal_sigma_m=road['gaussian_sigma_m'][world_ids,2])
        settings=dict(opacity=1.0,registered_opacity_upper=.999,
            opacity_role='optimistic diagnostic relaxation only; not a new trained or persisted asset',
            tangent_sigma='original per-candidate maximum, same native tangent axes',
            fixed='all old-active parameters; all colors/positions/normals/rotation/normal sigma',
            activate_existing=len(actids),append_new_geometry=len(appids),resolution=[1152,768])
        with torch.no_grad():
            for raw in FRAMES:
                value=geometry_render(ctx,raw,op,scales,feature_ids=pos)
                K=ctx.K_masks[raw];alpha=value['alpha'].cpu().numpy();new_weight=value['new_weight'].cpu().numpy()
                packet=dict(road_rgb=value['rgb'].cpu().numpy(),road_alpha=alpha,K=K,new_geometry_weight=new_weight)
                path=OUT/'rendered'/f'{raw:03d}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**packet)
                trained=load(ROOT/'appearance_fit/rendered'/f'{raw:03d}.npz')['road_alpha']
                domain=load(ROOT/'proposal/domains'/f'{raw:03d}.npz')['fit']
                row=dict(raw=raw,path=str(path),SHA256=sha(path),keys=list(packet),resolution=[1152,768],
                    fit_pixels=int(domain.sum()),trained_fit_below_01=int((domain&(trained<.1)).sum()),
                    max_fit_below_01=int((domain&(alpha<.1)).sum()),trained_fit_below_95=int((domain&(trained<.95)).sum()),
                    max_fit_below_95=int((domain&(alpha<.95)).sum()),alpha_change_min=float((alpha-trained).min()),
                    new_road_weight_on_K_max=float(np.abs(new_weight[K]).max()))
                rows.append(row)
                if row['new_road_weight_on_K_max']==0:
                    Kproof.append(dict(raw=raw,method='new road-only unit alphaT is exactly0 on full K; extra scene occluders cannot increase it',
                        road_only_new_weight_K_max=0.,joint_new_weight_K_max_bound=0.,extra_native_calls=0))
                else:
                    # Necessary only if road mass exists behind a kept occluder.
                    joint=geometry_render(ctx,raw,op,scales,joint=True,feature_ids=pos)
                    jmax=float(joint['new_weight'][:,K].abs().max()) if joint['new_weight'].ndim==3 else float(joint['new_weight'][K].abs().max())
                    Kproof.append(dict(raw=raw,method='actual joint native unit-weight qualification',joint_new_weight_K_max=jmax,extra_native_calls=1))
                    if jmax!=0:raise RuntimeError('optimistic diagnostic state intersects protected joint K')
                write(OUT/'manifest.json',dict(schema='surface_max_allowed_state_packets.v1',complete=False,settings=settings,rows=rows))
                print('MAX_STATE',raw,'nativeF',ctx.native_calls,'K',row['new_road_weight_on_K_max'],flush=True)
        ctx.assert_frozen();scene.assert_persistent_unchanged()
        if {str(p):sha(p) for p in frozen}!=binding:raise RuntimeError('formal files changed during read-only diagnostic')
        write(OUT/'manifest.json',dict(schema='surface_max_allowed_state_packets.v1',complete=True,settings=settings,rows=rows))
        write(OUT/'summary.json',dict(schema='surface_max_allowed_feasibility_summary.v1',rows=rows,K=Kproof,
            interpretation='CPU residual comparison uses the same certified pixel IDs as the trained state. Remaining max-state gaps locate sampling/protection/fixed-surface support limits, not a universal3DGS ceiling.',
            no_rigorous_global_negative_certificate=True))
        write(OUT/'receipt.json',dict(complete=True,optimizer_updates=0,native_forward_calls=ctx.native_calls,
            road_only_native_calls=5,necessary_joint_K_native_calls=ctx.native_calls-5,settings=settings,
            original_formal_files_unchanged=True,input_SHA256=binding,seconds=time.monotonic()-start,
            source_code=str(Path(__file__)),source_code_SHA256=sha(__file__),GPU_keep_on=True))
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),native_forward_calls=0 if ctx is None else ctx.native_calls,rows=rows,seconds=time.monotonic()-start));raise
    finally:
        if ctx is not None:ctx.close()
        lease=read(OUT/'gpu_lease.json');lease.update(active=False,ended_at=time.time());write(OUT/'gpu_lease.json',lease)

if __name__=='__main__':main()
