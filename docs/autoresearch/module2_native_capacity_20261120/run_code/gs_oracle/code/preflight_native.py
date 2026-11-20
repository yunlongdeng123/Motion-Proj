"""Root-authorized 8-forward/2-backward, zero-update native1920 preflight.

Default mode only prints the plan. Execution creates a new independent lease;
it never updates a legacy lease, production asset, optimizer, or source file.
"""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time,traceback
import numpy as np

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=CAMPAIGN/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
ASSET=CAMPAIGN/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OWNER='native_capacity_preflight';FRAMES=(86,110)
DEFAULT=G/'native_capacity_20261120_r1/gs_oracle/preflight_r1'


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,keys):
    with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in keys}


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    auth=dict(schema='native_capacity_zero_update_preflight.v1',GPU_authorized=True,phase='preflight',
        user_authorization='User authorized experiments pursuing practical fixed-GS capacity; root specifically authorized this8F/2B zero-update native preflight.',
        frames=list(FRAMES),image_wh=[1920,1280],max_native_forwards=8,max_native_backwards=2,optimizer_updates_allowed=0,
        source_asset=str(ASSET),scope_rows=89887,scope_role='frozen common scope; final formal qualification remains separate',
        power_policy='keep_GPU_on',production_scene_writes=False)
    write(output/'authorization.json',auth)
    lease=output/'gpu_lease.json';write(lease,dict(owner=OWNER,active=True,gpu_authorized=True,
        authorization=str(output/'authorization.json'),phase='preflight',optimizer_updates_allowed=0,keep_GPU_on=True))
    count=dict(native_forward_calls=0,native_backward_calls=0,optimizer_updates=0,rows=[])
    write(output/'native_counter.json',count);ctx=None;start=time.monotonic()
    bindings={str(Path(__file__).resolve()):sha(__file__)}
    paths=[ASSET/'road.npz',ASSET/'provenance.npz',ASSET/'persistent/scene.json',ASSET/'persistent/proof.json',
        Q/'permission/final_r1/teacher_evidence.npz',Q/'permission/avg_sparse_gh31_r1/operator/constants.npz',
        Q/'permission/avg_sparse_gh31_r1/operator/read.json',CAMPAIGN/'road_capacity_gpu_20261118_r1/code/road_capacity_native.py']
    try:
        bindings.update({str(p):sha(p)for p in paths})
        import torch
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(CAMPAIGN/'road_capacity_gpu_20261118_r1/code'),str(REPO/'scripts/worldsim_v82')]
        from road_capacity_native import NativeRoadContext
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        manifest=ASSET/'persistent/scene.json';proof=ASSET/'persistent/proof.json'
        scene,m,receipt=load_persistent_handoff(manifest,proof)
        for component in (scene.static,scene.dynamic,scene.environment):
            for parameter in component.parameters()if hasattr(component,'parameters')else():parameter.requires_grad_(False)
        low=m.camera_headers();data=dict(m.data);data['render_wh']=[1920,1280]
        high=SceneManifest(data,manifest.parent).camera_headers();cameras={}
        def camera(raw):
            if raw not in cameras:cameras[raw]=make_native_camera(high[raw])
            return cameras[raw]
        scene.camera_provider=camera
        ctx=NativeRoadContext(scene,m,receipt,ASSET,FRAMES,lease,OWNER);ctx._manifest_path=manifest;ctx._proof_path=proof
        ctx.require_lease();context_seconds=time.monotonic()-start
        scope=load(Q/'permission/final_r1/teacher_evidence.npz',['active_world_rows','common_write_active'])
        active=scope['active_world_rows'];eligible=scope['common_write_active']
        assert np.array_equal(active,ctx.active_world_rows)and eligible.dtype==bool and int(eligible.sum())==89887
        constants=load(Q/'permission/avg_sparse_gh31_r1/operator/constants.npz',['base','mass','row_ids'])
        loc=np.searchsorted(active,constants['row_ids'])
        assert np.array_equal(active[loc],constants['row_ids'])and np.array_equal(active[eligible],constants['row_ids'])
        valid=constants['mass']>1e-12
        colors=ctx.base_rgb.clone();positions=torch.as_tensor(loc[valid],device=colors.device)
        colors[positions]=torch.as_tensor(np.clip(constants['base'][valid],0,1),device=colors.device,dtype=colors.dtype)
        emask=torch.as_tensor(eligible,device=colors.device)
        assert torch.equal(colors[~emask],ctx.base_rgb[~emask])
        rng=np.random.default_rng(101);sign=torch.as_tensor(rng.choice([-1.,1.],(int(eligible.sum()),3)),device=colors.device,dtype=colors.dtype)
        margin=torch.minimum(colors[emask],1-colors[emask]);delta=.2*margin*sign
        minus=colors[emask]-delta;plus=colors[emask]+delta
        assert bool(((minus>=0)&(minus<=1)&(plus>=0)&(plus<=1)).all())
        initial=dict(context_seconds=context_seconds,active_rows=len(active),eligible_rows=int(eligible.sum()),
            base_available_rows=int(valid.sum()),base_fallback_rows=int((~valid).sum()),
            scope_world_rows_sha256=hashlib.sha256(active[eligible].astype('<i8').tobytes()).hexdigest(),
            F0='frozen GH31 operator.base at L=H=0, clippednativeRGB; readonlyparent exact',
            trainable_parameters='only89887x3 per-Gaussian colors; no optimizer instantiated',
            geometry_opacity_scales_detached=True)
        write(output/'context_ready.json',initial);print('CONTEXT_READY',json.dumps(initial),flush=True)
        def render(raw,value,label):
            if count['native_forward_calls']>=8:raise RuntimeError('8-forward preflight budget exhausted')
            ctx.require_lease();torch.cuda.synchronize();began=time.perf_counter()
            result=ctx.render_road(raw,value);torch.cuda.synchronize();seconds=time.perf_counter()-began
            count['native_forward_calls']+=1;count['rows'].append(dict(kind='forward',raw=raw,label=label,seconds=seconds))
            write(output/'native_counter.json',count);return result
        frame_rows=[]
        for raw in FRAMES:
            assert np.array_equal(low[raw]['W'],high[raw]['W'])and all(high[raw][k]==4*low[raw][k]for k in ['fx','fy','cx','cy'])
            targets_path=G/'quality_bench_20261120_r1/highres/targets'/f'{raw:03d}.npz'
            bindings[str(targets_path)]=sha(targets_path)
            target=load(targets_path,['T_A','T_C','T_S','A_pre_alpha','C_pre_alpha','S_pre_alpha'])
            with torch.no_grad():base=render(raw,colors,'F0')
            base_rgb=base['rgb'].detach().cpu().numpy().copy();fixed_alpha=base['alpha'].detach().reshape(1280,1920)
            parameter=minus.detach().clone().requires_grad_(True)
            cminus=ctx.base_rgb.index_copy(0,torch.as_tensor(np.flatnonzero(eligible),device=colors.device),parameter)
            # This graph receives individual native Gaussian RGB, not pixel material.
            vm=render(raw,cminus,'bounded_minus_with_RGB_objective_VJP');pm=vm['rgb']
            minus_alpha=vm['alpha'].detach().reshape(1280,1920).clone()
            probe=torch.zeros_like(pm);photo_parts={};mask_count={}
            for source,weight in [('A',1.),('C',.5),('S',1.)]:
                mask=torch.as_tensor(target[source+'_pre_alpha'],device=pm.device,dtype=torch.bool)
                teacher=torch.as_tensor(target['T_'+source],device=pm.device,dtype=pm.dtype)
                residual=pm-fixed_alpha[None]*teacher;n=int(mask.sum());assert n>0
                probe[:,mask]+=weight*2*residual.detach()[:,mask]/(3*n)
                photo_parts[source]=float(residual.detach().square()[:,mask].mean());mask_count[source]=n
            if count['native_backward_calls']>=2:raise RuntimeError('2-backward preflight budget exhausted')
            torch.cuda.synchronize();began=time.perf_counter()
            grad=torch.autograd.grad((pm*probe).sum(),parameter)[0]
            torch.cuda.synchronize();bseconds=time.perf_counter()-began
            count['native_backward_calls']+=1;count['rows'].append(dict(kind='backward',raw=raw,label='actual_fullRGB_objective_ci_VJP',seconds=bseconds))
            write(output/'native_counter.json',count)
            pm_cpu=pm.detach().cpu().numpy().copy();probe_cpu=probe.detach().cpu().numpy().copy()
            grad_cpu=grad.detach().cpu().numpy().copy()
            directional_gradient=float((grad.double()*(plus-minus).double()).sum())
            del pm,vm,grad,parameter,cminus,probe,residual,teacher,mask;gc.collect()
            with torch.no_grad():
                cplus=ctx.base_rgb.index_copy(0,torch.as_tensor(np.flatnonzero(eligible),device=colors.device),plus)
                vp=render(raw,cplus,'bounded_plus');pp_cpu=vp['rgb'].cpu().numpy().copy()
                midpoint=(minus+plus)/2
                cmid=ctx.base_rgb.index_copy(0,torch.as_tensor(np.flatnonzero(eligible),device=colors.device),midpoint)
                vi=render(raw,cmid,'bounded_midpoint');pi_cpu=vi['rgb'].cpu().numpy().copy()
            native_alphas=[base['alpha'],minus_alpha,vp['alpha'],vi['alpha']]
            alpha_exact=all(torch.equal(a.reshape(1280,1920),fixed_alpha)for a in native_alphas)
            directional_forward=float(np.sum((pp_cpu.astype(np.float64)-pm_cpu)*probe_cpu,dtype=np.float64))
            error=abs(directional_gradient-directional_forward);den=max(abs(directional_gradient),abs(directional_forward),1e-12)
            save=output/f'{raw:03d}_probe.npz'
            arrays=dict(F0_rgb=base_rgb,minus_rgb=pm_cpu,plus_rgb=pp_cpu,midpoint_rgb=pi_cpu,
                fixed_alpha=fixed_alpha.cpu().numpy(),ci_RGB_gradient=grad_cpu,active_world_rows=active,eligible_active=eligible,K=ctx.K_masks[raw])
            np.savez_compressed(save,**arrays)
            with np.load(save,allow_pickle=False)as z:roundtrip=all(np.array_equal(z[k],v)for k,v in arrays.items())
            row=dict(raw=raw,image_wh=[1920,1280],native_forward_calls=4,native_backward_calls=1,
                shader='actualordinaryGS perGaussiancolors_precomp',preclamp_linear_midpoint_max=float(np.max(np.abs(pi_cpu.astype(np.float64)-.5*(pm_cpu.astype(np.float64)+pp_cpu)))),
                midpoint_vs_F0_max=float(np.max(np.abs(pi_cpu.astype(np.float64)-base_rgb))),
                ci_gradient_finite=bool(np.isfinite(grad_cpu).all()),ci_gradient_L2=float(np.linalg.norm(grad_cpu.astype(np.float64))),
                ci_gradient_nonzero_rows=int((np.abs(grad_cpu).sum(1)>0).sum()),
                VJP_directional=directional_gradient,native_endpoint_directional=directional_forward,
                directional_absolute_error=error,directional_relative_error=error/den,
                alpha_exact=alpha_exact,save_reload_exact=roundtrip,photo_RGB_MSE=photo_parts,full_source_pixels=mask_count,
                K_pixels_native=int(ctx.K_masks[raw].sum()),timing_forward=[x['seconds']for x in count['rows']if x['kind']=='forward'and x['raw']==raw],
                timing_backward=bseconds,optimizer_updates=0,certificate='numerical native linearity/adjoint evidence; no global renderer error certificate')
            if not(row['ci_gradient_finite']and row['ci_gradient_nonzero_rows']>0 and alpha_exact and roundtrip):raise RuntimeError('preflight finite/alpha/serialization failure')
            frame_rows.append(row);write(output/'progress.json',dict(rows=frame_rows));print('FRAME_DONE',raw,json.dumps(row),flush=True)
            del base,vp,vi,cplus,cmid,base_rgb,minus_alpha,arrays,target,pm_cpu,pp_cpu,pi_cpu,probe_cpu,grad_cpu
            gc.collect();torch.cuda.empty_cache()
        ctx.assert_frozen()
        changed=[p for p,h in bindings.items()if sha(p)!=h]
        if changed:raise RuntimeError('frozen source changed: '+str(changed))
        result=dict(schema='native_capacity_zero_update_preflight_result.v1',completed=True,
            optimizer_updates=0,native_forward_calls=count['native_forward_calls'],native_backward_calls=count['native_backward_calls'],
            context=initial,frames=frame_rows,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),source_binding=bindings,source_fields_unchanged=True,
            production_assets_written=False,new_context_lease_only=True,GPU_keep_on=True,
            numerical_acceptance_thresholds='reported only; root will freeze prospective training/preflight numeric policy')
        write(output/'receipt.json',result);print('COMPLETED',json.dumps({k:v for k,v in result.items()if k not in ['source_binding','frames']}),flush=True)
    except Exception:
        write(output/'failed.json',dict(error=traceback.format_exc(),counter=count,optimizer_updates=0,seconds=time.monotonic()-start));raise
    finally:
        if ctx is not None:ctx.close()
        current=read(lease);current.update(active=False,ended_at=time.time());write(lease,current)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:run(args.output)
    else:print(json.dumps(dict(output=str(args.output),native_allowed_by_default=False,forwards=8,backwards=2,updates=0)))
