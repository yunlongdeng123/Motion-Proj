"""Fixed R2 CI-only numerical check. Original objective/normalizers, capped real FG."""
from pathlib import Path
import copy,hashlib,json,os,random,subprocess,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R2=C/'road_surface_residual_20261121_r2'
OUT=R2/'diagnostic/color_optimality';ASSET=R2/'appearance_fit/asset';R1=C/'road_surface_adaptive_20261121_r1'
COMMON=C/'road_common1152_20261121';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
B=C/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/native_capacity_20261120_r1'
sys.path[:0]=[str(R1/'code'),str(REPO/'scripts/worldsim_v82'),str(C/'road_capacity_gpu_20261118_r1/code'),str(B/'numerical_completion/code')]
from run_adaptive import geometry_render,pin,load,read,write,sha,save,cpu_clone,rng_state,restore_rng
from native_capacity_loss import NativeCapacityLoss
from fp64_reduction_loss import FP64ReductionLoss
from checkpointed_lbfgsb import CheckpointedLBFGSB
import torch
FRAMES=(86,110,134,179,198);OWNER='r2_color_optimality';CTX=None;LEDGER={};PRIOR_F=0

def count(k,n=1):LEDGER[k]=LEDGER.get(k,0)+n
def account():write(OUT/'compute_ledger.json',dict(counters=LEDGER,nativeF=PRIOR_F+(CTX.native_calls if CTX else 0),GPU_keep_on=True))
def status(phase,**extra):write(OUT/'status.json',dict(state='running',complete=False,phase=phase,PID=os.getpid(),updated_at=time.time(),**extra))
def close():
    global CTX,PRIOR_F
    if CTX is not None:PRIOR_F+=CTX.native_calls;CTX.close();CTX=None
    account();torch.cuda.empty_cache()
def context(asset):
    global CTX
    close()
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    from road_capacity_native import NativeRoadContext
    scene,manifest,loader=load_persistent_handoff(Path(asset)/'persistent/scene.json',Path(asset)/'persistent/proof.json')
    assert manifest.data['render_wh']==[1152,768]
    for obj in (scene.static,scene.dynamic,scene.environment):
        for p in obj.parameters() if hasattr(obj,'parameters') else ():p.requires_grad_(False)
    CTX=NativeRoadContext(scene,manifest,loader,asset,FRAMES,OUT/'gpu_lease.json',OWNER);return CTX
def array_sha(a):return hashlib.sha256(str((a.shape,a.dtype)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
def scalar_tree(x):
    if torch.is_tensor(x):return float(x.detach())
    if isinstance(x,dict):return {k:scalar_tree(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [scalar_tree(v) for v in x]
    return x
def difference(a,b,path='root'):
    # String/object/Fortran workspaces MUST compare exactly before numeric casts.
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype,path
        if a.dtype.kind in 'SUO':assert np.array_equal(a,b),path;return 0.
        if a.dtype.kind in 'biu':assert np.array_equal(a,b),path;return 0.
        assert np.isfinite(a).all() and np.isfinite(b).all(),path
        return float(np.abs(a-b).max()) if a.size else 0.
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        return max((difference(a[k],b[k],path+'.'+str(k)) for k in a),default=0.)
    if isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b),path
        return max((difference(x,y,path+'.'+str(i)) for i,(x,y) in enumerate(zip(a,b))),default=0.)
    if isinstance(a,(float,np.floating)):
        assert np.isfinite(a) and np.isfinite(b),path;return abs(float(a)-float(b))
    assert a==b,path;return 0.
def ci_stats(x,g):
    xx=np.asarray(x).reshape(-1);gg=np.asarray(g).reshape(-1);pg=np.where(((xx<=0)&(gg>0))|((xx>=1)&(gg<0)),0,gg)
    return dict(parameter_count=len(xx),ci_min=float(xx.min()),ci_max=float(xx.max()),at_zero=int((xx==0).sum()),at_one=int((xx==1).sum()),
        near_zero_1e6=int((xx<=1e-6).sum()),near_one_1e6=int((xx>=1-1e-6).sum()),gradient_abs_max=float(np.abs(gg).max()),
        gradient_L2=float(np.linalg.norm(gg)),projected_gradient_inf=float(np.abs(pg).max()))

def prepare_objective(ctx):
    scope=load(R2/'train_inputs/allowed_color_world_rows.npz');ids=scope['world_row_ids'].astype(np.int64)
    assert len(ids)==1191 and len(np.unique(ids))==1191
    pos=np.searchsorted(ctx.active_world_rows,ids);assert np.array_equal(ctx.active_world_rows[pos],ids)
    checkpoint=R2/'appearance_fit/checkpoint.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['schema']=='r2_appearance_full_state.v1' and state['cursor']==1200
    road=load(ASSET/'road.npz');assert np.array_equal(state['rgb'].clamp(0,1).numpy(),road['reference_srgb'][ids])
    frames={};overlap={}
    for raw in (86,110):
        formal=load(R2/'appearance_fit/rendered'/f'{raw:03d}.npz');covered=load(R2/'coverage_fit/rendered'/f'{raw:03d}.npz')
        assert np.array_equal(formal['road_alpha'],covered['road_alpha']);alpha=formal['road_alpha']
        parent=load(R1/'appearance_fit/rendered'/f'{raw:03d}.npz');d=load(R2/'proposal/domains'/f'{raw:03d}.npz')
        teacher=load(COMMON/'teacher'/f'raw{raw:03d}'/'supervision.npz');masks={k:d['appearance_'+k] for k in ['A','C','S']}
        frames[raw]=dict(alpha=np.ones_like(alpha),targets={k:alpha[None]*teacher['target_'+k+'_srgb'] for k in masks},contexts=masks,scores=masks)
        exported={}
        for k in masks:
            for prefix in ['RGB','context','score']:exported[prefix+'_'+k]=masks[k]
        path=OUT/'training_masks'/f'{raw:03d}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**exported)
        overlap[str(raw)]={}
        for k in ['C','S']:
            cert=d[k+'_source']&d['canonical_valid'];new=cert&(parent['road_alpha']<.95)&(alpha>=.95)
            overlap[str(raw)][k]=dict(appearance_pixels=int(masks[k].sum()),certified_newly_covered=int(new.sum()),included_in_training=int((new&masks[k]).sum()),
                excluded_from_training=int((new&~masks[k]).sum()),new_mask_SHA256=array_sha(new),training_mask_SHA256=array_sha(masks[k]),
                target_alpha_times_teacher_SHA256=array_sha(frames[raw]['targets'][k]),target_uses_current_coverage_alpha=True)
    original=NativeCapacityLoss(frames,device='cuda');original.restore(state['normalizers'])
    precise=FP64ReductionLoss(frames,device='cuda');precise.restore(state['normalizers'])
    assert original.snapshot()==state['normalizers'] and precise.snapshot()==state['normalizers']
    write(OUT/'objective_identity.json',dict(verified=True,original_checkpoint=str(checkpoint),checkpoint_SHA256=sha(checkpoint),
        original_normalizer_signature_exact=True,normalizers_recalibrated=False,snapshot=state['normalizers'],overlap=overlap,
        source_SHA256={str(p):sha(p) for p in [R2/'code/run_residual.py',B/'code/native_capacity_loss.py',B/'numerical_completion/code/fp64_reduction_loss.py',
            COMMON/'teacher/manifest.json',R2/'proposal/domains/086.npz',R2/'proposal/domains/110.npz',R2/'train_inputs/allowed_color_world_rows.npz']}))
    return ids,torch.as_tensor(pos,device='cuda',dtype=torch.long),original,precise

def save_asset(ctx,colors,out,label):
    out=Path(out)
    if (out/'persistent/proof.json').exists():return out
    if out.exists():raise FileExistsError('preserve incomplete asset '+str(out))
    source=load(ASSET/'road.npz');r={k:v.copy() for k,v in source.items()};p=load(ASSET/'provenance.npz')
    ids=load(R2/'train_inputs/allowed_color_world_rows.npz')['world_row_ids'];pos=np.searchsorted(ctx.active_world_rows,ids)
    r['reference_srgb'][ids]=colors[torch.as_tensor(pos,device='cuda')].detach().cpu().numpy()
    for key in source:
        if key!='reference_srgb':assert np.array_equal(r[key],source[key]),'fixed geometry changed '+key
    frozen=np.ones(len(r['active']),bool);frozen[ids]=False;assert np.array_equal(r['reference_srgb'][frozen],source['reference_srgb'][frozen])
    assert np.min(r['reference_srgb'])>=0 and np.max(r['reference_srgb'])<=1
    p['effective_row_rgb']=r['reference_srgb'].copy();out.mkdir(parents=True)
    np.savez_compressed(out/'road.npz',**r);np.savez_compressed(out/'provenance.npz',**p)
    pin(ASSET,out,dict(stage=label,source_asset=str(ASSET),CI_only=True,allowed_rows=1191,no_geometry_update=True))
    write(out/'authority_receipt.json',dict(complete=True,road_SHA256=sha(out/'road.npz'),provenance_SHA256=sha(out/'provenance.npz'),
        source_road_SHA256=sha(ASSET/'road.npz'),source_provenance_SHA256=sha(ASSET/'provenance.npz'),
        original_objective_identity_SHA256=sha(OUT/'objective_identity.json'),all_geometry_alpha_fixed=True,all_non1191_colors_bitexact=True,
        total_rows=len(r['active']),active_rows=int(r['active'].sum()),new_point_count=0))
    return out

def export(ctx,folder,label):
    from PIL import Image
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);rows=[]
    with torch.no_grad():
        for raw in FRAMES:
            road=geometry_render(ctx,raw);joint=geometry_render(ctx,raw,joint=True);K=ctx.K_masks[raw]
            data=dict(road_rgb=road['rgb'].cpu().numpy(),road_alpha=road['alpha'].cpu().numpy(),joint_rgb=joint['rgb'].cpu().numpy(),joint_alpha=joint['alpha'].cpu().numpy(),K=K)
            original=load(R2/'appearance_fit/rendered'/f'{raw:03d}.npz')
            assert np.array_equal(data['road_alpha'],original['road_alpha']) and np.array_equal(data['joint_alpha'],original['joint_alpha'])
            krgb=float(np.abs(data['joint_rgb']-original['joint_rgb'])[:,K].max());ka=float(np.abs(data['joint_alpha']-original['joint_alpha'])[K].max())
            if krgb!=0 or ka!=0:raise RuntimeError('PROTECTION_GATE numeric K changed')
            packet=folder/f'{raw:03d}.npz';np.savez_compressed(packet,**data);png={}
            for view in ['road','joint']:
                path=folder/f'{raw:03d}_{view}.png';Image.fromarray(np.rint(np.clip(data[view+'_rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)).save(path);png[view+'_png']=str(path)
            rows.append(dict(raw=raw,packet=str(packet),**png,stage=label,resolution=[1152,768],asset_dir=str(ctx.asset_dir),joint_K_RGB_max=krgb,joint_K_alpha_max=ka,alpha_exact_final=True))
    count(label+':export_nativeF',10);account();write(folder/'receipt.json',dict(complete=True,rows=rows));return rows

def main():
    global LEDGER,PRIOR_F
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
    watched=[ASSET/'road.npz',ASSET/'provenance.npz',ASSET/'persistent/scene.json',ASSET/'persistent/proof.json',R2/'resource_ledger.json',R2/'status.json',R2/'receipt.json',R2/'appearance_fit/checkpoint.pt']
    source={str(p):sha(p) for p in watched}
    if (OUT/'compute_ledger.json').exists():old=read(OUT/'compute_ledger.json');LEDGER=old['counters'];PRIOR_F=old['nativeF']
    write(OUT/'gpu_lease.json',dict(active=True,owner=OWNER,gpu_authorized=True,power_policy='keep_on'))
    try:
        status('objective_identity');ctx=context(ASSET);ids,pos,original,precise=prepare_objective(ctx);seed=ctx.base_rgb.detach();x0=seed[pos].cpu().numpy().astype(np.float64)
        free=seed[pos].clone().requires_grad_(True);colors=seed.index_copy(0,pos,free)
        predictions={r:geometry_render(ctx,r,colors=colors)['rgb'] for r in (86,110)}
        f32,_=original.objective(predictions,'detail');f64,_=precise.objective(predictions,'detail')
        g32=torch.autograd.grad(f32,free,retain_graph=True)[0];g64=torch.autograd.grad(f64,free)[0]
        fr=abs(float(f32)-float(f64))/max(abs(float(f64)),1e-12);gr=float((g32.double()-g64.double()).norm())/max(float(g64.double().norm()),1e-12)
        qualification=dict(verified=fr<=1e-6 and gr<=1e-5,function_relative=fr,gradient_L2_relative=gr,original_f=float(f32),FP64_reduction_f=float(f64),same_native_predictions=True,FP64_only_square_mean_scalar=True)
        count('qualification:nativeF',2);count('qualification:nativeB',4);write(OUT/'precision_qualification.json',qualification);account()
        if not qualification['verified']:raise RuntimeError('unchanged objective FP64 f/grad qualification failed')
        del original,free,colors,predictions,f32,f64,g32,g64
        current={};timings=[]
        def evaluator(role):
            def evaluate(x):
                start=time.monotonic();free=torch.as_tensor(x,device='cuda',dtype=torch.float32).requires_grad_(True)
                colors=seed.index_copy(0,pos,free.reshape(-1,3));pred={r:geometry_render(ctx,r,colors=colors)['rgb'] for r in (86,110)}
                value,parts=precise.objective(pred,'detail');gradient=torch.autograd.grad(value,free)[0]
                f=float(value.detach());g=gradient.detach().cpu().numpy().astype(np.float64);details=scalar_tree(parts)
                count(role+':FG');count(role+':nativeF',2);count(role+':nativeB',2);timings.append(time.monotonic()-start)
                current.clear();current.update(x_SHA256=array_sha(np.asarray(x)),objective=f,loss_terms=details,CI_gradient=ci_stats(x,g))
                if role=='numeric' and LEDGER[role+':FG']==1:
                    gap=CheckpointedLBFGSB.gap(x,g)
                    write(OUT/'initial_evaluation.json',dict(**current,FW_gap=gap,relative_FW_gap=gap/max(f,1e-3),same_Adam_final_CI=True))
                return f,g,details
            return evaluate
        identity=dict(source_SHA256=source,objective=sha(OUT/'objective_identity.json'),allowed=sha(R2/'train_inputs/allowed_color_world_rows.npz'),kind='R2_final_CI1191_box_fixed_geometry')
        smoke=OUT/'smoke';smoke.mkdir(exist_ok=True)
        if not (smoke/'receipt.json').exists():
            status('numeric_smoke');start=time.monotonic();solver=CheckpointedLBFGSB(x0,evaluator('smoke'),identity=dict(role='smoke',**identity),maxFG=50)
            first=solver.next_accepted();assert first is not None,'need a true next accepted point for workspace smoke'
            statepath=smoke/'solver_accepted1.npz';solver.save(statepath);save(smoke/'RNG.pt',rng_state('cuda'))
            expectedrow=solver.next_accepted();expected=solver.snapshot();solver.load(statepath);restore_rng(torch.load(smoke/'RNG.pt',map_location='cpu',weights_only=False),'cuda')
            actualrow=solver.next_accepted();diff=difference(expected,solver.snapshot());assert expectedrow is not None and actualrow is not None and diff<1e-5
            smoke_colors=seed.index_copy(0,pos,torch.as_tensor(solver.last_accepted_x.reshape(-1,3),device='cuda',dtype=torch.float32)).detach()
            asset=save_asset(ctx,smoke_colors,smoke/'asset','numeric_smoke');reference={}
            with torch.no_grad():reference={k:v.cpu().numpy() for k,v in geometry_render(ctx,110,colors=smoke_colors).items()}
            count('smoke:reload_reference_nativeF');ctx=context(asset)
            with torch.no_grad():loaded=geometry_render(ctx,110)
            count('smoke:reload_nativeF');reload={k:float(np.abs(reference[k]-loaded[k].cpu().numpy()).max()) for k in ['rgb','alpha']};assert max(reload.values())<=5e-6
            export(ctx,smoke/'rendered','smoke');estimate=2000*float(np.median(timings))+240
            if estimate>=3600:raise RuntimeError('hour_scale_smoke_requires_600s_before_formal; preserve qualification and stop for explicit controller extension')
            write(smoke/'receipt.json',dict(verified=True,full_workspace_disk_restore=True,Fortran_strings_exact=True,next_accepted_workspace_numeric_max=diff,
                expected_FG=expected['fg_calls'],accepted=solver.accepted,actual_replay_FG=LEDGER.get('smoke:FG'),asset_save_reload=reload,all5_K_exact_zero=True,
                next_stage_entry=True,seconds=time.monotonic()-start,estimated_max2000FG_seconds=estimate,ten_minute_required=False,formal_warm_start=False))
            ctx=context(ASSET)
        # Fresh fixed source; smoke is discarded and cannot warm-start this solver.
        seed=ctx.base_rgb.detach();run=OUT/'solver';run.mkdir(exist_ok=True);pointer=run/'latest.json'
        solver=CheckpointedLBFGSB(x0,evaluator('numeric'),identity=dict(role='numeric',**identity),maxFG=2000)
        accepted_details=None
        if pointer.exists():
            meta=read(pointer);solver.load(meta['workspace']);restore_rng(torch.load(meta['RNG'],map_location='cpu',weights_only=False),'cuda');accepted_details=meta['accepted_details']
        status('numeric',FG=solver.fg_calls,accepted=solver.accepted,budget=2000)
        while not solver.done:
            row=solver.next_accepted()
            if row is None:break
            accepted_details=copy.deepcopy(current);accepted_details['row']=row
            if solver.accepted%20==0 or row['relative_FW_gap']<=.01:
                path=run/f'workspace_FG{solver.fg_calls:04d}_A{solver.accepted:04d}.npz'
                if not path.exists():solver.save(path)
                rng=path.with_suffix('.RNG.pt');save(rng,rng_state('cuda'));write(pointer,dict(workspace=str(path),RNG=str(rng),accepted_details=accepted_details))
                status('numeric',FG=solver.fg_calls,accepted=solver.accepted,budget=2000,objective=row['objective'],FW_relative=row['relative_FW_gap']);account();print(json.dumps(row),flush=True)
            if row['relative_FW_gap']<=.01:solver.stop_at_training_gate()
        if accepted_details is None:accepted_details=copy.deepcopy(current)
        path=run/'final_workspace.npz';solver.save(path);save(run/'final_RNG.pt',rng_state('cuda'))
        np.savez_compressed(OUT/'accepted_CI.npz',world_row_ids=ids,CI_FP64=solver.last_accepted_x.reshape(-1,3),CI_native_FP32=solver.last_accepted_x.reshape(-1,3).astype(np.float32),gradient_FP64=solver.last_accepted_g.reshape(-1,3))
        colors=seed.index_copy(0,pos,torch.as_tensor(solver.last_accepted_x.reshape(-1,3),device='cuda',dtype=torch.float32)).detach()
        final=save_asset(ctx,colors,OUT/'new_numeric_asset','numeric_final');ctx=context(final);rows=export(ctx,OUT/'rendered','numeric')
        manifest=dict(schema='fixed_geometry_CI_numeric_exports.v1',complete=True,rows=rows,asset=str(final),source_asset=str(ASSET),allowed_rows=1191,new_points=0)
        write(OUT/'export_manifest.json',manifest);close();assert source=={str(p):sha(p) for p in watched},'R2 formal source changed'
        fw=solver.gap(solver.last_accepted_x,solver.last_accepted_g);receipt=dict(complete=True,FG=solver.fg_calls,accepted=solver.accepted,objective=solver.last_accepted_f,
            FW_gap=fw,relative_FW_gap=fw/max(solver.last_accepted_f,1e-3),termination=solver.reason,recipe=solver.recipe,ABI=solver.abi,
            last_accepted_CI_used=True,accepted_details=accepted_details,CI_gradient=ci_stats(solver.last_accepted_x,solver.last_accepted_g),
            source_SHA256=source,formal_source_unchanged=True,geometry_alpha_all5_exact=True,K_all5_exact_zero=True,new_point_count=0,
            numerical_diagnostic_only=True,strict_capacity_certificate=False,source_R2_formal1200_unchanged=True,final_asset=str(final),road_SHA256=sha(final/'road.npz'),
            objective_identity_SHA256=sha(OUT/'objective_identity.json'),initial_evaluation=read(OUT/'initial_evaluation.json'),
            numeric_script_SHA256=sha(__file__),GPU_keep_on=True)
        write(OUT/'numeric_receipt.json',receipt);status('CPU_evaluation')
        subprocess.run([sys.executable,str(OUT/'evaluate_numeric_cpu.py'),'--manifest',str(OUT/'export_manifest.json')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        write(OUT/'receipt.json',receipt);write(OUT/'status.json',dict(state='complete',phase='done',PID=os.getpid(),**receipt))
    except Exception:
        error=traceback.format_exc();account();write(OUT/'failure.json',dict(error=error,counters=LEDGER));write(OUT/'status.json',dict(state='failed',complete=False,error=error,PID=os.getpid()));raise
    finally:close();write(OUT/'gpu_lease.json',dict(active=False,owner=OWNER,gpu_authorized=True,power_policy='keep_on'))

if __name__=='__main__':main()
