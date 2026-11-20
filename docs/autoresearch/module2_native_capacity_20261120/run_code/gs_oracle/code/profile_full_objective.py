"""Authorized actual-native lifecycle profile; no formal run or warm start.

Caps:64 native forwards,16 per-view native backwards,8 total accepted parameter
updates including replays. The intended path uses4 Adam+3 box accepted updates.
"""
from pathlib import Path
import argparse,copy,gc,hashlib,json,random,sys,time,traceback
import numpy as np

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=CAMPAIGN/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
OWNER='native_capacity_profile';FRAMES=(86,110)
DEFAULT=B/'gs_oracle/full_objective_profile_r1'

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    authorization=dict(schema='native_capacity_full_objective_lifecycle.v1',GPU_authorized=True,formal_allowed=False,
        root_authorization='Explicit full-objective profile and lifecycle;8total parameter updates including Adam/projected/replays; no formal training.',
        frames=list(FRAMES),image_wh=[1920,1280],max_native_forwards=64,max_native_per_view_backwards=16,
        max_engineering_parameter_updates=8,keep_GPU_on=True,production_assets_written=False)
    write(out/'authorization.json',authorization)
    lease=out/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(out/'authorization.json'),phase='profile_only',formal_allowed=False))
    ledger=dict(native_forwards=0,native_per_view_backwards=0,torch_backward_invocations=0,
        engineering_parameter_updates=0,Adam_updates=0,box_accepted_updates=0,rows=[])
    ctx=None;start=time.monotonic();bindings={}
    try:
        import torch
        torch.set_num_threads(1);torch.manual_seed(101);np.random.seed(101);random.seed(101)
        torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import real_frames,make_context,field_and_reader
        from native_capacity_loss import NativeCapacityLoss
        from box_quadratic_solver import ProjectedBoxQuadratic,capture_solver_checkpoint,restore_solver_checkpoint
        from gs_color_oracle import ColorLayout,BoundedGSColors
        sourcefiles=[Path(__file__),B/'code/native_capacity_inputs.py',B/'code/native_capacity_loss.py',
            B/'shared_field/code/shared_field_gs_read.py',B/'gs_oracle/code/box_quadratic_solver.py',B/'gs_oracle/code/gs_color_oracle.py',
            CAMPAIGN/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded/road.npz',
            CAMPAIGN/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded/provenance.npz',
            CAMPAIGN/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded/persistent/scene.json',
            CAMPAIGN/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded/persistent/proof.json']
        bindings={str(p):sha(p)for p in sourcefiles}
        frames,data_binding=real_frames('seen')
        ctx=make_context(lease,OWNER);model,reader,field_binding=field_and_reader(ctx.base_rgb,'cuda')
        assert len(ctx.active_world_rows)==151988 and int(reader.eligible.sum())==89887
        layout=ColorLayout(ctx.active_world_rows,reader.eligible)
        model.set_stage('B')
        with torch.no_grad():ci0=reader.colors(model).detach().clone()
        assert torch.equal(ci0[~torch.as_tensor(reader.eligible,device=ci0.device)],ctx.base_rgb[~torch.as_tensor(reader.eligible,device=ci0.device)])
        initial_np=B/'shared_field/preparation_r1/initial_ci.npz'
        with np.load(initial_np,allow_pickle=False)as z:
            assert np.array_equal(z['active_world_rows'],ctx.active_world_rows)
            initial_cpu_RGB_max=float(np.max(np.abs(ci0.cpu().numpy()-z['ci0'])))
        write(out/'context_ready.json',dict(seconds=time.monotonic()-start,scope=89887,active=151988,
            ci0='common avg_A200 L/H0, true physicalSigma read',initial_CPU_CUDA_RGB_max=initial_cpu_RGB_max,
            new_lease=str(lease),source_binding=bindings,data_binding=data_binding,field_binding=field_binding))
        print('CONTEXT_READY',time.monotonic()-start,flush=True)

        def sync():torch.cuda.synchronize()
        def record():write(out/'native_counter.json',ledger)
        def render(raw,colors,label):
            if ledger['native_forwards']>=64:raise RuntimeError('64 native forward cap reached')
            ctx.require_lease();sync();t=time.perf_counter();v=ctx.render_road(raw,colors);sync()
            ledger['native_forwards']+=1;ledger['rows'].append(dict(kind='native_forward',raw=raw,label=label,seconds=time.perf_counter()-t));record()
            alpha=v['alpha'].detach().reshape(1280,1920)
            fixed=torch.as_tensor(frames[raw]['alpha'],device=alpha.device)
            if not torch.equal(alpha,fixed):raise RuntimeError('common native alpha differs from frozen objective input')
            return v['rgb']
        def backward(value,parameters,label):
            if ledger['native_per_view_backwards']+len(FRAMES)>16:raise RuntimeError('16 per-view native backward cap reached')
            sync();t=time.perf_counter();grads=torch.autograd.grad(value,parameters,allow_unused=False);sync()
            ledger['native_per_view_backwards']+=len(FRAMES);ledger['torch_backward_invocations']+=1
            ledger['rows'].append(dict(kind='native_backward',label=label,per_view_calls=len(FRAMES),torch_invocations=1,seconds=time.perf_counter()-t));record()
            return grads
        def accepted(kind,label):
            if ledger['engineering_parameter_updates']>=8:raise RuntimeError('8 engineering update cap reached')
            ledger['engineering_parameter_updates']+=1
            ledger['Adam_updates'if kind=='Adam'else'box_accepted_updates']+=1
            ledger['rows'].append(dict(kind='accepted_parameter_update',solver=kind,label=label));record()
        def cpu(x):
            if torch.is_tensor(x):return x.detach().cpu().clone()
            if isinstance(x,dict):return {k:cpu(v)for k,v in x.items()}
            if isinstance(x,list):return [cpu(v)for v in x]
            if isinstance(x,tuple):return tuple(cpu(v)for v in x)
            return copy.deepcopy(x)
        def difference(a,b):
            if torch.is_tensor(a):
                if a.shape!=b.shape or a.dtype!=b.dtype:return float('inf')
                return float((a.double()-b.double()).abs().max())if a.numel()else 0.
            if isinstance(a,np.ndarray):return float(np.max(abs(a.astype(float)-b.astype(float))))if a.size else 0.
            if isinstance(a,dict):return max([difference(a[k],b[k])for k in a]or[0.])if a.keys()==b.keys()else float('inf')
            if isinstance(a,(list,tuple)):return max([difference(x,y)for x,y in zip(a,b)]or[0.])if len(a)==len(b)else float('inf')
            if isinstance(a,(float,int,np.number)):return abs(float(a)-float(b))
            return 0. if a==b else float('inf')
        def rng():return dict(torch=torch.get_rng_state().clone(),numpy=copy.deepcopy(np.random.get_state()),python=copy.deepcopy(random.getstate()),cuda=[x.clone()for x in torch.cuda.get_rng_state_all()])
        def restore_rng(s):
            torch.set_rng_state(s['torch'].clone());np.random.set_state(copy.deepcopy(s['numpy']));random.setstate(copy.deepcopy(s['python']));torch.cuda.set_rng_state_all([x.clone()for x in s['cuda']])

        objective=NativeCapacityLoss(frames,device='cuda')
        with torch.no_grad():initial_predictions={raw:render(raw,ci0,'common_initial_calibration')for raw in FRAMES}
        normalizers=objective.calibrate(initial_predictions);write(out/'loss_snapshot.json',normalizers)
        del initial_predictions
        opt=torch.optim.Adam([{'params':[model.low],'lr':.01},{'params':[model.high],'lr':.003}],betas=(.9,.999),eps=1e-8,weight_decay=0)
        field_cursor=0;field_trace=[];field_stats=[]
        def field_snapshot(stage):return dict(schema='native_capacity_field_profile.v1',role='profile_not_formal',state=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),RNG=rng(),
            cursor=field_cursor,trace=copy.deepcopy(field_trace),stage=stage,loss=objective.snapshot(),source_binding=copy.deepcopy(bindings),sampler=dict(kind='fixed_full_batch',frames=list(FRAMES)))
        def field_restore(saved):
            nonlocal field_cursor,field_trace
            if saved['schema']!='native_capacity_field_profile.v1'or saved['role']!='profile_not_formal'or saved['source_binding']!=bindings or saved['sampler']!=dict(kind='fixed_full_batch',frames=list(FRAMES)):
                raise ValueError('field profile checkpoint identity differs')
            if saved['cursor']!=len(saved['trace']):raise ValueError('field checkpoint trace cursor differs')
            model.load_state_dict(copy.deepcopy(saved['state']));opt.load_state_dict(copy.deepcopy(saved['optimizer']));objective.restore(saved['loss']);restore_rng(saved['RNG'])
            field_cursor=saved['cursor'];field_trace=copy.deepcopy(saved['trace']);model.set_stage(saved['stage'])
        def field_step(stage,label):
            nonlocal field_cursor
            if ledger['engineering_parameter_updates']>=8:raise RuntimeError('engineering cap before Adam')
            model.set_stage(stage);opt.zero_grad(set_to_none=True);low_before=model.low.detach().clone();low_state_before=cpu(opt.state.get(model.low,{}))
            sync();t=time.perf_counter();colors=reader.colors(model)
            predictions={raw:render(raw,colors,label)for raw in FRAMES};value,parts=objective.objective(predictions,'detail')
            params=[p for p in(model.low,model.high)if p.requires_grad];grads=backward(value,params,label)
            for p,grad in zip(params,grads):p.grad=grad
            grad_rows={name:dict(finite=bool(torch.isfinite(p.grad).all()),L2=float(p.grad.norm()),nonzero=int((p.grad!=0).sum()))for name,p in [('low',model.low),('high',model.high)]if p.requires_grad}
            if not all(v['finite']and v['nonzero']>0 for v in grad_rows.values()):raise RuntimeError('field native gradient nonfinite/zero')
            opt.step();sync();accepted('Adam',label);field_cursor+=1;field_trace.append(stage)
            low_hold=stage!='B'or(torch.equal(model.low,low_before)and difference(cpu(opt.state.get(model.low,{})),low_state_before)==0)
            if not low_hold:raise RuntimeError('B changed low parameters or AdamL')
            row=dict(stage=stage,label=label,objective=float(value.detach()),step_seconds=time.perf_counter()-t,gradient=grad_rows,low_and_Adam_held_exact=low_hold,logical_cursor=field_cursor)
            field_stats.append(row);write(out/'field_progress.json',field_stats);print('FIELD_STEP',json.dumps(row),flush=True)
            del value,parts,predictions,colors,grads;gc.collect();return row
        field_step('B','B1');saved_B1=field_snapshot('B');torch.save(saved_B1,out/'B1_checkpoint.pt')
        field_step('B','B2_reference');expected_B2=field_snapshot('B')
        loaded=torch.load(out/'B1_checkpoint.pt',map_location='cpu',weights_only=False);field_restore(loaded)
        before_restore=difference(loaded,field_snapshot('B'))
        field_step('B','B2_replay');actual_B2=field_snapshot('B');replay_error=difference(expected_B2,actual_B2)
        if before_restore!=0 or replay_error>1e-6:raise RuntimeError('full field restore/replay failed')
        field_step('C','C_entry');field_after=field_snapshot('C');torch.save(field_after,out/'field_final_checkpoint.pt')
        with torch.no_grad():field_colors=reader.colors(model).detach().clone()

        oracle=BoundedGSColors(ci0,layout);positions=oracle.eligible_positions
        last={};box_stats=[]
        def evaluate_box(x,need_gradient):
            # The native color vector is evaluated in-box; no pixel-world overlay.
            parameter=torch.as_tensor(np.asarray(x),device=ci0.device,dtype=ci0.dtype).detach().requires_grad_(need_gradient)
            with torch.set_grad_enabled(need_gradient):
                colors=ci0.index_copy(0,positions,parameter.reshape(-1,3))
                predictions={raw:render(raw,colors,'box_gradient'if need_gradient else'box_backtracking_value')for raw in FRAMES}
                value,parts=objective.objective(predictions,'detail')
                gradient=backward(value,[parameter],'box_objective')[0]if need_gradient else None
            result=float(value.detach());details=dict(role='profile_only_convex_native_DETAIL')
            if need_gradient:last.update(colors=colors.detach().clone(),predictions={r:p.detach().clone()for r,p in predictions.items()})
            g=None if gradient is None else gradient.detach().cpu().numpy().astype(np.float64)
            del predictions,value,parts,colors,parameter,gradient;gc.collect();return result,g,details
        identity=dict(role='profile_only',scope=layout.signature,source_binding=bindings,loss=normalizers)
        solver=ProjectedBoxQuadratic(ci0[positions].cpu().numpy(),evaluate_box,identity=identity,
            L0=1.,backtrack_factor=2.,trial_decrease=2.,max_backtracks=12,comparison_atol=0.)
        def box_step(label):
            if ledger['engineering_parameter_updates']>=8:raise RuntimeError('engineering cap before box accepted step')
            sync();t=time.perf_counter();row=solver.step();sync();accepted('box',label);row.update(label=label,actual_step_seconds=time.perf_counter()-t)
            box_stats.append(row);write(out/'box_progress.json',box_stats);print('BOX_STEP',json.dumps(row),flush=True)
        box_step('box1');box1=capture_solver_checkpoint(solver,role='profile',frames=FRAMES,device='cuda');torch.save(box1,out/'box1_checkpoint.pt')
        box_step('box2_reference');expected_box2=capture_solver_checkpoint(solver,role='profile',frames=FRAMES,device='cuda')
        loaded_box=torch.load(out/'box1_checkpoint.pt',map_location='cpu',weights_only=False)
        restore_solver_checkpoint(loaded_box,solver,role='profile',frames=FRAMES,device='cuda')
        box_restore_error=difference(loaded_box,capture_solver_checkpoint(solver,role='profile',frames=FRAMES,device='cuda'))
        box_step('box2_replay');actual_box2=capture_solver_checkpoint(solver,role='profile',frames=FRAMES,device='cuda')
        box_replay_error=difference(expected_box2,actual_box2)
        if box_restore_error!=0 or box_replay_error>1e-6:raise RuntimeError('full box restore/replay failed')
        torch.save(actual_box2,out/'box_final_checkpoint.pt')
        with torch.no_grad():oracle.rgb.copy_(torch.as_tensor(solver.x,device=ci0.device,dtype=ci0.dtype).reshape_as(oracle.rgb))
        oracle.save(out/'oracle_colors.pt');oracle_loaded=BoundedGSColors.load(out/'oracle_colors.pt',device='cuda')
        if not torch.equal(oracle_loaded(),oracle()):raise RuntimeError('ci serializer reload differs')
        model.save(out/'world_field.pt',dict(role='engineering_profile_not_formal',reader='ordinary_GS_physical_Sigma_average',source_binding=bindings))
        reloaded_field,_=type(model).load(out/'world_field.pt',device='cuda')
        with torch.no_grad():field_requery=reader.colors(reloaded_field)
        field_query_error=float((field_requery-field_colors).abs().max())
        if field_query_error>1e-6:raise RuntimeError('authority field reload/requery differs')
        full_scope=np.zeros(len(ctx.road['active']),bool);full_scope[layout.eligible_world_rows]=True
        assets={}
        for name,colors in [('field',field_requery),('oracle',oracle_loaded())]:
            assets[name]=ctx.save_asset(colors,full_scope,out/(name+'_asset'),training_record=dict(role='engineering_profile_not_production',method=name,source_binding=bindings,world_field_authority=str(out/'world_field.pt')if name=='field'else None))
        # Public loader proofs inherit render_wh480. Rebind camera only in memory,
        # using the saved manifest's own physical intrinsics/extrinsics at1920.
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        public=[]
        for name,colors in [('field',field_requery),('oracle',oracle_loaded())]:
            with torch.no_grad():expected={r:render(r,colors,name+'_public_expected').detach().clone()for r in FRAMES}
            asset=assets[name];scene,m,_=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
            data=dict(m.data);data['render_wh']=[1920,1280];headers=SceneManifest(data,m.parent).camera_headers();cameras={}
            def camera(raw):
                if raw not in cameras:cameras[raw]=make_native_camera(headers[raw])
                return cameras[raw]
            scene.camera_provider=camera;provider=next(p for p in scene._providers if p.kind=='road')
            with torch.no_grad():
                for raw in FRAMES:
                    if ledger['native_forwards']>=64:raise RuntimeError('native cap before public clone')
                    ctx.require_lease();cam=camera(raw);sync();t=time.perf_counter()
                    actual=scene._bindings().raster(cam,scene.pipe,0,provider.rows(raw,cam,scene.pipe),scene._background);sync()
                    ledger['native_forwards']+=1;ledger['rows'].append(dict(kind='native_forward',raw=raw,label=name+'_public_clone',seconds=time.perf_counter()-t));record()
                    rgb_error=float((expected[raw]-actual['rgb']).abs().max());alpha_error=float((actual['alpha'].reshape(1280,1920)-torch.as_tensor(frames[raw]['alpha'],device=ci0.device)).abs().max())
                    row=dict(asset=name,raw=raw,RGB_max=rgb_error,alpha_max=alpha_error,RGB8_max=int((torch.round(expected[raw]*255)-torch.round(actual['rgb']*255)).abs().max()),image_wh=[1920,1280],public_proof_loaded=True,camera_rebound_from_own_manifest=True)
                    public.append(row)
                    if rgb_error>1e-6 or alpha_error!=0:raise RuntimeError('actual1920 public loader reload failed')
            scene.assert_persistent_unchanged();del provider,scene,expected;gc.collect();torch.cuda.empty_cache()
        ctx.assert_frozen()
        if any(sha(p)!=v for p,v in bindings.items()):raise RuntimeError('source changed during profile')
        result=dict(schema='native_capacity_full_objective_lifecycle_result.v1',completed=True,formal_updates=0,
            physical_ledger=ledger,loss_snapshot=str(out/'loss_snapshot.json'),field_steps=field_stats,box_steps=box_stats,
            field_full_restore_error=before_restore,field_next_step_replay_error=replay_error,
            box_full_restore_error=box_restore_error,box_next_step_replay_error=box_replay_error,
            field_reload_query_max=field_query_error,public_reload=public,source_binding=bindings,
            seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            source_fields_unchanged=True,profile_parameters_discard_for_formal=True,GPU_keep_on=True)
        write(out/'receipt.json',result);print('COMPLETED',json.dumps({k:v for k,v in result.items()if k not in('physical_ledger','source_binding','field_steps','box_steps')}),flush=True)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),physical_ledger=ledger,formal_updates=0,seconds=time.monotonic()-start));raise
    finally:
        if ctx is not None:ctx.close()
        state=json.loads(lease.read_text());state.update(active=False,ended_at=time.time());write(lease,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:run(a.output)
    else:print(json.dumps(dict(native_default=False,output=str(a.output),formal_updates=0,maxF=64,maxB=16,maxEngineeringUpdates=8)))
