"""Registered ordinary-GS experiments. Default is inspection, never execution.

The root launch contract and a fresh caller-owned GPU lease are both required.
No held metric is used for budgets, solver continuation, or checkpoint choice.
"""
from pathlib import Path
import argparse,copy,gc,hashlib,json,os,random,sys,time,traceback
import numpy as np
from native_ideal_inputs import ideal_frame_from_arrays,load_ideal_frames

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
FRAMES=(86,110);ARM_ORDER=('G_RGB','G_DETAIL','G_ORACLE','IDEAL_ORACLE')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);temporary=p.with_name(p.name+'.new')
    temporary.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temporary,p)
def scalar(x):
    if isinstance(x,dict):return {k:scalar(v)for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [scalar(v)for v in x]
    if hasattr(x,'detach'):
        a=x.detach().cpu();return a.item()if a.numel()==1 else a.tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


def validate_launch(auth,protocol_path,output):
    if auth.get('GPU_authorized')is not True or auth.get('formal_authorized')is not True:raise RuntimeError('formal GPU execution is not authorized')
    if auth.get('protocol_sha256')!=sha(protocol_path)or Path(auth.get('output','')).resolve()!=Path(output).resolve():raise RuntimeError('launch protocol/output binding differs')
    smoke_path=Path(auth['smoke_receipt']);smoke=read(smoke_path)
    rule=auth.get('engineering_rule','approximately_10_minute_real_lifecycle')
    if rule=='measured_queue_below_one_hour_complete_lifecycle':
        if not 0<float(auth.get('queue_upper_estimate_seconds',0))<3600:raise RuntimeError('below-one-hour measured queue estimate missing')
        throughput=Path(auth['throughput_receipt'])
        if auth.get('throughput_receipt_sha256')!=sha(throughput):raise RuntimeError('actual throughput receipt binding differs')
        minimum_elapsed=0.
    elif rule=='approximately_10_minute_real_lifecycle':minimum_elapsed=float(auth.get('minimum_smoke_seconds',600))
    else:raise RuntimeError('unknown real lifecycle engineering rule')
    if smoke.get('completed')is not True or float(smoke.get('elapsed_seconds',0))<=0 or float(smoke['elapsed_seconds'])<minimum_elapsed:
        raise RuntimeError('verified actual lifecycle smoke is required under registered time rule')
    if auth.get('smoke_receipt_sha256')!=sha(smoke_path):raise RuntimeError('smoke receipt binding differs')
    phases=smoke.get('phases',[])
    if not phases:raise RuntimeError('canonical smoke wrapper lacks raw receipt chain')
    seen=set();elapsed=0.
    for phase in phases:
        path=Path(phase['path']).resolve()
        if path in seen or phase['sha256']!=sha(path):raise RuntimeError('duplicate/changed original smoke phase receipt')
        seen.add(path);raw=read(path);value=raw
        for key in phase['elapsed_seconds_key'].split('.'):value=value[key]
        seconds=float(value)
        if not np.isfinite(seconds)or seconds<=0:raise RuntimeError('invalid original lifecycle duration')
        if raw.get('formal_updates',0)!=0:raise RuntimeError('smoke phase is a formal warm start')
        elapsed+=seconds
    if abs(elapsed-float(smoke['elapsed_seconds']))>1e-5:raise RuntimeError('smoke phase sum and wrapper elapsed differ')
    for key in ('shared_restore','RGB600_boundary','idealI_restore','public1920_fiveview','K'):
        if smoke.get('checks',{}).get(key)is not True:raise RuntimeError('smoke lifecycle check missing: '+key)
    if auth.get('formal_warm_start')not in (None,False):raise RuntimeError('formal cannot warm-start engineering states')


def oracle_relative_gap(row):return float(row['measured_box_FW_gap'])/max(float(row['objective']),1e-3)


def canonical_loss_snapshot(value):
    """JSON changes tuples to lists; restore only the declared tuple fields."""
    value=copy.deepcopy(value);spec=value['spec']
    spec['scales']=tuple(spec['scales']);spec['source_weights']=tuple(tuple(x)for x in spec['source_weights'])
    return value


def oracle_resume_phase(arm,cursor,trace,registered_maximum,recipe,source_identity,folder):
    maximum=registered_maximum;continued=False;converged=False
    last_row=copy.deepcopy(trace[-1])if trace else None
    if cursor>0 and(last_row is None or last_row.get('iteration')!=cursor):raise ValueError('oracle endpoint trace/cursor differs')
    if last_row is not None:last_row['relative_measured_FW_gap']=oracle_relative_gap(last_row)
    continuation_path=Path(folder)/'continuation.json'
    if arm=='G_ORACLE':
        if maximum!=2000:raise RuntimeError('registered realoracle initial budget must be2000')
        if continuation_path.exists():
            continuation=read(continuation_path);base_cp=Path(folder)/'checkpoint_2000.pt'
            if (continuation.get('max_total')!=4000 or continuation.get('fixed_recipe')!=recipe or
                continuation.get('source_identity')!=source_identity or continuation.get('checkpoint_sha256')!=sha(base_cp)):
                raise RuntimeError('continuation checkpoint/source/recipe binding differs')
            if cursor<2000:raise RuntimeError('continuation exists but explicit resume checkpoint predates2000')
            maximum=4000;continued=True
        elif cursor>2000:raise RuntimeError('oracle checkpoint beyond2000 lacks registered continuation')
        if last_row is not None and cursor%100==0 and oracle_relative_gap(last_row)<=.01:converged=True
    return maximum,continued,converged,last_row


class Ledger:
    def __init__(self,output,resume=False):
        self.output=Path(output);self.path=self.output/'physical_ledger.json'
        self.data=read(self.path)if resume and self.path.exists()else dict(native_forward_attempts=0,native_forwards=0,native_per_view_backwards=0,
            torch_backward_invocations=0,parameter_updates=0,shared_updates=0,box_accepted_updates=0,arms={},source='actual calls including failure/replay; never rewind on checkpoint restore')
        self.arm='initialization';self.old_highwater={}
    def add(self,key,n=1):
        self.data[key]=self.data.get(key,0)+int(n)
        a=self.data['arms'].setdefault(self.arm,{})
        a[key]=a.get(key,0)+int(n)
    def flush(self):write(self.path,self.data)


def run(args):
    protocol_path=Path(args.protocol);protocol=read(protocol_path);out=Path(args.output);auth=read(args.authorization)
    validate_launch(auth,protocol_path,out)
    if set(protocol['arms'])!=set(ARM_ORDER):raise RuntimeError('unregistered arm set')
    if out.exists()and not args.resume:raise FileExistsError('preserve existing run; use explicit own checkpoint resume')
    if args.resume and not out.exists():raise FileNotFoundError('resume output missing')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic();ctx=None;ledger=Ledger(out,args.resume)
    progress=read(out/'run_state.json')if args.resume else dict(completed_arms={},current_arm=None,checkpoint=None,logical_highwater={})
    try:
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code'),str(C/'road_capacity_gpu_20261118_r1/code')]
        from road_capacity_native import assert_gpu_lease
        assert_gpu_lease(args.lease,args.lease_owner)
        import torch
        torch.set_num_threads(1);torch.manual_seed(101);np.random.seed(101);random.seed(101);torch.cuda.reset_peak_memory_stats()
        from native_capacity_inputs import real_frames,make_context,field_and_reader,ASSET
        from native_capacity_loss import NativeCapacityLoss,LossSpec
        from shared_training_state import SharedTrainingState
        from gs_color_oracle import ColorLayout,BoundedGSColors
        from box_quadratic_solver import ProjectedBoxQuadratic,capture_solver_checkpoint,restore_solver_checkpoint
        sourcefiles=[Path(__file__),protocol_path,B/'code/native_capacity_loss.py',B/'code/native_capacity_inputs.py',
            B/'shared_field/code/shared_field_gs_read.py',B/'shared_field/code/shared_training_state.py',
            B/'gs_oracle/code/gs_color_oracle.py',B/'gs_oracle/code/box_quadratic_solver.py',B/'code/native_ideal_inputs.py',
            C/'road_capacity_gpu_20261118_r1/code/road_capacity_native.py']
        package=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit')
        sourcefiles += [package/n for n in ('persistent_road_handoff.py','v44_metric_raster.py','v44_factory.py','v44_scene.py','v44_manifest.py')]
        binding={str(p):sha(p)for p in sourcefiles}
        ideal_root=Path(protocol['arms']['IDEAL_ORACLE']['target_root'])
        for p in [ideal_root.parent/'preparation_audit_r2.json',*[ideal_root/f'{r:03d}.npz'for r in FRAMES]]:
            binding[str(p)]=sha(p)
        frames,data_binding=real_frames('seen');binding.update(data_binding['files'])
        rawctx=make_context(args.lease,args.lease_owner,frames=FRAMES)
        class CountingContext:
            def __getattr__(self,key):return getattr(rawctx,key)
            def render_road(self,raw,colors=None):
                rawctx.require_lease();ledger.add('native_forward_attempts')
                v=rawctx.render_road(raw,colors);ledger.add('native_forwards')
                if not torch.equal(v['alpha'].detach().reshape(1280,1920),torch.as_tensor(frames[raw]['alpha'],device=rawctx.base_rgb.device)):
                    raise RuntimeError('frozen native/objective alpha differs')
                return v
            def render_joint(self,raw,colors=None):
                rawctx.require_lease();ledger.add('native_forward_attempts');v=rawctx.render_joint(raw,colors);ledger.add('native_forwards');return v
        ctx=CountingContext()
        initial_model,reader,field_binding=field_and_reader(ctx.base_rgb,'cuda')
        f=field_binding['field']
        for key in ('source_common_avg_A200','permission','runtime'):
            binding[f[key]]=f[key+'_SHA256']
        binding.update(field_binding['read']['source_SHA256'])
        for name,module in tuple(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path and Path(path).is_file()and(name in ('utils.v4_road_render','gaussian_renderer')or'rasterization'in name):
                binding[str(Path(path).resolve())]=sha(path)
        for p in (ASSET/'road.npz',ASSET/'provenance.npz',ASSET/'persistent/scene.json',ASSET/'persistent/proof.json'):
            binding[str(p)]=sha(p)
        layout=ColorLayout(ctx.active_world_rows,reader.eligible)
        if len(ctx.active_world_rows)!=protocol['frozen']['active_rows']or int(reader.eligible.sum())!=protocol['frozen']['eligible_color_rows']:raise RuntimeError('registered world topology/scope differs')
        field_cache=out/'common_initial_field.pt';ci_cache=out/'common_initial_ci.npz';loss_cache=out/'common_loss_snapshot.json'
        objective=NativeCapacityLoss(frames,device='cuda')
        if not args.resume:
            with torch.no_grad():ci0=reader.colors(initial_model).detach().clone()
            initial_model.save(field_cache,dict(role='fresh_registered_common_A200L_H0',field_binding=field_binding))
            np.savez_compressed(ci_cache,ci0=ci0.cpu().numpy(),active_world_rows=ctx.active_world_rows,eligible_active=reader.eligible)
            cpu_source=B/'shared_field/preparation_r1/initial_ci.npz'
            with np.load(cpu_source,allow_pickle=False)as z:cpu_cuda_max=float(np.max(abs(ci0.cpu().numpy()-z['ci0'])))
            if cpu_cuda_max>1e-6:raise RuntimeError('common CPU/GPU initial physical read differs')
            with torch.no_grad():initial_predictions={r:ctx.render_road(r,ci0)['rgb']for r in FRAMES}
            (out/'common_initial').mkdir()
            for raw,P in initial_predictions.items():
                np.savez_compressed(out/'common_initial'/f'{raw:03d}_native.npz',road_rgb=P.cpu().numpy(),road_alpha=frames[raw]['alpha'],K=ctx.K_masks[raw])
            write(loss_cache,objective.calibrate(initial_predictions));del initial_predictions
            source_record=dict(binding=binding,data_binding=data_binding,field_binding=field_binding,scope_sha256=layout.signature,
                CPU_GPU_initial_RGB_max=cpu_cuda_max,common_initial_ci_sha256=sha(ci_cache),common_initial_field_sha256=sha(field_cache),loss_snapshot_sha256=sha(loss_cache))
            write(out/'source_freeze.json',source_record);write(out/'run_state.json',progress);ledger.flush()
        else:
            source_record=read(out/'source_freeze.json')
            if source_record['binding']!=binding or source_record['scope_sha256']!=layout.signature:raise RuntimeError('formal resume source/protocol/scope changed')
            for path,key in [(ci_cache,'common_initial_ci_sha256'),(field_cache,'common_initial_field_sha256'),(loss_cache,'loss_snapshot_sha256')]:
                if sha(path)!=source_record[key]:raise RuntimeError('formal common cache changed')
            with np.load(ci_cache,allow_pickle=False)as z:
                if not np.array_equal(z['active_world_rows'],ctx.active_world_rows)or not np.array_equal(z['eligible_active'],reader.eligible):raise RuntimeError('common initial row identity changed')
                ci0=torch.as_tensor(z['ci0'].copy(),device=ctx.base_rgb.device)
            objective.restore(canonical_loss_snapshot(read(loss_cache)))
        source_identity=dict(source_freeze_sha256=sha(out/'source_freeze.json'),role='registered_formal',scope_sha256=layout.signature)
        full_world=np.zeros(len(ctx.road['active']),bool);full_world[layout.eligible_world_rows]=True
        positions=torch.as_tensor(np.flatnonzero(reader.eligible),device=ci0.device)
        parent_joint={};results=copy.deepcopy(progress['completed_arms'])

        def write_log(folder,row):
            with (folder/'steps.jsonl').open('a')as f:f.write(json.dumps(scalar(row),allow_nan=False)+'\n')
        def hook(event):
            key=event['event'];n=event['amount']
            if key=='native_backward_calls':ledger.add('native_per_view_backwards',n)
            elif key=='backward_calls':ledger.add('torch_backward_invocations',n)
            elif key=='parameter_updates':ledger.add('parameter_updates',n);ledger.add('shared_updates',n)
        def endpoint(arm,colors,*,model=None,box=None,extra=None,eval_objective=None):
            folder=out/arm;rendered=folder/'rendered';rendered.mkdir(exist_ok=True)
            if model is not None:
                field_path=folder/'field.pt'
                if not field_path.exists():model.save(field_path,dict(arm=arm,purpose='formal',reader='ordinaryGS_Sigma_average',source_identity=source_identity,field_binding=field_binding))
                reloaded,_=type(model).load(field_path,device='cuda')
                with torch.no_grad():query=reader.colors(reloaded)
                error=float((query-colors).abs().max())
                if error>1e-6:raise RuntimeError('authority field reload/requery mismatch')
                colors=query.detach();del reloaded
            else:
                field_path=folder/'oracle_colors.pt';box_model=BoundedGSColors(ci0,layout)
                with torch.no_grad():box_model.rgb.copy_(torch.as_tensor(box.x,device=ci0.device,dtype=ci0.dtype).reshape_as(box_model.rgb))
                if not field_path.exists():box_model.save(field_path)
                reloaded=BoundedGSColors.load(field_path,device='cuda')
                if not torch.equal(reloaded(),box_model()):raise RuntimeError('oracle color serialization mismatch')
                colors=reloaded().detach();error=0.;del box_model,reloaded
            asset=folder/'asset'
            if not asset.exists():ctx.save_asset(colors,full_world,asset,training_record=dict(arm=arm,purpose='registered_formal_candidate_not_production',source_identity=source_identity,
                authority=str(field_path),ordinary_GS_primary=True,extra=extra))
            Krows=[];end_predictions={}
            with torch.no_grad():
                for raw in FRAMES:
                    road=ctx.render_road(raw,colors);joint=ctx.render_joint(raw,colors)
                    end_predictions[raw]=road['rgb']
                    if raw not in parent_joint:parent_joint[raw]=ctx.render_joint(raw)
                    mask=torch.as_tensor(ctx.K_masks[raw],device=colors.device)
                    kerror=float((joint['rgb']-parent_joint[raw]['rgb']).abs().amax(0)[mask].max())
                    aerror=float((joint['alpha']-parent_joint[raw]['alpha']).abs().max())
                    Krows.append(dict(raw=raw,pixels=int(mask.sum()),joint_K_RGB_max=kerror,joint_alpha_max=aerror))
                    if kerror!=0 or aerror!=0:raise RuntimeError('actual native1920 K/alpha changed')
                    path=rendered/f'{raw:03d}_native.npz'
                    if not path.exists():np.savez_compressed(path,road_rgb=road['rgb'].cpu().numpy(),road_alpha=road['alpha'].squeeze().cpu().numpy(),
                        joint_rgb=joint['rgb'].cpu().numpy(),joint_alpha=joint['alpha'].squeeze().cpu().numpy(),K=ctx.K_masks[raw])
            with torch.no_grad():
                data_objective=eval_objective if eval_objective is not None else objective
                detail_f,detail_terms=data_objective.objective(end_predictions,'detail')
                rgb_f,_=data_objective.objective(end_predictions,'rgb')
            result=dict(arm=arm,completed=True,asset=str(asset),asset_dir=str(asset),authority=str(field_path),authority_reload_query_RGB_max=error,
                checkpoint=progress.get('checkpoint')if progress.get('checkpoint_arm')==arm else None,
                source_bindings=str(out/'source_freeze.json'),common_initial_ci=str(ci_cache),
                data_DETAIL_objective=float(detail_f),data_RGB_objective=float(rgb_f),endpoint_data_terms=scalar(detail_terms),
                native_resolution=[1920,1280],K=Krows,source_identity=source_identity,
                public_loader_1920_final_independent_phase_pending=True,diagnostic_134_179_198_pending=True,
                promotion=False,extra=extra)
            if arm=='IDEAL_ORACLE':
                result.update(endpoint_own_I_RGB=float(rgb_f),endpoint_own_I_DETAIL=float(detail_f),endpoint_own_I_terms=scalar(detail_terms),
                    separate_synthetic_control_not_realteacher_ranking=True)
            else:
                result.update(endpoint_common_RGB=float(rgb_f),endpoint_common_DETAIL=float(detail_f),endpoint_common_terms=scalar(detail_terms),
                    endpoint_timing='post-update actual nativeRGB already calculated; no additional forward')
            write(folder/'endpoint.json',result);ledger.flush();return result

        for arm in ARM_ORDER:
            if arm in results:continue
            ledger.arm=arm;folder=out/arm;folder.mkdir(exist_ok=True);cfg=protocol['arms'][arm]
            progress['current_arm']=arm
            if cfg['type']=='shared_field':
                if cfg['steps']!=1200 or cfg['H_only_steps']!=600 or cfg['joint_LH_steps']!=600:raise RuntimeError('shared helper/budget protocol mismatch')
                model,_=type(initial_model).load(field_cache,device='cuda')
                expected=SharedTrainingState.CONFIG;configured=cfg['optimizer']
                if (configured['name']!='Adam'or configured['low_lr']!=expected['low_lr']or configured['high_lr']!=expected['high_lr']or
                    tuple(configured['betas'])!=tuple(expected['betas'])or configured['eps']!=expected['eps']or configured['weight_decay']!=expected['weight_decay']):
                    raise RuntimeError('shared optimizer config differs from registered protocol')
                state=SharedTrainingState(model,reader,ctx,objective,mode=cfg['loss'],source_identity=source_identity,
                    source_metadata=field_binding,purpose='formal',ledger_hook=hook)
                cp=progress.get('checkpoint')if progress.get('checkpoint_arm')==arm else None
                if cp:state.load(cp)
                while state.cursor<cfg['steps']:
                    rawctx.require_lease();t=time.monotonic();row=state.step();row['seconds']=time.monotonic()-t
                    write_log(folder,row);progress['logical_highwater'][arm]=max(progress['logical_highwater'].get(arm,0),state.cursor)
                    if state.cursor%100==0:
                        cp=folder/('checkpoint_%04d.pt'%state.cursor)
                        if not cp.exists():state.save(cp)
                        progress.update(checkpoint=str(cp),checkpoint_arm=arm)
                        print('CHECKPOINT',arm,state.cursor,row['loss'],flush=True)
                    write(out/'run_state.json',progress);ledger.flush()
                with torch.no_grad():colors=reader.colors(model).detach()
                results[arm]=endpoint(arm,colors,model=model,extra=dict(logical_steps=state.cursor,physical=state.physical,mode=cfg['loss']))
                del state,model,colors;gc.collect();torch.cuda.empty_cache()
            else:
                box_objective=objective;ideal_sources=None
                if arm=='IDEAL_ORACLE':
                    ideal_frames,ideal_sources=load_ideal_frames(protocol)
                    if any(binding.get(p)!=h for p,h in ideal_sources['files'].items()):raise RuntimeError('ideal inputs changed since common source freeze')
                    box_objective=NativeCapacityLoss(ideal_frames,spec=LossSpec(source_weights=(('I',1.),)),device='cuda')
                    with torch.no_grad():pred={r:ctx.render_road(r,ci0)['rgb']for r in FRAMES}
                    ideal_loss=folder/'initial_loss_snapshot.json'
                    if ideal_loss.exists():box_objective.restore(canonical_loss_snapshot(read(ideal_loss)))
                    else:write(ideal_loss,box_objective.calibrate(pred))
                    del pred
                box_identity=dict(common=source_identity,arm=arm,objective=box_objective.snapshot(),ideal_sources=ideal_sources)
                def evaluate(x,need_gradient):
                    parameter=torch.as_tensor(np.asarray(x),device=ci0.device,dtype=ci0.dtype).detach().requires_grad_(need_gradient)
                    with torch.set_grad_enabled(need_gradient):
                        colors=ci0.index_copy(0,positions,parameter.reshape(-1,3))
                        pred={r:ctx.render_road(r,colors)['rgb']for r in FRAMES};value,terms=box_objective.objective(pred,'detail')
                        if need_gradient:
                            grad=torch.autograd.grad(value,parameter)[0]
                            ledger.add('torch_backward_invocations');ledger.add('native_per_view_backwards',len(FRAMES))
                        else:grad=None
                    result=float(value.detach());gradient=None if grad is None else grad.detach().cpu().numpy().astype(np.float64)
                    ledger.flush() # includes solver construction and every rejected warm/backtracking query
                    del pred,value,terms,colors,parameter,grad;return result,gradient,dict(arm=arm,source='actual_native_GS_DETAIL')
                recipe=protocol['arms']['G_ORACLE']['recipe']
                solver=ProjectedBoxQuadratic(ci0[positions].cpu().numpy(),evaluate,identity=box_identity,**recipe)
                cp=progress.get('checkpoint')if progress.get('checkpoint_arm')==arm else None
                if cp:restore_solver_checkpoint(torch.load(cp,map_location='cpu',weights_only=False),solver,role='formal',frames=FRAMES,device='cuda')
                maximum,continued,converged,last_row=oracle_resume_phase(arm,solver.iteration,solver.trace,cfg['max_accepted_steps'],recipe,source_identity,folder)
                continuation_path=folder/'continuation.json'
                while True:
                    while solver.iteration<maximum and not converged:
                        rawctx.require_lease();row=solver.step();ledger.add('parameter_updates');ledger.add('box_accepted_updates')
                        row['relative_measured_FW_gap']=oracle_relative_gap(row);last_row=row;write_log(folder,row)
                        progress['logical_highwater'][arm]=max(progress['logical_highwater'].get(arm,0),solver.iteration)
                        if solver.iteration%100==0:
                            cp=folder/('checkpoint_%04d.pt'%solver.iteration)
                            if not cp.exists():torch.save(capture_solver_checkpoint(solver,role='formal',frames=FRAMES,device='cuda'),cp)
                            progress.update(checkpoint=str(cp),checkpoint_arm=arm)
                            print('CHECKPOINT',arm,solver.iteration,row['objective'],row['relative_measured_FW_gap'],flush=True)
                            if arm=='G_ORACLE'and row['relative_measured_FW_gap']<=.01:converged=True
                        write(out/'run_state.json',progress);ledger.flush()
                        if converged:break
                    if arm=='G_ORACLE'and not converged and not continued:
                        if solver.iteration!=2000:raise RuntimeError('unexpected continuation cursor')
                        cp=folder/'checkpoint_2000.pt';saved=torch.load(cp,map_location='cpu',weights_only=False)
                        restore_solver_checkpoint(saved,solver,role='formal',frames=FRAMES,device='cuda')
                        maximum=4000;continued=True;write(continuation_path,dict(reason='registered training measuredgap>1percent at2000, no held metric read',exact_checkpoint=str(cp),checkpoint_sha256=sha(cp),source_identity=source_identity,fixed_recipe=recipe,max_total=4000))
                        continue
                    break
                with torch.no_grad():colors=ci0.index_copy(0,positions,torch.as_tensor(solver.x,device=ci0.device,dtype=ci0.dtype).reshape(-1,3))
                results[arm]=endpoint(arm,colors,box=solver,extra=dict(accepted_steps=solver.iteration,continued=continued,measured_training_convergence=converged,
                    strict_numeric_certificate=False,last_train_record=last_row,ideal_sources=ideal_sources),eval_objective=box_objective)
                del solver,colors,box_objective;gc.collect();torch.cuda.empty_cache()
            progress['completed_arms']=results;progress.update(current_arm=None,checkpoint=None,checkpoint_arm=None)
            write(out/'run_state.json',progress);write(out/'results.json',dict(completed=False,arms=results,physical=ledger.data));ledger.flush()
        rawctx.assert_frozen()
        if any(sha(p)!=h for p,h in binding.items()):raise RuntimeError('registered source/input changed during experiment')
        feasible={k:results[k]['data_DETAIL_objective']for k in ('G_RGB','G_DETAIL','G_ORACLE')}
        best_feasible=min(feasible.values())
        logical_updates=sum((v['extra'].get('logical_steps',v['extra'].get('accepted_steps',0)))for v in results.values())
        result=dict(schema='registered_native_GS_experiments.v1',completed=True,arms=results,physical=ledger.data,
            feasible_real_DETAIL_training_points=feasible,known_feasible_oracle_objective_upper_bound=best_feasible,
            oracle_worse_than_known_feasible_shared=results['G_ORACLE']['data_DETAIL_objective']>min(feasible['G_RGB'],feasible['G_DETAIL']),
            practical_limit_not_established_if_oracle_worse_than_feasible=True,
            completed_logical_updates=logical_updates,engineering_replay_updates=max(0,ledger.data['parameter_updates']-logical_updates),
            source_identity=source_identity,source_freeze=str(out/'source_freeze.json'),seconds_this_process=time.monotonic()-started,
            GPU_keep_on=True,production_scene_modified=False,quality_PASS_not_inferred=True,
            final_fiveview_public1920_evaluation_pending=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(out/'results.json',result);print('COMPLETED',json.dumps(dict(arms=list(results),physical=ledger.data,output=str(out))),flush=True)
    except Exception:
        ledger.flush();write(out/('failure_%d.json'%int(time.time())),dict(error=traceback.format_exc(),physical=ledger.data,run_state=progress,
            seconds_this_process=time.monotonic()-started,prior_checkpoints_preserved=True,production_scene_modified=False));raise
    finally:
        if ctx is not None:ctx.close()
        lease_path=Path(args.lease)
        if lease_path.exists():
            state=read(lease_path)
            if state.get('owner')==args.lease_owner:
                state.update(active=False,ended_at=time.time());write(lease_path,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(B/'protocol.json'));p.add_argument('--output');p.add_argument('--authorization')
    p.add_argument('--lease');p.add_argument('--lease-owner');p.add_argument('--resume',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(json.dumps(dict(execute_default=False,protocol=a.protocol,arms=ARM_ORDER,GPU_context_loaded=False)))
    else:
        if not all((a.output,a.authorization,a.lease,a.lease_owner)):p.error('execution requires output, authorization, fresh lease and owner')
        run(a)
