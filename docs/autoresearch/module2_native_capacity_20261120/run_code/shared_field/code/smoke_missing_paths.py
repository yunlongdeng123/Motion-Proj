"""Authorized missing RGB600->601 and analytic-I lifecycle; disposable only."""
from pathlib import Path
import copy,gc,hashlib,json,random,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
B=G/'native_capacity_20261120_r1';P=B/'smoke_r1';R=P/'readonly_completion_r2';OUT=P/'missing_paths_r1';OWNER='native_capacity_smoke';FRAMES=(86,110)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):
    import torch
    if torch.is_tensor(a):return float((a.double()-b.double()).abs().max())if a.numel()else 0.
    if isinstance(a,np.ndarray):return float(np.max(abs(a.astype(float)-b.astype(float))))if a.size else 0.
    if isinstance(a,dict):return max([diff(a[k],b[k])for k in a if k!='physical']or[0.])if a.keys()==b.keys()else float('inf')
    if isinstance(a,(tuple,list)):return max([diff(x,y)for x,y in zip(a,b)]or[0.])if len(a)==len(b)else float('inf')
    return abs(float(a)-float(b))if isinstance(a,(int,float,np.number))else(0. if a==b else float('inf'))


def main():
    OUT.mkdir(parents=True,exist_ok=False);prior=read(R/'receipt.json');ledger=copy.deepcopy(prior['aggregate_physical_ledger']);origin=read(P/'receipt.json')
    if not prior['complete']or ledger['engineering_accepted_updates']!=2500:raise RuntimeError('unexpected prefix')
    write(OUT/'authorization.json',dict(GPU_authorized=True,formal_allowed=False,
        root_authorization='Fresh disposable G_RGB600->601 plus analyticI100/checkpoint replay; root extraaccepted<=1500, aggregate<=4000/F20000/B8000',
        inherited_engineering_updates=2500,additional_accepted_cap=1500,GPU_keep_on=True))
    lease=OUT/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(OUT/'authorization.json'),phase='smoke_missing_paths'))
    started=time.monotonic();ctx=None;additional_updates=0
    try:
        import torch
        torch.set_num_threads(1);torch.manual_seed(101);np.random.seed(101);random.seed(101);torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import real_frames,make_context,field_and_reader
        from native_capacity_loss import NativeCapacityLoss,LossSpec
        from native_ideal_inputs import load_ideal_frames
        from shared_training_state import SharedTrainingState
        from gs_color_oracle import ColorLayout,BoundedGSColors
        from box_quadratic_solver import ProjectedBoxQuadratic,capture_solver_checkpoint,restore_solver_checkpoint
        files=[Path(__file__),B/'code/native_ideal_inputs.py',B/'code/native_capacity_inputs.py',B/'code/native_capacity_loss.py',
            B/'shared_field/code/shared_training_state.py',B/'shared_field/code/shared_field_gs_read.py',B/'protocol.json',R/'receipt.json']
        binding={str(p):sha(p)for p in files};frames,data_binding=real_frames('seen');protocol=read(B/'protocol.json')
        ctx=make_context(lease,OWNER);model,reader,field_binding=field_and_reader(ctx.base_rgb,'cuda')
        fixed={r:torch.as_tensor(frames[r]['alpha'],device='cuda')for r in FRAMES}
        def call(label,raw,fn):
            if ledger['native_forwards']>=20000:raise RuntimeError('aggregate F cap')
            ctx.require_lease();ledger['native_forwards']+=1;ledger['forwards_by_label'][label]=ledger['forwards_by_label'].get(label,0)+1
            return fn()
        def road(raw,color,label):
            v=call(label,raw,lambda:ctx.render_road(raw,color))
            if not torch.equal(v['alpha'].squeeze().detach(),fixed[raw]):raise RuntimeError('fixed alpha changed')
            return v
        class ContextView:
            frames=FRAMES;active_world_rows=ctx.active_world_rows
            @property
            def native_calls(self):return ctx.native_calls
            def render_road(self,raw,color):return road(raw,color,'RGB_smoke_road')
        def accepted(kind,label):
            nonlocal additional_updates
            if additional_updates>=1500 or ledger['engineering_accepted_updates']>=4000:raise RuntimeError('aggregate accepted cap')
            additional_updates+=1;ledger['engineering_accepted_updates']+=1;ledger[kind+'_updates']+=1
            ledger['accepted_by_label'][label]=ledger['accepted_by_label'].get(label,0)+1
            if 'replay'in label:ledger['replay_updates']+=1
        label=['RGB_fresh']
        def hook(e):
            if e['event']=='native_backward_calls':
                ledger['native_per_view_backwards']+=e['amount']
                if ledger['native_per_view_backwards']>8000:raise RuntimeError('aggregate B cap')
            elif e['event']=='backward_calls':ledger['torch_backward_invocations']+=e['amount']
            elif e['event']=='parameter_updates':accepted('shared',label[0])
        with torch.no_grad():ci0=reader.colors(model).detach();initial={r:road(r,ci0,'shared_same_ci0')['rgb']for r in FRAMES}
        objective=NativeCapacityLoss(frames,device='cuda');objective.restore(torch.load(P/'loss_snapshot.pt',map_location='cpu',weights_only=False))
        state=SharedTrainingState(model,reader,ContextView(),objective,mode='rgb',source_identity={'smoke_missing_source':binding,'data':data_binding},
            source_metadata=field_binding,purpose='smoke',ledger_hook=hook)
        low0=model.low.detach().clone();low_moments_absent=True
        for k in range(600):
            row=state.step()
            if(k+1)%100==0:print('RGB_PROGRESS',k+1,float(row['loss']),flush=True)
        low_moments_absent=model.low not in state.optimizer.state
        if not torch.equal(low0,model.low)or not low_moments_absent or state.stage!='C':raise RuntimeError('RGB600 boundary freeze failed')
        state.save(OUT/'RGB600_checkpoint.pt');before=state.snapshot();label[0]='RGB601_reference';state.step();expected=state.snapshot()
        state.load(OUT/'RGB600_checkpoint.pt');restore_error=diff(before,state.snapshot());label[0]='RGB601_replay';state.step();next_error=diff(expected,state.snapshot())
        if restore_error!=0 or next_error>1e-6:raise RuntimeError('RGB formal boundary restore/nextstep failed')
        state.save(OUT/'RGB601_final_checkpoint.pt');model.save(OUT/'RGB_field.pt',dict(role='smoke_discard',source_metadata=field_binding,logical_cursor=601))
        rgb_colors=reader.colors(model).detach();torch.save(objective.snapshot(),OUT/'RGB_loss_snapshot.pt')
        ideal_frames,ideal_metadata=load_ideal_frames(protocol)
        ideal=NativeCapacityLoss(ideal_frames,spec=LossSpec(source_weights=(('I',1.),)),device='cuda');ideal.calibrate(initial)
        torch.save(ideal.snapshot(),OUT/'I_loss_snapshot.pt');write(OUT/'I_loss_snapshot.json',ideal.snapshot());del initial
        layout=ColorLayout(ctx.active_world_rows,reader.eligible);oracle=BoundedGSColors(ci0,layout);positions=oracle.eligible_positions
        def evaluate(x,gradient):
            if ledger['native_per_view_backwards']+2*int(gradient)>8000:raise RuntimeError('aggregate B cap')
            p=torch.as_tensor(x,device='cuda',dtype=ci0.dtype).detach().requires_grad_(gradient)
            with torch.set_grad_enabled(gradient):
                color=ci0.index_copy(0,positions,p.reshape(-1,3));pred={r:road(r,color,'I_gradient'if gradient else'I_value')['rgb']for r in FRAMES}
                v,_=ideal.objective(pred,'detail');g=torch.autograd.grad(v,p)[0]if gradient else None
                if gradient:ledger['native_per_view_backwards']+=2;ledger['torch_backward_invocations']+=1
            value=float(v.detach());array=None if g is None else g.detach().cpu().numpy().astype(np.float64)
            return value,array,{'role':'analytic_I_smoke_only','target_case':'marking_w120','premult_target_not_double_alpha':True}
        solver=ProjectedBoxQuadratic(ci0[positions].cpu().numpy(),evaluate,identity={'binding':binding,'ideal':ideal_metadata,'objective':ideal.snapshot()},**protocol['arms']['G_ORACLE']['recipe'])
        def step_I(which):
            row=solver.step();accepted('oracle',which)
            with(OUT/'I_steps.jsonl').open('a')as f:f.write(json.dumps(dict(row,label=which))+'\n')
        step_I('I_first');one=capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda');torch.save(one,OUT/'I1_checkpoint.pt')
        step_I('I_reference');expected=capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda')
        restore_solver_checkpoint(torch.load(OUT/'I1_checkpoint.pt',map_location='cpu',weights_only=False),solver,role='smoke_I',frames=FRAMES,device='cuda')
        Irestore=diff(one,capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda'));step_I('I_replay');Inext=diff(expected,capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda'))
        if Irestore!=0 or Inext>1e-6:raise RuntimeError('I restore/nextstep failed')
        while solver.iteration<100:step_I('I_continuous')
        torch.save(capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda'),OUT/'I100_checkpoint.pt')
        with torch.no_grad():oracle.rgb.copy_(torch.as_tensor(solver.x,device='cuda',dtype=ci0.dtype).reshape_as(oracle.rgb))
        oracle.save(OUT/'I_colors.pt');I_colors=oracle().detach();scope=np.zeros(len(ctx.road['active']),bool);scope[ctx.active_world_rows]=reader.eligible
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        publics=[];evaluations={}
        for name,color,loss in [('RGB',rgb_colors,objective),('I',I_colors,ideal)]:
            asset=ctx.save_asset(color,scope,OUT/(name+'_asset'),training_record=dict(role='smoke_discard',source_binding=binding,original_field_metadata=field_binding))
            with torch.no_grad():
                expect={r:road(r,color,'final_missing_eval')for r in FRAMES};value,_=loss.objective({r:v['rgb']for r,v in expect.items()},'rgb'if name=='RGB'else'detail')
                evaluations[name]=float(value)
            scene,m,_=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json');data=dict(m.data);data['render_wh']=[1920,1280]
            headers=SceneManifest(data,asset/'persistent').camera_headers();provider=next(p for p in scene._providers if p.kind=='road')
            with torch.no_grad():
                for raw in FRAMES:
                    cam=make_native_camera(headers[raw]);actual=call('missing_public_clone',raw,lambda:scene._bindings().raster(cam,scene.pipe,0,provider.rows(raw,cam,scene.pipe),scene._background))
                    row=dict(asset=name,raw=raw,RGB_max=float((actual['rgb']-expect[raw]['rgb']).abs().max()),alpha_max=float((actual['alpha']-expect[raw]['alpha']).abs().max()))
                    publics.append(row)
                    if row['RGB_max']>1e-6 or row['alpha_max']!=0:raise RuntimeError('missing path public reload failed')
            scene.assert_persistent_unchanged();del scene,provider,expect;gc.collect();torch.cuda.empty_cache()
        ctx.assert_frozen();state.assert_frozen(full=True)
        if any(sha(p)!=h for p,h in binding.items()):raise RuntimeError('source changed during missing-path smoke')
        elapsed=time.monotonic()-started
        receipt=dict(schema='native_capacity_smoke_missing_paths.v1',complete=True,formal_updates=0,additional_accepted_updates=additional_updates,
            RGB600_exact_low_and_empty_moments=low_moments_absent,RGB_restore_max=restore_error,RGB601_nextstep_max=next_error,
            RGB_cursor=state.cursor,I_cursor=solver.iteration,I_restore_max=Irestore,I_nextstep_max=Inext,
            I_metadata=ideal_metadata,I_target='qualified case1 road_target_P/alpha only fixed context; never second premultiply',
            evaluation=evaluations,public1920_reload=publics,aggregate_physical_ledger=ledger,
            additional_seconds=elapsed,aggregate_active_work_seconds=prior['measured_active_work_seconds']+elapsed,
            aggregate_wall_seconds=prior['aggregate_wall_seconds']+elapsed,source_binding=binding,
            all_parameters_discard_for_formal=True,GPU_keep_on=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(OUT/'receipt.json',receipt);print('COMPLETED',json.dumps({k:v for k,v in receipt.items()if k not in('aggregate_physical_ledger','source_binding','I_metadata')}),flush=True)
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),aggregate_ledger=ledger,additional_updates=additional_updates,seconds=time.monotonic()-started));raise
    finally:
        if ctx is not None:ctx.close()
        value=read(lease);value.update(active=False,owner=None,ended_at=time.time());write(lease,value)


if __name__=='__main__':main()
