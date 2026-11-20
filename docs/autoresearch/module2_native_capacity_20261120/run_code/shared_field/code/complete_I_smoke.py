"""Finish registered analytic-I lifecycle from fresh ci0; no RGB600 replay."""
from pathlib import Path
import copy,gc,json,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1';B=Q/'gpu_20261119/native_capacity_20261120_r1'
P=B/'smoke_r1';A=P/'RGB_replay_audit_r1';OUT=P/'I_completion_r1';OWNER='native_capacity_smoke';FRAMES=(86,110)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=False);prefix=read(A/'receipt.json');ledger=copy.deepcopy(prefix['aggregate_ledger']);started=time.monotonic();ctx=None;updates=0
    write(OUT/'authorization.json',dict(GPU_authorized=True,formal_allowed=False,root_authorization='AnalyticI100 and complete state/persistence; aggregate<=4000updates/20000F/8000B',maximum_new_updates=102))
    lease=OUT/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(OUT/'authorization.json')))
    try:
        import torch
        torch.set_num_threads(1)
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import make_context,field_and_reader,real_frames
        from native_capacity_loss import NativeCapacityLoss,LossSpec
        from native_ideal_inputs import load_ideal_frames
        from shared_training_state import SharedTrainingState
        from shared_field_gs_read import file_sha
        from gs_color_oracle import ColorLayout,BoundedGSColors
        from box_quadratic_solver import ProjectedBoxQuadratic,capture_solver_checkpoint,restore_solver_checkpoint
        from smoke_missing_paths import diff
        protocol=read(B/'protocol.json');ctx=make_context(lease,OWNER);model,reader,metadata=field_and_reader(ctx.base_rgb,'cuda')
        frames,data_binding=real_frames('seen');fixed={r:torch.as_tensor(frames[r]['alpha'],device='cuda')for r in FRAMES}
        def call(label,raw,fn):
            if ledger['native_forwards']>=20000:raise RuntimeError('F cap')
            ledger['native_forwards']+=1;ledger['forwards_by_label'][label]=ledger['forwards_by_label'].get(label,0)+1;return fn()
        def road(raw,color,label):
            v=call(label,raw,lambda:ctx.render_road(raw,color))
            if not torch.equal(v['alpha'].squeeze().detach(),fixed[raw]):raise RuntimeError('alpha changed')
            return v
        with torch.no_grad():ci0=reader.colors(model).detach().clone();initial={r:road(r,ci0,'I_initial_ci0')['rgb']for r in FRAMES}
        old=torch.load(A/'601_replay.pt',map_location='cpu',weights_only=False)
        lossRGB=NativeCapacityLoss(frames,device='cuda');lossRGB.restore(old['objective'])
        state=SharedTrainingState(model,reader,ctx,lossRGB,mode='rgb',source_identity=old['source_identity'],source_metadata=old['source_metadata'],purpose='smoke')
        state.restore(old)
        with torch.no_grad():RGBcolors=reader.colors(model).detach().clone()
        model.save(OUT/'RGB_field.pt',dict(role='smoke_discard',source_metadata=metadata,checkpoint=str(A/'601_replay.pt')))
        ideal_frames,ideal_metadata=load_ideal_frames(protocol)
        ideal=NativeCapacityLoss(ideal_frames,spec=LossSpec(source_weights=(('I',1.),)),device='cuda');ideal.calibrate(initial)
        torch.save(ideal.snapshot(),OUT/'I_loss_snapshot.pt');write(OUT/'I_loss_snapshot.json',ideal.snapshot());del initial
        layout=ColorLayout(ctx.active_world_rows,reader.eligible);oracle=BoundedGSColors(ci0,layout);pos=oracle.eligible_positions
        def evaluate(x,need_gradient):
            if ledger['native_per_view_backwards']+2*int(need_gradient)>8000:raise RuntimeError('B cap')
            p=torch.as_tensor(x,device='cuda',dtype=ci0.dtype).detach().requires_grad_(need_gradient)
            with torch.set_grad_enabled(need_gradient):
                color=ci0.index_copy(0,pos,p.reshape(-1,3));pred={r:road(r,color,'I_gradient'if need_gradient else'I_value')['rgb']for r in FRAMES}
                value,_=ideal.objective(pred,'detail');g=torch.autograd.grad(value,p)[0]if need_gradient else None
            if need_gradient:ledger['native_per_view_backwards']+=2;ledger['torch_backward_invocations']+=1
            return float(value.detach()),None if g is None else g.detach().cpu().numpy().astype(np.float64),{'role':'I_smoke_only','premult_target_correct':True}
        binding={str(p):file_sha(p)for p in [Path(__file__),B/'code/native_ideal_inputs.py',B/'protocol.json',A/'receipt.json']}
        solver=ProjectedBoxQuadratic(ci0[pos].cpu().numpy(),evaluate,identity={'source':binding,'ideal':ideal_metadata,'objective':ideal.snapshot()},**protocol['arms']['G_ORACLE']['recipe'])
        def step(label):
            nonlocal updates
            if updates>=102 or ledger['engineering_accepted_updates']>=4000:raise RuntimeError('update cap')
            row=solver.step();updates+=1;ledger['engineering_accepted_updates']+=1;ledger['oracle_updates']+=1
            ledger['accepted_by_label'][label]=ledger['accepted_by_label'].get(label,0)+1
            if 'replay'in label:ledger['replay_updates']+=1
            with(OUT/'I_steps.jsonl').open('a')as f:f.write(json.dumps(dict(row,label=label))+'\n')
        step('I1');one=capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda');torch.save(one,OUT/'I1.pt')
        step('I2_reference');reference=capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda')
        restore_solver_checkpoint(torch.load(OUT/'I1.pt',map_location='cpu',weights_only=False),solver,role='smoke_I',frames=FRAMES,device='cuda')
        restore_error=diff(one,capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda'))
        step('I2_replay');replay=capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda')
        replay_parts={k:diff(reference['solver'][k],replay['solver'][k])for k in('x','gradient','value','L')}
        if restore_error!=0:raise RuntimeError('I full restore not exact')
        while solver.iteration<100:step('I_continuous')
        torch.save(capture_solver_checkpoint(solver,role='smoke_I',frames=FRAMES,device='cuda'),OUT/'I100.pt')
        with torch.no_grad():oracle.rgb.copy_(torch.as_tensor(solver.x,device='cuda',dtype=ci0.dtype).reshape_as(oracle.rgb))
        oracle.save(OUT/'I_colors.pt');Icolors=oracle().detach();scope=np.zeros(len(ctx.road['active']),bool);scope[ctx.active_world_rows]=reader.eligible
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        public=[];evaluation={};Krows=[]
        with torch.no_grad():parent={r:call('I_completion_parent_joint',r,lambda r=r:ctx.render_joint(r))for r in FRAMES}
        for name,color,objective,mode in [('RGB',RGBcolors,lossRGB,'rgb'),('I',Icolors,ideal,'detail')]:
            asset=ctx.save_asset(color,scope,OUT/(name+'_asset'),training_record={'role':'smoke_discard','source':binding,'original_field_metadata':metadata})
            with torch.no_grad():expected={r:road(r,color,'I_completion_eval')for r in FRAMES};value,_=objective.objective({r:v['rgb']for r,v in expected.items()},mode)
            evaluation[name]=float(value)
            scene,m,_=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json');data=dict(m.data);data['render_wh']=[1920,1280]
            headers=SceneManifest(data,asset/'persistent').camera_headers();provider=next(p for p in scene._providers if p.kind=='road')
            with torch.no_grad():
                for raw in FRAMES:
                    cam=make_native_camera(headers[raw]);actual=call('I_completion_public_clone',raw,lambda:scene._bindings().raster(cam,scene.pipe,0,provider.rows(raw,cam,scene.pipe),scene._background))
                    row={'asset':name,'raw':raw,'RGB_max':float((actual['rgb']-expected[raw]['rgb']).abs().max()),'alpha_max':float((actual['alpha']-expected[raw]['alpha']).abs().max())};public.append(row)
                    if row['RGB_max']>1e-6 or row['alpha_max']!=0:raise RuntimeError('I/RGB public reload failed')
                    joint=call('I_completion_joint',raw,lambda:ctx.render_joint(raw,color));mask=torch.as_tensor(ctx.K_masks[raw],device='cuda')
                    Krows.append(dict(asset=name,raw=raw,K_pixels=int(mask.sum()),RGB_max=float((joint['rgb']-parent[raw]['rgb']).abs().amax(0)[mask].max()),alpha_max=float((joint['alpha']-parent[raw]['alpha']).abs().max())))
            scene.assert_persistent_unchanged();del scene,provider,expected;gc.collect();torch.cuda.empty_cache()
        ctx.assert_frozen();state.assert_frozen(full=True)
        if any(file_sha(p)!=h for p,h in binding.items()):raise RuntimeError('source changed')
        receipt=dict(schema='I_lifecycle_completion.v1',complete=True,I_full_restore_max=restore_error,I_nextstep_per_component=replay_parts,
            I_logical_cursor=solver.iteration,I_physical_accepted_updates=updates,I_metadata=ideal_metadata,
            public1920_reload=public,joint_K=Krows,evaluation=evaluation,aggregate_ledger=ledger,seconds=time.monotonic()-started,
            source_SHA256=binding,all_parameters_discard_for_formal=True,formal_updates=0,GPU_keep_on=True)
        write(OUT/'receipt.json',receipt);print('COMPLETED',json.dumps({k:v for k,v in receipt.items()if k not in('aggregate_ledger','I_metadata','source_SHA256')}),flush=True)
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),aggregate_ledger=ledger,updates=updates,seconds=time.monotonic()-started));raise
    finally:
        if ctx is not None:ctx.close()
        v=read(lease);v.update(active=False,owner=None,ended_at=time.time());write(lease,v)
if __name__=='__main__':main()
