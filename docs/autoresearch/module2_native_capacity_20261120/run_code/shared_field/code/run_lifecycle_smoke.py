"""One authorized, disposable >=600s actual native lifecycle smoke. No formal run."""
from pathlib import Path
import copy,gc,hashlib,json,random,sys,time,traceback
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
OUT=B/'smoke_r1';OWNER='native_capacity_smoke';FRAMES=(86,110);ALL=(86,110,134,179,198)

def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for a in iter(lambda:f.read(1024*1024),b''):h.update(a)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'authorization.json',dict(schema='native_capacity_long_lifecycle_smoke.v1',GPU_authorized=True,
        user_authorization='Root explicit2026-11-20 lifecycle smoke before formal queue; disposal only',
        formal_train=False,minimum_actual_seconds=600,maximum_compute_seconds=750,closing_IO_separate=True,
        engineering_updates_cap=2500,nativeF_cap=20000,native_perviewB_cap=8000,
        smoke_stage_counts=[20,2480],formal_stage_counts_unchanged=[600,600],source_method_and_LRs_unchanged=True,
        additional_readonly_K_views=[134,179,198],GPU_keep_on=True))
    lease=OUT/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,
        authorization=str(OUT/'authorization.json'),phase='smoke_only',formal_allowed=False))
    ledger=dict(native_forwards=0,native_per_view_backwards=0,torch_backward_invocations=0,
        engineering_accepted_updates=0,shared_updates=0,oracle_updates=0,replay_updates=0,failed_attempts=0,
        forwards_by_label={},accepted_by_label={})
    ctx=None;started=time.monotonic();compute_started=None;bindings={}
    try:
        prep=read(B/'qa/preparation_audit_r2.json');profile=read(B/'gs_oracle/full_objective_profile_completion_r1/receipt.json')
        assert prep['complete']and profile['completed']
        assert profile['aggregate_physical_ledger']['native_forwards']==58
        assert profile['aggregate_physical_ledger']['native_per_view_backwards']==16
        assert profile['aggregate_physical_ledger']['engineering_parameter_updates']==7
        import torch
        torch.set_num_threads(1);torch.manual_seed(101);np.random.seed(101);random.seed(101);torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import real_frames,make_context,field_and_reader
        from native_capacity_loss import NativeCapacityLoss
        from shared_training_state import SharedTrainingState
        from box_quadratic_solver import ProjectedBoxQuadratic,capture_solver_checkpoint,restore_solver_checkpoint
        from gs_color_oracle import ColorLayout,BoundedGSColors
        paths=[Path(__file__),B/'protocol.json',B/'code/native_capacity_inputs.py',B/'code/native_capacity_loss.py',
            B/'shared_field/code/shared_field_gs_read.py',B/'shared_field/code/shared_training_state.py',
            B/'gs_oracle/code/box_quadratic_solver.py',B/'gs_oracle/code/gs_color_oracle.py',
            B/'qa/preparation_audit_r2.json',B/'gs_oracle/full_objective_profile_completion_r1/receipt.json']
        bindings={str(p):sha(p)for p in paths};(OUT/'source_snapshot').mkdir()
        for p in paths:
            if p.suffix=='.py':(OUT/'source_snapshot'/p.name).write_bytes(p.read_bytes())
        frames,data_binding=real_frames('seen');ctx=make_context(lease,OWNER,frames=ALL)
        model,reader,field_binding=field_and_reader(ctx.base_rgb,'cuda');elig=torch.as_tensor(reader.eligible,device='cuda')
        assert len(ctx.base_rgb)==151988 and int(elig.sum())==89887
        fixed={raw:torch.as_tensor(frames[raw]['alpha'],device='cuda')for raw in FRAMES}
        protocol=read(B/'protocol.json')
        def sync():torch.cuda.synchronize()
        def progress():
            write(OUT/'progress.json',dict(seconds=time.monotonic()-started,
                compute_seconds=None if compute_started is None else time.monotonic()-compute_started,ledger=ledger))
        def call(label,raw,fn):
            if ledger['native_forwards']>=20000:raise RuntimeError('smoke nativeF cap')
            ctx.require_lease();ledger['native_forwards']+=1
            ledger['forwards_by_label'][label]=ledger['forwards_by_label'].get(label,0)+1
            value=fn()
            if label not in('joint_K','K_feature','public_clone')and raw in fixed:
                if not torch.equal(value['alpha'].detach().reshape(1280,1920),fixed[raw]):raise RuntimeError('road alpha changed')
            return value
        def road(raw,colors,label='train_road'):return call(label,raw,lambda:ctx.render_road(raw,colors))
        class TrainContext:
            frames=FRAMES;active_world_rows=ctx.active_world_rows
            @property
            def native_calls(self):return ctx.native_calls
            def render_road(self,raw,colors):return road(raw,colors,'shared_road')
        trainctx=TrainContext()
        def accepted(kind,label):
            if ledger['engineering_accepted_updates']>=2500:raise RuntimeError('smoke accepted-update cap')
            ledger['engineering_accepted_updates']+=1;ledger[kind+'_updates']+=1
            ledger['accepted_by_label'][label]=ledger['accepted_by_label'].get(label,0)+1
            if 'replay'in label:ledger['replay_updates']+=1
        def state_hook(e):
            if e['event']=='native_backward_calls':
                ledger['native_per_view_backwards']+=e['amount']
                if ledger['native_per_view_backwards']>8000:raise RuntimeError('smoke nativeB cap')
            elif e['event']=='backward_calls':ledger['torch_backward_invocations']+=e['amount']
            elif e['event']=='parameter_updates':accepted('shared',current_label[0])
        with torch.no_grad():ci0=reader.colors(model).detach().clone()
        objective=NativeCapacityLoss(frames,device='cuda')
        with torch.no_grad():initial={r:road(r,ci0,'same_initial_F0')['rgb']for r in FRAMES}
        objective.calibrate(initial);torch.save(objective.snapshot(),OUT/'loss_snapshot.pt');write(OUT/'loss_snapshot.json',objective.snapshot());del initial
        identity=dict(role='smoke_discard',source_bindings=bindings,data_binding=data_binding,field_binding=field_binding)
        current_label=['setup']
        def new_state(m):return SharedTrainingState(m,reader,trainctx,objective,mode='detail',
            source_identity=identity,source_metadata=field_binding,purpose='smoke',ledger_hook=state_hook,stage_counts=(20,2480))
        state=new_state(model)
        write(OUT/'context_ready.json',dict(seconds=time.monotonic()-started,active=151988,eligible=89887,
            frames=list(ALL),source_bindings=bindings,field_binding=field_binding,data_binding=data_binding))
        print('CONTEXT_READY',time.monotonic()-started,flush=True)

        # Same-sort actual alpha*T for every new1920 K; no screen mask override.
        Kdir=OUT/'K_qualification';Kdir.mkdir();Krows=[]
        with torch.no_grad():
            for raw in ALL:
                rows,owners,cam,_=ctx.scene._rows(raw,(10,),(),False)
                rows=dict(rows,mask=rows['mask']&~ctx.scene._contains(owners,(10,)))
                offset=len(rows['means3D'])-len(ci0)
                if not torch.equal(rows['means3D'][offset:],ctx.base_packet(raw)['means3D']):raise RuntimeError('joint provider road-tail identity differs')
                marker=rows['colors_precomp'].new_zeros((len(rows['means3D']),24));marker[:,:2]=rows['features'][:,:2]
                marker[offset:,5]=elig.to(marker.dtype)
                value=call('K_feature',raw,lambda:ctx.scene._bindings().raster(cam,ctx.scene.pipe,0,dict(rows,features=marker),ctx.scene._background))
                weight=value['extra'][3];K=torch.as_tensor(ctx.K_masks[raw],device='cuda')
                row=dict(raw=raw,K_native_pixels=int(K.sum()),eligible_joint_alphaT_K_max=float(weight[K].max()),
                    K_positive_pixels=int((weight[K]>0).sum()),scope=89887,resolution=[1920,1280],actual_same_sort=True)
                np.savez_compressed(Kdir/f'{raw:03d}.npz',eligible_joint_alphaT=weight.cpu().numpy(),K=ctx.K_masks[raw],joint_alpha=value['alpha'].squeeze().cpu().numpy())
                Krows.append(row);print('PROTECTION',json.dumps(row),flush=True)
                del rows,owners,cam,marker,value,weight,K;gc.collect()
        write(Kdir/'receipt.json',dict(complete=True,scope_unmodified=True,all_exact_zero=all(x['K_positive_pixels']==0 for x in Krows),rows=Krows,
            nonzero_policy='report current fixed-scope failure; do not alter scope or copy screen pixels; smoke remains disposable'))

        def difference(a,b):
            if torch.is_tensor(a):return float((a.double()-b.double()).abs().max())if a.numel()else 0.
            if isinstance(a,np.ndarray):return float(np.max(abs(a.astype(float)-b.astype(float))))if a.size else 0.
            if isinstance(a,dict):return max([difference(a[k],b[k])for k in a if k!='physical']or[0.])if a.keys()==b.keys()else float('inf')
            if isinstance(a,(list,tuple)):return max([difference(x,y)for x,y in zip(a,b)]or[0.])if len(a)==len(b)else float('inf')
            return abs(float(a)-float(b))if isinstance(a,(int,float,np.number))else(0. if a==b else float('inf'))
        def field_step(label):
            if ledger['engineering_accepted_updates']>=2500:raise RuntimeError('engineering cap before shared update')
            current_label[0]=label;t=time.monotonic();row=state.step();sync()
            compact={k:v for k,v in row.items()if k not in('terms','physical')};compact['seconds']=time.monotonic()-t
            with(OUT/'shared_steps.jsonl').open('a')as f:f.write(json.dumps(compact)+'\n')
            return row
        compute_started=time.monotonic()
        for _ in range(5):field_step('shared_B_initial')
        state.save(OUT/'B5_checkpoint.pt');B5=state.snapshot();field_step('shared_B_reference');expected=state.snapshot()
        state.load(OUT/'B5_checkpoint.pt');Brestore=difference(B5,state.snapshot());field_step('shared_B_replay');Breplay=difference(expected,state.snapshot())
        if Brestore!=0 or Breplay>1e-6:raise RuntimeError('shared B full restore/replay failed')
        while state.cursor<21:field_step('shared_BtoC_entry')
        state.save(OUT/'C21_checkpoint.pt');C21=state.snapshot();field_step('shared_C_reference');expected=state.snapshot()
        fresh_model,_=type(model).load(B/'shared_field/preparation_r1/initial_field.pt',device='cuda')
        new=new_state(fresh_model);new.load(OUT/'C21_checkpoint.pt')
        Crestore=difference(C21,new.snapshot());del state,model;state=new;model=fresh_model;gc.collect()
        field_step('shared_C_replay');Creplay=difference(expected,state.snapshot())
        if Crestore!=0 or Creplay>1e-6:raise RuntimeError('fresh shared C full restore/replay failed')
        layout=ColorLayout(ctx.active_world_rows,reader.eligible);oracle=BoundedGSColors(ci0,layout);positions=oracle.eligible_positions
        def box_evaluate(x,need_gradient):
            if ledger['native_per_view_backwards']+2*int(need_gradient)>8000:raise RuntimeError('B cap before oracle')
            parameter=torch.as_tensor(x,device='cuda',dtype=ci0.dtype).detach().requires_grad_(need_gradient)
            with torch.set_grad_enabled(need_gradient):
                colors=ci0.index_copy(0,positions,parameter.reshape(-1,3));pred={r:road(r,colors,'oracle_gradient'if need_gradient else'oracle_value')['rgb']for r in FRAMES}
                value,_=objective.objective(pred,'detail')
                if need_gradient:
                    gradient=torch.autograd.grad(value,parameter)[0]
                    ledger['native_per_view_backwards']+=2;ledger['torch_backward_invocations']+=1
                else:gradient=None
            f=float(value.detach());g=None if gradient is None else gradient.detach().cpu().numpy().astype(np.float64)
            del pred,colors,value,parameter,gradient;return f,g,dict(role='smoke_not_formal',objective='common_DETAIL')
        solver=ProjectedBoxQuadratic(ci0[positions].cpu().numpy(),box_evaluate,identity=identity,**protocol['arms']['G_ORACLE']['recipe'])
        def box_step(label):
            if ledger['engineering_accepted_updates']>=2500:raise RuntimeError('engineering cap before oracle update')
            row=solver.step();accepted('oracle',label)
            with(OUT/'oracle_steps.jsonl').open('a')as f:f.write(json.dumps(dict(row,label=label))+'\n')
            return row
        box_step('oracle_initial');one=capture_solver_checkpoint(solver,role='smoke',frames=FRAMES,device='cuda');torch.save(one,OUT/'oracle1_checkpoint.pt')
        box_step('oracle_reference');expected=capture_solver_checkpoint(solver,role='smoke',frames=FRAMES,device='cuda')
        restore_solver_checkpoint(torch.load(OUT/'oracle1_checkpoint.pt',map_location='cpu',weights_only=False),solver,role='smoke',frames=FRAMES,device='cuda')
        Orestore=difference(one,capture_solver_checkpoint(solver,role='smoke',frames=FRAMES,device='cuda'))
        box_step('oracle_replay');Oreplay=difference(expected,capture_solver_checkpoint(solver,role='smoke',frames=FRAMES,device='cuda'))
        if Orestore!=0 or Oreplay>1e-6:raise RuntimeError('oracle full restore/replay failed')
        lifecycle=dict(shared_B_restore_max=Brestore,shared_B_nextstep_max=Breplay,shared_fresh_C_restore_max=Crestore,shared_C_nextstep_max=Creplay,
            oracle_restore_max=Orestore,oracle_nextstep_max=Oreplay,stage_C_entered=True,fullstate_RNG_Adam_objective_included=True)
        write(OUT/'lifecycle.json',lifecycle);progress();print('LIFECYCLE_PASS',json.dumps(lifecycle),flush=True)
        oracle_stopped=False
        while time.monotonic()-compute_started<600:
            if ledger['engineering_accepted_updates']>=2500:break
            if time.monotonic()-compute_started>750:raise RuntimeError('750s compute cap')
            field_step('shared_continuous')
            if not oracle_stopped and ledger['engineering_accepted_updates']<2500:
                row=box_step('oracle_continuous')
                if solver.iteration%100==0 and row['measured_box_FW_gap']/max(row['objective'],1e-3)<=.01:oracle_stopped=True
            if ledger['engineering_accepted_updates']%10<2:
                progress();print('PROGRESS',json.dumps(dict(compute_s=time.monotonic()-compute_started,**{k:v for k,v in ledger.items()if not isinstance(v,dict)})),flush=True)
            if state.cursor%100==0:
                folder=OUT/'periodic';folder.mkdir(exist_ok=True);state.save(folder/f'shared_{state.cursor:04d}.pt')
        computational_seconds=time.monotonic()-compute_started
        state.save(OUT/'shared_final_checkpoint.pt');torch.save(capture_solver_checkpoint(solver,role='smoke',frames=FRAMES,device='cuda'),OUT/'oracle_final_checkpoint.pt')
        model.save(OUT/'world_field.pt',dict(role='engineering_long_smoke_discard',reader='ordinary_GS_physical_Sigma_GH31',source_metadata=field_binding,logical_cursor=state.cursor))
        reload_field,_=type(model).load(OUT/'world_field.pt',device='cuda')
        with torch.no_grad():
            field_colors=reader.colors(model).detach();field_requery=reader.colors(reload_field).detach()
            if not torch.equal(field_colors,field_requery):raise RuntimeError('field persistent requery changed ci')
            oracle.rgb.copy_(torch.as_tensor(solver.x,device='cuda',dtype=ci0.dtype).reshape_as(oracle.rgb))
        oracle.save(OUT/'oracle_colors.pt');oracle_reloaded=BoundedGSColors.load(OUT/'oracle_colors.pt',device='cuda')
        if not torch.equal(oracle_reloaded(),oracle()):raise RuntimeError('oracle ci reload failed')
        colors={'shared':field_requery,'oracle':oracle_reloaded()};scope=np.zeros(len(ctx.road['active']),bool);scope[ctx.active_world_rows]=reader.eligible
        assets={};evaluations={};expected={}
        for name,color in colors.items():
            if not torch.equal(color[~elig],ctx.base_rgb[~elig])or not bool(torch.isfinite(color).all()):raise RuntimeError('readonly/nonfinite output ci')
            assets[name]=ctx.save_asset(color,scope,OUT/(name+'_asset'),training_record=dict(role='engineering_smoke_discard_not_formal',source_bindings=bindings,source_metadata=field_binding))
            folder=OUT/'evaluation'/name;folder.mkdir(parents=True)
            with torch.no_grad():
                outputs={r:road(r,color,'final_eval')for r in FRAMES};value,parts=objective.objective({r:o['rgb']for r,o in outputs.items()},'detail')
                expected[name]={r:{k:v[k].detach().clone()for k in('rgb','alpha')}for r,v in outputs.items()}
                evaluations[name]=dict(DETAIL=float(value),logical_cursor=state.cursor if name=='shared'else solver.iteration,readonly_RGB_exact=True)
                for raw,v in outputs.items():np.savez_compressed(folder/f'{raw:03d}.npz',road_rgb=v['rgb'].cpu().numpy(),road_alpha=v['alpha'].squeeze().cpu().numpy(),K=ctx.K_masks[raw])
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        public=[]
        for name,asset in assets.items():
            scene,m,_=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
            data=dict(m.data);data['render_wh']=[1920,1280];headers=SceneManifest(data,asset/'persistent').camera_headers()
            provider=next(p for p in scene._providers if p.kind=='road')
            with torch.no_grad():
                for raw in FRAMES:
                    cam=make_native_camera(headers[raw]);actual=call('public_clone',raw,lambda:scene._bindings().raster(cam,scene.pipe,0,provider.rows(raw,cam,scene.pipe),scene._background))
                    target=expected[name][raw];row=dict(asset=name,raw=raw,RGB_max=float((actual['rgb']-target['rgb']).abs().max()),
                        alpha_max=float((actual['alpha']-target['alpha']).abs().max()),RGB8_max=int((torch.round(actual['rgb']*255)-torch.round(target['rgb']*255)).abs().max()),resolution=[1920,1280])
                    public.append(row)
                    if row['RGB_max']>1e-6 or row['alpha_max']!=0:raise RuntimeError('true1920 public reload failed')
            scene.assert_persistent_unchanged();del scene,provider;gc.collect();torch.cuda.empty_cache()
        joint=[]
        with torch.no_grad():
            for raw in FRAMES:
                parent=call('joint_K',raw,lambda:ctx.render_joint(raw));K=torch.as_tensor(ctx.K_masks[raw],device='cuda')
                for name,color in colors.items():
                    value=call('joint_K',raw,lambda:ctx.render_joint(raw,color));row=dict(asset=name,raw=raw,K_pixels=int(K.sum()),
                        K_RGB_max=float((value['rgb']-parent['rgb']).abs().amax(0)[K].max()),alpha_max=float((value['alpha']-parent['alpha']).abs().max()))
                    joint.append(row)
        ctx.assert_frozen();state.assert_frozen(full=True)
        if any(sha(p)!=h for p,h in bindings.items()):raise RuntimeError('executed source changed during smoke')
        receipt=dict(schema='native_capacity_long_smoke_result.v1',complete=computational_seconds>=600,formal_updates=0,
            measured_computational_seconds=computational_seconds,total_seconds=time.monotonic()-started,physical_ledger=ledger,
            shared_logical_cursor=state.cursor,oracle_logical_cursor=solver.iteration,shared_state_physical=state.physical,
            lifecycle=lifecycle,evaluation=evaluations,public1920_reload=public,joint_K=joint,
            source_only_K_qualification=str(Kdir/'receipt.json'),all5view_K_eligible_mass_zero=all(x['K_positive_pixels']==0 for x in Krows),
            source_bindings=bindings,field_metadata=field_binding,model_geometry_coverage_fixed=True,all_smoke_parameters_discard=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU_keep_on=True)
        write(OUT/'receipt.json',receipt);progress();print('COMPLETED',json.dumps({k:v for k,v in receipt.items()if k not in('source_bindings','field_metadata','physical_ledger')}),flush=True)
        if computational_seconds<600:raise RuntimeError('updates cap before minimum600s; preserve actual measured smoke')
    except Exception:
        ledger['failed_attempts']+=1;write(OUT/'failed.json',dict(error=traceback.format_exc(),ledger=ledger,seconds=time.monotonic()-started,formal_updates=0));raise
    finally:
        if ctx is not None:ctx.close()
        state=read(lease);state.update(active=False,owner=None,ended_at=time.time());write(lease,state)


if __name__=='__main__':main()
