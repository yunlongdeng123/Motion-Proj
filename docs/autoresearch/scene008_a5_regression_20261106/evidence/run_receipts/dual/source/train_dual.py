"""原 B 30K时程双车入口；正式预算5K→10K，smoke独立。"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse, json, sys, time, gc, hashlib
ROOT=Path(__file__).resolve().parent.parent
ENGINE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
TOOLS=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/tools')

class CompleteBoundary(Exception):pass

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','resume2step','formal','extend'],required=True)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke-seconds',type=float,default=540)
    ap.add_argument('--smoke-policy',choices=['adaptive500','long540'],default='adaptive500')
    ap.add_argument('--stop-iteration',type=int,default=10000)
    opts=ap.parse_args();out=opts.output;out.mkdir(parents=True,exist_ok=False)
    provenance=[Path(__file__),ROOT/'config.json',ROOT/'identity_contract.json']+sorted(ENGINE.rglob('*.py'))
    write(out/'runtime_source_manifest.json',dict(interpreter=sys.executable,executed_wrapper=str(Path(__file__).resolve()),
        files={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in provenance}))
    assert ROOT.name=='scene008_a5_old_dual30k_20261106_r1'
    budget=json.loads((ROOT/'active_budget.json').read_text())
    assert budget['dual_fresh_stop']==10000 and budget['learning_rate_horizon']==30000
    if opts.mode in ['resume2step','extend']:
        assert opts.checkpoint is not None
        start_iteration=int(opts.checkpoint.stem.replace('chkpnt',''))
        config=json.loads((opts.checkpoint.parent/'effective_config.json').read_text())
    else:
        assert opts.checkpoint is None
        start_iteration=0;config=json.loads((ROOT/'config.json').read_text())
    assert config['iterations']==config['position_lr_max_steps']==30000
    config.update(model_path=str(out),start_checkpoint=str(opts.checkpoint) if opts.checkpoint else None)
    diagnostic_keys=['checkpoint_iterations','save_iterations','test_iterations']
    if opts.mode=='formal':
        reference=json.loads((ROOT/'config.json').read_text())
        assert opts.stop_iteration==5000
        for key in diagnostic_keys:config[key]=sorted(set(config[key])|{5000})
        ignore=['model_path','start_checkpoint']+diagnostic_keys
        assert {k:v for k,v in config.items() if k not in ignore}=={k:v for k,v in reference.items() if k not in ignore}
        write(out/'user_budget_and_diagnostic_cadence.json',dict(total_formal_budget=10000,phase_stop=5000,
            added_complete_checkpoint_eval_node=5000,learning_rate_horizon=30000,method_changes={},
            note='Additional diagnostic evaluation is explicitly requested; its captured RNG state is continued exactly.'))
    if opts.mode=='extend':assert start_iteration==5000 and opts.stop_iteration==10000
    write(out/'effective_config.json',config)
    sys.path[:0]=[str(ENGINE),str(TOOLS)]
    import torch, numpy as np
    from omegaconf import OmegaConf
    import train
    from utils import paper_checkpoint as checkpoint
    from utils import hybrid30k_runtime as runtime
    from utils.general_utils import seed_everything
    from run_v4_arm import bind_scene_contract
    from compare_v1_states import differences
    assert Path(train.__file__).resolve()==ENGINE/'train.py' and torch.cuda.is_available()
    assert sorted(Path(config['source_path']).glob('points3d_dynamic_*.ply'))==[Path(config['source_path'])/f'points3d_dynamic_{i}.ply' for i in range(2)]
    bind_scene_contract(train,config)
    bound_scene=train.Scene
    def fixed_identity_scene(*args,**kwargs):
        scene=bound_scene(*args,**kwargs)
        identity=json.loads((ROOT/'identity_contract.json').read_text())
        assert identity['dynamic_actor_aliases']==[6,37] and identity['retained_aliases']==[5,26,18]
        assert identity['dynamic_source_real_points']==[11625,98452]
        assert len(scene.dynamic_gaussians.get_xyz)==identity['source_real_point_count']==110077
        assert len(scene.gaussians.get_xyz)==590681 and len(scene.getTrainCameras())==49
        write(out/'actual_identity_initialization.json',dict(status='PASS_FULL_ORIGINAL_A6_A37_INITIALIZATION',dynamic_aliases=[6,37],
            raw_track_ids=identity['dynamic_raw_track_ids'],dynamic_rows_before_any_update=len(scene.dynamic_gaussians.get_xyz),
            static_rows_before_any_update=len(scene.gaussians.get_xyz),dynamic_random_init_point=config['dynamic_random_init_point'],
            actor_ply_paths=config['followup_scene_spec']['actor_ply_paths'],camera_count=len(scene.getTrainCameras())))
        return scene
    train.Scene=fixed_identity_scene
    seed_everything(config['seed']);torch.set_num_threads(1)
    original_publish=checkpoint.publish_checkpoint_event;original_restore=checkpoint.restore_training_state
    original_record=runtime.record_step;started=time.monotonic();last=[0];diagnostic_node=[None]
    first_record_time=[None];smoke_prediction=[None]
    original_adam_step=torch.optim.Adam.step

    def finite_adam_step(optimizer,*args,**kwargs):
        if last[0]%100==0:
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError('nonfinite backward gradient')
        return original_adam_step(optimizer,*args,**kwargs)

    def record(args,iteration,camera,time_shift,static,dynamic,losses):
        if first_record_time[0] is None:first_record_time[0]=time.monotonic()
        original_record(args,iteration,camera,time_shift,static,dynamic,losses)
        values={k:float(v) for k,v in losses.items()}
        if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('nonfinite loss')
        last[0]=iteration
        if iteration%100==0:
            rec=dict(iteration=iteration,elapsed_seconds=time.monotonic()-started,
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                static_rows=len(static.get_xyz),dynamic_rows=len(dynamic.get_xyz),losses=values)
            with (out/'runtime.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            write(out/'progress.json',rec)
        if opts.mode=='smoke' and (iteration>=500 or time.monotonic()-started>=opts.smoke_seconds) and smoke_prediction[0] is None:
            observed=(time.monotonic()-first_record_time[0])/max(iteration-1,1)
            # Existing native single10K measured1730s; scale by full input-point ratio.
            # Add240s for the extra5K diagnostic evaluation and exact-resume entry.
            historical=1730*(110077/98452)
            predicted=max(observed*10000,historical)+240
            smoke_prediction[0]=dict(policy=opts.smoke_policy,observed_complete_updates_before_record=iteration-1,
                observed_train_seconds_per_update=observed,observed_train_10k_seconds=observed*10000,
                historical_single10k_seconds=1730,input_point_ratio=110077/98452,
                historical_scaled_10k_seconds=historical,extra5k_eval_resume_allowance_seconds=240,
                predicted_formal_total_seconds=predicted,long_smoke_required=predicted>=3600,
                long_experiment_threshold_seconds=3600,long_smoke_min_seconds=opts.smoke_seconds,
                excludes='smoke updates and zero-update reference evaluations')
            write(out/'smoke_time_policy.json',smoke_prediction[0])
        short_ready=opts.mode=='smoke' and opts.smoke_policy=='adaptive500' and smoke_prediction[0] is not None and not smoke_prediction[0]['long_smoke_required']
        long_ready=opts.mode=='smoke' and time.monotonic()-started>=opts.smoke_seconds
        if opts.mode=='smoke' and diagnostic_node[0] is None and (short_ready or long_ready):
            # Fresh smoke requests one native complete bundle at its time limit.
            diagnostic_node[0]=iteration
            before={key:list(args[key]) for key in diagnostic_keys}
            for key in diagnostic_keys:
                args[key]=sorted(set(args[key])|{iteration});config[key]=list(args[key])
            write(out/'effective_config.json',config)
            write(out/'diagnostic_schedule_smoke_only.json',dict(
                status='ADDED_ONE_SMOKE_ONLY_DIAGNOSTIC_NODE',fresh_run=True,iteration=iteration,
                elapsed_seconds=time.monotonic()-started,method_changes={},before=before,
                after={key:list(args[key]) for key in diagnostic_keys}))
        if opts.mode=='resume2step' and iteration>=start_iteration+3:
            # record_step 位于反向前，前两步已完整更新；此步不计为更新。
            write(out/'resume_entry.json',dict(status='PASS_NEXT_TRAINING_ENTRY',completed_extra_updates=2,entered_iteration=iteration))
            raise CompleteBoundary()

    def restore(path,iteration,scene,models,args,sgcm=None):
        stack,ema=original_restore(path,iteration,scene,models,args,sgcm=sgcm)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        current=checkpoint.capture_training_state(iteration,scene,stack,models,args,sgcm=sgcm,ema=ema)
        ignored=['model_path','start_checkpoint']
        config_differences={k:dict(saved=saved['config'].get(k),current=current['config'].get(k)) for k in set(saved['config'])|set(current['config']) if saved['config'].get(k)!=current['config'].get(k)}
        assert set(config_differences)<=set(ignored),config_differences
        for item in [saved,current]:item['config']={k:v for k,v in item['config'].items() if k not in ignored}
        errors=differences(saved,current,'continuation')
        for name,prefix in [('static','chkpnt'),('dynamic','dynamic_chkpnt'),('environment','env_light_chkpnt')]:
            old,step=torch.load(Path(path).parent/f'{prefix}{iteration}.pth',map_location='cpu',weights_only=False)
            errors+=differences(old,models[name].capture(),name);del old;gc.collect()
        write(out/'restore_audit.json',dict(status='PASS_EXACT_FULL_RESTORE' if not errors else 'FAIL',iteration=iteration,differences=errors,
            includes='parameters,Adam,Python/NumPy/Torch/CUDA RNG,camera sampler,SGCM,EMA,runtime',updates=0,
            bookkeeping_config_differences=config_differences,checkpoint=str(opts.checkpoint)))
        if errors:raise RuntimeError(str(errors))
        return stack,ema

    def publish(root,iteration,*args,**kwargs):
        event=original_publish(root,iteration,*args,**kwargs)
        write(out/'last_complete_node.json',dict(iteration=iteration,elapsed_seconds=time.monotonic()-started,event=event['model_path']))
        if (opts.mode=='smoke' and diagnostic_node[0]==iteration) or (opts.mode in ['formal','extend'] and iteration==opts.stop_iteration):
            write(out/'returned.json',dict(status='PASS_COMPLETE_BOUNDARY',mode=opts.mode,iteration=iteration,
                source_iteration=start_iteration,actual_additional_iterations=iteration-start_iteration,
                accounting='smoke_only_extra' if opts.mode=='smoke' else 'fresh_formal_total10000_in_5000_then_5000_phases',
                elapsed_seconds=time.monotonic()-started,
                learning_rate_horizon=30000,formal_stop=10000,phase_stop=opts.stop_iteration,max_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_reserved_bytes=torch.cuda.max_memory_reserved(),quality_accepted=False,human_verdict=None))
            raise CompleteBoundary()
        return event

    runtime.record_step=record;checkpoint.publish_checkpoint_event=publish;checkpoint.restore_training_state=restore
    torch.optim.Adam.step=finite_adam_step
    os.chdir(ENGINE)
    try:train.training(OmegaConf.create(config))
    except CompleteBoundary:pass
    if opts.mode!='resume2step':assert (out/'returned.json').exists()
    else:assert (out/'resume_entry.json').exists()
    print(json.dumps(dict(status='TRAINING_ENTRY_RETURNED',mode=opts.mode,last_entered=last[0],elapsed_seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
