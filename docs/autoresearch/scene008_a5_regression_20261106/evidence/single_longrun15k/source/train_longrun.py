"""Native single-a37 continuation from existing full11977; old engineering ledger unchanged."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['continue','resume2step'],required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stop-iteration',type=int,choices=[15000,20000,30000],required=True)
    opts=ap.parse_args();out=opts.output;out.mkdir(parents=True,exist_ok=False)
    provenance=[Path(__file__),ROOT/'config.json',ROOT/'identity_contract.json']+sorted(ENGINE.rglob('*.py'))
    write(out/'runtime_source_manifest.json',dict(interpreter=sys.executable,executed_wrapper=str(Path(__file__).resolve()),
        files={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in provenance}))
    start_iteration=int(opts.checkpoint.stem.replace('chkpnt',''))
    contract=json.loads((ROOT/'longrun_contract.json').read_text())
    assert start_iteration in [11977,15000,20000] and start_iteration<opts.stop_iteration
    if start_iteration==11977:assert opts.checkpoint.resolve()==Path(contract['source_checkpoint']).resolve()
    else:assert opts.checkpoint.parent.parent.resolve()==ROOT.resolve()
    config=json.loads((opts.checkpoint.parent/'effective_config.json').read_text())
    assert config['iterations']==config['position_lr_max_steps']==30000 and config['seed']==0
    config.update(model_path=str(out),start_checkpoint=str(opts.checkpoint))
    diagnostic_keys=['checkpoint_iterations','save_iterations','test_iterations']
    write(out/'effective_config.json',config)
    sys.path[:0]=[str(ENGINE),str(TOOLS)]
    import torch, numpy as np
    # Native loader writes a transform even on restore. Keep old inputs read-only.
    transform_source=Path(config['source_path'])/'transform_matrix.txt'
    transform_bytes=transform_source.read_bytes();original_savetxt=np.savetxt
    def protected_savetxt(filename,array,*args,**kwargs):
        if isinstance(filename,(str,os.PathLike)) and Path(filename).resolve()==transform_source.resolve():
            destination=out/'loader_transform_matrix.txt'
            result=original_savetxt(destination,array,*args,**kwargs)
            assert destination.read_bytes()==transform_bytes and transform_source.read_bytes()==transform_bytes
            write(out/'loader_transform_readonly_audit.json',dict(status='PASS_BYTE_EXACT_TRANSFORM_REDIRECTION',
                original=str(transform_source),destination=str(destination),sha256=hashlib.sha256(transform_bytes).hexdigest(),
                source_write=False,method_changes={}))
            return result
        return original_savetxt(filename,array,*args,**kwargs)
    np.savetxt=protected_savetxt
    from omegaconf import OmegaConf
    import train
    from utils import paper_checkpoint as checkpoint
    from utils import hybrid30k_runtime as runtime
    from utils.general_utils import seed_everything
    from run_v4_arm import bind_scene_contract
    from compare_v1_states import differences
    assert Path(train.__file__).resolve()==ENGINE/'train.py' and torch.cuda.is_available()
    assert list(Path(config['source_path']).glob('points3d_dynamic_*.ply'))==[Path(config['source_path'])/'points3d_dynamic_0.ply']
    import gaussian_renderer
    from scene import gaussian_model
    assert Path(gaussian_renderer.__file__).resolve()==ENGINE/'gaussian_renderer/__init__.py'
    assert Path(gaussian_model.__file__).resolve()==ENGINE/'scene/gaussian_model.py'
    for relative,expected in contract['native_core_sha256'].items():
        assert hashlib.sha256((ENGINE/relative).read_bytes()).hexdigest()==expected,relative
    bind_scene_contract(train,config)
    bound_scene=train.Scene
    def fixed_identity_scene(*args,**kwargs):
        scene=bound_scene(*args,**kwargs)
        identity=json.loads((ROOT/'identity_contract.json').read_text())
        assert identity['dynamic_actor_aliases']==[37]
        assert len(scene.dynamic_gaussians.get_xyz)==identity['source_real_point_count']==98452
        assert len(scene.gaussians.get_xyz)==591672 and len(scene.getTrainCameras())==49
        write(out/'actual_identity_initialization.json',dict(status='PASS_SINGLE_A37_INITIALIZATION',dynamic_aliases=[37],
            raw_track_id=identity['dynamic_raw_track_ids'][0],dynamic_rows_before_any_update=len(scene.dynamic_gaussians.get_xyz),
            static_rows_before_any_update=len(scene.gaussians.get_xyz),dynamic_random_init_point=config['dynamic_random_init_point'],
            actor_ply_paths=config['followup_scene_spec']['actor_ply_paths'],camera_count=len(scene.getTrainCameras())))
        return scene
    train.Scene=fixed_identity_scene
    seed_everything(config['seed']);torch.set_num_threads(1)
    original_publish=checkpoint.publish_checkpoint_event;original_restore=checkpoint.restore_training_state
    original_record=runtime.record_step;started=time.monotonic();last=[0];diagnostic_node=[None]
    original_adam_step=torch.optim.Adam.step

    def finite_adam_step(optimizer,*args,**kwargs):
        if last[0]%100==0:
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError('nonfinite backward gradient')
        return original_adam_step(optimizer,*args,**kwargs)

    def record(args,iteration,camera,time_shift,static,dynamic,losses):
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
        # Native strict restore and full-state equality happen before diagnostics.
        before={key:list(args[key]) for key in diagnostic_keys}
        for key in diagnostic_keys:
            args[key]=sorted(set(args[key])|{15000,20000,30000});config[key]=list(args[key])
        write(out/'effective_config.json',config)
        write(out/'diagnostic_schedule_after_exact_restore.json',dict(
            status='PASS_EXACT_RESTORE_THEN_REQUESTED_15K_20K_30K_NODES',
            before=before,after={key:list(args[key]) for key in diagnostic_keys},
            method_changes={},note='Requested15K diagnostic;20K/30K were original nodes. Native eval RNG state is captured and exactly continued.'))
        return stack,ema

    def publish(root,iteration,*args,**kwargs):
        event=original_publish(root,iteration,*args,**kwargs)
        write(out/'last_complete_node.json',dict(iteration=iteration,elapsed_seconds=time.monotonic()-started,event=event['model_path']))
        if opts.mode=='continue' and iteration==opts.stop_iteration:
            write(out/'returned.json',dict(status='PASS_COMPLETE_BOUNDARY',mode=opts.mode,iteration=iteration,
                source_iteration=start_iteration,actual_additional_iterations=iteration-start_iteration,
                accounting='new_continuation_only_prior1977_engineering_retained_without_recount',
                elapsed_seconds=time.monotonic()-started,learning_rate_horizon=30000,
                final_stop=30000,phase_stop=opts.stop_iteration,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
                quality_accepted=False,human_verdict=None))
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
