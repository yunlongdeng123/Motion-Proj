"""Frozen V41-B native scene training; engineering stops only after full saves.

Smoke uses fresh300s + strict-restored300s at complete checkpoint boundaries.
Formal always starts fresh,30K,5K cadence; no road/editor hooks are installed.
"""
import argparse,json,math,os,signal,sys,time
from pathlib import Path
from resource_control import require_start,stop_requested
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
SOURCE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_configs/scene0008_089_137_GN11.json')

class SavedBoundary(Exception):pass

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def config_for(scene,output,smoke,resume):
    config=json.loads(BASE.read_text()); sensor=json.loads((scene/'sensor_preparation.json').read_text())
    frames=sensor['frames'];names=[x['frame_id'] for x in frames];raws=[x['raw'] for x in frames];times=[i*.02-.48 for i in range(49)]
    nodes=list(range(500,30001,500)) if smoke else list(range(5000,30001,5000))
    changes=dict(source_path=str(scene),model_path=str(output),start_checkpoint=resume,start_frame=raws[0],end_frame=raws[-1],test_iterations=nodes,save_iterations=nodes,checkpoint_iterations=nodes,
        followup_scene_spec=dict(scene_id=scene.name,source_path=str(scene),frame_ids=names,camera_directory='image_0',frame_ids_with_camera=['image_0/'+x for x in names],raw_frame_range=[raws[0],raws[-1]],model_timestamps=times,original_wh=[1920,1280],render_wh=[480,320],anchor_frame=names[10],anchor_index=10,anchor_timestamp=times[10],actor_ply_paths=[str(scene/'points3d_dynamic_0.ply')]),followup_static_anchor_frame=names[10])
    config.update(changes)
    if resume:
        import torch
        step=int(Path(resume).stem.replace('chkpnt',''))
        saved=torch.load(Path(resume).parent/f'training_state{step}.pth',map_location='cpu')['config']
        assert saved['source_path']==str(scene) and saved['iterations']==30000
        config=dict(saved,model_path=str(output),start_checkpoint=resume)
        changes=dict(resumed_complete_saved_config=True,model_path=str(output),start_checkpoint=resume)
    return config,changes

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke-seconds',type=float,default=0);p.add_argument('--resume');a=p.parse_args()
    require_start();assert json.loads((ROOT/'research_policy.json').read_text())['allow_native_scene_training']
    assert not a.output.exists(),'No-clobber: use new attempt directories'
    assert json.loads((a.scene/'native_preparation/selection_reviewed.json').read_text())['status']=='agent_visually_reviewed'
    registry=json.loads((ROOT/'evaluation_admission_v2/registry.json').read_text())
    admitted=next(x for x in registry['scenes'] if x['context']==a.scene.name)
    assert admitted['raw_input_status']=='raw_checks_passed' and not admitted['structural_failures'] and not admitted['review_flags']
    assert len(list((a.scene/'sky/sky_0').glob('*.png')))==49
    a.output.mkdir(parents=True)
    config,changes=config_for(a.scene,a.output,bool(a.smoke_seconds),a.resume)
    write(a.output/'effective_config.json',config);write(a.output/'config_delta.json',dict(source=str(BASE),changes=changes,method_loss_sampling_lr_densification_unchanged=True,smoke_extra_not_formal=bool(a.smoke_seconds)))
    sys.path.insert(0,str(SOURCE));sys.path.insert(1,'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/tools')
    import torch,numpy as np,random
    from omegaconf import OmegaConf
    import train
    from run_v4_arm import bind_scene_contract
    import utils.paper_checkpoint as checkpoint
    import utils.hybrid30k_runtime as runtime
    from utils.general_utils import seed_everything
    assert Path(train.__file__).resolve()==SOURCE/'train.py'
    assert all(x not in (SOURCE/'train.py').read_text() for x in ['V41_BEGIN','V4_BEGIN'])
    bind_scene_contract(train,config);os.chdir(SOURCE);torch.cuda.set_device(0);seed_everything(config['seed'])
    started=time.monotonic();active_started=[None];stop=[False];last=[0];first=[None];finite_steps=[0]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    original_record=runtime.record_step
    def record(*args,**kwargs):
        iteration=int(args[1]);values=args[-1]
        if not all(math.isfinite(float(v)) for v in values.values() if isinstance(v,(int,float))):raise FloatingPointError(values)
        if active_started[0] is None:active_started[0]=time.monotonic();first[0]=iteration
        finite_steps[0]+=1;last[0]=iteration
        original_record(*args,**kwargs)
        # Resource stops get an immediate next-step full save, not a 5K wait.
        if stop[0] or stop_requested():
            for key in ['checkpoint_iterations','save_iterations','test_iterations']:
                if iteration not in getattr(args[0],key):getattr(args[0],key).append(iteration)
    runtime.record_step=record
    original_restore=checkpoint.restore_training_state
    def equal(x,y):
        if torch.is_tensor(x):return torch.is_tensor(y) and x.shape==y.shape and torch.equal(x.detach().cpu(),y.detach().cpu())
        if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
        if isinstance(x,dict):return isinstance(y,dict) and set(x)==set(y) and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return type(x)==type(y) and len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
        return x==y
    def restore(path,iteration,scene,models,args,**kw):
        result=original_restore(path,iteration,scene,models,args,**kw)
        state=torch.load(path,map_location='cpu');root=Path(path).parent
        checks={}
        for name,prefix in [('static','chkpnt'),('dynamic','dynamic_chkpnt'),('environment','env_light_chkpnt')]:
            saved,step=torch.load(root/f'{prefix}{iteration}.pth',map_location='cpu');assert step==iteration
            checks[name+'_parameters_and_optimizer']=equal(models[name].capture(),saved)
        checks.update(python_rng=equal(random.getstate(),state['python_rng']),numpy_rng=equal(np.random.get_state(),state['numpy_rng']),torch_rng=equal(torch.get_rng_state(),state['torch_rng']),cuda_rng=equal(torch.cuda.get_rng_state_all(),state['cuda_rng']),sampler=equal(scene.hybrid_sampler.capture(),state['hybrid_sampler']))
        write(a.output/'restore_verification.json',dict(iteration=iteration,checks=checks,all_passed=all(checks.values()),cuda_trajectory_bitwise_claimed=False));assert all(checks.values()),checks
        return result
    checkpoint.restore_training_state=restore
    original_publish=checkpoint.publish_checkpoint_event
    def publish(*args,**kwargs):
        event=original_publish(*args,**kwargs);iteration=int(args[1]);active=time.monotonic()-active_started[0]
        write(a.output/'latest_complete_node.json',dict(iteration=iteration,training_seconds_since_first_step=active,event=str(a.output/'checkpoint_events'/f'iteration_{iteration:06d}.json')))
        if stop[0] or stop_requested() or (a.smoke_seconds and active>=a.smoke_seconds):raise SavedBoundary()
        return event
    checkpoint.publish_checkpoint_event=publish
    reason='native_returned'
    try:train.training(OmegaConf.create(config))
    except SavedBoundary:reason='resource_saved' if stop[0] or stop_requested() else 'smoke_saved'
    finally:
        write(a.output/'cuda_memory_peak.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),scope='whole native training invocation including initialization,evaluation,save',nvml_is_separate=True))
    write(a.output/'returned.json',dict(reason=reason,iteration=last[0],first_iteration=first[0],extra_smoke=bool(a.smoke_seconds),observed_finite_step_calls=finite_steps[0],wall_seconds=time.monotonic()-started,training_seconds_since_first_step=time.monotonic()-active_started[0],native_source=str(SOURCE),quality_accepted=False,human_verdict=None))
    if not a.smoke_seconds and reason=='native_returned':
        assert last[0]==30000
        expected=set(range(5000,30001,5000))
        actual={int(p.stem.split('_')[-1]) for p in (a.output/'checkpoint_events').glob('*.json')}
        if not a.resume:assert expected.issubset(actual)

if __name__=='__main__':main()
