"""Explicit GPU gate/train entry, to be used ONLY after the user powers on AutoDL.

Never launches on import. No retries, continuation, extra training arms or shutdown.
The base engine must be copied to a NEW isolated directory and overlaid with
candidate/ first. This script verifies exact source hashes before importing it.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import sys
import subprocess
import time
import math
import traceback


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def validate_paths(engine,output,bundle,baseline_config,manifest,mode,gpu_gate=None):
    cfg=read(baseline_config);data=read(bundle);files=read(manifest)['files']
    if output.exists():raise FileExistsError('new output required; never overwrite/replay a run')
    if data['status']!='CPU_INPUT_BUNDLE_VERIFIED':raise RuntimeError('CPU real-input gate incomplete')
    if data['full49_observed']+data['prior']<=0:raise RuntimeError('road module has no real target')
    if cfg.get('seed')!=0 or cfg.get('iterations')!=30000:raise RuntimeError('wrong baseline configuration')
    if not all(cfg.get(k) for k in ['including_dynamic','paper_method','hybrid_method']):
        raise RuntimeError('actual native hybrid A configuration required')
    if 'full5_pipeline/inputs/scene0008_089_137' not in str(cfg['source_path']):
        raise RuntimeError('use actual A49 effective config, not old reference scene metadata')
    for rel,expected in files.items():
        if sha(engine/rel)!=expected['candidate_sha256']:raise RuntimeError('candidate source mismatch: '+rel)
    if mode=='train':
        if gpu_gate is None:raise RuntimeError('actual zero-update CUDA receipt required')
        gate=read(gpu_gate)
        if gate['status']!='PASSED_ACTUAL_CUDA_FORWARD_BACKWARD_ZERO_UPDATES' or gate['optimizer_updates']!=0:
            raise RuntimeError('invalid CUDA gate')
        for key,value in [('manifest_sha256',sha(manifest)),('bundle_sha256',sha(bundle)),('baseline_config_sha256',sha(baseline_config))]:
            if gate[key]!=value:raise RuntimeError('CUDA gate belongs to another candidate/input/config')
    return cfg,data


def main():
    p=argparse.ArgumentParser()
    for key in ['engine','output','bundle','baseline-config','readiness','manifest','binding-tools','gpu-lock']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=['gate','train'],required=True)
    p.add_argument('--gpu-gate',type=Path)
    a=p.parse_args()
    cfg,data=validate_paths(a.engine,a.output,a.bundle,a.baseline_config,a.manifest,a.mode,a.gpu_gate)
    from ablation_support import shared_admission, collective_storage_required
    arm_lock=shared_admission(a.output,a.manifest)
    variant=read(a.manifest)['variant']
    if a.output.parent.name!=variant:raise RuntimeError('wrong isolated arm output')
    # All six native checkpoint bundles +49 raw evals, known maxima +10%, plus8GB workspace.
    import shutil
    reserve=6*(cfg['densify_until_num_points']*640+cfg['dynamic_densify_until_num_points']*800+230e6)*1.1+8e9
    free=shutil.disk_usage(a.output.parent).free
    if free<reserve:raise RuntimeError(f'insufficient space for preserved six checkpoints: free={free}, reserve={reserve}')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    # shared_admission已核验所有CUDA进程属于当前白名单。
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    sys.path.insert(0,str(a.binding_tools));sys.path.insert(0,str(a.engine))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if torch.cuda.get_device_properties(0).total_memory<30*1024**3:raise RuntimeError('actual GPU has less than30GiB memory')
    torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    from omegaconf import OmegaConf
    import train
    if Path(train.__file__).resolve()!=(a.engine/'train.py').resolve():raise RuntimeError('wrong train import')
    os.chdir(a.engine)
    # Preserve all native optimization schedules apart from explicitly recorded v2 policy.
    cfg.update(model_path=str(a.output),start_checkpoint=None,seed=0,iterations=30000,
               checkpoint_iterations=list(range(5000,30001,5000)),
               save_iterations=list(range(5000,30001,5000)),test_iterations=list(range(5000,30001,5000)),
               v2_bundle=str(a.bundle),v2_zero_update_gate=a.mode=='gate')
    cfg['v2_policy']=data['config']
    cfg['v2_ablation_variant']=variant
    a.output.mkdir()
    write(a.output/'effective_config.json',cfg)
    write(a.output/'v2_run_contract.json',dict(mode=a.mode,logical_budget=30000,second_stage_updates=0,
        source_path=cfg['source_path'],same_seed=0,source_hashes=read(a.manifest),
        baseline_config_sha256=sha(a.baseline_config),bundle_sha256=sha(a.bundle),
        extra_optimizer_updates=0,road_objective_independent=True,
        free_bytes=free,reserved_bytes=reserve,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    import runtime_binding
    runtime_binding.bind_training_contract(train,cfg,read(a.readiness))
    # Native binding was designed for10K nodes: enforce six-node disk admission also at save.
    from utils import paper_checkpoint
    original_save=paper_checkpoint.save_components
    def save_with_admission(path,iteration,models):
        import time
        remaining=sum(step>=iteration for step in cfg['checkpoint_iterations'])
        # Conservative point-count-aware reserve; never delete older checkpoints.
        estimate=(len(models['static'].get_xyz)*640+len(models['dynamic'].get_xyz)*800+230e6)*1.1
        required=max(remaining*estimate+4e9,collective_storage_required(a.output.parent.parent))
        waiting=False
        while shutil.disk_usage(a.output).free<required:
            if not waiting:
                write(a.output/'storage_admission.json',dict(status='WAITING_STORAGE_MEMORY_PRESERVED',
                    completed_optimizer_steps=iteration,required_bytes=required,
                    free_bytes=shutil.disk_usage(a.output).free,automatic_replay=False))
                waiting=True
            time.sleep(30)
        if waiting:
            write(a.output/'storage_admission.json',dict(status='STORAGE_ADMISSION_RECOVERED',
                completed_optimizer_steps=iteration,required_bytes=required,
                free_bytes=shutil.disk_usage(a.output).free))
        return original_save(path,iteration,models)
    paper_checkpoint.save_components=save_with_admission
    from utils.general_utils import seed_everything
    from utils import v2_runtime_context
    ZeroUpdateGateComplete=v2_runtime_context.ZeroUpdateGateComplete
    started=time.monotonic()
    status=dict(status='initializing',mode=a.mode,pid=os.getpid(),parent_pid=os.getppid(),iteration=0,
        logical_budget=30000,second_stage_updates=0,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        GPU=torch.cuda.get_device_name(0),quality_accepted=False)
    write(a.output/'status.json',status)
    original_eval=train.complete_eval
    def observed_eval(*positional,**keyword):
        iteration=int(positional[1]);schedule=positional[2];losses=positional[6]
        numbers={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in losses.items()}
        if not all(math.isfinite(v) for v in numbers.values()):raise FloatingPointError('nonfinite actual training scalar')
        context=v2_runtime_context.ACTIVE_CONTEXT
        if iteration==1 or iteration%100==0 or iteration in schedule:
            counts=dict(context.optimizer_steps)
            if counts.get('static')!=iteration or counts.get('dynamic')!=iteration:
                raise RuntimeError('actual optimizer update count disagrees with logical iteration')
            status.update(status='evaluating' if iteration in schedule else 'training',iteration=iteration,
                wall_seconds=time.monotonic()-started,loss=numbers.get('loss'),psnr=numbers.get('psnr'),
                static_points=len(context.models[0].get_xyz),dynamic_points=len(context.models[1].get_xyz),
                optimizer_step_counts=counts,
                cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(a.output/'status.json',status)
        return original_eval(*positional,**keyword)
    train.complete_eval=observed_eval
    seed_everything(0)
    try:
        train.training(OmegaConf.create(cfg))
    except ZeroUpdateGateComplete:
        if a.mode!='gate':raise
        write(a.output/'gpu_gate_receipt.json',dict(status='PASSED_ACTUAL_CUDA_FORWARD_BACKWARD_ZERO_UPDATES',
            optimizer_updates=0,manifest_sha256=sha(a.manifest),bundle_sha256=sha(a.bundle),
            baseline_config_sha256=sha(a.baseline_config),GPU=torch.cuda.get_device_name(),
            actual_checks=read(a.output/'gpu_gate_details.json'),
            same_fresh_seed_required_for_training=True,quality_verified=False))
        status.update(status='zero_update_gate_passed',optimizer_step_counts=dict(v2_runtime_context.ACTIVE_CONTEXT.optimizer_steps))
        write(a.output/'status.json',status)
        return
    except BaseException as ex:
        context=v2_runtime_context.ACTIVE_CONTEXT
        status.update(status='failed_preserved_no_restart',error=repr(ex),traceback=traceback.format_exc(),
            optimizer_step_counts=dict(context.optimizer_steps) if context is not None else {},
            wall_seconds=time.monotonic()-started,automatic_restart=False)
        write(a.output/'status.json',status)
        raise
    if a.mode=='gate':raise RuntimeError('zero-update gate did not stop before Adam')
    for step in cfg['checkpoint_iterations']:
        event=read(a.output/'checkpoint_events'/f'iteration_{step:06d}.json')
        if event['status']!='complete' or len(event['checkpoints'])!=4 or len(event['frames'])!=49:
            raise RuntimeError('final six-node checkpoint/evaluation contract incomplete')
    status.update(status='complete',iteration=30000,wall_seconds=time.monotonic()-started,
        optimizer_step_counts=dict(v2_runtime_context.ACTIVE_CONTEXT.optimizer_steps))
    write(a.output/'status.json',status)
    # Native complete status is necessary, not final delivery/quality acceptance.
    write(a.output/'v2_training_returned.json',dict(status='TRAINING_RETURNED_FINAL_ASSET_AUDIT_PENDING',
        quality_accepted=False,full49_deletion_evaluation_still_required=True))


if __name__=='__main__':main()
