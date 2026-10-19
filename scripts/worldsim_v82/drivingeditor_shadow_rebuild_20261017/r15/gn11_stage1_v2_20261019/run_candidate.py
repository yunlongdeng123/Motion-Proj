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
    import fcntl
    lock=a.gpu_lock.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # All six native checkpoint bundles +49 raw evals, known maxima +10%, plus8GB workspace.
    import shutil
    reserve=6*(2.5e6*640+1.1e6*800+230e6)*1.1+8e9
    free=shutil.disk_usage(a.output.parent).free
    if free<reserve:raise RuntimeError(f'insufficient space for preserved six checkpoints: free={free}, reserve={reserve}')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    sys.path.insert(0,str(a.binding_tools));sys.path.insert(0,str(a.engine))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    from omegaconf import OmegaConf
    import train
    if Path(train.__file__).resolve()!=(a.engine/'train.py').resolve():raise RuntimeError('wrong train import')
    # Preserve all native optimization schedules apart from explicitly recorded v2 policy.
    cfg.update(model_path=str(a.output),start_checkpoint=None,seed=0,iterations=30000,
               checkpoint_iterations=list(range(5000,30001,5000)),
               save_iterations=list(range(5000,30001,5000)),test_iterations=list(range(5000,30001,5000)),
               v2_bundle=str(a.bundle),v2_zero_update_gate=a.mode=='gate')
    cfg['v2_policy']=data['config']
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
        required=remaining*estimate+4e9
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
    from utils.v2_runtime_context import ZeroUpdateGateComplete
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
        return
    if a.mode=='gate':raise RuntimeError('zero-update gate did not stop before Adam')
    # Native complete status is necessary, not final delivery/quality acceptance.
    write(a.output/'v2_training_returned.json',dict(status='TRAINING_RETURNED_FINAL_ASSET_AUDIT_PENDING',
        quality_accepted=False,full49_deletion_evaluation_still_required=True))


if __name__=='__main__':main()
