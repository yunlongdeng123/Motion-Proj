"""Native B algorithm unchanged; replace only worker GPU admission."""
from pathlib import Path
import sys,types,os
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E/'direct_implementation'))
import priority
priority._unshared_inspect_priority=priority.inspect_priority
from shared_gpu_policy import shared_slot,shared_priority
import run_native_timing_safe as safe
runner=safe.runner
def run_shared(args):
 paths,spec,ready,reference,cfg,gate,record,space=runner.validate_for_run(args)
 support=Path(record['training_support']);sys.path.insert(0,str(support))
 source=(support/'matrix_worker.py').read_text()
 old="""    with (BASE / 'campaign/gpu_slot.lock').open('a+') as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError('GPU已有计算任务: ' + apps)"""
 new="""    with shared_slot('B') as fd:
        admission = shared_priority()
        if not admission['allowed']:
            raise RuntimeError(str(admission))"""
 assert source.count(old)==1;source=source.replace(old,new)
 source=source.replace('        torch.cuda.set_device(0)','        torch.cuda.set_device(0)\n        torch.cuda.set_per_process_memory_fraction(13*1024**3/torch.cuda.get_device_properties(0).total_memory)')
 worker=types.ModuleType('parallel_native_B_worker');worker.__file__=str(support/'matrix_worker.py')
 worker.__dict__.update(shared_slot=shared_slot,shared_priority=shared_priority)
 exec(compile(source,worker.__file__,'exec'),worker.__dict__)
 worker.BASE=runner.REMOTE_BASE;worker.ROOT=Path(paths['campaign_root']);worker.REPOSITORY=Path(record['training_source'])
 worker.check_config=lambda arm:runner.validate_for_run(args)[4]
 worker.check_prepared=lambda:runner.validate_for_run(args)
 from runtime_binding import bind_training_contract
 worker.bind_training_contract=lambda training,config:bind_training_contract(training,config,ready)
 worker.validate_completion=lambda output,config:runner.completion_contract(output,config)
 runner.record_start_attempt(worker.ROOT,spec,space)
 worker.run_arm('GN11')
runner.run=run_shared
if __name__=='__main__':runner.main()
