"""最多两路训练（包括原主实验），三个指定消融各一次fresh30K。"""
from pathlib import Path
import argparse,datetime,fcntl,json,os,shutil,subprocess,time,traceback
from ablation_support import read,process,collective_storage_required

PYTHON='/root/autodl-tmp/envs/drivingeditor/bin/python'

def write(path,data):
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(tmp,path)

def main(C):
    group=C/'ablations';guard=(group/'group.lock').open('a+')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (group/'registry.json').exists():raise FileExistsError('preserve prior controller state; no replay')
    original=read(C/'training_launch.json');mainpid=original['pid'];mainproc=process(mainpid)
    mainstatus=read(C/'run30k/status.json')
    if not mainproc or mainproc['state']=='D' or mainstatus['status'] not in ['training','evaluating']:
        raise RuntimeError('current main training is not healthy for shared admission')
    if (Path('/sys/block/md0/md/array_state').read_text().strip() not in ['active','clean'] or
        Path('/sys/block/md0/md/degraded').read_text().strip()!='0' or
        Path('/sys/block/md0/md/sync_action').read_text().strip()!='idle'):
        raise RuntimeError('storage not healthy for new shared work')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if {int(x) for x in apps.splitlines() if x.strip()}!={mainpid}:raise RuntimeError('unexpected initial GPU process')
    total=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())*1024**2
    if 30*1024**3>total or mainstatus['cuda_peak_allocated_bytes']>8*1024**3:
        raise RuntimeError('shared allocation allowances do not fit observed main peak')
    registry=dict(status='PREPARING_AUTHORIZED_ROAD_ABLATIONS',controller_pid=os.getpid(),
        controller_starttime_ticks=process(os.getpid())['starttime_ticks'],main_campaign=str(C),main_pid=mainpid,
        main_starttime_ticks=mainproc['starttime_ticks'],baseline_config=read(C/'baseline_config.json'),
        children={},arm_queue=['real_only','prior_only','ownership_only'],max_concurrent_training=2,
        automatic_restart=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(group/'registry.json',registry)
    required=collective_storage_required(group);free=shutil.disk_usage(C).free
    if free<required:raise RuntimeError(f'collective preserved-assets reserve does not fit: {free} < {required}')
    registry['storage_admission']=dict(free_bytes=free,required_bytes=required)
    write(group/'registry.json',registry)
    source=group/'source_bundle';handles=[];workers={}
    global_lock=(C.parent/'gpu_slot.lock').open('a+');global_owned=False
    def maintain_global():
        nonlocal global_owned
        main_live=process(mainpid)
        if not global_owned and (not main_live or main_live['state']=='Z'):
            try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);global_owned=True
            except BlockingIOError:pass
    def launch(arm,mode):
        root=group/arm
        command=[PYTHON,str(source/'run_ablation.py'),'--engine',str(root/'engine'),
            '--output',str(root/('gate_001' if mode=='gate' else 'run30k')),
            '--bundle',str(C/'input_bundle_002/bundle.json'),'--baseline-config',str(C/'baseline_config.json'),
            '--readiness',str(C.parent/'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/metadata/scene0008_089_137/CPU_readiness.json'),
            '--manifest',str(root/'source_bundle/candidate_manifest.json'),
            '--binding-tools',str(C.parent/'followup_scene_prep_20261013/tools/followup'),
            '--gpu-lock',str(C.parent/'gpu_slot.lock'),'--mode',mode]
        if mode=='train':command+=['--gpu-gate',str(root/'gate_001/gpu_gate_receipt.json')]
        log=(root/(mode+'.log')).open('xb');handles.append(log)
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
        registry['children'][str(child.pid)]=dict(arm=arm,mode=mode,command=command,returncode=None)
        write(group/'registry.json',registry)
        write(root/(mode+'_launch.json'),dict(pid=child.pid,command=command,mode=mode,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_updates=0 if mode=='gate' else 30000,automatic_restart=False))
        return child
    try:
        pending=list(registry['arm_queue']);failed=False
        # gate和训练都占一个并发槽；正在运行的主实验也占槽，不暂停它。
        while True:
            maintain_global()
            codes={arm:p.poll() for arm,p in workers.items()}
            for arm,p in workers.items():registry['children'][str(p.pid)]['returncode']=codes[arm]
            main_live=process(mainpid)
            active=sum(code is None for code in codes.values())+(1 if main_live and main_live['state']!='Z' else 0)
            if any(code not in [None,0] for code in codes.values()):failed=True
            if pending and active<2 and not failed:
                arm=pending.pop(0)
                child=launch(arm,'gate')
                while child.poll() is None:maintain_global();time.sleep(.5)
                registry['children'][str(child.pid)]['returncode']=child.returncode
                write(group/'registry.json',registry)
                if child.returncode:
                    failed=True;registry['failed_gate_arm']=arm
                else:
                    receipt=read(group/arm/'gate_001/gpu_gate_receipt.json')
                    if receipt['optimizer_updates']!=0:raise RuntimeError('gate performed updates')
                    workers[arm]=launch(arm,'train')
                codes={name:p.poll() for name,p in workers.items()}
            registry['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            registry['pending_arms']=pending
            registry['status']='AUTHORIZED_ABLATIONS_RUNNING_MAX_TWO_TRAINING_INCLUDING_MAIN'
            if failed:registry['status']='FAILED_ARM_PRESERVED_PENDING_NOT_LAUNCHED_OTHER_AUTHORIZED_WORK_CONTINUES'
            write(group/'registry.json',registry)
            if all(code is not None for code in codes.values()) and (not pending or failed):break
            time.sleep(5)
        if failed or any(code!=0 for code in codes.values()):raise RuntimeError('one or more ablation workers/gates failed; no restart')
        for arm in workers:
            s=read(group/arm/'run30k/status.json')
            if s['status']!='complete' or any(s['optimizer_step_counts'][m]!=30000 for m in ['static','dynamic']):
                raise RuntimeError('ablation worker returned without complete30K')
        registry['status']='THREE_ABLATIONS_TRAINING_COMPLETE_AWAITING_ZERO_UPDATE49_AND_REPORT'
        write(group/'registry.json',registry)
    except BaseException as error:
        registry.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RESTART',error=repr(error),traceback=traceback.format_exc())
        write(group/'registry.json',registry)
        raise
    finally:
        for handle in handles:handle.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();main(a.campaign)
