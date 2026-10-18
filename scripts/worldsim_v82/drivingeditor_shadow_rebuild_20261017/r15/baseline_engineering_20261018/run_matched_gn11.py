"""Finite new five-ID GN11 30K, after the four diagnostic pilots.

This does not blindly start an instance phase with undecided constraints, and
never resumes or updates R15. A failed phase is preserved rather than restarted.
"""
from pathlib import Path
import os,sys,json,subprocess,time,datetime,traceback
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
OUT=ROOT/'baseline_engineering_20261018';DATA=OUT/'full5_pipeline'
BASE=Path('/root/autodl-tmp/drivingeditor_repro')
RUN=BASE/'campaign/r15_baseline_owner5_20261018'
SID='scene0008_089_137';M=DATA/'metadata'/SID
RUNNER=BASE/'campaign/followup_scene_prep_20261013/tools/followup/run_scene_gn11.py'
def write(state,**kw):
    (OUT/'full_queue_status.json').write_text(json.dumps(dict(state=state,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kw),indent=2))
def main():
    if (OUT/'matched_GN11_started.json').exists():raise FileExistsError('no duplicate GN11 phase')
    read=lambda p:json.loads(p.read_text())
    assert read(OUT/'channel_gate.json')['status']=='PASSED'
    assert read(OUT/'prior_trace_receipt.json')['status']=='PASSED_NATIVE_ADAPTER_RGB'
    assert read(OUT/'matched5_input_receipt.json')['body_masks_exactly_match_later_480x320_labels']
    assert len(read(M/'scene_spec.json')['selected_genuine_raw_track_ids'])==5
    assert read(M/'CPU_readiness.json')['train_cameras']==49
    assert not RUN.exists(),'separate run already exists: diagnose rather than overwrite'
    (OUT/'matched_GN11_started.json').write_text(json.dumps(dict(pid=os.getpid(),new_GN11_maximum_updates=30000,new_instance_updates=0)))
    wait_start=time.monotonic()
    while True:
        q=read(OUT/'queue_status.json')
        if q['state']=='PILOTS_COMPLETE_AWAITING_LOSS_RESTORE_AND_QUALITY_REVIEW':break
        if q['state'].startswith('FAILED'):raise RuntimeError('pilot failed; full GN11 remains gated')
        if time.monotonic()-wait_start>3600:raise TimeoutError('pilot wait exceeded finite allowance')
        write('WAITING_FOR_FOUR_DIAGNOSTIC_PILOTS',pilot=q,new_GN11_updates=0)
        time.sleep(20)
    cmd=[sys.executable,'-B','-u',str(RUNNER),'--verify-paired-steps','--scene-spec',str(M/'scene_spec.json'),
        '--readiness-file',str(M/'CPU_readiness.json'),'--reference-config',str(M/'reference_GN11.json'),
        '--source-repository',str(BASE/'campaign/followup_scene_prep_20261013/frozen_repository'),
        '--matrix-tools-dir',str(BASE/'campaign/hybrid30k_20261012/tools'),'--campaign-root',str(RUN),
        '--gate-file',str(DATA/'metadata/GN11_reference_acceptance_authorization.json')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCH_HOME=str(BASE/'cache/torch'))
    env['PATH']='/root/autodl-tmp/envs/drivingeditor/bin:/usr/local/cuda-11.8/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-11.8/lib64:/root/autodl-tmp/envs/drivingeditor/lib:'+env.get('LD_LIBRARY_PATH','')
    write('MATCHED_FIVE_GN11_CPU_PREPARING',command=cmd+['--prepare'])
    with (OUT/'matched_GN11_prepare.log').open('x') as f:subprocess.run(cmd+['--prepare'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    write('MATCHED_FIVE_GN11_30K_RUNNING',command=cmd+['--run'],run_root=str(RUN),new_GN11_maximum_updates=30000,new_instance_updates=0)
    with (OUT/'matched_GN11_train.log').open('x') as f:subprocess.run(cmd+['--run'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    model=RUN/'runs'/f'{SID}_hybrid_30000_seed0_GN11'
    status=read(model/'status.json');assert status['status']=='complete' and status['iteration']==30000
    write('MATCHED_FIVE_GN11_30K_COMPLETE_INSTANCE_PHASE_GATED_FOR_REVIEW',run_root=str(RUN),new_GN11_updates=30000,new_instance_updates=0,
        next='verify current masks, native opacity/geometry constraints, Adam/RNG restore, and static/actor/delete quality before the fresh five-ID 30K instance phase')
if __name__=='__main__':
    try:main()
    except BaseException:
        write('FAILED_MATCHED_PHASE_PRESERVED_NO_RESTART',error=traceback.format_exc());raise
