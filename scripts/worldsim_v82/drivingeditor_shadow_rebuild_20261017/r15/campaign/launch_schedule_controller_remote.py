from pathlib import Path
import subprocess,os,json,time
D=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/backend_restore_r15')
assert not (D/'full_controller_recovery_launch.json').exists()
proof=json.loads((D/'schedule_recovery_verified.json').read_text());assert proof['actual_main_updates']==2 and proof['Adam_moments_bitwise_equal']
progress=json.loads((D/'main/progress.json').read_text())
if progress['global_step']!=2:
    # The bounded1-update resume saves checkpoint2 before the next50-step log.
    # Record exact checkpoint count without inventing a missing RGB/loss sample.
    assert progress['global_step']==1
    (D/'main/progress_before_exact_count_receipt.json').write_text(json.dumps(progress,indent=2))
    progress.update(global_step=2,stage_step=2,loss=None,terms={},frame_id=None,elapsed_seconds=0.,count_source='actual checkpoint2 and nonempty Adam/model/render restore receipt; no photometric log fabricated')
    (D/'main/progress.json').write_text(json.dumps(progress,indent=2))
assert progress['global_step']==2
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';log=(D/'controller.log').open('ab');p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python',str(D/'controller.py')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
v=dict(pid=p.pid,at=time.time(),script='controller.py',start_step=2,total=45000,remaining=44998,implementation='implementation_r15_train_v3',log='controller.log',schedule_repair=True);(D/'full_controller_recovery_launch.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
