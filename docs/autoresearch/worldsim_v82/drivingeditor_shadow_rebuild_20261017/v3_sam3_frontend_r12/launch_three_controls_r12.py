from pathlib import Path
import subprocess,os,json,datetime
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=r/'ownership_controls_r12_sam3'
if (root/'controls_launch.json').exists():raise FileExistsError('no duplicate controller')
if not (root/'initialization_summary.json').exists() or not (root/'initialization_QUERY_gate.json').exists():raise RuntimeError('raw rebuild and initial body/road measurement report required')
assert (root/'shared_SAM3_root_review.json').exists()
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
with (root/'controls.log').open('xb') as log:p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python',str(r/'three_control_controller_r12.py')],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=p.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FINITE_3x3000_CONTROLLER',max_updates_total=9000,GPU_slot_per_arm=True);(root/'controls_launch.json').write_text(json.dumps(v));print(json.dumps(v))
