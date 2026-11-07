"""Bounded gate: existing smoke -> fresh replay/parity -> fresh400. No49 release."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1');PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def target_guard(phase):
 mf=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/target_manifest.json')
 assert sha(mf)=='d7cca4167e2bbb68347f63babae356320a1b0ddd60e580bf746e966f2ac8acbe'
 m=json.loads(mf.read_text());rows={x['raw']:x for x in m['rows']};cfg=json.loads((R/'F_smoke_r1/executed_config.json').read_text())
 prior=json.loads((R/'review/E4_existing_C_target_readiness_cpu.json').read_text());old={x['raw']:x for x in prior['rows']};files=[]
 for raw in [94,105,128,137]:
  actual=cfg['targets'][str(raw)];p=Path(actual['path']);expected=rows[raw]
  assert str(p).endswith(expected['file']) and sha(p)==actual['sha256']==expected['sha256']==old[raw]['target_sha256']
  files.append({'raw':raw,'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p),'manifest_file':expected['file'],'original_role':expected['role'],'current_role':'BUILD'})
 record={'status':'PASS_SOURCE_C_MANIFEST_AND_FOUR_ACTUAL_TARGET_FILES_CLOSED','phase':phase,'manifest':str(mf),'manifest_sha256':sha(mf),'files':files,
  'evidence_note':'Smoke recorded actual4SHA before optimization; this external manifest closure is performed before fresh formal and repeated after formal.'}
 write(R/'contract'/('F_target_guard_'+phase+'.json'),record);return record
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
assert sha(R/'source/F_colour_bake_core.py')=='7181bf20a68eb7b8ef8d3961cf88a777f7b6bba9be6e2b2c6f2a210f33e4518b'
zero=json.loads((R/'F_zero_identity4/receipt.json').read_text());assert zero['status']=='PASS_PUBLIC_BOUNDED_F_ZERO_DELTA_ALL4_KD_EXACT_BASELINE'
launch=json.loads((R/'contract/F_smoke_r1_launch.json').read_text())
while not (R/'F_smoke_r1/returned.json').exists():
 try:os.kill(launch['pid'],0)
 except ProcessLookupError:raise RuntimeError('Smoke exited without a passing receipt; formal not released')
 time.sleep(1)
smoke=json.loads((R/'F_smoke_r1/returned.json').read_text());assert smoke['status']=='PASS_SMOKE_SAVE_PUBLIC_EVAL_AND_DISPOSABLE_NEXT2'
assert smoke['all_original_model_binding_and_files_unchanged']
if smoke['formal_ETA_seconds_from_actual_smoke']>=3600:assert smoke['actual_training_smoke_seconds']>=540
target_guard('before_gate')
with (R/'F_restore_parity_r1.log').open('x') as log:
 p=subprocess.Popen([PY,'-B',str(R/'source/F_restore_parity_gate.py')],stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True)
write(R/'contract/F_restore_parity_r1_launch.json',{'pid':p.pid,'log':str(R/'F_restore_parity_r1.log'),'native_parity_forwards_planned':2,'restore_next2_updates_planned':2,'unix':time.time()});print(json.dumps({'phase':'fresh_restore_and_parity','pid':p.pid}),flush=True)
assert p.wait()==0,'Fresh restore or train-public parity failed; formal not released'
gate=json.loads((R/'F_restore_parity_r1/returned.json').read_text());assert gate['status']=='PASS_FRESH_FULL_RESTORE_NEXT2_AND_TRAIN_PUBLIC_PARITY'
before=target_guard('before_formal');write(R/'contract/F_formal_release.json',{'status':'RELEASED_FRESH_ZERO_400_AFTER_REAL_LIFECYCLE','smoke':smoke,'gate':gate,'target_manifest_guard':before,'formal_fresh_zero':True,'all49_auto_release':False})
cmd=[PY,'-B',str(R/'source/F_colour_bake_core.py'),'--mode','formal','--output',str(R/'F_formal400_r1')]
with (R/'F_formal400_r1.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True)
write(R/'contract/F_formal400_r1_launch.json',{'pid':p.pid,'command':cmd,'source_sha256':sha(R/'source/F_colour_bake_core.py'),'fresh_zero':True,'paired_steps':400,'unix':time.time(),'all49_auto_release':False});print(json.dumps({'phase':'fresh_formal400','pid':p.pid}),flush=True)
code=p.wait();after=target_guard('after_formal');assert before['files']==after['files']
result={'status':'COMPLETE_FIXED_LIFECYCLE_AND_FORMAL400' if code==0 else 'FORMAL_FAILED_PRESERVED','formal_exit_code':code,'target_inputs_unchanged':True,'no_automatic_all49':True,'shutdown':False,'reset_card':False}
write(R/'F_training_queue_returned.json',result);print(json.dumps(result),flush=True);raise SystemExit(code)
