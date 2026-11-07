from pathlib import Path
import json,sys,os,subprocess,time,hashlib,shutil
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=BASE/'scene008_a37_native_longrun30k_20261107_r1'
OLD=BASE/'scene008_a5_old_single30k_20261106_r1'
ENGINE=BASE/'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
SOURCE=OLD/'smoke_retry1'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def entry(p):return dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert ROOT.is_dir() and not (ROOT/'longrun_contract.json').exists()
for f in ['config.json','identity_contract.json']:shutil.copy2(OLD/f,ROOT/f)
smoke=json.loads((OLD/'smoke_summary.json').read_text())
assert smoke['status']=='PASS_REAL_GPU_SMOKE_SAVE_EVAL_FULL_RESTORE_NEXT_ENTRY' and smoke['smoke_train_wall_seconds']>=540
assert json.loads((OLD/'smoke_retry1_resume2step/restore_audit.json').read_text())['differences']==[]
core=['train.py','gaussian_renderer/__init__.py','scene/gaussian_model.py','scene/dynamic_gaussian_model.py','utils/paper_checkpoint.py','utils/hybrid30k_runtime.py']
core=[f for f in core if (ENGINE/f).exists()]
expected={'gaussian_renderer/__init__.py':'35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced',
    'scene/gaussian_model.py':'e5da10984f8836da1ac51c975823c3ca89bbf8c56f45ec4d4dcb071c06aafa72'}
hashes={f:entry(ENGINE/f)['sha256'] for f in core}
for f,h in expected.items():assert hashes[f]==h
protected=[SOURCE/f'{prefix}11977.pth' for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']]+[ENGINE/f for f in core]
write(ROOT/'longrun_contract.json',dict(status='AUTHORIZED_NATIVE_CONTINUATION',source_checkpoint=str(SOURCE/'chkpnt11977.pth'),
    source_iteration=11977,final_iteration=30000,new_updates=18023,node_iterations=[15000,20000,30000],
    native_core_sha256=hashes,protected_source_files={str(p):entry(p) for p in protected},
    lifecycle_smoke_reused=smoke,prior_smoke1977_reused_no_recount=True,prior_engineering_ledger6599_unchanged=True,
    lr_horizon=30000,seed=0,initial_static_rows=591672,initial_dynamic_rows=98452,
    method_changes={},diagnostic_changes=['additional15000 complete native checkpoint/eval'],
    resource_budget=dict(free_disk_required_GiB=24,expected_three_complete_nodes_GiB=15,peak_reserved_previous_bytes=smoke['max_reserved_bytes']),
    shutdown=False,reset_card=False))
env=os.environ.copy()
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]='1'
with (ROOT/'queue.log').open('x') as log:
    proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'source/run_single_longrun.py')],stdout=log,stderr=subprocess.STDOUT,
        cwd=ROOT,env=env,start_new_session=True)
(ROOT/'queue.pid').write_text(str(proc.pid)+'\n')
write(ROOT/'launch_receipt.json',dict(controller_pid=proc.pid,unix=time.time(),interpreter=sys.executable,shutdown=False,reset_card=False))
print(json.dumps(dict(controller_pid=proc.pid,root=str(ROOT),log=str(ROOT/'queue.log'))))
