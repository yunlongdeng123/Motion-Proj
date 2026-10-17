from pathlib import Path
import ast,json,base64,tempfile,datetime
W=Path(__file__).resolve().parent
source=(W/'full_controller.py').read_text(encoding='utf8')
old="""    if run.exists():raise FileExistsError('Do not start duplicate full experiment; inspect actual controller/latest checkpoint')
    state('INDEPENDENT_DINOV3_RANSAC_NEW_POINTSETS_ZERO_SGD')
    command(D/'dinov3_registration_r14.py',gpu=True,interpreter=str(R/'ownership_controls_r12_sam3/runtime_sam3_dinov3/bin/python'))
    step=0
"""
new="""    import hashlib
    receipt=json.loads((D/'controller_recovery_r1_preflight.json').read_text())
    assert receipt['global_step']==3000 and receipt['checkpoint_optimizer_rng_input_preserved']
    assert hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest()==receipt['checkpoint_sha256']
    assert hashlib.sha256((run/'pre_owner_003000.pt').read_bytes()).hexdigest()==receipt['checkpoint_sha256']
    assert json.loads((run/'progress.json').read_text())['global_step']==3000
    step=3000;G=D/'gates/003000'
    assert G.is_dir() and not list(G.iterdir()),'Recovery only adopts the verified empty gate from the failed status call'
    state('ACTUAL_OWNER_REVIEW',step=step,phase='recover status-call failure; zero repeated SGD; fresh full-scene evidence')
    audit(step,run,G/'before');command(D/'split_003000.py');audit(step,G/'split',G/'after_split');command(D/'owner_003000.py')
    state('DIAGNOSTIC_PREVIEW_NOT_STOP_GATE',step=step);command(D/'preview.py',gpu=True)
"""
assert source.count(old)==1
source=source.replace(old,new).replace('for end in boundaries:','for end in [v for v in boundaries if v>step]:')
source='"""Explicit once-only recovery at saved3000: no input/DINO/SGD replay."""\n'+source
ast.parse(source)
(W/'full_controller_resume_r1.py').write_text(source,encoding='utf8')
# Execute the actual status helper with the formerly conflicting detail field.
tree=ast.parse(source);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='state')
module=ast.Module(body=[fn],type_ignores=[])
with tempfile.TemporaryDirectory(dir=W,prefix='controller_status_check_') as tmp:
    ns=dict(D=Path(tmp),datetime=datetime,json=json);exec(compile(module,'status_helper','exec'),ns)
    ns['state']('ACTUAL_OWNER_REVIEW',step=3000,phase='fresh evidence')
    check=json.loads((Path(tmp)/'pipeline_state.json').read_text());assert check['status']=='ACTUAL_OWNER_REVIEW' and check['phase']=='fresh evidence'
ends=[v for v in list(range(1000,13001,1000))+list(range(14000,20001,1000))+[21000,22000,22400] if v>3000]
assert ends[0]==4000 and ends[-1]==22400 and sum(b-a for a,b in zip([3000]+ends[:-1],ends))==19400
result=dict(status='PASSED',status_and_phase_coexist=True,resume_step=3000,remaining_updates=19400,DINO_and_input_replay=False,train_snapshot_changed=False)
(W/'controller_recovery_r1_CPU_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
files=['full_controller.py','full_controller_resume_r1.py','gate_common_003000.py','gate_common_009000.py']
payload={name:base64.b64encode((W/name).read_bytes()).decode() for name in files}
remote='''from pathlib import Path
import base64,subprocess,os,json,time,hashlib,shutil,sys
import torch
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'ownership_rebuild_r14'
failure=json.loads((D/'pipeline_state.json').read_text());assert failure['status']=='ENGINEERING_FAILURE_PRESERVE_AND_DIAGNOSE_NO_SHUTDOWN' and 'multiple values' in failure['error']
assert not (D/'full_controller_recovery_launch.json').exists(),'Never launch recovery twice'
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:c=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
    except (OSError,UnicodeError):continue
    assert not (str(D) in c and any(t in c for t in ['full_controller.py','full_controller_resume_r1.py','run.py train','ownership_audit_r14.py'])),('Owned worker still live',p.name,c)
shutil.copy2(D/'pipeline_state.json',D/'engineering_failure_003000_state.json')
shutil.copy2(D/'full_controller_launch.json',D/'full_controller_initial_launch.json')
for name,value in PAYLOAD.items():
    p=D/name
    if p.exists():shutil.copy2(p,D/(name+'.pre_recovery_r1.bak'))
    p.write_bytes(base64.b64decode(value))
sys.path.insert(0,str(D));from gate_common_003000 import load,body_bank,road_bank,build_only
saved,cfg,data,model,opt=load('main/latest.pt')
assert saved['global_step']==3000 and saved['stage']=='geometry'
assert not list((D/'gates/003000').iterdir())
counts=[]
for a in [5,6,12,18,25,26,37]:
    body,road=body_bank(a),road_bank(a);build_only(body,data);build_only(road,data);counts.append(dict(actor=a,body=len(body['xyz']),road=len(road['xyz'])))
digest=hashlib.sha256((D/'main/latest.pt').read_bytes()).hexdigest()
assert digest==hashlib.sha256((D/'main/pre_owner_003000.pt').read_bytes()).hexdigest()
receipt=dict(global_step=3000,checkpoint_sha256=digest,checkpoint_optimizer_rng_input_preserved=True,Gaussians=len(model.field.xyz),BUILD_banks_checked=counts,failure=failure,training_updates_repeated=0,immutable_training_source_changed=False,helper_import='implementation_r14_final')
(D/'controller_recovery_r1_preflight.json').write_text(json.dumps(receipt,indent=2));del model,opt,saved,data
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (D/'controller_recovery_r1.log').open('xb') as log:
    process=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python',str(D/'full_controller_resume_r1.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=process.pid,at=time.time(),script='full_controller_resume_r1.py',log='controller_recovery_r1.log',resume_at=3000,total=22400,remaining_updates=19400,zero_SGD_replay=True,initial_controller_pid=2271)
(D/'full_controller_recovery_launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(dict(preflight=receipt,launch=launch)))
'''.replace('PAYLOAD',repr(payload))
(W/'deploy_controller_recovery_r1_remote.py').write_text(remote,encoding='utf8')
print(json.dumps(result))
