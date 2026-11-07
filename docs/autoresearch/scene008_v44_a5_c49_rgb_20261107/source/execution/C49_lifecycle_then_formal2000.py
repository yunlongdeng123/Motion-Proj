from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1');PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
assert sha(R/'source/C49_material_core.py')=='2222d3d4da7ffafe7fa83b5c69ca6b359d046055238373775252ba0f68af3275'
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
smokepid=json.loads((R/'contract/smoke600s_r1_launch.json').read_text())['pid']
while not (R/'smoke600s_r1/returned.json').exists():
 status=Path('/proc')/str(smokepid)/'stat'
 assert status.exists() and status.read_text().split()[2]!='Z','Smoke exited without passing receipt'
 time.sleep(1)
smoke=json.loads((R/'smoke600s_r1/returned.json').read_text());assert smoke['status']=='PASS_REAL600S_C49_SMOKE_SAVE_EVAL_DISPOSABLE_NEXT2' and smoke['actual_training_seconds_before_disposable']>=600
def run(command,stem):
 with (R/(stem+'.log')).open('x') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
 write(R/'contract'/(stem+'_launch.json'),{'pid':p.pid,'command':command,'unix':time.time()});print(json.dumps({'stage':stem,'pid':p.pid}),flush=True)
 code=p.wait();assert code==0,(stem,code)
run([PY,'-B',str(R/'source/C49_restore_next2_gate.py')],'smoke_freshrestore_r1')
gate=json.loads((R/'smoke_freshrestore_r1/returned.json').read_text());assert gate['status']=='PASS_C49_600S_FRESH_FULL_RESTORE_AND_NEXT2'
write(R/'contract/formal2000_release.json',{'status':'RELEASED_FRESH_CWARM0_ADAM0_AFTER_ACTUAL_600S_LIFECYCLE','smoke':smoke,'fresh_restore_gate':gate,'formal_start':0,'authorized_stop':2000,'intermediate_node':500,'further_nodes_require_root_release':[5000,10000],'quality_adoption':False,'shutdown':False})
run([PY,'-B',str(R/'source/C49_material_core.py'),'--mode','formal','--output',str(R/'formal2000_r1')],'formal2000_r1')
formal=json.loads((R/'formal2000_r1/returned.json').read_text());write(R/'queue_status.json',{'status':'COMPLETE_FRESH_FORMAL_NODE_AWAITING_ROOT_2K_RELEASE','actual_formal_step':formal['final_step'],'smoke_optimizer_updates_separate':smoke['actual_optimizer_updates']+gate['actual_optimizer_updates'],'no_automatic5K_or10K':True,'shutdown':False,'reset_card':False})
