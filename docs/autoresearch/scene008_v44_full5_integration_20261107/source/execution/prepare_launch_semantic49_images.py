from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,time
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
adapter=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(adapter)=='c177c97bea2c73e896fa925d3ecaa8ed490bb5b810e352ed77b2af23c5610c44'
parent=r/'contract/execution_contract_semantic_actor5_all49fit.json';assert sha(parent)=='d16cdc0661ec8e30da100e89e08decc6ed3928f1a662255adf482848ebc8faf0'
contract=copy.deepcopy(json.loads(parent.read_text()));contract['created_unix']=time.time()
contract['case_id']='ALL49_SEMANTIC_ACTOR5_FIXED_BUILD_SIDECAR_DIAGNOSTIC'
contract['scope']='all49_diagnostic_not_quality_adoption';contract['parent_contract_sha256']=sha(parent)
contract['evaluation_raws']=list(range(89,138));contract['baseline']['reused_cached_all49']=True
qa=r/'review/cpu_independent_candidate_semantic_a5_all49fit_4.json';q=json.loads(qa.read_text())
contract['prior_registered_gate_failures_retained']={'source':str(qa),'sha256':sha(qa),'failed_gates':q['failed_gates'],'not_waived':True}
contract['root_all49_diagnostic_release']={'granted':True,'basis':'Root viewed both complete four-frame KEEP/DELETE boards, confirmed visible roof/windshield recovery and retained the four raw94 photometric failures.',
 'quality_adoption':False,'publication':False,'new_training_or_parameter_fitting':False,'same_original_numeric_thresholds':True}
contract['all49_release']={'automatic':False,'explicit_root_diagnostic_release_granted':True,'quality_or_publication_granted':False}
dest=r/'contract/execution_contract_semantic_actor5_all49.json';assert not dest.exists();dest.write_text(json.dumps(contract,indent=2)+'\n')
s=(r/'source/render_v44_semantic_actor5_all49fit_candidate4.py').read_text()
changes={
 'contract/execution_contract_semantic_actor5_all49fit.json':'contract/execution_contract_semantic_actor5_all49.json',
 "contract['pilot_raws']":"contract['evaluation_raws']",
 "outdir=ROOT/'candidate_semantic_a5_all49fit_4'":"outdir=ROOT/'candidate_semantic_a5_49'",
 'RENDERED_FIXED_SEMANTIC_ACTOR5_SIDECAR_PILOT_PENDING_GATES_AND_ROOT_REVIEW':'RENDERED_ALL49_FIXED_SEMANTIC_ACTOR5_DIAGNOSTIC_PENDING_COMPLETE_GATES_AND_ROOT_REVIEW',
}
for before,after in changes.items():
 assert s.count(before)==1,(before,s.count(before));s=s.replace(before,after)
assert s.count('[94,105,130,137]')==3;s=s.replace('[94,105,130,137]','list(range(89,138))')
ast.parse(s);worker=r/'source/render_v44_semantic_actor5_all49.py';assert not worker.exists();worker.write_text(s)
assert sha(adapter)=='c177c97bea2c73e896fa925d3ecaa8ed490bb5b810e352ed77b2af23c5610c44'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]);assert free>=12288
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=[sys.executable,'-B',str(worker)]
with (r/'candidate_semantic_a5_49.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=r,env=env,start_new_session=True)
a={'status':'STARTED_ALL49_SEMANTIC_DIAGNOSTIC_ONLY','pid':p.pid,'command':cmd,'adapter_sha256':sha(adapter),'worker_sha256':sha(worker),
 'contract':str(dest),'contract_sha256':sha(dest),'raws':list(range(89,138)),'planned_joint_forwards':98,'optimization_updates':0,
 'quality_accepted':False,'prior_raw94_photo_failures_retained':True,'free_MiB':free,'unix':time.time(),'shutdown':False,'reset_card':False}
(r/'contract/candidate_semantic_actor5_all49_launch_receipt.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
