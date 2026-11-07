"""Correct the diagnostic slice for appended group7; preserve first failure."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=ROOT/'source/render_road_pilot4.py';s=original.read_text()
assert sha(original)=='19c36757a5eee371b454208f3d00b936796cea09720b8f858fed4d4fd69cf3f3'
assert s.count('float(np.abs(groups[5:]).max())==0.')==1
s=s.replace('float(np.abs(groups[5:]).max())==0.','float(np.abs(groups[5:7]).max())==0.')
assert s.count("outdir=ROOT/(variant+'4')")==1;s=s.replace("outdir=ROOT/(variant+'4')","outdir=ROOT/(variant+'4_retry1')")
s=s.replace("    output=rt.render(raw,scene)","    output=rt.render(raw,scene)\n    write(outdir/'kernel_execution_progress.json',dict(completed_native_forwards=2*len(rows)+(1 if state=='keep' else 2),raw=raw,state=state,optimization_updates=0))")
ast.parse(s);worker=ROOT/'source/render_road_pilot4_retry1.py';assert not worker.exists();worker.write_text(s)
failure={'status':'PRESERVED_FIRST_FORWARD_ENGINEERING_ASSERTION_FAILURE','variant':'AA','raw':94,'state':'keep','actual_native_forwards':1,'optimization_updates':0,
 'reason':'Inherited seven-group unused slice groups[5:] incorrectly included appended newroad group7; corrected to groups[5:7] only.',
 'old_worker_sha256':sha(original),'retry_worker_sha256':sha(worker),'adapter_asset_threshold_method_changed':False,
 'failed_log':str(ROOT/'AA4.log'),'C_launched_in_failed_attempt':False,'preserved_original_contract_sha256':sha(ROOT/'contract/pilot4_execution_contract.json')}
(ROOT/'contract/first_forward_engineering_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
statuses=[]
for variant in ['AA','C']:
 cmd=[sys.executable,'-B',str(worker),variant]
 with (ROOT/(variant+'4_retry1.log')).open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
 entry={'variant':variant,'pid':p.pid,'command':cmd,'worker_sha256':sha(worker),'unix':time.time(),'optimization_updates':0}
 (ROOT/'contract'/('launch_'+variant+'4_retry1.json')).write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry),flush=True)
 code=p.wait();statuses.append(dict(entry,exit_code=code));(ROOT/'pilot_pair_retry1_status.json').write_text(json.dumps({'completed':statuses,'all49_auto_release':False,'optimization_updates':0},indent=2)+'\n')
 if code!=0:raise SystemExit(code)
print(json.dumps({'status':'COMPLETE_AA_C_OFFICIAL4_RETRY1','optimization_updates':0,'preserved_failed_forward_count':1}),flush=True)
