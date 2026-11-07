from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
(ROOT/'diagnostic130').mkdir(exist_ok=False)
contract={'status':'SEALED_BEFORE_DIAGNOSTIC_GPU','pilot_raws':[130],'states':['keep','delete'],'variants':['baseline','AA','C'],
 'scope':'Separate fixed35-ray tire diagnostic; never included in official raw94/105/128/137 pooled metrics.',
 'planned_joint_forwards':6,'model_and_road_asset_parameters_unchanged':True,'optimization_updates':0,'all49_auto_release':False,
 'fixed_wheel_registry':str(ROOT/'contract/fixed_wheel_rays_v1.json'),'created_unix':time.time(),'shutdown':False,'reset_card':False}
cp=ROOT/'contract/diagnostic130_protocol.json';cp.write_text(json.dumps(contract,indent=2)+'\n')
workers={}
for variant in ['baseline','road']:
 src=ROOT/'source'/('render_latest_semantic_baseline4.py' if variant=='baseline' else 'render_road_pilot4_retry1.py');s=src.read_text()
 prior='contract/baseline4_protocol.json' if variant=='baseline' else 'contract/pilot4_execution_contract.json'
 assert s.count(prior)==1;s=s.replace(prior,'contract/diagnostic130_protocol.json')
 prior='SEALED_BEFORE_BASELINE_GPU' if variant=='baseline' else 'SEALED_BEFORE_CANDIDATE_GPU';assert s.count(prior)==1;s=s.replace(prior,'SEALED_BEFORE_DIAGNOSTIC_GPU')
 assert s.count('[94,105,128,137]')==3;s=s.replace('[94,105,128,137]','[130]')
 prior="outdir=ROOT/'baseline4'" if variant=='baseline' else "outdir=ROOT/(variant+'4_retry1')"
 after="outdir=ROOT/'diagnostic130/baseline'" if variant=='baseline' else "outdir=ROOT/'diagnostic130'/variant"
 assert s.count(prior)==1;s=s.replace(prior,after);ast.parse(s)
 p=ROOT/'source'/('render_tire130_'+variant+'.py');assert not p.exists();p.write_text(s);workers[variant]=p
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
done=[]
for variant in ['baseline','AA','C']:
 w=workers['baseline' if variant=='baseline' else 'road'];cmd=[sys.executable,'-B',str(w)]+([] if variant=='baseline' else [variant])
 with (ROOT/('diagnostic130_'+variant+'.log')).open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
 v={'variant':variant,'pid':p.pid,'command':cmd,'worker_sha256':hashlib.sha256(w.read_bytes()).hexdigest(),'unix':time.time(),'optimization_updates':0};print(json.dumps(v),flush=True)
 code=p.wait();done.append(dict(v,exit_code=code));(ROOT/'diagnostic130/status.json').write_text(json.dumps({'completed':done,'no_all49_release':True},indent=2)+'\n')
 if code!=0:raise SystemExit(code)
 # Expose only a diagnostic symlink; original official-four receipts remain unchanged.
 source=ROOT/'diagnostic130'/variant/'frames/130'
 official=ROOT/({'baseline':'baseline4','AA':'AA4_retry1','C':'C4_retry1'}[variant])/'frames/130'
 assert not official.exists();official.symlink_to(source,target_is_directory=True)
print(json.dumps({'status':'COMPLETE_THREE_VARIANT_RAW130_DIAGNOSTIC','actual_joint_forwards':6,'optimization_updates':0}),flush=True)
