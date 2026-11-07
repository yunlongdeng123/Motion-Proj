"""Freeze and replay the latest semantic baseline, without loading road AA/C."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
NEW=B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
SEM=B/'scene008_v44_full5_integration_20261107_r1'
ADAPTER=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ADAPTER)=='c177c97bea2c73e896fa925d3ecaa8ed490bb5b810e352ed77b2af23c5610c44'
for d in ['source','contract']: (NEW/d).mkdir(parents=True,exist_ok=True)
manifest=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json';selection=SEM/'semantic_actor5_calibration49/semantic_actor5_rows.npz'
assert sha(manifest)=='f77f33745925956cc582a0cb2b80f9e1bf2708ed41c7fcddecff8c891a4a937f'
assert sha(selection)=='3c3b0818a4794127915d2aed1d5b19da6b51ddbbf58fb35858d801724dbed0ec'
transform=SEM/'contract/full5_transform_matrix.txt';destination=NEW/'contract/full5_transform_matrix.txt'
if destination.exists():assert destination.read_bytes()==transform.read_bytes()
else:destination.write_bytes(transform.read_bytes())
contract={'status':'SEALED_BEFORE_BASELINE_GPU','schema':'scene008.latest_semantic_road_integration.baseline4.v1',
 'created_unix':time.time(),'pilot_raws':[94,105,128,137],'states':['keep','delete'],
 'baseline':'latest SemanticActor5V44Integration; adopted V44 + fixed all49 BUILD sidecar',
 'base_v44_contract':str(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json'),
 'adapter_sha256':sha(ADAPTER),'semantic_manifest':str(manifest),'semantic_manifest_sha256':sha(manifest),
 'selection_sha256':sha(selection),'transform_sha256':sha(destination),
 'cached_semantic_reference':str(SEM/'candidate_semantic_a5_49'),
 'variants_to_compare':['latest_semantic_baseline','AA_support','C_material_same_support'],
 'candidate_inputs_and_thresholds_to_be_sealed_before_candidate_GPU':True,
 'separate_tire_diagnostic_raws':[130],'training_updates':0,'background_solves':0,'planned_baseline_joint_forwards':8,
 'all49_automatic_release':False,'shutdown':False,'reset_card':False}
cp=NEW/'contract/baseline4_protocol.json';assert not cp.exists();cp.write_text(json.dumps(contract,indent=2)+'\n')
s=(SEM/'source/render_v44_semantic_actor5_all49fit_candidate4.py').read_text()
changes={
 "ROOT=B/'scene008_v44_full5_integration_20261107_r1'":"ROOT=B/'scene008_v44_a5_road_fusion_integration_20261107_r1'",
 'contract/execution_contract_semantic_actor5_all49fit.json':'contract/baseline4_protocol.json',
 'SEALED_BEFORE_CANDIDATE_GPU':'SEALED_BEFORE_BASELINE_GPU',
 " and (ROOT/'contract/root_baseline4_exact_png_audit.json').exists()":'',
 "ROOT/'semantic_actor5_calibration49/semantic_actor5_manifest.json'":"B/'scene008_v44_full5_integration_20261107_r1/semantic_actor5_calibration49/semantic_actor5_manifest.json'",
 "ROOT/'semantic_actor5_calibration49/semantic_actor5_rows.npz'":"B/'scene008_v44_full5_integration_20261107_r1/semantic_actor5_calibration49/semantic_actor5_rows.npz'",
 "outdir=ROOT/'candidate_semantic_a5_all49fit_4'":"outdir=ROOT/'baseline4'",
 'RENDERED_FIXED_SEMANTIC_ACTOR5_SIDECAR_PILOT_PENDING_GATES_AND_ROOT_REVIEW':'RENDERED_LATEST_SEMANTIC_BASELINE4_PENDING_CACHED_EXACT_REPLAY',
}
for before,after in changes.items():
 assert s.count(before)==1,(before,s.count(before));s=s.replace(before,after)
assert s.count('[94,105,130,137]')==3;s=s.replace('[94,105,130,137]','[94,105,128,137]')
ast.parse(s);worker=NEW/'source/render_latest_semantic_baseline4.py';assert not worker.exists();worker.write_text(s)
assert not (NEW/'baseline4').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]);assert free>=12288
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=[sys.executable,'-B',str(worker)]
with (NEW/'baseline4.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=NEW,env=env,start_new_session=True)
v={'status':'STARTED_LATEST_SEMANTIC_BASELINE4','pid':p.pid,'command':cmd,'contract_sha256':sha(cp),'source_sha256':sha(worker),'raws':[94,105,128,137],
 'free_MiB':free,'unix':time.time(),'actual_forwards_not_precounted':True,'optimization_updates':0,'shutdown':False,'reset_card':False}
(NEW/'contract/baseline4_launch.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
