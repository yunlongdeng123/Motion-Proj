from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
source=R/'source/render_L_Cwarm12_counterfactual.py';contract=R/'contract/L_Cwarm12_counterfactual_v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='c8a61d2a2fbc2b53837bf0edb36fea73c7c9b80b4d465847b46633876b710fee'
assert sha(contract)=='6acc266cdb28633d45596e88589d78451c03a30b01a1c80e33dad9a720ff9571'
assert json.loads(contract.read_text())['status']=='SEALED_BEFORE_L_NATIVE8F'
assert not (R/'J4_road_parent_operator/L_Cwarm12_render_v1').exists()
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(source)]
with (R/'J4_road_parent_operator/L_Cwarm12_render_v1.log').open('x') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'command':cmd,'source_sha256':sha(source),'contract_sha256':sha(contract),'unix':time.time(),'planned_F':8,'planned_B':0,'Adam_updates':0,'new_solver_calls':0,'no_automatic49_or_fit':True}
(R/'contract/L_Cwarm12_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
