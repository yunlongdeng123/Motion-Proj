from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=R/'source/measure_J_road_parent_operator.py';contract=R/'contract/J_parent_RGB_operator_v1.json'
assert sha(source)=='ad82a8c4ff9c72bb16fa1bc949147f827b6d5b62a68f69ef26b48e0fc0138207' and sha(contract)=='e2ef3c2656eb3c38fce208d840a28575818092ebb80aebc61d612d6ecf2e3bb5'
assert json.loads(contract.read_text())['status']=='SEALED_BEFORE_J_GPU'
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(source)]
with (R/'J4_road_parent_operator.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'command':cmd,'source_sha256':sha(source),'contract_sha256':sha(contract),'unix':time.time(),'planned_native_forwards':24,'planned_dummy_backwards':16,'optimizer_updates':0,'automatic49_or_training':False};(R/'contract/J_operator_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
