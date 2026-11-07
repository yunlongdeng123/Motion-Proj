from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
source=R/'source/probe_white_point_parents.py';contract=R/'contract/white_point_parent_diagnostic_v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='0198fae7ba7b09a1567be40d97ac3439879ebb87fb9b2c6e72ccd975d98a011b'
assert sha(contract)=='a0097c3f9b7f63bbc7f045940008dab0af1580535fc157a1d1acf7d2b966b437'
assert json.loads(contract.read_text())['status']=='SEALED_BEFORE_TWO_POINT_GPU'
assert not (R/'white_point_parent_diagnostic_v1').exists()
compile(source.read_text(),str(source),'exec',dont_inherit=True)
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(source)]
with (R/'white_point_parent_diagnostic_v1.log').open('x') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'command':cmd,'source_sha256':sha(source),'contract_sha256':sha(contract),'unix':time.time(),'planned_F':2,'planned_dummy_B':2,'optimizer_updates':0,'no_automatic_fit_or49':True}
(R/'contract/white_point_parent_diagnostic_v1_launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
