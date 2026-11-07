from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1');PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
assert json.loads((R/'F_zero_identity4/receipt.json').read_text())['status']=='PASS_PUBLIC_BOUNDED_F_ZERO_DELTA_ALL4_KD_EXACT_BASELINE'
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
source=R/'source/F_colour_bake_core.py'
cmd=[PY,'-B',str(source),'--mode','smoke','--output',str(R/'F_smoke_r1')]
with (R/'F_smoke_r1.log').open('x') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'command':cmd,'source_sha256':sha(source),'unix':time.time(),'formal_released_by_this_launcher':False,'planned_smoke_paired_updates':4,'disposable_next2_updates':2,'new_zero_identity_native_forwards':8}
write(R/'contract/F_smoke_r1_launch.json',record);print(json.dumps(record),flush=True)
