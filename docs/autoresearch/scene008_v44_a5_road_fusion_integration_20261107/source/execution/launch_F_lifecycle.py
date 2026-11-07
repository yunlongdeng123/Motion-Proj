from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1');PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=[PY,'-B',str(R/'source/F_lifecycle_then_formal400.py')]
with (R/'F_training_queue.log').open('x') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'command':cmd,'lifecycle_source_sha256':sha(R/'source/F_lifecycle_then_formal400.py'),'restore_gate_source_sha256':sha(R/'source/F_restore_parity_gate.py'),'core_source_sha256':sha(R/'source/F_colour_bake_core.py'),'unix':time.time(),'only_fresh400_after_lifecycle_pass':True,'all49_auto_release':False}
(R/'contract/F_training_queue_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
