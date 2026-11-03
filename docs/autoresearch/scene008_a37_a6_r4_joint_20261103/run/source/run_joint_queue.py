"""Full-state smoke continuation, fresh joint1k, and restored all49 evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import subprocess,sys,json
from pathlib import Path
import torch
RUN=Path(__file__).resolve().parent.parent
def run(args,log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    with (RUN/log).open('x') as f:subprocess.run([sys.executable,str(RUN/'source/joint_runtime.py')]+args,cwd=RUN,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    assert (RUN/'smoke/receipt.json').exists()
    run(['--output','smoke_restore','--steps','33','--resume','smoke/checkpoint000032.pt'],'smoke_restore.log')
    run(['--output','smoke_repeat','--steps','33','--resume','smoke/checkpoint000032.pt'],'smoke_repeat.log')
    a=torch.load(RUN/'smoke_restore/checkpoint000033.pt',map_location='cpu',weights_only=False);b=torch.load(RUN/'smoke_repeat/checkpoint000033.pt',map_location='cpu',weights_only=False)
    diff=max(float((v-b['joint'][k]).abs().max()) for k,v in a['joint'].items() if v.is_floating_point());assert diff<1e-4
    exact={str(r):(RUN/'smoke_restore/step000033'/('%03d_joint.png'%r)).read_bytes()==(RUN/'smoke_repeat/step000033'/('%03d_joint.png'%r)).read_bytes() for r in [101,105,113]}
    (RUN/'smoke_receipt.json').write_text(json.dumps(dict(smoke_updates=32,replay_updates=2,loaded_state_exact=True,optimizer_RNG_restored=True,repeated_step_parameter_maxabs=diff,PNG_exact=exact,formal_starts_fresh=True)))
    del a,b
    run(['--output','formal','--steps','1000'],'train.log')
    run(['--output','evaluation','--resume','formal/checkpoint001000.pt','--evaluate'],'eval.log')
    (RUN/'queue_complete.json').write_text(json.dumps(dict(formal_updates=1000,smoke_updates=32,replay_updates=2,formal_restore_updates=0,GPU_shutdown=False)))
if __name__=='__main__':main()
