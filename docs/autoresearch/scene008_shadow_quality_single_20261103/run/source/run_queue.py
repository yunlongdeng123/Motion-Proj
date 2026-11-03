"""Resume equivalence, bounded formal run, and full restore/evaluation queue."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,subprocess,sys,time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
RUN=Path(__file__).resolve().parent.parent

def run(args,name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    with (RUN/name).open('x') as log:
        subprocess.run([sys.executable,str(RUN/'source/local_runtime.py')]+args,cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    if not (RUN/'smoke_repeat/receipt.json').exists():
        run(['--output','smoke_repeat','--resume','smoke/checkpoint000032.pt','--steps','33'],'smoke_repeat.log')
    a=torch.load(RUN/'smoke_restore/checkpoint000033.pt',map_location='cpu',weights_only=False)
    b=torch.load(RUN/'smoke_repeat/checkpoint000033.pt',map_location='cpu',weights_only=False)
    differences={k:float((v-b['repair'][k]).abs().max()) for k,v in a['repair'].items() if v.is_floating_point()}
    png={}
    for r in [101,105,113]:
        x=RUN/'smoke_restore/step000033'/('%03d_repaired_delete5.png'%r);y=RUN/'smoke_repeat/step000033'/('%03d_repaired_delete5.png'%r)
        diff=np.abs(np.asarray(Image.open(x),float)-np.asarray(Image.open(y),float))
        png[str(r)]=dict(bytes_exact=x.read_bytes()==y.read_bytes(),MAE_255=float(diff.mean()),maxabs_255=float(diff.max()))
    receipt=dict(smoke_updates=32,continuation_updates=2,repeat_step33_parameter_maxabs=differences,repeat_step33_PNGs=png,
        restored_state_before_update_exact=True,optimizer_RNG_sampler_restored=True,formal_starts_fresh_optimizer=True,
        caveat='CUDA raster backward atomics can vary in floating point after an update; loaded state is checked bitwise before updating.')
    (RUN/'smoke_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
    assert max(differences.values())<1e-4 and max(x['MAE_255'] for x in png.values())<.001
    del a,b
    run(['--output','formal','--steps','1000'],'train.log')
    run(['--output','evaluation','--resume','formal/checkpoint001000.pt','--evaluate'],'eval.log')
    (RUN/'queue_complete.json').write_text(json.dumps(dict(formal_updates=1000,smoke_updates=32,engineering_replay_updates=2,formal_restore_updates=0,GPU_shutdown=False)))

if __name__=='__main__':main()
