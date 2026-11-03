"""Exercise editable coarse coverage settings without changing final inputs."""
import os,json,subprocess,sys
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent.parent
def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    with (RUN/'parameter_entry.log').open('x') as f:
        subprocess.run([sys.executable,str(RUN/'source/prepare_local.py'),'--output-name','coarse_inputs'],env=env,cwd=RUN,stdout=f,stderr=subprocess.STDOUT,check=True)
    rebuilt=np.load(RUN/'coarse_inputs/road_reference.npz');original=np.load(RUN/'inputs/coarse_reference.npz')
    fields=['active','edit_weight','road_opacity','reference_srgb','reference_linear']
    dif={k:float(np.abs(rebuilt[k].astype(float)-original[k].astype(float)).max()) for k in fields}
    assert max(dif.values())<1e-7
    result=dict(parameter_entry_executed=True,default_parameters_reproduce_final_coarse_inputs=True,maxabs=dif,optimizer_updates=0,
        workflow='new directory: prepare_local.py --output-name coarse_inputs; refine_road.py; start_quality.py',
        dependencies='original v41 B, R4 real199 evidence and original view records, R5 immutable native surface stats, V3 coordinates and fullclip199 sensors')
    (RUN/'parameter_entry_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
