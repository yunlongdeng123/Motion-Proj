import json, time
from pathlib import Path
import torch
start=time.time()
from gsplat.cuda._backend import _C
assert _C is not None
p=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1/cuda_build.json')
p.write_text(json.dumps({'module':str(_C.__file__),'seconds':time.time()-start,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'completed':True},indent=2))
print(p.read_text(),flush=True)
