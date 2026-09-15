import ast,importlib.metadata as m,json
from pathlib import Path
import torch,numpy,scipy
print('RUNTIME',torch.__version__,torch.cuda.is_available(),numpy.__version__,scipy.__version__,flush=True)
names=['appdirs','av','awscli','comet_ml','cryptography','tyro','gdown','ninja','h5py','ipywidgets','jaxtyping','jupyterlab','mediapy','msgpack','msgpack_numpy','nerfacc','open3d','opencv-python','Pillow','plotly','protobuf','pymeshlab','pyngrok','python-socketio','pyquaternion','rawpy','requests','rich','scikit-image','splines','tensorboard','torchmetrics','typing_extensions','viser','nuscenes-devkit','wandb','xatlas','trimesh','timm','gsplat','pytorch-msssim','pathos','packaging','zod','pandaset','av2','fastapi','numba','tinycudann']
out={}
for n in names:
 try:out[n]=m.version(n)
 except m.PackageNotFoundError:out[n]=None
print(json.dumps(out,indent=2),flush=True)
Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1/runtime_initial.json').write_text(json.dumps(out,indent=2))
