import json, importlib.metadata as m, subprocess
from pathlib import Path
import torch, numpy, cv2
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
names=['torch','torchvision','numpy','scipy','numba','av2','tyro','gsplat','neurad-studio','viser','pandaset','zod','dataclass-wizard','torchmetrics','nerfacc','tinycudann','splines','pytorch-msssim','protobuf','wandb']
versions={}
for n in names:
    try:versions[n]=m.version(n)
    except m.PackageNotFoundError:versions[n]=None
out={'versions':versions,'actual_cv2':cv2.__version__,'torch_cuda':torch.version.cuda,
    'repositories':{r:subprocess.check_output(['git','-C',str(B/r),'rev-parse','HEAD'],text=True).strip() for r in ['neurad-studio','SplatAD_splat','viser-splatad','pandaset-devkit']},
    'deviations':['AV2 0.3.6 instead of 0.2.1; AV2 input unused','Numba 0.65.1 instead of 0.57, NumPy 1.26.4 retained','OpenCV 4.11.0 instead of 4.10.0','Only backend=eager compile mode option removed for Torch2.4 compatibility','Optional exporters, Jupyter and cloud logging dependencies not installed; not exercised'],
    'isolation':'Separate site-package namespace using initial hardlink copy; package replacements only in splatad-impact; no modifications to prior experiment environments'}
(N/'runtime_final.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
