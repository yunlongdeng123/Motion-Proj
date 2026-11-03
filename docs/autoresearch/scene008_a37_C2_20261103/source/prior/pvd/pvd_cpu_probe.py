import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,types,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
sys.path.insert(0,str(ROOT))
import torch
from inference_core import Model,get_betas
started=time.monotonic()
args=types.SimpleNamespace(nc=3,svpoints=200,embed_dim=64,attention=True,dropout=.1)
model=Model(args,get_betas('linear',.0001,.02,1000),'mse','eps','fixedsmall')
checkpoint=torch.load(ROOT/'weights/car_completion_3199.pth',map_location='cpu')
state={key.replace('model.module.','model.',1):value for key,value in checkpoint['model_state'].items()}
model.load_state_dict(state,strict=True)
model.requires_grad_(False);model.model.eval();model.training=False
report=dict(state_dict_strict=True,parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            every_model_tensor_finite=all(bool(torch.isfinite(t).all()) for t in state.values()),
            torch=torch.__version__,gpu_initialized=torch.cuda.is_initialized(),
            source_checkpoint_epoch=checkpoint['epoch'],diffusion_steps=1000,
            fixed_partial_points=200,total_points=2048,new_noise_points=1848,
            state_key_prefix_map='model.module. -> model. only',wall_seconds=time.monotonic()-started)
(ROOT/'model_cpu_probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
