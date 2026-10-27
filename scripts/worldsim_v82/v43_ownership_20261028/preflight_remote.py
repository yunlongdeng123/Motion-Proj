"""Read-only inventory. Does not start old controllers or modify model data."""
import json, os, sys, time, importlib.util, shutil
from pathlib import Path
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
print('PYTHON',sys.executable,sys.version)
import torch
print('TORCH',torch.__version__,'CUDA',torch.cuda.is_available())
print('PACKAGES',{k:bool(importlib.util.find_spec(k)) for k in ['transformers','mmseg','cv2','sam2']})
for p in [Path('/root/autodl-tmp/models/worldsim_v6/r20_semantic_segformer_cityscapes/config.json'),Path('/root/autodl-tmp/models/legacy_v4_preprocess/segformer-b5-2c6f153/config.json')]:
    if p.exists():print('MODEL_CONFIG',str(p),p.read_text())
for p in sorted((R/'native_scene_training_v1').glob('*/status.json')):
    d=json.loads(p.read_text());run=p.parent/'formal30k'
    print('SCENE',p.parent.name,d.get('phase'),[f.name for f in run.glob('*30000.pth')])
print('V43_EXISTING',[str(p) for p in R.parent.glob('*v43*')])
print('RESOURCE_CONTROL')
for p in [R/'resource_state.json', R/'STOP', R/'code/resource_control.py']:
    if p.exists():print(str(p),p.read_text()[:6000])
