import json, pathlib, sys
import numpy as np
import cv2
from PIL import Image
REPO=pathlib.Path('/root/autodl-tmp/motion_proj_v77')
ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r2')
OLD=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1')
FULL=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-FULL-20260927/r1')
HW=(536,960); H,W=HW
sys.path.insert(0,str(REPO/'scripts/worldsim_v77'))
from geometry import project_bbox,resized_intrinsics,transform
from hybrid_masks import mask_contract,compose_background
def read(p):return json.loads(pathlib.Path(p).read_text())
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rgb(p):return np.array(Image.open(p).convert('RGB').resize((W,H),Image.Resampling.BILINEAR))
def mask(p):return np.array(Image.open(p).convert('L').resize((W,H),Image.Resampling.NEAREST))>127
def write_mask(p,m):Image.fromarray(m.astype('uint8')*255).save(p)
def load_masks(out,i):
 return {k:mask(out/k/f'{i:05}.png') for k in ['delete','protect','generate','observed','residual_delete','residual_generate']}
