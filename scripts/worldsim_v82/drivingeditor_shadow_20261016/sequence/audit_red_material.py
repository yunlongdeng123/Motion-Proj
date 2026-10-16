import os,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
s=torch.load(ROOT/'contribution_frozen_v1/final.pt',map_location='cpu',weights_only=False)['surface']
ref=s['reference'];ids=s['ids'];color=s['color_raw'].sigmoid();confidence=s['confidence']
red=lambda x:(x[:,0]-x[:,1]>.05)&(x[:,0]-x[:,2]>.05)
result={'reference_red_texels_in_confident_road':int((red(ref)&(confidence>.9)).sum()),
    'learned_red_receivers_in_confident_road':int((red(color)&(confidence[ids]>.9)).sum()),
    'criterion':'linear R>G+.05 and R>B+.05; not a semantic label','purpose':'test whether observed red fragments are baked in the new road material'}
(ROOT/'red_material_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
