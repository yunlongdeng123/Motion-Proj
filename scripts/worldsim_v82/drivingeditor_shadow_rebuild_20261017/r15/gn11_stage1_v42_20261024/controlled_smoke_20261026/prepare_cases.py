import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,shutil
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
sys.path.insert(0,'/root/autodl-tmp')
from fullclip_road_cpu_20261026 import loadmeta,FixedSurface,measured_patch_evidence,ray_box_blocked,transform
FULL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026')
OUT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
m,anchor,inv,plane=loadmeta();a=dict(np.load(FULL/'full199_initialization.npz'))
surface=FixedSurface(m,dict(keys=a['keys'],world=a['world_xyz'],surface_supported=a['surface_supported'],provenance=a['eligible'].astype(np.uint8),base_weight=a['source_valid'],normal_world=a['normal_world']))
manifest=[]
for case,label in [('a37',5),('a6',4)]:
    folder=OUT/'inputs'/case;(folder/'input').mkdir(parents=True,exist_ok=True);(folder/'masks').mkdir(exist_ok=True)
    for raw in [102,109,116]:
        z=np.load(f'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/frames/0008{raw:03d}.npz')
        mask=z['actor_labels']==label
        # Keep supplied silhouette; upstream registered 15px dilation applies once.
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/'masks'/f'{raw:03d}.png')
        shutil.copyfile(FULL/'rgb'/f'{raw:03d}.png',folder/'input'/f'{raw:03d}.png')
        manifest.append(dict(case=case,raw=raw,mask_pixels=int(mask.sum()),source='raw observed RGB; object-present is NOT deletion GT'))
(OUT/'inputs/frozen_cases.json').write_text(json.dumps(manifest,indent=2))
train=OUT/'dense_training';train.mkdir(exist_ok=True)
for raw in [89,96,102,109,116,123,130,137,141,143,145,147,149,151,153,155,157,159]:
    fp=train/f'{raw:03d}.npz'
    if fp.exists():continue
    f=np.load(FULL/'frames'/f'{raw:03d}.npz');w2world=f['w2l']@inv;r=surface.camera(f['K'],w2world)
    safe=np.maximum(r['unit_id'],0);query=np.flatnonzero(r['supported']&a['eligible'][safe]&~f['guard'].ravel())
    # Derive dense visible-road labels from same-frame small measured triangles.
    ok,dep,tri,conflict=measured_patch_evidence(f['cloud'],f['ground'],f['K'],f['w2l'],transform(r['world'][query],inv),r['pixels'][query],a['normal_local'][safe[query]])
    boxes=[dict(center=c,half=h,axes=ax) for c,h,ax in zip(f['box_center'],f['box_half'],f['box_axes'])]
    ok&=~ray_box_blocked(transform(r['world'][query],inv),np.linalg.inv(f['w2l'])[:3,3],boxes)
    visible=np.zeros(320*480,bool);visible[query[ok]]=True
    rgb=np.asarray(Image.open(FULL/'rgb'/f'{raw:03d}.png'),np.float32)/255.
    np.savez_compressed(fp,rgb=rgb,visible=visible.reshape(320,480),surface_depth=r['depth'].reshape(320,480),unit_id=r['unit_id'].reshape(320,480))
    print(json.dumps(dict(raw=raw,dense_visible_road=int(visible.sum()))),flush=True)
