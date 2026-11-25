"""Official VACE inpainting: the unknown RGB region must be neutral gray."""
import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,'/root/autodl-tmp/_wsjob')
from vace_run_clip import CLIPS,OUT,R5

base=Path(OUT);base.mkdir(exist_ok=True)
manifest=dict(schema='scene065.shared_official_inpainting.v1',mask_gray_value=127,
    masks=f'{R5}/masks_w8_swept_cd18',reference_images_used=False,clips={})
for clip,(lo,hi) in CLIPS.items():
    for name in ('src_video','src_mask'):(base/clip/name).mkdir(parents=True,exist_ok=True)
    for k,raw in enumerate(range(lo,hi+1)):
        src=cv2.imread(f'{R5}/input_frames/{raw:05d}.png')
        mask=cv2.imread(f'{R5}/masks_w8_swept_cd18/{raw:05d}.png',0)
        blind=src.copy();blind[mask>0]=127
        assert np.array_equal(blind[mask==0],src[mask==0]) and (blind[mask>0]==127).all()
        cv2.imwrite(str(base/clip/f'src_video/{k:05d}.png'),blind)
        cv2.imwrite(str(base/clip/f'src_mask/{k:05d}.png'),mask)
    manifest['clips'][clip]=dict(frames=[lo,hi],count=hi-lo+1,src_video=str(base/clip/'src_video'),
        src_mask=str(base/clip/'src_mask'),reference_images_used=False)
(base/'clips_manifest.json').write_text(json.dumps(manifest,indent=2))
old=base.parent/'scene065_vace13b_20261126_r1/smoke_087'
new=base/'smoke_087';new.mkdir(exist_ok=True)
shutil.copy2(old/'smoke_info.json',new/'smoke_info.json')
shutil.copytree(old/'attempts',new/'attempts',dirs_exist_ok=True)
print('Staged identical gray-masked PNG inputs for both backends; native GPU runtime smoke reused.',flush=True)
