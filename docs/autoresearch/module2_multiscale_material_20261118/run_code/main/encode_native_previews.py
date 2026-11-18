from pathlib import Path
import json,time
import numpy as np
from PIL import Image

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
rows=[]
for src in sorted(P.rglob('*_native.npz')):
    if time.time()-src.stat().st_mtime<3:continue
    raw=src.name.split('_')[0]
    if raw not in ['086','110','134','179','198']:continue
    with np.load(src,allow_pickle=False) as z:
        for key,mode in [('road_rgb','road'),('joint_rgb','joint')]:
            if key not in z:continue
            rgb=z[key]
            assert rgb.shape==(3,320,480) and np.isfinite(rgb).all()
            out=src.parent/f'{raw}_{mode}.webp'
            Image.fromarray(np.clip(np.rint(rgb.transpose(1,2,0)*255),0,255).astype(np.uint8)).save(out,lossless=True)
            rows.append(dict(source=str(src.relative_to(P)),key=key,output=str(out.relative_to(P)),
                encoding='native sRGB rounded/clipped8bit, lossless WebP; no recoloring or image correction'))
receipt=dict(schema='module2_actual_native_preview_encoding.v1',native_GPU_calls=0,optimizer_updates=0,rows=rows)
(P/'delivery/preview_encoding.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(images=len(rows),native_calls=0)))
