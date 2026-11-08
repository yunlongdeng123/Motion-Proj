import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
import torch
from mmseg.apis import init_segmentor,inference_segmentor
ROOT=Path(__file__).resolve().parent.parent
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108/inputs')
def main():
    dest=ROOT/'semantics';dest.mkdir(exist_ok=True);torch.set_num_threads(1)
    config='/root/autodl-tmp/external/dggt_segformer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py'
    checkpoint='/root/autodl-tmp/drivingeditor_repro/weights/segformer.b5.1024x1024.city.160k.pth'
    model=init_segmentor(config,checkpoint,device='cuda:0');start=time.monotonic();rows=[]
    frames=json.loads((BANK/'receipt.json').read_text())['frames']
    for frame in frames:
        path=dest/f'{frame["raw"]:03d}.png'
        if not path.exists():
            with torch.no_grad():labels=inference_segmentor(model,str(BANK/'image/image_0'/f'{frame["id"]}.png'))[0]
            Image.fromarray(labels.astype(np.uint8)).save(path)
        labels=np.array(Image.open(path));rows.append(dict(raw=frame['raw'],road_pixels=int((labels==0).sum()),sky_pixels=int((labels==10).sum())))
        print(frame['raw'],flush=True)
    (dest/'receipt.json').write_text(json.dumps(dict(status='COMPLETE_BUILD_SEMANTICS',frames=rows,model='Same original SegFormer B5 Cityscapes',config=config,checkpoint=checkpoint,seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':main()
