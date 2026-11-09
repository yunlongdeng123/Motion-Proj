"""BUILD-only SegFormer class maps and sky, with exact original pretrained model."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
def main():
    import torch
    from mmseg.apis import init_segmentor,inference_segmentor
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and not (a.output/'receipt.json').exists():
        deadline=time.monotonic()+900
        while not (a.output/'receipt.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('parallel semantic export did not complete')
            time.sleep(2)
    if (a.output/'receipt.json').exists():
        receipt=json.loads((a.output/'receipt.json').read_text());frames=json.loads((a.bank/'receipt.json').read_text())['frames']
        assert receipt['status']=='COMPLETE_BUILD_SEMANTICS' and [f['raw'] for f in frames]==[f['raw'] for f in receipt['frames']]
        assert all((a.output/f'{f["raw"]:03d}.png').exists() and (a.bank/'sky/sky_0'/f'{f["id"]}.png').exists() for f in frames)
        print('REUSE_COMPLETED_BUILD_SEMANTICS');return
    a.output.mkdir(exist_ok=False);sky=a.bank/'sky/sky_0';sky.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    config='/root/autodl-tmp/external/dggt_segformer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py'
    checkpoint='/root/autodl-tmp/drivingeditor_repro/weights/segformer.b5.1024x1024.city.160k.pth'
    model=init_segmentor(config,checkpoint,device='cuda:0');start=time.monotonic();rows=[]
    frames=json.loads((a.bank/'receipt.json').read_text())['frames']
    for frame in frames:
        with torch.no_grad():labels=inference_segmentor(model,str(a.bank/'image/image_0'/f'{frame["id"]}.png'))[0]
        Image.fromarray(labels.astype(np.uint8)).save(a.output/f'{frame["raw"]:03d}.png')
        Image.fromarray(np.uint8(labels==10)*255).save(sky/f'{frame["id"]}.png')
        rows.append(dict(raw=frame['raw'],road_pixels=int((labels==0).sum()),sky_pixels=int((labels==10).sum())))
        print(frame['raw'],flush=True)
    (a.output/'receipt.json').write_text(json.dumps(dict(status='COMPLETE_BUILD_SEMANTICS',frames=rows,model='Original SegFormer B5 Cityscapes',config=config,checkpoint=checkpoint,seconds=time.monotonic()-start,heldout_used=False),indent=2))
if __name__=='__main__':main()
