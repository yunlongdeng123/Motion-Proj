"""目标12自身GLB用于r30原位接回；单参考、单seed，不复用另一辆车身份。"""
from pathlib import Path
import sys,datetime,os,numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r5'

def mask():
    import torch,cv2
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    assert not (R/'registration.json').exists()
    src=read(R/'view_candidates.json')[0];assert (src['frame'],src['camera'])==(46,2)
    dump(R/'registration.json',dict(task_id=T.name,run_id='r5',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',source=src,
        role='One real rear/side original view selected by projected area and source-image review. SAM2 mask then existing frozen Hunyuan2.1. Opposite surfaces are prior, not evidence. GT pose and size only later placement.',
        seed=7740,shape_steps=50,guidance=7.5,octree=256,texture_views=6,texture_resolution=512,cpu_threads=4,stop='Review shape/0-vs180 orientation before texture or r30 putback. No seed/view search after generation.',human_verdict=None,failure_ledger_refs=['V77-F02']))
    im=np.array(Image.open(src['image']).convert('RGB'));bb=np.array(src['box'])*np.array([im.shape[1]/1024,im.shape[0]/576]*2)
    device=os.environ.get('SAM_DEVICE','cuda');torch.set_num_threads(4);cv2.setNumThreads(4);sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device=device))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):sam.set_image(im);m,score,_=sam.predict(box=bb,multimask_output=True)
    choice=int(np.argmax(score));sel=m[choice]>0;np.savez_compressed(R/'sam_candidates.npz',masks=m,scores=score)
    yy,xx=np.where(sel);box=[max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(im.shape[1],int(xx.max())+13),min(im.shape[0],int(yy.max())+13)]
    Image.fromarray(np.dstack([im,sel.astype('uint8')*255])).crop(box).save(R/'source_rgba.png');Image.fromarray(im).crop(box).save(R/'source_context.png');Image.fromarray(sel.astype('uint8')*255).crop(box).save(R/'source_mask.png')
    dump(R/'mask_state.json',dict(state='complete_pending_visual_review',device=device,choice=choice,scores=score.tolist(),crop=box,pixels=int(sel.sum()),human_verdict=None))

def model(stage):
    import hybrid_retained_actor_asset as asset
    asset.ROOT=R
    if stage=='shape':asset.shape()
    else:asset.paint()

if __name__=='__main__':
    if sys.argv[-1]=='mask':mask()
    else:model(sys.argv[-1])
