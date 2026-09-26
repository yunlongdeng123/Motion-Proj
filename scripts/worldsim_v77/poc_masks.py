"""两场景 GT-box 提示的 SAM2 视频 mask POC；GT 仅用于提示/限制漂移。"""
import argparse
import json
import pathlib
import sys
import time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from geometry import project_bbox, resized_intrinsics

ROOT = pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
SCENES = {
 'scene_0230': {'data': '/root/autodl-tmp/data/v76_vadgs/scene_0230', 'actor': '22', 'count': 50},
 'scene_0255': {'data': '/root/autodl-tmp/data/v76_vadgs/scene_0255', 'actor': '25', 'count': 100},
}

def boxes_for(scene, camera):
    root=pathlib.Path(scene['data'])
    ann=json.loads((root/'instances/instances_info.json').read_text())[scene['actor']]['frame_annotations']
    vals=np.loadtxt(root/'intrinsics'/f'{camera}.txt')
    k=np.array([[vals[0],0,vals[2]],[0,vals[1],vals[3]],[0,0,1]])
    k=resized_intrinsics(k,(1600,900),(384,688))
    result=[]
    for frame in range(scene['count']):
        if frame not in ann['frame_idx']:
            result.append(None);continue
        j=ann['frame_idx'].index(frame)
        pose=np.asarray(ann['obj_to_world'][j]);size=ann['box_size'][j]
        c2w=np.loadtxt(root/'extrinsics'/f'{frame:03d}_{camera}.txt')
        box=project_bbox(pose,size,c2w,k,(384,688))
        result.append(box)
    return result

def prepare(scene_name):
    scene=SCENES[scene_name];root=pathlib.Path(scene['data']);out=ROOT/scene_name
    out.mkdir(parents=True,exist_ok=True)
    metadata={'scene':scene_name,'actor':scene['actor'],'frames':scene['count'],
      'source':str(root),'camera_streams':{},'method':'SAM2.1 large; GT 3D box projected to initial 2D prompt and box gate; zero training'}
    for camera in range(6):
        boxes=boxes_for(scene,camera)
        areas=[(b[2]-b[0])*(b[3]-b[1]) if b else 0 for b in boxes]
        active=[i for i,a in enumerate(areas) if a>=400]
        metadata['camera_streams'][str(camera)]={'first':active[0] if active else None,
           'last':active[-1] if active else None,'active_frames':len(active),'max_area':round(max(areas)),'boxes':boxes}
        if not active:continue
        frames=out/'rgb'/f'cam{camera}';frames.mkdir(parents=True,exist_ok=True)
        for f in range(scene['count']):
            target=frames/f'{f:05d}.jpg'
            if target.exists():continue
            im=Image.open(root/'images'/f'{f:03d}_{camera}.jpg').convert('RGB').resize((688,384),Image.Resampling.BICUBIC)
            im.save(target,quality=95)
    (out/'selection.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({scene_name:{c:{k:v for k,v in d.items() if k!='boxes'} for c,d in metadata['camera_streams'].items()}},ensure_ascii=False),flush=True)

def segment(scene_name,limit=None):
    out=ROOT/scene_name;meta=json.loads((out/'selection.json').read_text())
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2_video_predictor
    ckpt='/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt'
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',ckpt,device='cuda')
    predictor.eval();print('SAM2_READY',flush=True)
    for camera in range(6):
        info=meta['camera_streams'][str(camera)]
        if info['first'] is None:continue
        start=time.monotonic();frames=out/'rgb'/f'cam{camera}'
        target=out/'masks'/f'cam{camera}';target.mkdir(parents=True,exist_ok=True)
        count=limit or meta['frames']
        if info['first']>=count:continue
        if all((target/f'{f:05d}.png').exists() for f in range(count)):continue
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(video_path=str(frames),offload_video_to_cpu=True,offload_state_to_cpu=True)
            first=info['first'];box=np.asarray(info['boxes'][first],np.float32)
            predictor.add_new_points_or_box(state,frame_idx=first,obj_id=1,box=box)
            masks={}
            for f,ids,logits in predictor.propagate_in_video(state,start_frame_idx=first,max_frame_num_to_track=count-first):
                if f>=count:break
                raw=(logits[0,0]>0).cpu().numpy().astype(np.uint8)
                b=info['boxes'][f]
                if b is None or (b[2]-b[0])*(b[3]-b[1])<400:raw[:]=0
                else:
                    gate=np.zeros_like(raw)
                    x0,y0,x1,y1=[int(round(x)) for x in b]
                    gate[max(0,y0-6):min(384,y1+7),max(0,x0-6):min(688,x1+7)]=1
                    raw &= gate
                masks[f]=raw
            predictor.reset_state(state)
        for f in range(count):
            raw=masks.get(f,np.zeros((384,688),np.uint8))
            Image.fromarray(raw*255,'L').save(target/f'{f:05d}.png')
        summary={'scene':scene_name,'camera':camera,'frames':count,
          'nonempty':sum(int(m.any()) for m in masks.values()),
          'areas':{str(f):int(m.sum()) for f,m in masks.items()},'elapsed_s':time.monotonic()-start,
          'GT_box_roles':'first prompt and per-frame gate','human_verdict':None}
        (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='areas'}),flush=True)
    del predictor;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','segment']);p.add_argument('--scene',choices=list(SCENES),required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();globals()[a.command](a.scene) if a.command=='prepare' else segment(a.scene,a.limit)
