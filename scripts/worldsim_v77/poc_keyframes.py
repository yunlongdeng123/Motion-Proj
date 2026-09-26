"""从 SAM2 mask/清晰度/GT 对齐视角选最多 8 帧，产出透明 actor 参考图。"""
import json
import pathlib
import sys
import cv2
import numpy as np
from PIL import Image

BASE=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
for scene_name in ['scene_0230','scene_0255']:
    root=BASE/scene_name;meta=json.loads((root/'selection.json').read_text())
    data=pathlib.Path(meta['source']);ann=json.loads((data/'instances/instances_info.json').read_text())[meta['actor']]['frame_annotations']
    rows=[]
    for cam in range(6):
        if meta['camera_streams'][str(cam)]['first'] is None:continue
        for f in range(meta['frames']):
            path=root/'masks'/f'cam{cam}'/f'{f:05d}.png'
            mask=np.asarray(Image.open(path).convert('L'))>0
            area=int(mask.sum())
            if area<500 or f not in ann['frame_idx']:continue
            box=meta['camera_streams'][str(cam)]['boxes'][f]
            if box is None:continue
            box_area=(box[2]-box[0])*(box[3]-box[1]);fill=min(1,area/max(box_area,1))
            image=np.asarray(Image.open(root/'rgb'/f'cam{cam}'/f'{f:05d}.jpg').convert('RGB'))
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            lap=cv2.Laplacian(gray,cv2.CV_32F)
            sharp=float(np.var(lap[mask]))
            edge=int(mask[:8].sum()+mask[-8:].sum()+mask[:,:8].sum()+mask[:,-8:].sum())/area
            j=ann['frame_idx'].index(f);pose=np.asarray(ann['obj_to_world'][j])
            c2w=np.loadtxt(data/'extrinsics'/f'{f:03d}_{cam}.txt')
            local=np.linalg.inv(pose)@np.r_[c2w[:3,3],1.]
            azimuth=float(np.degrees(np.arctan2(local[1],local[0])))
            rows.append({'frame':f,'camera':cam,'area':area,'fill':round(fill,3),'sharpness':round(sharp,2),
              'edge_fraction':round(edge,3),'view_azimuth_deg':round(azimuth,1)})
    areas=np.array([r['area'] for r in rows]);sharp=np.array([r['sharpness'] for r in rows]);
    for r in rows:
        r['score']=float(np.sqrt(r['area']/max(np.median(areas),1))
          *np.sqrt(max(r['sharpness'],1)/max(np.median(sharp),1))
          *(.4+.6*r['fill'])*max(.1,1-2*r['edge_fraction']))
    choices=[]
    while len(choices)<8:
        candidates=[]
        for r in rows:
            if any(abs(r['frame']-q['frame'])<5 and r['camera']==q['camera'] for q in choices):continue
            if not choices:val=r['score']
            else:
                angles=[abs((r['view_azimuth_deg']-q['view_azimuth_deg']+180)%360-180) for q in choices]
                val=r['score']*(.3+.7*min(90,min(angles))/90)
            candidates.append((val,r))
        if not candidates:break
        choices.append(max(candidates,key=lambda z:z[0])[1])
    out=root/'actor';out.mkdir(exist_ok=True)
    for rank,r in enumerate(choices):
        cam,f=r['camera'],r['frame']
        image=np.asarray(Image.open(root/'rgb'/f'cam{cam}'/f'{f:05d}.jpg').convert('RGB'))
        mask=np.asarray(Image.open(root/'masks'/f'cam{cam}'/f'{f:05d}.png').convert('L'))
        ys,xs=np.nonzero(mask)
        lo=np.array([xs.min(),ys.min()]);hi=np.array([xs.max()+1,ys.max()+1]);center=(lo+hi)/2
        side=int(max(hi-lo)*1.25);side=max(side,64)
        rgba=np.zeros((side,side,4),dtype=np.uint8)
        x0=int(round(center[0]-side/2));y0=int(round(center[1]-side/2))
        sx0=max(0,x0);sy0=max(0,y0);sx1=min(688,x0+side);sy1=min(384,y0+side)
        tx0=sx0-x0;ty0=sy0-y0
        rgba[ty0:ty0+sy1-sy0,tx0:tx0+sx1-sx0,:3]=image[sy0:sy1,sx0:sx1]
        rgba[ty0:ty0+sy1-sy0,tx0:tx0+sx1-sx0,3]=mask[sy0:sy1,sx0:sx1]
        path=out/f'key_{rank:02d}_f{f:03d}_cam{cam}.png'
        Image.fromarray(rgba,'RGBA').resize((512,512),Image.Resampling.LANCZOS).save(path)
        r['crop_path']=str(path)
    shape_candidates=[r for r in choices if r['edge_fraction']<.02 and r['fill']>=.55]
    shape_choice=max(shape_candidates or choices,key=lambda r:r['area'])
    manifest={'scene':scene_name,'actor':meta['actor'],'selected':choices,'shape_input':shape_choice['crop_path'],
      'shape_input_rule':'已选 keyframe 中：图像边界 mask 占比<2%、mask/GT 投影框面积>=55%，取 mask 面积最大；若无满足者退回面积最大',
      'selection_roles':'SAM2 area/fill, masked Laplacian sharpness, GT-aligned viewpoint for diversity; no visual-outcome tuning',
      'GT_input_role':'actor pose and camera c2w for angle; GT is POC aid',
      'omega_note':'Ω per-frame gauge is independent; cannot compare predicted camera azimuth across frames directly',
      'human_verdict':None}
    (out/'keyframes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'scene':scene_name,'choice':[(r['frame'],r['camera'],r['area'],r['view_azimuth_deg'],round(r['score'],2)) for r in choices]},ensure_ascii=False))
