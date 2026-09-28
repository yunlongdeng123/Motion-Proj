"""从正在落盘的本轮输出生成只读阶段预览，不改推理/写回。"""
from nine_common import *
import argparse,numpy as np
from PIL import Image,ImageDraw
from geometry import project_bbox
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--camera',type=int,required=True);a=p.parse_args()
s=next(s for s in read(ROOT/'registration.json')['scenes'] if s['name']==a.scene);base=ROOT/a.scene;stream=base/f'cam{a.camera}';frames=read(base/'camera_frames.json');ids=[f for f in [0,9,14,18,27,29] if (stream/'background'/f'{f:05}.png').exists() and (stream/'native'/f'{f:05}.png').exists()];assert ids
out=ROOT/'drive_preview';out.mkdir(exist_ok=True);path=out/f'{a.scene}_cam{a.camera}_through{max(ids):02}.jpg'
if not path.exists():
    panels=[]
    for f in ids:
        rgb=np.array(Image.open(stream/'rgb'/f'{f:05}.png').convert('RGB'));m=np.array(Image.open(stream/'model_mask'/f'{f:05}.png'))>0;wr=np.array(Image.open(stream/'write_mask'/f'{f:05}.png'))>0;scope=rgb.copy();scope[m]=(.5*scope[m]+.5*np.array([60,160,255])).astype('uint8');scope[wr]=(.5*scope[wr]+.5*np.array([255,190,0])).astype('uint8')
        actor=next(b for b in frames[f]['all_boxes'] if b['actor_id']==s['actor']);c,k=camera(frames[f],a.camera);bb=project_bbox(actor['pose'],actor['size_lwh'],c,k,(576,1024));marked=Image.fromarray(rgb)
        if bb:ImageDraw.Draw(marked).rectangle(bb,outline='yellow',width=2)
        else:bb=(0,0,1024,576)
        imgs=[marked,Image.fromarray(scope),Image.open(stream/'native'/f'{f:05}.png'),Image.open(stream/'background'/f'{f:05}.png')]
        panel=Image.new('RGB',(1376,424),(20,30,40));d=ImageDraw.Draw(panel)
        # 上全景下局部；同一时点四列使用同一个ROI，局部仅用于看残影/边缘。
        x0=max(0,int(bb[0])-24);x1=min(1024,int(bb[2])+24);y0=max(0,int(bb[1])-24);y1=min(576,int(bb[3])+24)
        for j,im in enumerate(imgs):
            panel.paste(im.resize((344,194)),(j*344,24));crop=im.crop((x0,y0,x1,y1));crop.thumbnail((344,194));panel.paste(crop,(j*344+(344-crop.width)//2,230));d.text((j*344+4,4),f'f{f:02} CAM{a.camera} '+['source','mask','native','writeback'][j],fill='white')
        panels.append(panel)
    sheet=Image.new('RGB',(1376,424*len(panels)))
    for j,im in enumerate(panels):sheet.paste(im,(0,j*424))
    sheet.save(path,quality=94)
print(path,flush=True)
