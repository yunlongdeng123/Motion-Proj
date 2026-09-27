"""固定表面的原视频/目标/原生/写回视频；不把留出失败筛掉。"""
from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from delete_full_query import Writer
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r1';reg=read(R/'registration.json');summ=[]
for s in reg['new_scenes']:
    c=s['primary_camera'];base=R/s['name']/f'cam{c}';done=(base/'drive/composite/00029.png').exists();out=base/'review'
    if out.exists():
        if (out/'summary.json').exists():summ.append(read(out/'summary.json'))
        continue
    if not done and s['name']!='processed_756':continue
    out.mkdir();keys=['original','target','scope']+(['native','delete'] if done else []);writers={k:Writer(out/f'{k}.mp4',(1024,576)) for k in keys};crops=[];full=[]
    stream=next(v for v in s['streams'] if v['camera']==c);boxes=np.array([v for v in stream['boxes'] if v is not None]);bb=[max(0,int(boxes[:,0].min())-70),max(0,int(boxes[:,1].min())-55),min(1024,int(boxes[:,2].max())+70),min(576,int(boxes[:,3].max())+60)]
    for f in range(30):
        im=np.array(Image.open(base/'rgb'/f'{f:05}.png').convert('RGB'));m=np.array(Image.open(base/'write_mask'/f'{f:05}.png'))>0;model=np.array(Image.open(base/'model_mask'/f'{f:05}.png'))>0;p=np.array(Image.open(base/'protect'/f'{f:05}.png'))>0
        target=im.copy();target[m]=(target[m]*.4+np.array([255,195,20])*.6).astype('uint8');target=Image.fromarray(target);draw=ImageDraw.Draw(target)
        if stream['boxes'][f] is not None:draw.rectangle(stream['boxes'][f],outline='yellow',width=2)
        scope=im.copy();scope[model]=(scope[model]*.5+np.array([50,150,255])*.5).astype('uint8');scope[p]=[20,220,180]
        images=dict(original=Image.fromarray(im),target=target,scope=Image.fromarray(scope))
        if done:
            images['native']=Image.open(base/'drive/native'/f'{f:05}.png').convert('RGB');images['delete']=Image.open(base/'drive/composite'/f'{f:05}.png').convert('RGB')
        for k,pic in images.items():
            q=pic.copy();ImageDraw.Draw(q).text((5,5),f"{s['name']} target{s['actor']} CAM{c} f{f} {s['split']} | {k}",fill='yellow');writers[k].write(q)
        selected=['target','native','delete'] if done else ['original','target','scope']
        if f in [0,14,29]:
            panel=Image.new('RGB',(1536,314),(20,30,40));d=ImageDraw.Draw(panel)
            for j,k in enumerate(selected):panel.paste(images[k].resize((512,288)),(j*512,26));d.text((j*512+4,5),f'{s["name"]} f{f} | {k}',fill='white')
            full.append(panel)
        if f in list(np.linspace(0,29,10).round().astype(int)):
            panel=Image.new('RGB',(1200,276),(20,30,40));d=ImageDraw.Draw(panel)
            for j,k in enumerate(selected):
                crop=images[k].crop(bb);crop.thumbnail((400,250));panel.paste(crop,(j*400,26));d.text((j*400+4,5),f'f{f} {k}',fill='white')
            crops.append(panel)
    for w in writers.values():w.close()
    for name,tiles in [('full',full),('crops',crops)]:
        sheet=Image.new('RGB',(tiles[0].width,sum(t.height for t in tiles)))
        y=0
        for pic in tiles:sheet.paste(pic,(0,y));y+=pic.height
        sheet.save(out/f'{name}.jpg',quality=95)
    row=dict(scene=s['name'],actor=s['actor'],camera=c,split=s['split'],generation_complete=done,frames=30,video_frames={k:v.count for k,v in writers.items()},crop_box=bb,assistant_visual_review='pending',human_verdict=None)
    dump(out/'summary.json',row);summ.append(row)
dump(R/'review_index.json',dict(scenes=summ,registered=6,generated=sum(s['generation_complete'] for s in summ),human_verdict=None));print('REVIEW_STREAMS',[(s['scene'],s['generation_complete']) for s in summ])
