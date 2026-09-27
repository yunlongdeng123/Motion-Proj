"""r30 B_t接回新目标资产，保存失败诊断；不执行未满足前提的MOVE。"""
from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from geometry import project_bbox
from delete_full_query import Writer,compose_actor
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r5';out=R/'review_marked';out.mkdir(exist_ok=False);keys=['original','background','no_occlusion','factual','pure_geometry'];writers={k:Writer(out/f'{k}.mp4',(688,384)) for k in keys};sheet=Image.new('RGB',(1720,218*10),(20,30,40));d=ImageDraw.Draw(sheet);rows=[];frames=read(R/'camera_frames.json');crop_sheet=Image.new('RGB',(1200,244*10),(20,30,40));cd=ImageDraw.Draw(crop_sheet)
for f in range(10):
    w=T/'r2'/f'f{f:03}';bg=np.array(Image.open(w/'hybrid_cam0.png').convert('RGB'));pure=np.array(Image.open(w/'geometry_cam0.png').convert('RGB'));z=np.load(w/'z_cam0.npy');rgba=np.array(Image.open(R/'actor_layers'/f'f{f:03}_cam0.png').convert('RGBA'));az=np.load(R/'actor_layers'/f'f{f:03}_cam0_depth.npy');unocc,_=compose_actor(bg,rgba,az,np.full(z.shape,np.inf));factual,stats=compose_actor(bg,rgba,az,z);geo,_=compose_actor(pure,rgba,az,z);rows.append(dict(frame=f,**stats));original=np.array(Image.open(frames[f]['views'][0]['image']).convert('RGB').resize((688,384),Image.Resampling.BICUBIC));images=[original,bg,unocc,factual,geo];c,k=camera(frames[f],0,hw=(384,688));actor=next(b for b in frames[f]['all_boxes'] if b['actor_id']=='12');bb=project_bbox(actor['pose'],actor['size_lwh'],c,k,(384,688))
    for j,(k,im) in enumerate(zip(keys,images)):
        pic=Image.fromarray(im);pd=ImageDraw.Draw(pic);pd.text((5,5),f'official000 target12 f{f} | {k} | FAILED APPEARANCE CONTROL',fill='yellow');pd.rectangle(bb,outline='yellow',width=1);writers[k].write(pic);sheet.paste(pic.resize((344,192)),(j*344,f*218+26));d.text((j*344+4,f*218+5),f'f{f} {k}',fill='white')
    box=[max(0,int(bb[0])-16),max(0,int(bb[1])-16),min(688,int(bb[2])+17),min(384,int(bb[3])+17)]
    for j,im in enumerate([original,bg,factual]):
        pic=Image.fromarray(im).crop(box);factor=min(400/pic.width,214/pic.height);pic=pic.resize((round(pic.width*factor),round(pic.height*factor)));crop_sheet.paste(pic,(j*400,f*244+26));cd.text((j*400+4,f*244+5),f'f{f} '+['original','r30+Omega B_t','factual (appearance fail)'][j],fill='white')
for w in writers.values():w.close()
sheet.save(out/'all10.jpg',quality=95);crop_sheet.save(out/'crops.jpg',quality=95);dump(out/'summary.json',dict(task_id=T.name,run_id='r5',frames=10,rows=rows,video_frames={k:w.count for k,w in writers.items()},asset='New target12 own Hunyuan2.1 asset; real f46/CAM2 reference; unseen sides are prior; all old assets preserved',assistant_verdict='appearance_not_accepted: green side/front mottling and weak wheels; yaw0 direction verified with observed-source0/180 control; no MOVE executed',human_verdict=None))
print('FACTUAL_REVIEW',rows)
