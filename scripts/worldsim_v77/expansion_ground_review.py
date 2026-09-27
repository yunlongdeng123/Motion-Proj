from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from geometry import project_bbox
from delete_full_query import compose_actor,Writer
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r7';out=R/'review';out.mkdir(exist_ok=False);writer=Writer(out/'grounded.mp4',(688,384));frames=read(T/'r5/camera_frames.json');sheet=Image.new('RGB',(1200,2440),(20,30,40));d=ImageDraw.Draw(sheet);rows=[]
for f in range(10):
    world=T/'r2'/f'f{f:03}';bg=np.array(Image.open(world/'hybrid_cam0.png').convert('RGB'));z=np.load(world/'z_cam0.npy');c,k=camera(frames[f],0,(384,688));b=next(b for b in frames[f]['all_boxes'] if b['actor_id']=='12');bb=project_bbox(b['pose'],b['size_lwh'],c,k,(384,688));images=[np.array(Image.open(frames[f]['views'][0]['image']).convert('RGB').resize((688,384),Image.Resampling.BICUBIC))]
    for rid in ['r5','r7']:
        base=T/rid/'actor_layers';rgba=np.array(Image.open(base/f'f{f:03}_cam0.png').convert('RGBA'));az=np.load(base/f'f{f:03}_cam0_depth.npy');comp,stats=compose_actor(bg,rgba,az,z);images.append(comp)
        if rid=='r7':
            rows.append(dict(frame=f,**stats));pic=Image.fromarray(comp);dr=ImageDraw.Draw(pic);dr.rectangle(bb,outline='yellow',width=1);dr.text((4,4),f'official000 / target12 f{f} | r7 fixedZ -0.146m | diagnostic',fill='yellow');writer.write(pic)
    box=[max(0,int(bb[0])-18),max(0,int(bb[1])-18),min(688,int(bb[2])+19),min(384,int(bb[3])+19)]
    for j,im in enumerate(images):
        pic=Image.fromarray(im).crop(box);scale=min(400/pic.width,214/pic.height);pic=pic.resize((round(pic.width*scale),round(pic.height*scale)));sheet.paste(pic,(j*400,f*244+26));d.text((j*400+4,f*244+5),f'f{f} '+['Original','r5 GT height','r7 ground diagnostic'][j],fill='white')
writer.close();sheet.save(out/'contact.jpg',quality=95);dump(out/'summary.json',dict(task_id=T.name,run_id='r7',frames=10,delta_z_m=read(R/'registration.json')['delta_z_m'],rows=rows,operation='factual_ground_control_only',move_executed=False,human_verdict=None));print('GROUND_REVIEW_DONE')
