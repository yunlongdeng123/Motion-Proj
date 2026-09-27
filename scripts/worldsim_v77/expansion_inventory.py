"""生成前只核对真实文件、标定和目标候选；不依据模型结果选scene。"""
from pathlib import Path
import sys,json,subprocess,datetime
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from geometry import project_bbox,resized_intrinsics
from video_review import scene_frame
from repair_common import dump,read

ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r1')
assert not ROOT.exists();ROOT.mkdir(parents=True)
files=subprocess.check_output(['rg','--files','/root/autodl-tmp/data'],text=True).splitlines()
roots=sorted({Path(p).parent.parent for p in files if p.endswith('/instances/instances_info.json')})
roots += [Path('/root/autodl-tmp/data/v76_vadgs')/s for s in ['scene_0230','scene_0255']]
rows=[];candidates=[];thumbs=[]
for root in roots:
    ims=sorted((root/'images').glob('*_0.jpg'))
    row=dict(root=str(root),name=root.name,front_frames=len(ims))
    rows.append(row)
    if len(ims)<30:continue
    inst=read(root/'instances/instances_info.json');views=[]
    for c in range(6):
        raw=np.loadtxt(root/'intrinsics'/f'{c}.txt');k=[[raw[0],0,raw[2]],[0,raw[1],raw[3]],[0,0,1]]
        views.append(dict(camera=c,intrinsics=k,original_wh=list(Image.open(root/'images'/f'000_{c}.jpg').size)))
    spec=dict(name=root.name,root=str(root),views=views)
    complete=all((root/'images'/f'{f:03}_{c}.jpg').exists() and (root/'extrinsics'/f'{f:03}_{c}.txt').exists() for f in range(30) for c in range(6)) and all((root/'lidar'/f'{f:03}.bin').exists() and (root/'lidar_pose'/f'{f:03}.txt').exists() for f in range(30))
    row['complete_first30_sixcams']=complete
    if not complete:continue
    for aid,a in inst.items():
        if a['class_name']!='vehicle.car' or not set(range(30)).issubset(a['frame_annotations']['frame_idx']):continue
        rs=[]
        for f in [0,14,29]:
            fr=scene_frame(spec,f,inst);b=next(b for b in fr['all_boxes'] if b['actor_id']==aid)
            rr=[]
            for c,v in enumerate(fr['views']):
                bb=project_bbox(b['pose'],b['size_lwh'],np.array(v['c2w']),resized_intrinsics(v['intrinsics'],v['original_wh'],[576,1024]),[576,1024])
                rr.append(dict(camera=c,box=bb,area=(bb[2]-bb[0])*(bb[3]-bb[1]) if bb else 0))
            rs.append(rr)
        # 固定首窗连续可见、适中尺度，防止用裁边/全画面物体强做小POC。
        areas=np.array([[x['area'] for x in rr] for rr in rs]);primary=int(np.argmax(areas.min(0)))
        if areas[:,primary].min()<1500 or areas[:,primary].max()>70000:continue
        boxes=[rr[primary]['box'] for rr in rs]
        if any(b[0]<8 or b[1]<8 or b[2]>1016 or b[3]>568 for b in boxes):continue
        # 首帧同相机非目标投影交叠只作选型信息，不当精确遮挡/合法性。
        fr=scene_frame(spec,0,inst);v=fr['views'][primary];bb=boxes[0];overlap=0
        for b in fr['all_boxes']:
            if b['actor_id']==aid:continue
            ob=project_bbox(b['pose'],b['size_lwh'],np.array(v['c2w']),resized_intrinsics(v['intrinsics'],v['original_wh'],[576,1024]),[576,1024])
            if ob:overlap+=max(0,min(bb[2],ob[2])-max(bb[0],ob[0]))*max(0,min(bb[3],ob[3])-max(bb[1],ob[1]))
        candidates.append(dict(root=str(root),scene=root.name,actor=aid,track_id=a['id'],camera=primary,areas=areas[:,primary].tolist(),boxes=boxes,overlap_fraction=min(1,float(overlap/areas[0,primary])),spec=spec,other_views=[c for c in range(6) if c!=primary and areas[:,c].max()>100]))
    top=sorted([c for c in candidates if c['root']==str(root)],key=lambda c:(c['overlap_fraction'], -min(c['areas']),int(c['actor'])))[:2]
    row['candidates']=len([c for c in candidates if c['root']==str(root)])
    row['track_ids']=[a['id'] for a in inst.values()]
    for c in top:
        im=Image.open(root/'images'/f"000_{c['camera']}.jpg").convert('RGB').resize((512,288));d=ImageDraw.Draw(im);b=np.array(c['boxes'][0])*.5;d.rectangle(b.tolist(),outline='yellow',width=3);tile=Image.new('RGB',(512,322),(20,30,40));tile.paste(im,(0,34));ImageDraw.Draw(tile).text((5,5),f"{root.parent.parent.parent.name}/{root.name} actor{c['actor']} CAM{c['camera']} overlap {c['overlap_fraction']:.2f}",fill='white');thumbs.append(tile)
sheet=Image.new('RGB',(1536,322*((len(thumbs)+2)//3)),(20,30,40))
for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*512,(i//3)*322))
sheet.save(ROOT/'inventory.jpg',quality=93)
dump(ROOT/'inventory.json',dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,candidates=candidates,selection_roles='File and source geometry screening before generation; no split selected yet; scene aliases and source history must be checked.'))
print(json.dumps(dict(roots=len(rows),complete=[{k:r.get(k) for k in ['root','front_frames','candidates']} for r in rows if r.get('complete_first30_sixcams')],candidates=len(candidates)),ensure_ascii=False),flush=True)
