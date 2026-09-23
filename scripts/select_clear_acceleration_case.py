"""按CPU投影/运动筛一个更易辨认的非ego加速输入，不读取生成视频。"""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motion_proj.cfbench.geometry import box_corners_world
from prepare_omnidreams_cfbench import pose_at

ROOT=Path('/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval')
OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/replace-speed-actor02')
OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads(Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/20260923-proposal-r2-10s/candidates.json').read_text())
used={(r['case']['dataset']['scene_id'],r['case']['target'].get('actor_key',r['case']['target'].get('source_asset_actor_key'))) for r in manifest['cases']}

def bbox(cam,K,pose,size):
    if pose is None:return None
    xyz=cam@box_corners_world(pose,size)
    if xyz[2].min()<=.1:return None
    fx,fy,cx,cy=K[:4]
    x=fx*xyz[0]/xyz[2]+cx; y=fy*xyz[1]/xyz[2]+cy
    box=np.array([max(0,x.min()),max(0,y.min()),min(1600,x.max()),min(900,y.max())])
    if box[2]<=box[0] or box[3]<=box[1]:return None
    return box,float(np.prod(box[2:]-box[:2])),float(np.mean(xyz[2]))

rows=[]
for scene in ['179','191','204']:
    root=ROOT/scene
    info=json.loads((root/'instances/instances_info.json').read_text())
    tracks={key:{int(f):np.array(p) for f,p in zip(r['frame_annotations']['frame_idx'],r['frame_annotations']['obj_to_world'])} for key,r in info.items()}
    sizes={key:np.median(r['frame_annotations']['box_size'],axis=0) for key,r in info.items()}
    for start in [0,20,40]:
        event=start+5; samples=[start,start+5,start+20,start+40,start+60,start+80,start+99]
        for camera in range(6):
            K=np.loadtxt(root/f'intrinsics/{camera}.txt')
            cams={f:np.linalg.inv(np.loadtxt(root/f'extrinsics/{f:03d}_{camera}.txt')) for f in samples}
            boxes={f:{key:bbox(cams[f],K,pose_at(poses,f),sizes[key]) for key,poses in tracks.items()} for f in samples}
            for key,r in info.items():
                if (scene,key)==('191','7') or not r['class_name'].startswith('vehicle.') or r['class_name'] in ['vehicle.bicycle','vehicle.motorcycle']:continue
                poses=tracks[key]
                end=pose_at(poses,event+(start+100-event)*1.5)
                factual_end=pose_at(poses,start+100)
                if end is None or factual_end is None:continue
                delta=float(np.linalg.norm(end[:3,3]-factual_end[:3,3]))
                if delta<5:continue
                areas=[]; overlaps=[]; distractions=[]; cf_visible=0
                for f in samples:
                    target=boxes[f][key]
                    q=f if f<event else event+(f-event)*1.5
                    edited=bbox(cams[f],K,pose_at(poses,q),sizes[key])
                    if edited is not None and edited[1]>=1200:cf_visible+=1
                    if target is None:continue
                    b,area,z=target
                    if area<1200:continue
                    areas.append(area)
                    overlap=0; competing=0
                    for other,ob in boxes[f].items():
                        if other==key or ob is None:continue
                        o,oa,oz=ob
                        inter=np.prod(np.maximum(0,np.minimum(b[2:],o[2:])-np.maximum(b[:2],o[:2])))
                        if oz<z:overlap=max(overlap,float(inter/area))
                        if info[other]['class_name']==r['class_name'] and oa>=area*.5:competing+=1
                    overlaps.append(overlap); distractions.append(competing)
                if len(areas)<6 or cf_visible<5 or max(overlaps)>.15:continue
                if boxes[event][key] is None or boxes[event][key][1]<2500:continue
                rows.append({'scene':scene,'camera':camera,'actor_key':key,'class_name':r['class_name'],
                             'entity_id':r.get('id',r.get('instance_id')), 'start':start,'event':event,
                             'min_area':min(areas),'median_area':float(np.median(areas)),
                             'max_nearer_overlap':max(overlaps),'same_class_large_count_max':max(distractions),
                             'same_class_large_count_mean':float(np.mean(distractions)),'endpoint_delta_m':delta})
rows.sort(key=lambda r:(r['same_class_large_count_max'],r['same_class_large_count_mean'],-r['min_area']))
(OUT/'selection.json').write_text(json.dumps(rows,indent=2)+'\n')
for n,r in enumerate(rows[:3]):
    root=ROOT/r['scene']; info=json.loads((root/'instances/instances_info.json').read_text()); ann=info[r['actor_key']]['frame_annotations']
    poses={int(f):np.array(p) for f,p in zip(ann['frame_idx'],ann['obj_to_world'])}; size=np.median(ann['box_size'],axis=0)
    im=Image.open(root/f"images/{r['event']:03d}_{r['camera']}.jpg").convert('RGB')
    b=bbox(np.linalg.inv(np.loadtxt(root/f"extrinsics/{r['event']:03d}_{r['camera']}.txt")),np.loadtxt(root/f"intrinsics/{r['camera']}.txt"),poses[r['event']],size)[0]
    draw=ImageDraw.Draw(im); draw.rectangle(b.tolist(),outline='yellow',width=6)
    draw.text((20,20),f"{r['scene']} cam {r['camera']} actor {r['actor_key']} start {r['start']}",fill='yellow')
    im.save(OUT/f'candidate-{n+1}.jpg',quality=90)
print(json.dumps({'count':len(rows),'top':rows[:5]},ensure_ascii=False))
