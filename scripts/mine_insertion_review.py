"""CPU-only shortlist: visible donor and inserted car, road-valid, no overlap."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import box, Polygon

sys.path.insert(0, '/root/autodl-tmp/motion_proj/scripts')
sys.path.insert(0, '/root/autodl-tmp/motion_proj')
from mine_lateral_review import V5, OLD, local_road, footprint
from motion_proj.cfbench.geometry import project_box, shifted_pose

OPTIONS = [
    ('425','1',5,90,'boston-seaport',V5),
    ('350','7',5,40,'singapore-queenstown',V5),
    ('350','0',0,80,'singapore-queenstown',V5),
    ('350','6',5,0,'singapore-queenstown',V5),
    ('756','14',0,60,'singapore-queenstown',V5),
    ('756','5',0,90,'singapore-queenstown',V5),
    ('756','12',2,40,'singapore-queenstown',V5),
    ('204','49',0,40,'boston-seaport',OLD),
    ('276','24',5,70,'singapore-onenorth',V5),
]
OFFSETS = [(x,y,0) for x in (-12,-8,-4,8,12) for y in (-3.5,0,3.5)]


def egoarea(pose):
    return Polygon([tuple((pose@np.array([x,y,0,1]))[:2])
                    for x,y in [(-1,-2.25),(1,-2.25),(1,2.25),(-1,2.25)]])


def iou(a,b):
    l=max(a[0],b[0]);t=max(a[1],b[1]);r=min(a[2],b[2]);d=min(a[3],b[3])
    inter=max(0,r-l)*max(0,d-t)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter else 0


def evaluate(option):
    scene,key,camera,start,location,base=option
    root=base/scene
    info=json.loads((root/'instances/instances_info.json').read_text())
    ann=info[key]['frame_annotations']
    poses=dict(zip(ann['frame_idx'],map(np.asarray,ann['obj_to_world'])))
    sizes=dict(zip(ann['frame_idx'],ann['box_size']))
    frames=list(range(start+5,start+100,5))
    if any(f not in poses for f in frames):return []
    path=np.array([poses[f][:2,3] for f in frames])
    region=box(path[:,0].min()-25,path[:,1].min()-25,path[:,0].max()+25,path[:,1].max()+25)
    road=local_road(location,region)
    other={}
    for f in frames:
        objs=[]
        for other_key,item in info.items():
            aa=item['frame_annotations'];fs=aa['frame_idx']
            if f not in fs:continue
            j=fs.index(f)
            objs.append((other_key,footprint(np.asarray(aa['obj_to_world'][j]),aa['box_size'][j])))
        other[f]=objs
    ego={f:egoarea(np.loadtxt(root/f'lidar_pose/{f:03d}.txt')) for f in frames}
    results=[]
    for offset in OFFSETS:
        inroad=visible=collisions=egohits=0
        best=(0,None,None,None)
        for f in frames:
            pose=shifted_pose(poses[f],offset)
            area=footprint(pose,sizes[f])
            inroad+=road.covers(area)
            collisions+=sum(area.intersects(x) for _,x in other[f])
            egohits+=area.intersects(ego[f])
            source=project_box(root,f,camera,poses[f],sizes[f])
            target=project_box(root,f,camera,pose,sizes[f])
            if source and target:
                visible+=1
                s=source['bbox_xyxy'];t=target['bbox_xyxy']
                sarea=(s[2]-s[0])*(s[3]-s[1]);tarea=(t[2]-t[0])*(t[3]-t[1])
                overlap=iou(s,t)
                edge=min(s[0],t[0],1600-s[2],1600-t[2],s[1],t[1],900-s[3],900-t[3])
                quality=min(sarea,tarea)*(1-overlap)
                if edge>=20 and quality>best[0]:best=(quality,f,round(sarea),round(tarea))
        if inroad==len(frames) and collisions==0 and egohits==0 and visible>=15 and best[0]>=10000:
            results.append({'scene':scene,'key':key,'camera':camera,'start':start,'offset':offset,
                            'road':inroad,'visible':visible,'annotated_actor_intersections':collisions,
                            'ego_footprint_intersections':egohits,'reference_frame':best[1],
                            'min_reference_area':min(best[2],best[3]),'quality':round(best[0]),
                            'donor_class':info[key]['class_name']})
    return sorted(results,key=lambda x:-x['quality'])


if __name__=='__main__':
    for option in OPTIONS:
        rows=evaluate(option)
        print(json.dumps({'option':option[:5],'qualified':len(rows),'top':rows[:12]}),flush=True)
