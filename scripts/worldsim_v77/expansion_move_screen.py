"""MOVE只筛选固定四个0.5m位移；未通过factual前不渲染或宣称合法。"""
from pathlib import Path
import sys,datetime,itertools,numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import Polygon
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from video_review import scene_frame
from geometry import transform,project_bbox,box_mask
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r6';R.mkdir(exist_ok=False);spec=read(T/'r3/frames.json')[0];data=Path(spec['root']);inst=read(data/'instances/instances_info.json');frames=[scene_frame(spec,f,inst) for f in range(30)]
def foot(p,s):
    return Polygon(transform(np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*np.array(s)/2,p)[:,:2])
actor0=next(b for b in frames[0]['all_boxes'] if b['actor_id']=='12');yaw=np.array(actor0['pose'])[:3,:3];rows=[];panels=[]
for axis,sgn in itertools.product([0,1],[-1,1]):
    delta=yaw[:,axis]*sgn*.5;delta[2]=0;gtgap=1e9;egogap=1e9;maxstatic=0;collisions=[]
    for fr in frames:
        actor=next(b for b in fr['all_boxes'] if b['actor_id']=='12');original=np.array(actor['pose']);size=actor['size_lwh'];other=[b for b in fr['all_boxes'] if b['actor_id']!='12'];ec=np.mean([np.array(v['c2w'])[:3,3] for v in fr['views']],0);ep=np.eye(4);fc=np.array(fr['views'][0]['c2w'])[:3,2];fc[2]=0;fc/=np.linalg.norm(fc);ep[:3,:3]=np.column_stack([fc,[-fc[1],fc[0],0],[0,0,1]]);ep[:3,3]=ec
        pts=transform(np.fromfile(data/'lidar'/f'{fr["frame"]:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(data/'lidar_pose'/f'{fr["frame"]:03}.txt'));keep=np.ones(len(pts),bool)
        for b in fr['all_boxes']:keep &= ~box_mask(pts,np.array(b['pose']),b['size_lwh'])
        for q in np.linspace(0,1,11):
            pose=original.copy();pose[:3,3]+=q*delta;poly=foot(pose,size);gap=min(poly.distance(foot(b['pose'],b['size_lwh'])) for b in other);gtgap=min(gtgap,gap);egogap=min(egogap,poly.distance(foot(ep,[5,2.4,2])))
            if gap==0:collisions.append(dict(frame=fr['frame'],fraction=float(q)))
            cp=transform(pts[keep],np.linalg.inv(pose));inside=(np.abs(cp[:,0])<size[0]/2)&(np.abs(cp[:,1])<size[1]/2)&(cp[:,2]>-size[2]/2+.3)&(cp[:,2]<size[2]/2);maxstatic=max(maxstatic,int(inside.sum()))
    row=dict(axis=axis,sign=sgn,delta_world=delta.tolist(),min_gt_gap=gtgap,min_ego_gap=egogap,max_static_points=maxstatic,collision_samples=collisions,geometric_candidate=bool(gtgap>=.5 and egogap>=.75 and maxstatic<=8));rows.append(row)
    for f in [0,14,29]:
        fr=frames[f];a=next(b for b in fr['all_boxes'] if b['actor_id']=='12');pose=np.array(a['pose']);pose[:3,3]+=delta;c,k=camera(fr,0);im=Image.open(fr['views'][0]['image']).convert('RGB').resize((768,432));d=ImageDraw.Draw(im)
        for p,color in [(a['pose'],'yellow'),(pose,'magenta')]:
            bb=project_bbox(p,a['size_lwh'],c,k,(576,1024))
            if bb:d.rectangle((np.array(bb)*.75).tolist(),outline=color,width=3)
        panel=Image.new('RGB',(768,458),(20,30,40));panel.paste(im,(0,26));ImageDraw.Draw(panel).text((4,5),f'axis{axis} sign{sgn} f{f} gt{gtgap:.2f} ego{egogap:.2f} static{maxstatic}',fill='white');panels.append(panel)
sheet=Image.new('RGB',(768*3,458*4))
for i,p in enumerate(panels):sheet.paste(p,((i%3)*768,(i//3)*458))
sheet.save(R/'candidates.jpg',quality=94)
dump(R/'registration_results.json',dict(task_id=T.name,run_id='r6',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),operation='MOVE_SCREEN_ONLY',actor='official_000/12',magnitude_m=.5,sweep_samples=11,frames=30,candidates=rows,prerequisite='Target12 r5 factual asset not yet visually accepted; no MOVE rendering. Geometric candidate does not certify roadway or dynamic response.',human_verdict=None,failure_ledger_refs=['V77-F02']));print('MOVE_SCREEN',[(r['axis'],r['sign'],r['min_gt_gap'],r['min_ego_gap'],r['max_static_points'],r['geometric_candidate']) for r in rows])
