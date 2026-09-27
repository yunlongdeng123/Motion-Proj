"""空位置INSERT只做几何筛选及源图标注，最终需查看图像，不以框零碰撞代道路合法。"""
from pathlib import Path
import sys,datetime,itertools
import numpy as np,cv2
from PIL import Image,ImageDraw
from shapely.geometry import Polygon
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera
from geometry import transform,box_mask,project_bbox
from video_review import scene_frame
from hybrid_road_height_prefix_control import local_ground
BASE=Path('/root/autodl-tmp/runs/worldsim_v77');ROOT=BASE/'WS-V77-EXPAND-EDIT-20260928/r3';assert not ROOT.exists();ROOT.mkdir()
old=read(BASE/'WS-V77-P0-24ACTOR-20260926/r1/registration.json');spec=next(s for s in old['scenes'] if s['name']=='official_000');inst=read(Path(spec['root'])/'instances/instances_info.json');frames=[scene_frame(spec,f,inst) for f in range(30)];source=read(BASE/'WS-V77-DELETE-FULL-20260927/r1/scene_0255/camera_frames.json')[0];actor=next(b for b in source['all_boxes'] if b['actor_id']=='25');size=np.array(actor['size_lwh']);g,n,d=local_ground(0);lidars=[]
def foot(p,s,margin=0):
    corners=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*(np.array(s)+[2*margin,2*margin,0])/2
    return Polygon(transform(corners,p)[:,:2])
for fr in frames:
    f=fr['frame'];pts=transform(np.fromfile(Path(spec['root'])/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(Path(spec['root'])/'lidar_pose'/f'{f:03}.txt'));valid=np.ones(len(pts),bool)
    for b in fr['all_boxes']:valid &= ~box_mask(pts,np.array(b['pose']),b['size_lwh'])
    lidars.append(pts[valid])
target=next(b for b in frames[0]['all_boxes'] if b['actor_id']=='12');candidates=[]
for camera_id in [0,5,2,4,1,3]:
    c,k=camera(frames[0],camera_id)
    for u,v in itertools.product([220,380,512,644,804],[390,440,490]):
        ray=c[:3,:3]@np.linalg.solve(k,np.array([u,v,1.]));den=n@ray
        if abs(den)<1e-6:continue
        t=-(n@c[:3,3]+d)/den
        if not 4<t<25:continue
        ground=c[:3,3]+t*ray;pose=np.array(target['pose']);pose[:3,3]=ground+[0,0,size[2]/2];fp=foot(pose,size);clear=1e9;occupied=[];ego_clear=1e9
        for fr,pts in zip(frames,lidars):
            for b in fr['all_boxes']:clear=min(clear,fp.distance(foot(np.array(b['pose']),b['size_lwh'])))
            # ego近似包络仅为保守检查；没有处理交通规则/响应。
            ec=np.mean([np.array(w['c2w'])[:3,3] for w in fr['views']],0);ep=np.array(fr['views'][0]['c2w']);forward=ep[:3,2].copy();forward[2]=0;forward/=np.linalg.norm(forward);ep[:3,:3]=np.column_stack([forward,[-forward[1],forward[0],0],[0,0,1]]);ep[:3,3]=ec;ego_clear=min(ego_clear,fp.distance(foot(ep,[5,2.4,2])))
            local=transform(pts,np.linalg.inv(pose));inside=(np.abs(local[:,0])<size[0]/2+.1)&(np.abs(local[:,1])<size[1]/2+.1)&(local[:,2]>-size[2]/2+.3)&(local[:,2]<size[2]/2+.2);occupied.append(int(inside.sum()))
        if clear<.8 or ego_clear<1 or max(occupied)>8:continue
        support=lidars[0];near=np.linalg.norm(support[:,:2]-ground[:2],axis=1)<2;ng=int((near&(np.abs(support@n+d)<.12)).sum())
        if ng<8:continue
        areas=[]
        for camid in range(6):
            cc,kk=camera(frames[0],camid);bb=project_bbox(pose,size,cc,kk,(576,1024));areas.append((bb[2]-bb[0])*(bb[3]-bb[1]) if bb else 0)
        if max(areas)<1000:continue
        candidates.append(dict(proposal=len(candidates),sample_camera=camera_id,pixel=[u,v],pose_world=pose.tolist(),size_lwh=size.tolist(),min_gt_gap_m=clear,min_ego_gap_m=ego_clear,max_static_points=max(occupied),local_road_support=ng,areas=areas))
candidates.sort(key=lambda r:(-sum(a>500 for a in r['areas']),-r['min_gt_gap_m'],-max(r['areas'])))
dump(ROOT/'proposal_registration.json',dict(task_id='WS-V77-EXPAND-EDIT-20260928',run_id='r3',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),operation='INSERT',asset=str(BASE/'WS-V77-DELETE-FULL-20260927/r1/scene_0255/actor.glb'),source_actor='scene_0255/25',destination_scene='official_000',screened_frames=list(range(30)),actual_render_frames=list(range(10)),candidates=candidates,scope='Geometric empty-space screen then assistant source-image review; no traffic legality certificate. First render only10frames with r30 B_t; no MOVE or claim3sINSERT.',human_verdict=None))
sheet=Image.new('RGB',(1536,322*min(8,len(candidates))),(20,30,40));draw=ImageDraw.Draw(sheet)
for i,row in enumerate(candidates[:8]):
    for j,camid in enumerate(np.argsort(row['areas'])[-3:][::-1]):
        im=Image.open(frames[0]['views'][int(camid)]['image']).convert('RGB').resize((512,288));c,k=camera(frames[0],int(camid));bb=project_bbox(row['pose_world'],size,c,k,(576,1024))
        if bb:ImageDraw.Draw(im).rectangle((np.array(bb)*.5).tolist(),outline='magenta',width=3)
        sheet.paste(im,(j*512,i*322+28));draw.text((j*512+4,i*322+4),f"proposal{row['proposal']} CAM{camid} gap{row['min_gt_gap_m']:.1f}m ego{row['min_ego_gap_m']:.1f}m points{row['max_static_points']}",fill='white')
sheet.save(ROOT/'proposals.jpg',quality=94);dump(ROOT/'frames.json',frames[:10]);print('PROPOSALS',len(candidates),candidates[:2],flush=True)
