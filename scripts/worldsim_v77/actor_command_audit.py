"""在完整已处理时间窗审计目标绑定与平移扫掠体，不将框间无碰撞当作道路合法。"""
import json
import pathlib
import numpy as np
from shapely.geometry import Polygon, MultiPoint

RUN=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
OUT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-ACTOR-COMMAND-AUDIT-20260926/r1')
SCENES={'scene_0230':('22',50,5,2),'scene_0255':('25',100,20,3)}

def box(row,frame):
    fa=row['frame_annotations']
    if frame not in fa['frame_idx']:return None
    j=fa['frame_idx'].index(frame)
    return np.array(fa['obj_to_world'][j],float),np.array(fa['box_size'][j],float)

def footprint(pose,size,delta=np.zeros(3)):
    corners=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*size[:2]/2
    xyz=np.c_[corners,np.zeros(4)]@pose[:3,:3].T+pose[:3,3]+delta
    return Polygon(xyz[:,:2])


def translation_sweep(source,target):
    """固定朝向纯平移矩形的连续扫掠体，涵盖端点之间的所有位置。"""
    return MultiPoint(list(source.exterior.coords)+list(target.exterior.coords)).convex_hull

def audit(instances,actor,count,delta):
    result=[]
    for f in range(count):
        item=box(instances[actor],f)
        if item is None:continue
        pose,size=item
        source=footprint(pose,size)
        target=footprint(pose,size,delta)
        swept=translation_sweep(source,target)
        peers=[]
        for aid,row in instances.items():
            if aid==actor:continue
            other=box(row,f)
            if other is None:continue
            op,os=other
            if abs(op[2,3]-pose[2,3])>(size[2]+os[2])/2:continue
            poly=footprint(op,os)
            if poly.distance(swept)>12:continue
            peers.append({'actor_id':aid,'class':row['class_name'],
                'baseline_overlap_m2':source.intersection(poly).area,
                'endpoint_overlap_m2':target.intersection(poly).area,
                'swept_overlap_m2':swept.intersection(poly).area,
                'swept_clearance_m':swept.distance(poly),
                'endpoint_clearance_m':target.distance(poly),
                'footprint_world':list(poly.exterior.coords)[:-1]})
        result.append({'frame':f,'time_s':f/10,'target_pose':pose.tolist(),'target_size_lwh':size.tolist(),
           'source_footprint_world':list(source.exterior.coords)[:-1],
           'destination_footprint_world':list(target.exterior.coords)[:-1],
           'swept_footprint_world':list(swept.exterior.coords)[:-1],
           'delta_actor_local_m':(pose[:3,:3].T@delta).tolist(),
           'endpoint_hits':[p for p in peers if p['endpoint_overlap_m2']>1e-5],
           'swept_hits':[p for p in peers if p['swept_overlap_m2']>1e-5],
           'peers':peers})
    return {'delta_world_m':delta.tolist(),'frames_checked':len(result),
      'endpoint_collision_frames':sum(bool(r['endpoint_hits']) for r in result),
      'swept_collision_frames':sum(bool(r['swept_hits']) for r in result),
      'endpoint_hit_ids':sorted({p['actor_id'] for r in result for p in r['endpoint_hits']}),
      'swept_hit_ids':sorted({p['actor_id'] for r in result for p in r['swept_hits']}),
      'minimum_swept_clearance_m':min((p['swept_clearance_m'] for r in result for p in r['peers']),default=None),
      'max_new_swept_overlap_m2':max((p['swept_overlap_m2']-p['baseline_overlap_m2'] for r in result for p in r['peers']),default=0),
      'frames':result}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 summary={'task_id':'WS-V77-ACTOR-COMMAND-AUDIT-20260926','training_steps':0,'new_model_forwards':0,
 'geometry_scope':'完整时间窗逐帧GT对象框，平面纯平移扫掠多边形精确凸包；检查同高度对象。没有道路可行驶区、静态场景网格或ego车身尺寸；不得把无框碰撞称为完整指令合法。',
 'human_feedback':'用户称本地 Blender 中 GLB 看起来没太大问题；要求先排查目标绑定和移动指令合法性。','scenes':[]}
 for name,(actor,count,ref,cam) in SCENES.items():
    instances=json.loads((RUN/name/'instances_snapshot.json').read_text())
    delta=np.array(json.loads((RUN/name/'actor/placement.json').read_text())['move_delta_world_m'])
    pose,size=box(instances[actor],ref)
    old=audit(instances,actor,count,delta)
    candidates=[]
    for length in [2.0,1.0,0.5]:
        for angle in range(0,360,45):
            direction=pose[:3,:3]@np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0])
            direction[2]=0;direction/=np.linalg.norm(direction)
            r=audit(instances,actor,count,direction*length)
            candidates.append({k:v for k,v in r.items() if k!='frames'}|{'angle_actor_deg':angle,'length_m':length})
    centers=np.array([box(instances[actor],f)[0][:3,3] for f in range(count) if box(instances[actor],f) is not None])
    row={'name':name,'actor_id':actor,'track_id':instances[actor]['id'],'class':instances[actor]['class_name'],
       'reference_frame':ref,'reference_camera':cam,'size_lwh':size.tolist(),
       'track_net_displacement_m':float(np.linalg.norm(centers[-1]-centers[0])),
       'track_max_deviation_from_first_m':float(np.linalg.norm(centers-centers[0],axis=1).max()),
       'old_move':old,'candidate_sweep':candidates,'human_verdict':None}
    (OUT/f'{name}_audit.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    compact={k:v for k,v in row.items() if k not in ['old_move','candidate_sweep']}
    compact['old_move']={k:v for k,v in old.items() if k!='frames'}
    compact['candidate_sweep']=candidates
    summary['scenes'].append(compact)
    print(json.dumps(compact,ensure_ascii=False),flush=True)
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
 main()
