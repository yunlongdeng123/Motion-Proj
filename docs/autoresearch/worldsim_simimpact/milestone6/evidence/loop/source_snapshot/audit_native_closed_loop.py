"""按下游事件审计原生短闭环；已知框间距与感知匹配，保留完整分母。"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
import sys
B=Path('/root/autodl-tmp/external/worldsim_simimpact');sys.path.insert(0,str(B/'NAVSIM'))
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--reference',type=Path);args=ap.parse_args()
reg=json.loads((args.run/'registration.json').read_text());assert reg['completed']
reference=args.reference or I/'lidar_policy'/f"{reg['scene']}_log_reference.json"
ref=json.loads(reference.read_text())
rows=json.loads((args.run/'policy_execution_summary.json').read_text())
times=np.array([r['time_s'] for r in ref['records']]);vehicle=get_pacifica_parameters()

def corners(x,y,w,l,h):
    p=np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]])
    return p@np.array([[np.cos(h),np.sin(h)],[-np.sin(h),np.cos(h)]])+[x,y]

def signed_gap(a,b):
    pa,pb=Polygon(a),Polygon(b)
    if not pa.intersects(pb):return pa.distance(pb)
    gaps=[]
    for e in np.r_[a[1:3]-a[:2],b[1:3]-b[:2]]:
        axis=np.array([-e[1],e[0]])/np.linalg.norm(e);aa=a@axis;bb=b@axis
        gaps.append(min(aa.max()-bb.min(),bb.max()-aa.min()))
    return -min(gaps)

def actors_at(t):
    hi=int(np.clip(np.searchsorted(times,t,side='right'),1,len(times)-1));lo=hi-1
    f=float(np.clip((t-times[lo])/(times[hi]-times[lo]),0,1))
    aa={x['instance']:x for x in ref['records'][lo]['boxes']};bb={x['instance']:x for x in ref['records'][hi]['boxes']}
    out=[]
    for k in aa.keys()&bb.keys():
        a=np.array(aa[k]['box']);b=np.array(bb[k]['box']);b[6]=a[6]+np.arctan2(np.sin(b[6]-a[6]),np.cos(b[6]-a[6]))
        out.append({'instance':k,'category':aa[k]['category'],'box':a*(1-f)+b*f})
    return out

def gap_summary(states,tt,margin=0.):
    best={'signed_box_clearance_m':float('inf')};n=0
    for s,t in zip(states,tt):
        if t<times[0]-1e-5 or t>times[-1]+1e-5:continue
        n+=1;x,y,h=s[:3];x+=vehicle.rear_axle_to_center*np.cos(h);y+=vehicle.rear_axle_to_center*np.sin(h)
        p=corners(x,y,vehicle.width+2*margin,vehicle.length+2*margin,h)
        for actor in actors_at(t):
            b=actor['box']
            g=signed_gap(p,corners(b[0],b[1],b[3],b[4],b[6]))
            if g<best['signed_box_clearance_m']:best={'signed_box_clearance_m':float(g),'time_s':float(t),'instance':actor['instance'],'category':actor['category']}
    best.update(evaluated_state_samples=n,has_annotated_overlap=best['signed_box_clearance_m']<0)
    return best

loop=[]
for row in [r for r in rows if r.get('actual_closed_loop')]:
    z=np.load(args.run/row['condition']/'executed_states.npz')
    loop.append({**row,'clearance':{str(m):gap_summary(z['states'],z['times'],m) for m in [0.,-.1,.1]}})
tt=np.arange(reg['steps']*5+1)*.1
gt=np.array([np.interp(tt,times,np.array([r['xy_yaw'] for r in ref['records']])[:,k]) for k in range(3)]).T
recorded_clearance={str(m):gap_summary(gt,tt,m) for m in [0.,-.1,.1]}

def detection(npz,index,threshold):
    # 官方头预测 vehicle 单类；0.5 是固定存在概率阈值。IoU 两个阈值都报告。
    state=npz['agent_states'];labels=npz['agent_labels']
    keep=(labels>0)&(state[:,3]>0)&(state[:,4]>0)
    pred=state[keep]
    pose=np.array(ref['records'][index]['pose']);inv=np.linalg.inv(pose)
    gt=[];ids=[]
    for actor in ref['records'][index]['boxes']:
        b=np.array(actor['box']);p=inv[:3,:3]@b[:3]+inv[:3,3]
        if not actor['category'].startswith('vehicle.') or actor['lidar_points']<5:continue
        if not (0<=p[0]<=32 and abs(p[1])<=16):continue
        h=b[6]-np.arctan2(pose[1,0],pose[0,0]);gt.append([p[0],p[1],h,b[4],b[3]]);ids.append(actor['instance'])
    polys_p=[Polygon(corners(p[0],p[1],p[4],p[3],p[2])) for p in pred]
    polys_g=[Polygon(corners(g[0],g[1],g[4],g[3],g[2])) for g in gt]
    iou=np.array([[p.intersection(g).area/max(p.union(g).area,1e-8) for g in polys_g] for p in polys_p]).reshape(len(pred),len(gt))
    matches=[]
    if len(pred) and len(gt):
        pp,gg=linear_sum_assignment(-iou)
        matches=[{'instance':ids[g],'IoU':float(iou[p,g]),'pred':pred[p].tolist(),'gt':gt[g]} for p,g in zip(pp,gg) if iou[p,g]>=threshold]
    return {'eligible_visible_forward_vehicles':len(gt),'matched':len(matches),'matches':matches,
        'unmatched_ids':sorted(set(ids)-{m['instance'] for m in matches}),
        'predicted_positive_count':int(keep.sum()),'IoU_threshold':threshold,
        'scope':'nuScenes adapter diagnostic, not a native detection benchmark or AP; annotations require >=5 real LiDAR points.'}

replay=[]
for index in sorted({r['index'] for r in rows if 'index' in r}):
    base=next(r for r in rows if r.get('index')==index and r['condition']=='real_sensor_teacher_forcing')
    folder=args.run/f'replay/frame_{index:02d}'
    zbase=np.load(folder/'real_sensor_teacher_forcing.npz')
    for row in [r for r in rows if r.get('index')==index]:
        z=np.load(folder/(row['condition']+'.npz'))
        delta=z['states'][:,:2]-zbase['states'][:,:2]
        near=z['states'][:6];t=row['time_s']+np.arange(6)*.1
        replay.append({**row,'ADE_delta_vs_real_m':row['ADE_vs_recorded_ego_m']-base['ADE_vs_recorded_ego_m'],
            'execution_endpoint_change_vs_real_m':float(np.linalg.norm(delta[-1])),
            'extra_minimum_acceleration_mps2':row['minimum_acceleration_mps2']-base['minimum_acceleration_mps2'],
            'next_half_second_clearance':gap_summary(near,t),
            'detection':{str(th):detection(z,index,th) for th in [.25,.5]},
            'semantic_argmax_changed_cells_vs_real':int((z['bev_semantic_map'].argmax(0)!=zbase['bev_semantic_map'].argmax(0)).sum()),
            'semantic_limit':'Prediction differences only; complete drivable-area/map truth not evaluated.'})
out={'scope':'Downstream-first pilot audit; no geometry interventions, no causal or broad safety conclusion.',
    'reference':str(reference),
    'checkpoint_step':reg['checkpoint_step'],'fitting_complete':reg['fitting_complete'],
    'recorded_ego_clearance':recorded_clearance,'closed_loop':loop,'replay':replay}
(args.run/'downstream_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'closed_loop':loop,'recorded_ego_clearance':recorded_clearance},indent=2))
