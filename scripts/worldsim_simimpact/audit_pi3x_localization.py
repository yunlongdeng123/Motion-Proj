"""计算空间恢复的车辆执行结果与框余量；保留近边界局限。"""
import json,sys,os
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
B=Path('/root/autodl-tmp/external/worldsim_simimpact');sys.path.insert(0,str(B/'NAVSIM'))
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
C=Path(os.environ.get('SIMIMPACT_SIMULATION_INPUT_ROOT','/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1/lidar_policy/pi3x_localization'))
ref=json.loads((C/'scene-0061_log_reference.json').read_text())
rows=json.loads((C/'policy_probe_summary.json').read_text());results=json.loads((C/'pdm_simulation_summary.json').read_text())
z=np.load(C/'pdm_simulation/scene-0061/states.npz');v=get_pacifica_parameters()
times=z['times'];rt=np.array([r['time_s'] for r in ref['records']]);actors=[]
for t in times:
 hi=int(np.clip(np.searchsorted(rt,t,side='right'),1,len(rt)-1));lo=hi-1;f=float(np.clip((t-rt[lo])/(rt[hi]-rt[lo]),0,1))
 aa={r['instance']:r for r in ref['records'][lo]['boxes']};bb={r['instance']:r for r in ref['records'][hi]['boxes']};frame=[]
 for key in aa.keys()&bb.keys():
  a=np.array(aa[key]['box']);b=np.array(bb[key]['box']);b[6]=a[6]+np.arctan2(np.sin(b[6]-a[6]),np.cos(b[6]-a[6]));frame.append((key,aa[key]['category'],a*(1-f)+b*f))
 actors.append(frame)
def corners(x,y,w,l,yaw):
 p=np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]])
 return p@np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])+[x,y]
def gap(a,b):
 pa,pb=Polygon(a),Polygon(b)
 if not pa.intersects(pb):return pa.distance(pb)
 vals=[]
 for e in np.concatenate([a[1:3]-a[:2],b[1:3]-b[:2]]):
  axis=np.array([-e[1],e[0]])/np.linalg.norm(e);aa=a@axis;bb=b@axis;vals.append(min(aa.max()-bb.min(),bb.max()-aa.min()))
 return -min(vals)
def clearance(traj,margin):
 best={'signed_clearance_m':float('inf')}
 for t,s,frame in zip(times,traj,actors):
  x,y,h=s[:3];x+=v.rear_axle_to_center*np.cos(h);y+=v.rear_axle_to_center*np.sin(h)
  w=v.width+2*margin;l=v.length+2*margin;e=corners(x,y,w,l,h)
  for key,cat,b in frame:
   if np.hypot(x-b[0],y-b[1])-.5*np.hypot(w,l)-.5*np.hypot(b[3],b[4])>max(0,best['signed_clearance_m']):continue
   g=gap(e,corners(b[0],b[1],b[3],b[4],b[6]))
   if g<best['signed_clearance_m']:best={'signed_clearance_m':float(g),'time_s':float(t),'instance':key,'category':cat}
 return best
out=[]
for row,result,traj in zip(rows,results,z['simulated']):
 q={**result,'changed_returns':row['changed_returns'],'region':row.get('region'),
     'max_abs_vs_prior_GPU_trajectory':row.get('max_abs_vs_prior_GPU_trajectory'),
     'clearance':{str(m):clearance(traj,m) for m in [0.,-.1,.1]}}
 out.append(q)
 print(q['condition'],q.get('variant'),q['actor_overlap_any'],round(q['clearance']['0.0']['signed_clearance_m'],5),q['max_abs_vs_prior_GPU_trajectory'],flush=True)
final={'scope':json.loads((C/'registration.json').read_text())['role'],'rows':out,
       'recorded_ego_clearance':{str(m):clearance(z['ground_truth_ego'],m) for m in [0.,-.1,.1]}}
(C/'localization_outcomes.json').write_text(json.dumps(final,indent=2))
