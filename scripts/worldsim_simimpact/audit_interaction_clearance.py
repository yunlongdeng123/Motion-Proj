"""定位新增交叠：对象、距离余量、框尺寸敏感性与真实点的相机覆盖。"""
import sys,json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1');L=R/'lidar_policy';name='scene-0061'
ref=json.loads((L/f'{name}_log_reference.json').read_text());rows=[r for r in json.loads((L/'policy_probe_summary.json').read_text()) if r['scene']==name]
tr=np.load(L/'pdm_simulation'/name/'states.npz');times=tr['times'];poses=tr['simulated'];record_times=np.array([r['time_s'] for r in ref['records']]);vehicle=get_pacifica_parameters()
def corners(x,y,w,l,yaw):
 c=np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]]);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]);return c@rot.T+[x,y]
def signed_gap(a,b):
 pa,pb=Polygon(a),Polygon(b)
 if not pa.intersects(pb):return pa.distance(pb)
 edges=np.concatenate([np.diff(np.r_[a,a[:1]],axis=0)[:2],np.diff(np.r_[b,b[:1]],axis=0)[:2]])
 depths=[]
 for e in edges:
  axis=np.array([-e[1],e[0]])/np.linalg.norm(e);aa=a@axis;bb=b@axis;depths.append(min(aa.max()-bb.min(),bb.max()-aa.min()))
 return -float(min(depths))
actor_lists=[]
for t in times:
 hi=int(np.clip(np.searchsorted(record_times,t,side='right'),1,len(record_times)-1));lo=hi-1;f=float(np.clip((t-record_times[lo])/(record_times[hi]-record_times[lo]),0,1))
 aa={a['instance']:a for a in ref['records'][lo]['boxes']};bb={a['instance']:a for a in ref['records'][hi]['boxes']};actors=[]
 for k in aa.keys()&bb.keys():
  a=np.array(aa[k]['box']);b=np.array(bb[k]['box']);b[6]=a[6]+np.arctan2(np.sin(b[6]-a[6]),np.cos(b[6]-a[6]));c=a*(1-f)+b*f;actors.append({'instance':k,'category':aa[k]['category'],'box':c.tolist()})
 actor_lists.append(actors)
results=[]
for row,trajectory in zip(rows,poses):
 rec={k:row[k] for k in ['condition','method','variant','protocol'] if k in row};details={}
 for margin in [0.,-.1,.1]:
  minima=[];best=np.inf;w=vehicle.width+2*margin;l=vehicle.length+2*margin
  for t,s,actors in zip(times,trajectory,actor_lists):
   x,y,yaw=s[:3];x+=vehicle.rear_axle_to_center*np.cos(yaw);y+=vehicle.rear_axle_to_center*np.sin(yaw);ego=corners(x,y,w,l,yaw)
   for a in actors:
    b=a['box'];bound=np.hypot(x-b[0],y-b[1])-.5*np.hypot(w,l)-.5*np.hypot(b[3],b[4])
    if bound>max(0,best):continue
    gap=signed_gap(ego,corners(b[0],b[1],b[3],b[4],b[6]));best=min(best,gap);minima.append({'time_s':float(t),'signed_clearance_m':float(gap),'instance':a['instance'],'category':a['category']})
  details[f'ego_margin_{margin:+.1f}m']=min(minima,key=lambda a:a['signed_clearance_m'])
 rec['clearance']=details;results.append(rec)
gtresult=[];best=np.inf
for t,s,actors in zip(times,tr['ground_truth_ego'],actor_lists):
 x,y,yaw=s[:3];x+=vehicle.rear_axle_to_center*np.cos(yaw);y+=vehicle.rear_axle_to_center*np.sin(yaw);e=corners(x,y,vehicle.width,vehicle.length,yaw)
 for a in actors:
  b=a['box'];bound=np.hypot(x-b[0],y-b[1])-.5*np.hypot(vehicle.width,vehicle.length)-.5*np.hypot(b[3],b[4])
  if bound>max(0,best):continue
  gap=signed_gap(e,corners(b[0],b[1],b[3],b[4],b[6]));best=min(best,gap);gtresult.append({'time_s':float(t),'signed_clearance_m':gap,'instance':a['instance'],'category':a['category']})
# GT point membership and known-camera FOV. Visibility/occlusion remain separate.
real=np.load(L/'scans'/name/'real.npz');gt=real['points'];inp=json.loads((R/'inputs'/f'{name}.json').read_text());ego=np.array(inp['views'][0]['world_from_ego_camera']);covered=np.zeros(len(gt),dtype=bool)
for v in inp['views'][:6]:
 T=np.linalg.inv(ego)@np.array(v['world_from_camera']);cp=(gt-T[:3,3])@T[:3,:3];uv=cp@np.array(v['intrinsics_original']).T;uv=uv[:,:2]/uv[:,2:];w,h=v['original_wh'];covered|=(cp[:,2]>.2)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
objects=[]
for a in ref['records'][0]['boxes']:
 b=np.array(a['box']);yaw=b[6];rot=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]);q=(gt-b[:3])@rot;mask=(abs(q)<np.array([b[4],b[3],b[5]])/2+.05).all(1)
 if not mask.any():continue
 stats=[]
 for method in ['dvgt1','vggt','omega512','pi3x']:
  for variant in ['six','twelve']:
   s=np.load(L/'scans'/name/f'{method}_{variant}_build_scale.npz');d=s['first_range'];delta=d-real['ranges'];stats.append({'method':method,'variant':variant,'n':int(mask.sum()),'in_camera_fov':int((mask&covered).sum()),'miss':int((mask&~np.isfinite(d)).sum()),'early_0p2':int((mask&(delta<-.2)).sum()),'late_0p2':int((mask&np.isfinite(d)&(delta>.2)).sum())})
 objects.append({'instance':a['instance'],'category':a['category'],'box':a['box'],'point_stats':stats})
out={'scene':name,'ego_parameters':{'width':vehicle.width,'length':vehicle.length,'rear_axle_to_center':vehicle.rear_axle_to_center},'executions':results,'recorded_ego_minimum_clearance':min(gtresult,key=lambda r:r['signed_clearance_m']),
 'initial_object_return_stats':objects,'coverage':'Known camera field of view only, not full occlusion visibility','scope':'Negative signed clearance is rectangle penetration; annotation / vehicle convention uncertainty remains.'}
(L/'clearance_audit.json').write_text(json.dumps(out,indent=2))
print('GT',out['recorded_ego_minimum_clearance'],flush=True)
for r in results:
 if r['condition']=='real' or (r.get('protocol')=='build_scale' and r['condition'] in ['full','missing_only']):print(r,flush=True)
