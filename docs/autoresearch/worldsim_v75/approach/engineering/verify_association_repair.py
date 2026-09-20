import json,sys,copy
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,'/root/autodl-tmp/motion_proj/scripts/worldsim_v75')
from rgb_idm_policy import MotionTracks,RGBIDMPolicy,IDMPolicy
from following_geometry import Route
R=Path('/root/autodl-tmp/runs/worldsim_v75'); src=R/'WS-V75-APPROACH-BASELINE-01/20260920-association-r2'; run=R/'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-r1'
a=json.loads((run/'association_audit.json').read_text()); row=next(r for r in a['rows'] if r['frame']==92); cost=np.array(row['cost']); ids=row['track_ids']; targetrow=ids.index(2); assert cost[targetrow,0]<1
oldi,oldj=linear_sum_assignment(cost); newi,newj=linear_sum_assignment(np.where(cost<=10,cost,11*(max(cost.shape)+1)))
assert not any(i==targetrow and j==0 for i,j in zip(oldi,oldj)); assert any(i==targetrow and j==0 for i,j in zip(newi,newj))
p=object.__new__(RGBIDMPolicy); p.idm=IDMPolicy(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.,decel_max=3.); p.tracker=MotionTracks(); protocol=json.loads((src/'protocol.json').read_text()); trajectory=np.load(Path(protocol['base'])/'trajectory.npz'); p.route=Route(trajectory['ego_world'][:,:3,3]); snap=json.loads((src/'tracker_at_t0.json').read_text()); p.tracker.previous_time=snap['previous_time']; p.tracker.serial=snap['serial']; p.tracker.tracks=[{k:np.array(v) if k in ['state','cov'] else v for k,v in t.items()} for t in snap['tracks']]
rows=[]
for old in json.loads((run/'gt_clean/decisions.json').read_text()):
 ps=old['policy']; state=ps['ego_state']; detections=copy.deepcopy(ps['detections']); p.tracker.update(detections,ps['timestamp_us']/1e6,state['speed_mps']*np.array([np.cos(state['yaw_rad']),np.sin(state['yaw_rad'])])); lead=p.choose_lead(detections,state); acc=p.acceleration(state['speed_mps'],lead); rows.append({'frame':old['observation_frame'],'old_track_id':ps['lead']['track_id'] if ps['lead'] else None,'fixed_track_id':lead['track_id'] if lead else None,'old_acceleration_mps2':ps['acceleration_mps2'],'fixed_acceleration_mps2':acc,'oracle_acceleration_mps2':old['oracle_acceleration_mps2']})
f=next(x for x in rows if x['frame']==92); assert f['fixed_track_id']==2 and f['fixed_acceleration_mps2']<0
result={'status':'passed','same_recorded_cost_verified':True,'observed_correct_match_distance_m':float(cost[targetrow,0]),'no_threshold_change':True,'new_detector_calls':0,'new_generator_calls':0,'role':'offline replay only, not a new closed loop','rows':rows}
(run/'association_repair_verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(f))
