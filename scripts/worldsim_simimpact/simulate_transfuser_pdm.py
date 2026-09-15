import os
"""官方NAVSIM非反应式4秒车辆仿真；不给未计算的地图/PDM总分命名。

保留官方Pacifica参数，nuScenes状态适配和演员框交叠单列；不是新视角传感器闭环。
"""
import sys,json
from pathlib import Path
import numpy as np
B=Path('/root/autodl-tmp/external/worldsim_simimpact');sys.path[:0]=[str(B/'NAVSIM'),str(B/'HUGSIM'),str(B/'HUGSIM/sim')]
R=Path(os.environ.get('SIMIMPACT_SIMULATION_INPUT_ROOT',str(Path(os.environ.get('SIMIMPACT_RUN_ROOT','/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1'))/'lidar_policy')))
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from hugsim_env.envs.hug_sim import fg_collision_det
sampling=TrajectorySampling(num_poses=40,interval_length=.1);vehicle=get_pacifica_parameters();sim=PDMSimulator(sampling)
inputs=json.loads((R/'policy_probe_summary.json').read_text());results=[]
for name in sorted({r['scene'] for r in inputs}):
    ref=json.loads((R/f'{name}_log_reference.json').read_text());rows=[r for r in inputs if r['scene']==name]
    causal='initial_acceleration_xy_mps2' in ref
    yaw_rate=float(ref.get('initial_yaw_rate_rad_s',0.));forward_speed=float(ref['initial_velocity_xy_mps'][0])
    steering=float(np.arctan(yaw_rate*vehicle.wheel_base/forward_speed)) if abs(forward_speed)>.1 else 0.
    initial=EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(0,0,0),rear_axle_velocity_2d=StateVector2D(*ref['initial_velocity_xy_mps']) if causal else StateVector2D(ref['initial_velocity_xy_mps'][0],0),
        rear_axle_acceleration_2d=StateVector2D(*ref['initial_acceleration_xy_mps2']) if causal else StateVector2D(0,0),tire_steering_angle=steering,time_point=TimePoint(1000000),vehicle_parameters=vehicle,angular_vel=yaw_rate)
    states=[]
    for row in rows:
        traj=Trajectory(np.array(row['trajectory']),trajectory_sampling=TrajectorySampling(num_poses=8,interval_length=.5))
        official=transform_trajectory(traj,initial);states.append(get_trajectory_as_array(official,sampling,initial.time_point))
    tracked=sim.simulate_proposals(np.stack(states),initial);base=tracked[next(i for i,r in enumerate(rows) if r['condition']=='real')]
    times=np.arange(41)*.1;record_times=np.array([r['time_s'] for r in ref['records']]);record_poses=np.array([r['xy_yaw'] for r in ref['records']]);record_poses[:,2]=np.unwrap(record_poses[:,2])
    gt=np.column_stack([np.interp(times,record_times,record_poses[:,i]) for i in range(3)])
    actor_lists=[]
    for t in times:
        hi=int(np.clip(np.searchsorted(record_times,t,side='right'),1,len(record_times)-1));lo=hi-1;f=float(np.clip((t-record_times[lo])/(record_times[hi]-record_times[lo]),0,1))
        aa={a['instance']:a['box'] for a in ref['records'][lo]['boxes']};bb={a['instance']:a['box'] for a in ref['records'][hi]['boxes']};boxes=[]
        for key in aa.keys()&bb.keys():
            a=np.array(aa[key]);b=np.array(bb[key]);b[6]=a[6]+np.arctan2(np.sin(b[6]-a[6]),np.cos(b[6]-a[6]));boxes.append((a*(1-f)+b*f).tolist())
        actor_lists.append(boxes)
    def overlap(poses):
        flags=[]
        for pose,boxes in zip(poses,actor_lists):
            x,y,yaw=pose[:3];center=[x+vehicle.rear_axle_to_center*np.cos(yaw),y+vehicle.rear_axle_to_center*np.sin(yaw)]
            flags.append(bool(fg_collision_det([*center,0,vehicle.width,vehicle.length,vehicle.height,yaw],boxes)))
        return flags
    gt_overlap=overlap(gt);out=R/'pdm_simulation'/name;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'states.npz',simulated=tracked,ground_truth_ego=gt,times=times)
    for row,s in zip(rows,tracked):
        delta=s[:,:2]-base[:,:2];errors=np.linalg.norm(s[:,:2]-gt[:,:2],axis=1);flags=overlap(s)
        res={k:row[k] for k in ['scene','condition','method','variant','protocol'] if k in row}
        res.update(final_position_change_vs_real_m=float(np.linalg.norm(delta[-1])),mean_position_change_vs_real_m=float(np.linalg.norm(delta,axis=1).mean()),
            final_forward_m=float(s[-1,0]),final_lateral_m=float(s[-1,1]),minimum_longitudinal_acceleration_mps2=float(s[:,5].min()),
            final_speed_mps=float(np.linalg.norm(s[-1,3:5])),ADE_vs_recorded_ego_m=float(errors.mean()),FDE_vs_recorded_ego_m=float(errors[-1]),
            actor_overlap_any=any(flags),actor_overlap_first_time_s=next((float(t) for t,f in zip(times,flags) if f),None),
            recorded_ego_actor_overlap_under_same_vehicle=any(gt_overlap),vehicle_model='Official NAVSIM/nuPlan Pacifica; dataset-adapter convention',initial_yaw_rate_rad_s=yaw_rate,initial_tire_steering_angle_rad=steering,
            scope='Official PDMSimulator 4s tracking. Actor overlap is a separate geometric diagnostic, not the full PDM score; no map/at-fault/reaction metric.')
        results.append(res)
    print(name,len(rows),'max_shift',max(r['final_position_change_vs_real_m'] for r in results if r['scene']==name),'overlaps',sum(r['actor_overlap_any'] for r in results if r['scene']==name),flush=True)
(R/'pdm_simulation_summary.json').write_text(json.dumps(results,indent=2))
