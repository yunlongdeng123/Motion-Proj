"""原生传感器与冻结策略的短闭环；日志 teacher forcing 单列，非完整 PDM 分数。"""
import argparse, json, sys, time, copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from native_splatad_bridge import NativeSplatADBridge, CAMERAS, matrix
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
sys.path.insert(0,str(B/'NAVSIM'))
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Lidar, Trajectory
from hugsim.dataparser import parse_raw
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state,ego_state_to_state_array

ap=argparse.ArgumentParser()
ap.add_argument('--fit',type=Path,required=True)
ap.add_argument('--checkpoint-dir',type=Path,required=True)
ap.add_argument('--checkpoint-step',type=int,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--scene',default='scene-0004')
ap.add_argument('--steps',type=int,default=8)
ap.add_argument('--replay-sensor-timing',choices=['logged','synchronized'],default='logged')
args=ap.parse_args()
if args.out.exists():raise RuntimeError(f'Preserve existing run: {args.out}')
args.out.mkdir(parents=True)
torch.manual_seed(20260915);np.random.seed(20260915);torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
views=json.loads((I/'inputs'/f'{args.scene}.json').read_text())['views']
initial_ref=json.loads((I/'lidar_policy'/f'{args.scene}_log_reference.json').read_text())
reg={'task_id':'WS-SIM-NATIVE-CLOSEDLOOP-01','run_id':args.out.name,'scene':args.scene,
    'fit':str(args.fit),'checkpoint_dir':str(args.checkpoint_dir),'replay_sensor_timing':args.replay_sensor_timing,
    'checkpoint_step':args.checkpoint_step,'fitting_complete':args.checkpoint_step>=30000,
    'seed':20260915,'steps':args.steps,'replan_interval_s':.5,'physics_interval_s':.1,
    'scope':'SplatAD renders all three policy cameras and LiDAR at actually executed poses. Official TransFuser seed0 and PDM controller/bicycle; nuScenes adapter, annotated actors follow the log. Not an official end-to-end benchmark score.',
    'conditions':['real_sensor_teacher_forcing','rendered_sensor_teacher_forcing_raw','rendered_sensor_teacher_forcing_median','lidar_only_teacher_forcing_raw','lidar_only_teacher_forcing_median','rgb_only_teacher_forcing','closedloop_raw','closedloop_median'],
    'sensor_protocol':'For closed loop, synchronized query-time cameras and a fixed initial-log LiDAR firing pattern with official rolling compensation; no future GT images, ranges or validity masks enter closed-loop policy. Assets were fit with full-scene RGB+LiDAR and dynamics, an extra-information reference.',
    'controls':'Same 1–80m point range for policy inputs, known sensor extrinsics, native raw and median readouts kept. Vehicle model/ego-reference convention remains a dataset adapter.',
    'limits':'Real sensors exist only on logged poses; teacher-forced comparison is distinct from closed-loop counterfactuals. A changed trajectory does not imply degradation, collision safety or reconstruction causality.',
    'failure_ledger_refs':['V74-H2-F15','V74-H2-F16','V74-H2-F17'],'failure_ledger_delta':'pending',
    'human_verdict':None,'completed':False,'start_unix':time.time()}
(args.out/'registration.json').write_text(json.dumps(reg,indent=2))
bridge=NativeSplatADBridge(args.fit,args.checkpoint_dir,args.checkpoint_step,views[0]['sample_token'])
front=bridge.nusc.get('sample_data',bridge.sample['data']['CAM_FRONT'])
t0=front['timestamp']/1e6
anchor=bridge.ego_pose(front);anchor_inv=np.linalg.inv(anchor)
v0=np.r_[initial_ref['initial_velocity_xy_mps'],0.]
a0=np.r_[initial_ref['initial_acceleration_xy_mps2'],0.]
_,_,omega0=bridge.status_at(front)
reg.update(camera_calibration=bridge.camera_checks,lidar_pattern=bridge.lidar_check,
           time_origin_absolute_s=t0,world_from_ego_initial=anchor.tolist(),dataparser_transform=bridge.A.tolist())
(args.out/'registration.json').write_text(json.dumps(reg,indent=2))

c=TransfuserConfig();c.latent=False
agent=TransfuserAgent(c,1e-4,str(I/'lidar_policy/assets/transfuser_seed_0.ckpt'))
agent.initialize();agent.cuda().eval()
vehicle=get_pacifica_parameters();sampling=TrajectorySampling(num_poses=40,interval_length=.1)
sim=PDMSimulator(sampling)
rows=[];policy_count=0;controller_count=0

def state(t,pose,v,a,omega=None):
    return EgoState.build_from_rear_axle(StateSE2(*pose),StateVector2D(*v[:2]),StateVector2D(*a[:2]),
        0 if omega is None or abs(v[0])<.1 else float(np.arctan(omega[2]*vehicle.wheel_base/v[0])),
        TimePoint(int(round((t-t0+1)*1e6))),vehicle,
        angular_vel=0. if omega is None else float(omega[2]))

def relative_pose(world):
    p=anchor_inv@world;return np.array([p[0,3],p[1,3],np.arctan2(p[1,0],p[0,0])])

def world_pose(s):
    h=s[2];p=np.eye(4);p[:2,:2]=[[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]];p[:2,3]=s[:2]
    return anchor@p

def predict(rgb,points,v,a):
    global policy_count
    data=parse_raw(({'rgb':rgb},{'ego_pos':[0,0,0],'ego_steer':0,'ego_velo':float(v[0]),'accelerate':float(a[0])}))['input']
    data.ego_statuses[-1].ego_velocity=np.array(v[:2],np.float32)
    data.ego_statuses[-1].ego_acceleration=np.array(a[:2],np.float32)
    pc=np.zeros((6,len(points)),np.float32);pc[:3]=points.T;data.lidars[-1]=Lidar(pc)
    f={}
    for b in agent.get_feature_builders():f.update(b.compute_features(data))
    with torch.no_grad():output=agent({k:x.unsqueeze(0).cuda() for k,x in f.items()})
    policy_count+=1
    return {k:x[0].detach().cpu().numpy() for k,x in output.items()}

def policy_points(scan,readout):
    p=scan[readout];origin=scan.get('policy_lidar_origin_ego',bridge.extrinsics['LIDAR_TOP'][:3,3])
    r=np.linalg.norm(p-origin,axis=1)
    return p[np.isfinite(p).all(1)&(r>1)&(r<80)]

def execute(output,initial):
    global controller_count
    traj=Trajectory(output['trajectory'],trajectory_sampling=TrajectorySampling(num_poses=8,interval_length=.5))
    planned=get_trajectory_as_array(transform_trajectory(traj,initial),sampling,initial.time_point)
    controller_count+=1
    return sim.simulate_proposals(planned[None],initial)[0]

def save_observation(folder,obs):
    folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(obs['rgb']['CAM_FRONT']).save(folder/'front.jpg',quality=93)
    np.savez_compressed(folder/'sensors.npz',**obs['scan'],
        ego_pose_world=obs['ego_pose_world'],
        sensor_times_absolute_s=np.array([obs['sensor_times_absolute_s'][c] for c in (*CAMERAS,'LIDAR_TOP')]),
        camera_poses=np.array([obs['camera_poses'][c] for c in CAMERAS]))

def evaluate(tracked,t):
    tt=t+np.arange(len(tracked))*.1
    gt=np.array([relative_pose(bridge.pose_at(float(x))) for x in tt])
    e=np.linalg.norm(tracked[:,:2]-gt[:,:2],axis=1)
    return {'ADE_vs_recorded_ego_m':float(e.mean()),'FDE_vs_recorded_ego_m':float(e[-1]),
        'minimum_acceleration_mps2':float(tracked[:,5].min()),'final_speed_mps':float(tracked[-1,3]),
        'evaluation':'Recorded-trajectory agreement only here; actor clearance is audited separately.'}

def append(row):
    rows.append(row);(args.out/'policy_execution_summary.json').write_text(json.dumps(rows,indent=2))
    print('NATIVE_RESULT',json.dumps({k:v for k,v in row.items() if k not in ['trajectory','states']}),flush=True)

# 位姿响应与不依赖 GT 距离的接口证据，不计自然研究条件。
base=bridge.render(anchor,t0,v0,omega0)
probe_pose=anchor.copy();probe_pose[:3,3]+=anchor[:3,1]*.25
moved=bridge.render(probe_pose,t0,v0,omega0)
dummy=bridge.lidar_scan(anchor,t0,v0,omega0,placeholder=7.)
qa={'camera_pixel_MAE_for_0p25m_lateral_shift':float(abs(base['rgb']['CAM_FRONT'].astype(float)-moved['rgb']['CAM_FRONT']).mean()),
    'raw_range_mean_abs_change_for_0p25m_shift_m':float(abs(base['scan']['raw_range']-moved['scan']['raw_range']).mean()),
    'raw_range_max_abs_change_for_dummy_GT_range':float(abs(base['scan']['raw_range']-dummy['raw_range']).max()),
    'pose_shift_m':.25,'scope':'Controlled API response only, not a natural badcase.'}
(args.out/'sensor_interface_qa.json').write_text(json.dumps(qa,indent=2))
assert qa['camera_pixel_MAE_for_0p25m_lateral_shift']>.01
assert qa['raw_range_mean_abs_change_for_0p25m_shift_m']>1e-5
assert qa['raw_range_max_abs_change_for_dummy_GT_range']<1e-6
save_observation(args.out/'pose_probe/original',base)
save_observation(args.out/'pose_probe/displaced',moved)
del base,moved,dummy

# 日志对照默认匹配各传感器实际采样时刻；同步模式仅用于复现 8k 试跑。
sample=bridge.sample
for j in range(args.steps):
    rgb,points,ego,sd=bridge.real_observation(sample);t=sd['timestamp']/1e6
    v,a,w=bridge.status_at(sd)
    if j==0:v,a=v0,a0
    current=state(t,relative_pose(ego),v,a,w)
    obs=bridge.render_log_sample(sample) if args.replay_sensor_timing=='logged' else bridge.render(ego,t,v,w)
    save_observation(args.out/f'replay/frame_{j:02d}',obs)
    for condition,imgs,p in [('real_sensor_teacher_forcing',rgb,points),
        ('rendered_sensor_teacher_forcing_raw',obs['rgb'],policy_points(obs['scan'],'raw')),
        ('rendered_sensor_teacher_forcing_median',obs['rgb'],policy_points(obs['scan'],'median')),
        ('lidar_only_teacher_forcing_raw',rgb,policy_points(obs['scan'],'raw')),
        ('lidar_only_teacher_forcing_median',rgb,policy_points(obs['scan'],'median')),
        ('rgb_only_teacher_forcing',obs['rgb'],points)]:
        output=predict(imgs,p,v,a);tracked=execute(output,current)
        np.savez_compressed(args.out/f'replay/frame_{j:02d}/{condition}.npz',**output,states=tracked)
        row={'condition':condition,'index':j,'time_s':t-t0,'points':len(p),**evaluate(tracked,t)}
        if j==0 and condition=='real_sensor_teacher_forcing':
            prior=json.loads((I/'lidar_policy/policy_outputs'/args.scene/'real.json').read_text())
            row['initial_policy_max_abs_vs_previous']=float(abs(output['trajectory']-np.array(prior['trajectory'])).max())
        append(row)
    sample=bridge.nusc.get('sample',sample['next'])

for readout in ['raw','median']:
    condition='closedloop_'+readout
    current=state(t0,[0,0,0],v0,a0,omega0)
    all_states=[ego_state_to_state_array(current)]
    for j in range(args.steps):
        t=t0+j*.5;s=ego_state_to_state_array(current)
        v=np.r_[s[3:5],0];a=np.r_[s[5:7],0];w=np.array([0.,0.,s[9]])
        obs=bridge.render(world_pose(s),t,v,w)
        folder=args.out/condition/f'frame_{j:02d}';save_observation(folder,obs)
        points=policy_points(obs['scan'],readout)
        output=predict(obs['rgb'],points,v,a)
        tracked=execute(output,current)
        np.savez_compressed(folder/'policy.npz',**output,forecast_execution=tracked)
        all_states.extend(tracked[1:6])
        current=state_array_to_ego_state(tracked[5],TimePoint(current.time_point.time_us+500000),vehicle)
        print('NATIVE_LOOP_STEP',condition,j,tracked[5,:3].tolist(),flush=True)
    actual=np.array(all_states)
    np.savez_compressed(args.out/condition/'executed_states.npz',states=actual,times=np.arange(len(actual))*.1)
    append({'condition':condition,'actual_closed_loop':True,'executed_seconds':args.steps*.5,
        'state_samples':len(actual),**evaluate(actual,t0)})

reg.update(completed=True,end_unix=time.time(),policy_forwards=policy_count,
           controller_forecasts=controller_count,actual_closed_loops=2,failure_ledger_delta='pending_analysis')
(args.out/'registration.json').write_text(json.dumps(reg,indent=2))
print('NATIVE_CLOSED_LOOP_COMPLETED',args.out,flush=True)
