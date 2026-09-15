import os
"""官方TransFuser消费真实/重建扫描；分离early、late、missing的因果诊断。

本脚本评价预测轨迹，不冒充完整闭环。nuScenes输入适配到官方NAVSIM特征，
真实扫描基线与所有干预共享RGB/车辆状态。oracle混合仅作归因，非同信息方法。
"""
import sys,json,copy,argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
B=Path('/root/autodl-tmp/external/worldsim_simimpact');sys.path[:0]=[str(B/'NAVSIM'),str(B/'HUGSIM'),str(B/'HUGSIM/sim')]
R=Path(os.environ.get('SIMIMPACT_RUN_ROOT','/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1'));O=R/'lidar_policy'
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Lidar
from hugsim.dataparser import parse_raw
from hugsim_env.envs.hug_sim import fg_collision_det
ap=argparse.ArgumentParser();ap.add_argument('--scene',nargs='*');args=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(20260915);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
c=TransfuserConfig();c.latent=False
agent=TransfuserAgent(c,1e-4,str(O/'assets/transfuser_seed_0.ckpt'));agent.initialize();agent.cuda().eval()
completed=[]
for line in (R/'lidar_raycast.log').read_text().splitlines():
    try:r=json.loads(line)
    except ValueError:continue
    if 'method' in r:completed.append(r)
all_rows=[]
for name in sorted(p.stem for p in (R/'inputs').glob('*.json')):
    if args.scene and name not in args.scene:continue
    ref_file=O/f'{name}_log_reference.json'
    if not ref_file.exists():continue
    ref=json.loads(ref_file.read_text());views=json.loads((R/'inputs'/f'{name}.json').read_text())['views']
    rgb={v['camera']:np.array(Image.open(v['image']).convert('RGB')) for v in views[:6] if v['camera'] in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']}
    info={'ego_pos':[0.,0.,0.],'ego_steer':0.,'ego_velo':float(ref['initial_velocity_xy_mps'][0]),'accelerate':0.}
    data=parse_raw(({'rgb':rgb},info))['input']
    if 'driving_command_onehot' in ref:
        command=np.asarray(ref['driving_command_onehot'],dtype=np.float32)
        assert command.shape==(4,) and set(command.tolist())<=set([0.,1.]) and command.sum()==1
        data.ego_statuses[-1].driving_command=command
    if 'initial_acceleration_xy_mps2' in ref:
        data.ego_statuses[-1].ego_velocity=np.array(ref['initial_velocity_xy_mps'],dtype=np.float32)
        data.ego_statuses[-1].ego_acceleration=np.array(ref['initial_acceleration_xy_mps2'],dtype=np.float32)
    gt=np.load(O/'scans'/name/'real.npz');gtpoints=gt['points'];origin=gt['origin'];directions=gt['directions'];ranges=gt['ranges']
    out=O/'policy_outputs'/name;out.mkdir(parents=True,exist_ok=True)
    def predict(points):
        pc=np.zeros((6,len(points)),dtype=np.float32);pc[:3]=points.T;data.lidars[-1]=Lidar(pc)
        features={}
        for b in agent.get_feature_builders():features.update(b.compute_features(data))
        features={k:v.unsqueeze(0).cuda() for k,v in features.items()}
        with torch.no_grad():pred=agent(features)
        return pred['trajectory'][0].cpu().numpy(),features['lidar_feature'][0].cpu().numpy()
    baseline,basehist=predict(gtpoints)
    records=ref['records'];times=np.array([r['time_s'] for r in records]);gtposes=np.array([r['xy_yaw'] for r in records]);gtposes[:,2]=np.unwrap(gtposes[:,2]);horizon=np.arange(1,9)*.5
    gttraj=np.column_stack([np.interp(horizon,times,gtposes[:,i]) for i in range(3)])
    def metrics(traj):
        dist=np.linalg.norm(traj[:,:2]-gttraj[:,:2],axis=1)
        intersections=[]
        # Use native HUGSIM foreground rectangle rule and its 1.6m x 3m ego footprint.
        for j,pose in enumerate(traj):
            rr=records[min(j+1,len(records)-1)];boxes=[b['box'] for b in rr['boxes']]
            flag=bool(fg_collision_det([*pose[:2],0,1.6,3.,1.5,float(pose[2])],boxes))
            intersections.append(flag)
        return {'ADE_vs_recorded_ego_m':float(dist.mean()),'FDE_vs_recorded_ego_m':float(dist[-1]),
            'planned_actor_intersection_at_0p5s_samples':intersections,'planned_actor_intersection_any':any(intersections),
            'four_second_forward_m':float(traj[-1,0]),'four_second_lateral_m':float(traj[-1,1])}
    base_row={'scene':name,'condition':'real','trajectory':baseline.tolist(),'metrics':metrics(baseline),'initial_velocity_mps':info['ego_velo'],
        'checkpoint':str(O/'assets/transfuser_seed_0.ckpt'),'latent':False,'strict_checkpoint_loading':True,'scope':'Official policy + feature builder with nuScenes adapter; planned trajectory, not closed-loop',
        'input_velocity_xy':data.ego_statuses[-1].ego_velocity.tolist(),'input_acceleration_xy':data.ego_statuses[-1].ego_acceleration.tolist(),
        'status_source':ref.get('status_source','Forward finite-difference speed; zero lateral speed and acceleration from original client'),
        'driving_command_onehot':data.ego_statuses[-1].driving_command.tolist(),'driving_command_source':ref.get('driving_command_source','Unchanged fixed official client command')}
    (out/'real.json').write_text(json.dumps(base_row,indent=2));np.save(out/'real_histogram.npy',basehist);all_rows.append(base_row)
    for rec in [r for r in completed if r['scene']==name]:
        key=f'{rec["method"]}_{rec["variant"]}_{rec["protocol"]}';scan=np.load(O/'scans'/name/f'{key}.npz');distance=scan['first_range'];finite=np.isfinite(distance);early=distance<ranges-.2;late=finite&(distance>ranges+.2)
        for condition in ['full','early_only','late_only','missing_only']:
            target=out/f'{key}_{condition}.json'
            if target.exists():all_rows.append(json.loads(target.read_text()));continue
            if condition=='full':points=scan['points']
            elif condition=='missing_only':points=gtpoints[finite]
            else:
                mask=early if condition=='early_only' else late;points=gtpoints.copy();points[mask]=origin+directions[mask]*distance[mask,None]
            traj,hist=predict(points);d=traj[:,:2]-baseline[:,:2]
            row={'scene':name,'method':rec['method'],'variant':rec['variant'],'protocol':rec['protocol'],'condition':condition,
                'trajectory':traj.tolist(),'metrics':metrics(traj),'points':len(points),
                'mean_trajectory_shift_vs_real_m':float(np.linalg.norm(d,axis=1).mean()),'final_trajectory_shift_vs_real_m':float(np.linalg.norm(d[-1])),
                'histogram_changed_cells':int((hist!=basehist).sum()),'histogram_L1':float(abs(hist-basehist).sum()),
                'new_nonzero_cells':int(((hist>0)&(basehist==0)).sum()),'removed_nonzero_cells':int(((hist==0)&(basehist>0)).sum()),
                'input_scope':'Natural full reconstructed readout' if condition=='full' else 'Oracle hybrid: only this observed error type changes; other beams retain real values',
                'evaluation_scope':'Predicted trajectory and annotated-actor overlap only; no claim of executed collision or complete static GT'}
            target.write_text(json.dumps(row,indent=2));np.save(out/f'{key}_{condition}_histogram.npy',hist);all_rows.append(row)
            print(name,key,condition,'shift',row['final_trajectory_shift_vs_real_m'],'progress',row['metrics']['four_second_forward_m'],'overlap',row['metrics']['planned_actor_intersection_any'],flush=True)
(O/'policy_probe_summary.json').write_text(json.dumps(all_rows,indent=2))
