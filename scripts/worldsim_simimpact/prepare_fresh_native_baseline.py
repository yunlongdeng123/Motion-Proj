"""八个冻结真实输入起点；共享已运行的官方策略/车辆组件，先验证基线。"""
import json,shutil,time
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from PIL import Image
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1');B=R/'baseline';P=Path('/root/autodl-tmp/motion_proj/scripts/worldsim_simimpact')
if B.exists():raise RuntimeError('Preserve baseline stage')
assert not json.loads((R/'extraction_result.json').read_text())['missing']
B.mkdir();(B/'inputs').mkdir();L=B/'lidar_policy';L.mkdir();(L/'assets').mkdir()
ckpt=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1/lidar_policy/assets/transfuser_seed_0.ckpt');assert ckpt.exists();(L/'assets/transfuser_seed_0.ckpt').symlink_to(ckpt.resolve())
(B/'lidar_raycast.log').write_text('');shutil.copy2(__file__,B/'source_snapshot.py')
for f in ['probe_transfuser_lidar.py','simulate_transfuser_pdm.py']:shutil.copy2(P/f,B/f)
src=json.loads((R/'raw_source_availability.json').read_text());M=R/'data/v1.0-trainval';tokens={t for s in src.values() for t in s['samples']}
ann={x['token']:x for x in json.loads((M/'sample_annotation.json').read_text()) if x['sample_token'] in tokens};by_sample={}
for a in ann.values():by_sample.setdefault(a['sample_token'],[]).append(a)
instances={x['token']:x for x in json.loads((M/'instance.json').read_text())};categories={x['token']:x['name'] for x in json.loads((M/'category.json').read_text())};rows=[]
order=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
def transform(rec):
    p=np.eye(4);p[:3,:3]=Quaternion(rec['rotation']).rotation_matrix;p[:3,3]=rec['translation'];return p
for scene,spec in src.items():
    meta=json.loads((R/'metadata'/f'{scene}.json').read_text());samples=meta['sample'];sd=meta['sample_data'];ego=meta['ego_pose'];cal=meta['calibrated_sensor']
    def pose(token):return transform(ego[sd[token]['ego_pose_token']])
    for j in range(8):
        name=scene+f'_start{j:02d}';seq=[samples[t] for t in spec['samples'][j:j+10]];s=seq[0];front=sd[s['data']['CAM_FRONT']];world=pose(front['token']);inv=np.linalg.inv(world);views=[]
        for k,ss in enumerate(seq[:2]):
            for camera in order:
                d=sd[ss['data'][camera]];c=cal[d['calibrated_sensor_token']];image=R/'data'/d['filename'];we=pose(d['token'])
                views.append({'sample_token':ss['token'],'sample_index':k,'camera':camera,'image':str(image),'timestamp_us':d['timestamp'],'world_from_camera':(we@transform(c)).tolist(),'world_from_ego_camera':we.tolist(),'intrinsics_original':c['camera_intrinsic'],'original_wh':list(Image.open(image).size)})
        (B/'inputs'/f'{name}.json').write_text(json.dumps({'scene':name,'source_scene':scene,'views':views,'role':'REAL_SENSOR_BASELINE_ONLY; no reconstruction inference'},indent=2))
        records=[]
        for k,ss in enumerate(seq):
            fd=sd[ss['data']['CAM_FRONT']];ep=inv@pose(fd['token']);boxes=[]
            for a in by_sample.get(ss['token'],[]):
                category=categories[instances[a['instance_token']]['category_token']]
                if not category.startswith(('vehicle.','human.','movable_object.barrier','movable_object.trafficcone')):continue
                ap=inv@transform(a);w,l,h=a['size'];boxes.append({'instance':a['instance_token'],'category':category,'box':[*ap[:3,3].tolist(),w,l,h,float(np.arctan2(ap[1,0],ap[0,0]))],'lidar_points':a['num_lidar_pts']})
            records.append({'index':k,'time_s':(fd['timestamp']-front['timestamp'])/1e6,'pose':ep.tolist(),'xy_yaw':[ep[0,3],ep[1,3],float(np.arctan2(ep[1,0],ep[0,0]))],'boxes':boxes})
        past=[front];past.append(sd[past[-1]['prev']]);past.append(sd[past[-1]['prev']]);past_poses=[inv@pose(d['token']) for d in past];positions=[p[:2,3] for p in past_poses];times=np.array([d['timestamp'] for d in past])/1e6;dt=np.diff(-times)
        v=(positions[0]-positions[1])/dt[0];old=(positions[1]-positions[2])/dt[1];a=(v-old)/np.mean(dt)
        yaw_rate=float(-np.arctan2(past_poses[1][1,0],past_poses[1][0,0])/dt[0])
        ref={'scene':name,'source_scene':scene,'world_from_ego_initial':world.tolist(),'initial_velocity_xy_mps':v.tolist(),'initial_acceleration_xy_mps2':a.tolist(),'initial_yaw_rate_rad_s':yaw_rate,'records':records,
             'status_source':'Backward finite differences from current and two previous CAM_FRONT sensor records, including causal yaw rate; no future state input. Steering derived by the same Pacifica bicycle relation as native loop.',
             'reference':'Future logged ego and actor boxes are evaluator only. Official inherited high-level command remains [0,1,0,0], not GT future routing.'}
        (L/f'{name}_log_reference.json').write_text(json.dumps(ref,indent=2))
        d=sd[s['data']['LIDAR_TOP']];wl=pose(d['token'])@transform(cal[d['calibrated_sensor_token']]);T=inv@wl;raw=np.fromfile(R/'data'/d['filename'],np.float32).reshape(-1,5)[:,:3];ranges=np.linalg.norm(raw,axis=1);valid=np.isfinite(raw).all(1)&(ranges>1)&(ranges<80);raw=raw[valid];ranges=ranges[valid];points=raw@T[:3,:3].T+T[:3,3];origin=T[:3,3];directions=(points-origin)/ranges[:,None]
        out=L/'scans'/name;out.mkdir(parents=True);np.savez_compressed(out/'real.npz',points=points,ranges=ranges,origin=origin,directions=directions,world_from_ego=world)
        row={'scene':scene,'case_id':name,'original_start_index':spec['original_start_index']+j,'sample_token':s['token'],'points':len(points),'initial_velocity_xy_mps':v.tolist(),'initial_acceleration_xy_mps2':a.tolist(),'initial_yaw_rate_rad_s':yaw_rate,'reference_horizon_last_time_s':records[-1]['time_s'],'reference_last_heading_change_rad':records[-1]['xy_yaw'][2]};rows.append(row);print('REAL_INPUT_READY',row,flush=True)
reg={'task_id':'WS-SIM-FRESH-NATIVE-01','run_id':'real_baseline_r1','scope':'Eight real logged starts per metadata-selected new log; official TransFuser/PDM adapter, not sensor feedback. All results preserved before fitting admission.',
    'rows':rows,'expected_policy_forwards':len(rows),'expected_vehicle_forecasts':len(rows),'high_level_command':[0,1,0,0],'command_source':'Unchanged official HUGSIM client default. No future GT route or motion input. Report this limitation if baseline fails.',
    'failure_ledger_refs':['V74-H2-F21'],'failure_ledger_delta':'pending','human_verdict':None,'prepared':True,'prepared_unix':time.time()}
(B/'registration.json').write_text(json.dumps(reg,indent=2));print('PREPARED_REAL_BASELINE',len(rows),flush=True)
