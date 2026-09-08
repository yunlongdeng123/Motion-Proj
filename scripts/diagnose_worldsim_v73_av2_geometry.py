"""One real old-development log diagnostic: beam geometry, time and camera domains."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.av2_geometry import AV2MetricLog,RING_CAMERAS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--log',type=Path,default=Path('/root/autodl-tmp/data/av2/sensor/val/02678d04-cc9f-3148-9f95-1ba66347dff9'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); start=time.monotonic()
    data=AV2MetricLog(args.log)
    paths=sorted((args.log/'sensors/lidar').glob('*.feather'))
    path=paths[5]
    candidates=[]
    for convention in ['up_lidar','down_lidar']:
        sweep=data.sweep(path,lower_ids_sensor=convention)
        sensor=sweep['world_from_sensor']
        local=np.einsum('nji,nj->ni',sensor[:,:3,:3],sweep['points_world_m']-sensor[:,:3,3])
        elevation=np.degrees(np.arctan2(local[:,2],np.linalg.norm(local[:,:2],axis=1)))
        rings=[]
        for ring in range(64):
            chosen=(sweep['laser_number']==ring)&sweep['valid_sensor_pose']
            angles=elevation[chosen]
            rings.append({'laser_number':ring,'returns':len(angles),
                'median_elevation_deg':float(np.median(angles)) if len(angles) else None,
                'median_absolute_deviation_deg':float(np.median(np.abs(angles-np.median(angles)))) if len(angles) else None})
        candidates.append({'lower_ids_sensor':convention,'rings':rings,
            'median_ring_mad_deg':float(np.median([r['median_absolute_deviation_deg'] for r in rings if r['returns']]))})
    preferred=min(candidates,key=lambda row:row['median_ring_mad_deg'])['lower_ids_sensor']
    sweep=data.sweep(path,lower_ids_sensor=preferred)
    # Compare against the common scan-time origin approximation on the same raw returns.
    reference=sweep['reference_world_from_ego']; laser=sweep['laser_number']
    other='down_lidar' if preferred=='up_lidar' else 'up_lidar'
    base_sensor=np.where((laser<32)[:,None],data.ego_from_sensor[preferred][:3,3],data.ego_from_sensor[other][:3,3])
    old_origin=base_sensor@reference[:3,:3].T+reference[:3,3]
    origin_change=np.linalg.norm(old_origin-sweep['origins_world_m'],axis=1)
    cameras=[]
    for channel in RING_CAMERAS:
        files=sorted((args.log/'sensors/cameras'/channel).glob('*.jpg'))
        file=min(files,key=lambda p:abs(int(p.stem)-sweep['timestamp_ns']))
        camera=data.camera(channel,file)
        direction=camera['world_from_camera'][:3,:3][:,2]
        ego,_=data.ego.at(camera['timestamp_ns'])
        direction=ego[:3,:3].T@direction
        cameras.append({'channel':channel,'timestamp_ns':camera['timestamp_ns'],
            'lidar_delta_ns':camera['timestamp_ns']-sweep['timestamp_ns'],
            'optical_axis_ego':direction.tolist(),'optical_axis_yaw_deg':float(np.degrees(np.arctan2(direction[1],direction[0]))),
            'image_hw':list(camera['image'].shape[-2:]),'valid_image_rect_xyxy':camera['valid_image_rect_xyxy'],
            'intrinsics':camera['intrinsics'].tolist(),'pose_known':camera['pose_known']})
    tracks=data.load_tracks()
    result={'status':'done','log_id':args.log.name,'role':'already_exposed_AV2_development',
        'new_confirmation_sensor_values_read':False,'scan_filename':str(path),'raw_returns':sweep['raw_points'],
        'invalid_pose_returns':int((~sweep['valid_sensor_pose']).sum()),
        'offset_ms_range':((np.array([sweep['point_timestamps_ns'].min(),sweep['point_timestamps_ns'].max()])-sweep['timestamp_ns'])/1e6).tolist(),
        'laser_group_candidates':candidates,'geometry_preferred_lower_ids_sensor':preferred,
        'laser_mapping_evidence':'old-development within-ring elevation dispersion under fixed provided calibration; inference, not a retrieved official index declaration',
        'scan_origin_approx_error_m_p50_p95_max':[float(np.median(origin_change)),float(np.quantile(origin_change,.95)),float(origin_change.max())],
        'camera_observations':cameras,'known_tracks':len(tracks),
        'scope':'input geometry only, no network inference or evaluation-method selection',
        'wall_s':time.monotonic()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['laser_group_candidates','camera_observations']}),flush=True)


if __name__=='__main__': main()
