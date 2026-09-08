"""Export an AV2 window to the shared V7.3 build-observation and Actor-ray format."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.av2_geometry import (
    AV2MetricLog,RING_CAMERAS,RIGID_VEHICLES,sizes_at,source_camera_yaws,camera_identity_weights)

BUILD=[5,15,20,30]; EVAL=[10,25]


def near_box(origins,directions,extents):
    parallel=np.abs(directions)<1e-10
    inv=1/np.where(parallel,1,directions)
    left=(-extents-origins)*inv; right=(extents-origins)*inv
    lower=np.where(parallel,-np.inf,np.minimum(left,right)).max(1)
    upper=np.where(parallel,np.inf,np.maximum(left,right)).min(1)
    return (upper>=np.maximum(lower,0))&~np.any(parallel&(np.abs(origins)>extents),axis=1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-plan',type=Path)
    parser.add_argument('--role',choices=['development','external_confirmation'],default='development')
    parser.add_argument('--nuscenes-metadata',type=Path,default=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval'))
    args=parser.parse_args(); start=time.monotonic(); torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('status.json',{'status':'running','phase':'metadata'})
    data=AV2MetricLog(args.log); tracks=data.load_tracks(); scene_id='av2-'+args.log.name
    if args.window_plan:
        plans=json.loads(args.window_plan.read_text())['logs']
        plan=next(row for row in plans if row['log_id']==args.log.name)
        stamps={int(k):int(v) for k,v in plan['lidar_timestamps_ns'].items()}
        camera_paths={(v['lidar_index'],v['channel']):args.log/v['filename'] for v in plan['build_camera_observations']}
    else:
        paths=sorted((args.log/'sensors/lidar').glob('*.feather'))
        stamps={index:int(paths[index].stem) for index in BUILD+EVAL}
        camera_paths={}
        for channel in RING_CAMERAS:
            paths=sorted((args.log/'sensors/cameras'/channel).glob('*.jpg'))
            for index in BUILD:
                if paths: camera_paths[(index,channel)]=min(paths,key=lambda p:abs(int(p.stem)-stamps[index]))
    selected={owner:track for owner,track in tracks.items() if track['category'] in RIGID_VEHICLES and
              np.any(track['poses'].at(np.array([stamps[i] for i in BUILD if i in stamps],dtype=np.int64))[1])}
    yaws=source_camera_yaws(args.nuscenes_metadata)
    save('input_protocol.json',{'dataset':'AV2 Sensor','role':args.role,'log_id':args.log.name,
         'selection':'all rigid vehicle tracks with known build-window pose, independent of LiDAR or prediction quality',
         'build_lidar_indices':BUILD,'evaluation_lidar_indices':EVAL,'selected_actors':len(selected),
         'camera_channels':RING_CAMERAS,'image_hw':[672,672],'lower_laser_ids_sensor':'up_lidar',
         'camera_identity':'calibration-azimuth interpolation of existing six nuScenes embeddings; no new parameters',
         'source_camera_yaws_deg':np.degrees(yaws).tolist(),
         'ownership':'per-return-time annotation boxes +0.1m, overlap excluded; proxy, not segmentation GT',
         'no_return_boundary':'observed returns only; no fabricated emitted-but-missing beams',
         'shared_training':False,'new_confirmation':args.role=='external_confirmation',
         'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    actor_frames={owner:[] for owner in selected}; observations=[]; scan_summary=[]
    for index in sorted(stamps):
        stamp=stamps[index]; path=args.log/'sensors/lidar'/f'{stamp}.feather'
        if not path.exists():
            scan_summary.append({'sample_index':index,'status':'missing_payload'}); continue
        sweep=data.sweep(path); count=sweep['raw_points']
        membership=np.zeros(count,dtype=np.int16); owners=np.full(count,'',dtype='U36')
        locals_by_owner={}; canonical=np.zeros((count,3),dtype=float)
        for owner,track in tracks.items():
            local,origins,directions,known=data.actor_coordinates(owner,sweep)
            inside=known&np.all(np.abs(local)<=sizes_at(track,sweep['point_timestamps_ns'])/2+.1,axis=1)
            first=inside&(membership==0); owners[first]=owner; canonical[first]=local[first]
            membership[inside]+=1
            if owner in selected: locals_by_owner[owner]=(local,origins,directions,known)
        owners[membership>1]='ambiguous'
        for owner,track in selected.items():
            local,origins,directions,known=locals_by_owner[owner]
            size=sizes_at(track,sweep['point_timestamps_ns'])
            near=known&near_box(origins,directions,size/2+.5)
            positive=(owners==owner)&(membership==1)
            actor_frames[owner].append({'sample_id':str(stamp),'sample_index':index,'timestamp_ns':stamp,'timestamp_us':stamp//1000,
                'point_timestamps_ns':torch.from_numpy(sweep['point_timestamps_ns'][near]),
                'role':'build' if index in BUILD else 'heldout_time',
                'origins_actor_m':torch.tensor(origins[near],dtype=torch.float32),
                'directions_actor':torch.tensor(directions[near],dtype=torch.float32),
                'observed_first_range_m':torch.tensor(sweep['observed_range_m'][near],dtype=torch.float32),
                'positive_actor':torch.from_numpy(positive[near]),'ambiguous_owner':torch.from_numpy((membership>1)[near]),
                'points_actor_m':torch.tensor(local[near],dtype=torch.float32),
                'size_lwh_m':torch.tensor(sizes_at(track,stamp),dtype=torch.float32),
                'raw_scan_points':count,'near_box_rays':int(near.sum()),'owned_points':int((positive&near).sum()),
                'pose_unknown_returns':int((~known).sum())})
        scan_summary.append({'sample_index':index,'timestamp_ns':stamp,'raw_returns':count,
                             'ambiguous_returns':int((membership>1).sum()),'sensor_pose_unknown':int((~sweep['valid_sensor_pose']).sum())})
        if index in BUILD:
            rigid=np.isin(owners,list(selected)); background=owners==''
            for channel in RING_CAMERAS:
                camera_path=camera_paths.get((index,channel))
                if camera_path is None or not camera_path.is_file(): continue
                camera=data.camera(channel,camera_path)
                if not camera['pose_known']: continue
                moved=sweep['points_world_m'].copy(); valid=sweep['valid_sensor_pose']&(rigid|background)
                actor_poses={}
                for owner,track in selected.items():
                    pose,known=track['poses'].at(camera['timestamp_ns'])
                    take=owners==owner
                    if known:
                        actor_poses[owner]=pose
                        moved[take]=canonical[take]@pose[:3,:3].T+pose[:3,3]
                    else: valid[take]=False
                camera_pose=camera['world_from_camera']
                xyz=(moved-camera_pose[:3,3])@camera_pose[:3,:3]
                homogeneous=xyz@camera['intrinsics'].T
                uv=homogeneous[:,:2]/np.maximum(homogeneous[:,2:],1e-8)
                left,top,right,bottom=camera['valid_image_rect_xyxy']
                valid&=(xyz[:,2]>.5)&(uv[:,0]>=left)&(uv[:,0]<=right-1)&(uv[:,1]>=top)&(uv[:,1]<=bottom-1)
                ids=np.flatnonzero(valid); pixels=np.rint(uv[ids]).astype(int)
                flat=pixels[:,1]*672+pixels[:,0]
                order=np.argsort(xyz[ids,2],kind='stable')
                _,unique=np.unique(flat[order],return_index=True); ids=ids[order[unique]]
                weights=camera_identity_weights(data.ego_from_sensor[channel],yaws)
                observations.append({'image':torch.from_numpy(camera['image']),
                    'uv':torch.tensor(uv[ids],dtype=torch.float32),'z_m':torch.tensor(xyz[ids,2],dtype=torch.float32),
                    'actor_mask':torch.from_numpy(rigid[ids]),'diagnostic_mask':torch.from_numpy(ids%5==0),
                    'owners':owners[ids].tolist(),'points_actor_m':torch.tensor(canonical[ids],dtype=torch.float32),
                    'world_from_actor':actor_poses,'world_from_camera':camera_pose,'intrinsics':camera['intrinsics'],
                    'sample_id':str(stamp),'camera_id':channel,'camera_time_ns':camera['timestamp_ns'],
                    'camera_time_us':camera['timestamp_ns']//1000,'lidar_time_us':stamp//1000,'lidar_time_ns':stamp,
                    'image_path':str(camera_path),'valid_image_rect_xyxy':camera['valid_image_rect_xyxy'],
                    'camera_embedding_weights':torch.from_numpy(weights)})
        save('status.json',{'status':'running','phase':'scans','scans_done':len(scan_summary),'views':len(observations)})
        print(json.dumps(scan_summary[-1]),flush=True)
    scene={'scene_id':scene_id,'log_id':args.log.name,'role':args.role,'requested_times':4,
           'available_times':len({v['sample_id'] for v in observations}),'views':observations,
           'window_actor_count':len(selected),'actor_count':len(selected),
           'point_ownership':'per-return-time unique box +0.1m proxy'}
    torch.save([scene],args.output/'build_observations.pt')
    entries=[]
    for owner,track in selected.items():
        rays=actor_frames[owner]; build=[r for r in rays if r['role']=='build']
        parts=[r['points_actor_m'][r['positive_actor']] for r in build]
        points=torch.unique(torch.cat(parts),dim=0) if parts else torch.empty(0,3)
        views=[i for i,view in enumerate(observations) if owner in view['world_from_actor']]
        subset=[observations[i] for i in views]; times=np.array([stamps[i] for i in BUILD if i in stamps],dtype=np.int64)
        poses,known=track['poses'].at(times); active=times[known]
        duration=float((active[-1]-active[0])/1e9) if len(active)>1 else 0
        speed=float(np.linalg.norm(poses[known][-1,:3,3]-poses[known][0,:3,3])/duration) if duration else None
        size=torch.tensor(np.median(sizes_at(track,active),axis=0),dtype=torch.float32)
        entry={'scene':scene_id,'log_id':args.log.name,'role':args.role,'owner':owner,'category':track['category'],
            'selection':'all known rigid vehicles with build-window pose','window_duration_s':duration,
            'translation_speed_mps':speed,'build_points':len(points),'actual_views':len(views),
            'heldout_frames':sum(r['role']=='heldout_time' for r in rays),
            'heldout_actor_returns':sum(r['owned_points'] for r in rays if r['role']=='heldout_time'),
            'status':'ready' if len(points) else 'unavailable_input',
            'file':scene_id+'__'+owner+'.pt'}
        if not len(points): entry['reason']='no measured build LiDAR; current main input protocol retains empty prediction'
        reference=subset[0]['camera_time_ns'] if subset else 0
        tensor=lambda key,shape:torch.tensor(np.stack([v[key] for v in subset]),dtype=torch.float32) if subset else torch.empty(shape)
        matrices=[np.linalg.inv(v['world_from_camera'])@v['world_from_actor'][owner] for v in subset]
        case={'metadata':entry,'rays':rays,'points_actor_m':points,'size_lwh_m':size,'view_indices':views,
            'camera_from_actor':torch.tensor(np.stack(matrices),dtype=torch.float32) if matrices else torch.empty(0,4,4),
            'intrinsics':tensor('intrinsics',(0,3,3)),
            'camera_ids':torch.tensor([RING_CAMERAS.index(v['camera_id']) for v in subset],dtype=torch.long),
            'camera_embedding_weights':torch.stack([v['camera_embedding_weights'] for v in subset]) if subset else torch.empty(0,6),
            'valid_image_rect_xyxy':tensor('valid_image_rect_xyxy',(0,4)),
            'time_offsets_s':torch.tensor([(v['camera_time_ns']-reference)/1e9 for v in subset],dtype=torch.float32),
            'image_hw':[672,672]}
        torch.save(case,args.output/entry['file']); entries.append(entry)
    result={'status':'done','cases':entries,'selection':'all_window_rigid','scenes':[{'scene':scene_id,'role':args.role,'log_id':args.log.name,'input_views':len(observations)}],
            'scans':scan_summary,'pose_boundary':'per-return nanosecond sensor/Actor and per-camera exposure poses; no double ego deskew',
            'source_camera_yaws_deg':np.degrees(yaws).tolist(),'wall_s':time.monotonic()-start,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    save('index.json',result); save('status.json',{'status':'done'})
    print(json.dumps({'status':'done','actors':len(entries),'ready':sum(e['status']=='ready' for e in entries),
                      'views':len(observations),'wall_s':result['wall_s'],'peak_rss_gib':result['peak_rss_gib']}),flush=True)


if __name__=='__main__': main()
