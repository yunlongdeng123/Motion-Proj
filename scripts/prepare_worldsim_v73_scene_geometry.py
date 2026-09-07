"""build侧排除所有已知注释物体的固定背景；额外时刻仅保存评价原始束。"""
import argparse,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.actor_rays import ActorRayDataset
from motion_proj.worldsim_v73.native_data import interpolate_pose
from motion_proj.worldsim_v72.data.nuscenes_camera import _transform
from motion_proj.worldsim_v73.scene_readout import pca_patch_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--role',choices=['fit','development','all'],default='development')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); torch.set_num_threads(2)
    def save(name,value):
        p=args.output/name; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(p)
    save('status.json',{'status':'running','phase':'input'})
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'native_run':str(args.native_run),'actor_data':str(args.actor_data),'role':args.role,
        'background':'all build first-return points outside EVERY known annotation box +0.1m at that scan time; fixed .06m PCA patches, no filling',
        'evaluation':'all original positive-range LiDAR beams at nonbuild sample indices modulo3==2 inside build window',
        'ownership':'unique known box+0.1m proxy; overlapping boxes ambiguous; outside all known boxes background proxy, not semantic GT',
        'boundary_band':'observed first-return endpoints within .2m of annotation box surface; not true surface/contact boundary',
        'time_boundary':'scan timestamp; known rigid translation/Slerp; no per-point deskew; no extrapolation',
        'source_test_read':False,'external_test_read':False})
    try:
        population=json.loads((args.actor_data/'index.json').read_text())['cases']
        scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)
        scenes=[s for s in scenes if args.role=='all' or s['role']==args.role]
        sensor=ActorRayDataset('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval',{s['scene_id'] for s in scenes})
        index=sensor.index; output=[]
        for scene in scenes:
            name=scene['scene_id']; target=args.output/name; target.mkdir()
            tracks=sensor.tracks[name]; owners={owner:i+1 for i,owner in enumerate(sorted(tracks))}
            cohort=[r for r in population if r['scene']==name]
            cohort_ids=[owners[r['owner']] for r in cohort]
            token=next(s['token'] for s in index.scenes if s['name']==name)
            samples=sorted([s for s in index.samples if s['scene_token']==token],key=lambda s:s['timestamp'])
            build_ids={v['sample_id'] for v in scene['views']}
            positions=[i for i,s in enumerate(samples) if s['token'] in build_ids]
            backgrounds=[]; frames=[]; build_counts=[]
            for i,sample in enumerate(samples):
                if not min(positions)<=i<=max(positions): continue
                is_build=sample['token'] in build_ids
                if not is_build and i%3!=2: continue
                lidar=index.sample_data[index.data_by_sample_channel[(sample['token'],'LIDAR_TOP')]]
                stamp=int(lidar['timestamp']); world=index.sensor_points_world(sample['token']).astype(np.float64)
                calibrated=index.calibrated[lidar['calibrated_sensor_token']]; ego=index.ego_poses[lidar['ego_pose_token']]
                sensor_pose=_transform(ego['translation'],ego['rotation'])@_transform(calibrated['translation'],calibrated['rotation'])
                origin=sensor_pose[:3,3]; vector=world-origin; ranges=np.linalg.norm(vector,axis=-1)
                valid=np.isfinite(ranges)&(ranges>0)
                world=world[valid]; vector=vector[valid]; ranges=ranges[valid]
                membership=np.zeros(len(world),np.int32); label=np.zeros(len(world),np.int32)
                boundary=np.zeros(len(world),bool); poses={}
                for owner,trajectory in tracks.items():
                    pose=interpolate_pose(trajectory,stamp)
                    if pose is None: continue
                    poses[owner]=pose.tolist()
                    local=(world-pose[:3,3])@pose[:3,:3]
                    extent=min(trajectory,key=lambda r:abs(r[0]-stamp))[2]/2
                    inside=np.all(np.abs(local)<=extent+.1,axis=-1)
                    membership+=inside; label[inside]=owners[owner]
                    q=np.abs(local)-extent
                    distance=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
                    boundary|=np.abs(distance)<=.2
                label[membership>1]=-2
                if is_build:
                    backgrounds.append(world[membership==0].astype(np.float32))
                    build_counts.append({'sample_index':i,'rays':len(world),'excluded_known_objects':int((membership>0).sum()),
                                         'background_points':int((membership==0).sum())})
                else:
                    filename='rays_'+str(i)+'.npz'
                    np.savez(target/filename,origin_world_m=origin.astype(np.float32),directions_world=(vector/ranges[:,None]).astype(np.float32),
                        observed_first_range_m=ranges.astype(np.float32),observed_owner=label,box_boundary_band=boundary)
                    frames.append({'sample_index':i,'sample_id':sample['token'],'timestamp_us':stamp,'file':filename,
                                   'rays':len(world),'world_from_actor':poses})
            background=np.unique(np.concatenate(backgrounds),axis=0)
            vertices,faces=pca_patch_surface(background)
            np.savez(target/'background.npz',vertices_world_m=vertices,faces=faces)
            entry={'scene':name,'role':scene['role'],'log_id':scene['log_id'],'owner_ids':owners,'cohort_ids':cohort_ids,
                'cohort':cohort,'build':build_counts,'frames':frames,'background_points':len(background),'background_triangles':len(faces)}
            (target/'scene.json').write_text(json.dumps(entry,indent=2)+'\n'); output.append(entry)
            save('status.json',{'status':'running','scenes_done':len(output),'scene':name,'elapsed_s':time.monotonic()-started})
            print(json.dumps({'scene':name,'background_points':len(background),'evaluation_rays':sum(f['rays'] for f in frames)}),flush=True)
        save('index.json',{'status':'done','scenes':output,'wall_s':time.monotonic()-started,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})
        save('status.json',{'status':'done'})
    except Exception as exc:
        save('status.json',{'status':'blocked','message':str(exc),'exception':type(exc).__name__})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
