"""Remove explicit background triangles contradicted by original build beams.

This is conservative triangle carving, not a TSDF reproduction. Only positive
first-return build measurements are read; heldout ray files are linked unchanged.
All returned pre-return intersections are considered, including later surfaces.
The BVH may coalesce coincident hits, so this single pass is not a guarantee of
zero residual build intrusion. No learned transparency or target-time rejection.
"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex,_transform
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-data',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--margin-m',type=float,default=.2)
    parser.add_argument('--ray-chunk',type=int,default=8192)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    manifest={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'parent_scene_data':str(args.scene_data),'dataset_root':str(args.dataset_root),'margin_m':args.margin_m,
        'method':'remove triangle if any original build first-return central ray intersects it in (0, range-margin)',
        'intersection_query':'one list_intersections pass; equal-depth coincident hits may be coalesced by Open3D',
        'ray_chunk':args.ray_chunk,'sensor_poses':'same scan-time nuScenes calibration as parent',
        'source_selection':'same parent background; no new surface support or actor changes',
        'heldout_values_read':False,'unknown_space':'unobserved directions and space beyond first return unchanged',
        'approximation':'whole triangle removed, not exact clipped subtriangle; coverage cost reported; finite width not assumed',
        'references':['https://lightfield.stanford.edu/papers/volrange/',
                      'https://voxblox.readthedocs.io/en/latest/pages/The-Voxblox-Node.html',
                      'https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html']}
    save('manifest.json',manifest); save('status.json',{'status':'running','phase':'load_metadata'})
    try:
        source=json.loads((args.scene_data/'index.json').read_text()); output=[]
        index=NuScenesCameraIndex(args.dataset_root)
        for scene in source['scenes']:
            tick=time.monotonic(); name=scene['scene']; folder=args.output/name; folder.mkdir()
            original=np.load(args.scene_data/name/'background.npz')
            vertices=original['vertices_world_m']; faces=original['faces']
            bvh=SurfaceBVH(vertices,faces); rejected=np.zeros(len(faces),bool); records=[]; build_rays=[]
            scene_token=next(s['token'] for s in index.scenes if s['name']==name)
            samples=sorted([s for s in index.samples if s['scene_token']==scene_token],key=lambda s:s['timestamp'])
            for build in scene['build']:
                sample=samples[build['sample_index']]
                lidar=index.sample_data[index.data_by_sample_channel[(sample['token'],'LIDAR_TOP')]]
                world=index.sensor_points_world(sample['token']).astype(np.float64)
                calibrated=index.calibrated[lidar['calibrated_sensor_token']]; ego=index.ego_poses[lidar['ego_pose_token']]
                pose=_transform(ego['translation'],ego['rotation'])@_transform(calibrated['translation'],calibrated['rotation'])
                vector=world-pose[:3,3]; ranges=np.linalg.norm(vector,axis=1)
                valid=np.isfinite(ranges)&(ranges>0); ranges=ranges[valid]
                directions=(vector[valid]/ranges[:,None]).astype(np.float32)
                build_rays.append((pose[:3,3],directions,ranges))
                counts={'sample_index':build['sample_index'],'sample_id':sample['token'],'build_rays':len(ranges),
                        'all_intersections':0,'free_intersections':0,'rays_contradicting_background':0}
                before=int(rejected.sum())
                if bvh.scene is not None:
                    for start in range(0,len(ranges),args.ray_chunk):
                        dirs=directions[start:start+args.ray_chunk]
                        rays=np.concatenate([np.broadcast_to(pose[:3,3],dirs.shape),dirs],axis=1).astype(np.float32)
                        hits=bvh.scene.list_intersections(o3d.core.Tensor(rays),nthreads=2)
                        ids=hits['ray_ids'].numpy().astype(np.int64)
                        distance=hits['t_hit'].numpy(); triangles=hits['primitive_ids'].numpy().astype(np.int64)
                        bad=(distance>1e-6)&(distance<ranges[start+ids]-args.margin_m)
                        rejected[triangles[bad]]=True
                        counts['all_intersections']+=len(ids)
                        counts['free_intersections']+=int(bad.sum())
                        counts['rays_contradicting_background']+=len(np.unique(ids[bad]))
                counts['new_rejected_triangles']=int(rejected.sum())-before
                records.append(counts)
                save('status.json',{'status':'running','phase':'build_carving','scene':name,
                                   'rejected_triangles':int(rejected.sum()),**counts})
            retained=np.flatnonzero(~rejected)
            np.savez(folder/'background.npz',vertices_world_m=vertices,faces=faces[retained],retained_parent_face_ids=retained)
            del bvh
            remaining=SurfaceBVH(vertices,faces[retained]); after_early=0; after_free=0.
            for origin,directions,ranges in build_rays:
                depth=remaining.cast(origin,directions)
                finite=np.isfinite(depth); after_early+=int((finite&(depth<ranges-args.margin_m)).sum())
                after_free+=float(np.maximum(ranges[finite]-args.margin_m-depth[finite],0).sum())
            del remaining,build_rays
            for frame in scene['frames']:
                (folder/frame['file']).symlink_to((args.scene_data/name/frame['file']).resolve())
            entry={**scene,'background_triangles':len(retained),
                'carving':{'original_triangles':len(faces),'removed_triangles':int(rejected.sum()),
                           'active_vertices':len(np.unique(faces[retained])),'build':records,
                           'remaining_build_early_rays':after_early,'remaining_build_free_sum_m':after_free,
                           'wall_s':time.monotonic()-tick}}
            (folder/'scene.json').write_text(json.dumps(entry,indent=2)+'\n'); output.append(entry)
            print(json.dumps({'scene':name,**{k:v for k,v in entry['carving'].items() if k!='build'}}),flush=True)
            del original,vertices,faces,rejected
        result={'status':'done','scenes':output,'wall_s':time.monotonic()-started,
                'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                'boundary':manifest['method']+'; build only, same heldout rays/owners/poses, no target-time carving'}
        save('index.json',result); save('status.json',{'status':'done'})
        print(json.dumps({k:v for k,v in result.items() if k!='scenes'}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
