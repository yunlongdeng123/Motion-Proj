"""AV2 static build background and raw heldout beams with per-return rigid time."""
import argparse,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.av2_geometry import AV2MetricLog,sizes_at
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH,pca_patch_surface


def carve(vertices,faces,build_rays,margin=.2,chunk=8192):
    scene=SurfaceBVH(vertices,faces); removed=np.zeros(len(faces),bool); contradicted=0
    if scene.scene is not None:
        for origins,directions,ranges in build_rays:
            for start in range(0,len(ranges),chunk):
                rays=np.concatenate([origins[start:start+chunk],directions[start:start+chunk]],axis=1).astype(np.float32)
                hit=scene.scene.list_intersections(o3d.core.Tensor(rays),nthreads=2)
                ids=hit['ray_ids'].numpy().astype(np.int64); distance=hit['t_hit'].numpy()
                bad=(distance>1e-6)&(distance<ranges[start+ids]-margin)
                removed[hit['primitive_ids'].numpy()[bad]]=True
                contradicted+=len(np.unique(ids[bad]))
    del scene
    kept=np.flatnonzero(~removed); scene=SurfaceBVH(vertices,faces[kept]); remaining=0; free_sum=0.
    for origins,directions,ranges in build_rays:
        depth=scene.cast(origins,directions); finite=np.isfinite(depth)
        remaining+=int((finite&(depth<ranges-margin)).sum())
        free_sum+=float(np.maximum(ranges[finite]-margin-depth[finite],0).sum())
    return kept,{'original_triangles':len(faces),'removed_triangles':int(removed.sum()),
                 'build_conflicting_rays_before':contradicted,'remaining_build_early_rays':remaining,
                 'remaining_build_free_sum_m':free_sum,'margin_m':margin,
                 'approximation':'whole triangle removal; one all-returned-intersections pass, coincident BVH hits can be coalesced'}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('status.json',{'status':'running','phase':'metadata'})
    try:
        actor_index=json.loads((args.actor_data/'index.json').read_text())
        cohort=[entry for entry in actor_index['cases'] if entry['log_id']==args.log.name]
        row=next((r for r in actor_index.get('logs',[]) if r['log_id']==args.log.name),actor_index)
        stamps={r['sample_index']:r['timestamp_ns'] for r in row['scans'] if 'timestamp_ns' in r}
        data=AV2MetricLog(args.log); tracks=data.load_tracks(); name='av2-'+args.log.name
        role=next(r['role'] for r in actor_index['scenes'] if r['log_id']==args.log.name)
        folder=args.output/name; folder.mkdir(); owners={owner:i+1 for i,owner in enumerate(sorted(tracks))}
        save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'log':str(args.log),'actor_data':str(args.actor_data),'role':role,'pose_mode':'per_return',
            'background':'same four build sweeps; reference-ego compensated world endpoints outside all valid known boxes+0.1m',
            'carving':'build-only original central beams before measured first return minus0.2m; explicit faces removed, no opacity',
            'uncarved_control':'background_uncarved.npz; same points and raw heldout observations',
            'time':'sensor origins and Actor membership at per-return ns; world endpoint uses reference ego transform once',
            'ownership':'unique valid-time annotation boxes; overlap separate; outside-known-box is proxy, not background semantic GT',
            'sensor_near_zone':'range<1m reporting stratum only; no source or evaluation removal based on this stratum',
            'unknown_pose':'invalid sensor pose excluded from build support; heldout raw beams retained as missing predictions',
            'references':['https://github.com/argoverse/av2-api/blob/main/src/av2/structures/sweep.py'],
            'external_confirmation_values_processed':role=='external_confirmation','model_quality_computed':False})
        backgrounds=[]; build_rays=[]; frames=[]; counts=[]
        for index,stamp in sorted(stamps.items()):
            sweep=data.sweep(args.log/'sensors/lidar'/f'{stamp}.feather'); n=sweep['raw_points']
            membership=np.zeros(n,np.int16); labels=np.zeros(n,np.int32); boundary=np.zeros(n,bool)
            for owner,track in tracks.items():
                local,_,_,known=data.actor_coordinates(owner,sweep)
                extent=sizes_at(track,sweep['point_timestamps_ns'])/2
                inside=known&np.all(np.abs(local)<=extent+.1,axis=1)
                membership+=inside; labels[inside]=owners[owner]
                q=np.abs(local)-extent
                distance=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
                boundary|=known&(np.abs(distance)<=.2)
            labels[membership>1]=-2
            origins=sweep['origins_world_m'].astype(np.float32); directions=sweep['directions_world'].astype(np.float32)
            ranges=sweep['observed_range_m'].astype(np.float32); known=sweep['valid_sensor_pose']
            if index in [5,15,20,30]:
                keep=known&(membership==0)
                backgrounds.append(sweep['points_world_m'][keep].astype(np.float32))
                build_rays.append((origins[known],directions[known],ranges[known]))
                counts.append({'sample_index':index,'sample_id':str(stamp),'rays':n,
                               'invalid_sensor_pose':int((~known).sum()),'excluded_known_objects':int((membership>0).sum()),
                               'background_points':int(keep.sum())})
            else:
                filename=f'rays_{index}.npz'
                np.savez(folder/filename,origin_world_m=origins,directions_world=directions,
                         observed_first_range_m=ranges,observed_owner=labels,box_boundary_band=boundary,
                         sensor_near_zone=ranges<1.,sensor_pose_known=known,point_timestamps_ns=sweep['point_timestamps_ns'])
                frames.append({'sample_index':index,'sample_id':str(stamp),'timestamp_ns':stamp,'file':filename,'rays':n})
            save('status.json',{'status':'running','phase':'scan_geometry','sample_index':index,'raw_returns':n})
        points=np.unique(np.concatenate(backgrounds),axis=0); del backgrounds
        vertices,faces=pca_patch_surface(points)
        np.savez(folder/'background_uncarved.npz',vertices_world_m=vertices,faces=faces)
        save('status.json',{'status':'running','phase':'build_free_carving','background_triangles':len(faces)})
        kept,carving=carve(vertices,faces,build_rays)
        np.savez(folder/'background.npz',vertices_world_m=vertices,faces=faces[kept],retained_parent_face_ids=kept)
        trajectories={entry['owner']:{'timestamps_ns':tracks[entry['owner']]['poses'].times.tolist(),
                                     'world_from_actor':tracks[entry['owner']]['poses'].matrices.tolist()}
                      for entry in cohort}
        scene={'scene':name,'log_id':args.log.name,'role':role,'cohort':cohort,'cohort_ids':[owners[e['owner']] for e in cohort],
               'owner_ids':owners,'pose_mode':'per_return','actor_trajectories':trajectories,'frames':frames,'build':counts,
               'background_points':len(points),'background_triangles':len(kept),'carving':carving}
        (folder/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
        result={'status':'done','scenes':[scene],'wall_s':time.monotonic()-started,
                'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        save('index.json',result); save('status.json',{'status':'done'})
        print(json.dumps({'status':'done','scene':name,'actors':len(cohort),'heldout_rays':sum(f['rays'] for f in frames),
                          'wall_s':result['wall_s'],'peak_rss_gib':result['peak_rss_gib'],'carving':carving}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
