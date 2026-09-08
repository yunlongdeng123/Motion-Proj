"""同一build背景、已知轨迹和全局首交点上的完整场景比较；纯CPU BVH。"""
import argparse,json,resource,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH,pca_patch_surface,compose_first,ray_statistics


def aggregate(rows):
    result={}
    for method in sorted({r['method'] for r in rows}):
        method_rows=[r for r in rows if r['method']==method]
        groups={}
        for group in method_rows[0]['groups']:
            scenes=defaultdict(list)
            for r in method_rows: scenes[(r['log_id'],r['scene'])].append(r['groups'][group])
            logs=defaultdict(list)
            for (log,scene),frames in scenes.items():
                n=sum(f['rays'] for f in frames); returned=sum(f['returned'] for f in frames)
                if not n: continue
                value={metric:sum(f[count] for f in frames)/n for metric,count in
                    [('hit_rate','hit'),('early_rate','early'),('miss_rate','miss'),('free_intrusion_m','free_intrusion_sum_m')]}
                value['returned_mae_m']=sum(f['returned_abs_error_sum_m'] for f in frames)/returned if returned else None
                logs[log].append(value)
            per_log={log:{metric:float(np.mean([v[metric] for v in values if v[metric] is not None]))
                if any(v[metric] is not None for v in values) else None for metric in
                ['hit_rate','early_rate','miss_rate','free_intrusion_m','returned_mae_m']} for log,values in logs.items()}
            groups[group]={'rays':sum(r['groups'][group]['rays'] for r in method_rows),'logs':len(logs),'per_log':per_log,
                'means':{metric:float(np.mean([v[metric] for v in per_log.values() if v[metric] is not None]))
                    if any(v[metric] is not None for v in per_log.values()) else None for metric in
                    ['hit_rate','early_rate','miss_rate','free_intrusion_m','returned_mae_m']}}
        result[method]=groups
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-data',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--model',action='append',default=[],help='NAME=completed surface run path')
    parser.add_argument('--background-file',default='background.npz',help='explicit registered background within each scene folder')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); torch.set_num_threads(2)
    models={name:Path(path) for name,path in [value.split('=',1) for value in args.model]}
    def save(name,value):
        p=args.output/name; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(p)
    manifest={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scene_data':str(args.scene_data),'actor_data':str(args.actor_data),'models':{k:str(v) for k,v in models.items()},
        'background_file':args.background_file,
        'method':'per-owner canonical CPU BVH + inverse rigid rays + global nearest distance; per-return time when scene supplies trajectories',
        'aggregation':'ray weighted within scene, then equal scenes within log and equal independent logs; differs from Actor-only means',
        'ownership':'all known annotation boxes+0.1m proxy, overlaps separate; no learned opacity or trajectory updates',
        'empty_domain':'missing Actor geometry and unobserved background are not filled; all observed beams retained including other annotated objects',
        'evaluation':'scene input roles retained; extra-time rays read for evaluation, no optimizer',
        'source_test_read':False,'external_test_read':False}
    save('manifest.json',manifest)
    save('status.json',{'status':'running','phase':'load'})
    try:
        scenes=json.loads((args.scene_data/'index.json').read_text())['scenes']; rows=[]; supports=[]
        roles=sorted({scene['role'] for scene in scenes})
        manifest.update(roles=roles,external_test_read='external_confirmation' in roles)
        save('manifest.json',manifest)
        for scene in scenes:
            name=scene['scene']; folder=args.scene_data/name
            bg=np.load(folder/args.background_file); background=SurfaceBVH(bg['vertices_world_m'],bg['faces'])
            trajectories={}
            if scene.get('pose_mode')=='per_return':
                from motion_proj.worldsim_v73.av2_geometry import TimedPoses
                trajectories={owner:TimedPoses(value['timestamps_ns'],value['world_from_actor'])
                              for owner,value in scene['actor_trajectories'].items()}
            frames=[]
            for frame in scene['frames']:
                raw=np.load(folder/frame['file']); origins=raw['origin_world_m']; directions=raw['directions_world']
                if 'sensor_pose_known' in raw:
                    known=raw['sensor_pose_known']; bg_depth=np.full(len(directions),np.inf,np.float32)
                    bg_depth[known]=background.cast(np.broadcast_to(origins,directions.shape)[known],directions[known])
                else: bg_depth=background.cast(origins,directions)
                frames.append((frame,{key:raw[key] for key in raw.files},bg_depth))
            for method in ['background_only','lidar_pca',*models]:
                actor_bvhs={}; count=[]
                if method!='background_only':
                    for entry in scene['cohort']:
                        owner=entry['owner']
                        if method=='lidar_pca':
                            case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=False)
                            vertices,faces=pca_patch_surface(case['points_actor_m'].numpy(),count=1536)
                            del case
                        else:
                            path=models[method]/(name+'__'+owner+'_surface.pt')
                            if not path.is_file(): path=models[method]/(owner+'_surface.pt')
                            # Missing artifact is an incomplete experiment, not a model empty prediction.
                            surface=torch.load(path,map_location='cpu',weights_only=False)
                            vertices=surface['vertices_actor_m'].numpy(); faces=surface['faces'].numpy()
                            del surface
                        actor_bvhs[owner]=SurfaceBVH(vertices,faces)
                        count.append({'owner':owner,'triangles':len(faces)})
                    supports.append({'scene':name,'method':method,'actors':count})
                for frame,raw,bg_depth in frames:
                    depths=[]; unavailable=[]; unknown_counts={}
                    for owner,bvh in actor_bvhs.items():
                        if scene.get('pose_mode')=='per_return':
                            trajectory=trajectories.get(owner)
                            if trajectory is None:
                                unavailable.append(owner); unknown_counts[owner]=len(raw['directions_world']); continue
                            depth,known=bvh.cast_per_time(raw['origin_world_m'],raw['directions_world'],
                                raw['point_timestamps_ns'],trajectory,raw.get('sensor_pose_known'))
                            unknown_counts[owner]=int((~known).sum())
                            if not known.any(): unavailable.append(owner)
                            depths.append((scene['owner_ids'][owner],depth))
                            continue
                        pose=frame['world_from_actor'].get(owner)
                        if pose is None: unavailable.append(owner); continue
                        depths.append((scene['owner_ids'][owner],bvh.cast(raw['origin_world_m'],raw['directions_world'],pose)))
                    depth,pred_owner,actor_depth=compose_first(bg_depth,depths)
                    truth=raw['observed_owner']; observed=raw['observed_first_range_m']
                    masks={'all_raw_returns':np.ones(len(observed),bool),'cohort_returns':np.isin(truth,scene['cohort_ids']),
                        'background_proxy_returns':truth==0,'other_annotated_returns':(truth>0)&~np.isin(truth,scene['cohort_ids']),
                        'ambiguous_returns':truth==-2,'annotation_box_boundary_band':raw['box_boundary_band']}
                    moving_ids=[scene['owner_ids'][entry['owner']] for entry in scene['cohort']
                                if (entry.get('translation_speed_mps') or 0)>2]
                    masks['moving_cohort_returns']=np.isin(truth,moving_ids)
                    if 'sensor_near_zone' in raw:
                        masks['sensor_near_zone_returns']=raw['sensor_near_zone']
                        masks['outside_sensor_near_zone_returns']=~raw['sensor_near_zone']
                    if 'sensor_pose_known' in raw: masks['sensor_pose_unknown_returns']=~raw['sensor_pose_known']
                    row={'scene':name,'log_id':scene['log_id'],'role':scene['role'],'sample_index':frame['sample_index'],
                        'method':method,'groups':{key:ray_statistics(depth,observed,mask) for key,mask in masks.items()},
                        'actors_without_pose':unavailable,
                        'actor_unknown_pose_rays':unknown_counts,
                        'background_return_actor_early':int(((truth==0)&(pred_owner>0)&(depth<observed-.2)).sum()),
                        'cohort_return_background_early':int((masks['cohort_returns']&(pred_owner==0)&(depth<observed-.2)).sum()),
                        'unique_proxy_owner_mismatch_returned':int(((truth>=0)&(pred_owner>=0)&(truth!=pred_owner)).sum()),
                        'unique_proxy_returned_denominator':int(((truth>=0)&(pred_owner>=0)).sum()),
                        'actor_background_near_tie_02m':int((np.isfinite(actor_depth)&np.isfinite(bg_depth)&
                            (np.abs(np.where(np.isfinite(actor_depth),actor_depth,0)-np.where(np.isfinite(bg_depth),bg_depth,0))<=.2)).sum())}
                    rows.append(row)
                    np.savez(args.output/(name+'__'+str(frame['sample_index'])+'__'+method+'.npz'),
                        first_depth_m=depth,first_owner=pred_owner,background_depth_m=bg_depth,actor_first_depth_m=actor_depth)
                save('status.json',{'status':'running','scene':name,'method':method,'frames_done':len(rows),'elapsed_s':time.monotonic()-started})
                print(json.dumps({'scene':name,'method':method,'frames_done':len(rows)}),flush=True)
            del background,frames
        save('summary.json',{'status':'done','frames':rows,'supports':supports,'statistics':aggregate(rows),'roles':roles,
            'statistics_by_role':{role:aggregate([row for row in rows if row['role']==role]) for role in roles},
            'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'scene input roles retained; box attribution/boundary proxy; finite build background incomplete; no counterfactual GT'})
        save('status.json',{'status':'done'})
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
