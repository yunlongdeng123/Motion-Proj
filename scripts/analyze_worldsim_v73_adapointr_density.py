"""Separate native point support from fixed-budget and full-density physical surfaces."""
import argparse,json,resource,subprocess,sys,time,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH,pca_patch_surface,ray_statistics
from summarize_worldsim_v73_global_results import summarize_actor,stage_statistics,paired


def point_statistics(rows):
    result={}
    for name in ['native_points','matched_centers']:
        logs=defaultdict(list)
        for row in rows:
            frames=row[name]; count=sum(f['points'] for f in frames)
            if not count: continue
            measured=sum(f['points'] for f in frames if f['distance_sum_m'] is not None)
            logs[row['actor']['log_id']].append({
                'distance_m':sum(f['distance_sum_m'] or 0 for f in frames)/measured if measured else None,
                'recall_02':sum(f['near_02'] for f in frames)/count})
        per_log={log:{metric:float(np.mean([r[metric] for r in values if r[metric] is not None]))
                         if any(r[metric] is not None for r in values) else None
                         for metric in ['distance_m','recall_02']} for log,values in logs.items()}
        result[name]={'per_log':per_log,'logs':len(per_log),'means':{
            metric:float(np.mean([r[metric] for r in per_log.values() if r[metric] is not None]))
            if any(r[metric] is not None for r in per_log.values()) else None for metric in ['distance_m','recall_02']}}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2); started=time.monotonic()
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'run':str(args.run),'actor_data':str(args.actor_data),'role':'development','optimizer_updates':0,
        'scope':'all 75 existing development Actors, including empty input and absent owned returns; fixed final outputs only',
        'density_control':'all native output points as .06m PCA patch centers, 20 nearest native neighbors; no sampling, hole filling, opacity or target-based construction',
        'comparison':'cached matched-budget surface results retained; dense CPU BVH is a density sensitivity, not a matched-budget method gain',
        'point_metrics':'measured target-to-point only; no unobserved predicted-to-sparse-target precision or complete Chamfer claim',
        'references':['https://cgl.ethz.ch/research/past_projects/apss/','https://github.com/hbb1/2d-gaussian-splatting'],
        'migration':'separate point support, surface construction and literal intersection; no APSS/2DGS performance reproduction claim'})
    save('status.json',{'status':'running'})
    try:
        entries=[r for r in json.loads((args.actor_data/'index.json').read_text())['cases'] if r['role']=='development']
        cached=json.loads((args.run/'summary.json').read_text())['final']
        matched=[summarize_actor(r) for r in cached if r['actor']['role']=='development']; dense=[]; points_rows=[]; raw_rows=[]
        for entry in entries:
            owner=entry['owner']; case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=False)
            native=torch.load(args.run/(owner+'_final_points.pt'),map_location='cpu',weights_only=False).numpy()
            surface=torch.load(args.run/(owner+'_surface.pt'),map_location='cpu',weights_only=False)
            centers=surface['centers_actor_m'].numpy(); del surface
            vertices,faces=pca_patch_surface(native); bvh=SurfaceBVH(vertices,faces)
            point_trees={name:cKDTree(value) if len(value) else None for name,value in [('native_points',native),('matched_centers',centers)]}
            point_row={'actor':entry,'native_point_count':len(native),'matched_patch_count':len(centers),
                       'dense_triangle_count':len(faces),'native_points':[],'matched_centers':[]}; frames=[]
            for frame in case['rays']:
                if frame['role']!='heldout_time': continue
                origins=frame['origins_actor_m'].numpy(); directions=frame['directions_actor'].numpy()
                observed=frame['observed_first_range_m'].numpy(); positive=frame['positive_actor'].numpy()
                target=frame['points_actor_m'][frame['positive_actor']].numpy(); depth=bvh.cast(origins,directions)
                distance=bvh.scene.compute_distance(o3d.core.Tensor(target.astype(np.float32)),nthreads=2).numpy() if len(target) and bvh.scene is not None else None
                owned=ray_statistics(depth,observed,positive); all_rays=ray_statistics(depth,observed,np.ones(len(observed),bool))
                frames.append({'sample_index':frame['sample_index'],'role':frame['role'],'owned_ray':owned,
                    'all_near_box_rays':len(observed),'mean_free_intrusion_m':all_rays['free_intrusion_m'] or 0.,
                    'positive_points':len(target),'positive_surface_mean_m':float(distance.mean()) if distance is not None else None,
                    'positive_surface_recall_02':float((distance<=.2).mean()) if distance is not None else (0. if len(target) else None),
                    'owned_miss_with_surface_within_02':int(((~np.isfinite(depth[positive]))&(distance<=.2)).sum()) if distance is not None else 0})
                for name,tree in point_trees.items():
                    d=tree.query(target,k=1,workers=2)[0] if len(target) and tree is not None else None
                    point_row[name].append({'sample_index':frame['sample_index'],'points':len(target),
                        'distance_sum_m':float(d.sum()) if d is not None else None,
                        'near_02':int((d<=.2).sum()) if d is not None else 0})
            row={'actor':entry,'frames':frames,'surface_patches':len(native),'seed_support':{'lidar_fallback':False}}
            raw_rows.append(row); dense.append(summarize_actor(row)); points_rows.append(point_row)
            del bvh,vertices,faces,case
            save('status.json',{'status':'running','actors_done':len(dense),'elapsed_s':time.monotonic()-started})
        result={'status':'done','final':raw_rows,'actors':{'dense':dense,'matched':matched},'point_rows':points_rows,
            'statistics':{'dense':stage_statistics(dense),'matched':stage_statistics(matched)},
            'paired_dense_minus_matched':paired(matched,dense),'point_statistics':point_statistics(points_rows),
            'moving_gt2mps':{'dense':stage_statistics([r for r in dense if (r.get('translation_speed_mps') or 0)>2]),
                            'matched':stage_statistics([r for r in matched if (r.get('translation_speed_mps') or 0)>2]),
                            'points':point_statistics([r for r in points_rows if (r['actor'].get('translation_speed_mps') or 0)>2])},
            'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'different surface density, fixed output construction; old development diagnostic only'}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({k:result[k] for k in ['status','statistics','point_statistics','wall_s','peak_rss_gib']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
