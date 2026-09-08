"""Assign fixed first-intersection errors to original query provenance.

Source 0 starts at build LiDAR; source 1 is a completion query. Both subsequently
share learned spatial/visual updates: these labels are not modality ablations.
"""
import argparse,json,resource,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH

def distribution(values):
    values=np.asarray(values)
    return {'count':len(values),'mean':float(values.mean()) if len(values) else None,
            'median':float(np.median(values)) if len(values) else None,
            'p95':float(np.quantile(values,.95)) if len(values) else None,
            'max':float(values.max()) if len(values) else None}

def aggregate(rows):
    result={'actors':len(rows),'logs':len({r['actor']['log_id'] for r in rows}),
            'actor_ray_occurrences':sum(r['all_rays'] for r in rows),
            'owned_ray_occurrences':sum(r['owned_rays'] for r in rows),'sources':{}}
    for source in ['evidence','completion']:
        totals={metric:sum(r['sources'][source][metric] for r in rows)
                for metric in ['patches','all_early','all_free_sum_m','owned_early','owned_hit','owned_late']}
        logs=defaultdict(lambda:defaultdict(list))
        for row in rows:
            item=row['sources'][source]; log=row['actor']['log_id']
            if row['all_rays']:
                logs[log]['free_contribution_m'].append(item['all_free_sum_m']/row['all_rays'])
            if row['owned_rays']:
                for metric in ['early','hit','late']:
                    logs[log]['owned_'+metric+'_contribution'].append(item['owned_'+metric]/row['owned_rays'])
            if item['center_to_build_m']['count']:
                logs[log]['center_to_build_mean_m'].append(item['center_to_build_m']['mean'])
                logs[log]['centers_outside_build_02_fraction'].append(item['centers_outside_build_02']/item['patches'])
        per_log={log:{metric:float(np.mean(v)) for metric,v in values.items()} for log,values in logs.items()}
        metrics=sorted({metric for values in per_log.values() for metric in values})
        totals['equal_log_means']={metric:float(np.mean([v[metric] for v in per_log.values() if metric in v])) for metric in metrics}
        totals['per_log']=per_log; result['sources'][source]=totals
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--model',action='append',required=True,help='NAME=fixed completed query-model run')
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2); started=time.monotonic()
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    models={name:Path(path) for name,path in (value.split('=',1) for value in args.model)}
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'actor_data':str(args.actor_data),'models':{k:str(v) for k,v in models.items()},'optimizer_updates':0,
        'scope':'all existing development Actors; no score selection; fixed saved surfaces, CPU first-intersection primitive IDs',
        'query_origin':'source0=initial build LiDAR query, source1=initial completion query; later spatial/visual information is shared',
        'assignment':'primitive ID // 8 maps to the original 8-face query patch; missing rays have no assigned patch',
        'geometry_boundary':'no surface deletion, reordering, opacity, radius change or target-based reconstruction',
        'statistics_boundary':'raw Actor-ray occurrences can repeat original beams across Actors; source contribution metrics first normalize within Actor and then average independent logs',
        'inference_boundary':'descriptive origin decomposition, not causal modality or query-removal ablation; outside build support is not itself an error'})
    save('status.json',{'status':'running'})
    try:
        entries=[r for r in json.loads((args.actor_data/'index.json').read_text())['cases'] if r['role']=='development']
        rows={method:[] for method in models}
        for entry in entries:
            case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=False)
            build=case['points_actor_m'].numpy(); frames=[f for f in case['rays'] if f['role']=='heldout_time']
            def concatenate(key,shape,dtype=np.float32):
                return np.concatenate([f[key].numpy() for f in frames]) if frames else np.empty(shape,dtype)
            origins=concatenate('origins_actor_m',(0,3)); directions=concatenate('directions_actor',(0,3))
            observed=concatenate('observed_first_range_m',(0,)); owned=concatenate('positive_actor',(0,),bool)
            measured=concatenate('points_actor_m',(0,3)); tree=cKDTree(build) if len(build) else None
            for method,run in models.items():
                surface=torch.load(run/(entry['owner']+'_surface.pt'),map_location='cpu',weights_only=False)
                vertices=surface['vertices_actor_m'].numpy(); faces=surface['faces'].numpy(); centers=surface['centers_actor_m'].numpy()
                source=surface['source'].numpy() if len(centers) else np.empty(0,np.int64)
                distance=tree.query(centers,k=1,workers=2)[0] if len(centers) else np.empty(0)
                bvh=SurfaceBVH(vertices,faces); depth=np.full(len(observed),np.inf,np.float32)
                patch=np.full(len(observed),-1,np.int64); origin_class=np.full(len(observed),-1,np.int64)
                if len(observed) and bvh.scene is not None:
                    query=np.concatenate([origins,directions],axis=-1).astype(np.float32)
                    intersections=bvh.scene.cast_rays(o3d.core.Tensor(query),nthreads=2)
                    depth=intersections['t_hit'].numpy(); finite=np.isfinite(depth)
                    patch[finite]=intersections['primitive_ids'].numpy()[finite].astype(np.int64)//8
                    origin_class[finite]=source[patch[finite]]
                finite=np.isfinite(depth); early=finite&(depth<observed-.2)
                hit=finite&(np.abs(depth-observed)<=.2); late=finite&(depth>observed+.2)
                free=np.where(early,observed-.2-depth,0.)
                sources={}
                for tag,name in [(0,'evidence'),(1,'completion')]:
                    patches=source==tag; rays=origin_class==tag
                    sources[name]={'patches':int(patches.sum()),'center_to_build_m':distribution(distance[patches]),
                        'centers_outside_build_02':int((distance[patches]>.2).sum()),
                        'all_early':int((early&rays).sum()),'all_free_sum_m':float(free[rays].sum()),
                        'owned_early':int((owned&early&rays).sum()),'owned_hit':int((owned&hit&rays).sum()),
                        'owned_late':int((owned&late&rays).sum()),
                        'owned_early_patch_to_build_m':distribution(distance[patch[owned&early&rays]])}
                row={'actor':entry,'all_rays':len(observed),'owned_rays':int(owned.sum()),
                    'owned_missing':int((owned&~finite).sum()),'sources':sources,
                    'maximum_vertex_center_offset_m':float(np.linalg.norm(vertices.reshape(-1,9,3)-centers[:,None],axis=-1).max()) if len(centers) else None}
                rows[method].append(row)
                if (entry.get('translation_speed_mps') or 0)>2:
                    take=owned; mask=finite[take]
                    np.savez_compressed(args.output/(method+'__'+entry['owner']+'_moving.npz'),
                        measured=measured[take],predicted=origins[take][mask]+depth[take][mask,None]*directions[take][mask],
                        predicted_return_mask=mask,predicted_range_m=depth[take],observed_range_m=observed[take],
                        source=origin_class[take],build=build,size_lwh_m=case['size_lwh_m'].numpy())
                del bvh,surface
            save('status.json',{'status':'running','actors_done':len(rows[next(iter(models))]),'actors_total':len(entries)})
        result={'status':'done','actors':rows,'statistics':{method:aggregate(value) for method,value in rows.items()},
            'moving_gt2mps':{method:aggregate([r for r in value if (r['actor'].get('translation_speed_mps') or 0)>2]) for method,value in rows.items()},
            'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'fixed-surface source decomposition only; shared updates prevent modality-causal attribution'}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({key:result[key] for key in ['status','statistics','wall_s','peak_rss_gib']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise

if __name__=='__main__': main()
