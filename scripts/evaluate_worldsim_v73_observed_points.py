"""Bidirectional point metrics on a fixed observed-ray query domain.

This is not full-surface precision: predicted points are literal first triangle
intersections on heldout beams whose measured first return belongs to the Actor.
Unseen canonical surface and no-return beams have no fabricated ground truth.
"""
import argparse,json,random,resource,statistics,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH,pca_patch_surface

METRICS=['beam_point_precision_02','beam_point_recall_02','beam_point_fscore_02',
         'predicted_to_measured_m','measured_to_predicted_m','beam_point_chamfer_l1_sum_m','literal_missing_rate']

def point_metrics(predicted,target,beam_count):
    n=len(predicted); m=len(target)
    result={key:None for key in METRICS}
    result.update(observed_actor_beams=beam_count,predicted_returns=n,measured_points=m)
    if not m: return result
    result['literal_missing_rate']=1-n/beam_count
    if not n:
        result.update(beam_point_precision_02=0.,beam_point_recall_02=0.,beam_point_fscore_02=0.)
        return result
    forward=cKDTree(target).query(predicted,k=1,workers=2)[0]
    backward=cKDTree(predicted).query(target,k=1,workers=2)[0]
    precision=float((forward<=.2).mean()); recall=float((backward<=.2).mean())
    result.update(beam_point_precision_02=precision,beam_point_recall_02=recall,
        beam_point_fscore_02=2*precision*recall/(precision+recall) if precision+recall else 0.,
        predicted_to_measured_m=float(forward.mean()),measured_to_predicted_m=float(backward.mean()),
        beam_point_chamfer_l1_sum_m=float(forward.mean()+backward.mean()))
    return result

def aggregate(rows):
    output={'actors':len(rows),'logs':len({r['actor']['log_id'] for r in rows}),
        'no_observed_actor_beams':sum(r['observed_actor_beams']==0 for r in rows),
        'observed_but_no_prediction':sum(r['observed_actor_beams']>0 and r['predicted_returns']==0 for r in rows),
        'observed_actor_beams':sum(r['observed_actor_beams'] for r in rows),
        'predicted_returns':sum(r['predicted_returns'] for r in rows),'metrics':{}}
    for metric in METRICS:
        logs=defaultdict(list)
        for row in rows:
            if row[metric] is not None: logs[row['actor']['log_id']].append(row[metric])
        values={log:statistics.mean(v) for log,v in logs.items()}
        output['metrics'][metric]={'mean':statistics.mean(values.values()) if values else None,
            'valid_actors':sum(len(v) for v in logs.values()),'logs':len(values),'per_log':values}
    return output

def paired(before,after):
    reference={r['actor']['owner']:r for r in before}; result={}
    for metric in METRICS:
        logs=defaultdict(list)
        for row in after:
            old=reference[row['actor']['owner']]
            if row[metric] is not None and old[metric] is not None:
                logs[row['actor']['log_id']].append(row[metric]-old[metric])
        values={log:statistics.mean(v) for log,v in logs.items()}; delta=list(values.values())
        rng=random.Random(7304)
        boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000)) if len(delta)>1 else []
        result[metric]={'mean_delta':statistics.mean(delta) if delta else None,'logs':len(delta),
            'paired_actors':sum(len(v) for v in logs.values()),'bootstrap95':[boot[249],boot[9749]] if boot else None,
            'per_log_delta':values}
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--model',action='append',default=[],help='NAME=completed fixed surface run')
    parser.add_argument('--compare',action='append',default=[],help='CANDIDATE=REFERENCE, explicit same-cohort paired comparison')
    parser.add_argument('--role',default='development'); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2); started=time.monotonic()
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    models={name:Path(path) for name,path in (value.split('=',1) for value in args.model)}
    manifest={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'actor_data':str(args.actor_data),'models':{k:str(v) for k,v in models.items()},'role':args.role,'optimizer_updates':0,
        'query_domain':'all registered heldout-time beams with unambiguous first return on this Actor; same beam domain for every method',
        'predicted_points':'literal first intersections with fixed explicit Actor surface; no opacity, hull, threshold pruning or target-supported sampling',
        'measurement_domain':'original measured endpoints in canonical coordinates, pooled across the registered heldout times',
        'metric_definition':'unsquared Euclidean nearest-neighbor distances, no far-outlier truncation; Chamfer is sum of the two means; precision/recall threshold 0.2 m',
        'empty_policy':'all cohort Actors retained; no measured Actor beams => unavailable; measured beams but no prediction => P/R/F=0 and distance undefined, counts/miss reported',
        'aggregation':'within Actor point metrics, then Actor mean within independent log, then equal log mean; paired intervals require at least two logs',
        'boundary':'observed-ray point-set diagnostic only, not complete canonical-surface precision or full-surface Chamfer; nearest-neighbor scores do not replace original-beam early/hit/free/miss',
        'reference':'https://github.com/THU-LYJ-Lab/SS3DM-Benchmark',
        'migration':'bidirectional point metrics separated from visibility and surface support; this sparse observed-ray protocol is not the SS3DM dense-GT benchmark'}
    save('manifest.json',manifest); save('status.json',{'status':'running'})
    try:
        entries=[r for r in json.loads((args.actor_data/'index.json').read_text())['cases'] if r['role']==args.role]
        rows={name:[] for name in ['lidar_pca',*models]}
        for entry in entries:
            case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=False)
            frames=[r for r in case['rays'] if r['role']=='heldout_time']
            origins=np.concatenate([r['origins_actor_m'][r['positive_actor']].numpy() for r in frames]) if frames else np.empty((0,3),np.float32)
            directions=np.concatenate([r['directions_actor'][r['positive_actor']].numpy() for r in frames]) if frames else np.empty((0,3),np.float32)
            target=np.concatenate([r['points_actor_m'][r['positive_actor']].numpy() for r in frames]) if frames else np.empty((0,3),np.float32)
            for method in rows:
                if method=='lidar_pca': vertices,faces=pca_patch_surface(case['points_actor_m'].numpy(),count=1536)
                else:
                    path=models[method]/(entry['scene']+'__'+entry['owner']+'_surface.pt')
                    if not path.is_file(): path=models[method]/(entry['owner']+'_surface.pt')
                    surface=torch.load(path,map_location='cpu',weights_only=False)
                    vertices=surface['vertices_actor_m'].numpy(); faces=surface['faces'].numpy()
                bvh=SurfaceBVH(vertices,faces)
                depth=bvh.cast(origins,directions) if len(origins) else np.empty(0,np.float32)
                finite=np.isfinite(depth); predicted=origins[finite]+depth[finite,None]*directions[finite]
                rows[method].append({'actor':entry,**point_metrics(predicted,target,len(origins))})
                np.savez_compressed(args.output/(method+'__'+entry['owner']+'.npz'),predicted=predicted,measured=target,
                                    predicted_return_mask=finite,predicted_range_m=depth)
                del bvh
            save('status.json',{'status':'running','actors_done':len(rows['lidar_pca']),'actors_total':len(entries)})
        result={'status':'done','manifest':manifest,'actors':rows,
            'statistics':{method:aggregate(value) for method,value in rows.items()},
            'paired_against_lidar_pca':{method:paired(rows['lidar_pca'],value) for method,value in rows.items() if method!='lidar_pca'},
            'paired_comparisons':{value:paired(rows[value.split('=',1)[1]],rows[value.split('=',1)[0]]) for value in args.compare},
            'moving_gt2mps':{method:aggregate([r for r in value if (r['actor'].get('translation_speed_mps') or 0)>2]) for method,value in rows.items()},
            'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({key:result[key] for key in ['status','statistics','wall_s','peak_rss_gib']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise

if __name__=='__main__': main()
