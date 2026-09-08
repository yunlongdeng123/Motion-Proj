"""Summarize saved zero-LiDAR predictions without creating new correspondences."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--fixed-run',type=Path,required=True)
parser.add_argument('--fusion-run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
fixed=json.loads((args.fixed_run/'summary.json').read_text())
keys={(r['actor']['scene'],r['actor']['owner']) for r in fixed['final']}
fusion=json.loads((args.fusion_run/'summary.json').read_text())
models={'joint_r5_visual_only':fixed['final'],
        'native_fusion':[r for r in fusion['final'] if (r['actor']['scene'],r['actor']['owner']) in keys]}
records={}; result={}
for name,rows in models.items():
    actors=[]
    for row in rows:
        for time_role in ['build','heldout_time']:
            frames=[r for r in row['frames'] if r['role']==time_role]
            rays=sum(r['all_near_box_rays'] for r in frames)
            owned=sum(r['owned_ray']['rays'] for r in frames)
            metrics={'free_intrusion_m':sum(r['mean_free_intrusion_m']*r['all_near_box_rays'] for r in frames if r['all_near_box_rays'])/rays if rays else None}
            for metric in ['hit_rate','early_rate','late_rate','miss_rate']:
                metrics[metric]=sum(r['owned_ray'][metric]*r['owned_ray']['rays'] for r in frames if r['owned_ray']['rays'])/owned if owned else None
            actors.append({'actor':row['actor'],'time_role':time_role,'near_box_rays':rays,'owned_rays':owned,
                           'surface_patches':row['surface_patches'],'seed_support':row['seed_support'],'metrics':metrics})
    records[name]=actors; result[name]={}
    for role in ['fit','development']:
        result[name][role]={}
        for time_role in ['build','heldout_time']:
            subset=[r for r in actors if r['actor']['role']==role and r['time_role']==time_role]
            logs=defaultdict(lambda:defaultdict(list))
            for row in subset:
                for metric,value in row['metrics'].items():
                    if value is not None: logs[row['actor']['log_id']][metric].append(value)
            per_log={log:{metric:float(np.mean(v)) for metric,v in metrics.items()} for log,metrics in logs.items()}
            metrics={metric:{'logs':sum(metric in v for v in per_log.values()),
                             'mean':float(np.mean([v[metric] for v in per_log.values() if metric in v]))}
                     for metric in sorted({m for v in per_log.values() for m in v})}
            result[name][role][time_role]={'actors':len(subset),'logs':len({r['actor']['log_id'] for r in subset}),
                'near_box_actor_ray_occurrences':sum(r['near_box_rays'] for r in subset),'owned_rays':sum(r['owned_rays'] for r in subset),
                'with_owned_rays':sum(r['owned_rays']>0 for r in subset),'with_surface':sum(r['surface_patches']>0 for r in subset),
                'coarse_fallback_actors':sum(r['seed_support'].get('coarse_fallback',False) for r in subset),
                'metrics':metrics,'per_log':per_log}
out={'statistics':result,'actors':records,
     'fixed_run':str(args.fixed_run),'fusion_run':str(args.fusion_run),
     'fixed_resources':{k:fixed[k] for k in ['wall_s','peak_gpu_gib','peak_rss_gib','optimizer_updates']},
     'boundary':'metadata-zero-LiDAR subgroup; fixed r5 input-path migration, not retraining; input-fit logs are not independent confirmation; one development owned ray prevents a precision conclusion; no bootstrap or new inference'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
