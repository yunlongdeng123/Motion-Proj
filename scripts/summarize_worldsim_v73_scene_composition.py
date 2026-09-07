"""复用已保存场景结果做独立日志配对，不重复推理或射线读出。"""
import argparse,json,random,statistics
from pathlib import Path


def paired(before,after):
    result={}
    for group,a in before.items():
        if group not in after: continue
        b=after[group]; metrics={}
        for metric in a['means']:
            delta={log:values[metric]-a['per_log'][log][metric] for log,values in b['per_log'].items()
                if log in a['per_log'] and values[metric] is not None and a['per_log'][log][metric] is not None}
            values=list(delta.values()); rng=random.Random(7306)
            boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000)) if values else []
            metrics[metric]={'logs':len(values),'mean_delta':statistics.mean(values) if values else None,
                'bootstrap95':[boot[249],boot[9749]] if values else None,'per_log_delta':delta,
                'improved_logs':sum(v>0 if metric=='hit_rate' else v<0 for v in values)}
        result[group]=metrics
    return result


parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--before',type=Path); parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--pair',action='append',default=[],help='BEFORE=AFTER method names within the saved scene run')
args=parser.parse_args(); summary=json.loads((args.run/'summary.json').read_text()); stages=summary['statistics']
result={'run':str(args.run),'aggregation':'paired equal independent log differences after within-scene ray weighting; 10000 log bootstrap samples seed7306',
    'boundary':'existing development data; five logs, no new-source confirmation; return count and missing remain in original summary',
    'methods_minus_lidar_pca':{name:paired(stages['lidar_pca'],stage) for name,stage in stages.items()
        if name not in ['background_only','lidar_pca']}}
result['requested_pairs']={value:paired(stages[value.split('=',1)[0]],stages[value.split('=',1)[1]]) for value in args.pair}
if args.before:
    before=json.loads((args.before/'summary.json').read_text())['statistics']
    result['before_run']=str(args.before)
    result['same_method_background_change']={name:paired(before[name],stage) for name,stage in stages.items() if name in before}
args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'methods_vs_pca_cohort':{k:v['cohort_returns'] for k,v in result['methods_minus_lidar_pca'].items()},
    'background_change_all_raw':result.get('same_method_background_change',{}).get('background_only',{}).get('all_raw_returns')}))
