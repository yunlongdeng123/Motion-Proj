"""先在Actor内按真实束/点计数汇总，再按独立日志等权报告配对变化。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

METRICS=['hit_rate','early_rate','miss_rate','free_intrusion_m','surface_distance_m','surface_recall_02']


def summarize_actor(row):
    frames=[r for r in row['frames'] if r['role']=='heldout_time']
    owned=sum(r['owned_ray']['rays'] for r in frames)
    rays=sum(r['all_near_box_rays'] for r in frames)
    points=sum(r['positive_points'] for r in frames)
    value={**row['actor'],'heldout_owned_rays':owned,'heldout_all_rays':rays,
           'surface_patches':row['surface_patches'],'extra_time_usage':row.get('extra_time_usage','evaluation_only')}
    for name in ['hit_rate','early_rate','miss_rate']:
        value[name]=sum(r['owned_ray']['rays']*(r['owned_ray'].get(name) or 0) for r in frames)/owned if owned else None
    value['free_intrusion_m']=sum(r['all_near_box_rays']*r['mean_free_intrusion_m'] for r in frames)/rays if rays else None
    distances=[r for r in frames if r['positive_surface_mean_m'] is not None]
    measured=sum(r['positive_points'] for r in distances)
    value['surface_distance_m']=sum(r['positive_points']*r['positive_surface_mean_m'] for r in distances)/measured if measured else None
    value['surface_recall_02']=sum(r['positive_points']*(r['positive_surface_recall_02'] or 0) for r in frames)/points if points else None
    support=row.get('seed_support',row)
    value['lidar_fallback']=support.get('lidar_fallback')
    value['native_candidates']=support.get('native_candidates')
    value['native_sensor_huber_m']=support.get('native_sensor_huber_m')
    return value


def stage_statistics(rows):
    output={}
    for role in ['fit','development',*sorted({row['role'] for row in rows}-{'fit','development'})]:
        selected=[r for r in rows if r['role']==role]
        result={'actors':len(selected),'logs':len({r['log_id'] for r in selected}),
                'moving_gt2mps':sum((r.get('translation_speed_mps') or 0)>2 for r in selected),
                'motion_unavailable':sum(r.get('translation_speed_mps') is None for r in selected),
                'build_under100':sum(r['build_points']<100 for r in selected),
                'no_heldout_owned_return':sum(r['heldout_owned_rays']==0 for r in selected),
                'no_predicted_surface':sum(r['surface_patches']==0 for r in selected),
                'lidar_fallback':sum(r['lidar_fallback'] is True for r in selected)}
        for metric in METRICS:
            logs=defaultdict(list)
            for row in selected:
                if row[metric] is not None: logs[row['log_id']].append(row[metric])
            values=[statistics.mean(v) for v in logs.values()]
            result[metric]={'mean':statistics.mean(values) if values else None,'logs':len(values)}
        output[role]=result
    return output


def paired(reference,final):
    before={r['owner']:r for r in reference}
    output={}
    for role in ['fit','development',*sorted({row['role'] for row in final}-{'fit','development'})]:
        metrics={}
        for metric in METRICS:
            logs=defaultdict(list)
            for row in final:
                if row['role']!=role or row['owner'] not in before: continue
                a=before[row['owner']][metric]; b=row[metric]
                if a is not None and b is not None: logs[row['log_id']].append(b-a)
            delta=[statistics.mean(v) for v in logs.values()]
            rng=random.Random(7304)
            boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000)) if delta else []
            metrics[metric]={'logs':len(delta),'mean_delta':statistics.mean(delta) if delta else None,
                'bootstrap95':[boot[249],boot[9749]] if boot else None,
                'improved_logs':sum((d>0 if metric in ['hit_rate','surface_recall_02'] else d<0) for d in delta),
                'per_log_delta':{k:statistics.mean(v) for k,v in logs.items()}}
        output[role]=metrics
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--fusion',type=Path)
    parser.add_argument('--reference',action='append',default=[],help='NAME=completed run; report paired final differences without rerunning inference')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    stages={name:[summarize_actor(row) for row in summary[key]] for name,key in
            [('lidar_pca','baseline'),('initial','initial'),('final','final')] if key in summary}
    if args.fusion:
        fusion=json.loads((args.fusion/'summary.json').read_text())
        stages['native_lidar_fusion']=[summarize_actor(row) for row in fusion['final']]
    reference_protocols={}
    for reference in args.reference:
        name,path=reference.split('=',1)
        other=json.loads((Path(path)/'summary.json').read_text())
        stages[name]=[summarize_actor(row) for row in other['final']]
        reference_protocols[name]={'run':path,'fit_label_times':other.get('fit_label_times'),
                                   'boundary':other.get('boundary')}
    result={'run':str(args.run),'status':summary['status'],
        'scope':summary.get('boundary','existing log cohort; no new-source confirmation'),
        'fit_extra_time_usage':'training_labels' if summary.get('fit_label_times') in ['all_window','full_track'] else 'evaluation_only',
        'development_extra_time_usage':'evaluation_only',
        'reference_protocols':reference_protocols,
        'aggregation':'within Actor weighted by observed rays/points, then scene/Actor mean within log and independent log mean',
        'denominator':'owned first-return outcomes include misses; free includes all raw near-box rays; unknown surface excluded',
        'stages':{name:stage_statistics(rows) for name,rows in stages.items()},'actors':stages,
        'paired_final_minus':{name:paired(rows,stages['final']) for name,rows in stages.items() if name!='final'}}
    moving={name:[row for row in rows if (row.get('translation_speed_mps') or 0)>2] for name,rows in stages.items()}
    result['moving_gt2mps']={'definition':'known trajectory mean translation speed over build window > 2 m/s; report independent log count',
        'stages':{name:stage_statistics(rows) for name,rows in moving.items()},
        'paired_final_minus':{name:paired(rows,moving['final']) for name,rows in moving.items() if name!='final'}}
    ready={name:[row for row in rows if row.get('input_status',row.get('status'))=='ready'] for name,rows in stages.items()}
    result['build_lidar_ready']={'definition':'original build/metadata input status ready, independent of prediction quality; full population remains primary',
        'stages':{name:stage_statistics(rows) for name,rows in ready.items()},
        'paired_final_minus':{name:paired(rows,ready['final']) for name,rows in ready.items() if name!='final'}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['stages'],ensure_ascii=False))


if __name__=='__main__': main()
