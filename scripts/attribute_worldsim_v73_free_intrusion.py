"""从已保存的真实束统计分解侵入；不重跑模型、不把非本Actor等同背景。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text())
    stages={}
    for stage in ['baseline','initial','final']:
        actors=[]
        for row in summary[stage]:
            frames=[f for f in row['frames'] if f['role']=='heldout_time']
            n=sum(f['all_near_box_rays'] for f in frames)
            if not n: continue
            total=sum(f['mean_free_intrusion_m']*f['all_near_box_rays'] for f in frames)
            owned=sum((f['owned_ray'].get('mean_intrusion_m') or 0)*f['owned_ray']['rays'] for f in frames)
            counts=sum((f['free_intrusion_rate'] or 0)*f['all_near_box_rays'] for f in frames)
            owned_counts=sum((f['owned_ray'].get('early_rate') or 0)*f['owned_ray']['rays'] for f in frames)
            actors.append({**row['actor'],'all_rays':n,'all_intrusion_m':total/n,
                'owned_contribution_m':owned/n,'nonowned_contribution_m':max(0,total-owned)/n,
                'all_intrusion_rate':counts/n,'nonowned_intrusion_rate_all_denominator':max(0,counts-owned_counts)/n,
                'intruding_rays_mean_severity_m':total/counts if counts>0 else None})
        roles={}
        for role in ['fit','development']:
            logs=defaultdict(list)
            for row in actors:
                if row['role']==role: logs[row['log_id']].append(row)
            per_log={key:{metric:statistics.mean(v[metric] for v in values) for metric in
                ['all_intrusion_m','owned_contribution_m','nonowned_contribution_m','all_intrusion_rate',
                 'nonowned_intrusion_rate_all_denominator']} for key,values in logs.items()}
            means={metric:statistics.mean(v[metric] for v in per_log.values()) for metric in
                ['all_intrusion_m','owned_contribution_m','nonowned_contribution_m','all_intrusion_rate',
                 'nonowned_intrusion_rate_all_denominator']}
            means['nonowned_share_of_log_mean_intrusion']=means['nonowned_contribution_m']/means['all_intrusion_m'] if means['all_intrusion_m']>0 else None
            roles[role]={'logs':len(logs),'means':means,'per_log':per_log}
        stages[stage]={'roles':roles,'actors':actors}
    result={'summary':str(args.summary),'stages':stages,
        'method':'count-weighted saved frame means; per-Actor quantities averaged within log, then equal independent log mean',
        'scope':'inside-window extra times; fit may be training labels, see fit_label_times',
        'fit_label_times':summary.get('fit_label_times','build'),
        'ownership_boundary':'owned is unique annotated-box membership proxy; complement combines other Actor, background, ambiguous/unassigned returns; not instance visibility ground truth',
        'precision':'subtraction of saved float means; tiny negative roundoff clipped at zero; no new model or ray evaluation'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({stage:value['roles']['development']['means'] for stage,value in stages.items()}))


if __name__=='__main__': main()
