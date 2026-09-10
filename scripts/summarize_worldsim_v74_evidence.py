import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--events',type=Path,nargs='*',default=[]);parser.add_argument('--mechanisms',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
def write(name,data):(args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
groups=defaultdict(list)
for source in args.events:
    data=json.loads((source/'summary.json').read_text())
    for r in data['records']:groups[(r['dataset'],r['method'])].append(r)
events=[]
for (ds,method),rows in groups.items():
    count=sum(r['proposed'] for r in rows);ub=sum(r['useful_build'] for r in rows);uq=sum(r['useful_query'] for r in rows)
    events.append({'dataset':ds,'method':method,'objects':len(rows),'proposed':count,'useful_build':ub,'useful_query':uq,
        'proposal_weighted_build_useful_rate':ub/count if count else None,'proposal_weighted_query_useful_rate':uq/count if count else None,
        'object_mean_build_useful_rate':float(np.mean([r['useful_build_rate'] for r in rows if r['useful_build_rate'] is not None])) if count else None,
        'object_mean_query_useful_rate':float(np.mean([r['useful_query_rate'] for r in rows if r['useful_query_rate'] is not None])) if count else None,
        'objects_with_first_query_degradation':sum(r['first_degradation'] is not None for r in rows),'first_degradation_assets':[{'case_id':r['case_id'],**r['first_degradation']} for r in rows if r['first_degradation']]})
write('event_summary.json',{'source_folders':[str(p) for p in args.events],'summary':events,'boundary':'post-hoc diagnostic, not inference inputs; proposal usefulness is not actual integer-selected quality'})
for e in events:print(e['dataset'],e['method'],'proposals',e['proposed'],'useful BUILD/QUERY',e['proposal_weighted_build_useful_rate'],e['proposal_weighted_query_useful_rate'],'decline objects',e['objects_with_first_query_degradation'])
if args.mechanisms:
    rows=json.loads((args.mechanisms/'results.json').read_text());groups=defaultdict(list)
    for r in rows:groups[(r['family'],r['method'])].append(r)
    summary=[]
    for (family,method),items in groups.items():
        valid=[r for r in items if r.get('metrics_query') is not None];out={'family':family,'method':method,'count':len(items),'engineering_errors':len(items)-len(valid),'physical_build_feasible':sum(r['record'].get('all_build_feasible',False) for r in valid),'face_budget_exceeded':sum(r['record'].get('faces',0)>4096 for r in valid)}
        for field in ['metrics_query','metrics_build','geometry']:
            out[field]={}
            for key in ['hit_rate','early_rate','miss_rate','mean_free_intrusion_m','positive_surface_recall_02','positive_surface_mean_m','any_correct_intersection','surface_to_gt_m','gt_to_surface_m','gt_recall_02']:
                vv=[r[field].get(key) for r in valid if r.get(field,{}).get(key) is not None]
                if vv:out[field][key]={'mean':float(np.mean(vv)),'defined_instances':len(vv)}
        secs=[r['record']['wall_seconds'] for r in valid];out['wall_seconds_sum']=sum(secs);out['wall_seconds_median']=float(np.median(secs)) if secs else None
        summary.append(out)
    write('mechanism_summary.json',{'source':str(args.mechanisms),'finished':(args.mechanisms/'resources.json').exists(),'instances_methods':len(rows),'summary':summary,'scope':'four full-geometry families x20; fixed-field and analytic domain subsystem experiments reported separately; synthetic C uses fixed nuScenes FIT network and is not an independent learned-distribution test'})
    print('Geometric mechanism instance-method records',len(rows),'finished',(args.mechanisms/'resources.json').exists())
