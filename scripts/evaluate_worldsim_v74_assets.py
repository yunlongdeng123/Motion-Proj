"""固定表面离线评价：全对象分母保留，BUILD和QUERY各自汇总。"""
import argparse,json,sys,time,resource
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.common import write_json,evaluate
from motion_proj.worldsim_v74.data import load_build

parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
manifest=json.loads((args.assets/'manifest.json').read_text());assets=json.loads((args.assets/'assets.json').read_text())
cfg=manifest['config'];cohort=json.loads((Path(cfg['data_root'])/'probe_cohort.json').read_text())
cases=cohort['cases']+cohort['missing_build_contract_queue'];methods=manifest['methods'];lookup={(r['dataset'],r['case_id'],r['method']):r for r in assets}
write_json(args.output/'manifest.json',{'task':'WS-V74-PROBE-EVALUATION-01','source_assets':str(args.assets),'seed':7401,
    'methods':methods,'cases':cases,'quality_role':'exposed dataset-specific DEV; not new FINAL','readout':'same V73 hard first triangles/nearest surface, no retuning','failure_ledger_refs':cfg['failure_ledger_refs']})
rows=[]
for case in cases:
    folder=Path(case['build_file']).parent;build=load_build(folder)
    with np.load(case['query_truth_file'],allow_pickle=False) as arrays:query={k:arrays[k] for k in arrays.files}
    for method in methods:
        out=args.output/case['dataset']/case['case_id']/method;out.mkdir(parents=True)
        record=lookup.get((case['dataset'],case['case_id'],method));source=Path(record['folder'])/'final.npz' if record else None
        if source is not None and source.exists():
            with np.load(source,allow_pickle=False) as arrays:v=arrays['vertices_actor_m'];f=arrays['faces']
            status=record['record']['status']
        elif case['build_points']==0:v=np.empty((0,3));f=np.empty((0,3),int);status='missing_build'
        else:
            rows.append({'dataset':case['dataset'],'log_id':case['log_id'],'case_id':case['case_id'],'method':method,'status':'missing_method_output','source_record':record,'metrics_query':None,'metrics_build':None});continue
        t=time.monotonic();q=evaluate(v,f,query,out/'query_rays.npz');query_seconds=time.monotonic()-t
        b=evaluate(v,f,build,out/'build_rays.npz')
        row={'dataset':case['dataset'],'log_id':case['log_id'],'case_id':case['case_id'],'owner':case['owner'],'method':method,'status':status,
            'build_points':case['build_points'],'build_rank_stratum':case.get('build_rank_stratum','missing_build'),
            'translation_speed_mps':case.get('translation_speed_mps'),'contract_missing_input':case['build_points']==0,
            'surface_path':str(source) if source else None,'face_budget_compliant':len(f)<=cfg['face_budget'],
            'metrics_build':b,'metrics_query':q,'query_wall_seconds':query_seconds,'reconstruction':record['record'] if record else None}
        write_json(out/'metrics.json',row);rows.append(row)
    write_json(args.output/'per_actor.json',rows)
    print(case['dataset'],case['case_id'],'evaluated',len(rows),flush=True)
keys=['hit_rate','early_rate','late_rate','miss_rate','mean_free_intrusion_m','positive_surface_mean_m','positive_surface_recall_02','any_correct_intersection','early_with_later_correct_support']
summary={};logs=[]
for dataset in ['nuscenes','av2']:
    summary[dataset]={}
    for method in methods:
        chosen=[r for r in rows if r['dataset']==dataset and r['method']==method];ss={}
        for scope in ['ready','all_with_missing_build']:
            population=[r for r in chosen if scope!='ready' or not r.get('contract_missing_input')]
            result={'objects':len(population),'missing_method_output':sum(r['metrics_query'] is None for r in population),
                'empty_surfaces':sum(r['metrics_query'] is not None and r['metrics_query']['faces']==0 for r in population),
                'no_owned_query':sum(r['metrics_query'] is not None and r['metrics_query']['rays']==0 for r in population),
                'budget_exceeded':sum(not r.get('face_budget_compliant',False) for r in population)}
            for role in ['metrics_build','metrics_query']:
                result[role]={}
                for metric in keys:
                    perlog=defaultdict(list)
                    for r in population:
                        value=r[role].get(metric) if r[role] is not None else None
                        if value is not None:perlog[r['log_id']].append(value)
                    means={k:float(np.mean(v)) for k,v in perlog.items()};values=list(means.values())
                    result[role][metric]={'mean':float(np.mean(values)) if values else None,'logs':len(values),'defined_objects':sum(map(len,perlog.values())),'per_log':means}
            ss[scope]=result
        summary[dataset][method]=ss
write_json(args.output/'summary.json',{'aggregation':'ray/point weighted within object, object means within log, independent log equal means; undefined denominators stay undefined','datasets':summary})
write_json(args.output/'resources.json',{'wall_seconds':time.monotonic()-start,'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
