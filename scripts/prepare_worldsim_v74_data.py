"""逐对象导出 V74 LiDAR 合同，求解输入与留出真值分开驻留。"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import resource
import subprocess
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/autodl-tmp/runs/worldsim_v73')
SOURCES={
    'nuscenes':BASE/'WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2',
    'av2':BASE/'WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1',
}
FLOAT_FIELDS={'origins_actor_m':(3,), 'directions_actor':(3,),
              'observed_first_range_m':(), 'points_actor_m':(3,)}
BOOL_FIELDS=['positive_actor','ambiguous_owner']

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def pack(frames):
    out={}
    for key,shape in FLOAT_FIELDS.items():
        out[key]=np.concatenate([r[key].numpy() for r in frames],axis=0) if frames else np.empty((0,*shape),np.float32)
    for key in BOOL_FIELDS:
        out[key]=np.concatenate([r[key].numpy() for r in frames]) if frames else np.empty(0,bool)
    counts=[len(r['observed_first_range_m']) for r in frames]
    out['frame_offsets']=np.array([0,*np.cumsum(counts)],np.int64)
    out['point_timestamps_ns']=np.concatenate([
        r['point_timestamps_ns'].numpy() if 'point_timestamps_ns' in r else
        np.full(n,r['timestamp_us']*1000,np.int64) for r,n in zip(frames,counts)
    ]) if frames else np.empty(0,np.int64)
    metadata=[{k:v for k,v in r.items() if not torch.is_tensor(v)} for r in frames]
    out['frame_metadata_json']=np.array(json.dumps(metadata))
    return out

def select_probe(entries,logs):
    chosen=[]
    for dataset,log_id in logs:
        pool=sorted([r for r in entries if r['dataset']==dataset and r['log_id']==log_id and r['build_points']>0],
                    key=lambda r:(r['build_points'],r['case_id']))
        # 只按 BUILD 点数排序分成三层，层内取首末，避免使用留出可用性选对象。
        per_log=[]
        for rank,indices in enumerate(np.array_split(np.arange(len(pool)),3)):
            positions=sorted(set([int(indices[0]),int(indices[-1])])) if len(indices) else []
            for position in positions:
                per_log.append({**pool[position],'build_rank_stratum':['lower','middle','upper'][rank]})
        chosen.extend(per_log)
    return chosen

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    started=time.monotonic()
    source_indices={name:json.loads((folder/'index.json').read_text()) for name,folder in SOURCES.items()}
    av2_ids=sorted({r['log_id'] for r in source_indices['av2']['cases']})
    roles={
        'nuscenes':{role:sorted({r['log_id'] for r in source_indices['nuscenes']['cases'] if r['role']==old})
                    for role,old in [('FIT','fit'),('DEV','development')]},
        'av2':{'FIT':av2_ids[3:],'DEV':av2_ids[:3]},
    }
    for data in roles.values():
        if set(data['FIT'])&set(data['DEV']):
            raise ValueError('训练与开发日志重复')
        data['FINAL']=[]
    config={'task':'WS-V74-P0-DATA-01','schema':'worldsim_v74.lidar.v1','seed':7401,'confirmation_seed':7402,
        'dataset_roles':roles,'fit_policy':'separate training per dataset; no cross-dataset transfer required',
        'av2_role_policy':'old exposed 20 logs: first 3 IDs DEV, remaining 17 FIT; new FINAL in separate plan',
        'nuscenes_final_status':'unavailable_fresh_identity; historical exposure is not cleared by changing names',
        'primary_max_triangles':4096,'capacity_max_triangles':16384,'evaluation_hit_band_m':.2,
        'epsilon_obs':'pending FIT calibration before DEV quality; evaluation hit band is not noise sigma',
        'visual_inputs':False,'source_indices':{k:str(v/'index.json') for k,v in SOURCES.items()},
        'output':str(args.output),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'failure_ledger_refs':['V73-F02','V73-F03','V73-F04','V73-F05','V73-F09','V74-F01','V74-F02']}
    write_json(ROOT/'configs/worldsim_v74/data_roles.json',config)
    write_json(args.output/'config.json',config)
    write_json(args.output/'status.json',{'status':'running','completed_cases':0})
    entries=[]
    maximum_endpoint_residual=0.
    for dataset,index in source_indices.items():
        for row in sorted(index['cases'],key=lambda r:(r['log_id'],r['scene'],r['owner'])):
            source=SOURCES[dataset]/row['file']
            case=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
            role='FIT' if row['log_id'] in roles[dataset]['FIT'] else 'DEV'
            key=row['scene']+'__'+row['owner']
            folder=args.output/dataset/role/key
            folder.mkdir(parents=True)
            build_frames=[r for r in case['rays'] if r['role']=='build']
            query_frames=[r for r in case['rays'] if r['role']=='heldout_time']
            if {r['sample_id'] for r in build_frames}&{r['sample_id'] for r in query_frames}:
                raise ValueError('同一观测同时进入 BUILD 和 QUERY: '+key)
            if len(build_frames)+len(query_frames)!=len(case['rays']):
                raise ValueError('未知观测角色: '+key)
            build=pack(build_frames)
            build['support_points_actor_m']=case['points_actor_m'].numpy()
            build['size_lwh_m']=case['size_lwh_m'].numpy()
            np.savez(folder/'build.npz',**build)
            query=pack(query_frames)
            query_input={k:query[k] for k in ['origins_actor_m','directions_actor','point_timestamps_ns','frame_offsets']}
            np.savez(folder/'query_rays.npz',**query_input)
            np.savez(folder/'query_truth.npz',**query)
            # 只对第一份实际输出核对坐标语义，避免逐次重跑 smoke。
            if not entries and len(build['origins_actor_m']):
                endpoints=build['origins_actor_m']+build['directions_actor']*build['observed_first_range_m'][:,None]
                maximum_endpoint_residual=float(np.max(np.linalg.norm(endpoints-build['points_actor_m'],axis=1)))
            meta={'case_id':key,'dataset':dataset,'log_id':row['log_id'],'scene':row['scene'],'owner':row['owner'],
                  'role':role,'category':row['category'],'source_case':str(source),
                  'build_points':len(case['points_actor_m']),'build_frame_ids':[r['sample_id'] for r in build_frames],
                  'build_sample_indices':[r['sample_index'] for r in build_frames],
                  'translation_speed_mps':row.get('translation_speed_mps'),
                  'timing':'per-return AV2 ns' if dataset=='av2' else 'nuScenes scan timestamp approximation',
                  'ownership':'unique annotation box +0.1m proxy, overlap excluded; not segmentation GT',
                  'unknown_policy':'after observed first return unknown; no synthetic no-return rays',
                  'original_status':row['status']}
            write_json(folder/'build_metadata.json',meta)
            entry={**meta,'build_file':str(folder/'build.npz'),'query_rays_file':str(folder/'query_rays.npz'),
                   'query_truth_file':str(folder/'query_truth.npz'),'build_rays':len(build['observed_first_range_m']),
                   'query_rays':len(query['observed_first_range_m']),
                   'query_owned_returns':int(query['positive_actor'].sum()),
                   'query_frame_ids':[r['sample_id'] for r in query_frames],
                   'build_ambiguous_returns':int(build['ambiguous_owner'].sum())}
            entries.append(entry)
            del case,build,query,query_input
            if len(entries)%100==0:
                state={'status':'running','completed_cases':len(entries),'elapsed_s':time.monotonic()-started}
                write_json(args.output/'status.json',state)
                print(json.dumps(state),flush=True)
    probe_logs=[('nuscenes',x) for x in roles['nuscenes']['DEV']]+[('av2',x) for x in roles['av2']['DEV']]
    probe=select_probe(entries,probe_logs)
    missing=[e for e in entries if (e['dataset'],e['log_id']) in probe_logs and not e['build_points']]
    totals={}
    for dataset in roles:
        totals[dataset]={}
        for role in ['FIT','DEV']:
            rows=[r for r in entries if r['dataset']==dataset and r['role']==role]
            totals[dataset][role]={'logs':len({r['log_id'] for r in rows}),'cases':len(rows),
                'ready':sum(r['build_points']>0 for r in rows),'missing_build':sum(r['build_points']==0 for r in rows),
                'without_owned_query':sum(r['query_owned_returns']==0 for r in rows),
                'build_points':sum(r['build_points'] for r in rows),'build_rays':sum(r['build_rays'] for r in rows),
                'query_rays':sum(r['query_rays'] for r in rows),'query_owned_returns':sum(r['query_owned_returns'] for r in rows)}
    summary={'task':config['task'],'status':'done','cases':len(entries),'totals':totals,'probe_cases':len(probe),
        'probe_logs':len(probe_logs),'probe_missing_build_cases':len(missing),
        'probe_without_owned_query':sum(r['query_owned_returns']==0 for r in probe),
        'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        'first_case_endpoint_residual_m':maximum_endpoint_residual,
        'output_bytes':sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
        'optimizer_updates':0,'model_quality_evaluated':False,'new_surface_metrics':False,
        'failure_ledger_delta':'none; V74-F01/F02 remain','source_policy':'same V73 coordinate/ray semantics, all cases retained'}
    write_json(args.output/'index.json',{'schema':config['schema'],'cases':entries})
    probe_manifest={'selection':'5 nuScenes DEV + first 3 old AV2 IDs; 2 per BUILD rank tercile, no QUERY-quality selection',
                    'logs':probe_logs,'cases':probe,'missing_build_contract_queue':missing,
                    'scope':'method comparison only for selected build-available objects; full population index retained'}
    write_json(args.output/'probe_cohort.json',probe_manifest)
    write_json(ROOT/'docs/autoresearch/worldsim_v74/p0/probe_cohort.json',probe_manifest)
    write_json(args.output/'summary.json',summary)
    write_json(ROOT/'docs/autoresearch/worldsim_v74/p0/data_summary.json',summary)
    write_json(args.output/'status.json',{'status':'done','completed_cases':len(entries)})
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
