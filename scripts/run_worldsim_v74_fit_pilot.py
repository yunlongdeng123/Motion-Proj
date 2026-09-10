"""一次 FIT 元数据固定试运行，核对真实导出合同和所需算力，不读取 DEV。"""
import argparse,json,sys,time,traceback,resource
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.data import load_build
from motion_proj.worldsim_v74.a_wex import reconstruct
from motion_proj.worldsim_v74.common import write_json,evaluate
from motion_proj.worldsim_v73.surface_readout import _select_first_triangle

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text())
index=json.loads((Path(cfg['data_root'])/'index.json').read_text())['cases']
selected=[]
for dataset in ['nuscenes','av2']:
    pool=sorted([r for r in index if r['dataset']==dataset and r['role']=='FIT' and r['build_points']>=3],key=lambda r:(r['build_points'],r['case_id']))
    selected.extend([pool[len(pool)//2],pool[int(len(pool)*.95)]])
write_json(args.output/'manifest.json',{'task':'WS-V74-FIT-PILOT-01','seed':7401,'cases':selected,'methods':['WEX','A3_milp'],'config':cfg,
    'query_evaluated':False,'failure_ledger_refs':cfg['failure_ledger_refs']})
rows=[]
for row in selected:
    b=load_build(Path(row['build_file']).parent)
    for method in ['WEX','A3_milp']:
        folder=args.output/row['dataset']/row['case_id']/method
        params={**cfg,**cfg['dataset_parameters'][row['dataset']]}
        try:
            v,f,record=reconstruct(b,params,method,folder)
            metrics=evaluate(v,f,b,folder/'build_rays.npz')
            # 只对首个 FIT 资产做一次既有首交点读出等价核对，之后不重复。
            if not rows:
                depth,_=_select_first_triangle(torch.as_tensor(v,dtype=torch.float32,device='cuda'),torch.as_tensor(f,dtype=torch.long,device='cuda'),
                    torch.as_tensor(b['origins_actor_m'],device='cuda'),torch.as_tensor(b['directions_actor'],device='cuda'))
                saved=np.load(folder/'build_rays.npz')['first_range_m'];finite=np.isfinite(saved)
                err=float(np.max(np.abs(saved[finite]-depth.cpu().numpy()[finite]))) if finite.any() else 0.
                record['legacy_readout_max_range_delta_m']=err
                record['legacy_readout_hit_mask_equal']=bool(np.array_equal(finite,torch.isfinite(depth).cpu().numpy()))
            rows.append({'dataset':row['dataset'],'case_id':row['case_id'],'method':method,'reconstruction':record,'metrics_build':metrics})
            write_json(folder/'build_metrics.json',metrics)
            print(row['dataset'],row['build_points'],method,json.dumps(record),flush=True)
        except Exception as exc:
            folder.mkdir(parents=True,exist_ok=True);(folder/'error.txt').write_text(traceback.format_exc())
            rows.append({'dataset':row['dataset'],'case_id':row['case_id'],'method':method,'status':'engineering_error','error':str(exc)})
            print(traceback.format_exc(),flush=True)
    write_json(args.output/'results.json',rows)
write_json(args.output/'resources.json',{'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
