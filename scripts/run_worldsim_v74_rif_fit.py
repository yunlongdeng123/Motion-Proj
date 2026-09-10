"""RIF/B0/B1/B2 在已固定 FIT pilot 四对象上的首次重建。"""
import argparse,json,sys,time,traceback,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.b_rif import reconstruct
from motion_proj.worldsim_v74.common import write_json,evaluate
from motion_proj.worldsim_v74.data import load_build
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text())
prior=json.loads((ROOT/'docs/autoresearch/worldsim_v74/a_wex/fit_pilot_manifest.json').read_text());rows=[]
write_json(args.output/'manifest.json',{'task':'WS-V74-B-FIT-PILOT-01','seed':7401,'cases':prior['cases'],'config':cfg,'query_loaded':False,'failure_ledger_refs':cfg['failure_ledger_refs']})
for case in prior['cases']:
    b=load_build(Path(case['build_file']).parent)
    for method in ['RIF','B0_sampled','B1_exact','B2_regular']:
        folder=args.output/case['dataset']/case['case_id']/method
        try:
            v,f,record=reconstruct(b,{**cfg,**cfg['dataset_parameters'][case['dataset']]},method,folder)
            metric=evaluate(v,f,b,folder/'build_rays.npz');write_json(folder/'build_metrics.json',metric)
            rows.append({'dataset':case['dataset'],'case_id':case['case_id'],'method':method,'reconstruction':record,'metrics_build':metric})
            print(case['dataset'],case['build_points'],method,json.dumps(record),flush=True)
        except Exception as e:
            folder.mkdir(parents=True,exist_ok=True);(folder/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
            rows.append({'dataset':case['dataset'],'case_id':case['case_id'],'method':method,'status':'engineering_error','error':str(e)})
        write_json(args.output/'results.json',rows)
write_json(args.output/'resources.json',{'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
