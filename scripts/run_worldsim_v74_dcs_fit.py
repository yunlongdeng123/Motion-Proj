"""DCS 首次 FIT 资产与连续八边形/导出三角首交点合同。"""
import argparse,json,sys,traceback,resource
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.c_dcs import reconstruct,geometry_context,patches_from_parameters,patch_incidence,export_patches
from motion_proj.worldsim_v74.common import write_json,evaluate,ray_state
from motion_proj.worldsim_v74.data import load_build
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text())
prior=json.loads((ROOT/'docs/autoresearch/worldsim_v74/a_wex/fit_pilot_manifest.json').read_text());rows=[]
write_json(args.output/'manifest.json',{'task':'WS-V74-C-FIT-PILOT-01','seed':7401,'cases':prior['cases'],'config':cfg,'query_loaded':False,'failure_ledger_refs':cfg['failure_ledger_refs']})
for case in prior['cases']:
    b=load_build(Path(case['build_file']).parent);params={**cfg,**cfg['dataset_parameters'][case['dataset']],'dcs_checkpoints':cfg['dcs_checkpoints'][case['dataset']]}
    for method in ['DCS','C0_fixed','C1_residual','C2_local_network','C3_no_demand']:
        folder=args.output/case['dataset']/case['case_id']/method
        try:
            v,f,record=reconstruct(b,params,method,folder)
            metric=evaluate(v,f,b,folder/'build_rays.npz');write_json(folder/'build_metrics.json',metric)
            if not rows:
                with np.load(folder/'final.npz') as raw:patches={k:raw[k][raw['selected']] for k in ['center','u','v','normal','scale']}
                A,E=patch_incidence(patches,b,params['epsilon_obs_m']);state=ray_state(v,f,b,params['epsilon_obs_m'])
                record['analytic_mesh_any_correct_equal']=bool(np.array_equal(np.asarray(A.sum(1)).ravel()>0,state['any_correct'][np.flatnonzero(b['positive_actor']&~b['ambiguous_owner'])]))
                record['analytic_mesh_early_equal']=bool(np.array_equal(np.asarray(E.sum(1)).ravel()>0,state['early']))
            rows.append({'dataset':case['dataset'],'case_id':case['case_id'],'method':method,'reconstruction':record,'metrics_build':metric})
            print(case['dataset'],case['build_points'],method,json.dumps(record),flush=True)
        except Exception as e:
            folder.mkdir(parents=True,exist_ok=True);(folder/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
            rows.append({'dataset':case['dataset'],'case_id':case['case_id'],'method':method,'status':'engineering_error','error':str(e)})
        write_json(args.output/'results.json',rows)
write_json(args.output/'resources.json',{'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
