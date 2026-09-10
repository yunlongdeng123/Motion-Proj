"""先生成完整队列资产，再由独立入口评价；不在求解进程读取 QUERY。"""
import argparse,json,sys,time,resource,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.data import load_build
from motion_proj.worldsim_v74.common import write_json,save_mesh

parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text())
cohort=json.loads((Path(cfg['data_root'])/'probe_cohort.json').read_text())
cases=cohort['cases'];start=time.monotonic();rows=[]
write_json(args.output/'manifest.json',{'task':'WS-V74-METHOD-TOURNAMENT-01','run':args.output.name,'phase':'mini_real_assets',
    'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'seed':7401,'methods':args.methods,'config':cfg,'cases':cases,
    'query_loaded':False,'failure_ledger_refs':cfg['failure_ledger_refs']})
for case in cases:
    b=load_build(Path(case['build_file']).parent)
    for method in args.methods:
        folder=args.output/case['dataset']/case['case_id']/method
        params={**cfg,**cfg['dataset_parameters'][case['dataset']]}
        params['dcs_checkpoints']=cfg.get('dcs_checkpoints',{}).get(case['dataset'],{})
        try:
            if method.startswith('A') or method=='WEX':from motion_proj.worldsim_v74.a_wex import reconstruct
            elif method.startswith('B') or method=='RIF':from motion_proj.worldsim_v74.b_rif import reconstruct
            elif method.startswith('C') or method=='DCS':from motion_proj.worldsim_v74.c_dcs import reconstruct
            else:raise ValueError(method)
            v,f,record=reconstruct(b,params,method,folder)
            rows.append({'dataset':case['dataset'],'log_id':case['log_id'],'case_id':case['case_id'],'method':method,'record':record,'folder':str(folder)})
            print(case['dataset'],case['case_id'],method,json.dumps(record),flush=True)
        except Exception as exc:
            folder.mkdir(parents=True,exist_ok=True);(folder/'error.txt').write_text(traceback.format_exc())
            rows.append({'dataset':case['dataset'],'log_id':case['log_id'],'case_id':case['case_id'],'method':method,'record':{'status':'engineering_error','error':str(exc)},'folder':str(folder)})
            print(traceback.format_exc(),flush=True)
        write_json(args.output/'progress.json',{'completed':len(rows),'total':len(cases)*len(args.methods),'last':rows[-1],'wall_seconds':time.monotonic()-start})
    write_json(args.output/'assets.json',rows)
write_json(args.output/'resources.json',{'wall_seconds':time.monotonic()-start,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
    'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
